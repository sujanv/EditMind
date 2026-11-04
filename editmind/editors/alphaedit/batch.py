"""Batch-AlphaEdit: High-Throughput Batched Null-Space Projection Knowledge Editing.

Allows simultaneous insertion of N distinct factual associations in a single closed-form
matrix projection while guaranteeing orthogonality to the subspace of preserved reference keys.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
import time
import torch

from editmind.core.base_editor import BaseKnowledgeEditor
from editmind.core.types import EditRequest, EditResult
from editmind.core.registry import register_editor
from editmind.editors.rome.compute_v import compute_v_target


@register_editor("batch_alphaedit")
class BatchAlphaEditor(BaseKnowledgeEditor):
    """Batched Null-Space Projection Editor for multi-fact simultaneous updating."""

    @property
    def name(self) -> str:
        return "batch_alphaedit"

    def _sample_preserve_keys(self, target_layer: int, sample_count: int = 15) -> torch.Tensor:
        """Collects activation keys from reference prompts to form the preservation subspace."""
        model = self.model_wrapper.model
        mlp = model.layers[target_layer].mlp

        sample_prompts = [
            "The capital of France is",
            "The Colosseum is in",
            "Big Ben is located in",
            "Earth orbits around the",
            "Shakespeare wrote the play",
            "Turing is known as father of",
            "Python programming was designed by",
            "Apple is led by CEO",
            "Tesla was founded by",
            "The Louvre Museum is in",
        ]
        keys = []
        for p in sample_prompts[:sample_count]:
            input_ids = self.model_wrapper.tokenize(p)
            with torch.no_grad():
                _, hiddens = model(input_ids, output_hidden_states=True)
                layer_in = hiddens[target_layer]
                normed = model.layers[target_layer].ln2(layer_in)
                k = mlp.act(mlp.fc1(normed))[0, -1, :].detach()
                keys.append(k)

        # Shape: [in_dim, M]
        return torch.stack(keys, dim=1)

    def edit(self, request: EditRequest) -> EditResult:
        """Single request fallback delegating to batch_edit."""
        results = self.batch_edit([request])
        return results[0]

    def batch_edit(self, requests: List[EditRequest]) -> List[EditResult]:
        """Applies N edits simultaneously via closed-form batched null-space projection."""
        if not requests:
            return []

        start_time = time.time()
        ckpt_id = self.create_checkpoint()

        target_layer = self.config.get("target_layer", 2)
        v_steps = self.config.get("v_num_grad_steps", 20)
        v_lr = self.config.get("v_lr", 0.4)
        sample_count = self.config.get("null_space_preserve_samples", 10)
        reg_lambda = self.config.get("reg_lambda", 1e-3)

        # Pre-measure target probabilities
        pre_probs = []
        for req in requests:
            p_target, p_old = self.measure_target_probabilities(
                req.prompt, req.target_new, req.ground_truth
            )
            pre_probs.append((p_target, p_old))

        # 1. Collect K_* and V_* for all requests in the batch
        k_list = []
        v_list = []
        for req in requests:
            input_ids = self.model_wrapper.tokenize(req.prompt)
            subj_idx = max(0, input_ids.shape[1] - 2)
            k_star, v_star = compute_v_target(
                model_wrapper=self.model_wrapper,
                prompt=req.prompt,
                target_token=req.target_new,
                target_layer_idx=target_layer,
                subject_token_idx=subj_idx,
                num_steps=v_steps,
                lr=v_lr,
                weight_decay=1e-3,
            )
            k_list.append(k_star)
            v_list.append(v_star)

        # K_mat: [in_dim, N], V_mat: [hidden_dim, N]
        K_mat = torch.stack(k_list, dim=1)
        V_mat = torch.stack(v_list, dim=1)

        mlp = self.model_wrapper.model.layers[target_layer].mlp
        W = mlp.down_proj.weight  # [hidden_dim, in_dim]

        # Residuals: Delta_V = V_* - W K_* [hidden_dim, N]
        cur_V = torch.matmul(W, K_mat)
        delta_V = V_mat - cur_V

        # 2. Compute null-space projector: P_null = I - K_pres (K_pres^T K_pres + lambda I)^-1 K_pres^T
        K_pres = self._sample_preserve_keys(target_layer, sample_count=sample_count)
        M = K_pres.shape[1]
        gram = torch.matmul(K_pres.t(), K_pres) + reg_lambda * torch.eye(M, device=K_pres.device)
        inv_gram = torch.linalg.pinv(gram)

        # Projected keys: K_null = K_mat - K_pres (inv_gram (K_pres^T K_mat))
        proj_coeffs = torch.matmul(inv_gram, torch.matmul(K_pres.t(), K_mat))  # [M, N]
        K_null = K_mat - torch.matmul(K_pres, proj_coeffs)  # [in_dim, N]

        # 3. Solve for Delta W: Delta W = Delta_V (K_null^T K_mat + epsilon I)^-1 K_null^T
        N = len(requests)
        coupling = torch.matmul(K_null.t(), K_mat) + 1e-4 * torch.eye(N, device=K_mat.device)
        inv_coupling = torch.linalg.pinv(coupling)  # [N, N]

        delta_W = torch.matmul(delta_V, torch.matmul(inv_coupling, K_null.t()))  # [hidden_dim, in_dim]
        delta_norm = float(torch.norm(delta_W).item())

        with torch.no_grad():
            W.add_(delta_W)

        elapsed = time.time() - start_time
        time_per_edit = elapsed / max(1, N)

        # Measure post probabilities and build results
        results = []
        for i, req in enumerate(requests):
            p_post_target, p_post_old = self.measure_target_probabilities(
                req.prompt, req.target_new, req.ground_truth
            )
            p_pre_t, p_pre_o = pre_probs[i]
            success = p_post_target > p_pre_t

            res = EditResult(
                request_id=req.request_id,
                editor_name=self.name,
                success=success,
                execution_time_sec=time_per_edit,
                pre_edit_target_prob=p_pre_t,
                post_edit_target_prob=p_post_target,
                pre_edit_old_prob=p_pre_o,
                post_edit_old_prob=p_post_old,
                delta_weight_norm=delta_norm,
                checkpoint_id=ckpt_id,
                details={
                    "target_layer": target_layer,
                    "batch_size": N,
                    "null_space_dim": K_pres.shape[1],
                    "delta_norm": delta_norm,
                },
            )
            self.edit_history.append(res)
            results.append(res)

        return results
