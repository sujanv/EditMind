"""AlphaEdit: Null-Space Projection Knowledge Editing."""

from __future__ import annotations
from typing import Dict, Any, Optional, List
import time
import torch

from editmind.core.base_editor import BaseKnowledgeEditor
from editmind.core.types import EditRequest, EditResult
from editmind.core.registry import register_editor
from editmind.editors.rome.compute_v import compute_v_target


@register_editor("alphaedit")
class AlphaEditEditor(BaseKnowledgeEditor):
    """AlphaEdit: Projects weight updates onto the null space of preserved activation representations."""

    @property
    def name(self) -> str:
        return "alphaedit"

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

        # Shape: [in_dim, num_samples]
        return torch.stack(keys, dim=1)

    def edit(self, request: EditRequest) -> EditResult:
        start_time = time.time()
        ckpt_id = self.create_checkpoint()

        target_layer = self.config.get("target_layer", 2)
        v_steps = self.config.get("v_num_grad_steps", 25)
        v_lr = self.config.get("v_lr", 0.4)
        sample_count = self.config.get("null_space_preserve_samples", 10)
        reg_lambda = self.config.get("reg_lambda", 1e-3)

        p_pre_target, p_pre_old = self.measure_target_probabilities(
            request.prompt, request.target_new, request.ground_truth
        )

        input_ids = self.model_wrapper.tokenize(request.prompt)
        subj_idx = max(0, input_ids.shape[1] - 2)

        # 1. Compute target vector v* and key k*
        k_star, v_star = compute_v_target(
            model_wrapper=self.model_wrapper,
            prompt=request.prompt,
            target_token=request.target_new,
            target_layer_idx=target_layer,
            subject_token_idx=subj_idx,
            num_steps=v_steps,
            lr=v_lr,
            weight_decay=1e-3,
        )

        mlp = self.model_wrapper.model.layers[target_layer].mlp
        W = mlp.down_proj.weight # [hidden_dim, in_dim]
        cur_v = torch.matmul(W, k_star)
        residual = v_star - cur_v # [hidden_dim]

        # 2. Compute null-space projector: P_null = I - K (K^T K + lambda I)^-1 K^T
        K_preserve = self._sample_preserve_keys(target_layer, sample_count=sample_count) # [in_dim, M]
        M = K_preserve.shape[1]
        gram = torch.matmul(K_preserve.t(), K_preserve) + reg_lambda * torch.eye(M, device=K_preserve.device)
        inv_gram = torch.linalg.pinv(gram) # [M, M]

        # Project k_star onto null space: k_proj = k_star - K_preserve @ (K_preserve^T K_preserve)^-1 @ (K_preserve^T k_star)
        proj_coeff = torch.matmul(inv_gram, torch.matmul(K_preserve.t(), k_star)) # [M]
        k_null = k_star - torch.matmul(K_preserve, proj_coeff) # [in_dim]

        # If k_null is non-zero, use it; else fallback to regularized k_star
        norm_null = torch.norm(k_null).item()
        if norm_null > 1e-4:
            denom = torch.dot(k_null, k_star) + 1e-6
            delta_W = torch.outer(residual, k_null) / denom
        else:
            denom = torch.dot(k_star, k_star) + 1e-6
            delta_W = torch.outer(residual, k_star) / denom

        delta_norm = float(torch.norm(delta_W).item())
        with torch.no_grad():
            W.add_(delta_W)

        elapsed = time.time() - start_time
        p_post_target, p_post_old = self.measure_target_probabilities(
            request.prompt, request.target_new, request.ground_truth
        )
        success = p_post_target > p_pre_target

        result = EditResult(
            request_id=request.request_id,
            editor_name=self.name,
            success=success,
            execution_time_sec=elapsed,
            pre_edit_target_prob=p_pre_target,
            post_edit_target_prob=p_post_target,
            pre_edit_old_prob=p_pre_old,
            post_edit_old_prob=p_post_old,
            delta_weight_norm=delta_norm,
            checkpoint_id=ckpt_id,
            details={
                "target_layer": target_layer,
                "null_space_dim": K_preserve.shape[1],
                "delta_norm": delta_norm,
            },
        )
        self.edit_history.append(result)
        return result
