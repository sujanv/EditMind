"""Mass-Editing Memory in a Transformer (MEMIT) implementation."""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import time
import torch

from editmind.core.base_editor import BaseKnowledgeEditor
from editmind.core.types import EditRequest, EditResult
from editmind.core.registry import register_editor
from editmind.editors.memit.distribute import (
    compute_memit_keys_and_residuals,
    solve_layer_weight_delta,
)


@register_editor("memit")
class MEMITEditor(BaseKnowledgeEditor):
    """Mass-Editing Memory in a Transformer (MEMIT) across multiple layers."""

    @property
    def name(self) -> str:
        return "memit"

    def edit(self, request: EditRequest) -> EditResult:
        """Single-request wrapper around batch editing."""
        batch_results = self.batch_edit([request])
        return batch_results[0]

    def batch_edit(self, requests: List[EditRequest]) -> List[EditResult]:
        """Edits multiple facts simultaneously by distributing residual updates across layers."""
        start_time = time.time()
        ckpt_id = self.create_checkpoint()

        layers = self.config.get("layers", [2])
        v_steps = self.config.get("v_num_grad_steps", 30)
        v_lr = self.config.get("v_lr", 0.5)

        # Pre-edit measurements
        pre_metrics = []
        for req in requests:
            p_target, p_old = self.measure_target_probabilities(
                req.prompt, req.target_new, req.ground_truth
            )
            pre_metrics.append((p_target, p_old))

        prompts = [req.prompt for req in requests]
        targets = [req.target_new for req in requests]
        subjects = [req.subject or req.prompt.split()[-1] for req in requests]

        # 1. Compute keys and target representations for each layer
        keys_by_layer, targets_by_layer = compute_memit_keys_and_residuals(
            model_wrapper=self.model_wrapper,
            prompts=prompts,
            targets=targets,
            subjects=subjects,
            target_layers=layers,
            v_steps=v_steps,
            v_lr=v_lr,
        )

        # 2. Distribute residuals across layers
        num_layers = len(layers)
        total_delta_norm = 0.0

        for idx, layer_idx in enumerate(layers):
            mlp = self.model_wrapper.model.layers[layer_idx].mlp
            W = mlp.down_proj.weight # [hidden_dim, intermediate_dim]
            K_l = keys_by_layer[idx] # [intermediate_dim, N]
            V_star_l = targets_by_layer[idx] # [hidden_dim, N]

            # Current predictions at layer
            cur_outputs = torch.matmul(W, K_l) # [hidden_dim, N]
            
            # Residual fraction for this layer
            fraction = 1.0 / float(num_layers)
            R_l = (V_star_l - cur_outputs) * fraction

            delta_W = solve_layer_weight_delta(K=K_l, R=R_l, reg_lambda=1e-3)
            delta_norm = float(torch.norm(delta_W).item())
            total_delta_norm += delta_norm

            with torch.no_grad():
                mlp.down_proj.weight.add_(delta_W)

        elapsed = time.time() - start_time

        # Post-edit measurements
        results: List[EditResult] = []
        for i, req in enumerate(requests):
            p_post_target, p_post_old = self.measure_target_probabilities(
                req.prompt, req.target_new, req.ground_truth
            )
            p_pre_target, p_pre_old = pre_metrics[i]
            success = p_post_target > p_pre_target

            res = EditResult(
                request_id=req.request_id,
                editor_name=self.name,
                success=success,
                execution_time_sec=elapsed / len(requests),
                pre_edit_target_prob=p_pre_target,
                post_edit_target_prob=p_post_target,
                pre_edit_old_prob=p_pre_old,
                post_edit_old_prob=p_post_old,
                delta_weight_norm=total_delta_norm,
                checkpoint_id=ckpt_id,
                details={
                    "distributed_layers": layers,
                    "batch_size": len(requests),
                },
            )
            self.edit_history.append(res)
            results.append(res)

        return results
