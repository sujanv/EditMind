"""Rank-One Model Editing (ROME) implementation."""

from __future__ import annotations
from typing import Dict, Any, Optional
import time
import torch

from editmind.core.base_editor import BaseKnowledgeEditor
from editmind.core.types import EditRequest, EditResult
from editmind.core.registry import register_editor
from editmind.editors.rome.compute_v import compute_v_target


@register_editor("rome")
class ROMEEditor(BaseKnowledgeEditor):
    """Rank-One Model Editing (Locate-and-Edit) for causal language models."""

    @property
    def name(self) -> str:
        return "rome"

    def edit(self, request: EditRequest) -> EditResult:
        start_time = time.time()
        ckpt_id = self.create_checkpoint()

        target_layer = self.config.get("target_layer", 3)
        v_steps = self.config.get("v_num_grad_steps", 25)
        v_lr = self.config.get("v_lr", 0.1)
        clamp_norm = self.config.get("clamp_norm_factor", 4.0)

        # Pre-edit verification
        p_pre_target, p_pre_old = self.measure_target_probabilities(
            request.prompt, request.target_new, request.ground_truth
        )

        # Subject token index
        subj_str = request.subject or request.prompt.split()[-1]
        input_ids = self.model_wrapper.tokenize(request.prompt)
        subj_ids = self.model_wrapper.tokenize(subj_str)
        subj_idx = max(0, input_ids.shape[1] - 2)

        # Step 1: Optimize target vector v* and extract key k*
        k_star, v_star = compute_v_target(
            model_wrapper=self.model_wrapper,
            prompt=request.prompt,
            target_token=request.target_new,
            target_layer_idx=target_layer,
            subject_token_idx=subj_idx,
            num_steps=v_steps,
            lr=v_lr,
        )

        # Step 2: Rank-one closed-form weight update on MLP down_proj
        # down_proj: intermediate_dim -> hidden_dim
        # down_proj(x) = x @ W^T + b => W has shape [hidden_dim, intermediate_dim]
        mlp = self.model_wrapper.model.layers[target_layer].mlp
        W = mlp.down_proj.weight  # Shape: [hidden_dim, intermediate_dim]

        # Residual vector: v* - W @ k*
        cur_v = torch.matmul(W, k_star)
        residual = v_star - cur_v

        # Covariance C = I (identity assumption or uncentered empirical covariance)
        denom = torch.dot(k_star, k_star) + 1e-7
        # delta_W = residual (x) k*^T / (k*^T k*)
        delta_W = torch.outer(residual, k_star) / denom

        # Optional norm clamping
        delta_norm = float(torch.norm(delta_W).item())
        orig_norm = float(torch.norm(W).item())
        if delta_norm > clamp_norm * orig_norm:
            delta_W = delta_W * (clamp_norm * orig_norm / delta_norm)
            delta_norm = float(torch.norm(delta_W).item())

        # Apply update
        with torch.no_grad():
            mlp.down_proj.weight.add_(delta_W)

        # Post-edit verification
        p_post_target, p_post_old = self.measure_target_probabilities(
            request.prompt, request.target_new, request.ground_truth
        )

        elapsed = time.time() - start_time
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
                "v_steps": v_steps,
                "delta_norm": delta_norm,
            },
        )
        self.edit_history.append(result)
        return result
