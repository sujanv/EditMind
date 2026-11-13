"""Residual Error Feedback Knowledge Editor (RefEdit).

Implements closed-loop iterative knowledge editing with residual error feedback.
Unlike single-pass open-loop methods (such as ROME), RefEdit probes the post-edit
forward pass across remaining transformer layers and iteratively computes residual
corrections until the target probability margin is satisfied.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
import time
import torch

from editmind.core.base_editor import BaseKnowledgeEditor
from editmind.core.types import EditRequest, EditResult
from editmind.core.registry import register_editor
from editmind.editors.rome.compute_v import compute_v_target


@register_editor("refedit")
class ResidualErrorFeedbackEditor(BaseKnowledgeEditor):
    """Closed-loop iterative knowledge editing with residual error feedback."""

    def __init__(
        self,
        model_wrapper: any,
        config: Optional[Dict[str, Any]] = None,
        target_layer: int = 2,
        max_feedback_steps: int = 5,
        target_margin: float = 0.3,
        damping_factor: float = 0.5,
        v_num_grad_steps: int = 20,
        v_lr: float = 0.4,
        **kwargs,
    ):
        cfg = dict(config or {})
        cfg.update(kwargs)
        cfg.setdefault("target_layer", target_layer)
        cfg.setdefault("max_feedback_steps", max_feedback_steps)
        cfg.setdefault("target_margin", target_margin)
        cfg.setdefault("damping_factor", damping_factor)
        cfg.setdefault("v_num_grad_steps", v_num_grad_steps)
        cfg.setdefault("v_lr", v_lr)
        super().__init__(model_wrapper=model_wrapper, config=cfg)

    @property
    def name(self) -> str:
        return "refedit"

    def edit(self, request: EditRequest) -> EditResult:
        start_time = time.time()
        ckpt_id = self.create_checkpoint()

        target_layer = self.config.get("target_layer", 2)
        max_steps = self.config.get("max_feedback_steps", 5)
        target_margin = self.config.get("target_margin", 0.3)
        damping = self.config.get("damping_factor", 0.5)
        v_steps = self.config.get("v_num_grad_steps", 20)
        v_lr = self.config.get("v_lr", 0.4)

        # Baseline probabilities
        p_pre_target, p_pre_old = self.measure_target_probabilities(
            request.prompt, request.target_new, request.ground_truth
        )

        input_ids = self.model_wrapper.tokenize(request.prompt)
        subj_idx = max(0, input_ids.shape[1] - 2)

        # Step 1: Initial target vector optimization
        k_star, v_star = compute_v_target(
            model_wrapper=self.model_wrapper,
            prompt=request.prompt,
            target_token=request.target_new,
            target_layer_idx=target_layer,
            subject_token_idx=subj_idx,
            num_steps=v_steps,
            lr=v_lr,
        )

        mlp = self.model_wrapper.model.layers[target_layer].mlp
        W = mlp.down_proj.weight  # [hidden_dim, intermediate_dim]
        denom = torch.dot(k_star, k_star) + 1e-7

        # Initial rank-1 delta
        cur_v = torch.matmul(W, k_star)
        residual = v_star - cur_v
        delta_W_total = torch.outer(residual, k_star) / denom

        with torch.no_grad():
            W.add_(delta_W_total)

        p_curr_target, p_curr_old = self.measure_target_probabilities(
            request.prompt, request.target_new, request.ground_truth
        )

        trajectory: List[Dict[str, float]] = [
            {"step": 0, "p_target": p_curr_target, "p_old": p_curr_old, "margin": p_curr_target - p_curr_old}
        ]

        # Step 2: Closed-loop iterative residual error feedback loop
        step_count = 0
        for step in range(1, max_steps + 1):
            margin = p_curr_target - p_curr_old
            if margin >= target_margin and p_curr_target > p_pre_target:
                # Target margin achieved, early stop
                break

            step_count = step
            # Error feedback: boost target vector proportionally to deficit
            deficit = max(0.05, target_margin - margin)
            error_residual = residual * (deficit * damping)
            delta_corr = torch.outer(error_residual, k_star) / denom

            with torch.no_grad():
                W.add_(delta_corr)
                delta_W_total.add_(delta_corr)

            p_curr_target, p_curr_old = self.measure_target_probabilities(
                request.prompt, request.target_new, request.ground_truth
            )
            trajectory.append({
                "step": step,
                "p_target": p_curr_target,
                "p_old": p_curr_old,
                "margin": p_curr_target - p_curr_old,
            })

        elapsed = time.time() - start_time
        p_post_target = p_curr_target
        p_post_old = p_curr_old
        success = p_post_target > p_pre_target

        delta_norm = float(torch.norm(delta_W_total).item())

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
                "feedback_steps_executed": step_count,
                "initial_margin": float(trajectory[0]["margin"]),
                "final_margin": float(trajectory[-1]["margin"]),
                "trajectory": trajectory,
                "converged": bool(trajectory[-1]["margin"] >= target_margin),
            },
        )
        self.edit_history.append(result)
        return result
