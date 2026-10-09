"""Constrained Layer Fine-Tuning (FT-L) baseline."""

from __future__ import annotations
from typing import Dict, Any, Optional
import time
import torch
import torch.nn.functional as F

from editmind.core.base_editor import BaseKnowledgeEditor
from editmind.core.types import EditRequest, EditResult
from editmind.core.registry import register_editor


@register_editor("ft_l")
class FTLEditor(BaseKnowledgeEditor):
    """Constrained Fine-Tuning with L-inf / L2 norm projection."""

    def __init__(self, model_wrapper: Any, config: Optional[Dict[str, Any]] = None):
        super().__init__(model_wrapper, config)
        self.target_layer = self.config.get("target_layer", 2)
        self.num_steps = self.config.get("num_steps", 20)
        self.lr = self.config.get("lr", 1e-2)
        self.max_norm = self.config.get("max_norm", 0.05)
        self.norm_type = self.config.get("norm_constraint", "l_inf")

    @property
    def name(self) -> str:
        return "ft_l"

    def edit(self, request: EditRequest) -> EditResult:
        start_time = time.time()
        ckpt_id = self.create_checkpoint()

        p_pre_target, p_pre_old = self.measure_target_probabilities(
            request.prompt, request.target_new, request.ground_truth
        )

        model = self.model_wrapper.model
        tokenizer = self.model_wrapper.tokenizer
        target_id = tokenizer.token2id.get(request.target_new)
        if target_id is None:
            target_id = tokenizer._add_token(request.target_new)

        input_ids = self.model_wrapper.tokenize(request.prompt)
        target_tensor = torch.tensor([target_id], device=input_ids.device)

        mlp = model.layers[self.target_layer].mlp
        target_weight = mlp.down_proj.weight
        orig_weight = target_weight.detach().clone()

        # Optimize only target layer down_proj
        optimizer = torch.optim.Adam([target_weight], lr=self.lr)

        for step in range(self.num_steps):
            optimizer.zero_grad()
            logits, _ = model(input_ids)
            loss = F.cross_entropy(logits[0, -1, :].unsqueeze(0), target_tensor)
            loss.backward()
            optimizer.step()

            # Norm constraint projection
            with torch.no_grad():
                delta = target_weight - orig_weight
                if self.norm_type == "l_inf":
                    delta_clamped = torch.clamp(delta, -self.max_norm, self.max_norm)
                else: # l2 norm
                    norm = torch.norm(delta)
                    if norm > self.max_norm:
                        delta_clamped = delta * (self.max_norm / norm)
                    else:
                        delta_clamped = delta
                target_weight.copy_(orig_weight + delta_clamped)

        delta_final = torch.norm(target_weight - orig_weight).item()
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
            delta_weight_norm=float(delta_final),
            checkpoint_id=ckpt_id,
            details={
                "target_layer": self.target_layer,
                "norm_type": self.norm_type,
                "max_norm": self.max_norm,
            },
        )
        self.edit_history.append(result)
        return result
