"""LoRA-based Parameter-Efficient Knowledge Editing."""

from __future__ import annotations
from typing import Dict, Any, Optional
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from editmind.core.base_editor import BaseKnowledgeEditor
from editmind.core.types import EditRequest, EditResult
from editmind.core.registry import register_editor


class LoRALinear(nn.Module):
    """Wraps a linear module with Low-Rank Adapter matrices."""

    def __init__(self, base_linear: nn.Linear, rank: int = 4, alpha: float = 8.0):
        super().__init__()
        self.base_linear = base_linear
        self.rank = rank
        self.scaling = alpha / rank

        in_dim = base_linear.in_features
        out_dim = base_linear.out_features

        self.lora_A = nn.Parameter(torch.zeros(rank, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out

    def merge_to_base(self):
        with torch.no_grad():
            self.base_linear.weight.add_(torch.matmul(self.lora_B, self.lora_A) * self.scaling)


@register_editor("lora_edit")
class LoRAEditor(BaseKnowledgeEditor):
    """LoRA Knowledge Editor."""

    def __init__(self, model_wrapper: Any, config: Optional[Dict[str, Any]] = None):
        super().__init__(model_wrapper, config)
        self.target_layer = self.config.get("target_layer", 2)
        self.rank = self.config.get("rank", 4)
        self.lr = self.config.get("lr", 5e-2)
        self.num_steps = self.config.get("num_steps", 20)

    @property
    def name(self) -> str:
        return "lora_edit"

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
        orig_down = mlp.down_proj
        lora_layer = LoRALinear(orig_down, rank=self.rank)
        mlp.down_proj = lora_layer

        optimizer = torch.optim.Adam([lora_layer.lora_A, lora_layer.lora_B], lr=self.lr)

        for step in range(self.num_steps):
            optimizer.zero_grad()
            logits, _ = model(input_ids)
            loss = F.cross_entropy(logits[0, -1, :].unsqueeze(0), target_tensor)
            loss.backward()
            optimizer.step()

        # Merge adapter into base weights and restore original module type
        lora_layer.merge_to_base()
        mlp.down_proj = orig_down

        delta_norm = float(torch.norm(torch.matmul(lora_layer.lora_B, lora_layer.lora_A) * lora_layer.scaling).item())
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
                "target_layer": self.target_layer,
                "rank": self.rank,
            },
        )
        self.edit_history.append(result)
        return result
