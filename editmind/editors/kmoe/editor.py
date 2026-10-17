"""Knowledge Mixture of Experts Editor (K-MoE).

Routes continual knowledge edits to domain-specialized low-rank experts
via a dynamic gating network to prevent cross-domain parameter interference.
"""

from dataclasses import dataclass, field
import math
import time
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from editmind.core.base_editor import BaseKnowledgeEditor
from editmind.core.registry import register_editor
from editmind.core.types import EditRequest, EditResult
from editmind.models.model_wrapper import BaseModelWrapper


class LowRankExpert(nn.Module):
    """Individual low-rank adapter expert specialized in a knowledge domain."""

    def __init__(self, in_features: int, out_features: int, rank: int = 4, scale: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scale = scale

        self.adapter_A = nn.Parameter(torch.randn(rank, in_features) * (1.0 / math.sqrt(in_features)))
        self.adapter_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, in_features]
        delta = (x @ self.adapter_A.t()) @ self.adapter_B.t()
        return delta * (self.scale / max(1, self.rank))


class KMoERoutingLayer(nn.Module):
    """Dynamic Gating Network and Mixture-of-Experts Module for a transformer layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int = 4,
        top_k: int = 1,
        rank: int = 4,
        scale: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)

        # Routing gate projection: W_gate: in_features -> num_experts
        self.gate = nn.Linear(in_features, num_experts, bias=False)
        nn.init.normal_(self.gate.weight, std=0.02)

        # Expert ensemble
        self.experts = nn.ModuleList([
            LowRankExpert(in_features, out_features, rank=rank, scale=scale)
            for _ in range(num_experts)
        ])
        self.is_active = True

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns: (expert_output, top_k_indices, routing_weights)."""
        if not self.is_active:
            zero_out = torch.zeros(*x.shape[:-1], self.out_features, device=x.device, dtype=x.dtype)
            return zero_out, torch.zeros(1, dtype=torch.long), torch.zeros(1)

        # Pool sequence to compute routing logits: [batch, in_features]
        pooled_x = x.mean(dim=1) if x.dim() == 3 else x
        gate_logits = self.gate(pooled_x)  # [batch, num_experts]
        gate_probs = F.softmax(gate_logits, dim=-1)

        # Select Top-K experts
        topk_weights, topk_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        # Re-normalize top-k weights
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Dispatch through selected experts
        combined_output = torch.zeros(*x.shape[:-1], self.out_features, device=x.device, dtype=x.dtype)
        batch_size = x.shape[0]

        for b in range(batch_size):
            for k_idx in range(self.top_k):
                expert_idx = int(topk_indices[b, k_idx].item())
                w = topk_weights[b, k_idx]
                expert_out = self.experts[expert_idx](x[b : b + 1])
                combined_output[b : b + 1] += w * expert_out

        return combined_output, topk_indices, topk_weights


@register_editor("kmoe")
class KnowledgeMoEEditor(BaseKnowledgeEditor):
    """Knowledge Mixture-of-Experts (K-MoE) Lifelong Editor."""

    def __init__(
        self,
        model_wrapper: BaseModelWrapper,
        config: Optional[Dict[str, Any]] = None,
        num_experts: int = 4,
        top_k: int = 1,
        rank: int = 4,
        target_layer: int = 2,
        lr: float = 0.05,
        steps: int = 30,
        **kwargs,
    ):
        super().__init__(model_wrapper, config)
        cfg = config or {}
        self.num_experts = kwargs.get("num_experts", cfg.get("num_experts", num_experts))
        self.top_k = kwargs.get("top_k", cfg.get("top_k", top_k))
        self.rank = kwargs.get("rank", cfg.get("rank", rank))
        self.target_layer = kwargs.get("target_layer", cfg.get("target_layer", target_layer))
        self.lr = kwargs.get("lr", cfg.get("lr", lr))
        self.steps = kwargs.get("steps", cfg.get("steps", steps))

        # Map of installed K-MoE routing modules per layer
        self.installed_layers: Dict[int, KMoERoutingLayer] = {}
        self.hooks: List[Any] = []
        self._ensure_kmoe_attached(self.target_layer)

    @property
    def name(self) -> str:
        return "kmoe"

    def _ensure_kmoe_attached(self, layer_idx: int) -> KMoERoutingLayer:
        """Installs K-MoE routing module on the target layer if not already present."""
        if layer_idx in self.installed_layers:
            return self.installed_layers[layer_idx]

        layer = self.model_wrapper.get_layer(layer_idx)
        # Determine hidden dim
        hidden_dim = getattr(self.model_wrapper.model, "hidden_dim", None)
        if hidden_dim is None:
            # Fallback to inspecting layer weights
            for p in layer.parameters():
                if p.dim() == 2:
                    hidden_dim = min(p.shape[0], p.shape[1])
                    break
        hidden_dim = hidden_dim or 64

        moe_module = KMoERoutingLayer(
            in_features=hidden_dim,
            out_features=hidden_dim,
            num_experts=self.num_experts,
            top_k=self.top_k,
            rank=self.rank,
        ).to(self.model_wrapper.device)

        # Hook to inject expert residual into layer forward output
        def hook_fn(module, inp, output):
            x = inp[0] if isinstance(inp, tuple) else inp
            residual, _, _ = moe_module(x)
            if isinstance(output, tuple):
                return (output[0] + residual,) + output[1:]
            return output + residual

        hook = layer.register_forward_hook(hook_fn)
        self.hooks.append(hook)
        self.installed_layers[layer_idx] = moe_module
        return moe_module

    def edit(self, request: EditRequest) -> EditResult:
        """Applies knowledge update to the selected expert for request.subject."""
        start_time = time.time()
        moe_module = self._ensure_kmoe_attached(self.target_layer)

        # Measure baseline probability
        probs_pre = self.model_wrapper.predict_next_token_probs(request.prompt)
        p_pre_target = probs_pre.get(request.target_new, 0.0)

        # Identify routing decision for prompt
        input_ids = self.model_wrapper.tokenize(request.prompt)
        target_ids = self.model_wrapper.tokenizer.encode(request.target_new)
        target_token_id = target_ids[0] if target_ids else 0

        # Run forward pass to find expert chosen by gate
        with torch.no_grad():
            if hasattr(self.model_wrapper.model, "wte"):
                embeds = self.model_wrapper.model.wte(input_ids)
                pooled = embeds.mean(dim=1)
            else:
                pooled = torch.ones(1, moe_module.in_features, device=self.model_wrapper.device)
            gate_logits = moe_module.gate(pooled)
            chosen_expert_idx = int(torch.argmax(gate_logits, dim=-1).item())

        expert = moe_module.experts[chosen_expert_idx]

        # Train parameters of the assigned expert specifically
        optimizer = torch.optim.AdamW(
            [expert.adapter_A, expert.adapter_B],
            lr=self.lr,
            weight_decay=1e-4,
        )

        for step in range(self.steps):
            optimizer.zero_grad()
            out = self.model_wrapper.model(input_ids)
            logits = out[0] if isinstance(out, tuple) else (out.logits if hasattr(out, "logits") else out)
            loss = F.cross_entropy(logits[0, -1:], torch.tensor([target_token_id], device=self.model_wrapper.device))
            loss.backward()
            optimizer.step()

        probs_post = self.model_wrapper.predict_next_token_probs(request.prompt)
        p_post_target = probs_post.get(request.target_new, 0.0)

        elapsed = time.time() - start_time
        success = p_post_target > p_pre_target or p_post_target >= 0.20

        metadata = {
            "assigned_expert": chosen_expert_idx,
            "total_experts": self.num_experts,
            "target_layer": self.target_layer,
            "adapter_rank": self.rank,
        }

        return EditResult(
            request_id=request.request_id,
            editor_name=self.name,
            success=success,
            execution_time_sec=elapsed,
            pre_edit_target_prob=p_pre_target,
            post_edit_target_prob=p_post_target,
            details=metadata,
        )
