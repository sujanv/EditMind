"""Quantized Knowledge Editor (Q-Edit) for Low-Bit Quantized Large Language Models.

Enables knowledge modification on 4-bit and 8-bit quantized models without dequantizing
the base neural weights, using low-rank side-channel residual adapters (B * A) optimized
against target token probabilities.
"""

from dataclasses import dataclass, field
import math
import time
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from editmind.core.base_editor import BaseKnowledgeEditor
from editmind.core.registry import register_editor
from editmind.core.types import EditRequest, EditResult
from editmind.models.model_wrapper import BaseModelWrapper


class QuantizedLinearAdapter(nn.Module):
    """Low-rank side-channel residual adapter attached to a quantized linear layer."""

    def __init__(self, in_features: int, out_features: int, rank: int = 4, scale: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scale = scale

        # Low-rank factor matrices: Delta W = (scale / rank) * (B @ A)
        self.adapter_A = nn.Parameter(torch.randn(rank, in_features) * (1.0 / math.sqrt(in_features)))
        self.adapter_B = nn.Parameter(torch.zeros(out_features, rank))
        self.is_active = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_active:
            return torch.zeros(*x.shape[:-1], self.out_features, device=x.device, dtype=x.dtype)
        # x: [batch, seq, in_features]
        # output: [batch, seq, out_features]
        delta = (x @ self.adapter_A.t()) @ self.adapter_B.t()
        return delta * (self.scale / max(1, self.rank))


def simulate_weight_quantization(tensor: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """Simulates uniform affine quantization to N bits (e.g., 4-bit or 8-bit)."""
    qmin = -(2 ** (bits - 1))
    qmax = (2 ** (bits - 1)) - 1
    max_val = torch.max(torch.abs(tensor))
    if max_val == 0:
        return tensor

    scale = max_val / qmax
    q_tensor = torch.clamp(torch.round(tensor / scale), qmin, qmax)
    return q_tensor * scale


@register_editor("qedit")
class QuantizedKnowledgeEditor(BaseKnowledgeEditor):
    """Quantized Knowledge Editor for memory-constrained and quantized LLMs."""

    def __init__(
        self,
        model_wrapper: BaseModelWrapper,
        config: Optional[Dict[str, Any]] = None,
        target_layer: Optional[int] = None,
        rank: Optional[int] = None,
        bits: Optional[int] = None,
        steps: Optional[int] = None,
        lr: Optional[float] = None,
        **kwargs,
    ):
        cfg = dict(config or {})
        super().__init__(model_wrapper, config=cfg)
        self.target_layer = target_layer if target_layer is not None else cfg.get("target_layer", max(0, self.model_wrapper.num_layers // 2))
        self.rank = rank if rank is not None else cfg.get("rank", 4)
        self.bits = bits if bits is not None else cfg.get("bits", 4)
        self.steps = steps if steps is not None else cfg.get("steps", 35)
        self.lr = lr if lr is not None else cfg.get("lr", 0.05)
        self.installed_adapters: Dict[int, QuantizedLinearAdapter] = {}

    @property
    def name(self) -> str:
        return "qedit"

    def _get_target_linear(self, layer_idx: int) -> Optional[nn.Linear]:
        """Extracts the MLP out-projection layer in the transformer."""
        layer = self.model_wrapper.get_layer(layer_idx)
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "c_proj"):
            return layer.mlp.c_proj
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "dense_4h_to_h"):
            return layer.mlp.dense_4h_to_h
        for m in layer.modules():
            if isinstance(m, nn.Linear):
                return m
        return None

    def edit(self, request: EditRequest) -> EditResult:
        """Applies a knowledge edit to the quantized model via low-rank residual adaptation."""
        start_time = time.time()
        prompt = request.prompt
        target_new = request.target_new
        old_target = request.ground_truth

        # 1. Measure initial probabilities
        pre_new_prob, pre_old_prob = self.measure_target_probabilities(prompt, target_new, old_target)

        # 2. Get target linear module and simulate quantized base weights
        linear_module = self._get_target_linear(self.target_layer)
        if linear_module is None:
            return EditResult(
                request_id=request.request_id,
                editor_name="Q-Edit",
                success=False,
                execution_time_sec=time.time() - start_time,
                pre_edit_target_prob=pre_new_prob,
                post_edit_target_prob=pre_new_prob,
                details={"error": "Target linear layer not found for adapter hook."},
            )

        # Ensure base weights are frozen and simulate quantization
        linear_module.weight.requires_grad = False
        with torch.no_grad():
            linear_module.weight.data = simulate_weight_quantization(linear_module.weight.data, bits=self.bits)

        # 3. Attach QuantizedLinearAdapter
        device = self.model_wrapper.device
        in_dim = linear_module.in_features
        out_dim = linear_module.out_features
        adapter = QuantizedLinearAdapter(in_features=in_dim, out_features=out_dim, rank=self.rank).to(device)

        # Register forward hook to inject adapter output into linear module
        def adapter_hook(module, inp, out):
            return out + adapter(inp[0])

        hook_handle = linear_module.register_forward_hook(adapter_hook)

        # 4. Optimize adapter parameters (B, A) to maximize P(target_new)
        optimizer = torch.optim.Adam([adapter.adapter_A, adapter.adapter_B], lr=self.lr)
        tokenizer = self.model_wrapper.tokenizer
        target_tokens = tokenizer.encode(target_new)
        target_token_id = target_tokens[0] if target_tokens else 0
        prompt_tokens = tokenizer.encode(prompt)
        input_ids = torch.tensor([prompt_tokens], device=device)

        for _ in range(self.steps):
            optimizer.zero_grad()
            out = self.model_wrapper.forward(input_ids)
            logits = out if torch.is_tensor(out) else out.logits
            last_logits = logits[0, -1, :]
            loss = F.cross_entropy(last_logits.unsqueeze(0), torch.tensor([target_token_id], device=device))
            l2_reg = 1e-4 * (torch.norm(adapter.adapter_A) + torch.norm(adapter.adapter_B))
            total_loss = loss + l2_reg
            total_loss.backward()
            optimizer.step()

        # 5. Measure post-edit probabilities
        post_new_prob, post_old_prob = self.measure_target_probabilities(prompt, target_new, old_target)

        # Calculate effective delta norm
        with torch.no_grad():
            delta_norm = float(torch.norm(adapter.adapter_B @ adapter.adapter_A).item())

        self.installed_adapters[self.target_layer] = adapter

        success = post_new_prob > pre_new_prob
        res = EditResult(
            request_id=request.request_id,
            editor_name="Q-Edit",
            success=success,
            execution_time_sec=time.time() - start_time,
            pre_edit_target_prob=pre_new_prob,
            post_edit_target_prob=post_new_prob,
            pre_edit_old_prob=pre_old_prob,
            post_edit_old_prob=post_old_prob,
            delta_weight_norm=delta_norm,
            details={
                "target_layer": self.target_layer,
                "quantized_bits": self.bits,
                "rank": self.rank,
                "steps": self.steps,
            },
        )
        self.edit_history.append(res)
        return res
