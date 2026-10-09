"""PyTorch Forward Hooks for activation caching, intervention, and inspection."""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Callable
import torch
import torch.nn as nn
from contextlib import contextmanager


class ActivationCacher:
    """Collects and stores module activations during forward execution."""

    def __init__(self, modules: Dict[str, nn.Module]):
        self.modules = modules
        self.activations: Dict[str, List[torch.Tensor]] = {}
        self._handles: List[Any] = []

    def _make_hook(self, name: str):
        def hook(module, input_tensor, output_tensor):
            out = output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor
            if name not in self.activations:
                self.activations[name] = []
            self.activations[name].append(out.detach().clone())
        return hook

    def __enter__(self):
        self.activations.clear()
        for name, module in self.modules.items():
            handle = module.register_forward_hook(self._make_hook(name))
            self._handles.append(handle)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


@contextmanager
def hook_activation_intervention(
    module: nn.Module,
    intervention_fn: Callable[[torch.Tensor], torch.Tensor],
):
    """Context manager to intercept and modify module output."""
    def forward_hook(mod, inp, out):
        if isinstance(out, tuple):
            modified = intervention_fn(out[0])
            return (modified,) + out[1:]
        return intervention_fn(out)

    handle = module.register_forward_hook(forward_hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def hook_activation_replacement(
    module: nn.Module,
    target_token_idx: int,
    replacement_vector: torch.Tensor,
):
    """Replaces activation at a specific sequence position with a replacement vector."""
    def forward_hook(mod, inp, out):
        is_tuple = isinstance(out, tuple)
        tensor = out[0] if is_tuple else out
        tensor_clone = tensor.clone()
        # tensor shape: [batch, seq_len, hidden_dim]
        if target_token_idx < tensor_clone.shape[1]:
            tensor_clone[:, target_token_idx, :] = replacement_vector
        if is_tuple:
            return (tensor_clone,) + out[1:]
        return tensor_clone

    handle = module.register_forward_hook(forward_hook)
    try:
        yield
    finally:
        handle.remove()
