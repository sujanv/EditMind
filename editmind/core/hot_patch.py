"""Dynamic Knowledge Hot-Patching and Selective Delta-Patch Management Engine.

Enables runtime registration, hot-enabling, hot-disabling, scaling, and surgical
unbaking of specific knowledge edit patches without full-model rollbacks or
interfering with subsequent edits.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import time
import uuid
import torch
import numpy as np

from editmind.models.model_wrapper import UnifiedModelWrapper


@dataclass
class HotPatch:
    """Low-rank parameter delta representing a modular knowledge edit."""
    patch_id: str
    subject: str
    prompt: str
    target_new: str
    layer_idx: int
    target_param: str  # e.g. "mlp.down_proj.weight" or "attn.c_proj.weight"
    u: torch.Tensor    # [out_features, rank]
    v: torch.Tensor    # [in_features, rank]
    scaling: float = 1.0
    enabled: bool = True
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def reconstruct_delta(self) -> torch.Tensor:
        """Reconstructs the full delta weight matrix: scaling * (u @ v^T)."""
        return self.scaling * torch.matmul(self.u, self.v.t())

    def to_dict(self) -> Dict[str, Any]:
        delta = self.reconstruct_delta()
        return {
            "patch_id": self.patch_id,
            "subject": self.subject,
            "prompt": self.prompt,
            "target_new": self.target_new,
            "layer_idx": self.layer_idx,
            "target_param": self.target_param,
            "scaling": self.scaling,
            "enabled": self.enabled,
            "delta_norm": round(float(torch.norm(delta).item()), 4),
            "created_at": self.created_at,
        }


class HotPatchEngine:
    """Manages dynamic modular knowledge patches with runtime hot-toggling."""

    def __init__(self, model_wrapper: UnifiedModelWrapper):
        self.wrapper = model_wrapper
        self.patches: Dict[str, HotPatch] = {}
        # Keep base unpatched copies of affected weights for zero-drift reconstruction
        self._base_weights: Dict[Tuple[int, str], torch.Tensor] = {}

    def _get_param_tensor(self, layer_idx: int, target_param: str) -> torch.Tensor:
        layer = self.wrapper.get_layer(layer_idx)
        parts = target_param.split(".")
        curr = layer
        for p in parts[:-1]:
            curr = getattr(curr, p)
        return getattr(curr, parts[-1])

    def register_patch(
        self,
        subject: str,
        prompt: str,
        target_new: str,
        layer_idx: int,
        delta_tensor: torch.Tensor,
        target_param: str = "mlp.down_proj.weight",
        scaling: float = 1.0,
        patch_id: Optional[str] = None,
        rank: int = 2,
    ) -> HotPatch:
        """Decomposes a delta tensor into low-rank factors and registers hot-patch."""
        pid = patch_id or f"patch_{str(uuid.uuid4())[:8]}"

        # Record base weight before first patch on this layer/param
        key = (layer_idx, target_param)
        if key not in self._base_weights:
            orig = self._get_param_tensor(layer_idx, target_param)
            self._base_weights[key] = orig.detach().clone()

        # Low-rank SVD decomposition of delta_tensor: Delta = U @ S @ V^T
        # We absorb sqrt(S) into U and V
        delta = delta_tensor.detach().float()
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        effective_rank = min(rank, len(S))

        sqrt_S = torch.sqrt(S[:effective_rank])
        u_factor = U[:, :effective_rank] * sqrt_S.unsqueeze(0)
        v_factor = Vh[:effective_rank, :].t() * sqrt_S.unsqueeze(0)

        patch = HotPatch(
            patch_id=pid,
            subject=subject,
            prompt=prompt,
            target_new=target_new,
            layer_idx=layer_idx,
            target_param=target_param,
            u=u_factor,
            v=v_factor,
            scaling=scaling,
            enabled=True,
        )
        self.patches[pid] = patch

        # Immediately apply delta to the live model weight
        with torch.no_grad():
            w_tensor = self._get_param_tensor(layer_idx, target_param)
            w_tensor.add_(patch.reconstruct_delta().to(w_tensor.dtype))

        return patch

    def hot_toggle(self, patch_id: str, enabled: bool) -> bool:
        """Surgically toggles a patch on or off in the live model weights."""
        if patch_id not in self.patches:
            raise KeyError(f"Patch ID '{patch_id}' not found.")

        patch = self.patches[patch_id]
        if patch.enabled == enabled:
            return False  # No change

        delta = patch.reconstruct_delta()
        w_tensor = self._get_param_tensor(patch.layer_idx, patch.target_param)

        with torch.no_grad():
            if enabled:
                # Re-enable: add delta back
                w_tensor.add_(delta.to(w_tensor.dtype))
            else:
                # Disable: subtract delta out
                w_tensor.sub_(delta.to(w_tensor.dtype))

        patch.enabled = enabled
        return True

    def set_patch_scaling(self, patch_id: str, scaling: float) -> bool:
        """Dynamically adjusts scaling coefficient of an active hot-patch."""
        if patch_id not in self.patches:
            raise KeyError(f"Patch ID '{patch_id}' not found.")

        patch = self.patches[patch_id]
        old_scaling = patch.scaling
        diff_scaling = scaling - old_scaling

        if patch.enabled and abs(diff_scaling) > 1e-6:
            # Adjust delta proportionally
            patch.scaling = 1.0
            unit_delta = patch.reconstruct_delta()
            w_tensor = self._get_param_tensor(patch.layer_idx, patch.target_param)
            with torch.no_grad():
                w_tensor.add_((unit_delta * diff_scaling).to(w_tensor.dtype))

        patch.scaling = scaling
        return True

    def delete_patch(self, patch_id: str) -> bool:
        """Surgically un-bakes a patch from model weights and deletes it from registry."""
        if patch_id not in self.patches:
            return False

        # If enabled, remove its delta contribution first
        if self.patches[patch_id].enabled:
            self.hot_toggle(patch_id, enabled=False)

        del self.patches[patch_id]
        return True

    def compute_patch_interference_matrix(self) -> Dict[str, Any]:
        """Calculates pairwise cosine similarity between all registered hot-patches."""
        pids = list(self.patches.keys())
        n = len(pids)
        matrix = np.zeros((n, n), dtype=np.float32)

        deltas = [self.patches[pid].reconstruct_delta().flatten() for pid in pids]

        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i, j] = 1.0
                else:
                    norm_i = torch.norm(deltas[i])
                    norm_j = torch.norm(deltas[j])
                    if norm_i > 1e-7 and norm_j > 1e-7:
                        sim = float(torch.dot(deltas[i], deltas[j]) / (norm_i * norm_j))
                        matrix[i, j] = sim

        return {
            "patch_ids": pids,
            "matrix": matrix.round(4).tolist(),
            "max_interference": float(np.max(np.abs(matrix - np.eye(n)))) if n > 1 else 0.0,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Returns overview of all active and inactive registered patches."""
        active = sum(1 for p in self.patches.values() if p.enabled)
        return {
            "total_patches": len(self.patches),
            "active_patches": active,
            "inactive_patches": len(self.patches) - active,
            "patches": [p.to_dict() for p in self.patches.values()],
        }
