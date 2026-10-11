"""Knowledge Delta Checkpointing, Atomic Rollback, and Patch Serialization Engine.

Enables saving, applying, and rolling back lightweight weight deltas (Delta W)
produced by knowledge editing methods without re-instantiating full base models.
"""

from dataclasses import dataclass, field
import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional
import torch

from editmind.core.types import KnowledgeTriple, EditRequest
from editmind.models.model_wrapper import BaseModelWrapper


@dataclass
class EditPatch:
    """Represents a lightweight, serializable knowledge edit delta."""
    patch_id: str
    editor_name: str
    target_triple: Dict[str, Any]
    layer_deltas: Dict[str, torch.Tensor]  # parameter_name -> delta_tensor
    timestamp: float = field(default_factory=time.time)
    sha256: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def compute_hash(self) -> str:
        """Computes SHA-256 fingerprint of parameter deltas and metadata."""
        hasher = hashlib.sha256()
        hasher.update(self.patch_id.encode())
        hasher.update(self.editor_name.encode())
        hasher.update(json.dumps(self.target_triple, sort_keys=True).encode())
        for param_name in sorted(self.layer_deltas.keys()):
            hasher.update(param_name.encode())
            hasher.update(self.layer_deltas[param_name].cpu().numpy().tobytes())
        return hasher.hexdigest()

    def verify_integrity(self) -> bool:
        """Verifies delta tensor cryptographic signature."""
        return self.sha256 == self.compute_hash()


class CheckpointManager:
    """Manages knowledge delta snapshots, rollbacks, and fast-forward state restoration."""

    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = storage_dir
        if storage_dir:
            os.makedirs(storage_dir, exist_ok=True)
        self.patch_history: List[EditPatch] = []

    def capture_deltas(
        self,
        model_before: BaseModelWrapper,
        model_after: BaseModelWrapper,
        editor_name: str,
        target_triple: KnowledgeTriple,
        patch_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EditPatch:
        """Computes Delta W = W_after - W_before across all modified parameters."""
        deltas: Dict[str, torch.Tensor] = {}

        params_before = dict(model_before.model.named_parameters())
        params_after = dict(model_after.model.named_parameters())

        for name, p_after in params_after.items():
            if name in params_before:
                p_before = params_before[name]
                diff = p_after.detach() - p_before.detach()
                if torch.any(torch.abs(diff) > 1e-8):
                    deltas[name] = diff.cpu().clone()

        pid = patch_id or f"patch_{int(time.time())}_{len(self.patch_history)+1}"
        patch = EditPatch(
            patch_id=pid,
            editor_name=editor_name,
            target_triple=target_triple.to_dict(),
            layer_deltas=deltas,
            timestamp=time.time(),
            metadata=metadata or {},
        )
        patch.sha256 = patch.compute_hash()
        self.patch_history.append(patch)
        return patch

    def apply_patch(self, model: BaseModelWrapper, patch: EditPatch) -> None:
        """Applies delta forward: W <- W + Delta W."""
        if not patch.verify_integrity():
            raise ValueError(f"Patch {patch.patch_id} failed cryptographic SHA-256 integrity verification.")

        params = dict(model.model.named_parameters())
        with torch.no_grad():
            for name, delta in patch.layer_deltas.items():
                if name in params:
                    params[name].add_(delta.to(params[name].device))

    def rollback_patch(self, model: BaseModelWrapper, patch: EditPatch) -> None:
        """Reverses delta: W <- W - Delta W."""
        if not patch.verify_integrity():
            raise ValueError(f"Patch {patch.patch_id} failed cryptographic SHA-256 integrity verification.")

        params = dict(model.model.named_parameters())
        with torch.no_grad():
            for name, delta in patch.layer_deltas.items():
                if name in params:
                    params[name].sub_(delta.to(params[name].device))

    def rollback_latest(self, model: BaseModelWrapper) -> Optional[EditPatch]:
        """Rolls back the most recent edit patch from the history."""
        if not self.patch_history:
            return None
        latest = self.patch_history.pop()
        self.rollback_patch(model, latest)
        return latest

    def rollback_all(self, model: BaseModelWrapper) -> int:
        """Rolls back all patches in reverse order, restoring original model state."""
        count = 0
        while self.patch_history:
            self.rollback_latest(model)
            count += 1
        return count

    def save_patch(self, patch: EditPatch, filepath: str) -> None:
        """Saves an edit patch to disk."""
        data = {
            "patch_id": patch.patch_id,
            "editor_name": patch.editor_name,
            "target_triple": patch.target_triple,
            "timestamp": patch.timestamp,
            "sha256": patch.sha256,
            "metadata": patch.metadata,
            "layer_deltas": patch.layer_deltas,
        }
        torch.save(data, filepath)

    def load_patch(self, filepath: str) -> EditPatch:
        """Loads an edit patch from disk and verifies integrity."""
        data = torch.load(filepath, map_location="cpu", weights_only=False)
        patch = EditPatch(
            patch_id=data["patch_id"],
            editor_name=data["editor_name"],
            target_triple=data["target_triple"],
            layer_deltas=data["layer_deltas"],
            timestamp=data["timestamp"],
            sha256=data["sha256"],
            metadata=data.get("metadata", {}),
        )
        if not patch.verify_integrity():
            raise ValueError(f"Loaded patch {patch.patch_id} from {filepath} is corrupted or tampered.")
        return patch
