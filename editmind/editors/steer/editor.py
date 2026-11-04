"""Representation Engineering & Activation Steering Editor (ActAdd / Steering Vectors).

Modifies model knowledge and factual responses by adding contrastive activation steering
vectors directly to intermediate transformer layer representations during the forward pass,
achieving zero weight modification (Delta W = 0) and instantaneous reversibility.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from editmind.core.base_editor import BaseKnowledgeEditor
from editmind.core.registry import register_editor
from editmind.core.types import EditRequest, EditResult
from editmind.models.model_wrapper import BaseModelWrapper


@dataclass
class SteeringVector:
    """Stores contrastive activation steering vector and metadata."""
    layer_idx: int
    vector: torch.Tensor
    coefficient: float
    norm_clamped: bool
    target_token: str
    prompt: str


@register_editor("steer")
class SteeringVectorEditor(BaseKnowledgeEditor):
    """Activation Steering Vector (ActAdd) Knowledge Editor."""

    def __init__(
        self,
        model_wrapper: BaseModelWrapper,
        config: Optional[Dict[str, Any]] = None,
        target_layer: int = 2,
        coefficient: float = 1.5,
        max_norm: float = 5.0,
        **kwargs,
    ):
        super().__init__(model_wrapper, config)
        cfg = config or {}
        self.target_layer = kwargs.get("target_layer", cfg.get("target_layer", target_layer))
        self.coefficient = kwargs.get("coefficient", cfg.get("coefficient", coefficient))
        self.max_norm = kwargs.get("max_norm", cfg.get("max_norm", max_norm))

        self.active_vectors: Dict[str, SteeringVector] = {}
        self.hooks: List[Any] = []
        self._hook_installed = False

    @property
    def name(self) -> str:
        return "steer"

    def _ensure_hook_installed(self):
        """Attaches dynamic activation injection hook to the target transformer block."""
        if self._hook_installed:
            return

        layer = self.model_wrapper.get_layer(self.target_layer)

        def steering_hook(module, inp, output):
            if not self.active_vectors:
                return output

            # Extract residual hidden tensor
            if isinstance(output, tuple):
                h = output[0]
                rest = output[1:]
                is_tuple = True
            else:
                h = output
                rest = None
                is_tuple = False

            # Add combined steering vectors to last token position
            total_delta = torch.zeros_like(h)
            for sv in self.active_vectors.values():
                v = sv.vector.to(h.device, dtype=h.dtype)
                total_delta[:, -1, :] += sv.coefficient * v

            modified_h = h + total_delta
            if is_tuple:
                return (modified_h,) + rest
            return modified_h

        hook = layer.register_forward_hook(steering_hook)
        self.hooks.append(hook)
        self._hook_installed = True

    def compute_steering_vector(
        self,
        prompt: str,
        target_new: str,
        ground_truth: Optional[str] = None,
    ) -> torch.Tensor:
        """Derives a contrastive steering vector from target assertion representations."""
        # Pair positive prompt with target, contrast with baseline
        pos_text = f"{prompt.strip()} {target_new.strip()}"
        neg_text = f"{prompt.strip()} {ground_truth.strip()}" if ground_truth else prompt.strip()

        pos_ids = self.model_wrapper.tokenize(pos_text)
        neg_ids = self.model_wrapper.tokenize(neg_text)

        with torch.no_grad():
            out_pos = self.model_wrapper.model(pos_ids, output_hidden_states=True)
            pos_states = out_pos[1] if isinstance(out_pos, tuple) and len(out_pos) > 1 else None

            out_neg = self.model_wrapper.model(neg_ids, output_hidden_states=True)
            neg_states = out_neg[1] if isinstance(out_neg, tuple) and len(out_neg) > 1 else None

            if pos_states is not None and neg_states is not None and self.target_layer < len(pos_states):
                h_pos = pos_states[self.target_layer][0, -1, :]
                h_neg = neg_states[self.target_layer][0, -1, :]
                diff = h_pos - h_neg
            else:
                # Fallback to output embeddings if hidden states not directly exposed
                hidden_dim = getattr(self.model_wrapper.model, "hidden_dim", 64)
                target_ids = self.model_wrapper.tokenizer.encode(target_new)
                t_id = target_ids[0] if target_ids else 0
                if hasattr(self.model_wrapper.model, "wte"):
                    diff = self.model_wrapper.model.wte.weight[t_id].clone()
                else:
                    diff = torch.randn(hidden_dim, device=self.model_wrapper.device)

        # Normalize and clamp to max_norm
        norm = torch.norm(diff, p=2) + 1e-8
        if norm > self.max_norm:
            diff = (diff / norm) * self.max_norm
            clamped = True
        else:
            clamped = False

        return diff

    def edit(self, request: EditRequest) -> EditResult:
        """Performs knowledge editing by injecting a steering vector."""
        start_time = time.time()
        self._ensure_hook_installed()

        # Measure baseline probabilities
        probs_pre = self.model_wrapper.predict_next_token_probs(request.prompt)
        p_pre_target = probs_pre.get(request.target_new, 0.0)

        # Derive steering vector
        vector = self.compute_steering_vector(
            prompt=request.prompt,
            target_new=request.target_new,
            ground_truth=request.ground_truth,
        )

        sv = SteeringVector(
            layer_idx=self.target_layer,
            vector=vector,
            coefficient=self.coefficient,
            norm_clamped=bool(torch.norm(vector, p=2) >= self.max_norm - 1e-4),
            target_token=request.target_new,
            prompt=request.prompt,
        )

        vector_key = request.request_id or f"{request.prompt}_{request.target_new}"
        self.active_vectors[vector_key] = sv

        # Measure post-intervention probabilities
        probs_post = self.model_wrapper.predict_next_token_probs(request.prompt)
        p_post_target = probs_post.get(request.target_new, 0.0)

        elapsed = time.time() - start_time
        success = p_post_target > p_pre_target or p_post_target >= 0.15

        metadata = {
            "target_layer": self.target_layer,
            "coefficient": self.coefficient,
            "max_norm": self.max_norm,
            "vector_norm": float(torch.norm(vector, p=2).item()),
            "active_vectors_count": len(self.active_vectors),
            "weight_modification": 0.0,  # Delta W = 0 guaranteed
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

    def remove_steering_vector(self, vector_key: str) -> bool:
        """Removes a specific steering vector."""
        if vector_key in self.active_vectors:
            del self.active_vectors[vector_key]
            return True
        return False

    def clear_all_steering(self):
        """Clears all active steering vectors."""
        self.active_vectors.clear()
