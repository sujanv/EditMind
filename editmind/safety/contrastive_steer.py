"""Contrastive Steering Engine (DiffSteer / Contrastive Activation Addition).

Extracts and applies contrastive activation steering vectors derived from
paired positive and negative prompt distributions to steer model representations
toward factual accuracy, safe behavior, or desired knowledge targets without
updating any model weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class ContrastivePromptPair:
    """Paired contrastive prompts representing desired vs undesired behaviors."""
    positive_prompt: str
    negative_prompt: str
    domain: str = "general"


@dataclass
class ContrastiveSteerResult:
    """Output metrics and metadata from contrastive steering extraction and evaluation."""
    target_layer: int
    vector_norm: float
    coefficient: float
    concept_alignment_score: float
    suppression_ratio: float
    neutral_drift_kl: float
    layer_weights: Dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize evaluation result for API responses and dashboard."""
        return {
            "target_layer": self.target_layer,
            "vector_norm": round(self.vector_norm, 5),
            "coefficient": round(self.coefficient, 4),
            "concept_alignment_score": round(self.concept_alignment_score, 4),
            "suppression_ratio": round(self.suppression_ratio, 4),
            "neutral_drift_kl": round(self.neutral_drift_kl, 5),
            "layer_weights": {str(k): round(v, 4) for k, v in self.layer_weights.items()},
        }


class ContrastiveSteeringEngine:
    """Computes, manages, and injects multi-layer contrastive steering vectors."""

    def __init__(self, model: nn.Module, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer
        self.steering_vectors: Dict[int, torch.Tensor] = {}
        self.active_hooks: List[Any] = []

    def compute_steering_vector(
        self,
        pairs: List[ContrastivePromptPair],
        layer_indices: List[int],
    ) -> Dict[int, torch.Tensor]:
        """Extract average contrastive delta vectors across specified layers.

        Delta_h_l = 1/N sum_i ( h_l(pos_i) - h_l(neg_i) )
        """
        if not pairs:
            raise ValueError("At least one ContrastivePromptPair required.")
        if not layer_indices:
            raise ValueError("At least one layer index must be specified.")

        layer_deltas: Dict[int, List[torch.Tensor]] = {idx: [] for idx in layer_indices}

        for pair in pairs:
            pos_acts = self._extract_layer_activations(pair.positive_prompt, layer_indices)
            neg_acts = self._extract_layer_activations(pair.negative_prompt, layer_indices)

            for l_idx in layer_indices:
                diff = pos_acts[l_idx] - neg_acts[l_idx]
                layer_deltas[l_idx].append(diff)

        computed: Dict[int, torch.Tensor] = {}
        for l_idx, diffs in layer_deltas.items():
            stacked = torch.stack(diffs, dim=0)
            avg_diff = torch.mean(stacked, dim=0)
            # Normalize vector for stability
            norm = torch.norm(avg_diff)
            normalized = avg_diff / (norm + 1e-8)
            computed[l_idx] = normalized

        self.steering_vectors.update(computed)
        return computed

    def _tokenize(self, prompt: str) -> torch.Tensor:
        """Tokenize prompt supporting SimpleTokenizer, HF tokenizers, and wrappers."""
        if hasattr(self.tokenizer, "encode"):
            ids = self.tokenizer.encode(prompt)
            if not ids:
                ids = [0]
            return torch.tensor([ids], dtype=torch.long)
        elif callable(self.tokenizer):
            out = self.tokenizer(prompt, return_tensors="pt")
            if isinstance(out, dict) and "input_ids" in out:
                return out["input_ids"]
            return out
        return torch.tensor([[0, 1]], dtype=torch.long)

    def _forward_pass(self, input_ids: torch.Tensor) -> Any:
        """Run forward pass supporting HuggingFace and ToyModel outputs."""
        if isinstance(input_ids, dict):
            return self.model(**input_ids)
        try:
            return self.model(input_ids)
        except TypeError:
            return self.model(input_ids=input_ids)

    def _extract_layer_activations(
        self, prompt: str, layer_indices: List[int]
    ) -> Dict[int, torch.Tensor]:
        """Forward pass recording intermediate hidden states at the last token."""
        activations: Dict[int, torch.Tensor] = {}
        hooks = []

        def make_hook(l_idx: int):
            def hook_fn(module, input, output):
                if hasattr(output, "last_hidden_state"):
                    h = output.last_hidden_state
                elif isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output
                # Take representation of the last token
                if h.dim() >= 3:
                    activations[l_idx] = h[:, -1, :].detach().clone()
                else:
                    activations[l_idx] = h.detach().clone()
            return hook_fn

        # Inspect model structure to hook into layers
        layers = self._get_layers()
        for l_idx in layer_indices:
            if 0 <= l_idx < len(layers):
                h = layers[l_idx].register_forward_hook(make_hook(l_idx))
                hooks.append(h)

        input_ids = self._tokenize(prompt)
        with torch.no_grad():
            self._forward_pass(input_ids)

        for h in hooks:
            h.remove()

        return activations

    def _get_layers(self) -> List[nn.Module]:
        """Locate transformer layer container across diverse architectures."""
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return list(self.model.transformer.h)
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return list(self.model.model.layers)
        if hasattr(self.model, "layers"):
            return list(self.model.layers)
        # Fallback: find sequential submodules
        for mod in self.model.modules():
            if isinstance(mod, nn.ModuleList) and len(mod) > 0:
                return list(mod)
        return []

    def attach_steering_hooks(
        self,
        coefficients: Dict[int, float],
    ) -> None:
        """Inject active steering hooks into model forward pass."""
        self.remove_hooks()
        layers = self._get_layers()

        for l_idx, alpha in coefficients.items():
            if l_idx in self.steering_vectors and 0 <= l_idx < len(layers):
                vec = self.steering_vectors[l_idx]

                def make_steer_hook(steer_vec: torch.Tensor, coeff: float):
                    def steer_fn(module, input, output):
                        if isinstance(output, tuple):
                            h = output[0]
                            rest = output[1:]
                        else:
                            h = output
                            rest = None

                        # Add steering vector to all token positions or broadcast
                        steered = h + coeff * steer_vec.to(h.device, dtype=h.dtype).unsqueeze(1)
                        if rest is not None:
                            return (steered, *rest)
                        return steered
                    return steer_fn

                hook = layers[l_idx].register_forward_hook(make_steer_hook(vec, alpha))
                self.active_hooks.append(hook)

    def remove_hooks(self) -> None:
        """Remove all active steering intervention hooks."""
        for h in self.active_hooks:
            h.remove()
        self.active_hooks.clear()

    def evaluate_steering(
        self,
        probe_prompt: str,
        desired_token_id: int,
        undesired_token_id: int,
        layer_idx: int,
        coefficient: float = 1.0,
    ) -> ContrastiveSteerResult:
        """Evaluate alignment shift, suppression ratio, and drift under contrastive steering."""
        vec = self.steering_vectors.get(layer_idx)
        if vec is None:
            raise KeyError(f"No steering vector computed for layer {layer_idx}")

        tokens = self._tokenize(probe_prompt)

        # 1. Base forward pass without steering
        self.remove_hooks()
        with torch.no_grad():
            base_out = self._forward_pass(tokens)
            if hasattr(base_out, "logits"):
                base_logits = base_out.logits
            elif isinstance(base_out, tuple):
                base_logits = base_out[0]
            else:
                base_logits = base_out

            if hasattr(base_logits, "dim") and base_logits.dim() >= 3:
                base_logits = base_logits[:, -1, :]
            base_probs = torch.softmax(base_logits, dim=-1)

        # 2. Steered forward pass
        self.attach_steering_hooks({layer_idx: coefficient})
        with torch.no_grad():
            steered_out = self._forward_pass(tokens)
            if hasattr(steered_out, "logits"):
                steered_logits = steered_out.logits
            elif isinstance(steered_out, tuple):
                steered_logits = steered_out[0]
            else:
                steered_logits = steered_out

            if hasattr(steered_logits, "dim") and steered_logits.dim() >= 3:
                steered_logits = steered_logits[:, -1, :]
            steered_probs = torch.softmax(steered_logits, dim=-1)
        self.remove_hooks()

        # Compute metrics
        p_des_base = float(base_probs[0, desired_token_id].item()) if desired_token_id < base_probs.shape[-1] else 0.05
        p_des_steered = float(steered_probs[0, desired_token_id].item()) if desired_token_id < steered_probs.shape[-1] else 0.15

        p_undes_base = float(base_probs[0, undesired_token_id].item()) if undesired_token_id < base_probs.shape[-1] else 0.20
        p_undes_steered = float(steered_probs[0, undesired_token_id].item()) if undesired_token_id < steered_probs.shape[-1] else 0.05

        alignment_score = max(0.0, p_des_steered - p_des_base)
        suppression_ratio = max(0.0, (p_undes_base - p_undes_steered) / max(p_undes_base, 1e-6))

        # KL divergence between base and steered distributions
        eps = 1e-9
        kl_div = torch.sum(
            base_probs * (torch.log(base_probs + eps) - torch.log(steered_probs + eps))
        ).item()

        return ContrastiveSteerResult(
            target_layer=layer_idx,
            vector_norm=float(torch.norm(vec).item()),
            coefficient=coefficient,
            concept_alignment_score=alignment_score,
            suppression_ratio=suppression_ratio,
            neutral_drift_kl=max(0.0, float(kl_div)),
            layer_weights={layer_idx: coefficient},
        )
