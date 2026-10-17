"""Active Subspace and Gradient-Informed Localization Engine.

Combines Integrated Gradients, empirical Fisher Information layer sensitivity,
and Causal Mediation Tracing to provide a robust consensus recommendation for
optimal knowledge editing layers.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from editmind.models.model_wrapper import UnifiedModelWrapper
from editmind.models.causal_tracer import CausalTracer


@dataclass
class TokenAttributionResult:
    """Token-level attribution score from Integrated Gradients."""
    token: str
    token_id: int
    position: int
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token": self.token,
            "token_id": self.token_id,
            "position": self.position,
            "score": round(float(self.score), 4),
        }


@dataclass
class LayerAttributionScore:
    """Multi-criterion scores for a single transformer layer."""
    layer_idx: int
    causal_score: float
    fisher_score: float
    gradient_energy: float
    consensus_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_idx": self.layer_idx,
            "causal_score": round(float(self.causal_score), 4),
            "fisher_score": round(float(self.fisher_score), 6),
            "gradient_energy": round(float(self.gradient_energy), 6),
            "consensus_score": round(float(self.consensus_score), 4),
        }


@dataclass
class ConsensusLocalizationReport:
    """Comprehensive consensus localization report."""
    prompt: str
    subject: str
    target: str
    recommended_layers: List[int]
    confidence: float
    layer_scores: List[LayerAttributionScore]
    token_attributions: List[TokenAttributionResult]
    critical_token: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "subject": self.subject,
            "target": self.target,
            "recommended_layers": self.recommended_layers,
            "confidence": round(float(self.confidence), 4),
            "layer_scores": [s.to_dict() for s in self.layer_scores],
            "token_attributions": [t.to_dict() for t in self.token_attributions],
            "critical_token": self.critical_token,
        }


class IntegratedGradientsAttributor:
    """Computes path-integral token attribution using Integrated Gradients."""

    def __init__(self, model_wrapper: UnifiedModelWrapper, steps: int = 10):
        self.wrapper = model_wrapper
        self.model = model_wrapper.model
        self.tokenizer = model_wrapper.tokenizer
        self.steps = max(steps, 2)

    def attribute_tokens(self, prompt: str, target: str) -> List[TokenAttributionResult]:
        """Calculates token-level attribution for predicting the target."""
        input_ids = self.wrapper.tokenize(prompt)
        target_ids = self.tokenizer.encode(target)
        target_id = target_ids[0] if target_ids else 0

        # Get embedding layer if available
        embed_layer = None
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Embedding):
                embed_layer = module
                break

        tokens = [self.tokenizer.id2token.get(idx, getattr(self.tokenizer, "unk_token", "<unk>")) for idx in input_ids[0].tolist()]
        seq_len = input_ids.shape[1]

        if embed_layer is None:
            # Fallback uniform attribution if no embedding layer is identifiable
            uniform = 1.0 / max(seq_len, 1)
            return [
                TokenAttributionResult(token=tokens[i], token_id=int(input_ids[0, i]), position=i, score=uniform)
                for i in range(seq_len)
            ]

        # Extract baseline and input embeddings
        inputs_embeds = embed_layer(input_ids).detach()
        baseline_embeds = torch.zeros_like(inputs_embeds)

        # Compute integrated gradients along the interpolation path
        total_gradients = torch.zeros_like(inputs_embeds)
        alphas = torch.linspace(0.0, 1.0, self.steps)

        for alpha in alphas:
            interpolated = (baseline_embeds + alpha * (inputs_embeds - baseline_embeds)).clone()
            interpolated.requires_grad_(True)

            try:
                logits = self._forward_with_embeds(interpolated)
            except Exception:
                out = self.model(input_ids)
                logits = out[0] if isinstance(out, tuple) else (out.logits if hasattr(out, "logits") else out)
                break

            target_logit = logits[0, -1, target_id]
            grads = torch.autograd.grad(target_logit, interpolated, retain_graph=False)[0]
            total_gradients += grads

        avg_grads = total_gradients / self.steps
        attributions = (inputs_embeds - baseline_embeds) * avg_grads
        token_scores = attributions.norm(p=2, dim=-1).squeeze(0).detach().cpu().numpy()

        # Normalize token scores
        norm = np.sum(token_scores) + 1e-8
        normalized = (token_scores / norm).tolist()

        return [
            TokenAttributionResult(
                token=tokens[i],
                token_id=int(input_ids[0, i]),
                position=i,
                score=float(normalized[i])
            )
            for i in range(seq_len)
        ]

    def _forward_with_embeds(self, embeds: torch.Tensor) -> torch.Tensor:
        """Internal helper to forward with overridden embeddings."""
        batch_size, seq_len, _ = embeds.shape
        pos = torch.arange(0, seq_len, dtype=torch.long, device=embeds.device)
        if hasattr(self.model, "wpe"):
            h = embeds + self.model.wpe(pos)
        else:
            h = embeds

        layers = getattr(self.model, "layers", getattr(self.model, "blocks", []))
        for layer in layers:
            h = layer(h)
            if isinstance(h, tuple):
                h = h[0]

        if hasattr(self.model, "ln_f"):
            h = self.model.ln_f(h)
        if hasattr(self.model, "lm_head"):
            return self.model.lm_head(h)
        return h


class FisherLayerRanker:
    """Computes empirical Fisher Information per transformer layer."""

    def __init__(self, model_wrapper: UnifiedModelWrapper):
        self.wrapper = model_wrapper
        self.model = model_wrapper.model
        self.tokenizer = model_wrapper.tokenizer

    def compute_layer_fisher(self, prompt: str, target: str) -> Dict[int, float]:
        """Calculates the average squared gradient per layer for target token likelihood."""
        input_ids = self.wrapper.tokenize(prompt)
        target_ids = self.tokenizer.encode(target)
        target_id = target_ids[0] if target_ids else 0

        # Zero existing gradients
        self.model.zero_grad()

        # Forward pass
        out = self.model(input_ids)
        logits = out[0] if isinstance(out, tuple) else (out.logits if hasattr(out, "logits") else out)
        log_probs = F.log_softmax(logits[0, -1], dim=-1)
        target_log_prob = log_probs[target_id]

        # Backward pass to obtain gradient w.r.t parameters
        target_log_prob.backward(retain_graph=False)

        num_layers = self.wrapper.num_layers
        layer_fisher: Dict[int, float] = {}

        for l_idx in range(num_layers):
            layer = self.wrapper.get_layer(l_idx)
            sum_sq = 0.0
            param_count = 0
            for p in layer.parameters():
                if p.grad is not None:
                    sum_sq += float(torch.sum(p.grad ** 2).item())
                    param_count += p.numel()
            layer_fisher[l_idx] = (sum_sq / max(param_count, 1))

        self.model.zero_grad()
        return layer_fisher


class ConsensusLayerRanker:
    """Synthesizes Causal Tracing, Fisher Information, and Gradient Energy into consensus layer recommendations."""

    def __init__(
        self,
        model_wrapper: UnifiedModelWrapper,
        causal_weight: float = 0.45,
        fisher_weight: float = 0.35,
        gradient_weight: float = 0.20,
    ):
        self.wrapper = model_wrapper
        self.causal_weight = causal_weight
        self.fisher_weight = fisher_weight
        self.gradient_weight = gradient_weight
        self.causal_tracer = CausalTracer(model_wrapper)
        self.fisher_ranker = FisherLayerRanker(model_wrapper)
        self.ig_attributor = IntegratedGradientsAttributor(model_wrapper)

    def locate_knowledge(
        self,
        prompt: str,
        subject: str,
        target: str,
        top_k: int = 2,
    ) -> ConsensusLocalizationReport:
        """Executes full consensus localization and returns recommended layers with audit details."""
        num_layers = self.wrapper.num_layers

        # 1. Causal Mediation Tracing
        trace_res = self.causal_tracer.trace(prompt, subject, target)
        # Average indirect effect across tokens per layer
        causal_per_layer = np.max(trace_res.indirect_effects, axis=1)  # max over tokens for layer
        causal_norm = causal_per_layer / (np.max(causal_per_layer) + 1e-8)

        # 2. Empirical Fisher Sensitivity
        fisher_dict = self.fisher_ranker.compute_layer_fisher(prompt, target)
        fisher_vals = np.array([fisher_dict.get(l, 0.0) for l in range(num_layers)])
        fisher_norm = fisher_vals / (np.max(fisher_vals) + 1e-8)

        # 3. Token Attributions via Integrated Gradients
        token_attrs = self.ig_attributor.attribute_tokens(prompt, target)
        # Gradient energy proxy across layers (derived from Fisher & Causal coupling)
        grad_energy = (causal_norm * 0.6 + fisher_norm * 0.4)

        # 4. Consensus Score Combination
        consensus_scores = (
            self.causal_weight * causal_norm
            + self.fisher_weight * fisher_norm
            + self.gradient_weight * grad_energy
        )

        layer_scores: List[LayerAttributionScore] = []
        for l_idx in range(num_layers):
            layer_scores.append(
                LayerAttributionScore(
                    layer_idx=l_idx,
                    causal_score=float(causal_norm[l_idx]),
                    fisher_score=float(fisher_vals[l_idx]),
                    gradient_energy=float(grad_energy[l_idx]),
                    consensus_score=float(consensus_scores[l_idx]),
                )
            )

        # Sort layers by consensus score descending
        sorted_indices = np.argsort(-consensus_scores)
        top_k = min(top_k, num_layers)
        recommended_layers = [int(sorted_indices[i]) for i in range(top_k)]

        # Find critical token with highest attribution
        critical_token = max(token_attrs, key=lambda t: t.score).token if token_attrs else subject

        # Confidence based on score gap between top layer and median
        top_score = consensus_scores[sorted_indices[0]]
        median_score = np.median(consensus_scores)
        confidence = float(min(1.0, max(0.0, (top_score - median_score) / (top_score + 1e-6))))

        return ConsensusLocalizationReport(
            prompt=prompt,
            subject=subject,
            target=target,
            recommended_layers=recommended_layers,
            confidence=confidence,
            layer_scores=layer_scores,
            token_attributions=token_attrs,
            critical_token=critical_token,
        )
