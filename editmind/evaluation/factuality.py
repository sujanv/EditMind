"""Semantic Entropy and Post-Edit Factuality Stability Evaluator.

Evaluates hallucination susceptibility, semantic entropy across stochastic sampling,
and consistency of edited facts under diverse generation conditions.
Based on Semantic Entropy formulations (Farquhar et al., Nature 2024).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math
import torch
import torch.nn.functional as F

from editmind.core.types import EditRequest, KnowledgeTriple
from editmind.models.model_wrapper import BaseModelWrapper


@dataclass
class SemanticCluster:
    """A cluster of semantically equivalent model generations."""
    canonical_text: str
    frequency: int
    probability: float
    members: List[str] = field(default_factory=list)


@dataclass
class FactualityReport:
    """Report detailing post-edit factuality, semantic entropy, and hallucination stability."""
    prompt: str
    target_new: str
    num_samples: int
    temperature: float
    semantic_entropy: float
    is_confidently_stable: bool
    target_adherence_rate: float
    hallucination_risk_score: float
    clusters: List[Dict[str, Any]] = field(default_factory=list)
    sampled_generations: List[str] = field(default_factory=list)


class SemanticEntropyEvaluator:
    """Evaluates the stability and certainty of edited knowledge under stochastic sampling."""

    def __init__(self, entropy_threshold: float = 0.65, similarity_threshold: float = 0.80):
        self.entropy_threshold = entropy_threshold
        self.similarity_threshold = similarity_threshold

    def sample_generations(
        self,
        model: BaseModelWrapper,
        prompt: str,
        num_samples: int = 5,
        max_new_tokens: int = 12,
        temperature: float = 0.7,
    ) -> List[str]:
        """Samples multiple continuations using stochastic temperature sampling."""
        generations = []
        device = model.device
        tokenizer = model.tokenizer

        tokens = tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens], device=device)

        for _ in range(num_samples):
            curr_ids = input_ids.clone()
            sampled_tokens = []
            for _ in range(max_new_tokens):
                with torch.no_grad():
                    out = model.forward(curr_ids)
                    logits = out if torch.is_tensor(out) else out.logits
                    next_token_logits = logits[0, -1, :] / max(1e-4, temperature)
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    sampled_tokens.append(next_token.item())
                    curr_ids = torch.cat([curr_ids, next_token.unsqueeze(0)], dim=1)

            text = tokenizer.decode(sampled_tokens).strip()
            generations.append(text)

        return generations

    def cluster_generations(self, generations: List[str]) -> List[SemanticCluster]:
        """Groups generations into equivalence classes based on lexical and containment matching."""
        if not generations:
            return []

        clusters: List[SemanticCluster] = []
        for gen in generations:
            norm_gen = gen.lower().strip()
            matched = False
            for cluster in clusters:
                norm_canon = cluster.canonical_text.lower().strip()
                # Containment or exact match
                if norm_gen == norm_canon or norm_gen in norm_canon or norm_canon in norm_gen:
                    cluster.members.append(gen)
                    cluster.frequency += 1
                    matched = True
                    break
            if not matched:
                clusters.append(
                    SemanticCluster(
                        canonical_text=gen,
                        frequency=1,
                        probability=0.0,
                        members=[gen],
                    )
                )

        total = len(generations)
        for cluster in clusters:
            cluster.probability = cluster.frequency / total

        return clusters

    def compute_semantic_entropy(self, clusters: List[SemanticCluster]) -> float:
        """Computes Shannon semantic entropy over equivalence classes: - sum(p * log(p))."""
        entropy = 0.0
        for cluster in clusters:
            p = cluster.probability
            if p > 0:
                entropy -= p * math.log(p)
        return round(float(entropy), 4)

    def evaluate_edit_stability(
        self,
        model: BaseModelWrapper,
        prompt: str,
        target_new: str,
        num_samples: int = 5,
        temperature: float = 0.7,
    ) -> FactualityReport:
        """Runs stochastic sampling and assesses factuality stability for an edited fact."""
        samples = self.sample_generations(
            model=model,
            prompt=prompt,
            num_samples=num_samples,
            temperature=temperature,
        )

        clusters = self.cluster_generations(samples)
        entropy = self.compute_semantic_entropy(clusters)

        # Count how many samples contain the new target fact
        target_lower = target_new.lower().strip()
        adherence_count = sum(1 for s in samples if target_lower in s.lower())
        adherence_rate = adherence_count / max(1, len(samples))

        # Hallucination risk is high when entropy is high or adherence to the edited target is low
        hallucination_risk = round(max(0.0, min(1.0, (1.0 - adherence_rate) * 0.6 + (entropy / 2.0) * 0.4)), 3)
        stable = (entropy <= self.entropy_threshold) and (adherence_rate >= 0.5)

        cluster_dicts = [
            {
                "canonical": c.canonical_text,
                "frequency": c.frequency,
                "probability": round(c.probability, 3),
                "members": c.members,
            }
            for c in clusters
        ]

        return FactualityReport(
            prompt=prompt,
            target_new=target_new,
            num_samples=num_samples,
            temperature=temperature,
            semantic_entropy=entropy,
            is_confidently_stable=stable,
            target_adherence_rate=round(adherence_rate, 3),
            hallucination_risk_score=hallucination_risk,
            clusters=cluster_dicts,
            sampled_generations=samples,
        )
