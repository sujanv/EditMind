"""Counterfactual Generalization Stress-Tester and Robustness Auditor.

Audits whether an applied knowledge edit generalizes across diverse syntactic
rephrasings, distractor prefixes, and adversarial framing, or whether the model
merely memorized the exact surface token sequence of the training prompt.
Computes the Semantic Generalization Robustness Index (SGRI) and Prompt Sensitivity Variance (PSV).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np

from editmind.models.model_wrapper import UnifiedModelWrapper
from editmind.core.types import EditRequest


@dataclass
class PerturbationProbe:
    """Individual perturbed prompt evaluation result."""
    prompt: str
    category: str  # "rephrase", "distractor", "inversion", "conversational"
    target_prob: float
    old_prob: float
    passed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "category": self.category,
            "target_prob": round(float(self.target_prob), 4),
            "old_prob": round(float(self.old_prob), 4),
            "passed": self.passed,
        }


@dataclass
class GeneralizationAuditReport:
    """Comprehensive evaluation report for prompt generalization robustness."""
    base_prompt: str
    target_new: str
    ground_truth: Optional[str]
    base_target_prob: float
    total_perturbations: int
    passed_perturbations: int
    generalization_pass_rate: float
    semantic_generalization_robustness_index: float  # SGRI
    prompt_sensitivity_variance: float              # PSV
    category_breakdown: Dict[str, Dict[str, float]]
    probes: List[PerturbationProbe]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_prompt": self.base_prompt,
            "target_new": self.target_new,
            "ground_truth": self.ground_truth,
            "base_target_prob": round(float(self.base_target_prob), 4),
            "total_perturbations": self.total_perturbations,
            "passed_perturbations": self.passed_perturbations,
            "generalization_pass_rate": round(float(self.generalization_pass_rate), 4),
            "semantic_generalization_robustness_index": round(
                float(self.semantic_generalization_robustness_index), 4
            ),
            "prompt_sensitivity_variance": round(float(self.prompt_sensitivity_variance), 6),
            "category_breakdown": self.category_breakdown,
            "probes": [p.to_dict() for p in self.probes],
        }


class GeneralizationAuditor:
    """Stress-tests edited models against syntactic and contextual perturbations."""

    def __init__(self, model_wrapper: UnifiedModelWrapper):
        self.wrapper = model_wrapper

    def generate_perturbations(
        self,
        base_prompt: str,
        subject: Optional[str] = None,
    ) -> List[Tuple[str, str]]:
        """Generates category-labeled prompt perturbations."""
        subj = subject or base_prompt.split()[-1]
        cleaned_prompt = base_prompt.strip().rstrip(".:;,?")

        perturbations: List[Tuple[str, str]] = []

        # 1. Distractor prefixes
        distractor_prefixes = [
            "According to official verified records,",
            "In recent confirmed documentation,",
            "It is universally understood that",
            "Scholars have noted that",
        ]
        for prefix in distractor_prefixes:
            perturbations.append((f"{prefix} {cleaned_prompt} is", "distractor"))

        # 2. Syntactic rephrasing
        perturbations.append((f"Which location serves as the destination for {subj}?", "rephrase"))
        perturbations.append((f"The exact entity associated with {subj} is", "rephrase"))
        perturbations.append((f"Tell me about {subj}, specifically", "rephrase"))

        # 3. Conversational / user framing
        perturbations.append((f"Could you please confirm if {cleaned_prompt} is", "conversational"))
        perturbations.append((f"User query: {cleaned_prompt} ->", "conversational"))

        # 4. Grammatical inversion
        perturbations.append((f"Regarding {subj}, the primary value is", "inversion"))

        return perturbations

    def audit_generalization(
        self,
        base_prompt: str,
        target_new: str,
        ground_truth: Optional[str] = None,
        subject: Optional[str] = None,
        custom_perturbations: Optional[List[Tuple[str, str]]] = None,
    ) -> GeneralizationAuditReport:
        """Evaluates model generalization across all perturbed formulations."""
        base_probs = self.wrapper.predict_next_token_probs(base_prompt)
        base_target_prob = float(base_probs.get(target_new, 0.0))

        candidates = custom_perturbations or self.generate_perturbations(base_prompt, subject=subject)

        probes: List[PerturbationProbe] = []
        target_probs: List[float] = []
        category_stats: Dict[str, Dict[str, List[float]]] = {}

        for p_text, cat in candidates:
            probs = self.wrapper.predict_next_token_probs(p_text)
            p_tgt = float(probs.get(target_new, 0.0))
            p_old = float(probs.get(ground_truth, 0.0)) if ground_truth else 0.0

            passed = p_tgt > p_old
            probe = PerturbationProbe(
                prompt=p_text,
                category=cat,
                target_prob=p_tgt,
                old_prob=p_old,
                passed=passed,
            )
            probes.append(probe)
            target_probs.append(p_tgt)

            if cat not in category_stats:
                category_stats[cat] = {"target_probs": [], "passed": []}
            category_stats[cat]["target_probs"].append(p_tgt)
            category_stats[cat]["passed"].append(1.0 if passed else 0.0)

        total = len(probes)
        passed_count = sum(1 for p in probes if p.passed)
        pass_rate = (passed_count / max(1, total))

        # Semantic Generalization Robustness Index (SGRI)
        # Ratio of average perturbed target prob to base target prob
        avg_perturbed = float(np.mean(target_probs)) if target_probs else 0.0
        denom = base_target_prob if base_target_prob > 1e-5 else 1.0
        sgri = min(2.0, avg_perturbed / denom)

        # Prompt Sensitivity Variance (PSV)
        psv = float(np.var(target_probs)) if target_probs else 0.0

        # Category breakdown
        breakdown: Dict[str, Dict[str, float]] = {}
        for cat, data in category_stats.items():
            breakdown[cat] = {
                "mean_target_prob": round(float(np.mean(data["target_probs"])), 4),
                "pass_rate": round(float(np.mean(data["passed"])), 4),
            }

        return GeneralizationAuditReport(
            base_prompt=base_prompt,
            target_new=target_new,
            ground_truth=ground_truth,
            base_target_prob=base_target_prob,
            total_perturbations=total,
            passed_perturbations=passed_count,
            generalization_pass_rate=pass_rate,
            semantic_generalization_robustness_index=sgri,
            prompt_sensitivity_variance=psv,
            category_breakdown=breakdown,
            probes=probes,
        )
