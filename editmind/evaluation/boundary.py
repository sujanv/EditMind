"""Knowledge Boundary & Out-of-Distribution (OOD) Neighborhood Calibrator.

Evaluates whether knowledge edits remain sharply localized to the intended target
entity or cause diffuse "semantic leakage" into adjacent neighboring entities.
Computes the Boundary Sharpness Index (BSI) and Neighborhood Leakage Ratio (NLR).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np

from editmind.models.model_wrapper import UnifiedModelWrapper


# Default neighbor entities for prominent entity categories
DEFAULT_NEIGHBORS: Dict[str, List[Dict[str, str]]] = {
    "eiffel tower": [
        {"entity": "Louvre Museum", "prompt": "The Louvre Museum is located in", "expected": "Paris"},
        {"entity": "Notre-Dame", "prompt": "Notre-Dame Cathedral is located in", "expected": "Paris"},
        {"entity": "Arc de Triomphe", "prompt": "The Arc de Triomphe is located in", "expected": "Paris"},
    ],
    "france": [
        {"entity": "Spain", "prompt": "The capital of Spain is", "expected": "Madrid"},
        {"entity": "Germany", "prompt": "The capital of Germany is", "expected": "Berlin"},
        {"entity": "Italy", "prompt": "The capital of Italy is", "expected": "Rome"},
    ],
    "messi": [
        {"entity": "Cristiano Ronaldo", "prompt": "Cristiano Ronaldo plays for", "expected": "Al-Nassr"},
        {"entity": "Neymar", "prompt": "Neymar plays for", "expected": "Al-Hilal"},
        {"entity": "Mbappe", "prompt": "Kylian Mbappe plays for", "expected": "Real Madrid"},
    ],
}


@dataclass
class NeighborProbe:
    """Individual neighboring entity probe result."""
    neighbor_entity: str
    prompt: str
    expected_target: str
    target_new_prob: float      # Unintended bleed probability of edit's target
    expected_target_prob: float  # Intended probability of neighbor's true target
    leakage_detected: bool       # True if target_new_prob > expected_target_prob

    def to_dict(self) -> Dict[str, Any]:
        return {
            "neighbor_entity": self.neighbor_entity,
            "prompt": self.prompt,
            "expected_target": self.expected_target,
            "target_new_prob": round(float(self.target_new_prob), 4),
            "expected_target_prob": round(float(self.expected_target_prob), 4),
            "leakage_detected": self.leakage_detected,
        }


@dataclass
class BoundaryCalibrationReport:
    """Diagnostic report evaluating neighborhood containment and boundary sharpness."""
    subject: str
    target_new: str
    target_efficacy: float
    boundary_sharpness_index: float      # BSI in [0, 1]
    neighborhood_leakage_ratio: float    # NLR in [0, 1]
    neighbor_preservation_score: float   # Mean retention of neighbor true facts
    neighbor_leakage_score: float        # Mean bleed of new target into neighbors
    is_well_calibrated: bool
    probes: List[NeighborProbe]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "target_new": self.target_new,
            "target_efficacy": round(float(self.target_efficacy), 4),
            "boundary_sharpness_index": round(float(self.boundary_sharpness_index), 4),
            "neighborhood_leakage_ratio": round(float(self.neighborhood_leakage_ratio), 4),
            "neighbor_preservation_score": round(float(self.neighbor_preservation_score), 4),
            "neighbor_leakage_score": round(float(self.neighbor_leakage_score), 4),
            "is_well_calibrated": self.is_well_calibrated,
            "probes": [p.to_dict() for p in self.probes],
        }


class KnowledgeBoundaryCalibrator:
    """Audits spatial containment and neighborhood isolation of knowledge modifications."""

    def __init__(self, model_wrapper: UnifiedModelWrapper):
        self.wrapper = model_wrapper

    def calibrate_boundary(
        self,
        prompt: str,
        subject: str,
        target_new: str,
        neighbor_entities: Optional[List[Dict[str, str]]] = None,
    ) -> BoundaryCalibrationReport:
        """Audits target efficacy and probes semantic neighborhood for unintended bleed."""
        # 1. Target efficacy on edited prompt
        target_probs = self.wrapper.predict_next_token_probs(prompt)
        target_eff = float(target_probs.get(target_new, 0.0))

        # 2. Select neighborhood probes
        subj_lower = subject.lower().strip()
        neighbors = neighbor_entities
        if not neighbors:
            neighbors = DEFAULT_NEIGHBORS.get(subj_lower, [
                {"entity": f"{subject} neighbor A", "prompt": f"Related fact about {subject} neighbor A is", "expected": "true_value"},
                {"entity": f"{subject} neighbor B", "prompt": f"Related fact about {subject} neighbor B is", "expected": "true_value"},
            ])

        probes: List[NeighborProbe] = []
        bleed_probs: List[float] = []
        retention_probs: List[float] = []

        for n in neighbors:
            p_text = n["prompt"]
            exp = n["expected"]
            n_probs = self.wrapper.predict_next_token_probs(p_text)

            p_bleed = float(n_probs.get(target_new, 0.0))
            p_exp = float(n_probs.get(exp, 0.0))

            leakage = p_bleed > p_exp and p_bleed > 0.15
            probe = NeighborProbe(
                neighbor_entity=n["entity"],
                prompt=p_text,
                expected_target=exp,
                target_new_prob=p_bleed,
                expected_target_prob=p_exp,
                leakage_detected=leakage,
            )
            probes.append(probe)
            bleed_probs.append(p_bleed)
            retention_probs.append(p_exp)

        avg_bleed = float(np.mean(bleed_probs)) if bleed_probs else 0.0
        avg_retention = float(np.mean(retention_probs)) if retention_probs else 0.5

        # 3. Calculate Boundary Sharpness Index (BSI)
        # BSI = target_efficacy * (1 - avg_bleed) * max(0.1, avg_retention)
        bsi = float(target_eff * (1.0 - min(1.0, avg_bleed)) * min(1.0, max(0.1, avg_retention)))

        # 4. Neighborhood Leakage Ratio (NLR)
        denom = target_eff if target_eff > 1e-4 else 1.0
        nlr = float(min(1.0, avg_bleed / denom))

        is_calibrated = bsi >= 0.2 and nlr <= 0.35 and not any(p.leakage_detected for p in probes)

        return BoundaryCalibrationReport(
            subject=subject,
            target_new=target_new,
            target_efficacy=target_eff,
            boundary_sharpness_index=bsi,
            neighborhood_leakage_ratio=nlr,
            neighbor_preservation_score=avg_retention,
            neighbor_leakage_score=avg_bleed,
            is_well_calibrated=is_calibrated,
            probes=probes,
        )
