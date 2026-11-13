"""Plasticity-Stability Pareto Frontier Analyzer for Continual Knowledge Editing.

Analyzes the trade-off between model plasticity (ability to assimilate new knowledge)
and model stability (preservation of previously injected and baseline knowledge).
Computes Collapse Step (K_collapse), Area Under Retention Curve (AURC), Interference
Dispersion Index (IDI), and discovers the optimal Pareto operating boundary.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np

from editmind.continual.engine import ContinualKnowledgeEditor, ContinualEditTrajectory
from editmind.core.base_editor import BaseKnowledgeEditor
from editmind.core.types import EditRequest


@dataclass
class ParetoPoint:
    """Individual trade-off point at editing sequence step t."""
    step: int
    instantaneous_efficacy: float
    cumulative_retention: float
    plasticity_score: float
    stability_score: float
    harmonic_tradeoff: float
    weight_drift_norm: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "instantaneous_efficacy": round(self.instantaneous_efficacy, 4),
            "cumulative_retention": round(self.cumulative_retention, 4),
            "plasticity_score": round(self.plasticity_score, 4),
            "stability_score": round(self.stability_score, 4),
            "harmonic_tradeoff": round(self.harmonic_tradeoff, 4),
            "weight_drift_norm": round(self.weight_drift_norm, 4),
        }


@dataclass
class ParetoFrontierReport:
    """Comprehensive analysis report on plasticity-stability trade-off."""
    editor_name: str
    trajectory_length: int
    collapse_step: Optional[int]
    aurc: float  # Area Under Retention Curve (normalized to [0, 1])
    interference_dispersion_index: float  # Dispersion of degradation across memory keys
    optimal_operating_point: int  # Step index maximizing harmonic trade-off
    pareto_points: List[ParetoPoint] = field(default_factory=list)
    dominant_over: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "editor_name": self.editor_name,
            "trajectory_length": self.trajectory_length,
            "collapse_step": self.collapse_step,
            "aurc": round(self.aurc, 4),
            "interference_dispersion_index": round(self.interference_dispersion_index, 4),
            "optimal_operating_point": self.optimal_operating_point,
            "dominant_over": self.dominant_over,
            "pareto_points": [p.to_dict() for p in self.pareto_points],
        }


class ParetoPlasticityAuditor:
    """Audits lifelong continual editing trajectories for plasticity-stability boundaries."""

    def __init__(self, collapse_threshold: float = 0.5):
        self.collapse_threshold = collapse_threshold

    def audit_trajectory(
        self,
        trajectory: ContinualEditTrajectory,
        editor_name: str = "unknown",
    ) -> ParetoFrontierReport:
        """Audits a single ContinualEditTrajectory and computes Pareto metrics."""
        M = trajectory.interference_matrix
        T = M.shape[0]

        if T == 0:
            return ParetoFrontierReport(
                editor_name=editor_name,
                trajectory_length=0,
                collapse_step=None,
                aurc=0.0,
                interference_dispersion_index=0.0,
                optimal_operating_point=0,
                pareto_points=[],
            )

        pareto_points: List[ParetoPoint] = []
        retention_curve: List[float] = []
        collapse_step: Optional[int] = None

        for t in range(T):
            # Instantaneous efficacy: fact t probed immediately after edit t
            inst_eff = float(M[t, t])
            # Cumulative retention: average efficacy of all facts 0..t probed at step t
            retained_facts = M[: t + 1, t]
            cum_ret = float(np.mean(retained_facts))
            retention_curve.append(cum_ret)

            # Check collapse point
            if cum_ret < self.collapse_threshold and collapse_step is None:
                collapse_step = t + 1  # 1-indexed count

            # Harmonic trade-off F1 = 2 * P * S / (P + S + 1e-9)
            denom = inst_eff + cum_ret + 1e-9
            harmonic = (2.0 * inst_eff * cum_ret) / denom

            drift = 0.0
            if t < len(trajectory.results):
                drift = trajectory.results[t].delta_weight_norm or 0.0

            pareto_points.append(
                ParetoPoint(
                    step=t + 1,
                    instantaneous_efficacy=inst_eff,
                    cumulative_retention=cum_ret,
                    plasticity_score=inst_eff,
                    stability_score=cum_ret,
                    harmonic_tradeoff=harmonic,
                    weight_drift_norm=drift,
                )
            )

        # 1. Area Under Retention Curve (trapezoidal integration normalized by T - 1)
        if T == 1:
            aurc = retention_curve[0]
        else:
            # Normalized trapezoid rule for equidistant steps
            aurc = float((retention_curve[0] + retention_curve[-1] + 2.0 * sum(retention_curve[1:-1])) / (2.0 * (T - 1)))

        # 2. Interference Dispersion Index (IDI): variance of final column retention
        final_column = M[:, T - 1]
        idi = float(np.var(final_column))

        # 3. Optimal operating point (step with max harmonic tradeoff)
        optimal_step = max(pareto_points, key=lambda p: p.harmonic_tradeoff).step

        return ParetoFrontierReport(
            editor_name=editor_name,
            trajectory_length=T,
            collapse_step=collapse_step,
            aurc=aurc,
            interference_dispersion_index=idi,
            optimal_operating_point=optimal_step,
            pareto_points=pareto_points,
        )

    def compare_editors(
        self,
        editors: Dict[str, BaseKnowledgeEditor],
        requests: List[EditRequest],
    ) -> Dict[str, ParetoFrontierReport]:
        """Runs continual streams across multiple editors and calculates Pareto dominance."""
        reports: Dict[str, ParetoFrontierReport] = {}

        for name, editor in editors.items():
            continual_engine = ContinualKnowledgeEditor(editor)
            trajectory = continual_engine.run_sequential_stream(requests)
            reports[name] = self.audit_trajectory(trajectory, editor_name=name)

        # Compute pairwise Pareto dominance
        for name_a, report_a in reports.items():
            for name_b, report_b in reports.items():
                if name_a == name_b:
                    continue
                # A dominates B if A is strictly better in either AURC or optimal tradeoff
                # and no worse in either.
                a_beats_b_aurc = report_a.aurc > report_b.aurc
                a_beats_b_retention = (
                    report_a.pareto_points[-1].cumulative_retention
                    > report_b.pareto_points[-1].cumulative_retention
                )
                if (a_beats_b_aurc and report_a.aurc >= report_b.aurc - 1e-4) and (
                    a_beats_b_retention or a_beats_b_aurc
                ):
                    report_a.dominant_over.append(name_b)

        return reports
