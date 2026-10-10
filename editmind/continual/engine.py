"""Continual lifelong knowledge editing engine and interference tracking."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import numpy as np
import time

from editmind.core.base_editor import BaseKnowledgeEditor
from editmind.core.types import EditRequest, EditResult
from editmind.evaluation.metrics import compute_efficacy_score


@dataclass
class ContinualEditTrajectory:
    """Stores full trajectory of sequential edits and retention matrix."""
    requests: List[EditRequest]
    results: List[EditResult]
    interference_matrix: np.ndarray  # Shape: [T, T]
    average_retention: float
    catastrophic_forgetting_rate: float
    wall_clock_time_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_edits": len(self.requests),
            "average_retention": round(self.average_retention, 4),
            "catastrophic_forgetting_rate": round(self.catastrophic_forgetting_rate, 4),
            "wall_clock_time_sec": round(self.wall_clock_time_sec, 2),
            "interference_matrix": self.interference_matrix.round(4).tolist(),
        }


class ContinualKnowledgeEditor:
    """Evaluates sequential knowledge editing and tracks mutual interference over time."""

    def __init__(self, editor: BaseKnowledgeEditor):
        self.editor = editor
        self.wrapper = editor.model_wrapper

    def run_sequential_stream(
        self,
        requests: List[EditRequest],
    ) -> ContinualEditTrajectory:
        """Applies edits sequentially and records the T x T interference matrix.
        
        M[i, j] records the efficacy of fact i after edit j has been applied.
        """
        start_time = time.time()
        T = len(requests)
        interference_matrix = np.zeros((T, T), dtype=np.float32)
        edit_results: List[EditResult] = []

        for step, req in enumerate(requests):
            # Apply edit step
            res = self.editor.edit(req)
            edit_results.append(res)

            # Probe all facts applied so far (from 0 to step)
            for prev_idx in range(step + 1):
                prev_req = requests[prev_idx]
                score = compute_efficacy_score(self.wrapper, prev_req)
                interference_matrix[prev_idx, step] = float(score)

        elapsed = time.time() - start_time

        # Calculate final retention (column T-1)
        final_retention = float(np.mean(interference_matrix[:, T - 1]))

        # Calculate catastrophic forgetting rate:
        # F_i = max_{t >= i} M[i, t] - M[i, T-1]
        forgetting_rates = []
        for i in range(T - 1):
            max_achieved = np.max(interference_matrix[i, i:])
            final_val = interference_matrix[i, T - 1]
            forgetting_rates.append(max(0.0, max_achieved - final_val))
        avg_forgetting = float(np.mean(forgetting_rates)) if forgetting_rates else 0.0

        return ContinualEditTrajectory(
            requests=requests,
            results=edit_results,
            interference_matrix=interference_matrix,
            average_retention=final_retention,
            catastrophic_forgetting_rate=avg_forgetting,
            wall_clock_time_sec=elapsed,
        )
