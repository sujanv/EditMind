"""Knowledge Editing Evaluator for single and batch editing tasks."""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import time
import numpy as np

from editmind.core.base_editor import BaseKnowledgeEditor
from editmind.core.types import EditRequest, EditResult, EvaluationMetrics
from editmind.data.dataset import KnowledgeDataset
from editmind.evaluation.metrics import (
    compute_efficacy_score,
    compute_generality_score,
    compute_locality_score,
    compute_portability_score,
)


class KnowledgeEditorEvaluator:
    """Evaluates an editor across benchmark datasets with comprehensive metrics."""

    def __init__(self, editor: BaseKnowledgeEditor):
        self.editor = editor
        self.wrapper = editor.model_wrapper

    def evaluate_request(self, req: EditRequest) -> EvaluationMetrics:
        """Applies an edit and computes comprehensive metrics."""
        start_time = time.time()
        res = self.editor.edit(req)
        edit_latency = (time.time() - start_time) * 1000.0

        efficacy = compute_efficacy_score(self.wrapper, req)
        generality = compute_generality_score(self.wrapper, req)
        locality = compute_locality_score(self.wrapper, req)
        portability = compute_portability_score(self.wrapper, req)

        return EvaluationMetrics(
            efficacy=float(efficacy),
            generality=float(generality),
            locality=float(locality),
            portability=float(portability),
            ripple_retention=1.0,
            edit_latency_ms=edit_latency,
            delta_norm=res.delta_weight_norm,
        )

    def evaluate_dataset(self, dataset: KnowledgeDataset) -> EvaluationMetrics:
        """Evaluates editor across entire dataset with sequential retention tracking."""
        efficacies: List[float] = []
        generalities: List[float] = []
        localities: List[float] = []
        portabilities: List[float] = []
        latencies: List[float] = []
        delta_norms: List[float] = []

        checkpoint = self.editor.create_checkpoint()

        for req in dataset:
            t0 = time.time()
            res = self.editor.edit(req)
            latencies.append((time.time() - t0) * 1000.0)
            delta_norms.append(res.delta_weight_norm)

            efficacies.append(compute_efficacy_score(self.wrapper, req))
            generalities.append(compute_generality_score(self.wrapper, req))
            localities.append(compute_locality_score(self.wrapper, req))
            portabilities.append(compute_portability_score(self.wrapper, req))

        # Check retention of earlier edits (ripple degradation test)
        retained_count = 0
        test_sample = dataset.requests[:min(5, len(dataset))]
        for req in test_sample:
            if compute_efficacy_score(self.wrapper, req) >= 0.5:
                retained_count += 1
        retention = retained_count / len(test_sample) if test_sample else 1.0

        # Rollback model to baseline
        self.editor.rollback(checkpoint)

        return EvaluationMetrics(
            efficacy=float(np.mean(efficacies)) if efficacies else 0.0,
            generality=float(np.mean(generalities)) if generalities else 0.0,
            locality=float(np.mean(localities)) if localities else 0.0,
            portability=float(np.mean(portabilities)) if portabilities else 0.0,
            ripple_retention=float(retention),
            edit_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
            delta_norm=float(np.mean(delta_norms)) if delta_norms else 0.0,
        )
