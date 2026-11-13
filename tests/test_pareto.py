"""Unit tests for Plasticity-Stability Pareto Frontier Analyzer."""

import numpy as np
import pytest
from editmind.models import UnifiedModelWrapper
from editmind.core.registry import EditorRegistry
from editmind.data import load_counterfact_dataset
from editmind.continual.engine import ContinualKnowledgeEditor, ContinualEditTrajectory
from editmind.continual.pareto import (
    ParetoPoint,
    ParetoFrontierReport,
    ParetoPlasticityAuditor,
)


def test_pareto_auditor_synthetic_trajectory():
    # Synthetic 4x4 matrix with decaying retention
    M = np.array([
        [0.95, 0.85, 0.70, 0.45],
        [0.00, 0.90, 0.80, 0.60],
        [0.00, 0.00, 0.92, 0.75],
        [0.00, 0.00, 0.00, 0.88],
    ], dtype=np.float32)

    traj = ContinualEditTrajectory(
        requests=[],
        results=[],
        interference_matrix=M,
        average_retention=float(np.mean(M[:, 3])),
        catastrophic_forgetting_rate=0.2,
        wall_clock_time_sec=1.5,
    )

    auditor = ParetoPlasticityAuditor(collapse_threshold=0.7)
    report = auditor.audit_trajectory(traj, editor_name="synthetic_editor")

    assert report.editor_name == "synthetic_editor"
    assert report.trajectory_length == 4
    assert len(report.pareto_points) == 4
    # Step 1: retention 0.95 >= 0.7
    # Step 2: retention (0.85+0.90)/2 = 0.875 >= 0.7
    # Step 3: retention (0.70+0.80+0.92)/3 = 0.806 >= 0.7
    # Step 4: retention (0.45+0.60+0.75+0.88)/4 = 0.67 < 0.7 => collapse at step 4
    assert report.collapse_step == 4
    assert 0.0 <= report.aurc <= 1.0
    assert report.interference_dispersion_index >= 0.0
    assert 1 <= report.optimal_operating_point <= 4

    d = report.to_dict()
    assert d["collapse_step"] == 4
    assert "pareto_points" in d


def test_pareto_auditor_live_model():
    wrapper = UnifiedModelWrapper.create_toy_model(hidden_dim=64, num_layers=4)
    editor = EditorRegistry.create("grace", model_wrapper=wrapper, config={"target_layer": 2, "inner_steps": 5})
    continual = ContinualKnowledgeEditor(editor)

    dataset = load_counterfact_dataset(sample_count=3)
    trajectory = continual.run_sequential_stream(dataset.requests)

    auditor = ParetoPlasticityAuditor(collapse_threshold=0.3)
    report = auditor.audit_trajectory(trajectory, editor_name="grace")

    assert isinstance(report, ParetoFrontierReport)
    assert report.editor_name == "grace"
    assert report.trajectory_length == 3
    assert len(report.pareto_points) == 3
    for pt in report.pareto_points:
        assert 0.0 <= pt.instantaneous_efficacy <= 1.0
        assert 0.0 <= pt.cumulative_retention <= 1.0
        assert 0.0 <= pt.harmonic_tradeoff <= 1.0
