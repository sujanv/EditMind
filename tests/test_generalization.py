"""Unit tests for Counterfactual Generalization Stress-Tester."""

import pytest
import torch
from editmind.models import UnifiedModelWrapper
from editmind.evaluation.generalization import (
    PerturbationProbe,
    GeneralizationAuditReport,
    GeneralizationAuditor,
)


@pytest.fixture
def toy_wrapper():
    torch.manual_seed(42)
    return UnifiedModelWrapper.create_toy_model(hidden_dim=64, num_layers=4)


def test_generalization_generation(toy_wrapper):
    auditor = GeneralizationAuditor(toy_wrapper)
    perturbations = auditor.generate_perturbations(
        base_prompt="The capital of France is",
        subject="France",
    )
    assert len(perturbations) >= 5
    categories = {cat for _, cat in perturbations}
    assert "distractor" in categories
    assert "rephrase" in categories


def test_generalization_auditor(toy_wrapper):
    auditor = GeneralizationAuditor(toy_wrapper)
    report = auditor.audit_generalization(
        base_prompt="The capital of France is",
        target_new="Paris",
        ground_truth="Rome",
        subject="France",
    )

    assert isinstance(report, GeneralizationAuditReport)
    assert report.total_perturbations >= 5
    assert 0.0 <= report.generalization_pass_rate <= 1.0
    assert report.semantic_generalization_robustness_index >= 0.0
    assert report.prompt_sensitivity_variance >= 0.0
    assert "distractor" in report.category_breakdown
    assert len(report.probes) == report.total_perturbations

    d = report.to_dict()
    assert "semantic_generalization_robustness_index" in d
    assert "prompt_sensitivity_variance" in d
    assert len(d["probes"]) == report.total_perturbations
