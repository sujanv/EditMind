"""Tests for Concept Erasure, Representation Scrubbing, and Orthogonal Gradient Unlearning."""

import pytest
import torch
from editmind.models import UnifiedModelWrapper
from editmind.safety import (
    ConceptEraser,
    OrthogonalGradientUnlearner,
    ConceptUnlearnReport,
)


@pytest.fixture
def toy_wrapper():
    torch.manual_seed(42)
    return UnifiedModelWrapper.create_toy_model(hidden_dim=64, num_layers=4)


def test_concept_eraser(toy_wrapper):
    eraser = ConceptEraser(toy_wrapper)

    forget_prompts = [
        "The secret passcode of Alice is",
        "The classified encryption key of Alice is",
        "Alice confidential password was",
    ]
    retain_prompts = [
        "The capital of France is Paris",
        "Shakespeare was a famous English writer",
        "Berlin is the capital of Germany",
    ]

    report = eraser.erase_concept(
        concept_name="confidential_credentials",
        forget_prompts=forget_prompts,
        retain_prompts=retain_prompts,
        target_layers=[2, 3],
    )

    assert isinstance(report, ConceptUnlearnReport)
    assert report.concept_name == "confidential_credentials"
    assert report.forget_samples_count == 3
    assert report.retain_samples_count == 3
    assert 0.0 <= report.pre_leakage_score <= 1.0
    assert 0.0 <= report.post_leakage_score <= 1.0
    assert 0.0 <= report.utility_retention_score <= 1.0
    assert 0.0 <= report.unlearning_completeness <= 1.0
    assert report.scrubbed_layers == [2, 3]
    assert report.execution_time_sec > 0.0

    rep_dict = report.to_dict()
    assert rep_dict["concept_name"] == "confidential_credentials"
    assert "pre_leakage_score" in rep_dict


def test_orthogonal_gradient_unlearner(toy_wrapper):
    unlearner = OrthogonalGradientUnlearner(toy_wrapper, lr=1e-3)
    loss = unlearner.unlearn_orthogonal(
        forget_prompt="The passcode is",
        forget_target="Paris",
        retain_prompt="The capital of France is",
        retain_target="Paris",
        layer_idx=1,
    )
    assert loss >= 0.0
