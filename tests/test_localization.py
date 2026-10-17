"""Tests for gradient-informed consensus layer localization."""

import pytest
import torch
from editmind.models.model_wrapper import UnifiedModelWrapper
from editmind.models.localization import (
    ConsensusLayerRanker,
    FisherLayerRanker,
    IntegratedGradientsAttributor,
    ConsensusLocalizationReport,
)


@pytest.fixture
def toy_wrapper():
    torch.manual_seed(42)
    return UnifiedModelWrapper.create_toy_model(hidden_dim=64, num_layers=4)


def test_fisher_layer_ranker(toy_wrapper):
    ranker = FisherLayerRanker(toy_wrapper)
    scores = ranker.compute_layer_fisher("The capital of France is", "Paris")
    assert len(scores) == 4
    for layer_idx, score in scores.items():
        assert 0 <= layer_idx < 4
        assert score >= 0.0


def test_integrated_gradients_attributor(toy_wrapper):
    attributor = IntegratedGradientsAttributor(toy_wrapper, steps=5)
    results = attributor.attribute_tokens("The capital of France is", "Paris")
    assert len(results) > 0
    total_score = sum(r.score for r in results)
    assert abs(total_score - 1.0) < 0.1  # normalized sum ~ 1.0
    for r in results:
        assert isinstance(r.token, str)
        assert r.position >= 0


def test_consensus_layer_ranker(toy_wrapper):
    ranker = ConsensusLayerRanker(toy_wrapper)
    report = ranker.locate_knowledge(
        prompt="The capital of France is",
        subject="France",
        target="Paris",
        top_k=2,
    )
    assert isinstance(report, ConsensusLocalizationReport)
    assert len(report.recommended_layers) == 2
    for l in report.recommended_layers:
        assert 0 <= l < 4
    assert 0.0 <= report.confidence <= 1.0
    assert len(report.layer_scores) == 4
    assert len(report.token_attributions) > 0

    report_dict = report.to_dict()
    assert report_dict["prompt"] == "The capital of France is"
    assert "recommended_layers" in report_dict
    assert "confidence" in report_dict
