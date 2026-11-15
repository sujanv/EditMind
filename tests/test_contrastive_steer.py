"""Unit tests for Contrastive Steering Engine (DiffSteer)."""

import pytest
import torch

from editmind.models import UnifiedModelWrapper
from editmind.safety.contrastive_steer import (
    ContrastivePromptPair,
    ContrastiveSteerResult,
    ContrastiveSteeringEngine,
)


@pytest.fixture
def toy_wrapper():
    torch.manual_seed(42)
    return UnifiedModelWrapper.create_toy_model(hidden_dim=64, num_layers=4)


def test_contrastive_steering_vector_computation(toy_wrapper):
    engine = ContrastiveSteeringEngine(toy_wrapper.model, toy_wrapper.tokenizer)

    pairs = [
        ContrastivePromptPair(
            positive_prompt="The capital of France is Rome",
            negative_prompt="The capital of France is Paris",
        ),
        ContrastivePromptPair(
            positive_prompt="Paris is not the capital of France, Rome is",
            negative_prompt="Rome is not the capital of France, Paris is",
        ),
    ]

    computed = engine.compute_steering_vector(pairs, layer_indices=[1, 2])
    assert 1 in computed
    assert 2 in computed
    assert computed[1].shape[-1] == 64
    assert pytest.approx(torch.norm(computed[1]).item(), abs=1e-3) == 1.0
    assert pytest.approx(torch.norm(computed[2]).item(), abs=1e-3) == 1.0


def test_contrastive_steering_evaluation(toy_wrapper):
    engine = ContrastiveSteeringEngine(toy_wrapper.model, toy_wrapper.tokenizer)

    pairs = [
        ContrastivePromptPair(
            positive_prompt="Positive truthful context",
            negative_prompt="Negative untruthful context",
        )
    ]
    engine.compute_steering_vector(pairs, layer_indices=[2])

    result = engine.evaluate_steering(
        probe_prompt="The capital of France is",
        desired_token_id=2,
        undesired_token_id=1,
        layer_idx=2,
        coefficient=1.5,
    )

    assert isinstance(result, ContrastiveSteerResult)
    assert result.target_layer == 2
    assert result.coefficient == 1.5
    assert result.vector_norm > 0.0
    assert result.neutral_drift_kl >= 0.0

    d = result.to_dict()
    assert "concept_alignment_score" in d
    assert "suppression_ratio" in d
    assert "neutral_drift_kl" in d


def test_steering_hooks_lifecycle(toy_wrapper):
    engine = ContrastiveSteeringEngine(toy_wrapper.model, toy_wrapper.tokenizer)
    pairs = [ContrastivePromptPair("Good prompt", "Bad prompt")]
    engine.compute_steering_vector(pairs, [2])

    assert len(engine.active_hooks) == 0
    engine.attach_steering_hooks({2: 2.0})
    assert len(engine.active_hooks) > 0

    engine.remove_hooks()
    assert len(engine.active_hooks) == 0
