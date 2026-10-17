"""Unit tests for Knowledge Mixture-of-Experts (K-MoE) editor."""

import pytest
import torch
from editmind.models import UnifiedModelWrapper
from editmind.core.types import EditRequest, EditResult
from editmind.core.registry import EditorRegistry
from editmind.editors.kmoe import (
    KnowledgeMoEEditor,
    KMoERoutingLayer,
    LowRankExpert,
)


@pytest.fixture
def toy_wrapper():
    torch.manual_seed(42)
    return UnifiedModelWrapper.create_toy_model(hidden_dim=64, num_layers=4)


def test_low_rank_expert():
    expert = LowRankExpert(in_features=64, out_features=64, rank=4)
    x = torch.randn(2, 5, 64)
    out = expert(x)
    assert out.shape == (2, 5, 64)


def test_kmoe_routing_layer():
    moe = KMoERoutingLayer(in_features=64, out_features=64, num_experts=4, top_k=1, rank=4)
    x = torch.randn(2, 5, 64)
    out, topk_indices, weights = moe(x)
    assert out.shape == (2, 5, 64)
    assert topk_indices.shape == (2, 1)
    assert weights.shape == (2, 1)


def test_kmoe_editor_execution(toy_wrapper):
    editor = EditorRegistry.create("kmoe", model_wrapper=toy_wrapper, target_layer=2)
    assert isinstance(editor, KnowledgeMoEEditor)
    assert editor.name == "kmoe"

    req = EditRequest(
        prompt="The capital of France is",
        target_new="Rome",
        ground_truth="Paris",
        subject="France",
    )

    res = editor.edit(req)
    assert isinstance(res, EditResult)
    assert res.editor_name == "kmoe"
    assert res.execution_time_sec > 0.0
    assert "assigned_expert" in res.details
    assert "total_experts" in res.details
    assert 0 <= res.details["assigned_expert"] < 4
