"""Unit tests for Steering Vector (ActAdd / Representation Engineering) Editor."""

import pytest
import torch
from editmind.models import UnifiedModelWrapper
from editmind.core.types import EditRequest, EditResult
from editmind.core.registry import EditorRegistry
from editmind.editors.steer import SteeringVectorEditor, SteeringVector


@pytest.fixture
def toy_wrapper():
    torch.manual_seed(42)
    return UnifiedModelWrapper.create_toy_model(hidden_dim=64, num_layers=4)


def test_steering_vector_editor_registry(toy_wrapper):
    editor = EditorRegistry.create("steer", model_wrapper=toy_wrapper, target_layer=2)
    assert isinstance(editor, SteeringVectorEditor)
    assert editor.name == "steer"
    assert editor.target_layer == 2


def test_compute_steering_vector(toy_wrapper):
    editor = SteeringVectorEditor(toy_wrapper, target_layer=2, max_norm=4.0)
    v = editor.compute_steering_vector(
        prompt="The capital of France is",
        target_new="Rome",
        ground_truth="Paris",
    )
    assert isinstance(v, torch.Tensor)
    assert v.dim() == 1
    assert v.shape[0] == 64
    assert torch.norm(v, p=2) <= 4.0 + 1e-4


def test_steering_vector_edit_execution(toy_wrapper):
    editor = EditorRegistry.create("steer", model_wrapper=toy_wrapper, target_layer=2, coefficient=2.0)
    
    req = EditRequest(
        prompt="The capital of France is",
        target_new="Rome",
        ground_truth="Paris",
        subject="France",
    )

    res = editor.edit(req)
    assert isinstance(res, EditResult)
    assert res.editor_name == "steer"
    assert res.execution_time_sec > 0.0
    assert "target_layer" in res.details
    assert res.details["weight_modification"] == 0.0
    assert res.details["active_vectors_count"] == 1

    # Verify vector can be cleared/removed
    assert len(editor.active_vectors) == 1
    editor.clear_all_steering()
    assert len(editor.active_vectors) == 0
