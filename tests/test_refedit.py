"""Unit tests for Residual Error Feedback Knowledge Editor (RefEdit)."""

import pytest
import torch
from editmind.models import UnifiedModelWrapper
from editmind.core.types import EditRequest, EditResult
from editmind.core.registry import EditorRegistry
from editmind.editors.refedit import ResidualErrorFeedbackEditor


@pytest.fixture
def toy_wrapper():
    torch.manual_seed(42)
    return UnifiedModelWrapper.create_toy_model(hidden_dim=64, num_layers=4)


def test_refedit_registry(toy_wrapper):
    editor = EditorRegistry.create("refedit", model_wrapper=toy_wrapper, target_layer=2)
    assert isinstance(editor, ResidualErrorFeedbackEditor)
    assert editor.name == "refedit"


def test_refedit_execution(toy_wrapper):
    editor = ResidualErrorFeedbackEditor(
        toy_wrapper,
        target_layer=2,
        max_feedback_steps=3,
        v_num_grad_steps=10,
        target_margin=0.2,
    )
    req = EditRequest(
        prompt="The Eiffel Tower is in",
        target_new="Rome",
        ground_truth="Paris",
        subject="Eiffel Tower",
    )
    result = editor.edit(req)
    assert isinstance(result, EditResult)
    assert result.editor_name == "refedit"
    assert result.execution_time_sec > 0.0
    assert result.delta_weight_norm > 0.0
    assert "feedback_steps_executed" in result.details
    assert "trajectory" in result.details
    assert len(result.details["trajectory"]) >= 1


def test_refedit_rollback(toy_wrapper):
    editor = ResidualErrorFeedbackEditor(
        toy_wrapper,
        target_layer=1,
        max_feedback_steps=2,
        v_num_grad_steps=5,
    )
    mlp = toy_wrapper.model.layers[1].mlp
    orig_weight = mlp.down_proj.weight.clone()

    req = EditRequest(
        prompt="The capital of Spain is",
        target_new="Lisbon",
        ground_truth="Madrid",
        subject="Spain",
    )
    res = editor.edit(req)
    assert not torch.allclose(mlp.down_proj.weight, orig_weight)

    # Rollback checkpoint
    assert res.checkpoint_id is not None
    restored = editor.rollback(res.checkpoint_id)
    assert restored is True
    assert torch.allclose(mlp.down_proj.weight, orig_weight, atol=1e-6)
