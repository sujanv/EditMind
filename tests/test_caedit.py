"""Unit tests for Cross-Attention Knowledge Editor (CA-Edit)."""

import pytest
import torch
from editmind.models import UnifiedModelWrapper
from editmind.core.types import EditRequest, EditResult
from editmind.core.registry import EditorRegistry
from editmind.editors.caedit import CrossAttentionEditor


@pytest.fixture
def toy_wrapper():
    torch.manual_seed(42)
    return UnifiedModelWrapper.create_toy_model(hidden_dim=64, num_layers=4)


def test_caedit_registry(toy_wrapper):
    editor = EditorRegistry.create("caedit", model_wrapper=toy_wrapper, target_layer=2)
    assert isinstance(editor, CrossAttentionEditor)
    assert editor.name == "caedit"


def test_caedit_execution(toy_wrapper):
    editor = CrossAttentionEditor(toy_wrapper, target_layer=2, v_num_grad_steps=10)
    req = EditRequest(
        prompt="The Eiffel Tower is located in",
        target_new="Berlin",
        ground_truth="Paris",
        subject="Eiffel Tower",
    )
    result = editor.edit(req)
    assert isinstance(result, EditResult)
    assert result.editor_name == "caedit"
    assert result.execution_time_sec > 0.0
    assert result.delta_weight_norm > 0.0
    assert "mechanism" in result.details
    assert result.details["mechanism"] == "attention_projection_rank1"
    assert result.details["target_layer"] == 2


def test_caedit_rollback(toy_wrapper):
    editor = CrossAttentionEditor(toy_wrapper, target_layer=1, v_num_grad_steps=5)
    block = toy_wrapper.get_layer(1)
    orig_weight = block.attn.c_proj.weight.clone()

    req = EditRequest(
        prompt="The capital of Spain is",
        target_new="Lisbon",
        ground_truth="Madrid",
        subject="Spain",
    )
    res = editor.edit(req)
    assert not torch.allclose(block.attn.c_proj.weight, orig_weight)

    # Rollback checkpoint
    assert res.checkpoint_id is not None
    restored = editor.rollback(res.checkpoint_id)
    assert restored is True
    assert torch.allclose(block.attn.c_proj.weight, orig_weight, atol=1e-6)
