"""Unit tests for Fine-Tuning Baselines (FT-L and LoRA-Edit)."""

import pytest
from editmind.models import UnifiedModelWrapper
from editmind.core.types import EditRequest
from editmind.editors.ft import FTLEditor, LoRAEditor


def test_ft_l_edit():
    wrapper = UnifiedModelWrapper.create_toy_model(hidden_dim=64, num_layers=4)
    editor = FTLEditor(model_wrapper=wrapper, config={"target_layer": 2, "num_steps": 15})

    req = EditRequest(
        prompt="The Eiffel Tower is in",
        target_new="Rome",
        ground_truth="Paris",
        subject="Eiffel Tower",
    )

    res = editor.edit(req)
    assert res.editor_name == "ft_l"
    assert res.post_edit_target_prob > res.pre_edit_target_prob
    assert res.delta_weight_norm > 0.0

    # Rollback
    rolled_back = editor.rollback(res.checkpoint_id)
    assert rolled_back is True


def test_lora_edit():
    wrapper = UnifiedModelWrapper.create_toy_model(hidden_dim=64, num_layers=4)
    editor = LoRAEditor(model_wrapper=wrapper, config={"target_layer": 2, "num_steps": 15})

    req = EditRequest(
        prompt="The Eiffel Tower is in",
        target_new="Rome",
        ground_truth="Paris",
        subject="Eiffel Tower",
    )

    res = editor.edit(req)
    assert res.editor_name == "lora_edit"
    assert res.post_edit_target_prob > res.pre_edit_target_prob
    assert res.delta_weight_norm > 0.0

    # Rollback
    rolled_back = editor.rollback(res.checkpoint_id)
    assert rolled_back is True
