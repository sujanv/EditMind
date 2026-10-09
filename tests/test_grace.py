"""Unit tests for GRACE (General Retrieval and Adaptation using Compact Extensions)."""

import pytest
from editmind.models import UnifiedModelWrapper
from editmind.core.types import EditRequest
from editmind.editors.grace import GRACEEditor


def test_grace_edit():
    wrapper = UnifiedModelWrapper.create_toy_model(hidden_dim=64, num_layers=4)
    editor = GRACEEditor(model_wrapper=wrapper, config={"target_layer": 2, "inner_steps": 15})

    req = EditRequest(
        prompt="The Eiffel Tower is in",
        target_new="Rome",
        ground_truth="Paris",
        subject="Eiffel Tower",
    )

    res = editor.edit(req)
    assert res.editor_name == "grace"
    assert res.delta_weight_norm == 0.0 # Base weights preserved
    assert res.post_edit_target_prob > res.pre_edit_target_prob
    assert len(editor.codebook) == 1

    # Rollback
    rolled_back = editor.rollback(res.checkpoint_id)
    assert rolled_back is True
    assert len(editor.codebook) == 0
    editor.close()
