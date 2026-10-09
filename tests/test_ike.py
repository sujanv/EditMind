"""Unit tests for IKE (In-Context Knowledge Editing)."""

import pytest
from editmind.models import UnifiedModelWrapper
from editmind.core.types import EditRequest
from editmind.editors.ike import IKEEditor


def test_ike_edit():
    wrapper = UnifiedModelWrapper.create_toy_model(hidden_dim=64, num_layers=4)
    editor = IKEEditor(model_wrapper=wrapper)

    req = EditRequest(
        prompt="The Eiffel Tower is in",
        target_new="Rome",
        ground_truth="Paris",
        subject="Eiffel Tower",
    )

    res = editor.edit(req)
    assert res.editor_name == "ike"
    assert res.delta_weight_norm == 0.0 # Non-parametric
    assert res.post_edit_target_prob > res.pre_edit_target_prob
    assert len(editor.knowledge_store) == 1

    # Rollback
    rolled_back = editor.rollback(res.checkpoint_id)
    assert rolled_back is True
    assert len(editor.knowledge_store) == 0
