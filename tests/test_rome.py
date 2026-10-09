"""Unit tests for ROME (Rank-One Model Editing)."""

import pytest
from editmind.models import UnifiedModelWrapper
from editmind.core.types import EditRequest
from editmind.editors.rome import ROMEEditor


def test_rome_edit_single_fact():
    wrapper = UnifiedModelWrapper.create_toy_model(hidden_dim=64, num_layers=4)
    editor = ROMEEditor(model_wrapper=wrapper, config={"target_layer": 2, "v_num_grad_steps": 15})

    req = EditRequest(
        prompt="The Eiffel Tower is in",
        target_new="Rome",
        ground_truth="Paris",
        subject="Eiffel Tower",
    )

    res = editor.edit(req)
    assert res.editor_name == "rome"
    assert res.post_edit_target_prob > res.pre_edit_target_prob
    assert res.delta_weight_norm > 0.0

    # Test rollback
    assert len(editor.get_edit_history()) == 1
    rolled_back = editor.rollback(res.checkpoint_id)
    assert rolled_back is True

    probs_after_rollback = wrapper.predict_next_token_probs(req.prompt)
    assert abs(probs_after_rollback.get("Rome", 0.0) - res.pre_edit_target_prob) < 1e-4
