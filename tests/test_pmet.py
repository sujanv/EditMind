"""Unit tests for PMET (Penalizing Momentum to Enhance Knowledge Editing)."""

import pytest
from editmind.models import UnifiedModelWrapper
from editmind.core.types import EditRequest
from editmind.editors.pmet import PMETEditor


def test_pmet_edit():
    wrapper = UnifiedModelWrapper.create_toy_model(hidden_dim=64, num_layers=4)
    editor = PMETEditor(model_wrapper=wrapper, config={"target_layer": 2, "v_num_grad_steps": 25})

    req = EditRequest(
        prompt="The Eiffel Tower is in",
        target_new="Rome",
        ground_truth="Paris",
        subject="Eiffel Tower",
    )

    res = editor.edit(req)
    assert res.editor_name == "pmet"
    assert res.post_edit_target_prob > res.pre_edit_target_prob
    assert res.delta_weight_norm > 0.0

    # Rollback test
    rolled_back = editor.rollback(res.checkpoint_id)
    assert rolled_back is True
