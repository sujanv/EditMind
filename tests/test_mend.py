"""Unit tests for MEND (Model Editor Networks with Gradient Decomposition)."""

import pytest
from editmind.models import UnifiedModelWrapper
from editmind.core.types import EditRequest
from editmind.editors.mend import MENDEditor


def test_mend_edit():
    wrapper = UnifiedModelWrapper.create_toy_model(hidden_dim=64, num_layers=4)
    editor = MENDEditor(model_wrapper=wrapper, config={"target_layer": 2, "lr": 0.05})

    req = EditRequest(
        prompt="The Eiffel Tower is in",
        target_new="Rome",
        ground_truth="Paris",
        subject="Eiffel Tower",
    )

    res = editor.edit(req)
    assert res.editor_name == "mend"
    assert res.delta_weight_norm > 0.0
    assert len(editor.get_edit_history()) == 1

    # Rollback test
    rolled_back = editor.rollback(res.checkpoint_id)
    assert rolled_back is True
