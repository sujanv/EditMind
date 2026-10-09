"""Unit tests for MEMIT (Mass-Editing Memory in a Transformer)."""

import pytest
from editmind.models import UnifiedModelWrapper
from editmind.core.types import EditRequest
from editmind.editors.memit import MEMITEditor


def test_memit_batch_editing():
    wrapper = UnifiedModelWrapper.create_toy_model(hidden_dim=64, num_layers=4)
    editor = MEMITEditor(model_wrapper=wrapper, config={"layers": [1, 2], "v_num_grad_steps": 12})

    reqs = [
        EditRequest(
            prompt="The Eiffel Tower is in",
            target_new="Rome",
            ground_truth="Paris",
            subject="Eiffel Tower",
        ),
        EditRequest(
            prompt="Messi plays for",
            target_new="Miami",
            ground_truth="PSG",
            subject="Messi",
        ),
    ]

    results = editor.batch_edit(reqs)
    assert len(results) == 2
    for r in results:
        assert r.editor_name == "memit"
        assert r.post_edit_target_prob > r.pre_edit_target_prob
        assert r.delta_weight_norm > 0.0

    # Test rollback
    rolled_back = editor.rollback(results[0].checkpoint_id)
    assert rolled_back is True
