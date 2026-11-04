"""Unit tests for Batch-AlphaEdit (Multi-Fact Null-Space Projection Knowledge Editing)."""

import pytest
import torch
from editmind.models import UnifiedModelWrapper
from editmind.core.types import EditRequest, EditResult
from editmind.core.registry import EditorRegistry
from editmind.editors.alphaedit import BatchAlphaEditor


@pytest.fixture
def toy_wrapper():
    torch.manual_seed(42)
    return UnifiedModelWrapper.create_toy_model(hidden_dim=64, num_layers=4)


def test_batch_alphaedit_registry(toy_wrapper):
    editor = EditorRegistry.create("batch_alphaedit", model_wrapper=toy_wrapper)
    assert isinstance(editor, BatchAlphaEditor)
    assert editor.name == "batch_alphaedit"


def test_batch_alphaedit_single_and_batch_execution(toy_wrapper):
    editor = BatchAlphaEditor(toy_wrapper)

    req1 = EditRequest(
        prompt="The Eiffel Tower is in",
        target_new="Rome",
        ground_truth="Paris",
        subject="Eiffel Tower",
    )
    req2 = EditRequest(
        prompt="The Colosseum is in",
        target_new="Paris",
        ground_truth="Rome",
        subject="Colosseum",
    )

    results = editor.batch_edit([req1, req2])
    assert len(results) == 2
    for res in results:
        assert isinstance(res, EditResult)
        assert res.editor_name == "batch_alphaedit"
        assert res.delta_weight_norm > 0.0
        assert res.details["batch_size"] == 2
        assert "null_space_dim" in res.details

    # Test single edit fallback
    res_single = editor.edit(req1)
    assert isinstance(res_single, EditResult)
    assert res_single.editor_name == "batch_alphaedit"
