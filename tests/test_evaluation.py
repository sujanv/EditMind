"""Unit tests for Evaluation Suite and MethodComparator."""

import pytest
from editmind.models import UnifiedModelWrapper
from editmind.core.types import EditRequest
from editmind.core.registry import EditorRegistry
from editmind.data import load_counterfact_dataset
from editmind.evaluation import KnowledgeEditorEvaluator, MethodComparator


def test_evaluator_request():
    wrapper = UnifiedModelWrapper.create_toy_model(hidden_dim=64, num_layers=4)
    editor = EditorRegistry.create("rome", model_wrapper=wrapper, config={"target_layer": 2, "v_num_grad_steps": 10})
    evaluator = KnowledgeEditorEvaluator(editor)

    req = EditRequest(
        prompt="The Eiffel Tower is in",
        target_new="Rome",
        ground_truth="Paris",
        subject="Eiffel Tower",
        rephrase_prompts=["The location of Eiffel Tower is"],
        locality_prompts=[{"prompt": "The Colosseum is in", "target": "Rome"}],
    )

    metrics = evaluator.evaluate_request(req)
    assert 0.0 <= metrics.efficacy <= 1.0
    assert 0.0 <= metrics.locality <= 1.0
    assert metrics.edit_latency_ms > 0.0


def test_method_comparator():
    wrapper = UnifiedModelWrapper.create_toy_model(hidden_dim=64, num_layers=4)
    comparator = MethodComparator(wrapper, methods=["rome", "grace", "ike"])
    ds = load_counterfact_dataset(sample_count=2)

    results = comparator.compare_on_dataset(ds)
    assert "rome" in results
    assert "grace" in results
    assert "ike" in results

    table = comparator.generate_markdown_table(results)
    assert "ROME" in table
    assert "GRACE" in table
    assert "IKE" in table
