"""Unit tests for Multilingual Knowledge Transfer."""

import pytest
from editmind.models import UnifiedModelWrapper
from editmind.core.registry import EditorRegistry
from editmind.multilingual import (
    load_multilingual_benchmark,
    CrossLingualEvaluator,
)


def test_cross_lingual_transfer():
    wrapper = UnifiedModelWrapper.create_toy_model(hidden_dim=64, num_layers=4)
    editor = EditorRegistry.create("grace", model_wrapper=wrapper, config={"target_layer": 2, "inner_steps": 10})
    evaluator = CrossLingualEvaluator(editor)

    dataset = load_multilingual_benchmark()
    assert len(dataset) >= 2
    first = dataset[0]

    report = evaluator.evaluate_fact_transfer(first)
    assert report.fact_subject == "Eiffel Tower"
    assert "es" in report.transfer_scores
    assert "fr" in report.transfer_scores
    assert 0.0 <= report.average_transfer_rate <= 1.0
