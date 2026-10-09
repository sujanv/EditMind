"""Unit tests for EditMind core types and registry."""

import pytest
from editmind.core.types import (
    KnowledgeTriple,
    EditRequest,
    EditResult,
    EvaluationMetrics,
)
from editmind.core.registry import EditorRegistry, register_editor
from editmind.core.base_editor import BaseKnowledgeEditor
from editmind.core.config import EditMindConfig, load_config


def test_knowledge_triple():
    triple = KnowledgeTriple(
        subject="Eiffel Tower",
        relation="located in",
        target="Rome",
        old_target="Paris"
    )
    assert str(triple) == "(Eiffel Tower, located in, Rome)"
    data = triple.to_dict()
    reconstructed = KnowledgeTriple.from_dict(data)
    assert reconstructed.target == "Rome"
    assert reconstructed.old_target == "Paris"


def test_edit_request_and_result():
    req = EditRequest(
        prompt="The Eiffel Tower is in",
        target_new="Rome",
        ground_truth="Paris",
        subject="Eiffel Tower",
        relation="located in",
        rephrase_prompts=["Where is the Eiffel Tower located? The city of"],
    )
    assert req.triple is not None
    assert req.triple.subject == "Eiffel Tower"

    result = EditResult(
        request_id=req.request_id,
        editor_name="rome",
        success=True,
        execution_time_sec=0.12,
        pre_edit_target_prob=0.01,
        post_edit_target_prob=0.88,
        delta_weight_norm=0.05,
    )
    assert result.success is True
    assert result.post_edit_target_prob > result.pre_edit_target_prob


def test_evaluation_metrics():
    metrics = EvaluationMetrics(
        efficacy=0.95,
        generality=0.85,
        locality=0.90,
        portability=0.75,
        ripple_retention=0.88,
    )
    assert metrics.composite_score > 0.8
    assert "composite_score" in metrics.to_dict()


def test_registry():
    @register_editor("dummy_test_editor")
    class DummyEditor(BaseKnowledgeEditor):
        @property
        def name(self):
            return "dummy_test_editor"

        def edit(self, request):
            return EditResult(
                request_id=request.request_id,
                editor_name=self.name,
                success=True,
                execution_time_sec=0.01,
            )

    assert "dummy_test_editor" in EditorRegistry.list_available()
    inst = EditorRegistry.create("dummy_test_editor", model_wrapper=None)
    assert inst.name == "dummy_test_editor"
