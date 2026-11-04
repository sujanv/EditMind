"""Unit tests for Factual Reciprocal Consistency and Bidirectional Knowledge Auditor."""

import pytest
import torch
from editmind.models import UnifiedModelWrapper
from editmind.core.types import EditRequest
from editmind.core.registry import EditorRegistry
from editmind.evaluation.reciprocal import (
    FactualReciprocalAuditor,
    ReciprocalAuditResult,
    ReciprocalPairProbe,
    CANONICAL_INVERSES,
)


@pytest.fixture
def toy_wrapper():
    torch.manual_seed(42)
    return UnifiedModelWrapper.create_toy_model(hidden_dim=64, num_layers=4)


def test_reciprocal_probe_generation(toy_wrapper):
    auditor = FactualReciprocalAuditor(toy_wrapper)
    probe = auditor.generate_reciprocal_probe(
        subject="France",
        relation="capital",
        target="Paris",
    )
    assert isinstance(probe, ReciprocalPairProbe)
    assert "France" in probe.forward_prompt
    assert probe.forward_target == "Paris"
    assert "Paris" in probe.inverse_prompt
    assert probe.inverse_target == "France"
    assert probe.relation == "capital"
    assert probe.inverse_relation == "capital_of"


def test_custom_inverse_rule_registration(toy_wrapper):
    auditor = FactualReciprocalAuditor(toy_wrapper)
    auditor.register_inverse_rule(
        relation="mentored",
        inverse_relation="mentored_by",
        forward_template="{subject} mentored the student",
        inverse_template="{target} was mentored by the advisor",
    )
    probe = auditor.generate_reciprocal_probe(
        subject="Socrates",
        relation="mentored",
        target="Plato",
    )
    assert probe.relation == "mentored"
    assert probe.inverse_relation == "mentored_by"
    assert "Socrates mentored the student" == probe.forward_prompt
    assert "Plato was mentored by the advisor" == probe.inverse_prompt


def test_audit_edit_reciprocity_execution(toy_wrapper):
    auditor = FactualReciprocalAuditor(toy_wrapper)
    result = auditor.audit_edit_reciprocity(
        subject="France",
        relation="capital",
        target_new="Rome",
    )
    assert isinstance(result, ReciprocalAuditResult)
    assert result.subject == "France"
    assert result.target == "Rome"
    assert 0.0 <= result.directional_asymmetry_gap <= 1.0
    assert 0.0 <= result.reciprocal_consistency_score <= 1.0
    assert isinstance(result.reversal_curse_detected, bool)
    assert len(result.summary) > 0

    d = result.to_dict()
    assert "directional_asymmetry_gap" in d
    assert "reciprocal_consistency_score" in d
    assert "reversal_curse_detected" in d


def test_reversal_curse_detection_edge_cases(toy_wrapper):
    auditor = FactualReciprocalAuditor(toy_wrapper)
    
    # Mock predict_next_token_probs to simulate extreme reversal curse
    original_predict = toy_wrapper.predict_next_token_probs
    try:
        def mock_predict(prompt: str):
            if "capital of France" in prompt:
                return {"Rome": 0.95}
            elif "capital is Rome" in prompt:
                return {"France": 0.01}
            return {}

        toy_wrapper.predict_next_token_probs = mock_predict
        res = auditor.audit_edit_reciprocity(
            subject="France",
            relation="capital",
            target_new="Rome",
        )
        assert res.reversal_curse_detected is True
        assert res.directional_asymmetry_gap > 0.8
        assert "Reversal Curse detected" in res.summary
    finally:
        toy_wrapper.predict_next_token_probs = original_predict
