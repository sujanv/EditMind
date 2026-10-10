"""Unit tests for Safety Guardrails, Conflict Detection, and Machine Unlearning."""

import pytest
from editmind.models import UnifiedModelWrapper
from editmind.core.types import EditRequest
from editmind.safety import (
    KnowledgeConflictDetector,
    MachineUnlearner,
)


def test_conflict_detection():
    detector = KnowledgeConflictDetector()

    req1 = EditRequest(
        prompt="The Eiffel Tower is in",
        subject="Eiffel Tower",
        relation="located in",
        target_new="Rome",
        ground_truth="Paris",
    )
    rep1 = detector.check_request(req1)
    assert rep1.has_conflict is False
    detector.register_fact(req1)

    # Attempt direct contradiction: Eiffel Tower -> Berlin
    req2 = EditRequest(
        prompt="The Eiffel Tower is in",
        subject="Eiffel Tower",
        relation="located in",
        target_new="Berlin",
        ground_truth="Paris",
    )
    rep2 = detector.check_request(req2)
    assert rep2.has_conflict is True
    assert rep2.conflict_type == "contradiction"


def test_machine_unlearning():
    wrapper = UnifiedModelWrapper.create_toy_model(hidden_dim=64, num_layers=4)
    unlearner = MachineUnlearner(wrapper, editor_name="grace")

    res = unlearner.unlearn_fact(
        prompt="The secret passcode of Alice is",
        target_to_erase="Paris", # simulate sensitive token
        subject="passcode",
        replacement_token="unknown",
    )

    assert res.target_erased == "Paris"
    assert res.success is True
    assert res.execution_time_sec > 0.0
