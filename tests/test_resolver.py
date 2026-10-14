import pytest
from editmind.core.types import EditRequest
from editmind.safety.resolver import (
     KnowledgeConflictResolver,
     ResolutionStrategy,
     ResolutionDecision,
 )


def test_resolver_first_entry_and_idempotence():
    resolver = KnowledgeConflictResolver()
    req = EditRequest(
        prompt="The capital of France is",
        target_new="Paris",
        subject="France",
        relation="capital is",
    )
    decision = resolver.resolve(req)
    assert decision.action == "ACCEPTED"
    assert decision.effective_target == "Paris"

    # Same target should be accepted idempotently
    decision2 = resolver.resolve(req)
    assert decision2.action == "ACCEPTED"
    assert decision2.effective_target == "Paris"


def test_resolver_latest_wins():
    resolver = KnowledgeConflictResolver(default_strategy=ResolutionStrategy.LATEST_WINS)
    req1 = EditRequest(
        prompt="The president of CountryX is",
        target_new="Alice",
        subject="CountryX",
        relation="president",
    )
    resolver.resolve(req1)

    req2 = EditRequest(
        prompt="The president of CountryX is",
        target_new="Bob",
        subject="CountryX",
        relation="president",
    )
    decision = resolver.resolve(req2)
    assert decision.action == "SUPERSEDED"
    assert decision.effective_target == "Bob"
    assert decision.resolved_fact.version == 2


def test_resolver_confidence_weighting():
    resolver = KnowledgeConflictResolver()
    req1 = EditRequest(
        prompt="The founder of StartupY is",
        target_new="Carol",
        subject="StartupY",
        relation="founder",
    )
    resolver.resolve(req1, strategy=ResolutionStrategy.CONFIDENCE_WEIGHTED, confidence=0.90)

    # Low confidence should be rejected
    req2 = EditRequest(
        prompt="The founder of StartupY is",
        target_new="Dave",
        subject="StartupY",
        relation="founder",
    )
    decision_low = resolver.resolve(req2, strategy=ResolutionStrategy.CONFIDENCE_WEIGHTED, confidence=0.50)
    assert decision_low.action == "REJECTED"
    assert decision_low.effective_target == "Carol"

    # High confidence should supersede
    decision_high = resolver.resolve(req2, strategy=ResolutionStrategy.CONFIDENCE_WEIGHTED, confidence=0.95)
    assert decision_high.action == "SUPERSEDED"
    assert decision_high.effective_target == "Dave"


def test_resolver_multi_valued_merge():
    resolver = KnowledgeConflictResolver()
    req1 = EditRequest(
        prompt="Languages spoken in Switzerland include",
        target_new="German",
        subject="Switzerland",
        relation="spoken language",
    )
    resolver.resolve(req1)

    req2 = EditRequest(
        prompt="Languages spoken in Switzerland include",
        target_new="French",
        subject="Switzerland",
        relation="spoken language",
    )
    decision = resolver.resolve(req2)
    assert decision.action == "MERGED"
    assert "French" in decision.effective_target
    assert "German" in decision.effective_target


def test_resolver_temporal_scoping():
    resolver = KnowledgeConflictResolver()
    req1 = EditRequest(
        prompt="Lionel Messi plays for",
        target_new="Barcelona",
        subject="Lionel Messi",
        relation="plays for",
    )
    resolver.resolve(req1, valid_from="2004", valid_to="2021")

    req2 = EditRequest(
        prompt="Lionel Messi plays for",
        target_new="Inter Miami",
        subject="Lionel Messi",
        relation="plays for",
    )
    decision = resolver.resolve(req2, strategy=ResolutionStrategy.TEMPORAL_SCOPING, valid_from="2023")
    assert decision.action == "TEMPORAL_SPLIT"
    assert decision.effective_target == "Inter Miami"
    assert decision.resolved_fact.valid_from == "2023"
