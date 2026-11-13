"""Unit tests for Dynamic Knowledge Domain Router."""

import pytest
from editmind.routing import (
    KnowledgeDomainRouter,
    DomainClassificationResult,
    RouterMetrics,
    DEFAULT_DOMAINS,
)


def test_default_domain_routing():
    router = KnowledgeDomainRouter()

    # Geography domain test
    res_geo = router.route_prompt("The capital of France is located near Paris")
    assert isinstance(res_geo, DomainClassificationResult)
    assert res_geo.detected_domain == "geography"
    assert res_geo.recommended_paradigm == "rome"
    assert res_geo.confidence > 0.0

    # Privacy domain test
    res_priv = router.route_prompt("The secret root password and confidential token is")
    assert res_priv.detected_domain == "privacy_pii"
    assert res_priv.recommended_paradigm == "grace"

    # Transient override test
    res_steer = router.route_prompt("Hypothetically suppose we temporarily imagine")
    assert res_steer.detected_domain == "transient_override"
    assert res_steer.recommended_paradigm == "steer"

    # Science/Tech test
    res_sci = router.route_prompt("The planet orbit physics and software algorithm was invented")
    assert res_sci.detected_domain == "science_tech"
    assert res_sci.recommended_paradigm == "alphaedit"


def test_custom_domain_registration():
    router = KnowledgeDomainRouter()
    router.register_domain(
        domain_name="medical",
        keywords=["symptom", "disease", "treatment", "medicine", "dosage"],
        recommended_paradigm="alphaedit",
        description="Clinical & medical knowledge requiring null-space safety",
    )

    res = router.route_prompt("The primary symptom and treatment dosage for diabetes is")
    assert res.detected_domain == "medical"
    assert res.recommended_paradigm == "alphaedit"
    assert "medical" in res.domain_scores


def test_router_metrics_aggregation():
    router = KnowledgeDomainRouter()
    router.route_prompt("The capital of Spain is Madrid")
    router.route_prompt("The confidential password is root")
    router.route_prompt("The author of Hamlet was Shakespeare")

    metrics = router.get_metrics()
    assert isinstance(metrics, RouterMetrics)
    assert metrics.total_dispatches == 3
    assert len(metrics.domain_counts) > 0
    assert len(metrics.paradigm_counts) > 0
    assert 0.0 <= metrics.average_confidence <= 1.0

    d = metrics.to_dict()
    assert d["total_dispatches"] == 3
    assert "average_confidence" in d
