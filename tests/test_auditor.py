import pytest
from editmind.safety.auditor import KnowledgeDriftAuditor, AuditReport
from editmind.models.model_wrapper import UnifiedModelWrapper


def test_auditor_identical_models():
    model_pre = UnifiedModelWrapper.create_toy_model(hidden_dim=64, num_layers=4, seed=42)
    model_post = UnifiedModelWrapper.create_toy_model(hidden_dim=64, num_layers=4, seed=42)

    auditor = KnowledgeDriftAuditor()
    report = auditor.audit(model_pre, model_post, reference_corpus=["The sky is blue today."])

    assert isinstance(report, AuditReport)
    assert report.verdict == "PASS"
    assert report.is_safe is True
    assert report.mean_kl_divergence < 0.01
    assert abs(report.perplexity_drift_ratio - 1.0) < 0.05
    assert not report.entropy_collapse_detected


def test_auditor_detects_drift():
    model_pre = UnifiedModelWrapper.create_toy_model(hidden_dim=64, num_layers=4, seed=42)
    model_post = UnifiedModelWrapper.create_toy_model(hidden_dim=64, num_layers=4, seed=999)

    # Very strict thresholds to trigger warning/critical
    auditor = KnowledgeDriftAuditor(max_allowed_kl=0.0001, max_allowed_ppl_ratio=1.0001)
    report = auditor.audit(model_pre, model_post, reference_corpus=["Testing knowledge drift detection."])

    assert report.verdict in ("WARNING", "CRITICAL")
    assert len(report.violations) > 0


def test_entropy_collapse_detection():
    model = UnifiedModelWrapper.create_toy_model(hidden_dim=64, num_layers=4, seed=42)
    auditor = KnowledgeDriftAuditor()

    collapse, entropy = auditor.check_entropy_collapse(model, "Once upon a time")
    assert isinstance(collapse, bool)
    assert entropy > 0.0
