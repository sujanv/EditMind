"""Unit tests for Multi-Concept Orthogonal Subspace Decoupling Engine."""

import pytest
import torch
from editmind.models import UnifiedModelWrapper
from editmind.safety.multiconcept import (
    ConceptDefinition,
    MultiConceptUnlearnReport,
    JointSubspaceEraser,
)


@pytest.fixture
def toy_wrapper():
    torch.manual_seed(42)
    return UnifiedModelWrapper.create_toy_model(hidden_dim=64, num_layers=4)


def test_joint_subspace_erasure(toy_wrapper):
    eraser = JointSubspaceEraser(toy_wrapper)

    c1 = ConceptDefinition(
        name="credentials",
        forget_prompts=[
            "The root password is",
            "The classified private key is",
        ],
    )
    c2 = ConceptDefinition(
        name="pii_ssn",
        forget_prompts=[
            "The social security number of John is",
            "Confidential taxpayer ID number was",
        ],
    )

    retain = [
        "The capital of France is Paris",
        "Shakespeare was an English playwright",
        "Water boils at 100 degrees Celsius",
    ]

    report = eraser.erase_multiple_concepts(
        concepts=[c1, c2],
        retain_prompts=retain,
        target_layers=[2],
        max_subspace_rank=2,
    )

    assert isinstance(report, MultiConceptUnlearnReport)
    assert set(report.concept_names) == {"credentials", "pii_ssn"}
    assert report.scrubbed_layers == [2]
    assert report.subspace_rank >= 1
    assert 0.0 <= report.subspace_orthogonality_divergence <= 1.0
    assert report.utility_retention_score > 0.0
    assert report.joint_unlearning_efficiency > 0.0
    assert report.post_leakage_scores["credentials"] < report.pre_leakage_scores["credentials"]
    assert report.post_leakage_scores["pii_ssn"] < report.pre_leakage_scores["pii_ssn"]

    d = report.to_dict()
    assert "subspace_orthogonality_divergence" in d
    assert "joint_unlearning_efficiency" in d
