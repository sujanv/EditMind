"""Unit tests for Knowledge Boundary & OOD Neighborhood Calibrator."""

import pytest
import torch
from editmind.models import UnifiedModelWrapper
from editmind.evaluation.boundary import (
    NeighborProbe,
    BoundaryCalibrationReport,
    KnowledgeBoundaryCalibrator,
)


@pytest.fixture
def toy_wrapper():
    torch.manual_seed(42)
    return UnifiedModelWrapper.create_toy_model(hidden_dim=64, num_layers=4)


def test_boundary_calibrator_default_neighbors(toy_wrapper):
    calibrator = KnowledgeBoundaryCalibrator(toy_wrapper)
    report = calibrator.calibrate_boundary(
        prompt="The Eiffel Tower is located in",
        subject="Eiffel Tower",
        target_new="Rome",
    )

    assert isinstance(report, BoundaryCalibrationReport)
    assert report.subject == "Eiffel Tower"
    assert report.target_new == "Rome"
    assert len(report.probes) == 3
    assert 0.0 <= report.boundary_sharpness_index <= 1.0
    assert 0.0 <= report.neighborhood_leakage_ratio <= 1.0
    assert 0.0 <= report.neighbor_preservation_score <= 1.0
    assert 0.0 <= report.neighbor_leakage_score <= 1.0

    d = report.to_dict()
    assert "boundary_sharpness_index" in d
    assert "neighborhood_leakage_ratio" in d
    assert len(d["probes"]) == 3


def test_boundary_calibrator_custom_neighbors(toy_wrapper):
    calibrator = KnowledgeBoundaryCalibrator(toy_wrapper)
    custom_neighbors = [
        {"entity": "Apple", "prompt": "The founder of Apple is", "expected": "Steve Jobs"},
        {"entity": "Microsoft", "prompt": "The founder of Microsoft is", "expected": "Bill Gates"},
    ]
    report = calibrator.calibrate_boundary(
        prompt="Tesla CEO is",
        subject="Tesla",
        target_new="Tim Cook",
        neighbor_entities=custom_neighbors,
    )

    assert len(report.probes) == 2
    assert report.probes[0].neighbor_entity == "Apple"
    assert report.probes[1].neighbor_entity == "Microsoft"
