"""Unit tests for FastAPI endpoints."""

from fastapi.testclient import TestClient
from editmind.server.app import app

client = TestClient(app)


def test_health_endpoint():
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_methods_endpoint():
    res = client.get("/api/methods")
    assert res.status_code == 200
    methods = res.json()["methods"]
    assert "rome" in methods
    assert "memit" in methods
    assert "pmet" in methods
    assert "alphaedit" in methods
    assert "grace" in methods


def test_trace_endpoint():
    res = client.post("/api/trace", json={
        "prompt": "The Eiffel Tower is in",
        "subject": "Eiffel Tower",
        "target": "Paris",
    })
    assert res.status_code == 200
    data = res.json()
    assert "svg_heatmap" in data
    assert "critical_layer" in data


def test_edit_endpoint():
    res = client.post("/api/edit", json={
        "prompt": "The Eiffel Tower is in",
        "subject": "Eiffel Tower",
        "target_new": "Rome",
        "ground_truth": "Paris",
        "method": "rome",
    })
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["post_edit_target_prob"] > data["pre_edit_target_prob"]


def test_compare_endpoint():
    res = client.post("/api/compare", json={
        "prompt": "The Eiffel Tower is in",
        "subject": "Eiffel Tower",
        "target_new": "Rome",
        "ground_truth": "Paris",
        "methods": ["rome", "grace"],
    })
    assert res.status_code == 200
    data = res.json()
    assert "svg_radar" in data
    assert "rome" in data["comparison"]


def test_unlearn_endpoint():
    res = client.post("/api/unlearn", json={
        "prompt": "The secret password of root is",
        "target_to_erase": "Paris",
        "editor": "grace",
    })
    assert res.status_code == 200
    data = res.json()
    assert data["target_erased"] == "Paris"
    assert data["success"] is True


def test_continual_endpoint():
    res = client.post("/api/continual", json={
        "editor": "grace",
        "requests": [
            {"prompt": "The Eiffel Tower is in", "target_new": "Rome", "ground_truth": "Paris", "subject": "Eiffel Tower"},
            {"prompt": "Messi plays for", "target_new": "Miami", "ground_truth": "PSG", "subject": "Messi"},
        ]
    })
    assert res.status_code == 200
    data = res.json()
    assert "trajectory" in data
    assert "svg_matrix" in data


def test_conflict_endpoint():
    res = client.post("/api/check-conflict", json={
        "prompt": "The Eiffel Tower is in",
        "subject": "Eiffel Tower",
        "target_new": "Rome",
    })
    assert res.status_code == 200
    data = res.json()
    assert "has_conflict" in data
