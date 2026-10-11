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


def test_multihop_endpoint():
    res = client.post("/api/multihop", json={
        "anchor_prompt": "Lionel Messi plays for",
        "anchor_subject": "Lionel Messi",
        "anchor_relation": "plays for",
        "anchor_target": "Inter Miami",
        "downstream_queries": [
            {
                "prompt": "The home stadium of the team Lionel Messi plays for is",
                "target": "Chase Stadium",
                "expected_previous": "Parc des Princes",
                "hop_depth": 2,
            }
        ],
        "method": "grace",
    })
    assert res.status_code == 200
    data = res.json()
    assert "compositionality_score" in data
    assert "logical_soundness" in data
    assert len(data["hop_results"]) == 1


def test_audit_endpoint():
    res = client.post("/api/audit", json={
        "reference_texts": ["The sky is blue."],
        "max_allowed_kl": 2.0,
        "max_allowed_ppl_ratio": 2.0,
    })
    assert res.status_code == 200
    data = res.json()
    assert "verdict" in data
    assert "pre_edit_perplexity" in data
    assert "post_edit_perplexity" in data
    assert "mean_kl_divergence" in data


def test_patches_and_rollback_endpoint():
    res = client.get("/api/patches")
    assert res.status_code == 200
    data = res.json()
    assert "total_patches" in data
    assert "patches" in data

    res_rollback = client.post("/api/rollback", json={"rollback_all": False})
    assert res_rollback.status_code == 200
    data_rollback = res_rollback.json()
    assert "status" in data_rollback
