"""Unit tests for Wave 8 Server Endpoints (Delta Compression and Contrastive Steering)."""

from fastapi.testclient import TestClient
from editmind.server.app import app

client = TestClient(app)


def test_delta_compression_endpoint():
    res = client.post("/api/delta/compress", json={
        "method": "int8",
        "sparsity": 0.8,
        "rows": 32,
        "cols": 64,
    })
    assert res.status_code == 200
    data = res.json()
    assert data["method"] == "int8"
    assert data["compression_ratio"] > 3.0
    assert data["fidelity_cosine_sim"] > 0.99
    assert "compressed_bytes" in data


def test_delta_compression_sparse_endpoint():
    res = client.post("/api/delta/compress", json={
        "method": "sparse_topk",
        "sparsity": 0.9,
        "rows": 20,
        "cols": 20,
    })
    assert res.status_code == 200
    data = res.json()
    assert data["method"] == "sparse_topk"
    assert "relative_l2_error" in data


def test_contrastive_steer_endpoint():
    res = client.post("/api/steer/contrastive", json={
        "positive_prompt": "The capital of France is Rome",
        "negative_prompt": "The capital of France is Paris",
        "probe_prompt": "The capital of France is",
        "desired_token_id": 2,
        "undesired_token_id": 1,
        "layer_idx": 2,
        "coefficient": 1.2,
    })
    assert res.status_code == 200
    data = res.json()
    assert data["target_layer"] == 2
    assert data["coefficient"] == 1.2
    assert "concept_alignment_score" in data
    assert "suppression_ratio" in data
    assert "neutral_drift_kl" in data
