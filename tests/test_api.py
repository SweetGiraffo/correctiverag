"""
Unit and integration tests for FastAPI backend endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_api_status(client):
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "loaded_samples_count" in data
    assert "config" in data


def test_api_config_get_and_update(client):
    # Get current config
    get_res = client.get("/api/config")
    assert get_res.status_code == 200
    cfg = get_res.json()
    assert "retrieval_algorithm" in cfg

    # Update config
    post_res = client.post("/api/config", json={"top_k_passages": 4, "retrieval_algorithm": "hybrid"})
    assert post_res.status_code == 200
    updated = post_res.json()
    assert updated["top_k_passages"] == 4
    assert updated["retrieval_algorithm"] == "hybrid"

    # Restore default
    client.post("/api/config", json={"top_k_passages": 3, "retrieval_algorithm": "ppr"})


def test_api_samples(client):
    response = client.get("/api/samples?limit=5")
    assert response.status_code == 200
    samples = response.json()
    assert len(samples) > 0
    s0 = samples[0]
    assert "id" in s0
    assert "question" in s0
    assert "answer" in s0


def test_api_query_endpoint(client):
    # Get a sample query to run
    samples = client.get("/api/samples?limit=1").json()
    sample = samples[0]

    payload = {
        "query": sample["question"],
        "sample_id": sample["id"]
    }

    response = client.post("/api/query", json=payload)
    assert response.status_code == 200
    res_data = response.json()

    assert res_data["query"] == sample["question"]
    assert res_data["final_answer"] != ""
    assert len(res_data["execution_trace"]) >= 4
    assert len(res_data["retrieved_passages"]) > 0
