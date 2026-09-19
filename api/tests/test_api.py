"""
Tests for the toxic comment detector API.

These run against the actual trained model (not mocked), so make sure
MODEL_SOURCE points to a valid model directory/repo before running:

    pytest tests/ -v
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))
from main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_toxic_comment(client):
    response = client.post(
        "/predict", json={"text": "You are a worthless idiot and everyone hates you"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["is_toxic"] is True
    assert len(data["scores"]) == 6
    # toxic label specifically should be flagged for this input
    toxic_score = next(s for s in data["scores"] if s["label"] == "toxic")
    assert toxic_score["flagged"] is True


def test_predict_benign_comment(client):
    response = client.post(
        "/predict", json={"text": "Thank you for the helpful explanation, this really cleared things up!"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["is_toxic"] is False


def test_predict_response_shape(client):
    response = client.post("/predict", json={"text": "This is a normal sentence."})
    data = response.json()
    assert "text" in data
    assert "is_toxic" in data
    assert "scores" in data
    for score in data["scores"]:
        assert "label" in score
        assert "probability" in score
        assert 0.0 <= score["probability"] <= 1.0
        assert "flagged" in score


def test_predict_empty_text_rejected(client):
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422  # Pydantic validation error


def test_predict_missing_field_rejected(client):
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_text_too_long_rejected(client):
    response = client.post("/predict", json={"text": "a" * 6000})
    assert response.status_code == 422
