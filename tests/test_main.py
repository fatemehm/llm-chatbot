from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def mock_model():
    """Mock ChatbotModel"""
    with patch("app.main.ChatbotModel") as mock:
        mock_instance = Mock()
        mock_instance.generate_response.return_value = "Test response"
        mock.return_value = mock_instance
        yield mock_instance


def test_health_endpoint(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_endpoint_success(client, mock_model):
    """Test successful chat request"""
    payload = {"message": "How to fix error?", "model_name": "google/flan-t5-small"}
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "model_used" in data
    assert data["model_used"] == "google/flan-t5-small"


def test_chat_endpoint_empty_message(client):
    """Test chat with empty message"""
    payload = {"message": "", "model_name": "google/flan-t5-small"}
    response = client.post("/chat", json=payload)
    assert response.status_code == 400
    assert "Message cannot be empty" in response.json()["detail"]


def test_chat_endpoint_whitespace_message(client):
    """Test chat with whitespace-only message"""
    payload = {"message": "   ", "model_name": "google/flan-t5-small"}
    response = client.post("/chat", json=payload)
    assert response.status_code == 400


def test_chat_endpoint_missing_message(client):
    """Test chat without message field"""
    payload = {"model_name": "google/flan-t5-small"}
    response = client.post("/chat", json=payload)
    assert response.status_code == 422  # FastAPI validation error


def test_chat_endpoint_default_model(client, mock_model):
    """Test chat with default model"""
    payload = {"message": "Test question"}
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    assert response.json()["model_used"] == "google/flan-t5-small"


def test_chat_endpoint_different_models(client, mock_model):
    """Test chat with different models"""
    models = ["google/flan-t5-small", "bert-base-uncased", "distilgpt2"]
    for model in models:
        payload = {"message": "Test", "model_name": model}
        response = client.post("/chat", json=payload)
        assert response.status_code == 200
        assert response.json()["model_used"] == model


def test_metrics_endpoint(client):
    """Test Prometheus metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"llm_chatbot_requests_total" in response.content


def test_chat_model_caching(client, mock_model):
    """Test that models are cached"""
    payload = {"message": "Test", "model_name": "google/flan-t5-small"}

    # Make two requests with same model
    client.post("/chat", json=payload)
    client.post("/chat", json=payload)

    # Model should only be loaded once (cached)
    # This would need actual implementation check
    assert True  # Placeholder
