# tests/test_api.py
from fastapi.testclient import TestClient
from app.main import app
from unittest.mock import patch

client = TestClient(app)


@patch("app.model.ChatbotModel.generate_response", return_value="Mock response")
def test_chat_endpoint(mock_generate):
    response = client.post("/chat", json={"text": "Test query"})
    assert response.status_code == 200
    assert response.json() == {"response": "Mock response"}
