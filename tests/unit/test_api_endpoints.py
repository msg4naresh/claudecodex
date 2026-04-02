"""
Unit tests for Claude-Bedrock Proxy API endpoints.

This module contains unit tests for:
- FastAPI endpoint functionality
- Request/response validation  
- Error handling scenarios
- Mocked Bedrock API interactions

Tests use FastAPI TestClient with mocked AWS Bedrock calls to ensure
reliable testing without external dependencies.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from claudecodex.server import app


# === FIXTURES ===

@pytest.fixture(autouse=True)
def force_bedrock_provider(monkeypatch):
    """Ensure tests use the Bedrock provider to match mocks."""
    monkeypatch.setenv("LLM_PROVIDER", "bedrock")


@pytest.fixture
def mock_bedrock_client():
    """Provides a pre-configured mock Bedrock client."""
    with patch('claudecodex.bedrock.get_bedrock_client') as mock:
        mock_client = MagicMock()
        mock_client.converse.return_value = {
            'output': {
                'message': {
                    'content': [{'text': 'Hello! How can I help you today?'}],
                    'role': 'assistant'
                },
                'stopReason': 'end_turn'
            },
            'usage': {
                'inputTokens': 10,
                'outputTokens': 8
            }
        }
        mock.return_value = mock_client
        yield mock_client


# === TEST DATA ===

REQUEST_TEMPLATE = {
    "model": "claude-3-5-sonnet",
    "max_tokens": 1000,
    "messages": [{"role": "user", "content": "Hello"}],
    "system": "You are a helpful assistant",
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "stop_sequences": ["END"],
    "stream": False,
    "metadata": {"user_id": "123"}
}


# === HELPERS ===

def assert_successful_response(response, expected_text: str):
    """Assert standard successful response structure."""
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["role"] == "assistant"
    assert data["type"] == "message"
    assert len(data["content"]) == 1
    assert data["content"][0]["type"] == "text"
    assert data["content"][0]["text"] == expected_text
    assert data["stop_reason"] == "end_turn"
    assert "usage" in data
    assert data["usage"]["input_tokens"] == 10
    assert data["usage"]["output_tokens"] == 8


# === TEST CLIENT ===

client = TestClient(app)

# === TESTS ===


def test_create_message_success(mock_bedrock_client):
    """Test successful message creation with all parameters."""
    response = client.post("/v1/messages", json=REQUEST_TEMPLATE)
    assert_successful_response(response, "Hello! How can I help you today?")


def test_bedrock_error_handling(mock_bedrock_client):
    """Test error handling when AWS Bedrock API fails."""
    from botocore.exceptions import ClientError
    
    # Configure mock to raise error
    error_response = {'Error': {'Code': 'ValidationException', 'Message': 'Invalid model ID'}}
    mock_bedrock_client.converse.side_effect = ClientError(error_response, 'Converse')
    
    request_data = {"model": "invalid-model", "max_tokens": 1000, "messages": [{"role": "user", "content": "Hello"}]}
    response = client.post("/v1/messages", json=request_data)
    
    assert response.status_code == 500
    assert "Bedrock error" in response.json()["detail"]


def test_invalid_request_format():
    """Test validation of invalid request formats."""
    request_data = {"model": "claude-3-5-sonnet", "messages": [{"role": "user", "content": "Hello"}]}
    response = client.post("/v1/messages", json=request_data)
    assert response.status_code == 422


def test_token_counting_endpoint():
    """Test token counting endpoint functionality."""
    request_data = {
        "model": "claude-3-5-sonnet",
        "messages": [{"role": "user", "content": "Hello world"}],
        "system": "You are helpful"
    }
    response = client.post("/v1/messages/count_tokens", json=request_data)
    
    assert response.status_code == 200
    data = response.json()
    assert "input_tokens" in data
    assert data["input_tokens"] > 0


def test_health_endpoint():
    """Test health check endpoint for monitoring."""
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "model" in data


def test_root_endpoint():
    """Test root endpoint for server information."""
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "model" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
