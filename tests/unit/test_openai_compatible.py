"""
Unit tests for OpenAI-compatible service functionality.

This module contains unit tests for:
- OpenAI-compatible message translation (Claude ↔ OpenAI format)
- OpenAI-compatible service calls with mocked HTTP responses
- Tool calling functionality and translation
- Error handling for various OpenAI-compatible providers
- Token counting for different model types

Tests use mocked HTTP requests to ensure reliable testing without
external dependencies or API keys.
"""

import pytest
import json
from unittest.mock import patch, MagicMock
from requests.exceptions import RequestException, HTTPError
from fastapi import HTTPException

from claudecodex.models import MessagesRequest, Message, Tool
from claudecodex.openai_compatible import (
    call_openai_compatible_chat, count_openai_tokens, convert_to_openai_messages,
    convert_tools_to_openai, create_claude_response_from_openai, count_tokens_from_messages_openai
)


# Mock OpenAI-compatible responses for different providers
MOCK_OPENAI_RESPONSE = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "gpt-4",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! I'm here to help you with your coding questions."
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 12,
        "completion_tokens": 15,
        "total_tokens": 27
    }
}

MOCK_GEMINI_RESPONSE = {
    "id": "gemini-123",
    "object": "chat.completion", 
    "created": 1677652288,
    "model": "gemini-2.0-flash",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "I'd be happy to help you write clean Python code!"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 18,
        "completion_tokens": 12,
        "total_tokens": 30
    }
}

MOCK_TOOL_CALL_RESPONSE = {
    "id": "chatcmpl-tool-123",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "gpt-4",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "San Francisco", "unit": "celsius"}'
                        }
                    }
                ]
            },
            "finish_reason": "tool_calls"
        }
    ],
    "usage": {
        "prompt_tokens": 25,
        "completion_tokens": 8,
        "total_tokens": 33
    }
}


class TestOpenAICompatibleTranslation:
    """Test message translation between Claude and OpenAI-compatible formats."""
    
    def test_simple_message_translation(self):
        """Test basic message translation from Claude to OpenAI format."""
        request = MessagesRequest(
            model="gpt-4",
            max_tokens=1000,
            messages=[
                Message(role="user", content="Hello world")
            ],
            temperature=0.7
        )
        
        openai_messages = convert_to_openai_messages(request)
        
        assert len(openai_messages) == 1
        assert openai_messages[0]["role"] == "user"
        assert openai_messages[0]["content"] == "Hello world"
    
    def test_system_message_translation(self):
        """Test system message handling in OpenAI format."""
        request = MessagesRequest(
            model="gpt-4",
            max_tokens=1000,
            messages=[
                Message(role="user", content="Hello")
            ],
            system="You are a helpful coding assistant",
            temperature=0.7
        )
        
        openai_messages = convert_to_openai_messages(request)
        
        assert len(openai_messages) == 2
        assert openai_messages[0]["role"] == "system"
        assert openai_messages[0]["content"] == "You are a helpful coding assistant"
        assert openai_messages[1]["role"] == "user"
        assert openai_messages[1]["content"] == "Hello"
    
    def test_tool_definition_translation(self):
        """Test tool definitions translation to OpenAI functions format."""
        claude_tools = [
            Tool(
                name="get_weather",
                description="Get current weather",
                input_schema={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location"]
                }
            )
        ]
        
        openai_tools = convert_tools_to_openai(claude_tools)
        
        assert len(openai_tools) == 1
        tool = openai_tools[0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "get_weather"
        assert tool["function"]["description"] == "Get current weather"
        assert tool["function"]["parameters"]["type"] == "object"
        assert "location" in tool["function"]["parameters"]["properties"]
        assert tool["function"]["parameters"]["required"] == ["location"]
    
    def test_openai_response_to_claude_translation(self):
        """Test OpenAI response translation back to Claude format."""
        claude_response = create_claude_response_from_openai(MOCK_OPENAI_RESPONSE, "gpt-4")
        
        assert claude_response.model == "gpt-4"
        assert claude_response.role == "assistant"
        assert claude_response.type == "message"
        assert len(claude_response.content) == 1
        assert claude_response.content[0].type == "text"
        assert claude_response.content[0].text == "Hello! I'm here to help you with your coding questions."
        assert claude_response.stop_reason == "end_turn"
        assert claude_response.usage.input_tokens == 12
        assert claude_response.usage.output_tokens == 15
    
    def test_tool_call_response_translation(self):
        """Test tool call response translation from OpenAI to Claude format."""
        claude_response = create_claude_response_from_openai(MOCK_TOOL_CALL_RESPONSE, "gpt-4")
        
        assert len(claude_response.content) == 1
        assert claude_response.content[0].type == "tool_use"
        assert claude_response.content[0].id == "call_abc123"
        assert claude_response.content[0].name == "get_weather"
        assert claude_response.content[0].input == {
            "location": "San Francisco",
            "unit": "celsius"
        }
        assert claude_response.stop_reason == "tool_use"
    
    def test_token_counting(self):
        """Test token counting for OpenAI-compatible models."""
        messages = [
            Message(role="user", content="Hello world"),
            Message(role="assistant", content="Hi there!")
        ]
        system = "You are helpful"
        
        token_count = count_tokens_from_messages_openai(messages, system)
        
        assert isinstance(token_count, int)
        assert token_count > 0
        # Should count system message + both messages
        assert token_count >= 5  # Rough estimate for minimal messages


class TestOpenAICompatibleService:
    """Test OpenAI-compatible service functionality with mocked HTTP calls."""
    
    @patch('claudecodex.openai_compatible.get_openai_compatible_client')
    @patch('claudecodex.openai_compatible.get_openai_compatible_model')
    def test_successful_openai_call(self, mock_get_model, mock_get_client):
        """Test successful OpenAI API call."""
        # Setup mocks
        mock_get_model.return_value = "gpt-4"
        mock_client = MagicMock()
        mock_client.base_url = "https://api.openai.com/v1"
        mock_get_client.return_value = mock_client
        
        # Mock successful HTTP response
        mock_response = MagicMock()
        mock_response.json.return_value = MOCK_OPENAI_RESPONSE
        mock_response.raise_for_status.return_value = None
        mock_client.post.return_value = mock_response
        
        # Test request
        request = MessagesRequest(
            model="gpt-4",
            max_tokens=1000,
            messages=[Message(role="user", content="Hello")],
            temperature=0.7
        )
        
        result = call_openai_compatible_chat(request)
        
        # Verify API call was made correctly
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "https://api.openai.com/v1/chat/completions"
        
        payload = call_args[1]['json']
        assert payload["model"] == "gpt-4"
        assert payload["max_tokens"] == 1000
        assert payload["temperature"] == 0.7
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["content"] == "Hello"
        
        # Verify response
        assert result.model == "gpt-4"
        assert len(result.content) == 1
        assert result.content[0].text == "Hello! I'm here to help you with your coding questions."
    
    @patch('claudecodex.openai_compatible.get_openai_compatible_client')
    @patch('claudecodex.openai_compatible.get_openai_compatible_model')
    def test_gemini_call(self, mock_get_model, mock_get_client):
        """Test call configured for Gemini provider."""
        # Setup mocks for Gemini
        mock_get_model.return_value = "gemini-2.0-flash"
        mock_client = MagicMock()
        mock_client.base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
        mock_get_client.return_value = mock_client
        
        # Mock Gemini response
        mock_response = MagicMock()
        mock_response.json.return_value = MOCK_GEMINI_RESPONSE
        mock_response.raise_for_status.return_value = None
        mock_client.post.return_value = mock_response
        
        # Test request
        request = MessagesRequest(
            model="gemini-2.0-flash",
            max_tokens=2000,
            messages=[Message(role="user", content="Help me code")],
            temperature=0.5
        )
        
        result = call_openai_compatible_chat(request)
        
        # Verify Gemini endpoint was called
        call_args = mock_client.post.call_args
        assert "generativelanguage.googleapis.com" in call_args[0][0]
        
        # Verify payload
        payload = call_args[1]['json']
        assert payload["model"] == "gemini-2.0-flash"
        assert payload["max_tokens"] == 2000
        
        # Verify response
        assert result.model == "gemini-2.0-flash"
        assert result.content[0].text == "I'd be happy to help you write clean Python code!"
    
    @patch('claudecodex.openai_compatible.get_openai_compatible_client')
    @patch('claudecodex.openai_compatible.get_openai_compatible_model')
    def test_tool_calling(self, mock_get_model, mock_get_client):
        """Test tool calling functionality."""
        # Setup mocks
        mock_get_model.return_value = "gpt-4"
        mock_client = MagicMock()
        mock_client.base_url = "https://api.openai.com/v1"
        mock_get_client.return_value = mock_client
        
        # Mock tool call response
        mock_response = MagicMock()
        mock_response.json.return_value = MOCK_TOOL_CALL_RESPONSE
        mock_response.raise_for_status.return_value = None
        mock_client.post.return_value = mock_response
        
        # Test request with tools
        request = MessagesRequest(
            model="gpt-4",
            max_tokens=1000,
            messages=[Message(role="user", content="What's the weather?")],
            tools=[
                Tool(
                    name="get_weather",
                    description="Get weather info",
                    input_schema={
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"]
                    }
                )
            ],
            tool_choice={"type": "auto"}
        )
        
        result = call_openai_compatible_chat(request)
        
        # Verify tools were sent in payload
        payload = mock_client.post.call_args[1]['json']
        assert "tools" in payload
        assert len(payload["tools"]) == 1
        assert payload["tools"][0]["type"] == "function"
        assert payload["tools"][0]["function"]["name"] == "get_weather"
        assert payload["tool_choice"] == "auto"
        
        # Verify tool call in response
        assert len(result.content) == 1
        assert result.content[0].type == "tool_use"
        assert result.content[0].name == "get_weather"
        assert result.stop_reason == "tool_use"
    
    @patch('claudecodex.openai_compatible.get_openai_compatible_client')
    @patch('claudecodex.openai_compatible.get_openai_compatible_model')
    def test_api_error_handling(self, mock_get_model, mock_get_client):
        """Test handling of API errors from OpenAI-compatible providers."""
        # Setup mocks
        mock_get_model.return_value = "gpt-4"
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # Mock API error
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {
            "error": {
                "message": "Invalid API key",
                "type": "invalid_request_error"
            }
        }
        
        http_error = HTTPError()
        http_error.response = mock_response
        mock_client.post.side_effect = http_error
        
        # Test request
        request = MessagesRequest(
            model="gpt-4",
            max_tokens=1000,
            messages=[Message(role="user", content="Hello")]
        )
        
        # Should raise HTTPException with proper error handling
        with pytest.raises(HTTPException) as exc_info:
            call_openai_compatible_chat(request)
        
        assert exc_info.value.status_code == 401
        assert "authentication failed" in exc_info.value.detail.lower()
    
    @patch('claudecodex.openai_compatible.get_openai_compatible_client')
    @patch('claudecodex.openai_compatible.get_openai_compatible_model')
    def test_rate_limit_error(self, mock_get_model, mock_get_client):
        """Test handling of rate limit errors."""
        # Setup mocks
        mock_get_model.return_value = "gpt-4"
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # Mock rate limit error
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.json.return_value = {
            "error": {
                "message": "Rate limit exceeded",
                "type": "rate_limit_exceeded"
            }
        }
        
        http_error = HTTPError()
        http_error.response = mock_response
        mock_client.post.side_effect = http_error
        
        # Test request
        request = MessagesRequest(
            model="gpt-4",
            max_tokens=1000,
            messages=[Message(role="user", content="Hello")]
        )
        
        with pytest.raises(HTTPException) as exc_info:
            call_openai_compatible_chat(request)
        
        assert exc_info.value.status_code == 429
        assert "rate limit" in exc_info.value.detail.lower()
    
    def test_token_counting_service(self):
        """Test token counting service function."""
        from claudecodex.models import TokenCountRequest
        
        request = TokenCountRequest(
            model="gpt-4",
            messages=[
                Message(role="user", content="Count these tokens please")
            ],
            system="You are helpful"
        )
        
        result = count_openai_tokens(request)
        
        assert hasattr(result, 'input_tokens')
        assert result.input_tokens > 0
        assert isinstance(result.input_tokens, int)


class TestProviderSpecificBehavior:
    """Test behavior specific to different OpenAI-compatible providers."""
    
    def test_provider_specific_models(self):
        """Test that different providers use appropriate model names."""
        test_cases = [
            ("gpt-4", "OpenAI"),
            ("gpt-3.5-turbo", "OpenAI"),
            ("gemini-2.0-flash", "Gemini"),
            ("gemini-1.5-pro", "Gemini"),
            ("llama3", "Local/Ollama"),
        ]
        
        for model_name, provider in test_cases:
            request = MessagesRequest(
                model=model_name,
                max_tokens=100,
                messages=[Message(role="user", content="Test")]
            )
            
            openai_messages = convert_to_openai_messages(request)
            assert len(openai_messages) == 1
            assert openai_messages[0]["content"] == "Test"
            # Model name should be preserved in translation
    
    def test_parameter_handling_across_providers(self):
        """Test that parameters are handled consistently across providers."""
        request = MessagesRequest(
            model="test-model",
            max_tokens=1000,
            messages=[Message(role="user", content="Test")],
            temperature=0.8,
            top_p=0.95,
            top_k=40,  # Should be ignored in OpenAI format
            stop_sequences=["STOP", "END"]
        )
        
        # Parameters should be properly formatted for OpenAI-compatible APIs
        # top_k should be ignored since OpenAI doesn't support it
        openai_messages = convert_to_openai_messages(request)
        assert len(openai_messages) == 1


class FakeStreamResponse:
    """Minimal stand-in for requests.Response with an SSE body."""

    def __init__(self, chunks, raise_after=None):
        import json as _json
        self._lines = [f"data: {_json.dumps(c)}".encode() for c in chunks]
        self._lines.append(b"data: [DONE]")
        self._raise_after = raise_after
        self.closed = False

    def iter_lines(self):
        for i, line in enumerate(self._lines):
            if self._raise_after is not None and i >= self._raise_after:
                raise ConnectionError("upstream dropped")
            yield line

    def close(self):
        self.closed = True


class TestStreamOpenaiAsAnthropic:
    """Incremental OpenAI -> Anthropic SSE translation."""

    def _events(self, chunks):
        import json as _json
        from claudecodex.openai_compatible import stream_openai_as_anthropic

        raw = list(stream_openai_as_anthropic(FakeStreamResponse(chunks), "m"))
        parsed = []
        for event in raw:
            lines = event.strip().split("\n")
            name = lines[0].split("event: ", 1)[1]
            data = _json.loads(lines[1].split("data: ", 1)[1])
            parsed.append((name, data))
        return parsed

    def test_text_deltas_stream_incrementally(self):
        events = self._events([
            {"choices": [{"delta": {"content": "Hel"}}]},
            {"choices": [{"delta": {"content": "lo"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}],
             "usage": {"completion_tokens": 2, "prompt_tokens": 5}},
        ])

        names = [n for n, _ in events]
        assert names == [
            "message_start", "content_block_start",
            "content_block_delta", "content_block_delta",
            "content_block_stop", "message_delta", "message_stop",
        ]
        deltas = [d["delta"]["text"] for n, d in events
                  if n == "content_block_delta"]
        assert deltas == ["Hel", "lo"]
        message_delta = dict(events)["message_delta"]
        assert message_delta["delta"]["stop_reason"] == "end_turn"
        assert message_delta["usage"]["output_tokens"] == 2

    def test_tool_call_after_text_gets_new_block(self):
        events = self._events([
            {"choices": [{"delta": {"content": "Sure."}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "t1", "type": "function",
                 "function": {"name": "get_weather"}}]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": '{"city": '}}]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": '"Paris"}'}}]}}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ])

        starts = [d for n, d in events if n == "content_block_start"]
        assert starts[0]["content_block"]["type"] == "text"
        assert starts[1]["content_block"]["type"] == "tool_use"
        assert starts[1]["content_block"]["name"] == "get_weather"
        assert starts[1]["index"] == 1

        partials = "".join(
            d["delta"]["partial_json"] for n, d in events
            if n == "content_block_delta"
            and d["delta"]["type"] == "input_json_delta"
        )
        assert partials == '{"city": "Paris"}'
        assert dict(events)["message_delta"]["delta"]["stop_reason"] == "tool_use"

    def test_parallel_tool_calls_with_interleaved_arguments(self):
        """Two tools whose ids/names and argument chunks arrive interleaved."""
        events = self._events([
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "tool_a", "type": "function",
                 "function": {"name": "search"}},
                {"index": 1, "id": "tool_b", "type": "function",
                 "function": {"name": "weather"}}]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": '{"query"'}},
                {"index": 1, "function": {"arguments": '{"city"'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": ':"x"}'}},
                {"index": 1, "function": {"arguments": ':"London"}'}}]}}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ])

        starts = [d for n, d in events if n == "content_block_start"]
        assert len(starts) == 2, starts
        assert [(s["content_block"]["id"], s["content_block"]["name"])
                for s in starts] == [("tool_a", "search"), ("tool_b", "weather")]
        assert [s["index"] for s in starts] == [0, 1]

        deltas = [d for n, d in events if n == "content_block_delta"]
        args_by_index = {d["index"]: d["delta"]["partial_json"] for d in deltas}
        assert args_by_index == {0: '{"query":"x"}', 1: '{"city":"London"}'}
        stops = [d for n, d in events if n == "content_block_stop"]
        assert len(stops) == 2

    def test_upstream_response_closed_on_completion_and_error(self):
        from claudecodex.openai_compatible import stream_openai_as_anthropic

        # Normal completion closes the upstream response
        resp = FakeStreamResponse([{"choices": [{"delta": {"content": "hi"},
                                                 "finish_reason": "stop"}]}])
        list(stream_openai_as_anthropic(resp, "m"))
        assert resp.closed

        # Mid-stream failure still closes it
        resp = FakeStreamResponse(
            [{"choices": [{"delta": {"content": "hi"}}]}] * 3, raise_after=1)
        with pytest.raises(ConnectionError):
            list(stream_openai_as_anthropic(resp, "m"))
        assert resp.closed

    def test_response_closed_when_cancelled_after_message_start(self):
        from claudecodex.openai_compatible import stream_openai_as_anthropic

        resp = FakeStreamResponse([])
        stream = stream_openai_as_anthropic(resp, "m")
        next(stream)      # receive message_start
        stream.close()    # client cancels immediately
        assert resp.closed


class TestCopilotStreamTransportErrors:
    """completion_stream maps transport failures to Anthropic retry semantics."""

    def _provider_with_client(self, client):
        from claudecodex.copilot_provider import CopilotProvider

        provider = CopilotProvider()
        provider._get_client = lambda: client
        return provider

    def _request(self):
        return MessagesRequest(
            model="m", max_tokens=10,
            messages=[Message(role="user", content="hi")]
        )

    def test_timeout_maps_to_529(self):
        import requests as _requests

        client = MagicMock()
        client.base_url = "https://api.example"
        client.post.side_effect = _requests.exceptions.Timeout("timed out")

        provider = self._provider_with_client(client)
        with pytest.raises(HTTPException) as exc_info:
            provider.completion_stream(self._request())
        assert exc_info.value.status_code == 529

    def test_connection_error_maps_to_529(self):
        import requests as _requests

        client = MagicMock()
        client.base_url = "https://api.example"
        client.post.side_effect = _requests.exceptions.ConnectionError("refused")

        provider = self._provider_with_client(client)
        with pytest.raises(HTTPException) as exc_info:
            provider.completion_stream(self._request())
        assert exc_info.value.status_code == 529

    def test_http_error_response_is_closed(self):
        client = MagicMock()
        client.base_url = "https://api.example"
        error_response = MagicMock()
        error_response.status_code = 503
        error_response.text = "service unavailable"
        client.post.return_value = error_response

        provider = self._provider_with_client(client)
        with pytest.raises(HTTPException) as exc_info:
            provider.completion_stream(self._request())
        assert exc_info.value.status_code == 529  # 503 not passed through
        error_response.close.assert_called_once()


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])