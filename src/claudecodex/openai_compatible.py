"""
OpenAI-compatible backend implementation for Claude Codex.

This module consolidates all OpenAI-compatible functionality for providers implementing
the OpenAI Chat Completions API standard, including:
- OpenAI (api.openai.com)
- Google Gemini (generativelanguage.googleapis.com)
- Azure OpenAI
- Local models (Ollama, LM Studio, etc.)
- Other OpenAI-compatible providers

Features:
- HTTP client initialization and configuration
- Message translation between Claude API and OpenAI Chat Completions format
- Core service logic for making OpenAI-compatible API calls
- Token counting and response processing
"""

import os
import json
import uuid
import logging
from typing import List, Dict, Any, Optional, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from requests.exceptions import RequestException
from fastapi import HTTPException

from claudecodex.models import (
    MessagesRequest,
    MessagesResponse,
    Message,
    SystemContent,
    ContentBlockText,
    ContentBlockToolUse,
    Usage,
    TokenCountRequest,
    TokenCountResponse,
)


logger = logging.getLogger(__name__)


# === OPENAI COMPATIBLE CLIENT ===


def get_openai_compatible_client():
    """Get configured HTTP session for OpenAI-compatible providers."""
    try:
        api_key = os.environ.get("OPENAICOMPATIBLE_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="OPENAICOMPATIBLE_API_KEY environment variable is required",
            )

        base_url = os.environ.get(
            "OPENAICOMPATIBLE_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai",
        )

        # Create session with retry strategy
        session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Set default headers
        session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "claude-bedrock-proxy/1.0.0",
            }
        )

        # Store base URL for later use
        session.base_url = base_url

        return session

    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create OpenAI-compatible client: {str(e)}",
        )


def get_openai_compatible_model() -> str:
    """Get model name from environment variables."""
    return os.environ.get("OPENAI_MODEL", "gemini-2.0-flash")


def get_openai_compatible_base_url() -> str:
    """Get base URL from environment variables."""
    return os.environ.get(
        "OPENAICOMPATIBLE_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta/openai",
    )


# === OPENAI COMPATIBLE TRANSLATOR ===


def extract_system_message_openai(request: MessagesRequest) -> Optional[str]:
    """Extract system message for OpenAI format."""
    # First check system field
    if request.system:
        if isinstance(request.system, str):
            return request.system
        elif isinstance(request.system, list):
            system_text = ""
            for block in request.system:
                if hasattr(block, "text"):
                    system_text += block.text + "\n"
            return system_text.strip()

    # Also check for system messages in messages list
    for msg in request.messages:
        if msg.role == "system" and isinstance(msg.content, str):
            return msg.content

    return None


def convert_to_openai_messages(request: MessagesRequest) -> List[Dict[str, Any]]:
    """Convert Claude messages to OpenAI format."""
    openai_messages = []

    # Add system message first if present
    system_message = extract_system_message_openai(request)
    if system_message:
        openai_messages.append({"role": "system", "content": system_message})

    for msg in request.messages:
        if msg.role == "system":
            continue  # Already handled above

        # Handle content conversion
        if isinstance(msg.content, str):
            # Simple string content
            openai_messages.append({"role": msg.role, "content": msg.content})
        else:
            # Handle content blocks
            if (
                len(msg.content) == 1
                and hasattr(msg.content[0], "type")
                and msg.content[0].type == "text"
            ):
                # Single text block - simplify to string
                openai_messages.append(
                    {"role": msg.role, "content": msg.content[0].text}
                )
            else:
                # Multiple blocks or special content types
                content_parts = []
                tool_calls = []

                for block in msg.content:
                    if hasattr(block, "type"):
                        if block.type == "text":
                            content_parts.append(block.text)
                        elif block.type == "tool_use":
                            # Convert tool use to OpenAI format
                            tool_calls.append(
                                {
                                    "id": block.id,
                                    "type": "function",
                                    "function": {
                                        "name": block.name,
                                        "arguments": json.dumps(block.input)
                                        if isinstance(block.input, dict)
                                        else block.input,
                                    },
                                }
                            )
                        elif block.type == "tool_result":
                            # Tool results become separate user messages in OpenAI
                            if isinstance(block.content, str):
                                content_text = block.content
                            else:
                                content_text = str(block.content)

                            openai_messages.append(
                                {
                                    "role": "tool",
                                    "content": content_text,
                                    "tool_call_id": block.tool_use_id,
                                }
                            )

                # Create message with text content and/or tool calls
                message = {"role": msg.role}

                if content_parts:
                    message["content"] = " ".join(content_parts)
                elif not tool_calls:
                    message["content"] = ""

                if tool_calls:
                    message["tool_calls"] = tool_calls

                if "content" in message or "tool_calls" in message:
                    openai_messages.append(message)

    return openai_messages


def convert_tools_to_openai(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert Claude tools to OpenAI format."""
    openai_tools = []

    for tool in tools:
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema,
            },
        }
        openai_tools.append(openai_tool)

    return openai_tools


def estimate_token_count_openai(text: str) -> int:
    """Estimate token count using character heuristics."""
    return max(1, len(text) // 4)


def count_tokens_from_messages_openai(
    messages: List[Message], system: Optional[Union[str, List[SystemContent]]] = None
) -> int:
    """Count tokens from messages for OpenAI models."""
    total_tokens = 0

    # Count system tokens
    if system:
        if isinstance(system, str):
            total_tokens += estimate_token_count_openai(system)
        elif isinstance(system, list):
            for block in system:
                if hasattr(block, "text"):
                    total_tokens += estimate_token_count_openai(block.text)

    # Count message tokens
    for msg in messages:
        if isinstance(msg.content, str):
            total_tokens += estimate_token_count_openai(msg.content)
        else:
            for block in msg.content:
                if hasattr(block, "type") and block.type == "text":
                    total_tokens += estimate_token_count_openai(block.text)
                elif hasattr(block, "type") and block.type == "tool_result":
                    content_str = (
                        str(block.content)
                        if not isinstance(block.content, str)
                        else block.content
                    )
                    total_tokens += estimate_token_count_openai(content_str)

    return total_tokens


def create_claude_response_from_openai(
    openai_response: Dict[str, Any], model_id: str
) -> MessagesResponse:
    """Convert OpenAI response to Claude format."""
    # Extract first choice (OpenAI can return multiple choices)
    if not openai_response.get("choices"):
        raise ValueError("No choices in OpenAI response")

    choice = openai_response["choices"][0]
    message = choice["message"]

    # Extract content blocks
    content_blocks = []

    # Handle text content
    if message.get("content"):
        content_blocks.append(ContentBlockText(type="text", text=message["content"]))

    # Handle tool calls
    if message.get("tool_calls"):
        for tool_call in message["tool_calls"]:
            if tool_call["type"] == "function":
                function = tool_call["function"]
                # Parse arguments back to dict if it's a string
                try:
                    arguments = (
                        json.loads(function["arguments"])
                        if isinstance(function["arguments"], str)
                        else function["arguments"]
                    )
                except (json.JSONDecodeError, ValueError):
                    arguments = function["arguments"]

                content_blocks.append(
                    ContentBlockToolUse(
                        type="tool_use",
                        id=tool_call["id"],
                        name=function["name"],
                        input=arguments,
                    )
                )

    # Ensure we have at least one content block
    if not content_blocks:
        content_blocks.append(ContentBlockText(type="text", text=""))

    # Map stop reason
    finish_reason = choice.get("finish_reason", "stop")
    if finish_reason == "tool_calls":
        stop_reason = "tool_use"
    elif finish_reason == "length":
        stop_reason = "max_tokens"
    elif finish_reason == "content_filter":
        stop_reason = "stop_sequence"
    else:  # 'stop' or other
        stop_reason = "end_turn"

    # Extract usage information
    usage_info = openai_response.get("usage", {})

    return MessagesResponse(
        id=f"msg_{uuid.uuid4().hex[:24]}",
        model=model_id,
        content=content_blocks,
        stop_reason=stop_reason,
        usage=Usage(
            input_tokens=usage_info.get("prompt_tokens", 0),
            output_tokens=usage_info.get("completion_tokens", 0),
        ),
    )


# === OPENAI COMPATIBLE SERVICE ===


def call_openai_compatible_chat(request: MessagesRequest) -> MessagesResponse:
    """Execute Claude API request via OpenAI-compatible provider."""
    try:
        # Get OpenAI client and model
        openai_client = get_openai_compatible_client()
        model_id = get_openai_compatible_model()

        # Convert messages to OpenAI format
        openai_messages = convert_to_openai_messages(request)

        # Build request payload
        payload = {
            "model": model_id,
            "messages": openai_messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature or 0.7,
        }

        # Add optional parameters
        if request.top_p is not None:
            payload["top_p"] = request.top_p

        # Handle tool configuration
        if request.tools:
            openai_tools = convert_tools_to_openai(request.tools)
            payload["tools"] = openai_tools

            # Handle tool choice
            if request.tool_choice:
                if request.tool_choice.get("type") == "tool":
                    payload["tool_choice"] = {
                        "type": "function",
                        "function": {"name": request.tool_choice["name"]},
                    }
                elif request.tool_choice.get("type") == "auto":
                    payload["tool_choice"] = "auto"
                elif request.tool_choice.get("type") == "any":
                    payload["tool_choice"] = "required"  # OpenAI's equivalent

        # Handle stop sequences
        if request.stop_sequences:
            payload["stop"] = request.stop_sequences

        # Log request details (similar to bedrock_service pattern)
        logger.debug(
            f"OpenAI request: model={model_id}, messages={len(openai_messages)}"
        )
        if request.tools:
            logger.debug(f"Tools: {len(request.tools)} functions")
        if request.stream:
            logger.warning("Streaming requested but not yet implemented for OpenAI")

        # Make the API call
        response = openai_client.post(
            f"{openai_client.base_url}/chat/completions", json=payload, timeout=120
        )

        # Check for HTTP errors
        response.raise_for_status()

        # Parse response
        response_data = response.json()

        # Convert response back to Claude format
        return create_claude_response_from_openai(response_data, model_id)

    except RequestException as e:
        if hasattr(e, "response") and e.response is not None:
            try:
                error_data = e.response.json()

                # Handle different error response formats
                # Gemini returns errors as a list: [{"error": {...}}]
                # OpenAI returns errors as a dict: {"error": {...}}
                if isinstance(error_data, list) and len(error_data) > 0:
                    error_obj = error_data[0].get("error", {})
                else:
                    error_obj = error_data.get("error", {})

                error_message = error_obj.get("message", str(e))
                status_code = e.response.status_code

                # Map OpenAI error codes to appropriate HTTP status codes
                if status_code == 401:
                    error_message = f"OpenAI authentication failed: {error_message}"
                elif status_code == 403:
                    error_message = f"OpenAI access forbidden: {error_message}"
                elif status_code == 429:
                    error_message = f"OpenAI rate limit exceeded: {error_message}"
                elif status_code >= 500:
                    error_message = f"OpenAI server error: {error_message}"
                else:
                    error_message = f"OpenAI API error: {error_message}"

                logger.error(f"OpenAI API error ({status_code}): {error_message}")
                raise HTTPException(status_code=status_code, detail=error_message)

            except json.JSONDecodeError:
                error_message = f"OpenAI API error: {e.response.text[:200]}"
                logger.error(error_message)
                raise HTTPException(
                    status_code=e.response.status_code, detail=error_message
                )
        else:
            error_message = f"OpenAI connection error: {str(e)}"
            logger.error(error_message)
            raise HTTPException(status_code=500, detail=error_message)

    except json.JSONDecodeError as e:
        error_message = f"Failed to parse OpenAI response: {str(e)}"
        logger.error(error_message)
        raise HTTPException(status_code=500, detail=error_message)

    except Exception as e:
        logger.exception(f"Unexpected error calling OpenAI: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


def count_openai_tokens(request: TokenCountRequest) -> TokenCountResponse:
    """Count tokens for OpenAI-compatible models."""
    try:
        token_count = count_tokens_from_messages_openai(
            request.messages, request.system
        )
        return TokenCountResponse(input_tokens=token_count)

    except Exception as e:
        logger.exception(f"Error counting tokens for OpenAI: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error counting tokens: {str(e)}")
