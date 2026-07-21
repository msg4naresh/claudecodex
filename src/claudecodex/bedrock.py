"""
AWS Bedrock provider implementation for Claude Codex.

This module consolidates all Bedrock-related functionality:
- AWS Bedrock Runtime client initialization and configuration
- Message translation between Claude API and Bedrock Converse API formats
- Core service logic for making Bedrock API calls
- Token counting and response processing
"""

import os
import base64
import uuid
import logging
from typing import List, Dict, Any, Optional, Union

import boto3
from botocore.exceptions import ClientError, BotoCoreError
from botocore.config import Config
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
    TokenCountResponse
)
from claudecodex.provider import ProviderEntry, anthropic_sse, detect_model_family


logger = logging.getLogger(__name__)


# === BEDROCK CLIENT ===


_bedrock_client = None


def get_bedrock_client():
    """Get configured AWS Bedrock Runtime client (cached after first call)."""
    global _bedrock_client
    if _bedrock_client is not None:
        return _bedrock_client
    try:
        # Get region from environment or use default
        region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

        config = Config(
            region_name=region, retries={"max_attempts": 3, "mode": "adaptive"}
        )

        # Use specified AWS profile or default to "saml"
        profile_name = os.environ.get("AWS_PROFILE", "saml")
        session = boto3.Session(profile_name=profile_name)

        _bedrock_client = session.client(
            "bedrock-runtime", region_name=region, config=config
        )
        return _bedrock_client

    except (BotoCoreError, ClientError) as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to create bedrock client: {str(e)}"
        )


# Per-family model IDs for Claude Code's /model switching. These are AWS's
# standardized cross-region inference-profile IDs (the "us." prefix), which
# are consistent across accounts once Bedrock model access is granted -
# verified live against `aws bedrock list-inference-profiles`.
_BEDROCK_FAMILY_MODELS = {
    "opus": "us.anthropic.claude-opus-4-8",
    "sonnet": "us.anthropic.claude-sonnet-4-6",
    "haiku": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
}

# Per-family env var overrides, for accounts where the standard profile
# IDs above aren't available or a different snapshot is preferred.
_BEDROCK_FAMILY_ENV = {
    "opus": "BEDROCK_MODEL_ID_OPUS",
    "sonnet": "BEDROCK_MODEL_ID_SONNET",
    "haiku": "BEDROCK_MODEL_ID_HAIKU",
}


def get_model_id(requested_model: Optional[str] = None) -> str:
    """Get Bedrock model ID from environment variables with validation.

    BEDROCK_MODEL_ID, if set, always wins (explicit operator pin) - matches
    prior behavior exactly. Otherwise, resolve the requested model's family
    (opus/sonnet/haiku) to a BEDROCK_MODEL_ID_<FAMILY> override if one is
    configured, else to the standard cross-region profile ID for that
    family, to honor Claude Code's /model switching.
    """
    if "BEDROCK_MODEL_ID" in os.environ:
        model_id = os.environ["BEDROCK_MODEL_ID"]
    else:
        family = detect_model_family(requested_model)
        family_env = _BEDROCK_FAMILY_ENV.get(family)
        model_id = (
            (os.environ.get(family_env) if family_env else None)
            or _BEDROCK_FAMILY_MODELS.get(family)
            or _BEDROCK_FAMILY_MODELS["haiku"]
        )

    # Validate model ID follows AWS demo pattern
    valid_prefixes = ("anthropic.claude", "us.anthropic.claude")
    if not any(model_id.startswith(prefix) for prefix in valid_prefixes):
        logger.warning(f"Model ID may not be supported by Converse API: {model_id}")

    return model_id


# === BEDROCK TRANSLATOR ===


def extract_system_message(request: MessagesRequest) -> Optional[str]:
    """Extract system message from request."""
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

    # Also check for system messages in a messages list
    for msg in request.messages:
        if msg.role == "system" and isinstance(msg.content, str):
            return msg.content

    return None


def convert_to_bedrock_messages(request: MessagesRequest) -> List[Dict[str, Any]]:
    """Convert Claude messages to Bedrock format."""
    bedrock_messages = []

    for msg in request.messages:
        if msg.role == "system":
            continue  # Handle separately in system parameter

        # Handle content conversion
        if isinstance(msg.content, str):
            # Simple string content
            bedrock_messages.append(
                {"role": msg.role, "content": [{"text": msg.content}]}
            )
        else:
            # Handle content blocks
            content_parts = []
            for block in msg.content:
                if hasattr(block, "type"):
                    if block.type == "text":
                        content_parts.append({"text": block.text})
                    elif block.type == "tool_use":
                        # Convert tool use to Bedrock format
                        content_parts.append(
                            {
                                "toolUse": {
                                    "toolUseId": block.id,
                                    "name": block.name,
                                    "input": block.input
                                }
                            }
                        )
                    elif block.type == "tool_result":
                        # Convert tool result to Bedrock format - following AWS demo pattern
                        tool_result_content = []

                        # Handle different content types based on AWS demo best practices
                        if isinstance(block.content, str):
                            # Simple text content
                            tool_result_content.append({"text": block.content})
                        elif isinstance(block.content, list):
                            # Handle list of content blocks (following demo pattern)
                            for content_item in block.content:
                                if isinstance(content_item, dict):
                                    if "text" in content_item:
                                        tool_result_content.append(
                                            {"text": content_item["text"]}
                                        )
                                    else:
                                        # Structured data as JSON (AWS demo uses this pattern)
                                        tool_result_content.append(
                                            {"json": content_item}
                                        )
                                elif isinstance(content_item, str):
                                    tool_result_content.append({"text": content_item})
                                else:
                                    tool_result_content.append({"json": content_item})
                        elif isinstance(block.content, dict):
                            # Structured data as JSON (AWS demo pattern)
                            tool_result_content.append({"json": block.content})
                        else:
                            # Fallback to text representation
                            tool_result_content.append({"text": str(block.content)})

                        content_parts.append(
                            {
                                "toolResult": {
                                    "toolUseId": block.tool_use_id,
                                    "content": tool_result_content
                                }
                            }
                        )
                    elif block.type == "image":
                        # Convert base64 image to Bedrock format
                        source = block.source
                        if source.get("type") == "base64" and "data" in source:
                            image_format = source.get("media_type", "image/png").split("/")[-1]
                            content_parts.append(
                                {
                                    "image": {
                                        "format": image_format,
                                        "source": {
                                            "bytes": base64.b64decode(source["data"])
                                        }
                                    }
                                }
                            )

            if content_parts:
                bedrock_messages.append({"role": msg.role, "content": content_parts})

    return bedrock_messages


def estimate_token_count(text: str) -> int:
    """Estimate token count using character heuristics."""
    return max(1, len(text) // 4)


def count_tokens_from_messages(
    messages: List[Message], system: Optional[Union[str, List[SystemContent]]] = None
) -> int:
    """Count tokens from messages for Bedrock models."""
    total_tokens = 0

    # Count system tokens
    if system:
        if isinstance(system, str):
            total_tokens += estimate_token_count(system)
        elif isinstance(system, list):
            for block in system:
                if hasattr(block, "text"):
                    total_tokens += estimate_token_count(block.text)

    # Count message tokens
    for msg in messages:
        if isinstance(msg.content, str):
            total_tokens += estimate_token_count(msg.content)
        else:
            for block in msg.content:
                if hasattr(block, "type") and block.type == "text":
                    total_tokens += estimate_token_count(block.text)
                elif hasattr(block, "type") and block.type == "tool_result":
                    content_str = (
                        str(block.content)
                        if not isinstance(block.content, str)
                        else block.content
                    )
                    total_tokens += estimate_token_count(content_str)

    return total_tokens


BEDROCK_STOP_REASON_MAP = {
    "end_turn": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "stop_sequence": "stop_sequence",
    "content_filtered": "end_turn",  # Map content filtering to end_turn
}


def create_claude_response(
    bedrock_response: Dict[str, Any], model_id: str
) -> MessagesResponse:
    """Convert Bedrock response to Claude format."""
    # Extract response content blocks
    content_blocks = []
    bedrock_content = bedrock_response["output"]["message"]["content"]

    for block in bedrock_content:
        if "text" in block:
            # Text content block
            content_blocks.append(ContentBlockText(type="text", text=block["text"]))
        elif "toolUse" in block:
            # Tool use content block
            tool_use = block["toolUse"]
            content_blocks.append(
                ContentBlockToolUse(
                    type="tool_use",
                    id=tool_use["toolUseId"],
                    name=tool_use["name"],
                    input=tool_use["input"]
                )
            )

    # Ensure we have at least one content block
    if not content_blocks:
        content_blocks.append(ContentBlockText(type="text", text=""))

    usage_info = bedrock_response.get("usage", {})

    # Map Bedrock stop reasons to Claude format
    bedrock_stop_reason = bedrock_response.get("stopReason", "end_turn")
    stop_reason = BEDROCK_STOP_REASON_MAP.get(bedrock_stop_reason, "end_turn")

    return MessagesResponse(
        id=f"msg_{uuid.uuid4().hex[:24]}",
        model=model_id,
        content=content_blocks,
        stop_reason=stop_reason,
        usage=Usage(
            input_tokens=usage_info.get("inputTokens", 0),
            output_tokens=usage_info.get("outputTokens", 0)
        )
    )


# === BEDROCK SERVICE ===


# Maps AWS error codes to HTTP status so Claude Code's retry logic works,
# shared by both the sync and streaming call paths.
BEDROCK_ERROR_STATUS_MAP = {
    "ThrottlingException": 429,
    "TooManyRequestsException": 429,
    "ValidationException": 400,
    "AccessDeniedException": 403,
    "ResourceNotFoundException": 404,
    "ServiceUnavailableException": 529,
    "ModelTimeoutException": 529,
    "ModelNotReadyException": 529,
}


def build_converse_params(request: MessagesRequest, model_id: str) -> Dict[str, Any]:
    """Build Bedrock Converse API parameters from a Claude request. Shared
    by the sync (converse) and streaming (converse_stream) call paths."""
    bedrock_messages = convert_to_bedrock_messages(request)
    system_message = extract_system_message(request)

    inference_config = {"maxTokens": request.max_tokens}
    if request.temperature is not None:
        inference_config["temperature"] = request.temperature
    if request.top_p is not None:
        inference_config["topP"] = request.top_p
    if request.top_k is not None:
        inference_config["topK"] = request.top_k
    if request.stop_sequences:
        inference_config["stopSequences"] = request.stop_sequences

    converse_params = {
        "modelId": model_id,
        "messages": bedrock_messages,
        "inferenceConfig": inference_config
    }

    if system_message:
        converse_params["system"] = [{"text": system_message}]

    if request.tools:
        tool_config = {"tools": []}
        for tool in request.tools:
            tool_config["tools"].append({
                "toolSpec": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "inputSchema": {"json": tool.input_schema}
                }
            })

        # Handle tool choice - following AWS demo pattern
        if request.tool_choice:
            if isinstance(request.tool_choice, dict):
                if request.tool_choice.get("type") == "tool":
                    tool_config["toolChoice"] = {
                        "tool": {"name": request.tool_choice["name"]}
                    }
                elif request.tool_choice.get("type") == "auto":
                    tool_config["toolChoice"] = {"auto": {}}
                elif request.tool_choice.get("type") == "any":
                    tool_config["toolChoice"] = {"any": {}}
            elif request.tool_choice == "auto":
                tool_config["toolChoice"] = {"auto": {}}
            elif request.tool_choice == "any":
                tool_config["toolChoice"] = {"any": {}}

        converse_params["toolConfig"] = tool_config

    return converse_params


def call_bedrock_converse(request: MessagesRequest) -> MessagesResponse:
    """Execute Claude API request via AWS Bedrock."""
    try:
        bedrock_client = get_bedrock_client()
        model_id = get_model_id(request.model)
        converse_params = build_converse_params(request, model_id)

        response = bedrock_client.converse(**converse_params)

        return create_claude_response(response, model_id)

    except ClientError as e:
        error_info = e.response.get("Error", {})
        error_code = error_info.get("Code", "")
        error_message = error_info.get("Message", str(e))
        logger.error(f"Bedrock error ({error_code}): {error_message}")
        raise HTTPException(
            status_code=BEDROCK_ERROR_STATUS_MAP.get(error_code, 500),
            detail=f"Bedrock error: {error_message}"
        )

    except Exception as e:
        logger.exception(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


_BEDROCK_STREAM_ERROR_KEYS = (
    "internalServerException", "modelStreamErrorException",
    "validationException", "throttlingException",
    "serviceUnavailableException",
)


def stream_bedrock_as_anthropic(bedrock_stream, model_id: str):
    """Translate a Bedrock Converse stream into Anthropic SSE events.

    Verified live against a real account: Bedrock's stream is strictly
    sequential per content block (block N fully starts/deltas/stops before
    block N+1 begins, even for multiple parallel tool calls), so unlike the
    OpenAI-compatible path this needs no index-based accumulation - each
    event maps directly to the matching Anthropic event as it arrives.

    One quirk, also confirmed live: Bedrock sends an explicit
    contentBlockStart for tool_use blocks, but omits it entirely for plain
    text blocks - the first event for a text block is straight to
    contentBlockDelta. We synthesize the missing content_block_start on
    first delta so Anthropic clients (which require it) still get one.
    """
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    usage: Dict[str, Any] = {}
    stop_reason = "end_turn"
    started_indices = set()

    try:
        yield anthropic_sse("message_start", {
            "type": "message_start",
            "message": {
                "id": message_id, "type": "message", "role": "assistant",
                "model": model_id, "content": [],
                "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0}
            }
        })

        for event in bedrock_stream:
            if "contentBlockStart" in event:
                start = event["contentBlockStart"]
                index = start["contentBlockIndex"]
                block_start = start.get("start", {})
                if "toolUse" in block_start:
                    tool_use = block_start["toolUse"]
                    content_block = {
                        "type": "tool_use",
                        "id": tool_use["toolUseId"],
                        "name": tool_use["name"],
                        "input": {}
                    }
                else:
                    content_block = {"type": "text", "text": ""}
                started_indices.add(index)
                yield anthropic_sse("content_block_start", {
                    "type": "content_block_start", "index": index,
                    "content_block": content_block
                })

            elif "contentBlockDelta" in event:
                delta_event = event["contentBlockDelta"]
                index = delta_event["contentBlockIndex"]
                delta = delta_event.get("delta", {})
                if index not in started_indices:
                    # Bedrock doesn't send contentBlockStart for text blocks
                    started_indices.add(index)
                    yield anthropic_sse("content_block_start", {
                        "type": "content_block_start", "index": index,
                        "content_block": {"type": "text", "text": ""}
                    })
                if "text" in delta:
                    yield anthropic_sse("content_block_delta", {
                        "type": "content_block_delta", "index": index,
                        "delta": {"type": "text_delta", "text": delta["text"]}
                    })
                elif "toolUse" in delta:
                    partial = delta["toolUse"].get("input", "")
                    if partial:
                        yield anthropic_sse("content_block_delta", {
                            "type": "content_block_delta", "index": index,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": partial
                            }
                        })

            elif "contentBlockStop" in event:
                index = event["contentBlockStop"]["contentBlockIndex"]
                yield anthropic_sse(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": index}
                )

            elif "messageStop" in event:
                bedrock_stop_reason = event["messageStop"].get(
                    "stopReason", "end_turn"
                )
                stop_reason = BEDROCK_STOP_REASON_MAP.get(
                    bedrock_stop_reason, "end_turn"
                )

            elif "metadata" in event:
                usage = event["metadata"].get("usage", {})

            elif any(k in event for k in _BEDROCK_STREAM_ERROR_KEYS):
                error_key = next(k for k in event if k in _BEDROCK_STREAM_ERROR_KEYS)
                error_detail = event[error_key]
                raise RuntimeError(
                    f"Bedrock stream error ({error_key}): "
                    f"{error_detail.get('message', error_detail)}"
                )

        yield anthropic_sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": usage.get("outputTokens", 0)}
        })
        yield anthropic_sse("message_stop", {"type": "message_stop"})
    finally:
        bedrock_stream.close()


def count_request_tokens(request: TokenCountRequest) -> TokenCountResponse:
    """Count tokens for Bedrock models."""
    try:
        token_count = count_tokens_from_messages(request.messages, request.system)
        return TokenCountResponse(input_tokens=token_count)

    except Exception as e:
        logger.exception(f"Error counting tokens: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error counting tokens: {str(e)}")


# === PROVIDER CLASS ===


class BedrockProvider:
    """AWS Bedrock provider implementation."""

    def completion(self, request: MessagesRequest) -> MessagesResponse:
        """Get completion from AWS Bedrock (converts to Bedrock Converse API format)."""
        return call_bedrock_converse(request)

    def completion_stream(self, request: MessagesRequest):
        """Stream a completion from Bedrock as Anthropic SSE events, incrementally."""
        bedrock_client = get_bedrock_client()
        model_id = get_model_id(request.model)
        converse_params = build_converse_params(request, model_id)

        try:
            response = bedrock_client.converse_stream(**converse_params)
        except ClientError as e:
            error_info = e.response.get("Error", {})
            error_code = error_info.get("Code", "")
            error_message = error_info.get("Message", str(e))
            logger.error(f"Bedrock error ({error_code}): {error_message}")
            raise HTTPException(
                status_code=BEDROCK_ERROR_STATUS_MAP.get(error_code, 500),
                detail=f"Bedrock error: {error_message}"
            )

        return stream_bedrock_as_anthropic(response["stream"], model_id)

    def count_tokens(self, request: TokenCountRequest) -> TokenCountResponse:
        """Count tokens for a request."""
        return count_request_tokens(request)


def describe_bedrock() -> Dict[str, Any]:
    """Runtime info for /, /health, /dashboard."""
    return {
        "provider": "bedrock",
        "model": get_model_id(),
        "region": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        "profile": os.environ.get("AWS_PROFILE", "saml")
    }


def validate_bedrock_config() -> bool:
    """Basic env vars are optional due to AWS credential chain; real
    validity is only knowable on the first actual API call."""
    return True


PROVIDER = ProviderEntry(
    name="bedrock",
    factory=BedrockProvider,
    describe=describe_bedrock,
    validate=validate_bedrock_config,
)
