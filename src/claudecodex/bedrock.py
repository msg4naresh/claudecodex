"""
AWS Bedrock backend implementation for Claude Codex.

This module consolidates all Bedrock-related functionality:
- AWS Bedrock Runtime client initialization and configuration
- Message translation between Claude API and Bedrock Converse API formats
- Core service logic for making Bedrock API calls
- Token counting and response processing
"""

import os
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
    TokenCountResponse,
)


logger = logging.getLogger(__name__)


# === BEDROCK CLIENT ===


def get_bedrock_client():
    """Get configured AWS Bedrock Runtime client."""
    try:
        # Get region from environment or use default
        region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

        config = Config(
            region_name=region, retries={"max_attempts": 3, "mode": "adaptive"}
        )

        # Use specified AWS profile or default to "saml"
        profile_name = os.environ.get("AWS_PROFILE", "saml")
        session = boto3.Session(profile_name=profile_name)

        return session.client("bedrock-runtime", region_name=region, config=config)

    except (BotoCoreError, ClientError) as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to create bedrock client: {str(e)}"
        )


def get_model_id() -> str:
    """Get Bedrock model ID from environment variables with validation."""
    model_id = os.environ.get(
        "BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    )

    # Validate model ID follows AWS demo pattern
    valid_prefixes = ("anthropic.claude", "us.anthropic.claude", "cohere.command")
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

    # Also check for system messages in messages list
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
                                    "input": block.input,
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
                                    "content": tool_result_content,
                                }
                            }
                        )
                    # Note: Bedrock Converse doesn't support images yet, would need different handling

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
                    input=tool_use["input"],
                )
            )

    # Ensure we have at least one content block
    if not content_blocks:
        content_blocks.append(ContentBlockText(type="text", text=""))

    usage_info = bedrock_response.get("usage", {})

    # Map Bedrock stop reasons to Claude format
    bedrock_stop_reason = bedrock_response["output"].get("stopReason", "end_turn")
    stop_reason_mapping = {
        "end_turn": "end_turn",
        "tool_use": "tool_use",
        "max_tokens": "max_tokens",
        "stop_sequence": "stop_sequence",
        "content_filtered": "end_turn",  # Map content filtering to end_turn
    }
    stop_reason = stop_reason_mapping.get(bedrock_stop_reason, "end_turn")

    return MessagesResponse(
        id=f"msg_{uuid.uuid4().hex[:24]}",
        model=model_id,
        content=content_blocks,
        stop_reason=stop_reason,
        usage=Usage(
            input_tokens=usage_info.get("inputTokens", 0),
            output_tokens=usage_info.get("outputTokens", 0),
        ),
    )


# === BEDROCK SERVICE ===


def call_bedrock_converse(request: MessagesRequest) -> MessagesResponse:
    """Execute Claude API request via AWS Bedrock."""
    try:
        # Get Bedrock client and model ID
        bedrock_client = get_bedrock_client()
        model_id = get_model_id()

        # Convert messages and extract system
        bedrock_messages = convert_to_bedrock_messages(request)
        system_message = extract_system_message(request)

        # Build inference configuration
        inference_config = {
            "temperature": request.temperature or 0.7,
            "maxTokens": request.max_tokens,
        }

        # Add optional parameters
        if request.top_p is not None:
            inference_config["topP"] = request.top_p
        if request.top_k is not None:
            inference_config["topK"] = request.top_k

        # Prepare Bedrock Converse request
        converse_params = {
            "modelId": model_id,
            "messages": bedrock_messages,
            "inferenceConfig": inference_config,
        }

        # Add system message if present
        if system_message:
            converse_params["system"] = [{"text": system_message}]

        # Handle tool configuration
        if request.tools:
            tool_config = {"tools": []}

            for tool in request.tools:
                bedrock_tool = {
                    "toolSpec": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "inputSchema": {"json": tool.input_schema},
                    }
                }
                tool_config["tools"].append(bedrock_tool)

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

        # Make the API call
        response = bedrock_client.converse(**converse_params)

        # Convert response back to Claude format
        return create_claude_response(response, model_id)

    except ClientError as e:
        error_message = e.response.get("Error", {}).get("Message", str(e))
        logger.error(f"Bedrock error: {error_message}")
        raise HTTPException(status_code=500, detail=f"Bedrock error: {error_message}")

    except Exception as e:
        logger.exception(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


def count_request_tokens(request: TokenCountRequest) -> TokenCountResponse:
    """Count tokens for Bedrock models."""
    try:
        token_count = count_tokens_from_messages(request.messages, request.system)
        return TokenCountResponse(input_tokens=token_count)

    except Exception as e:
        logger.exception(f"Error counting tokens: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error counting tokens: {str(e)}")
