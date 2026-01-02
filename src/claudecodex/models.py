"""
Pydantic models for Claude API and Bedrock request/response handling.

This module defines all the data models used for:
- Claude API request/response structures
- AWS Bedrock Converse API compatibility
- Token counting and usage tracking
- Content block types (text, image, tool use, tool result)
"""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union, Literal


class ContentBlockText(BaseModel):
    """Text content block for Claude API messages."""

    type: Literal["text"]
    text: str


class ContentBlockImage(BaseModel):
    """Image content block for Claude API messages (not yet supported by Bedrock Converse)."""

    type: Literal["image"]
    source: Dict[str, Any]


class ContentBlockToolUse(BaseModel):
    """Tool use content block for Claude API messages."""

    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]


class ContentBlockToolResult(BaseModel):
    """Tool result content block for Claude API messages."""

    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], List[Any], Any]


class SystemContent(BaseModel):
    """System message content block."""

    type: Literal["text"]
    text: str


class Tool(BaseModel):
    """Tool definition for Claude API."""

    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]


class Message(BaseModel):
    """Individual message in a Claude API conversation."""

    role: Literal["user", "assistant", "system"]
    content: Union[
        str,
        List[
            Union[
                ContentBlockText,
                ContentBlockImage,
                ContentBlockToolUse,
                ContentBlockToolResult,
            ]
        ],
    ]


class MessagesRequest(BaseModel):
    """Complete Claude API messages request structure."""

    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, str]] = None
    request_id: Optional[str] = None
    user_id: Optional[str] = None


class Usage(BaseModel):
    """Token usage information for Claude API responses."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class MessagesResponse(BaseModel):
    """Complete Claude API messages response structure."""

    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: List[Union[ContentBlockText, ContentBlockToolUse]]
    type: Literal["message"] = "message"
    stop_reason: Optional[
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]
    ] = None
    stop_sequence: Optional[str] = None
    usage: Usage


class TokenCountRequest(BaseModel):
    """Request structure for token counting endpoint."""

    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, str]] = None
    request_id: Optional[str] = None
    user_id: Optional[str] = None


class TokenCountResponse(BaseModel):
    """Response structure for token counting endpoint."""

    input_tokens: int
