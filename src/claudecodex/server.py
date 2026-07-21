"""
Complete FastAPI server for Claude Codex.

Single hackable file containing all server functionality:
- Provider routing and auto-detection (Bedrock vs OpenAI-compatible)
- Claude API compatible endpoints (/v1/messages, /v1/messages/count_tokens)
- FastAPI application with CORS middleware
- Request/response logging and monitoring
- Health checks and status endpoints

This consolidated approach makes the entire server easy to understand and modify.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from claudecodex.models import (
    MessagesRequest,
    MessagesResponse,
    TokenCountRequest,
    TokenCountResponse
)
from claudecodex.provider import LLMProvider
from claudecodex.bedrock import BedrockProvider, get_model_id
from claudecodex.openai_compatible import (
    OpenAICompatibleProvider,
    get_openai_compatible_model,
    get_openai_compatible_base_url
)
from claudecodex.copilot_provider import CopilotProvider, get_copilot_model
from claudecodex.copilot import COPILOT_BASE_URL, COPILOT_TOKEN_ENDPOINT
from claudecodex.logging_config import setup_logging, log_request_response


# === PROVIDER ROUTING ===

logger = logging.getLogger(__name__)
ProviderType = Literal["bedrock", "openai_compatible", "copilot"]


def get_provider_type() -> ProviderType:
    """Get configured provider type from environment variables."""
    # Check for explicit provider selection first
    explicit_provider = os.environ.get("LLM_PROVIDER")
    if explicit_provider:
        provider = explicit_provider.lower()
        if provider not in ["bedrock", "openai_compatible", "copilot"]:
            raise ValueError(
                f"Unsupported LLM provider: {provider}. Must be 'bedrock', 'openai_compatible', or 'copilot'"
            )
        return provider

    # Default to Copilot when not explicitly set
    return "copilot"


_provider_instances: dict = {}


def get_provider() -> LLMProvider:
    """Get provider instance based on configuration (cached per provider type)."""
    provider_type = get_provider_type()

    if provider_type not in _provider_instances:
        if provider_type == "bedrock":
            _provider_instances[provider_type] = BedrockProvider()
        elif provider_type == "openai_compatible":
            _provider_instances[provider_type] = OpenAICompatibleProvider()
        elif provider_type == "copilot":
            _provider_instances[provider_type] = CopilotProvider()
        else:
            raise ValueError(f"Unsupported provider: {provider_type}")

    return _provider_instances[provider_type]


def call_llm_service(request: MessagesRequest) -> MessagesResponse:
    """Route request to appropriate provider service."""
    provider_type = get_provider_type()
    provider = get_provider()
    logger.debug(f"Routing request to {provider_type} provider")
    return provider.completion(request)


def count_llm_tokens(request: TokenCountRequest) -> TokenCountResponse:
    """Route token counting to appropriate provider."""
    provider_type = get_provider_type()
    provider = get_provider()
    logger.debug(f"Routing token count request to {provider_type} provider")
    return provider.count_tokens(request)


def get_provider_info() -> dict:
    """Get current provider configuration info."""
    provider = get_provider_type()

    if provider == "bedrock":
        return {
            "provider": "bedrock",
            "model": get_model_id(),
            "region": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            "profile": os.environ.get("AWS_PROFILE", "saml")
        }
    elif provider == "openai_compatible":
        return {
            "provider": "openai_compatible",
            "model": get_openai_compatible_model(),
            "base_url": get_openai_compatible_base_url(),
            "api_key_configured": bool(os.environ.get("OPENAICOMPATIBLE_API_KEY"))
        }
    elif provider == "copilot":
        token_file = os.environ.get("COPILOT_TOKEN_FILE")
        if token_file:
            token_path = Path(token_file).expanduser()
        else:
            token_path = Path.home() / ".copilot_token"
        return {
            "provider": "copilot",
            "model": get_copilot_model(),
            "base_url": COPILOT_BASE_URL,
            "token_cached": token_path.exists(),
            "token_endpoint": COPILOT_TOKEN_ENDPOINT,
        }
    else:
        return {"provider": "unknown", "error": f"Unsupported provider: {provider}"}


def validate_provider_config() -> bool:
    """Validate current provider configuration is complete."""
    try:
        provider = get_provider_type()

        if provider == "openai_compatible":
            api_key = os.environ.get("OPENAICOMPATIBLE_API_KEY")
            return bool(api_key)

        if provider == "copilot":
            # Copilot can always fall back to device flow; treat as valid.
            return True

        if provider == "bedrock":
            # Basic env vars are optional due to AWS credential chain
            return True

        return False

    except Exception as e:
        logger.error(f"Provider config validation failed: {str(e)}")
        return False


# === FASTAPI APPLICATION ===

# Initialize logging
logger, request_logger = setup_logging()

# Create FastAPI application
app = FastAPI(
    title="Claude Multi-Provider Proxy Server",
    description="Claude API compatible server supporting multiple LLM providers (Bedrock, OpenAI)",
    version="1.0.0"
)

# CORS: same-origin only. The proxy is consumed by non-browser clients
# (Claude Code) and the dashboard is served from this origin; a wildcard
# here would let arbitrary websites read dashboard data via the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8082",
        "http://127.0.0.1:8082",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# HTTP status -> Anthropic error type (drives Claude Code's retry behavior)
ANTHROPIC_ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    429: "rate_limit_error",
    500: "api_error",
    529: "overloaded_error",
}


@app.exception_handler(HTTPException)
async def anthropic_error_handler(request: Request, exc: HTTPException):
    """Return errors in Anthropic API error format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "type": "error",
            "error": {
                "type": ANTHROPIC_ERROR_TYPES.get(exc.status_code, "api_error"),
                "message": str(exc.detail)
            }
        }
    )


# === STREAMING ===


def sse_event(event: str, data: dict) -> str:
    """Format a single server-sent event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def stream_response_events(result: MessagesResponse):
    """Emit a complete MessagesResponse as Anthropic-format SSE events."""
    yield sse_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": result.id,
                "type": "message",
                "role": "assistant",
                "model": result.model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": result.usage.input_tokens,
                    "output_tokens": 0
                }
            }
        }
    )

    for index, block in enumerate(result.content):
        if block.type == "text":
            yield sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "text", "text": ""}
                }
            )
            yield sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "text_delta", "text": block.text}
                }
            )
        elif block.type == "tool_use":
            yield sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": {}
                    }
                }
            )
            yield sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(block.input)
                    }
                }
            )
        yield sse_event(
            "content_block_stop",
            {"type": "content_block_stop", "index": index}
        )

    yield sse_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": result.stop_reason,
                "stop_sequence": result.stop_sequence
            },
            "usage": {"output_tokens": result.usage.output_tokens}
        }
    )
    yield sse_event("message_stop", {"type": "message_stop"})


# === API ENDPOINTS ===


@app.post("/v1/messages")
def create_message(request: MessagesRequest):
    """Handle /v1/messages endpoint - main Claude API compatibility.

    Sync endpoint on purpose: providers use blocking HTTP clients, so FastAPI
    runs this in its threadpool, letting concurrent requests proceed in parallel.
    """
    start_time = time.time()

    try:
        provider = get_provider()

        # True incremental streaming when the provider supports it
        if request.stream and hasattr(provider, "completion_stream"):
            event_stream = provider.completion_stream(request)

            def logged_stream():
                # Accumulate a response summary from our own events for logging
                text_parts, tool_names = [], []
                usage_out, stop_reason = 0, None
                stream_error = None
                try:
                    try:
                        for event in event_stream:
                            yield event
                            try:
                                payload = json.loads(event.split("data: ", 1)[1])
                            except (IndexError, json.JSONDecodeError):
                                continue
                            ptype = payload.get("type")
                            if (ptype == "content_block_delta"
                                    and payload["delta"].get("type") == "text_delta"):
                                text_parts.append(payload["delta"]["text"])
                            elif (ptype == "content_block_start"
                                    and payload["content_block"]["type"] == "tool_use"):
                                tool_names.append(
                                    payload["content_block"].get("name", "?"))
                            elif ptype == "message_delta":
                                usage_out = payload.get("usage", {}) \
                                    .get("output_tokens", 0)
                                stop_reason = payload["delta"].get("stop_reason")
                    except Exception as e:
                        # Headers are already sent (200); surface the failure
                        # as an Anthropic-style SSE error event.
                        stream_error = str(e)
                        logger.error(f"Stream failed mid-flight: {stream_error}")
                        yield sse_event("error", {
                            "type": "error",
                            "error": {"type": "api_error", "message": stream_error}
                        })
                finally:
                    content = []
                    if text_parts:
                        content.append(
                            {"type": "text", "text": "".join(text_parts)[:500]}
                        )
                    content.extend(
                        {"type": "tool_use", "name": name} for name in tool_names
                    )
                    log_request_response(
                        request_logger=request_logger,
                        main_logger=logger,
                        endpoint="/v1/messages",
                        request_data=request.model_dump(),
                        response_data={
                            "streamed": True,
                            "content": content,
                            "stop_reason": stop_reason,
                            "usage": {"input_tokens": 0, "output_tokens": usage_out},
                        },
                        duration=time.time() - start_time,
                        status_code=500 if stream_error else 200,
                        error=stream_error,
                        provider_info=get_provider_info()
                    )

            return StreamingResponse(
                logged_stream(), media_type="text/event-stream"
            )

        result = provider.completion(request)
        duration = time.time() - start_time

        # Log successful request/response
        provider_info = get_provider_info()
        log_request_response(
            request_logger=request_logger,
            main_logger=logger,
            endpoint="/v1/messages",
            request_data=request.model_dump(),
            response_data=result.model_dump(),
            duration=duration,
            status_code=200,
            provider_info=provider_info
        )

        if request.stream:
            # Fallback: providers without completion_stream get replay SSE
            return StreamingResponse(
                stream_response_events(result),
                media_type="text/event-stream"
            )
        return result

    except Exception as e:
        duration = time.time() - start_time
        error_msg = str(e)
        status_code = e.status_code if isinstance(e, HTTPException) else 500

        # Log failed request
        provider_info = get_provider_info()
        log_request_response(
            request_logger=request_logger,
            main_logger=logger,
            endpoint="/v1/messages",
            request_data=request.model_dump(),
            response_data={},
            duration=duration,
            status_code=status_code,
            error=error_msg,
            provider_info=provider_info
        )

        raise


@app.post("/v1/messages/count_tokens")
def count_tokens(request: TokenCountRequest):
    """Handle /v1/messages/count_tokens endpoint (sync: providers block)."""
    start_time = time.time()

    try:
        result = count_llm_tokens(request)
        duration = time.time() - start_time

        # Log successful token count
        provider_info = get_provider_info()
        log_request_response(
            request_logger=request_logger,
            main_logger=logger,
            endpoint="/v1/messages/count_tokens",
            request_data=request.model_dump(),
            response_data={"input_tokens": result.input_tokens},
            duration=duration,
            status_code=200,
            provider_info=provider_info
        )

        return result

    except Exception as e:
        duration = time.time() - start_time
        error_msg = str(e)
        status_code = e.status_code if isinstance(e, HTTPException) else 500

        # Log failed token count
        provider_info = get_provider_info()
        log_request_response(
            request_logger=request_logger,
            main_logger=logger,
            endpoint="/v1/messages/count_tokens",
            request_data=request.model_dump(),
            response_data={},
            duration=duration,
            status_code=status_code,
            error=error_msg,
            provider_info=provider_info
        )

        raise


@app.get("/")
async def root():
    """Root endpoint with server info."""
    provider_info = get_provider_info()
    return {
        "message": "Claude Code-Compatible Multi-Provider Server",
        "provider": provider_info["provider"],
        "model": provider_info["model"]
    }


@app.get("/dashboard")
async def dashboard():
    """Live monitoring dashboard (self-contained HTML)."""
    from claudecodex.dashboard import DASHBOARD_HTML
    return HTMLResponse(DASHBOARD_HTML)


@app.get("/dashboard/data")
def dashboard_data(limit: int = 200):
    """JSON feed for the dashboard: recent request summaries, newest first."""
    from claudecodex.dashboard import read_recent_requests
    return read_recent_requests(limit)


@app.get("/health")
async def health():
    """Health check endpoint."""
    provider_info = get_provider_info()
    return {
        "status": "healthy",
        "provider": provider_info["provider"],
        "model": provider_info["model"]
    }
