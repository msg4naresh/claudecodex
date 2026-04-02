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
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

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


def get_provider() -> LLMProvider:
    """Get provider instance based on configuration."""
    provider_type = get_provider_type()

    if provider_type == "bedrock":
        return BedrockProvider()
    elif provider_type == "openai_compatible":
        return OpenAICompatibleProvider()
    elif provider_type == "copilot":
        return CopilotProvider()
    else:
        raise ValueError(f"Unsupported provider: {provider_type}")


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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log HTTP requests and responses."""

    # Capture request details for POST requests
    if request.method == "POST":
        body = await request.body()
        if body:
            try:
                json.loads(body.decode())
            except json.JSONDecodeError:
                {"raw": body.decode()[:200]}

    # Process request
    response = await call_next(request)

    # The duration is calculated and used in the endpoint.

    return response


# === API ENDPOINTS ===


@app.post("/v1/messages")
async def create_message(request: MessagesRequest):
    """Handle /v1/messages endpoint - main Claude API compatibility."""
    start_time = time.time()

    try:
        result = call_llm_service(request)
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

        return result

    except Exception as e:
        duration = time.time() - start_time
        error_msg = str(e)

        # Log failed request
        provider_info = get_provider_info()
        log_request_response(
            request_logger=request_logger,
            main_logger=logger,
            endpoint="/v1/messages",
            request_data=request.model_dump(),
            response_data={},
            duration=duration,
            status_code=500,
            error=error_msg,
            provider_info=provider_info
        )

        raise


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: TokenCountRequest):
    """Handle /v1/messages/count_tokens endpoint."""
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

        # Log failed token count
        provider_info = get_provider_info()
        log_request_response(
            request_logger=request_logger,
            main_logger=logger,
            endpoint="/v1/messages/count_tokens",
            request_data=request.model_dump(),
            response_data={},
            duration=duration,
            status_code=500,
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


@app.get("/health")
async def health():
    """Health check endpoint."""
    provider_info = get_provider_info()
    return {
        "status": "healthy",
        "provider": provider_info["provider"],
        "model": provider_info["model"]
    }
