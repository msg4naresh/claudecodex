"""
Complete FastAPI server for Claude Codex.

Single hackable file containing all server functionality:
- Backend routing and auto-detection (Bedrock vs OpenAI-compatible)
- Claude API compatible endpoints (/v1/messages, /v1/messages/count_tokens)
- FastAPI application with CORS middleware
- Request/response logging and monitoring
- Health checks and status endpoints

This consolidated approach makes the entire server easy to understand and modify.
"""

import os
import time
import logging
from typing import Literal
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from claudecodex.models import (
    MessagesRequest,
    MessagesResponse,
    TokenCountRequest,
    TokenCountResponse,
)
from claudecodex.bedrock import (
    call_bedrock_converse,
    count_request_tokens,
    get_model_id,
)
from claudecodex.openai_compatible import (
    call_openai_compatible_chat,
    count_openai_tokens,
    get_openai_compatible_model,
    get_openai_compatible_base_url,
)
from claudecodex.logging_config import setup_logging, log_request_response


# === BACKEND ROUTING ===

logger = logging.getLogger(__name__)
BackendType = Literal["bedrock", "openai_compatible"]


def get_backend_type() -> BackendType:
    """Get configured backend type from environment variables."""
    # Check for explicit backend selection first
    explicit_backend = os.environ.get("LLM_BACKEND")
    if explicit_backend:
        backend = explicit_backend.lower()
        if backend not in ["bedrock", "openai_compatible"]:
            raise ValueError(
                f"Unsupported LLM backend: {backend}. Must be 'bedrock' or 'openai_compatible'"
            )
        return backend

    # Default to openai_compatible backend
    return "openai_compatible"


def call_llm_service(request: MessagesRequest) -> MessagesResponse:
    """Route request to appropriate backend service."""
    backend = get_backend_type()

    logger.debug(f"Routing request to {backend} backend")

    if backend == "bedrock":
        return call_bedrock_converse(request)
    elif backend == "openai_compatible":
        return call_openai_compatible_chat(request)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def count_llm_tokens(request: TokenCountRequest) -> TokenCountResponse:
    """Route token counting to appropriate backend."""
    backend = get_backend_type()

    logger.debug(f"Routing token count request to {backend} backend")

    if backend == "bedrock":
        return count_request_tokens(request)
    elif backend == "openai_compatible":
        return count_openai_tokens(request)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def get_backend_info() -> dict:
    """Get current backend configuration info."""
    backend = get_backend_type()

    if backend == "bedrock":
        return {
            "backend": "bedrock",
            "model": get_model_id(),
            "region": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            "profile": os.environ.get("AWS_PROFILE", "saml"),
        }
    elif backend == "openai_compatible":
        return {
            "backend": "openai_compatible",
            "model": get_openai_compatible_model(),
            "base_url": get_openai_compatible_base_url(),
            "api_key_configured": bool(os.environ.get("OPENAICOMPATIBLE_API_KEY")),
        }
    else:
        return {"backend": "unknown", "error": f"Unsupported backend: {backend}"}


def validate_backend_config() -> bool:
    """Validate current backend configuration is complete."""
    try:
        backend = get_backend_type()

        if backend == "openai_compatible":
            api_key = os.environ.get("OPENAICOMPATIBLE_API_KEY")
            return bool(api_key)

        if backend == "bedrock":
            # Basic env vars are optional due to AWS credential chain
            return True

        return False

    except Exception as e:
        logger.error(f"Backend config validation failed: {str(e)}")
        return False


# === FASTAPI APPLICATION ===

# Initialize logging
logger, request_logger = setup_logging()

# Create FastAPI application
app = FastAPI(
    title="Claude Multi-Backend Proxy Server",
    description="Claude API compatible server supporting multiple LLM backends (Bedrock, OpenAI)",
    version="1.0.0",
)

# Add CORS middleware
# Note: allow_origins=["*"] is permissive - restrict in production if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === API ENDPOINTS ===


@app.post("/v1/messages")
async def create_message(request: MessagesRequest):
    """Handle /v1/messages endpoint - main Claude API compatibility."""
    start_time = time.time()

    try:
        result = call_llm_service(request)
        duration = time.time() - start_time

        # Log successful request/response
        backend_info = get_backend_info()
        log_request_response(
            request_logger=request_logger,
            main_logger=logger,
            endpoint="/v1/messages",
            request_data=request.model_dump(),
            response_data=result.model_dump(),
            duration=duration,
            status_code=200,
            backend_info=backend_info,
        )

        return result

    except Exception as e:
        duration = time.time() - start_time
        error_msg = str(e)

        # Log failed request
        backend_info = get_backend_info()
        log_request_response(
            request_logger=request_logger,
            main_logger=logger,
            endpoint="/v1/messages",
            request_data=request.model_dump(),
            response_data={},
            duration=duration,
            status_code=500,
            error=error_msg,
            backend_info=backend_info,
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
        backend_info = get_backend_info()
        log_request_response(
            request_logger=request_logger,
            main_logger=logger,
            endpoint="/v1/messages/count_tokens",
            request_data=request.model_dump(),
            response_data={"input_tokens": result.input_tokens},
            duration=duration,
            status_code=200,
            backend_info=backend_info,
        )

        return result

    except Exception as e:
        duration = time.time() - start_time
        error_msg = str(e)

        # Log failed token count
        backend_info = get_backend_info()
        log_request_response(
            request_logger=request_logger,
            main_logger=logger,
            endpoint="/v1/messages/count_tokens",
            request_data=request.model_dump(),
            response_data={},
            duration=duration,
            status_code=500,
            error=error_msg,
            backend_info=backend_info,
        )

        raise


@app.get("/")
async def root():
    """Root endpoint with server info."""
    backend_info = get_backend_info()
    return {
        "message": "Claude Code-Compatible Multi-Backend Server",
        "backend": backend_info["backend"],
        "model": backend_info["model"],
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    backend_info = get_backend_info()
    return {
        "status": "healthy",
        "backend": backend_info["backend"],
        "model": backend_info["model"],
    }
