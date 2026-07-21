"""GitHub Copilot provider wired through the OpenAI-compatible pipeline."""

from __future__ import annotations

import os
import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from fastapi import HTTPException

from claudecodex.copilot import (
    CopilotAuth,
    COPILOT_HEADERS,
    COPILOT_BASE_URL,
    COPILOT_TOKEN_ENDPOINT,
)
from claudecodex.models import (
    MessagesRequest,
    MessagesResponse,
    TokenCountRequest,
    TokenCountResponse,
)
from claudecodex.openai_compatible import (
    get_openai_compatible_client,
    call_openai_compatible_chat,
    count_tokens_from_messages_openai,
    build_openai_payload,
    post_streaming_completion,
    stream_openai_as_anthropic,
)
from claudecodex.provider import ProviderEntry, detect_model_family

logger = logging.getLogger(__name__)

# Verified against Copilot's live /models endpoint - Copilot uses dots, not
# hyphens, in Claude model IDs (e.g. "claude-sonnet-4.6").
_COPILOT_FAMILY_MODELS = {
    "opus": "claude-opus-4.8",
    "sonnet": "claude-sonnet-4.6",
    "haiku": "claude-haiku-4.5",
}


def get_copilot_model(requested_model: Optional[str] = None) -> str:
    """Resolve the Copilot model to use for a request.

    COPILOT_MODEL, if set, always wins (explicit operator pin). Otherwise,
    honor Claude Code's /model switching by mapping the requested model's
    family (opus/sonnet/haiku) to the matching Copilot model ID, falling
    back to the sonnet default when the family can't be determined.
    """
    if "COPILOT_MODEL" in os.environ:
        return os.environ["COPILOT_MODEL"]
    family = detect_model_family(requested_model)
    return _COPILOT_FAMILY_MODELS.get(family, _COPILOT_FAMILY_MODELS["sonnet"])


class CopilotProvider:
    """GitHub Copilot provider using Copilot auth + OpenAI-compatible transport."""

    def __init__(self):
        self._auth = CopilotAuth(
            oauth_token=os.environ.get("COPILOT_OAUTH_TOKEN"),
            token_file=os.environ.get("COPILOT_TOKEN_FILE"),
        )
        self._client: Optional[requests.Session] = None
        self._client_lock = threading.Lock()
        self._session_token: Optional[str] = None

    def _get_client(self) -> requests.Session:
        """Get or create session using the current Copilot session token."""
        with self._client_lock:
            session_token = self._auth.get_session_token()
            if (
                self._client is None
                or self._session_token != session_token
            ):
                self._client = get_openai_compatible_client(
                    api_key=session_token,
                    base_url=COPILOT_BASE_URL,
                    extra_headers={**COPILOT_HEADERS},
                )
                self._session_token = session_token

            return self._client

    def completion(self, request: MessagesRequest) -> MessagesResponse:
        """Route completion through Copilot's OpenAI-compatible endpoint."""
        client = self._get_client()
        model_id = get_copilot_model(request.model)

        try:
            # Copilot only returns tool_calls for Claude models in streaming
            # mode, so always stream upstream and aggregate.
            return call_openai_compatible_chat(
                request,
                client=client,
                model_id=model_id,
                stream_upstream=True,
            )
        except HTTPException as e:
            # If auth fails, invalidate cached session for next request
            if e.status_code == 401:
                self._auth.invalidate_session()
                with self._client_lock:
                    self._client = None
                    self._session_token = None
            raise

    def completion_stream(self, request: MessagesRequest):
        """Stream a completion as Anthropic SSE events, incrementally."""
        client = self._get_client()
        model_id = get_copilot_model(request.model)

        payload = build_openai_payload(request, model_id)
        payload["stream"] = True

        def invalidate_session_on_401(response: requests.Response) -> None:
            if response.status_code == 401:
                self._auth.invalidate_session()
                with self._client_lock:
                    self._client = None
                    self._session_token = None

        response = post_streaming_completion(
            client, payload, "Copilot", on_response=invalidate_session_on_401
        )
        return stream_openai_as_anthropic(response, model_id)

    def count_tokens(self, request: TokenCountRequest) -> TokenCountResponse:
        """Estimate token count using the OpenAI-compatible heuristic."""
        token_count = count_tokens_from_messages_openai(request.messages, request.system)
        return TokenCountResponse(input_tokens=token_count)


def describe_copilot() -> Dict[str, Any]:
    """Runtime info for /, /health, /dashboard."""
    token_file = os.environ.get("COPILOT_TOKEN_FILE")
    token_path = (
        Path(token_file).expanduser() if token_file else Path.home() / ".copilot_token"
    )
    return {
        "provider": "copilot",
        "model": get_copilot_model(),
        "base_url": COPILOT_BASE_URL,
        "token_cached": token_path.exists(),
        "token_endpoint": COPILOT_TOKEN_ENDPOINT,
    }


def validate_copilot_config() -> bool:
    """Copilot can always fall back to the OAuth device flow, so treat as
    valid regardless of whether a token is already cached."""
    return True


PROVIDER = ProviderEntry(
    name="copilot",
    factory=CopilotProvider,
    describe=describe_copilot,
    validate=validate_copilot_config,
)
