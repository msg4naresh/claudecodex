"""GitHub Copilot provider wired through the OpenAI-compatible pipeline."""

from __future__ import annotations

import os
import logging
import threading
from typing import Optional

import requests
from fastapi import HTTPException

from claudecodex.copilot import (
    CopilotAuth,
    COPILOT_HEADERS,
    COPILOT_BASE_URL,
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
)

logger = logging.getLogger(__name__)


def get_copilot_model() -> str:
    """Return configured Copilot model (defaults to gpt-4o)."""
    return os.environ.get("COPILOT_MODEL", "gpt-4o")


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
        model_id = get_copilot_model()

        try:
            return call_openai_compatible_chat(
                request,
                client=client,
                model_id=model_id,
            )
        except HTTPException as e:
            # If auth fails, invalidate cached session for next request
            if e.status_code == 401:
                self._auth.invalidate_session()
                with self._client_lock:
                    self._client = None
                    self._session_token = None
            raise

    def count_tokens(self, request: TokenCountRequest) -> TokenCountResponse:
        """Estimate token count using the OpenAI-compatible heuristic."""
        token_count = count_tokens_from_messages_openai(request.messages, request.system)
        return TokenCountResponse(input_tokens=token_count)
