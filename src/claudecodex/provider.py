"""
Provider interface for LLM providers.
"""

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol
from claudecodex.models import (
    MessagesRequest,
    MessagesResponse,
    TokenCountRequest,
    TokenCountResponse
)


def anthropic_sse(event: str, data: Dict[str, Any]) -> str:
    """Format one Anthropic-style server-sent event. Shared by every
    streaming provider translator (openai_compatible.py, bedrock.py)."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def detect_model_family(requested_model: Optional[str]) -> Optional[str]:
    """Detect which Claude model family a client requested, from its model
    string. Used so providers hosting real Claude models (Bedrock, Copilot)
    can honor Claude Code's /model switching. Returns "opus", "sonnet",
    "haiku", or None if the string doesn't name a recognizable family.
    """
    if not requested_model:
        return None
    lowered = requested_model.lower()
    for family in ("opus", "sonnet", "haiku"):
        if family in lowered:
            return family
    return None


class LLMProvider(Protocol):
    """
    Provider protocol for LLM providers.

    Each provider handles:
    1. Converting Claude Code requests to provider-specific format
    2. Calling the provider API
    3. Converting provider responses back to Claude Code format

    """

    def completion(self, request: MessagesRequest) -> MessagesResponse:
        """
        Get completion from LLM provider.

        Internally handles format conversion:
        - Claude Code MessagesRequest → Provider format
        - Call provider API
        - Provider response → Claude Code MessagesResponse

        Args:
            request: Claude-formatted messages request

        Returns:
            Claude-formatted messages response

        Note:
            Format conversion is transparent to callers.
        """
        ...

    def count_tokens(self, request: TokenCountRequest) -> TokenCountResponse:
        """
        Count tokens for a request.

        May involve conversion to provider format for accurate counting.

        Args:
            request: Token count request with messages

        Returns:
            Token count response with input token count
        """
        ...


@dataclass(frozen=True)
class ProviderEntry:
    """Everything the registry needs to offer one provider, owned by that
    provider's own module. server.py talks only to the registry - it never
    branches on a provider's name, env vars, or metadata shape.
    """

    name: str
    factory: Callable[[], LLMProvider]
    describe: Callable[[], Dict[str, Any]]  # runtime info for /, /health, /dashboard
    validate: Callable[[], bool]  # True if config looks usable at startup
