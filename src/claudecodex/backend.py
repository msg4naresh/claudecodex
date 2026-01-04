"""
Backend interface for LLM providers.
"""

from typing import Protocol
from claudecodex.models import (
    MessagesRequest,
    MessagesResponse,
    TokenCountRequest,
    TokenCountResponse
)


class LLMBackend(Protocol):
    """
    Backend protocol for LLM providers.

    Each backend handles:
    1. Converting Claude Code requests to provider-specific format
    2. Calling the provider API
    3. Converting provider responses back to Claude Code format

    """

    def completion(self, request: MessagesRequest) -> MessagesResponse:
        """
        Get completion from LLM backend.

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
