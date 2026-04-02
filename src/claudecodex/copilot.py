"""GitHub Copilot OAuth device flow, session token management, and API constants."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

COPILOT_CLIENT_ID = "Iv1.b507a08c87ecfe98"

COPILOT_BASE_URL = "https://api.githubcopilot.com"
COPILOT_TOKEN_ENDPOINT = "https://api.github.com/copilot_internal/v2/token"

GITHUB_DEVICE_CODE_URL = "https://github.com/login/device/code"
GITHUB_ACCESS_TOKEN_URL = "https://github.com/login/oauth/access_token"

COPILOT_HEADERS = {
    "editor-version": "vscode/1.85.0",
    "Copilot-Integration-Id": "vscode-chat",
    "editor-plugin-version": "copilot-chat/0.22.4",
    "user-agent": "GithubCopilot/1.155.0",
    "Accept": "application/json",
}


class CopilotError(Exception):
    """Base exception for all Copilot auth errors."""


class CopilotAuthError(CopilotError):
    """OAuth or device flow authentication failure."""


class CopilotTokenExpiredError(CopilotAuthError):
    """OAuth or session token has expired."""


class CopilotAuth:
    """Handles OAuth device flow and session token lifecycle.

    Args:
        oauth_token: Explicit OAuth token (skips device flow and file lookup).
        token_file: Path to cached OAuth token file. Defaults to ~/.copilot_token.
    """

    _DEVICE_FLOW_TIMEOUT_SECONDS = 900
    _SESSION_TOKEN_REFRESH_BUFFER_SECONDS = 60
    _DEFAULT_SESSION_LIFETIME_SECONDS = 1500
    _HTTP_TIMEOUT_SECONDS = 10
    _DEVICE_FLOW_POLL_INTERVAL_SECONDS = 5
    _DEVICE_FLOW_BACKOFF_INCREMENT_SECONDS = 5
    _DEVICE_FLOW_MAX_INTERVAL_SECONDS = 60

    def __init__(
        self,
        oauth_token: str | None = None,
        token_file: str | Path | None = None,
    ):
        self._oauth_token = oauth_token.strip() if oauth_token else None
        self._token_file = (
            Path(token_file).expanduser() if token_file else Path.home() / ".copilot_token"
        )
        self._session_token: Optional[str] = None
        self._session_token_expiry: float = 0
        self._lock = threading.Lock()
        self._refresh_lock = threading.Lock()

    def get_session_token(self) -> str:
        """Get a valid Copilot session token, refreshing if needed."""
        # Fast path: return cached token if still valid
        with self._lock:
            if self._session_token and time.time() < (
                self._session_token_expiry - self._SESSION_TOKEN_REFRESH_BUFFER_SECONDS
            ):
                return self._session_token

        # Slow path: refresh token; only one thread should perform the refresh
        with self._refresh_lock:
            # Another thread may have refreshed while we waited
            with self._lock:
                if self._session_token and time.time() < (
                    self._session_token_expiry - self._SESSION_TOKEN_REFRESH_BUFFER_SECONDS
                ):
                    return self._session_token

            oauth_token = self._resolve_oauth_token()

            try:
                resp = requests.get(
                    COPILOT_TOKEN_ENDPOINT,
                    headers={
                        **COPILOT_HEADERS,
                        "Authorization": f"token {oauth_token}",
                    },
                    timeout=self._HTTP_TIMEOUT_SECONDS,
                )
            except requests.exceptions.Timeout as e:
                raise TimeoutError("Timed out fetching Copilot session token") from e
            except requests.exceptions.ConnectionError as e:
                raise ConnectionError(
                    f"Cannot connect to Copilot token endpoint: {e}"
                ) from e

            if resp.status_code == 401:
                logger.warning("OAuth token rejected. Clearing cached token.")
                self._reset_auth_state()
                raise CopilotTokenExpiredError(
                    "Copilot OAuth token expired. Re-authenticate."
                )

            try:
                resp.raise_for_status()
            except requests.HTTPError as e:
                raise CopilotAuthError(
                    f"Failed to fetch session token (HTTP {resp.status_code}): {resp.text[:200]}"
                ) from e

            data = resp.json()
            token = data.get("token")
            if not token:
                raise CopilotAuthError(
                    f"Invalid session token response: {data}"
                )

            session_expiry = self._parse_expiry(token)

            with self._lock:
                self._session_token = token
                self._session_token_expiry = session_expiry

            logger.info(
                "Copilot session token obtained (expires in %ds)",
                int(session_expiry - time.time()),
            )
            return token

    def invalidate_session(self):
        """Force session token refresh on next call."""
        with self._lock:
            self._session_token = None
            self._session_token_expiry = 0

    def _reset_auth_state(self):
        """Clear all auth state and delete the token file.

        CALLER MUST HOLD self._lock.
        """
        with self._lock:
            self._session_token = None
            self._session_token_expiry = 0
            self._oauth_token = None
            try:
                self._token_file.unlink()
                logger.info("Removed cached token file: %s", self._token_file)
            except FileNotFoundError:
                pass
            except OSError as e:
                logger.warning("Failed to remove cached token file %s: %s", self._token_file, e)

    def _resolve_oauth_token(self) -> str:
        """Get OAuth token from explicit value, file, or device flow."""
        if self._oauth_token is not None:
            return self._oauth_token

        try:
            token = self._token_file.read_text().strip()
            if token:
                self._oauth_token = token
                return token
            else:
                # Empty file: clean up to avoid repeated empty reads
                try:
                    self._token_file.unlink()
                except FileNotFoundError:
                    pass
                except OSError as e:
                    logger.warning("Failed to remove empty token file %s: %s", self._token_file, e)
        except FileNotFoundError:
            token = None
        except Exception as e:
            logger.warning("Failed to read token file %s: %s", self._token_file, e)
            token = None

        logger.info("No Copilot OAuth token found. Starting device flow...")
        token = self._run_device_flow()
        self._token_file.write_text(token)
        self._token_file.chmod(0o600)
        self._oauth_token = token
        logger.info("OAuth token saved to %s", self._token_file)
        return token

    def _run_device_flow(self) -> str:
        """Run GitHub OAuth device flow to get an access token."""
        try:
            resp = requests.post(
                GITHUB_DEVICE_CODE_URL,
                json={"client_id": COPILOT_CLIENT_ID, "scope": "read:user"},
                headers={"Accept": "application/json"},
                timeout=self._HTTP_TIMEOUT_SECONDS,
            )
        except requests.exceptions.Timeout as e:
            raise CopilotAuthError("Timed out requesting device code") from e
        except requests.exceptions.ConnectionError as e:
            raise CopilotAuthError(
                f"Cannot connect to GitHub for device flow: {e}"
            ) from e

        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            raise CopilotAuthError(
                f"Failed to request device code (HTTP {resp.status_code}): {resp.text[:200]}"
            ) from e

        data = resp.json()

        device_code = data["device_code"]
        user_code = data["user_code"]
        verification_uri = data["verification_uri"]
        interval = data.get("interval", self._DEVICE_FLOW_POLL_INTERVAL_SECONDS)

        logger.warning(
            "\n%s\n  GitHub Copilot Authentication Required\n%s\n"
            "  1. Open: %s\n  2. Enter code: %s\n  Waiting for authorization...",
            "=" * 60,
            "=" * 60,
            verification_uri,
            user_code,
        )

        deadline = time.time() + self._DEVICE_FLOW_TIMEOUT_SECONDS
        while True:
            if time.time() > deadline:
                raise CopilotAuthError(
                    "Device flow timed out. Please restart and complete authorization within 15 minutes."
                )

            time.sleep(interval)

            try:
                resp = requests.post(
                    GITHUB_ACCESS_TOKEN_URL,
                    json={
                        "client_id": COPILOT_CLIENT_ID,
                        "device_code": device_code,
                        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    },
                    headers={"Accept": "application/json"},
                    timeout=self._HTTP_TIMEOUT_SECONDS,
                )
            except requests.exceptions.Timeout:
                logger.warning("Token poll timed out, retrying...")
                continue
            except requests.exceptions.ConnectionError as e:
                raise CopilotAuthError(
                    f"Connection lost during device flow: {e}"
                ) from e

            try:
                resp.raise_for_status()
            except requests.HTTPError as e:
                raise CopilotAuthError(
                    f"Failed to poll for token (HTTP {resp.status_code}): {resp.text[:200]}"
                ) from e

            result = resp.json()

            if "access_token" in result:
                logger.info("Device flow authorization successful")
                return result["access_token"]

            error = result.get("error")
            if error == "authorization_pending":
                continue
            if error == "slow_down":
                interval = min(
                    interval + self._DEVICE_FLOW_BACKOFF_INCREMENT_SECONDS,
                    self._DEVICE_FLOW_MAX_INTERVAL_SECONDS,
                )
                continue
            if error == "expired_token":
                raise CopilotAuthError("Device code expired. Please restart.")
            raise CopilotAuthError(
                f"Device flow error: {error} - {result.get('error_description', '')}"
            )

    def _parse_expiry(self, token: str) -> float:
        """Parse session token expiry from the token's embedded exp= field."""
        for part in token.split(";"):
            if part.startswith("exp="):
                try:
                    return float(part.split("=", 1)[1])
                except (ValueError, IndexError):
                    logger.warning(
                        "Could not parse exp= value from token fragment: %r", part
                    )
                    break
        logger.warning(
            "Could not parse session token expiry; defaulting to 1500s from now"
        )
        return time.time() + self._DEFAULT_SESSION_LIFETIME_SECONDS
