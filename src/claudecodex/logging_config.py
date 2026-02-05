"""
Minimal request/response logging for Claude Codex.

Goal: make it easy for a human to see what Claude Code requested and
what the LLM returned, without noisy formatting or fragile parsing.
"""

import json
import logging
import os
import re
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

_LOG_DIR = "logs"
_FULL_DIR = os.path.join(_LOG_DIR, "requests_full")
_PREVIEW_LEN = 200
_LOG_FULL_PAYLOADS = True

_SENSITIVE_KEYS = {
    "authorization",
    "api_key",
    "api-key",
    "x-api-key",
    "apikey",
    "token",
    "access_token",
    "secret",
    "password",
    "openai_api_key",
    "openaicompatible_api_key",
}


def _load_log_settings() -> None:
    global _PREVIEW_LEN, _LOG_FULL_PAYLOADS
    _PREVIEW_LEN = int(os.environ.get("LOG_PREVIEW_LEN", str(_PREVIEW_LEN)))
    _LOG_FULL_PAYLOADS = os.environ.get("LOG_FULL_PAYLOADS", "1").lower() not in (
        "0",
        "false",
        "no",
    )


def setup_logging(log_dir: str = "logs") -> Tuple[logging.Logger, logging.Logger]:
    """
    Configure logging.

    Returns:
        (main_logger, request_logger)
    """
    global _LOG_DIR, _FULL_DIR
    _LOG_DIR = log_dir
    _FULL_DIR = os.path.join(_LOG_DIR, "requests_full")
    _load_log_settings()
    os.makedirs(_FULL_DIR, exist_ok=True)

    main_logger = logging.getLogger("claudecodex")
    main_logger.setLevel(logging.WARNING)
    if not main_logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.WARNING)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        main_logger.addHandler(handler)

    request_logger = logging.getLogger("claudecodex.requests")
    request_logger.setLevel(logging.INFO)
    if not request_logger.handlers:
        file_handler = logging.FileHandler(os.path.join(_LOG_DIR, "requests.log"))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        request_logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        request_logger.addHandler(console_handler)

    return main_logger, request_logger


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.replace("\n", " ").replace("\r", " ").split())


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _collect_text_from_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type")
                if block_type == "text":
                    text = block.get("text")
                    if text:
                        parts.append(text)
                elif block_type == "tool_use":
                    name = block.get("name", "tool")
                    parts.append(f"[tool_use:{name}]")
                elif block_type == "tool_result":
                    parts.append("[tool_result]")
        return " ".join(parts)
    return str(content)


def _extract_tools_used(response_data: Dict[str, Any]) -> list[str]:
    tools = []
    for block in response_data.get("content", []) or []:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            name = block.get("name")
            if name:
                tools.append(name)
    return tools


def _extract_request_id(request_data: Dict[str, Any]) -> str:
    rid = request_data.get("request_id")
    if not rid:
        metadata = request_data.get("metadata") or {}
        rid = metadata.get("request_id") if isinstance(metadata, dict) else None
    return str(rid) if rid else uuid.uuid4().hex


def _safe_file_id(request_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", request_id).strip("_")
    if not safe:
        safe = uuid.uuid4().hex
    return safe[:64]


def _is_sensitive_key(key: str) -> bool:
    key_lower = key.lower()
    return key_lower in _SENSITIVE_KEYS or ("key" in key_lower and "api" in key_lower)


def _redact(obj: Any) -> Any:
    if isinstance(obj, dict):
        redacted = {}
        for k, v in obj.items():
            if _is_sensitive_key(k):
                redacted[k] = "***"
            else:
                redacted[k] = _redact(v)
        return redacted
    if isinstance(obj, list):
        return [_redact(item) for item in obj]
    return obj


def _write_full_payload(
    file_id: str,
    payload: Dict[str, Any],
) -> Optional[str]:
    if not _LOG_FULL_PAYLOADS:
        return None
    try:
        path = os.path.abspath(os.path.join(_FULL_DIR, f"{file_id}.json"))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)
        return path
    except OSError:
        return None


def log_request_response(
    request_logger: logging.Logger,
    main_logger: logging.Logger,
    endpoint: str,
    request_data: Dict[str, Any],
    response_data: Dict[str, Any],
    duration: float,
    status_code: int,
    provider_info: Dict[str, Any],
    error: Optional[str] = None,
):
    """
    Log a single request/response pair with a short preview and a full payload file.
    """
    try:
        now = datetime.now()
        timestamp = now.strftime("%H:%M:%S")
        provider = provider_info.get("provider", "unknown")
        model = provider_info.get("model", "unknown")
        request_id = _extract_request_id(request_data)
        file_id = _safe_file_id(request_id)

        msg_count = len(request_data.get("messages", []) or [])
        max_tokens = request_data.get("max_tokens", "N/A")
        temperature = request_data.get("temperature", "N/A")
        tools_count = len(request_data.get("tools") or [])
        tool_choice = request_data.get("tool_choice")

        messages = request_data.get("messages") or []
        request_text = _collect_text_from_content(
            messages[-1].get("content") if messages else ""
        )
        request_preview = _truncate(
            _normalize_whitespace(str(request_text)), _PREVIEW_LEN
        )

        response_text = _collect_text_from_content(response_data.get("content"))
        response_preview = _truncate(
            _normalize_whitespace(str(response_text)), _PREVIEW_LEN
        )
        tools_used = _extract_tools_used(response_data)

        usage = response_data.get("usage") or {}
        in_tokens = usage.get("input_tokens")
        out_tokens = usage.get("output_tokens")
        usage_str = (
            f"{in_tokens}->{out_tokens}"
            if in_tokens is not None and out_tokens is not None
            else (f"{out_tokens}" if out_tokens is not None else "N/A")
        )
        stop_reason = response_data.get("stop_reason")

        full_payload = {
            "id": request_id,
            "timestamp": now.isoformat(),
            "endpoint": endpoint,
            "provider": provider,
            "model": model,
            "status_code": status_code,
            "duration_ms": round(duration * 1000, 2),
            "request": request_data,
            "response": response_data,
            "error": error,
        }
        full_path = _write_full_payload(file_id, _redact(full_payload))

        req_line = (
            f"REQ {timestamp} id={request_id} provider={provider} model={model} "
            f"msgs={msg_count} max={max_tokens} temp={temperature} "
            f"tools={tools_count} tool_choice={tool_choice} "
            f'preview="{request_preview}"'
        )
        if full_path:
            req_line += f" full={full_path}"
        request_logger.info(req_line)

        if error:
            res_line = (
                f"RES {timestamp} id={request_id} status={status_code} "
                f'dur={duration:.3f}s error="{_normalize_whitespace(str(error))}"'
            )
        else:
            tools_used_str = f"[{', '.join(tools_used)}]" if tools_used else "[]"
            res_line = (
                f"RES {timestamp} id={request_id} status={status_code} dur={duration:.3f}s "
                f"usage={usage_str} stop={stop_reason} tools_used={tools_used_str} "
                f'preview="{response_preview}"'
            )
        if full_path:
            res_line += f" full={full_path}"
        request_logger.info(res_line)
    except Exception as e:
        main_logger.warning("logging failed: %s", e)
