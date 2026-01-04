"""
Enhanced logging configuration for the Claude Codex proxy server.

This module provides comprehensive monitoring logging with:
- Color-coded request/response flows for easy visual tracking
- Structured JSON logs for machine parsing
- Performance metrics and timing information
- Error tracking and debugging capabilities
- Real-time console monitoring with visual indicators
"""

import logging
import json
import os
from datetime import datetime
# Note: get_backend_info imported dynamically to avoid circular imports


# Color constants
COLORS = {
    "DEBUG": "\033[36m",  # Cyan
    "INFO": "\033[32m",  # Green
    "WARNING": "\033[33m",  # Yellow
    "ERROR": "\033[31m",  # Red
    "CRITICAL": "\033[35m",  # Magenta
}

REQUEST_COLOR = "\033[34m"  # Blue
SUCCESS_COLOR = "\033[32m"  # Green
ERROR_COLOR = "\033[31m"  # Red
TIMING_COLOR = "\033[36m"  # Cyan
MODEL_COLOR = "\033[33m"  # Yellow
TOKEN_COLOR = "\033[35m"  # Magenta
TOOL_COLOR = "\033[36m"  # Cyan
TOOL_NAME_COLOR = "\033[33m"  # Yellow
TOOL_INPUT_COLOR = "\033[37m"  # Light Gray
METRICS_COLOR = "\033[36m"  # Cyan
PREFIX_COLOR = "\033[37m"  # Light Gray
CLAUDE_REQUEST_COLOR = "\033[35m"  # Magenta

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"


class ColoredFormatter(logging.Formatter):
    """
    Enhanced formatter with color-coded request/response flows.

    Color scheme for monitoring:
    - Requests (→): Blue for incoming requests
    - Responses (←): Green for successful responses, Red for errors
    - System: Yellow for warnings, Magenta for critical issues
    - Timing: Cyan for performance metrics
    """

    def format(self, record):
        """Apply intelligent color formatting based on message content."""
        message = record.getMessage()

        if message.startswith("🔧 TOOL_CALL"):
            formatted_msg = self._format_tool_call(message)
        elif message.startswith("[Claude Code Request]"):
            formatted_msg = self._format_request(message)
        elif message.startswith("[LLM Response]"):
            formatted_msg = self._format_response(message)
        elif message.startswith("→"):
            formatted_msg = self._format_legacy_request(message)
        elif message.startswith("←"):
            formatted_msg = self._format_legacy_response(message)
        elif message.startswith("─"):
            formatted_msg = f"{DIM}{message}{RESET}"
        elif "|" in message and (
            "tokens" in message or "ms" in message or "s" in message
        ):
            formatted_msg = f"{TIMING_COLOR}{message}{RESET}"
        else:
            color = COLORS.get(record.levelname, RESET)
            record.levelname = f"{color}{BOLD}{record.levelname}{RESET}"
            return super().format(record)

        record.msg = formatted_msg
        record.args = ()
        return super().format(record)

    def _format_tool_call(self, message):
        """Format tool call messages with colors."""
        parts = message[14:].split(" | ")  # Skip "🔧 TOOL_CALL | "
        if len(parts) >= 3:
            tool_name = parts[0]
            tool_id = parts[1]
            tool_input = " | ".join(parts[2:])
            return f"{TOOL_COLOR}{BOLD}🔧 TOOL_CALL{RESET} | {TOOL_NAME_COLOR}{BOLD}{tool_name}{RESET} | {DIM}{tool_id}{RESET} | {TOOL_INPUT_COLOR}{tool_input}{RESET}"
        else:
            return f"{TOOL_COLOR}{BOLD}🔧 TOOL_CALL{RESET} {TOOL_COLOR}{message[14:].strip()}{RESET}"

    def _format_request(self, message):
        """Format Claude Code Request messages with colors."""
        content = message[len("[Claude Code Request]") :].strip()
        formatted_msg = f"{PREFIX_COLOR}[{RESET}{CLAUDE_REQUEST_COLOR}{BOLD}Claude Code Request{RESET}{PREFIX_COLOR}]{RESET} {REQUEST_COLOR}{BOLD}→{RESET} {content}"

        # Highlight specific parts
        formatted_msg = formatted_msg.replace(
            "tools", f"{TOOL_COLOR}{BOLD}tools{RESET}"
        )
        for model in [
            "gemini-2.0-flash",
            "claude-3-sonnet",
            "claude-3-haiku",
            "gpt-4",
            "gpt-3.5"
        ]:
            if model in formatted_msg:
                formatted_msg = formatted_msg.replace(
                    model, f"{MODEL_COLOR}{BOLD}{model}{RESET}"
                )

        return formatted_msg

    def _format_response(self, message):
        """Format LLM Response messages with colors."""
        content = message[len("[LLM Response]") :].strip()

        if "ERROR" in content:
            return f"{PREFIX_COLOR}[{RESET}{ERROR_COLOR}{BOLD}LLM Response{RESET}{PREFIX_COLOR}]{RESET} {ERROR_COLOR}{BOLD}←{RESET} {ERROR_COLOR}{content[1:].strip()}{RESET}"

        formatted_msg = f"{PREFIX_COLOR}[{RESET}{SUCCESS_COLOR}{BOLD}LLM Response{RESET}{PREFIX_COLOR}]{RESET} {SUCCESS_COLOR}{BOLD}←{RESET} {content[1:].strip()}"
        return self._apply_tool_highlights(formatted_msg)

    def _format_legacy_request(self, message):
        """Format legacy request messages with colors."""
        formatted_msg = (
            f"{REQUEST_COLOR}{BOLD}→{RESET} {REQUEST_COLOR}{message[1:].strip()}{RESET}"
        )
        return self._highlight_tools_text(formatted_msg)

    def _format_legacy_response(self, message):
        """Format legacy response messages with colors."""
        if "ERROR" in message:
            return (
                f"{ERROR_COLOR}{BOLD}←{RESET} {ERROR_COLOR}{message[1:].strip()}{RESET}"
            )

        formatted_msg = (
            f"{SUCCESS_COLOR}{BOLD}←{RESET} {SUCCESS_COLOR}{message[1:].strip()}{RESET}"
        )
        return self._highlight_tools_text(formatted_msg)

    def _apply_tool_highlights(self, formatted_msg):
        """Apply tool and metrics highlighting to response messages."""
        import re

        # Highlight performance metrics
        formatted_msg = re.sub(
            r"(\d+\.\d+s)", f"{TIMING_COLOR}{BOLD}\\1{RESET}", formatted_msg
        )
        formatted_msg = re.sub(
            r"(\d+→\d+ tok)", f"{TOKEN_COLOR}{BOLD}\\1{RESET}", formatted_msg
        )
        formatted_msg = re.sub(
            r"(\d+\.\d+ tok/s)", f"{METRICS_COLOR}{BOLD}\\1{RESET}", formatted_msg
        )

        # Apply tool highlighting
        formatted_msg = self._highlight_tools_text(formatted_msg)

        # Highlight stop reason
        formatted_msg = re.sub(
            r"stop:(🔧)?(\w+)",
            f"stop:{TOOL_COLOR}\\1{RESET}{METRICS_COLOR}{BOLD}\\2{RESET}",
            formatted_msg
        )
        return formatted_msg

    def _highlight_tools_text(self, formatted_msg):
        """Apply consistent tool highlighting across message types."""
        import re

        formatted_msg = formatted_msg.replace(
            "TOOLS:", f"{TOOL_COLOR}{BOLD}TOOLS:{RESET}"
        )
        formatted_msg = formatted_msg.replace("🔧", f"{TOOL_COLOR}🔧{RESET}")
        formatted_msg = formatted_msg.replace(
            "tools", f"{TOOL_COLOR}{BOLD}tools{RESET}"
        )

        # Highlight tool names in TOOLS section
        tools_match = re.search(r'TOOLS: (.+?)(?:\s\||\s*"|\s*$)', formatted_msg)
        if tools_match:
            tools_content = tools_match.group(1)
            tool_names = re.findall(r"🔧([^(]+)", tools_content)
            for tool_name in tool_names:
                formatted_msg = formatted_msg.replace(
                    f"🔧{tool_name}",
                    f"{TOOL_COLOR}🔧{RESET}{TOOL_NAME_COLOR}{BOLD}{tool_name}{RESET}"
                )

        return formatted_msg


def setup_logging(log_dir: str = "logs") -> tuple[logging.Logger, logging.Logger]:
    """
    Configure logging for the Claude-Bedrock proxy server.

    Args:
        log_dir: Directory to store log files (created if it doesn't exist)

    Returns:
        Tuple of (main_logger, request_logger) for different logging purposes

    Creates clean request/response only logging:
        - requests.log: Clean request/response logs only
        - Console: Minimal output for essential messages only
    """
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)

    # Setup main application logger (minimal console output only)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.WARNING)  # Only show warnings and errors

    # Console handler for errors only
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_formatter = ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    # Add minimal handler to main logger
    logger.addHandler(console_handler)

    # Create separate logger for clean request/response logs only
    request_logger = logging.getLogger("requests")
    request_logger.setLevel(logging.INFO)

    # Request/Response file handler WITHOUT colors (clean)
    request_log_handler = logging.FileHandler(f"{log_dir}/requests.log")
    request_log_handler.setLevel(logging.INFO)
    request_formatter = logging.Formatter(
        "%(message)s"
    )  # Plain formatter for clean file logs
    request_log_handler.setFormatter(request_formatter)
    request_logger.addHandler(request_log_handler)

    # Also output request/response to console
    request_console_handler = logging.StreamHandler()
    request_console_handler.setLevel(logging.INFO)
    request_console_formatter = logging.Formatter("%(message)s")
    request_console_handler.setFormatter(request_console_formatter)
    request_logger.addHandler(request_console_handler)

    return logger, request_logger


def _extract_request_content(request_data: dict) -> str:
    """Extract content from request messages."""
    if not request_data.get("messages"):
        return ""

    last_msg = request_data["messages"][-1]
    if not last_msg.get("content"):
        return ""

    if isinstance(last_msg["content"], list):
        content = last_msg["content"][0].get("text", "")
    else:
        content = str(last_msg["content"])

    return f' | "{content}"' if content else ""


def _extract_response_content(response_data: dict) -> tuple[list, str]:
    """Extract tool calls and text content from response."""
    tool_calls = []
    text_content = ""

    if not response_data.get("content"):
        return tool_calls, text_content

    for block in response_data["content"]:
        if block.get("type") == "text":
            text_content = block.get("text", "")
        elif block.get("type") == "tool_use":
            tool_name = block.get("name", "unknown")
            tool_input = block.get("input", {})
            tool_calls.append(
                f"🔧{tool_name}({json.dumps(tool_input) if tool_input else ''})"
            )

    return tool_calls, text_content


def _build_metrics_info(response_data: dict, duration: float) -> str:
    """Build performance metrics string."""
    metrics = []

    if "usage" in response_data:
        usage = response_data["usage"]
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        tokens_per_sec = output_tokens / duration if duration > 0 else 0
        metrics.append(f"{input_tokens}→{output_tokens} tok")
        if tokens_per_sec > 0:
            metrics.append(f"{tokens_per_sec:.1f} tok/s")

    if response_data.get("stop_reason"):
        stop_reason = response_data["stop_reason"]
        if stop_reason == "tool_use":
            metrics.append(f"stop:🔧{stop_reason}")
        else:
            metrics.append(f"stop:{stop_reason}")

    return " | " + " | ".join(metrics) if metrics else ""


def _log_tool_calls(request_logger: logging.Logger, response_data: dict, error: str):
    """Log individual tool calls with detailed formatting."""
    if error or not response_data.get("content"):
        return

    for block in response_data["content"]:
        if block.get("type") == "tool_use":
            tool_name = block.get("name", "unknown")
            tool_input = block.get("input", {})
            tool_id = block.get("id", "no-id")

            tool_log = f"🔧 TOOL_CALL | {tool_name} | id:{tool_id} | input:{json.dumps(tool_input, indent=None)}"
            request_logger.info(tool_log)


def _write_structured_log(
    endpoint: str,
    backend: str,
    model: str,
    request_data: dict,
    response_data: dict,
    msg_count: int,
    max_tokens,
    temperature,
    duration: float,
    status_code: int,
    error: str,
):
    """Write structured JSON log for machine parsing."""
    json_log_data = {
        "timestamp": datetime.now().isoformat(),
        "endpoint": endpoint,
        "backend": backend,
        "model": model,
        "request": {
            "message_count": msg_count,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "has_tools": bool(request_data.get("tools")),
            "request_headers": dict(request_data.get("headers") or {})
        },
        "response": {
            "status_code": status_code,
            "duration_ms": round(duration * 1000, 2),
            "error": error,
            "usage": response_data.get("usage") if not error else None
        },
        "request_id": request_data.get("request_id"),
        "user_id": request_data.get("user_id"),
    }

    json_log_file = os.path.join("logs", "structured.jsonl")
    os.makedirs("logs", exist_ok=True)
    try:
        with open(json_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(json_log_data) + "\n")
    except Exception:
        pass  # Don't let logging failures break the app


def log_request_response(
    request_logger: logging.Logger,
    main_logger: logging.Logger,
    endpoint: str,
    request_data: dict,
    response_data: dict,
    duration: float,
    status_code: int,
    backend_info: dict,
    error: str = None
):
    """
    Enhanced request/response logging with visual monitoring indicators.

    Provides color-coded, structured logs optimized for real-time monitoring
    with clear visual separation between requests and responses.
    """
    timestamp = datetime.now().strftime("%H:%M:%S")

    # Get backend info for context
    backend = backend_info.get("backend", "unknown")
    model = backend_info.get("model", "unknown")

    # Extract request details
    msg_count = len(request_data.get("messages", []))
    max_tokens = request_data.get("max_tokens", "N/A")
    temperature = request_data.get("temperature", "N/A")

    # Build request log components
    request_preview = _extract_request_content(request_data)
    tools_info = (
        f" | {len(request_data['tools'])} tools" if request_data.get("tools") else ""
    )

    # Log incoming request
    request_log = f"[Claude Code Request] → [{timestamp}] {model} ({backend}) | {msg_count} msgs | max:{max_tokens} T:{temperature}{tools_info}{request_preview}"
    request_logger.info(request_log)

    # Log response
    if error:
        response_log = (
            f"[LLM Response] ← ERROR {status_code} | {duration:.3f}s | {error}"
        )
    else:
        tool_calls, text_content = _extract_response_content(response_data)

        # Build response components
        tool_calls_info = f" | TOOLS: {', '.join(tool_calls)}" if tool_calls else ""
        response_preview = f' | "{text_content}"' if text_content else ""
        metrics_info = _build_metrics_info(response_data, duration)

        response_log = f"[LLM Response] ← {status_code} | {duration:.3f}s{metrics_info}{tool_calls_info}{response_preview}"

    request_logger.info(response_log)

    # Log individual tool calls
    _log_tool_calls(request_logger, response_data, error)

    # Write structured log
    _write_structured_log(
        endpoint,
        backend,
        model,
        request_data,
        response_data,
        msg_count,
        max_tokens,
        temperature,
        duration,
        status_code,
        error
    )

    # Visual separator for readability
    request_logger.info("─" * 80)
