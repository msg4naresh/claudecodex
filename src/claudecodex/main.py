#!/usr/bin/env python3
"""
Main entry point for Claude Codex.

Clean startup script that initializes and runs the FastAPI server.
The server provides Claude API compatibility for multiple LLM providers.

Usage:
    python main.py

The server will start on http://0.0.0.0:8082 by default.

Features:
    - Auto-detects provider (Bedrock or OpenAI-compatible)
    - Environment variable configuration
    - Structured logging and monitoring
    - Claude API compatible endpoints
"""

import sys
import uvicorn
import logging
import os
from dotenv import load_dotenv

from claudecodex.server import app, get_provider_info, get_provider_type, validate_provider_config


def main():
    """
    Start the Claude Codex server.

    Loads configuration, validates it, displays provider info, and starts
    the server. All server logic is contained in server.py for easy hacking.
    """
    # Load environment variables from .env file
    load_dotenv()

    # Get port from environment variable, fallback to 8082
    port = int(os.getenv("SERVER_PORT", 8082))
    # Bind localhost only by default; set SERVER_HOST=0.0.0.0 to expose
    host = os.getenv("SERVER_HOST", "127.0.0.1")

    # Setup logging for main entry point
    logging.basicConfig(level=logging.INFO)

    # Fail fast on an unusable provider config instead of starting a
    # server that would 500 on every request
    try:
        provider_type = get_provider_type()
    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)

    if not validate_provider_config():
        print(
            f"Configuration error: provider '{provider_type}' is missing "
            f"required configuration (see README for required env vars)."
        )
        sys.exit(1)

    # Display minimal startup information
    provider_info = get_provider_info()
    print(f"Claude Codex starting on http://localhost:{port}")
    print(f"Provider: {provider_info['provider']} | Model: {provider_info['model']}")

    # Start the server
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
