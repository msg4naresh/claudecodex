#!/usr/bin/env python3
"""
Main entry point for Claude Codex.

Clean startup script that initializes and runs the FastAPI server.
The server provides Claude API compatibility for multiple LLM backends.

Usage:
    python main.py

The server will start on http://0.0.0.0:8082 by default.

Features:
    - Auto-detects backend (Bedrock or OpenAI-compatible)
    - Environment variable configuration
    - Structured logging and monitoring
    - Claude API compatible endpoints
"""

import uvicorn
import logging
import os
from dotenv import load_dotenv

from claudecodex.server import app, get_backend_info


def main():
    """
    Start the Claude Codex server.

    Loads configuration, displays backend info, and starts the server.
    All server logic is contained in server.py for easy hacking.
    """
    # Load environment variables from .env file
    load_dotenv()

    # Get port from environment variable, fallback to 8082
    port = int(os.getenv("SERVER_PORT", 8082))

    # Setup logging for main entry point
    logging.basicConfig(level=logging.INFO)

    # Display minimal startup information
    backend_info = get_backend_info()
    print(f"Claude Codex starting on http://localhost:{port}")
    print(f"Backend: {backend_info['backend']} | Model: {backend_info['model']}")

    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
