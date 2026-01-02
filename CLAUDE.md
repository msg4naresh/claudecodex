# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

```bash
# Setup
uv venv                                     # Create virtual environment
source .venv/bin/activate                   # Activate (macOS/Linux)
uv pip sync requirements.txt                # Install dependencies

# Running
claudecodex                                 # Start server (after installation)
python -m claudecodex.main                  # Or run directly from source
export ANTHROPIC_BASE_URL=http://localhost:8082  # Point Claude Code to proxy

# Linting & Formatting
ruff check src                              # Check for issues
ruff format src                             # Format code
ruff check src --fix                        # Auto-fix issues

# Testing
pytest tests/ -v                            # All tests
pytest tests/unit/ -v                       # Unit tests only (fast)
pytest tests/integration/ -v                # Integration tests (requires API keys)
pytest tests/unit/test_api_endpoints.py::TestMessages -v  # Single test class
pytest tests/unit/test_api_endpoints.py::TestMessages::test_messages_endpoint -v  # Single test

# Backend-specific testing
LLM_BACKEND=bedrock pytest tests/integration/test_bedrock_integration.py -v
LLM_BACKEND=openai_compatible pytest tests/integration/test_gemini_integration.py -v
```

## Architecture Overview

**ClaudeCodeRelayX** is a hackable Claude API proxy that intercepts Claude Code requests and routes them to configurable LLM backends (AWS Bedrock, OpenAI, Google Gemini, local models). The design prioritizes simplicity and transparency—pure HTTP clients with no AI framework dependencies.

### Request Flow

```
Claude Code
  ↓ (ANTHROPIC_BASE_URL=localhost:8082)
  POST /v1/messages
  ↓
server.py (FastAPI)
  ├─ Backend detection (environment variables)
  ├─ Request validation (Pydantic models)
  ├─ Logging & performance tracking
  ↓
bedrock.py OR openai_compatible.py
  ├─ Message translation (Claude ↔ Provider format)
  ├─ Tool handling (function calling schema)
  ├─ API call with error handling
  ├─ Response translation
  ↓
AWS Bedrock / OpenAI / Gemini / Local LLM
```

### Core Source Files

**`src/claudecodex/main.py`** (62 lines)
- Entry point for the `claudecodex` CLI command
- Loads environment variables from `.env`
- Starts FastAPI server on configured port (default: 8082)
- Displays active backend on startup

**`src/claudecodex/server.py`** (278 lines)
- FastAPI application with all endpoints: `/v1/messages`, `/v1/messages/count_tokens`, `/health`
- Backend routing logic: selects active backend based on env vars
- Request/response logging with performance metrics
- CORS middleware for cross-origin requests
- Error handling and validation

**`src/claudecodex/bedrock.py`** (327 lines)
- AWS Bedrock backend implementation
- Message translation: Claude API ↔ AWS Bedrock Converse API format
- Tool calling support (tool_use, tool_result conversion)
- Token counting via Bedrock API
- Retry logic with exponential backoff

**`src/claudecodex/openai_compatible.py`** (447 lines)
- OpenAI-compatible backend supporting: OpenAI, Google Gemini, Azure, Ollama, LM Studio, custom endpoints
- Message translation: Claude API ↔ OpenAI Chat Completions format
- Tool calling support (function calling schema conversion)
- Token estimation (with character-based fallback)
- Retry strategy with exponential backoff

**`src/claudecodex/models.py`** (114 lines)
- Pydantic models for Claude API schema
- Request models: `MessagesRequest`, `TokenCountRequest`, `Message`, `Tool`
- Response models: `MessagesResponse`, `TokenCountResponse`, `Usage`
- Content block types: text, image, tool_use, tool_result

**`src/claudecodex/logging_config.py`** (382 lines)
- Comprehensive logging system with three tiers:
  1. **Console output** - Color-coded real-time summary
  2. **logs/requests.log** - Detailed plain text request/response log
  3. **logs/structured.jsonl** - Machine-readable JSON Lines for parsing
- Tracks request timing, token counts, tool calls
- Performance metrics (tokens per second, latency)

### Backend Selection Logic

Priority order (in `server.py:get_backend_type()`):
1. If `LLM_BACKEND` env var is set → use that backend
2. Otherwise → default to `openai_compatible`

### Translation Layer Pattern

Each backend implements a consistent translation pattern:
- `extract_system_message()` - Extract system prompt from messages
- `convert_to_[provider]_messages()` - Claude message format → Provider format
- `convert_tools_to_[provider]()` - Tool schema conversion
- `create_claude_response_from_[provider]()` - Provider response → Claude format

This pattern makes adding new backends straightforward.

### Environment Variables

**Backend Selection**
```bash
LLM_BACKEND=bedrock              # Force Bedrock backend (or openai_compatible)
```

**Server Configuration**
```bash
SERVER_PORT=8082                 # Server port (default: 8082)
```

**AWS Bedrock** (if using bedrock backend)
```bash
AWS_PROFILE=saml                 # AWS profile name (default: saml)
AWS_DEFAULT_REGION=us-east-1     # AWS region (default: us-east-1)
BEDROCK_MODEL_ID=us.anthropic.claude-haiku-4-5-20251001-v1:0  # Model ID (optional)
```

**OpenAI-Compatible** (for openai_compatible backend)
```bash
OPENAICOMPATIBLE_API_KEY=your-key          # Required API key
OPENAICOMPATIBLE_BASE_URL=endpoint-url     # Provider-specific endpoint (default: Google Gemini)
OPENAI_MODEL=model-name                    # Model identifier (default: gemini-2.0-flash)
```

**Claude Code Integration**
```bash
ANTHROPIC_BASE_URL=http://localhost:8082  # Point Claude Code to proxy
```

### Testing Structure

**`tests/unit/`** - Fast unit tests with mocked external services
- `test_api_endpoints.py` - FastAPI endpoint tests with mocked Bedrock
- `test_openai_compatible.py` - Message translation and OpenAI mocking

**`tests/integration/`** - Full system tests with real API calls (skipped without credentials)
- `test_bedrock_integration.py` - End-to-end Bedrock testing with real AWS calls
- `test_gemini_integration.py` - End-to-end Gemini testing with real API calls

Run integration tests only when API keys are available. Tests will skip gracefully if credentials are missing.