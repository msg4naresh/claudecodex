# Claude Codex

Monitor and intercept Claude Code API requests with multi-provider LLM support.

A hackable Claude API proxy for monitoring AI agent requests and connecting multiple LLM providers. No complex AI frameworks - just FastAPI, requests, and clean code you can easily modify.

## Quick Start

### 1. Install

```bash
pip install .
```

### 2. Run

**With Gemini (default):**
```bash
export OPENAICOMPATIBLE_API_KEY=your-gemini-key
claudecodex
```

**With AWS Bedrock:**
```bash
export LLM_PROVIDER=bedrock
claudecodex
```

**With GitHub Copilot (GPT-4o / Claude via your Copilot subscription):**
```bash
export LLM_PROVIDER=copilot
export COPILOT_MODEL=gpt-4o          # or claude-3.5-sonnet
claudecodex
```
On first run, a GitHub device flow will prompt you to authorize in your browser. The token is saved to `~/.copilot_token` for future use.

### 3. Connect Claude Code

```bash
export ANTHROPIC_BASE_URL=http://localhost:8082
```

That's it! Claude Code will now send requests through your proxy.

---

## What You Can Do

- **Monitor requests** - See all Claude Code API calls in real-time with color-coded logs
- **Switch providers** - Run the same workflow on Claude, GPT-4, Gemini, or local models
- **Debug workflows** - Watch which tools get called and when
- **Extend capabilities** - Simple translation layer makes adding new providers straightforward

## Architecture

```
┌─────────────┐    ANTHROPIC_BASE_URL=localhost:8082    ┌─────────────────┐
│ Claude Code │ ──────────────────────────────────────► │  Claude Codex   │
└─────────────┘            POST /v1/messages            │   (server.py)   │
                                                        └─────────────────┘
                                                                  │
                                                      get_provider() │
                                                    (Protocol-based) │
                                                                  │
                           ┌──────────────────────────────────────┼──────────────────────────────────────┐
                           │                                      │                                      │
                           ▼                                      ▼                                      ▼
                 ┌─────────────────┐                   ┌─────────────────┐                   ┌─────────────────┐
                 │BedrockProvider  │                   │OpenAICompatible │                   │  Your Custom    │
                 │  (bedrock.py)   │                   │    Provider     │                   │    Provider     │
                 │                 │                   │(openai_compat.py│                   │ (LLMProvider)   │
                 │Implements:      │                   │                 │                   │                 │
                 │ LLMProvider     │                   │Implements:      │                   │Implements:      │
                 │  Protocol       │                   │ LLMProvider     │                   │ LLMProvider     │
                 │                 │                   │  Protocol       │                   │  Protocol       │
                 └─────────────────┘                   └─────────────────┘                   └─────────────────┘
                           │                                      │                                      │
                           ▼                                      ▼                                      ▼
                 ┌─────────────────┐                   ┌─────────────────┐                   ┌─────────────────┐
                 │  AWS Bedrock    │                   │   OpenAI API    │                   │    Any LLM      │
                 │ Converse API    │                   │ Chat Completions│                   │   Provider      │
                 │                 │                   │                 │                   │                 │
                 │ • Claude Sonnet │                   │ • GPT-4/3.5     │                   │ • Local Models  │
                 │ • Claude Haiku  │                   │ • Gemini 2.0    │                   │ • Custom APIs   │
                 │ • Claude Opus   │                   │ • Ollama/LM Std │                   │ • Fine-tuned    │
                 └─────────────────┘                   └─────────────────┘                   └─────────────────┘
```

**How it works:**
1. Claude Code sends requests to Claude Codex (localhost:8082)
2. `server.py` routes to provider instance via `get_provider()`
3. Provider implements `LLMProvider` protocol with `completion()` and `count_tokens()` methods
4. Each provider translates Claude API ↔ Provider API formats
5. Response flows back to Claude Code

**Provider Support:**
- **AWS Bedrock**: Claude Sonnet/Haiku/Opus (native format)
- **OpenAI**: GPT-4, GPT-3.5 (via OpenAI API)
- **Google Gemini**: 2.0-flash, 1.5-pro (via OpenAI-compatible API)
- **Local**: Ollama, LM Studio (via OpenAI-compatible API)


## Project Goals

### Primary: Hackable LLM Request Monitoring
- **See what Claude Code is actually doing** - intercept and log all requests/responses
- **Monitor tool calling patterns** - watch which tools get called and when
- **Debug agentic workflows** - understand multi-step task execution in real-time
- **No AI frameworks required** - pure FastAPI + requests, easy to modify and extend

### Secondary: LLM Provider Flexibility
- **Connect any LLM** - swap between Claude, GPT-4, Gemini, local models instantly
- **Extend time limits** - run longer workflows by routing through your own infrastructure
- **Add new providers** - protocol-based design makes adding new LLMs straightforward
- **Custom model access** - use fine-tuned models or experimental endpoints

### Adding a New Provider

The protocol-based architecture makes it easy to add new LLM providers:

1. **Create a new provider file** (e.g., `src/claudecodex/custom_provider.py`)
2. **Implement the `LLMProvider` protocol**:
   ```python
   from claudecodex.provider import LLMProvider
   from claudecodex.models import MessagesRequest, MessagesResponse, TokenCountRequest, TokenCountResponse

   class CustomProvider:
       """Your custom LLM provider."""

       def completion(self, request: MessagesRequest) -> MessagesResponse:
           # 1. Convert Claude format to your provider's format
           # 2. Call your provider's API
           # 3. Convert response back to Claude format
           pass

       def count_tokens(self, request: TokenCountRequest) -> TokenCountResponse:
           # Count tokens for your provider
           pass
   ```

3. **Register in `server.py`**:
   ```python
   from claudecodex.custom_provider import CustomProvider

   def get_provider() -> LLMProvider:
       provider_type = get_provider_type()

       if provider_type == "bedrock":
           return BedrockProvider()
       elif provider_type == "openai_compatible":
           return OpenAICompatibleProvider()
       elif provider_type == "custom":
           return CustomProvider()
       # ...
   ```

Reference the existing `BedrockProvider` and `OpenAICompatibleProvider` implementations for translation patterns.

## Why Build This?

As developers working with AI agents, we often wonder:
- What requests is Claude Code actually making?
- How does it decide which tools to call?
- Can I run the same workflow on a different/cheaper model?
- How can I extend session time limits?

This proxy gives you **full visibility and control** without complex AI frameworks.

## Providers

- **AWS Bedrock**: Claude Sonnet/Haiku/Opus (`LLM_PROVIDER=bedrock`)
- **Google Gemini**: gemini-2.0-flash (`LLM_PROVIDER=openai_compatible` + Gemini config)
- **OpenAI**: GPT-4/3.5 (`LLM_PROVIDER=openai_compatible` + OpenAI config)
- **Local**: Ollama, LM Studio (`LLM_PROVIDER=openai_compatible` + local config)

---

## For Developers

### Development Setup

This project uses `uv` for package management and `ruff` for linting and formatting.

1. **Install `uv`:**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   uv venv
   uv pip sync requirements.txt
   ```

3. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate  # macOS/Linux
   .venv\Scripts\activate     # Windows
   ```

4. **Lint and format:**
   ```bash
   ruff check src
   ruff format src
   ```

### Testing

```bash
pytest tests/ -v              # All tests
pytest tests/unit/ -v         # Unit tests only (fast)
pytest tests/integration/ -v  # Integration tests (requires API keys)
```

### Install for Development

```bash
pip install -e .  # Install in editable mode
```

## Configuration

**Important:** Do not commit the `.env` file to version control. It should be added to your `.gitignore` file.

### Environment Variables

```bash
# Provider selection
LLM_PROVIDER=bedrock|openai_compatible|copilot  # default: openai_compatible
SERVER_PORT=8082                       # default: 8082

# Bedrock provider
AWS_PROFILE=your-profile               # default: saml
AWS_DEFAULT_REGION=us-east-1           # default: us-east-1
BEDROCK_MODEL_ID=model-id              # default: us.anthropic.claude-sonnet-4-*

# OpenAI-compatible provider
OPENAICOMPATIBLE_API_KEY=your-key      # required for openai_compatible
OPENAICOMPATIBLE_BASE_URL=endpoint-url # provider-specific
OPENAI_MODEL=model-name                # default: gemini-2.0-flash

# GitHub Copilot provider
COPILOT_MODEL=gpt-4o                   # default: gpt-4o (also: claude-3.5-sonnet)
COPILOT_OAUTH_TOKEN=your-token         # optional: skip device flow
COPILOT_TOKEN_FILE=~/.copilot_token    # default: ~/.copilot_token

# Claude Code integration
ANTHROPIC_BASE_URL=http://localhost:8082
```

## File Structure

```
├── pyproject.toml                    # Modern packaging configuration
├── src/claudecodex/                  # Core package
│   ├── __init__.py                   # Package info
│   ├── main.py                       # Entry point
│   ├── server.py                     # FastAPI server & routing
│   ├── provider.py                   # LLMProvider protocol definition
│   ├── bedrock.py                    # AWS Bedrock provider
│   ├── openai_compatible.py          # OpenAI-compatible provider
│   ├── copilot.py                    # GitHub Copilot provider
│   ├── models.py                     # Pydantic models
│   └── logging_config.py             # Monitoring & logging
├── tests/                            # Test package
│   ├── unit/                         # Fast tests with mocks
│   │   ├── test_api_endpoints.py     # API endpoint tests
│   │   └── test_openai_compatible.py # OpenAI service tests
│   └── integration/                  # Full system tests
│       ├── test_bedrock_integration.py # Bedrock end-to-end tests
│       └── test_gemini_integration.py  # Gemini end-to-end tests
├── logs/                             # Generated log files
├── requirements.txt                  # Dependencies
└── README.md                         # This file
```

**Protocol-Based Architecture**: Extensible design makes adding new providers easy
- `provider.py` - `LLMProvider` protocol interface
- `server.py` - FastAPI server with provider routing and API endpoints
- `bedrock.py` - AWS Bedrock provider implementing `LLMProvider`
- `openai_compatible.py` - OpenAI-compatible provider implementing `LLMProvider`

## API Functions

### Provider Protocol (`provider.py`)
- `LLMProvider` - Protocol defining provider interface:
  - `completion(request)` - Get completion from LLM
  - `count_tokens(request)` - Count tokens for a request

### Server Functions (`server.py`)
- `get_provider_type()` - Determine provider from environment
- `get_provider()` - Get provider instance (returns `LLMProvider`)
- `call_llm_service(request)` - Route requests to providers
- `count_llm_tokens(request)` - Token counting
- `get_provider_info()` - Runtime configuration info
- `create_message(request)` - `/v1/messages` endpoint
- `count_tokens(request)` - `/v1/messages/count_tokens` endpoint
- `health()` - `/health` endpoint

### Bedrock Provider (`bedrock.py`)
- `BedrockProvider` - Provider class implementing `LLMProvider`
  - `completion(request)` - Get completion from AWS Bedrock
  - `count_tokens(request)` - Count tokens for Bedrock
- `call_bedrock_converse(request)` - AWS Bedrock API calls
- `count_request_tokens(request)` - Bedrock token counting
- `get_bedrock_client()` - AWS client setup
- `convert_to_bedrock_messages()` - Claude → Bedrock format
- `create_claude_response()` - Bedrock → Claude format

### OpenAI-Compatible Provider (`openai_compatible.py`)
- `OpenAICompatibleProvider` - Provider class implementing `LLMProvider`
  - `completion(request)` - Get completion from OpenAI-compatible provider
  - `count_tokens(request)` - Count tokens for OpenAI
- `call_openai_compatible_chat(request)` - OpenAI API calls
- `count_openai_tokens(request)` - Token estimation
- `get_openai_compatible_client()` - HTTP session setup
- `convert_to_openai_messages()` - Claude → OpenAI format
- `create_claude_response_from_openai()` - OpenAI → Claude format

