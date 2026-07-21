# Claude Codex

[![CI](https://github.com/msg4naresh/claudecodex/actions/workflows/ci.yml/badge.svg)](https://github.com/msg4naresh/claudecodex/actions/workflows/ci.yml)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/github/license/msg4naresh/claudecodex)

**Run Claude Code on any LLM backend — and see every request it makes.**

Claude Codex is a transparent, Anthropic-API-compatible proxy. Point Claude Code at it and your agent sessions run on AWS Bedrock, GitHub Copilot, Gemini, or any OpenAI-compatible endpoint — with full tool-use agent loops, SSE streaming, and a color-coded live log of every request, tool call, and token count.


- **Full protocol fidelity** — real Claude Code sessions work end to end: streaming SSE events, multi-turn tool loops, Anthropic error types that preserve client retry behavior
- **Provider-verified** — an e2e test matrix runs every provider/model pair through completion, tool round-trips, streaming, and token counting (it has caught real upstream API quirks, like GitHub Copilot omitting tool calls in non-streaming responses)
- **Hackable by design** — no AI frameworks; FastAPI + requests + one `LLMProvider` protocol. Adding a provider is one file.

## Quick Start

### 1. Install

```bash
pip install .
```

### 2. Run

**With GitHub Copilot (default):**
```bash
export LLM_PROVIDER=copilot   # optional; copilot is default when unset
# Optional model pin (disables /model switching):
# export COPILOT_MODEL=claude-sonnet-4.6
claudecodex
```

**With Gemini:**
```bash
export OPENAICOMPATIBLE_API_KEY=your-gemini-key
export LLM_PROVIDER=openai_compatible
claudecodex
```

**With AWS Bedrock:**
```bash
export LLM_PROVIDER=bedrock
claudecodex
```

On first Copilot run, a GitHub device flow will prompt you to authorize in your browser. The token is saved to `~/.copilot_token` for future use.

### 3. Connect Claude Code

```bash
export ANTHROPIC_BASE_URL=http://localhost:8082
export ANTHROPIC_AUTH_TOKEN=dummy   # any non-empty string; skips the login prompt
```

That's it! Claude Code will now send requests through your proxy.

---

## Troubleshooting

**`address already in use` on startup** — a previous `claudecodex` process is still holding the port. Find and stop it:

```bash
lsof -nP -iTCP:8082 -sTCP:LISTEN   # find what's using the port (swap 8082 for SERVER_PORT if changed)
kill $(lsof -t -nP -iTCP:8082 -sTCP:LISTEN)   # stop it
```

Then restart `claudecodex` normally.

---

## What You Can Do

- **Watch it live** - open [http://localhost:8082/dashboard](http://localhost:8082/dashboard) for a real-time view: requests, errors, latency, token usage, tool calls
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

Providers register themselves in `registry.py` - **`server.py` never needs to change** to add one.

1. **Create a new provider file** (e.g., `src/claudecodex/custom_provider.py`)
2. **Implement the `LLMProvider` protocol, plus a `PROVIDER` entry**:
   ```python
   from claudecodex.provider import LLMProvider, ProviderEntry
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

       # Optional: def completion_stream(self, request) -> incremental SSE events
       # (see bedrock.py or copilot_provider.py). Providers without it fall
       # back to replay streaming automatically.

   def describe_custom() -> dict:
       """Runtime info shown at /, /health, /dashboard."""
       return {"provider": "custom", "model": "your-model-id"}

   def validate_custom_config() -> bool:
       """Checked at startup; False makes claudecodex exit with a clear error."""
       return True

   PROVIDER = ProviderEntry(
       name="custom",
       factory=CustomProvider,
       describe=describe_custom,
       validate=validate_custom_config,
   )
   ```

3. **Add one import + one entry in `registry.py`** - the only file that changes:
   ```python
   from claudecodex.custom_provider import PROVIDER as _custom

   _PROVIDERS = {p.name: p for p in (_bedrock, _openai_compatible, _copilot, _custom)}
   ```

Reference the existing `bedrock.py` and `openai_compatible.py` for translation patterns.

## Why Build This?

As developers working with AI agents, we often wonder:
- What requests is Claude Code actually making?
- How does it decide which tools to call?
- Can I run the same workflow on a different/cheaper model?
- How can I extend session time limits?

This proxy gives you **full visibility and control** without complex AI frameworks.

## Providers

- **GitHub Copilot** (default): Claude Sonnet/Haiku/Opus, GPT models via your Copilot subscription (`LLM_PROVIDER=copilot`)
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
   uv pip sync requirements-dev.txt   # runtime + pytest/ruff for development
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
RUN_E2E=1 pytest tests/e2e/ -v  # Provider/model matrix (real API calls; skips without creds)
```

### Install for Development

```bash
pip install -e ".[dev]"  # Editable mode + pytest/ruff
```

## Configuration

**Important:** Do not commit the `.env` file to version control. It should be added to your `.gitignore` file.

### Environment Variables

```bash
# Provider selection
LLM_PROVIDER=bedrock|openai_compatible|copilot  # default: copilot
SERVER_PORT=8082                       # default: 8082
SERVER_HOST=127.0.0.1                  # default: 127.0.0.1 (localhost only; 0.0.0.0 exposes to network)

# Bedrock provider
AWS_PROFILE=your-profile               # default: saml
AWS_DEFAULT_REGION=us-east-1           # default: us-east-1
BEDROCK_MODEL_ID=model-id              # optional: pins one model; if set, /model switching is a no-op
BEDROCK_MODEL_ID_OPUS=model-id         # optional: override the opus profile ID (default: us.anthropic.claude-opus-4-8)
BEDROCK_MODEL_ID_SONNET=model-id       # optional: override the sonnet profile ID (default: us.anthropic.claude-sonnet-4-6)
BEDROCK_MODEL_ID_HAIKU=model-id        # optional: override the haiku profile ID (default: us.anthropic.claude-haiku-4-5-20251001-v1:0)

# OpenAI-compatible provider
OPENAICOMPATIBLE_API_KEY=your-key      # required for openai_compatible
OPENAICOMPATIBLE_BASE_URL=endpoint-url # provider-specific
OPENAI_MODEL=model-name                # default: gemini-2.0-flash

# GitHub Copilot provider
COPILOT_MODEL=claude-sonnet-4.6         # default: claude-sonnet-4.6 (also: gpt-4o, claude-haiku-4.5); if set, /model switching is a no-op
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
│   ├── server.py                     # FastAPI server, routing & streaming
│   ├── provider.py                   # LLMProvider protocol + ProviderEntry
│   ├── registry.py                   # Provider registry - the only file that changes to add one
│   ├── bedrock.py                    # AWS Bedrock provider
│   ├── openai_compatible.py          # OpenAI-compatible provider + SSE translation
│   ├── copilot.py                    # GitHub Copilot auth (OAuth device flow)
│   ├── copilot_provider.py           # GitHub Copilot provider (streaming)
│   ├── dashboard.py                  # Live monitoring dashboard (/dashboard)
│   ├── models.py                     # Pydantic models
│   └── logging_config.py             # Monitoring & logging
├── tests/                            # Test package
│   ├── unit/                         # Fast tests with mocks
│   │   ├── test_api_endpoints.py     # API endpoint tests
│   │   └── test_openai_compatible.py # Translation & streaming tests
│   ├── integration/                  # Full system tests
│   │   ├── test_bedrock_integration.py # Bedrock end-to-end tests
│   │   └── test_gemini_integration.py  # Gemini end-to-end tests
│   └── e2e/                          # Provider/model matrix (RUN_E2E=1)
│       └── test_provider_matrix.py   # Completion/tools/streaming/tokens per provider
├── logs/                             # Generated log files
├── requirements.txt                  # Dependencies
└── README.md                         # This file
```

**Registry-Based Architecture**: Extensible design makes adding new providers easy - `server.py` never needs to change
- `provider.py` - `LLMProvider` protocol + `ProviderEntry` (what a provider registers)
- `registry.py` - Maps provider names to their `ProviderEntry`; the one file touched to add a provider
- `server.py` - FastAPI server with routing and API endpoints, talks only to the registry
- `bedrock.py` - AWS Bedrock provider implementing `LLMProvider`
- `openai_compatible.py` - OpenAI-compatible provider implementing `LLMProvider`

## API Functions

### Provider Protocol (`provider.py`)
- `LLMProvider` - Protocol defining provider interface:
  - `completion(request)` - Get completion from LLM
  - `completion_stream(request)` - Optional: incremental SSE events (providers without it fall back to replay streaming)
  - `count_tokens(request)` - Count tokens for a request
- `ProviderEntry` - What a provider registers: `name`, `factory`, `describe()`, `validate()`
- `anthropic_sse()` - Shared Anthropic SSE event formatter
- `detect_model_family()` - Maps a requested model string to opus/sonnet/haiku for `/model` switching

### Registry (`registry.py`)
- `get_entry(name)` - Look up a provider's `ProviderEntry`
- `provider_names()` - List all registered provider names
- `DEFAULT_PROVIDER` - Provider used when `LLM_PROVIDER` is unset

### Server Functions (`server.py`)
- `get_provider_type()` - Determine provider from environment
- `get_provider()` - Get provider instance (returns `LLMProvider`, cached per type)
- `call_llm_service(request)` - Route requests to providers
- `count_llm_tokens(request)` - Token counting
- `validate_provider_config()` - Checked at startup (`main.py`); a provider reporting invalid config exits the process with a clear error instead of starting a server that would 500 on every request
- `get_provider_info()` - Runtime configuration info
- `create_message(request)` - `/v1/messages` endpoint (JSON or SSE streaming)
- `count_tokens(request)` - `/v1/messages/count_tokens` endpoint
- `dashboard()` / `dashboard_data()` - `/dashboard` live monitor + JSON feed
- `health()` - `/health` endpoint

### Bedrock Provider (`bedrock.py`)
- `BedrockProvider` - Provider class implementing `LLMProvider`
  - `completion(request)` / `completion_stream(request)` - Get a completion from AWS Bedrock, JSON or incremental SSE
  - `count_tokens(request)` - Count tokens for Bedrock
- `build_converse_params(request, model_id)` - Shared Converse API param builder (sync + streaming)
- `call_bedrock_converse(request)` - AWS Bedrock API calls
- `stream_bedrock_as_anthropic()` - Bedrock Converse stream → Anthropic SSE events (incremental)
- `count_request_tokens(request)` - Bedrock token counting
- `get_bedrock_client()` - AWS client setup
- `convert_to_bedrock_messages()` - Claude → Bedrock format
- `create_claude_response()` - Bedrock → Claude format
- `PROVIDER` - This provider's `ProviderEntry`, registered in `registry.py`

### OpenAI-Compatible Provider (`openai_compatible.py`)
- `OpenAICompatibleProvider` - Provider class implementing `LLMProvider`
  - `completion(request)` / `completion_stream(request)` - Get a completion, JSON or incremental SSE
  - `count_tokens(request)` - Count tokens for OpenAI
- `call_openai_compatible_chat(request)` - OpenAI API calls
- `count_openai_tokens(request)` - Token estimation
- `get_openai_compatible_client()` - HTTP session setup
- `convert_to_openai_messages()` - Claude → OpenAI format
- `create_claude_response_from_openai()` - OpenAI → Claude format
- `build_openai_payload()` - Shared Chat Completions payload builder (sync + streaming)
- `post_streaming_completion()` - Shared streaming POST + error handling, used by this provider and Copilot
- `stream_openai_as_anthropic()` - OpenAI SSE chunks → Anthropic SSE events (incremental)
- `aggregate_openai_stream()` - Assemble a full response from an SSE stream
- `PROVIDER` - This provider's `ProviderEntry`, registered in `registry.py`

### GitHub Copilot Provider (`copilot_provider.py` + `copilot.py`)
- `CopilotProvider` - Provider class implementing `LLMProvider`
  - `completion(request)` / `completion_stream(request)` - via Copilot's OpenAI-compatible API
- `CopilotAuth` (`copilot.py`) - OAuth device flow + session token lifecycle
- `PROVIDER` - This provider's `ProviderEntry`, registered in `registry.py`
