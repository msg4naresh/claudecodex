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

Requires **Python 3.11+**. Bedrock also needs working AWS credentials; the OpenAI-compatible provider needs an API key for your chosen endpoint.

### 1. Install

```bash
git clone https://github.com/msg4naresh/claudecodex.git
cd claudecodex
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

**With AWS Bedrock:**
```bash
export LLM_PROVIDER=bedrock
claudecodex
```

**With Gemini:**
```bash
export OPENAICOMPATIBLE_API_KEY=your-gemini-key
export LLM_PROVIDER=openai_compatible
claudecodex
```

On first Copilot run, a GitHub device flow will prompt you to authorize in your browser. The token is saved to `~/.copilot_token` for future use.

### 3. Connect Claude Code

```bash
export ANTHROPIC_BASE_URL=http://localhost:8082
export ANTHROPIC_AUTH_TOKEN=dummy   # any non-empty string; skips the login prompt
```

That's it — Claude Code now sends requests through your proxy. Open [http://localhost:8082/dashboard](http://localhost:8082/dashboard) for a real-time view of requests, errors, latency, token usage, and tool calls.

## Troubleshooting

**`address already in use` on startup** — a previous `claudecodex` process is still holding the port. Find and stop it:

```bash
lsof -nP -iTCP:8082 -sTCP:LISTEN   # find what's using the port (swap 8082 for SERVER_PORT if changed)
kill $(lsof -t -nP -iTCP:8082 -sTCP:LISTEN)   # stop it
```

Then restart `claudecodex` normally.

## Why

- **See what Claude Code is actually doing** — intercept and log every request, response, and tool call in real time
- **Run the same workflow on any model** — swap between Claude, GPT, Gemini, or local models by changing one env var
- **Extend session limits** — route longer workflows through your own infrastructure

## Architecture

```
                       ┌───────────────────────────────┐
   Claude Code  ──────►│          Claude Codex         │
   (ANTHROPIC_BASE_URL)│           (server.py)         │
   POST /v1/messages   └───────────────┬───────────────┘
                                       │ get_provider()  (Protocol-based)
                   ┌───────────────────┼───────────────────┐
                   ▼                   ▼                   ▼
          ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
          │ BedrockProvider │ │ OpenAICompatible│ │  Your Custom    │
          │  (bedrock.py)   │ │    Provider     │ │    Provider     │
          │  LLMProvider    │ │  LLMProvider    │ │  LLMProvider    │
          └────────┬────────┘ └────────┬────────┘ └────────┬────────┘
                   ▼                   ▼                   ▼
          ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
          │  AWS Bedrock    │ │   OpenAI API    │ │    Any LLM      │
          │  Converse API   │ │ Chat Completions│ │    Provider     │
          └─────────────────┘ └─────────────────┘ └─────────────────┘
```

Claude Code sends requests to Claude Codex, `server.py` routes them to a provider instance via `get_provider()`, and the provider translates Claude API ↔ provider API formats in both directions. Providers register themselves in `registry.py`, so `server.py` never changes to add one.

> GitHub Copilot is its own provider (`LLM_PROVIDER=copilot`, with its own OAuth device flow), but it reuses the OpenAI-compatible translation layer under the hood — Copilot's API speaks the OpenAI Chat Completions format.

**Supported providers:**

| Provider | `LLM_PROVIDER` | Models |
|----------|----------------|--------|
| GitHub Copilot (default) | `copilot` | Claude Sonnet/Haiku/Opus, GPT models via your Copilot subscription |
| AWS Bedrock | `bedrock` | Claude Sonnet/Haiku/Opus (native Converse API) |
| Google Gemini | `openai_compatible` | gemini-2.0-flash |
| OpenAI | `openai_compatible` | GPT-4, GPT-3.5 |
| Local | `openai_compatible` | Ollama, LM Studio |

## Adding a New Provider

Adding a provider is one file plus a one-line registry edit — **`server.py` never changes.**

1. **Create `src/claudecodex/custom_provider.py`** with a class implementing the `LLMProvider` protocol (structural — no subclassing) and a module-level `PROVIDER` entry:
   ```python
   class CustomProvider:
       def completion(self, request: MessagesRequest) -> MessagesResponse: ...
       def count_tokens(self, request: TokenCountRequest) -> TokenCountResponse: ...
       # Optional: completion_stream(request) for incremental SSE;
       # providers without it fall back to replay streaming automatically.

   PROVIDER = ProviderEntry(
       name="custom",
       factory=CustomProvider,
       describe=lambda: {"provider": "custom", "model": "your-model-id"},  # shown at /, /health, /dashboard
       validate=lambda: True,  # False → claudecodex exits at startup with a clear error
   )
   ```

2. **Register it in `registry.py`** — the only other line that changes:
   ```python
   from claudecodex.custom_provider import PROVIDER as _custom
   _PROVIDERS = {p.name: p for p in (_bedrock, _openai_compatible, _copilot, _custom)}
   ```

Each provider converts Claude API ↔ its own API format in both directions — see `bedrock.py` and `openai_compatible.py` for the translation patterns.

## For Developers

This project uses `uv` for package management and `ruff` for linting and formatting.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment and install runtime + dev dependencies
uv venv
uv pip sync requirements-dev.txt
source .venv/bin/activate        # macOS/Linux  (Windows: .venv\Scripts\activate)

# Lint and format
ruff check src
ruff format src

# Install in editable mode
pip install -e ".[dev]"
```

### Testing

```bash
pytest tests/ -v                 # All tests
pytest tests/unit/ -v            # Unit tests only (fast)
pytest tests/integration/ -v     # Integration tests (requires API keys)
RUN_E2E=1 pytest tests/e2e/ -v   # Provider/model matrix (real API calls; skips without creds)
```

## Configuration

Set these via environment variables or a `.env` file. **Do not commit `.env`** — add it to `.gitignore`.

```bash
# Provider selection
LLM_PROVIDER=bedrock|openai_compatible|copilot  # default: copilot
SERVER_PORT=8082                       # default: 8082
SERVER_HOST=127.0.0.1                  # default: 127.0.0.1 (localhost only; 0.0.0.0 exposes to network)

# GitHub Copilot provider
COPILOT_MODEL=claude-sonnet-4.6         # default: claude-sonnet-4.6 (also: gpt-4o, claude-haiku-4.5); if set, /model switching is a no-op
COPILOT_OAUTH_TOKEN=your-token         # optional: skip device flow
COPILOT_TOKEN_FILE=~/.copilot_token    # default: ~/.copilot_token

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

# Claude Code integration
ANTHROPIC_BASE_URL=http://localhost:8082
```

## File Structure

```
├── pyproject.toml                    # Packaging configuration
├── src/claudecodex/                  # Core package
│   ├── main.py                       # Entry point
│   ├── server.py                     # FastAPI server, routing & streaming
│   ├── provider.py                   # LLMProvider protocol + ProviderEntry
│   ├── registry.py                   # Provider registry — the only file that changes to add one
│   ├── bedrock.py                    # AWS Bedrock provider
│   ├── openai_compatible.py          # OpenAI-compatible provider + SSE translation
│   ├── copilot.py                    # GitHub Copilot auth (OAuth device flow)
│   ├── copilot_provider.py           # GitHub Copilot provider (streaming)
│   ├── dashboard.py                  # Live monitoring dashboard (/dashboard)
│   ├── models.py                     # Pydantic models
│   └── logging_config.py             # Monitoring & logging
├── tests/                            # unit/ (mocked), integration/ (real APIs), e2e/ (provider matrix)
├── logs/                             # Generated log files
├── requirements.txt                  # Runtime dependencies
└── requirements-dev.txt              # Runtime + pytest/ruff for development
```

## License

See [LICENSE](LICENSE).
