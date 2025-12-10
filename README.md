# Claude Codex

Monitor and intercept Claude Code API requests with multi-backend LLM support.

A hackable Claude API proxy for monitoring AI agent requests and connecting multiple LLM backends. No complex AI frameworks - just FastAPI, requests, and clean code you can easily modify.

**NEW: LLM Council Mode** - Query multiple models simultaneously, have them review each other's responses, and synthesize a superior answer. Inspired by [Karpathy's llm-council](https://github.com/karpathy/llm-council).

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
export LLM_BACKEND=bedrock
claudecodex
```

### 3. Connect Claude Code

```bash
export ANTHROPIC_BASE_URL=http://localhost:8082
```

That's it! Claude Code will now send requests through your proxy.

---

## What You Can Do

- **Monitor requests** - See all Claude Code API calls in real-time with color-coded logs
- **Switch backends** - Run the same workflow on Claude, GPT-4, Gemini, or local models
- **Debug workflows** - Watch which tools get called and when
- **Extend capabilities** - Simple translation layer makes adding new backends straightforward

## Architecture

```
┌─────────────┐    ANTHROPIC_BASE_URL=localhost:8082    ┌─────────────────┐
│ Claude Code │ ──────────────────────────────────────► │  Claude Codex   │
└─────────────┘            POST /v1/messages            │   (server.py)   │
                                                        └─────────────────┘
                                                                  │
                                                    Auto-Detection │
                                                   (env vars/keys) │
                                                                  │
                           ┌──────────────────────────────────────┼──────────────────────────────────────┐
                           │                                      │                                      │
                           ▼                                      ▼                                      ▼
                 ┌─────────────────┐                   ┌─────────────────┐                   ┌─────────────────┐
                 │  AWS Bedrock    │                   │OpenAI Compatible│                   │  Future Backend │
                 │  (bedrock.py)   │                   │  (openai_compat │                   │                 │
                 │                 │                   │      .py)       │                   │                 │
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
2. Claude Codex auto-detects backend (Bedrock vs OpenAI-compatible)
3. Claude Codex translates Claude API ↔ Provider API formats
4. Response flows back to Claude Code

**Backend Support:**
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

### Secondary: LLM Backend Flexibility
- **Connect any LLM** - swap between Claude, GPT-4, Gemini, local models instantly
- **Extend time limits** - run longer workflows by routing through your own infrastructure  
- **Add new backends** - simple translator pattern makes adding new LLMs easy
- **Custom model access** - use fine-tuned models or experimental endpoints

## Why Build This?

As developers working with AI agents, we often wonder:
- What requests is Claude Code actually making?
- How does it decide which tools to call?
- Can I run the same workflow on a different/cheaper model?
- How can I extend session time limits?

This proxy gives you **full visibility and control** without complex AI frameworks.

## Backends

- **AWS Bedrock**: Claude Sonnet/Haiku/Opus (`LLM_BACKEND=bedrock`)
- **Google Gemini**: gemini-2.0-flash (`LLM_BACKEND=openai_compatible` + Gemini config)
- **OpenAI**: GPT-4/3.5 (`LLM_BACKEND=openai_compatible` + OpenAI config)
- **Local**: Ollama, LM Studio (`LLM_BACKEND=openai_compatible` + local config)
- **Council Mode**: Multi-model fusion (`LLM_BACKEND=council`) - see below

---

## LLM Council Mode (Model Fusion)

Query multiple LLMs simultaneously, have them peer-review each other, and synthesize a superior response. This implements a 3-stage consensus mechanism inspired by [Karpathy's llm-council](https://github.com/karpathy/llm-council).

### How It Works

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         STAGE 1: RESPOND                                │
│   Query all council members in parallel                                 │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐                   │
│   │ Gemini  │  │  GPT-4  │  │ Claude  │  │ Llama   │                   │
│   └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘                   │
│        │            │            │            │                         │
│        ▼            ▼            ▼            ▼                         │
│   Response A   Response B   Response C   Response D                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    STAGE 2: REVIEW (Anonymized)                         │
│   Each model reviews and ranks other responses                          │
│   (Identities hidden: "Response A, B, C, D")                            │
│                                                                         │
│   Rankings collected → Aggregate scores calculated                      │
│   Best insights extracted from all reviews                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      STAGE 3: SYNTHESIZE                                │
│   Chairman model receives:                                              │
│   • All original responses                                              │
│   • Aggregate rankings                                                  │
│   • Key insights from reviews                                           │
│                                                                         │
│   Chairman synthesizes → FINAL SUPERIOR RESPONSE                        │
└─────────────────────────────────────────────────────────────────────────┘
```

### Quick Start (Council Mode)

```bash
# Configure your models
export COUNCIL_MODELS=gemini-2.0-flash,gpt-4o,claude-sonnet-4
export COUNCIL_CHAIRMAN=claude-sonnet-4
export COUNCIL_MODE=full  # full | fast | race

# API keys for each provider
export GEMINI_API_KEY=your-gemini-key
export OPENAI_API_KEY=your-openai-key
# (Claude via Bedrock uses AWS credentials)

# Run
claudecodex
```

### Council Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `full` | All 3 stages: respond → review → synthesize | Best quality, most thorough |
| `fast` | Skip review: respond → synthesize directly | Faster, still multi-model fusion |
| `race` | First successful response wins | Lowest latency, automatic failover |

### Model Specification

Models can be specified in `COUNCIL_MODELS` using these formats:

```bash
# Auto-detect provider from model name
COUNCIL_MODELS=gemini-2.0-flash,gpt-4o,claude-sonnet-4

# Explicit provider prefix
COUNCIL_MODELS=google:gemini-2.0-flash,openai:gpt-4o,bedrock:claude-sonnet-4

# Mix of local and cloud models
COUNCIL_MODELS=ollama:llama3.2,gemini-2.0-flash,gpt-4o
```

Supported providers:
- `openai` / `gpt` → OpenAI API
- `gemini` / `google` → Google Gemini (via OpenAI-compatible)
- `anthropic` / `claude` / `bedrock` → AWS Bedrock
- `ollama` / `local` → Local models (Ollama, LM Studio)

### Why Use Council Mode?

- **Higher accuracy** - Multiple perspectives catch errors and hallucinations
- **Better reasoning** - Synthesis combines the best insights from each model
- **Automatic failover** - If one model fails, others continue
- **Model comparison** - See how different models approach the same problem
- **Reduced bias** - Anonymized review prevents model favoritism

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
# Backend selection
LLM_BACKEND=bedrock|openai_compatible|council  # default: auto-detect
SERVER_PORT=8082                               # default: 8082

# Bedrock backend
AWS_PROFILE=your-profile               # default: saml
AWS_DEFAULT_REGION=us-east-1           # default: us-east-1
BEDROCK_MODEL_ID=model-id              # default: us.anthropic.claude-sonnet-4-*

# OpenAI-compatible backend
OPENAICOMPATIBLE_API_KEY=your-key      # required for openai_compatible
OPENAICOMPATIBLE_BASE_URL=endpoint-url # provider-specific
OPENAI_MODEL=model-name                # default: gemini-2.0-flash

# Council mode (multi-model fusion)
COUNCIL_MODELS=model1,model2,model3    # comma-separated list
COUNCIL_CHAIRMAN=model-name            # synthesis model (default: first in list)
COUNCIL_MODE=full|fast|race            # default: full
COUNCIL_TIMEOUT=120                    # seconds per model (default: 120)
GEMINI_API_KEY=your-key                # for Gemini models in council
OPENAI_API_KEY=your-key                # for OpenAI models in council

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
│   ├── bedrock.py                    # AWS Bedrock backend
│   ├── openai_compatible.py          # OpenAI-compatible backend
│   ├── council.py                    # LLM Council (multi-model fusion)
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

**Simple Architecture**: Core functionality organized in focused files
- `server.py` - FastAPI server with backend routing and API endpoints
- `bedrock.py` - AWS Bedrock integration with Claude API translation
- `openai_compatible.py` - OpenAI-compatible API integration (OpenAI, Gemini, local models)
- `council.py` - Multi-model fusion with 3-stage consensus mechanism

## API Functions

### Server Functions (`server.py`)
- `get_backend_type()` - Determine backend from environment  
- `call_llm_service(request)` - Route requests to backends
- `count_llm_tokens(request)` - Token counting
- `get_backend_info()` - Runtime configuration info
- `create_message(request)` - `/v1/messages` endpoint
- `count_tokens(request)` - `/v1/messages/count_tokens` endpoint
- `health()` - `/health` endpoint

### Bedrock Backend (`bedrock.py`)
- `call_bedrock_converse(request)` - AWS Bedrock API calls
- `count_request_tokens(request)` - Bedrock token counting
- `get_bedrock_client()` - AWS client setup
- `convert_to_bedrock_messages()` - Claude → Bedrock format
- `create_claude_response()` - Bedrock → Claude format

### OpenAI-Compatible Backend (`openai_compatible.py`)
- `call_openai_compatible_chat(request)` - OpenAI API calls
- `count_openai_tokens(request)` - Token estimation
- `get_openai_compatible_client()` - HTTP session setup
- `convert_to_openai_messages()` - Claude → OpenAI format
- `create_claude_response_from_openai()` - OpenAI → Claude format

### Council Backend (`council.py`)
- `call_council(request)` - Execute 3-stage consensus process
- `collect_responses(config, request)` - Stage 1: parallel model queries
- `conduct_reviews(config, request, responses)` - Stage 2: anonymous peer review
- `synthesize_response(config, request, responses, reviews)` - Stage 3: chairman synthesis
- `get_council_config()` - Parse council configuration from env vars
- `get_council_info()` - Runtime council info for health endpoint

