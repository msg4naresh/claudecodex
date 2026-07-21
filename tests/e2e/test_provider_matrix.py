"""End-to-end provider/model matrix tests using real API calls.

Runs every (provider, model) pair in PROVIDER_MODELS through the full
FastAPI stack: completion, tool use, streaming, and token counting.

Gated behind RUN_E2E=1 so regular test runs stay fast and offline:

    RUN_E2E=1 pytest tests/e2e/ -v

Providers whose credentials are missing are skipped automatically:
- copilot: needs ~/.copilot_token or COPILOT_OAUTH_TOKEN
- bedrock: needs a resolvable AWS profile (AWS_PROFILE, default: saml)
- openai_compatible: needs OPENAICOMPATIBLE_API_KEY

To cover more models, add entries to PROVIDER_MODELS.
"""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from claudecodex.server import app

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_E2E") != "1", reason="set RUN_E2E=1 to run e2e tests"
)

client = TestClient(app)

# (provider, model env var, model). model=None uses the provider default.
PROVIDER_MODELS = [
    ("copilot", "COPILOT_MODEL", "claude-sonnet-4.6"),
    ("copilot", "COPILOT_MODEL", "claude-sonnet-4.5"),
    ("copilot", "COPILOT_MODEL", "claude-haiku-4.5"),
    ("copilot", "COPILOT_MODEL", "gpt-4o"),
    ("bedrock", "BEDROCK_MODEL_ID", None),
    ("openai_compatible", "OPENAI_MODEL", "gemini-2.0-flash"),
]

MATRIX_IDS = [f"{p}-{m or 'default'}" for p, _, m in PROVIDER_MODELS]

# First matrix entry per provider, for the per-provider tests.
_seen = set()
PROVIDER_FIRST = [
    entry for entry in PROVIDER_MODELS
    if entry[0] not in _seen and not _seen.add(entry[0])
]
FIRST_IDS = [f"{p}-{m or 'default'}" for p, _, m in PROVIDER_FIRST]

WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get the current weather for a city",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
    }
}


def _has_credentials(provider: str) -> bool:
    if provider == "copilot":
        if os.environ.get("COPILOT_OAUTH_TOKEN"):
            return True
        token_file = os.environ.get("COPILOT_TOKEN_FILE", "~/.copilot_token")
        return Path(token_file).expanduser().exists()
    if provider == "bedrock":
        try:
            import boto3
            profile = os.environ.get("AWS_PROFILE", "saml")
            return boto3.Session(profile_name=profile).get_credentials() is not None
        except Exception:
            return False
    if provider == "openai_compatible":
        return bool(os.environ.get("OPENAICOMPATIBLE_API_KEY"))
    return False


def _select_provider(provider, env_var, model, monkeypatch):
    if not _has_credentials(provider):
        pytest.skip(f"no credentials for {provider}")
    monkeypatch.setenv("LLM_PROVIDER", provider)
    if model:
        monkeypatch.setenv(env_var, model)


@pytest.mark.parametrize("provider,env_var,model", PROVIDER_MODELS, ids=MATRIX_IDS)
def test_completion(provider, env_var, model, monkeypatch):
    """Every provider/model pair returns a non-empty text completion."""
    _select_provider(provider, env_var, model, monkeypatch)

    response = client.post("/v1/messages", json={
        "model": model or "e2e-test",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Reply with the single word PONG"}]
    })

    assert response.status_code == 200, response.text
    data = response.json()
    text_blocks = [b for b in data["content"] if b["type"] == "text"]
    assert text_blocks and text_blocks[0]["text"].strip(), data
    assert data["stop_reason"] in ("end_turn", "max_tokens")
    assert data["usage"]["output_tokens"] > 0


@pytest.mark.parametrize("provider,env_var,model", PROVIDER_FIRST, ids=FIRST_IDS)
def test_tool_use(provider, env_var, model, monkeypatch):
    """Each provider returns a tool_use block with stop_reason=tool_use."""
    _select_provider(provider, env_var, model, monkeypatch)

    response = client.post("/v1/messages", json={
        "model": model or "e2e-test",
        "max_tokens": 200,
        "messages": [{
            "role": "user",
            "content": "What is the weather in Paris? Use the get_weather tool."
        }],
        "tools": [WEATHER_TOOL]
    })

    assert response.status_code == 200, response.text
    data = response.json()
    tool_blocks = [b for b in data["content"] if b["type"] == "tool_use"]
    assert tool_blocks, f"no tool_use block: {data}"
    assert tool_blocks[0]["name"] == "get_weather"
    assert data["stop_reason"] == "tool_use"

    # Round-trip: send the tool result back and expect a final text answer
    followup = client.post("/v1/messages", json={
        "model": model or "e2e-test",
        "max_tokens": 200,
        "messages": [
            {
                "role": "user",
                "content": "What is the weather in Paris? Use the get_weather tool."
            },
            {"role": "assistant", "content": data["content"]},
            {"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": tool_blocks[0]["id"],
                "content": [{"type": "text", "text": "22C and sunny"}]
            }]},
        ],
        "tools": [WEATHER_TOOL]
    })

    assert followup.status_code == 200, followup.text
    followup_data = followup.json()
    final_text = " ".join(
        b["text"] for b in followup_data["content"] if b["type"] == "text"
    )
    assert final_text.strip(), followup_data
    assert followup_data["stop_reason"] == "end_turn"


@pytest.mark.parametrize("provider,env_var,model", PROVIDER_FIRST, ids=FIRST_IDS)
def test_streaming(provider, env_var, model, monkeypatch):
    """Each provider returns a well-formed Anthropic SSE event sequence."""
    _select_provider(provider, env_var, model, monkeypatch)

    response = client.post("/v1/messages", json={
        "model": model or "e2e-test",
        "max_tokens": 50,
        "stream": True,
        "messages": [{"role": "user", "content": "Reply with the single word PONG"}]
    })

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [
        line.split("event: ", 1)[1]
        for line in response.text.splitlines()
        if line.startswith("event: ")
    ]
    assert events[0] == "message_start"
    assert events[-1] == "message_stop"
    assert "content_block_delta" in events


@pytest.mark.parametrize("provider,env_var,model", PROVIDER_FIRST, ids=FIRST_IDS)
def test_count_tokens(provider, env_var, model, monkeypatch):
    """Each provider counts tokens for a simple request."""
    _select_provider(provider, env_var, model, monkeypatch)

    response = client.post("/v1/messages/count_tokens", json={
        "model": model or "e2e-test",
        "messages": [{"role": "user", "content": "Hello, how are you today?"}]
    })

    assert response.status_code == 200, response.text
    assert response.json()["input_tokens"] > 0
