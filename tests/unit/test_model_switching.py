"""Unit tests for Claude Code /model switching support.

Covers the shared family-detection helper and its per-provider resolution:
Copilot (hardcoded, verified against the live API) and Bedrock (opt-in via
per-family env vars, since real inference-profile IDs vary by AWS account).
"""

import pytest

from claudecodex.provider import detect_model_family


class TestDetectModelFamily:
    def test_detects_known_families(self):
        assert detect_model_family("claude-opus-4-6") == "opus"
        assert detect_model_family("claude-sonnet-4-5-20250929") == "sonnet"
        assert detect_model_family("claude-haiku-4-5-20251001-v1:0") == "haiku"

    def test_case_insensitive(self):
        assert detect_model_family("Claude-Sonnet-4-6") == "sonnet"

    def test_unknown_or_missing_returns_none(self):
        assert detect_model_family("gpt-4o") is None
        assert detect_model_family("") is None
        assert detect_model_family(None) is None


class TestCopilotModelSwitching:
    def test_family_mapping_when_no_explicit_pin(self, monkeypatch):
        from claudecodex.copilot_provider import get_copilot_model

        monkeypatch.delenv("COPILOT_MODEL", raising=False)
        assert get_copilot_model("claude-opus-4-6") == "claude-opus-4.8"
        assert get_copilot_model("claude-sonnet-4-5") == "claude-sonnet-4.6"
        assert get_copilot_model("claude-haiku-4-5-20251001-v1:0") == "claude-haiku-4.5"

    def test_unrecognized_family_falls_back_to_sonnet(self, monkeypatch):
        from claudecodex.copilot_provider import get_copilot_model

        monkeypatch.delenv("COPILOT_MODEL", raising=False)
        assert get_copilot_model("gpt-4o") == "claude-sonnet-4.6"
        assert get_copilot_model(None) == "claude-sonnet-4.6"

    def test_explicit_env_pin_always_wins(self, monkeypatch):
        from claudecodex.copilot_provider import get_copilot_model

        monkeypatch.setenv("COPILOT_MODEL", "gpt-4o")
        assert get_copilot_model("claude-opus-4-6") == "gpt-4o"


class TestBedrockModelSwitching:
    def test_family_mapping_when_no_explicit_pin(self, monkeypatch):
        """Verified against `aws bedrock list-inference-profiles` for a
        real account: these cross-region profile IDs are standardized."""
        from claudecodex.bedrock import get_model_id

        monkeypatch.delenv("BEDROCK_MODEL_ID", raising=False)
        monkeypatch.delenv("BEDROCK_MODEL_ID_OPUS", raising=False)
        monkeypatch.delenv("BEDROCK_MODEL_ID_SONNET", raising=False)
        monkeypatch.delenv("BEDROCK_MODEL_ID_HAIKU", raising=False)

        assert get_model_id("claude-opus-4-6") == "us.anthropic.claude-opus-4-8"
        assert get_model_id("claude-sonnet-4-5") == "us.anthropic.claude-sonnet-4-6"
        assert (get_model_id("claude-haiku-4-5-20251001-v1:0")
                == "us.anthropic.claude-haiku-4-5-20251001-v1:0")

    def test_unrecognized_family_falls_back_to_haiku(self, monkeypatch):
        from claudecodex.bedrock import get_model_id

        monkeypatch.delenv("BEDROCK_MODEL_ID", raising=False)
        monkeypatch.delenv("BEDROCK_MODEL_ID_OPUS", raising=False)
        monkeypatch.delenv("BEDROCK_MODEL_ID_SONNET", raising=False)
        monkeypatch.delenv("BEDROCK_MODEL_ID_HAIKU", raising=False)

        assert get_model_id("gpt-4o") == "us.anthropic.claude-haiku-4-5-20251001-v1:0"
        assert get_model_id(None) == "us.anthropic.claude-haiku-4-5-20251001-v1:0"

    def test_family_env_override_used_when_configured(self, monkeypatch):
        from claudecodex.bedrock import get_model_id

        monkeypatch.delenv("BEDROCK_MODEL_ID", raising=False)
        monkeypatch.setenv("BEDROCK_MODEL_ID_OPUS", "us.anthropic.claude-opus-custom:0")

        assert get_model_id("claude-opus-4-6") == "us.anthropic.claude-opus-custom:0"

    def test_explicit_pin_wins_over_family_override(self, monkeypatch):
        from claudecodex.bedrock import get_model_id

        monkeypatch.setenv("BEDROCK_MODEL_ID", "us.anthropic.claude-pinned:0")
        monkeypatch.setenv("BEDROCK_MODEL_ID_OPUS", "us.anthropic.claude-opus-custom:0")

        assert get_model_id("claude-opus-4-6") == "us.anthropic.claude-pinned:0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
