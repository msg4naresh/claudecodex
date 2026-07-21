"""Unit tests for the Bedrock Converse Stream -> Anthropic SSE translator.

Event shapes here were captured from a real `converse_stream` call against
a live AWS account (see stream_bedrock_as_anthropic's docstring) - Bedrock
processes content blocks strictly sequentially, unlike the OpenAI-compatible
path, so no index-accumulation is needed here.
"""

import json

import pytest

from claudecodex.bedrock import stream_bedrock_as_anthropic


class FakeBedrockStream:
    """Minimal stand-in for the boto3 EventStream returned by converse_stream."""

    def __init__(self, events, raise_after=None):
        self._events = events
        self._raise_after = raise_after
        self.closed = False

    def __iter__(self):
        for i, event in enumerate(self._events):
            if self._raise_after is not None and i >= self._raise_after:
                raise ConnectionError("stream dropped")
            yield event

    def close(self):
        self.closed = True


def _parse_events(raw_events):
    parsed = []
    for event in raw_events:
        lines = event.strip().split("\n")
        name = lines[0].split("event: ", 1)[1]
        data = json.loads(lines[1].split("data: ", 1)[1])
        parsed.append((name, data))
    return parsed


class TestStreamBedrockAsAnthropic:
    def test_text_streams_incrementally_with_synthesized_block_start(self):
        """Bedrock omits contentBlockStart for text blocks (confirmed live) -
        the translator must synthesize one so Anthropic clients still get it."""
        stream = FakeBedrockStream([
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"delta": {"text": "Hel"}, "contentBlockIndex": 0}},
            {"contentBlockDelta": {"delta": {"text": "lo"}, "contentBlockIndex": 0}},
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {"inputTokens": 5, "outputTokens": 2}}},
        ])
        events = _parse_events(stream_bedrock_as_anthropic(stream, "m"))

        names = [n for n, _ in events]
        assert names == [
            "message_start", "content_block_start",
            "content_block_delta", "content_block_delta",
            "content_block_stop", "message_delta", "message_stop",
        ]
        start = dict(events)["content_block_start"]
        assert start["content_block"] == {"type": "text", "text": ""}
        deltas = [d["delta"]["text"] for n, d in events if n == "content_block_delta"]
        assert deltas == ["Hel", "lo"]
        message_delta = dict(events)["message_delta"]
        assert message_delta["delta"]["stop_reason"] == "end_turn"
        assert message_delta["usage"]["output_tokens"] == 2
        assert stream.closed

    def test_parallel_tool_calls_sequential_blocks(self):
        """Real Bedrock behavior: block 0 fully completes before block 1
        starts, verified against a live account with two tool calls."""
        stream = FakeBedrockStream([
            {"messageStart": {"role": "assistant"}},
            {"contentBlockStart": {"start": {"toolUse": {
                "toolUseId": "tool_a", "name": "search"}}, "contentBlockIndex": 0}},
            {"contentBlockDelta": {"delta": {"toolUse": {"input": ""}},
                                    "contentBlockIndex": 0}},
            {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"q":"x"}'}},
                                    "contentBlockIndex": 0}},
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"contentBlockStart": {"start": {"toolUse": {
                "toolUseId": "tool_b", "name": "weather"}}, "contentBlockIndex": 1}},
            {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"city":"Paris"}'}},
                                    "contentBlockIndex": 1}},
            {"contentBlockStop": {"contentBlockIndex": 1}},
            {"messageStop": {"stopReason": "tool_use"}},
        ])
        events = _parse_events(stream_bedrock_as_anthropic(stream, "m"))

        starts = [d for n, d in events if n == "content_block_start"]
        assert len(starts) == 2
        assert [(s["content_block"]["id"], s["content_block"]["name"])
                for s in starts] == [("tool_a", "search"), ("tool_b", "weather")]
        assert [s["index"] for s in starts] == [0, 1]

        deltas = [d for n, d in events if n == "content_block_delta"]
        args_by_index = {d["index"]: d["delta"]["partial_json"] for d in deltas}
        assert args_by_index == {0: '{"q":"x"}', 1: '{"city":"Paris"}'}
        assert dict(events)["message_delta"]["delta"]["stop_reason"] == "tool_use"

    def test_stream_error_event_raises(self):
        stream = FakeBedrockStream([
            {"messageStart": {"role": "assistant"}},
            {"throttlingException": {"message": "rate limited"}},
        ])
        gen = stream_bedrock_as_anthropic(stream, "m")
        next(gen)  # message_start
        with pytest.raises(RuntimeError, match="rate limited"):
            list(gen)
        assert stream.closed

    def test_closed_on_mid_stream_failure(self):
        stream = FakeBedrockStream(
            [{"messageStart": {"role": "assistant"}}] * 3, raise_after=1
        )
        with pytest.raises(ConnectionError):
            list(stream_bedrock_as_anthropic(stream, "m"))
        assert stream.closed

    def test_closed_when_cancelled_after_message_start(self):
        stream = FakeBedrockStream([
            {"contentBlockStart": {"start": {}, "contentBlockIndex": 0}},
        ])
        gen = stream_bedrock_as_anthropic(stream, "m")
        next(gen)  # message_start
        gen.close()  # client cancels immediately
        assert stream.closed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
