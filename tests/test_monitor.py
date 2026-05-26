"""Tests for Phase 1 stream monitoring."""

from unittest.mock import MagicMock

import pytest

from thought_brake._monitor import stream_and_monitor
from thought_brake.config import EarlyStopConfig
from thought_brake.types import StopReason

from .conftest import FailingStream, MockStream, content_chunk, reasoning_chunk, usage_chunk


def _mock_client(chunks):
    client = MagicMock()
    client.chat.completions.create.return_value = MockStream(chunks)
    return client


def test_natural_finish_with_content() -> None:
    chunks = [
        reasoning_chunk("Let me think. "),
        reasoning_chunk("OK got it。"),
        content_chunk("The answer is 42."),
    ]
    result = stream_and_monitor(_mock_client(chunks), "model", [], EarlyStopConfig())
    assert result.stop_reason == StopReason.NATURAL
    assert "42" in result.content
    assert "Let me think" in result.reasoning


def test_soft_truncation_at_sentence_boundary() -> None:
    cfg = EarlyStopConfig(soft_budget=10, hard_limit=200)
    # First chunk is under budget; second pushes over and ends with 。
    chunks = [
        reasoning_chunk("123456789"),      # 9 chars, under budget
        reasoning_chunk("0extra text。"),  # now > 10, ends with 。
        reasoning_chunk("this should not be collected"),
    ]
    result = stream_and_monitor(_mock_client(chunks), "model", [], cfg)
    assert result.stop_reason == StopReason.SOFT
    assert result.stop_detail == "soft_budget=10"
    assert result.content == ""
    assert "this should not be collected" not in result.reasoning


def test_hard_truncation() -> None:
    cfg = EarlyStopConfig(soft_budget=5, hard_limit=10)
    # Soft budget is 5 but no sentence-ending punctuation until after hard limit
    chunks = [
        reasoning_chunk("12345"),      # 5 chars, hits soft_budget but no punctuation
        reasoning_chunk("678901"),     # total 11 chars → hard limit
        reasoning_chunk("should not appear"),
    ]
    result = stream_and_monitor(_mock_client(chunks), "model", [], cfg)
    assert result.stop_reason == StopReason.HARD
    assert result.stop_detail == "hard_limit=10"
    assert "should not appear" not in result.reasoning


def test_no_reasoning_content_natural() -> None:
    chunks = [content_chunk("Direct answer.")]
    result = stream_and_monitor(_mock_client(chunks), "model", [], EarlyStopConfig())
    assert result.stop_reason == StopReason.NATURAL
    assert result.content == "Direct answer."
    assert result.reasoning == ""


def test_no_stop_detector_collects_full_baseline_reasoning() -> None:
    cfg = EarlyStopConfig(detector="none", soft_budget=5, hard_limit=10)
    chunks = [
        reasoning_chunk("12345"),
        reasoning_chunk("678901"),
        content_chunk("answer"),
    ]

    result = stream_and_monitor(_mock_client(chunks), "model", [], cfg)

    assert result.stop_reason == StopReason.NATURAL
    assert result.reasoning == "12345678901"
    assert result.content == "answer"


def test_stream_usage_is_collected_when_enabled() -> None:
    cfg = EarlyStopConfig(track_token_usage=True)
    client = _mock_client(
        [
            reasoning_chunk("think"),
            content_chunk("answer"),
            usage_chunk(10, 20, 30, reasoning_tokens=15),
        ]
    )

    result = stream_and_monitor(client, "model", [], cfg)

    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert call_kwargs["stream_options"] == {"include_usage": True}
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 20
    assert result.usage.total_tokens == 30
    assert result.usage.reasoning_tokens == 15


def test_initial_network_error_propagates() -> None:
    def _bad_create(**_):
        raise ConnectionError("network down")

    client = MagicMock()
    client.chat.completions.create.side_effect = _bad_create

    with pytest.raises(ConnectionError, match="network down"):
        stream_and_monitor(client, "model", [], EarlyStopConfig())


def test_stream_error_after_partial_reasoning_is_interrupted() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = FailingStream([reasoning_chunk("partial")])

    result = stream_and_monitor(client, "model", [], EarlyStopConfig())
    assert result.stop_reason == StopReason.INTERRUPTED
    assert result.stop_detail == "stream_interrupted"
    assert result.reasoning == "partial"
    assert result.content == ""
