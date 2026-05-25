"""Integration tests for ThoughtBrakeClient (all API calls mocked)."""

import os
from unittest.mock import MagicMock, patch

import pytest

from thought_brake import EarlyStopConfig, StopReason, ThoughtBrakeClient

from .conftest import FailingStream, MockStream, content_chunk, reasoning_chunk, usage_chunk


def _make_client(cfg: EarlyStopConfig | None = None) -> ThoughtBrakeClient:
    client = ThoughtBrakeClient(
        api_key="test-key",
        base_url="http://localhost",
        model="test-model",
        config=cfg,
    )
    client._openai = MagicMock()
    return client


def _attach_stream(client: ThoughtBrakeClient, chunks) -> None:
    client._openai.chat.completions.create.return_value = MockStream(chunks)


def test_natural_finish_no_phase2() -> None:
    client = _make_client()
    chunks = [reasoning_chunk("thinking。"), content_chunk("answer")]
    _attach_stream(client, chunks)

    resp = client.chat([{"role": "user", "content": "hi"}])

    assert resp.content == "answer"
    assert resp.metrics.stop_reason == StopReason.NATURAL
    assert not resp.metrics.phase2_used
    # Only one API call (Phase 1)
    assert client._openai.chat.completions.create.call_count == 1


def test_soft_truncation_triggers_phase2() -> None:
    cfg = EarlyStopConfig(soft_budget=5, hard_limit=100)
    client = _make_client(cfg)

    phase1_chunks = [reasoning_chunk("12345。"), reasoning_chunk("extra")]
    phase2_chunks = [content_chunk("phase2 answer")]

    client._openai.chat.completions.create.side_effect = [
        MockStream(phase1_chunks),
        MockStream(phase2_chunks),
    ]

    resp = client.chat([{"role": "user", "content": "q"}])

    assert resp.content == "phase2 answer"
    assert resp.metrics.stop_reason == StopReason.SOFT
    assert resp.metrics.phase2_used
    assert not resp.metrics.phase2_failed
    assert client._openai.chat.completions.create.call_count == 2


def test_interrupted_phase1_with_partial_reasoning_triggers_phase2() -> None:
    client = _make_client()

    client._openai.chat.completions.create.side_effect = [
        FailingStream([reasoning_chunk("partial reasoning")]),
        MockStream([content_chunk("phase2 answer")]),
    ]

    resp = client.chat([{"role": "user", "content": "q"}])

    assert resp.content == "phase2 answer"
    assert resp.metrics.stop_reason == StopReason.INTERRUPTED
    assert resp.metrics.phase2_used
    assert not resp.metrics.phase2_failed


def test_phase2_failure_triggers_fallback() -> None:
    cfg = EarlyStopConfig(soft_budget=5, hard_limit=100, fallback_on_phase2_fail=True)
    client = _make_client(cfg)

    phase1_chunks = [reasoning_chunk("12345。")]

    fallback_resp = MagicMock()
    fallback_resp.choices[0].message.content = "fallback answer"

    client._openai.chat.completions.create.side_effect = [
        MockStream(phase1_chunks),
        RuntimeError("prefill rejected"),   # Phase 2 fails
        fallback_resp,                       # fallback non-streaming call
    ]

    resp = client.chat([{"role": "user", "content": "q"}])

    assert resp.content == "fallback answer"
    assert resp.metrics.phase2_failed


def test_token_usage_metrics_are_combined_across_phases() -> None:
    cfg = EarlyStopConfig(
        soft_budget=5,
        hard_limit=100,
        track_token_usage=True,
    )
    client = _make_client(cfg)

    phase1_chunks = [
        usage_chunk(10, 20, 30, reasoning_tokens=18),
        reasoning_chunk("12345。"),
    ]
    phase2_chunks = [
        content_chunk("phase2 answer"),
        usage_chunk(40, 5, 45, reasoning_tokens=0),
    ]
    client._openai.chat.completions.create.side_effect = [
        MockStream(phase1_chunks),
        MockStream(phase2_chunks),
    ]

    resp = client.chat([{"role": "user", "content": "q"}])

    assert resp.metrics.phase1_total_tokens == 30
    assert resp.metrics.phase2_total_tokens == 45
    assert resp.metrics.total_prompt_tokens == 50
    assert resp.metrics.total_completion_tokens == 25
    assert resp.metrics.total_tokens == 75
    assert resp.metrics.total_reasoning_tokens == 18


def test_fallback_prompt_uses_configured_template() -> None:
    cfg = EarlyStopConfig(
        soft_budget=5,
        hard_limit=100,
        fallback_excerpt_chars=7,
        fallback_assistant_template="partial={reasoning}",
        fallback_user_prompt="answer only",
    )
    client = _make_client(cfg)

    fallback_resp = MagicMock()
    fallback_resp.choices[0].message.content = "fallback answer"

    client._openai.chat.completions.create.side_effect = [
        MockStream([reasoning_chunk("1234567890。")]),
        RuntimeError("prefill rejected"),
        fallback_resp,
    ]

    client.chat([{"role": "user", "content": "q"}])

    fallback_messages = client._openai.chat.completions.create.call_args.kwargs["messages"]
    assert fallback_messages[-2] == {"role": "assistant", "content": "partial=1234567"}
    assert fallback_messages[-1] == {"role": "user", "content": cfg.final_answer_prompt}


def test_early_stop_disabled_passthrough() -> None:
    cfg = EarlyStopConfig(enable=False)
    client = _make_client(cfg)

    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = "direct answer"
    client._openai.chat.completions.create.return_value = mock_resp

    resp = client.chat([{"role": "user", "content": "q"}])

    assert resp.content == "direct answer"
    assert not resp.metrics.phase2_used
    # Called without stream=True
    call_kwargs = client._openai.chat.completions.create.call_args.kwargs
    assert call_kwargs.get("stream") is not True


def test_missing_model_raises() -> None:
    with patch.dict(os.environ, {}, clear=True):
        with patch("thought_brake.client.load_dotenv"):  # prevent .env from loading
            with pytest.raises(ValueError, match="Model name is required"):
                ThoughtBrakeClient(api_key="k", base_url="u")
