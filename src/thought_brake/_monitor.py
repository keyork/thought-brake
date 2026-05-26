"""Phase 1: stream reasoning tokens and cut when budget is exceeded."""

from typing import Any, cast

from openai import OpenAI

from ._utils import get_reasoning_content, get_usage, usage_kwargs
from .config import EarlyStopConfig
from .detectors import ReasoningDetector, build_detector
from .types import ChatMessage, Phase1Result, StopReason, TokenUsage


def stream_and_monitor(
    client: OpenAI,
    model: str,
    messages: list[ChatMessage],
    config: EarlyStopConfig,
    detector: ReasoningDetector | None = None,
    **api_kwargs: Any,
) -> Phase1Result:
    """Stream the model and return when thinking ends or a budget is hit.

    Returns a Phase1Result with:
      - reasoning: all thinking text collected so far
      - content: answer text (non-empty only on NATURAL finish)
      - stop_reason: NATURAL | SOFT | HARD | INTERRUPTED
    """
    reasoning_buf: list[str] = []
    content_buf: list[str] = []
    reasoning_chars = 0
    stop_reason = StopReason.NATURAL
    stop_detail = ""
    active_detector = detector or build_detector(config)
    usage = None

    try:
        create = cast(Any, client.chat.completions.create)
        request_kwargs = usage_kwargs(config.track_token_usage, {**api_kwargs})
        with create(
            model=model,
            messages=messages,
            stream=True,
            **request_kwargs,
        ) as stream:
            for chunk in stream:
                chunk_usage = get_usage(chunk)
                if chunk_usage is not None:
                    usage = chunk_usage

                try:
                    delta = chunk.choices[0].delta
                except (IndexError, AttributeError):
                    continue

                # Model transitioned to answer phase
                content_piece = getattr(delta, "content", None)
                if content_piece:
                    content_buf.append(content_piece)
                    continue

                reasoning_piece = get_reasoning_content(delta)
                if not reasoning_piece:
                    continue

                reasoning_buf.append(reasoning_piece)
                reasoning_chars += len(reasoning_piece)

                decision = active_detector.update(reasoning_piece, reasoning_chars)
                stop_detail = decision.detail
                if decision.should_stop:
                    stop_reason = decision.reason
                    break
    except Exception:
        if not reasoning_buf:
            raise
        # Network interruption or JSON parse failure: return whatever was collected.
        # Phase 2 will attempt to salvage the partial reasoning.
        stop_reason = StopReason.INTERRUPTED
        stop_detail = "stream_interrupted"

    return Phase1Result(
        reasoning="".join(reasoning_buf),
        content="".join(content_buf),
        stop_reason=stop_reason,
        stop_detail=stop_detail,
        usage=usage or TokenUsage(),
    )
