from typing import Any

from .types import TokenUsage


def get_reasoning_content(delta: Any) -> str | None:
    """Extract reasoning_content from a streaming delta object.

    Tries direct attribute access first (some SDK builds patch this in),
    then falls back to model_extra (the openai SDK's dict for unknown fields).
    """
    val = getattr(delta, "reasoning_content", None)
    if val:
        return str(val)
    extra = getattr(delta, "model_extra", None)
    if extra and isinstance(extra, dict):
        v = extra.get("reasoning_content")
        return str(v) if v else None
    return None


def _get_value(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _get_nested_value(obj: Any, *names: str) -> Any:
    value = obj
    for name in names:
        value = _get_value(value, name)
        if value is None:
            return None
    return value


def get_usage(chunk_or_response: Any) -> TokenUsage | None:
    """Extract token usage from OpenAI-compatible response objects."""
    usage = _get_value(chunk_or_response, "usage")
    if usage is None:
        return None

    prompt_tokens = _get_value(usage, "prompt_tokens")
    completion_tokens = _get_value(usage, "completion_tokens")
    total_tokens = _get_value(usage, "total_tokens")
    reasoning_tokens = _get_nested_value(
        usage, "completion_tokens_details", "reasoning_tokens"
    )

    if (
        prompt_tokens is None
        and completion_tokens is None
        and total_tokens is None
        and reasoning_tokens is None
    ):
        return None

    return TokenUsage(
        prompt_tokens=int(prompt_tokens) if prompt_tokens is not None else None,
        completion_tokens=int(completion_tokens) if completion_tokens is not None else None,
        total_tokens=int(total_tokens) if total_tokens is not None else None,
        reasoning_tokens=int(reasoning_tokens) if reasoning_tokens is not None else None,
    )


def usage_kwargs(track_usage: bool, api_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return kwargs with stream usage enabled when requested."""
    if not track_usage:
        return api_kwargs
    stream_options = api_kwargs.get("stream_options", {})
    if not isinstance(stream_options, dict):
        raise TypeError("stream_options must be a dict when token usage tracking is enabled")
    return {
        **api_kwargs,
        "stream_options": {
            **stream_options,
            "include_usage": True,
        },
    }


def estimate_text_tokens(text: str) -> int:
    """Rough local token estimate used when streaming usage is unavailable."""
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    non_cjk = len(text) - cjk
    return max(1, cjk + (non_cjk + 3) // 4)


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        total += 4
        total += estimate_text_tokens(str(message.get("role", "")))
        total += estimate_text_tokens(str(message.get("content", "")))
    return total
