from typing import Any


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
