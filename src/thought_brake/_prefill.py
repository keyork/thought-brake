"""Phase 2: force the model to generate a final answer from partial reasoning."""

import re
from typing import Any, cast

from openai import OpenAI

from ._utils import (
    estimate_messages_tokens,
    estimate_text_tokens,
    get_reasoning_content,
    get_usage,
    usage_kwargs,
)
from .config import EarlyStopConfig
from .types import ChatMessage, Phase2Result, TokenUsage

_LEAK_START_RE = re.compile(
    r"^\s*(?:\d+[.)、]\s*)?(?:\*\*)?(?:分析请求|理解问题|识别问题|"
    r"Analyze the Request|Understand the Goal|Step\s+1|步骤\s*1)",
    re.IGNORECASE,
)
_FINAL_MARKER_RE = re.compile(
    r"(?:答案是|最终答案是|最终答案[:：]|Final answer[:：]|The answer is)",
    re.IGNORECASE,
)
_META_STEP_RE = re.compile(
    r"\d+[.)、]\s+\*{0,2}(?:生成|构建|输出|最终|对照|格式|润色|检查|验证|确认)"
    r"[^：:\n]{0,30}[：:]\s*\*{0,2}\s*",
)
_META_INSTRUCTION_RE = re.compile(
    r"^(?:\d+[.)、]\s+\*{0,2}(?:生成|构建|输出|最终|对照|格式|润色|检查|验证|确认)"
    r"[^：:\n]{0,30}[：:]|[*•]\s*(?:目标|是否|输出|不要|没有|严格|遵守))",
    re.MULTILINE,
)


def _deduplicate_tail(text: str) -> str:
    n = len(text)
    for length in range(n // 2, 0, -1):
        if text[n - 2 * length : n - length] == text[n - length :]:
            return text[: n - length]
    return text


def _extract_question(messages: list[ChatMessage]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return ""


def _render_conversation(messages: list[ChatMessage], limit: int) -> str:
    lines: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "unknown"))
        if role in {"system", "developer"}:
            continue
        content = str(msg.get("content", ""))
        if content:
            lines.append(f"{role}: {content}")

    rendered = "\n".join(lines).strip()
    if limit <= 0 or len(rendered) <= limit:
        return rendered
    return rendered[-limit:].lstrip()


def _preserved_control_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
    return [
        msg
        for msg in messages
        if msg.get("role") in {"system", "developer"}
    ]


def build_prefill_messages(
    original_messages: list[ChatMessage],
    partial_reasoning: str,
    config: EarlyStopConfig,
) -> list[ChatMessage]:
    """Construct messages for the prefill continuation mode."""
    prefill_content = (
        config.reasoning_start_tag
        + partial_reasoning
        + config.finalize_hint
        + config.reasoning_end_tag
        + config.phase2_answer_prefix
    )
    messages: list[ChatMessage] = [
        *original_messages,
        {"role": "assistant", "content": prefill_content},
    ]
    if not config.phase2_skip_user_prompt:
        messages.append({"role": "user", "content": config.final_answer_prompt})
    return messages


def _compose_reasoning_excerpt(
    text: str, head_chars: int, tail_chars: int, *, min_tail_threshold: int = 400
) -> str:
    n = len(text)
    if n <= head_chars:
        return text
    if tail_chars <= 0 or n <= head_chars + tail_chars + 30:
        return text[:head_chars]
    if n < min_tail_threshold:
        return text[:head_chars]
    head = text[:head_chars].rstrip()
    tail = text[n - tail_chars :].lstrip()
    return f"{head}\n……\n{tail}"


def build_direct_messages(
    original_messages: list[ChatMessage],
    partial_reasoning: str,
    config: EarlyStopConfig,
) -> list[ChatMessage]:
    question = _extract_question(original_messages)
    conversation = _render_conversation(
        original_messages, config.phase2_direct_conversation_chars
    )
    excerpt = _compose_reasoning_excerpt(
        partial_reasoning,
        config.phase2_direct_head_chars,
        config.phase2_direct_tail_chars,
    )
    user_content = config.phase2_direct_template.format(
        conversation=conversation, question=question, reasoning=excerpt
    )
    return [
        *_preserved_control_messages(original_messages),
        {"role": "user", "content": user_content},
    ]


def clean_final_answer(content: str) -> str:
    """Remove leaked reasoning drafts and meta-framework from Phase 2 output."""
    text = content.strip()
    if not text:
        return ""

    marker_matches = list(_FINAL_MARKER_RE.finditer(text))
    non_initial = [m for m in marker_matches if m.start() > 0]
    if non_initial:
        return _deduplicate_tail(text[non_initial[-1].start() :].strip())

    meta_matches = list(_META_STEP_RE.finditer(text))
    if meta_matches:
        after = text[meta_matches[-1].end() :]
        lines: list[str] = []
        for line in after.strip().splitlines():
            raw_line = line.strip()
            if _META_INSTRUCTION_RE.match(raw_line):
                continue
            cleaned = raw_line.lstrip("* ").strip()
            if cleaned:
                lines.append(cleaned)
        if lines:
            return _deduplicate_tail(" ".join(lines))

    if _LEAK_START_RE.search(text):
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        for paragraph in reversed(paragraphs):
            if not _LEAK_START_RE.search(paragraph):
                return _deduplicate_tail(paragraph)
        return _deduplicate_tail(text)

    return _deduplicate_tail(text)


def run_phase2(
    client: OpenAI,
    model: str,
    messages: list[ChatMessage],
    partial_reasoning: str,
    config: EarlyStopConfig,
    **api_kwargs: Any,
) -> Phase2Result:
    """Send the Phase 2 request and collect the model's final answer."""
    if config.phase2_mode == "direct":
        phase2_messages = build_direct_messages(messages, partial_reasoning, config)
    else:
        phase2_messages = build_prefill_messages(messages, partial_reasoning, config)
    estimated_prompt_tokens = estimate_messages_tokens(phase2_messages)

    content_parts: list[str] = []
    usage = None

    phase2_kwargs: dict[str, Any] = usage_kwargs(config.track_token_usage, {**api_kwargs})
    if config.phase2_temperature >= 0:
        phase2_kwargs["temperature"] = config.phase2_temperature
    if config.phase2_max_tokens > 0:
        phase2_kwargs["max_tokens"] = config.phase2_max_tokens
    if config.phase2_mode == "direct" and config.phase2_disable_thinking:
        existing_extra_body = phase2_kwargs.get("extra_body", {})
        if not isinstance(existing_extra_body, dict):
            raise TypeError("extra_body must be a dict when Phase 2 direct mode is enabled")
        phase2_extra_body = config.phase2_extra_body or {}
        if existing_extra_body or phase2_extra_body:
            phase2_kwargs["extra_body"] = {
                **existing_extra_body,
                **phase2_extra_body,
            }

    create = cast(Any, client.chat.completions.create)
    with create(
        model=model,
        messages=phase2_messages,
        stream=True,
        **phase2_kwargs,
    ) as stream:
        for chunk in stream:
            chunk_usage = get_usage(chunk)
            if chunk_usage is not None:
                usage = chunk_usage

            try:
                delta = chunk.choices[0].delta
            except (IndexError, AttributeError):
                continue

            content = getattr(delta, "content", None)
            if content:
                content_parts.append(content)
                continue

            leftover = get_reasoning_content(delta)
            if leftover:
                content_parts.append(leftover)

    raw = "".join(content_parts)
    content = clean_final_answer(raw) if config.clean_phase2_answer else raw
    estimated_completion_tokens = estimate_text_tokens(content)
    return Phase2Result(
        content=content,
        usage=usage or TokenUsage(),
        estimated_prompt_tokens=estimated_prompt_tokens,
        estimated_completion_tokens=estimated_completion_tokens,
        estimated_total_tokens=estimated_prompt_tokens + estimated_completion_tokens,
    )


def run_prefill(
    client: OpenAI,
    model: str,
    messages: list[ChatMessage],
    partial_reasoning: str,
    config: EarlyStopConfig,
    **api_kwargs: Any,
) -> str:
    """Backward-compatible wrapper that returns only the final answer."""
    return run_phase2(client, model, messages, partial_reasoning, config, **api_kwargs).content
