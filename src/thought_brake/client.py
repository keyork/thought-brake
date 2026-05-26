import os
from typing import Any, cast

from dotenv import load_dotenv
from openai import OpenAI

from ._monitor import stream_and_monitor
from ._prefill import run_phase2
from ._utils import estimate_messages_tokens, estimate_text_tokens, get_usage
from .config import EarlyStopConfig
from .types import ChatMessage, ChatResponse, RequestMetrics, StopReason


def _sum_optional(a: int | None, b: int | None) -> int | None:
    if a is None and b is None:
        return None
    return (a or 0) + (b or 0)


def _sum_exact(a: int | None, b: int | None) -> int | None:
    if a is None or b is None:
        return None
    return a + b


def _usage_source(api_total: int | None, estimated_total: int | None) -> str:
    if api_total is not None:
        return "api"
    if estimated_total is not None:
        return "estimate"
    return "none"


class ThoughtBrakeClient:
    """LLM API client wrapper that applies early stopping to reasoning models.

    Reads credentials from environment variables (THOUGHT_BRAKE_* preferred,
    OPENAI_* as fallback):
        THOUGHT_BRAKE_API_KEY   / OPENAI_API_KEY
        THOUGHT_BRAKE_BASE_URL  / OPENAI_BASE_URL
        THOUGHT_BRAKE_MODEL     (required if not passed to constructor)
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        config: EarlyStopConfig | None = None,
    ) -> None:
        load_dotenv(override=False)

        resolved_key = (
            api_key
            or os.environ.get("THOUGHT_BRAKE_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        resolved_url = (
            base_url
            or os.environ.get("THOUGHT_BRAKE_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
        )
        resolved_model = (
            model
            or os.environ.get("THOUGHT_BRAKE_MODEL")
            or os.environ.get("OPENAI_MODEL")
        )
        if not resolved_model:
            raise ValueError(
                "Model name is required. "
                "Set THOUGHT_BRAKE_MODEL env var or pass model= to constructor."
            )

        self.model = resolved_model
        self.config = config or EarlyStopConfig.from_env()
        self._openai = OpenAI(api_key=resolved_key, base_url=resolved_url)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[ChatMessage],
        config: EarlyStopConfig | None = None,
        **api_kwargs: Any,
    ) -> ChatResponse:
        """Send messages and return a ChatResponse.

        Early stopping is transparent: the returned content is always the
        model's final answer, regardless of whether Phase 2 was triggered.

        Args:
            messages: Chat message list.
            config:   Per-call config override. Falls back to instance config.
            **api_kwargs: Extra params forwarded to the API (temperature, etc.).
        """
        cfg = config or self.config

        if not cfg.enable:
            return self._passthrough(messages, **api_kwargs)

        # Phase 1 --------------------------------------------------------
        phase1 = stream_and_monitor(self._openai, self.model, messages, cfg, **api_kwargs)
        estimated_phase1_prompt_tokens = estimate_messages_tokens(messages)
        estimated_phase1_completion_tokens = estimate_text_tokens(
            phase1.reasoning + phase1.content
        )
        estimated_phase1_total_tokens = (
            estimated_phase1_prompt_tokens + estimated_phase1_completion_tokens
        )
        metrics = RequestMetrics(
            reasoning_chars=len(phase1.reasoning),
            stop_reason=phase1.stop_reason,
            stop_detail=phase1.stop_detail,
            phase1_prompt_tokens=phase1.usage.prompt_tokens,
            phase1_completion_tokens=phase1.usage.completion_tokens,
            phase1_total_tokens=phase1.usage.total_tokens,
            phase1_reasoning_tokens=phase1.usage.reasoning_tokens,
            estimated_phase1_prompt_tokens=estimated_phase1_prompt_tokens,
            estimated_phase1_completion_tokens=estimated_phase1_completion_tokens,
            estimated_phase1_total_tokens=estimated_phase1_total_tokens,
            estimated_total_tokens=estimated_phase1_total_tokens,
        )
        metrics.total_prompt_tokens = phase1.usage.prompt_tokens
        metrics.total_completion_tokens = phase1.usage.completion_tokens
        metrics.total_tokens = phase1.usage.total_tokens
        metrics.total_reasoning_tokens = phase1.usage.reasoning_tokens
        metrics.token_usage_source = _usage_source(
            phase1.usage.total_tokens, estimated_phase1_total_tokens
        )

        if phase1.stop_reason == StopReason.NATURAL:
            return ChatResponse(
                content=phase1.content,
                reasoning=phase1.reasoning,
                metrics=metrics,
            )

        # Phase 2 --------------------------------------------------------
        metrics.phase2_used = True
        try:
            phase2 = run_phase2(
                self._openai, self.model, messages, phase1.reasoning, cfg, **api_kwargs
            )
            content = phase2.content
            metrics.phase2_prompt_tokens = phase2.usage.prompt_tokens
            metrics.phase2_completion_tokens = phase2.usage.completion_tokens
            metrics.phase2_total_tokens = phase2.usage.total_tokens
            metrics.phase2_reasoning_tokens = phase2.usage.reasoning_tokens
            metrics.estimated_phase2_prompt_tokens = phase2.estimated_prompt_tokens
            metrics.estimated_phase2_completion_tokens = phase2.estimated_completion_tokens
            metrics.estimated_phase2_total_tokens = phase2.estimated_total_tokens
            metrics.estimated_total_tokens = _sum_optional(
                estimated_phase1_total_tokens, phase2.estimated_total_tokens
            )
            metrics.total_prompt_tokens = _sum_exact(
                phase1.usage.prompt_tokens, phase2.usage.prompt_tokens
            )
            metrics.total_completion_tokens = _sum_exact(
                phase1.usage.completion_tokens, phase2.usage.completion_tokens
            )
            metrics.total_tokens = _sum_exact(
                phase1.usage.total_tokens, phase2.usage.total_tokens
            )
            metrics.total_reasoning_tokens = _sum_exact(
                phase1.usage.reasoning_tokens, phase2.usage.reasoning_tokens
            )
            metrics.token_usage_source = _usage_source(
                metrics.total_tokens, metrics.estimated_total_tokens
            )
            if not content:
                raise RuntimeError("Phase 2 returned empty content")
        except Exception:
            metrics.phase2_failed = True
            if cfg.fallback_on_phase2_fail:
                content = self._fallback(messages, phase1.reasoning, cfg, **api_kwargs)
            else:
                content = phase1.content  # may be empty

        return ChatResponse(
            content=content,
            reasoning=phase1.reasoning,
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _passthrough(self, messages: list[ChatMessage], **api_kwargs: Any) -> ChatResponse:
        create = cast(Any, self._openai.chat.completions.create)
        resp = create(
            model=self.model,
            messages=messages,
            **api_kwargs,
        )
        usage = get_usage(resp)
        estimated_prompt_tokens = estimate_messages_tokens(messages)
        estimated_completion_tokens = estimate_text_tokens(resp.choices[0].message.content or "")
        estimated_total_tokens = estimated_prompt_tokens + estimated_completion_tokens
        return ChatResponse(
            content=resp.choices[0].message.content or "",
            reasoning="",
            metrics=RequestMetrics(
                stop_reason=StopReason.NATURAL,
                phase1_prompt_tokens=usage.prompt_tokens if usage else None,
                phase1_completion_tokens=usage.completion_tokens if usage else None,
                phase1_total_tokens=usage.total_tokens if usage else None,
                total_prompt_tokens=usage.prompt_tokens if usage else None,
                total_completion_tokens=usage.completion_tokens if usage else None,
                total_tokens=usage.total_tokens if usage else None,
                estimated_phase1_prompt_tokens=estimated_prompt_tokens,
                estimated_phase1_completion_tokens=estimated_completion_tokens,
                estimated_phase1_total_tokens=estimated_total_tokens,
                estimated_total_tokens=estimated_total_tokens,
                token_usage_source=_usage_source(
                    usage.total_tokens if usage else None, estimated_total_tokens
                ),
            ),
        )

    def _fallback(
        self,
        messages: list[ChatMessage],
        partial_reasoning: str,
        config: EarlyStopConfig,
        **api_kwargs: Any,
    ) -> str:
        """Last-resort: ask the model to summarise the partial reasoning."""
        excerpt = partial_reasoning[: config.fallback_excerpt_chars].rstrip()
        fallback_messages = [
            *messages,
            {
                "role": "assistant",
                "content": config.fallback_assistant_template.format(reasoning=excerpt),
            },
            {"role": "user", "content": config.final_answer_prompt},
        ]
        create = cast(Any, self._openai.chat.completions.create)
        resp = create(
            model=self.model,
            messages=fallback_messages,
            **api_kwargs,
        )
        return resp.choices[0].message.content or ""
