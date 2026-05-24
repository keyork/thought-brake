import os
from typing import Any, cast

from dotenv import load_dotenv
from openai import OpenAI

from ._monitor import stream_and_monitor
from ._prefill import run_prefill
from .config import EarlyStopConfig
from .types import ChatMessage, ChatResponse, RequestMetrics, StopReason


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
        metrics = RequestMetrics(
            reasoning_chars=len(phase1.reasoning),
            stop_reason=phase1.stop_reason,
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
            content = run_prefill(
                self._openai, self.model, messages, phase1.reasoning, cfg, **api_kwargs
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
        return ChatResponse(
            content=resp.choices[0].message.content or "",
            reasoning="",
            metrics=RequestMetrics(stop_reason=StopReason.NATURAL),
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
