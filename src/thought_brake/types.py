from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

ChatMessage = dict[str, Any]
DetectorName = Literal[
    "budget",
    "compression",
    "keyword",
    "ngram",
    "semantic",
    "bocpd",
    "none",
]
Phase2Mode = Literal["prefill", "direct"]


class StopReason(StrEnum):
    NATURAL = "natural"  # model finished thinking on its own
    SOFT = "soft"  # exceeded soft_budget, cut at sentence boundary
    HARD = "hard"  # exceeded hard_limit, forced cut
    INTERRUPTED = "interrupted"  # stream failed after partial reasoning was collected


@dataclass
class TokenUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None


@dataclass
class Phase1Result:
    reasoning: str
    content: str  # non-empty only when stop_reason is NATURAL
    stop_reason: StopReason
    stop_detail: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass
class Phase2Result:
    content: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    estimated_prompt_tokens: int | None = None
    estimated_completion_tokens: int | None = None
    estimated_total_tokens: int | None = None


@dataclass
class RequestMetrics:
    reasoning_chars: int = 0
    stop_reason: StopReason = StopReason.NATURAL
    stop_detail: str = ""
    phase2_used: bool = False
    phase2_failed: bool = False
    phase1_prompt_tokens: int | None = None
    phase1_completion_tokens: int | None = None
    phase1_total_tokens: int | None = None
    phase1_reasoning_tokens: int | None = None
    phase2_prompt_tokens: int | None = None
    phase2_completion_tokens: int | None = None
    phase2_total_tokens: int | None = None
    phase2_reasoning_tokens: int | None = None
    total_prompt_tokens: int | None = None
    total_completion_tokens: int | None = None
    total_tokens: int | None = None
    total_reasoning_tokens: int | None = None
    estimated_phase1_prompt_tokens: int | None = None
    estimated_phase1_completion_tokens: int | None = None
    estimated_phase1_total_tokens: int | None = None
    estimated_phase2_prompt_tokens: int | None = None
    estimated_phase2_completion_tokens: int | None = None
    estimated_phase2_total_tokens: int | None = None
    estimated_total_tokens: int | None = None
    token_usage_source: str = "none"


@dataclass
class ChatResponse:
    content: str
    reasoning: str
    metrics: RequestMetrics
