from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

ChatMessage = dict[str, Any]
DetectorName = Literal["budget", "compression", "keyword", "ngram", "semantic", "none"]
Phase2Mode = Literal["prefill", "direct"]


class StopReason(StrEnum):
    NATURAL = "natural"  # model finished thinking on its own
    SOFT = "soft"  # exceeded soft_budget, cut at sentence boundary
    HARD = "hard"  # exceeded hard_limit, forced cut
    INTERRUPTED = "interrupted"  # stream failed after partial reasoning was collected


@dataclass
class Phase1Result:
    reasoning: str
    content: str  # non-empty only when stop_reason is NATURAL
    stop_reason: StopReason


@dataclass
class RequestMetrics:
    reasoning_chars: int = 0
    stop_reason: StopReason = StopReason.NATURAL
    phase2_used: bool = False
    phase2_failed: bool = False


@dataclass
class ChatResponse:
    content: str
    reasoning: str
    metrics: RequestMetrics
