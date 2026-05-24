"""Streaming reasoning detectors.

Detectors are deliberately small and stateful. They receive reasoning text as it
streams in and decide whether Phase 1 should stop.
"""

import gzip
import re
from dataclasses import dataclass
from typing import Protocol

from .config import EarlyStopConfig
from .types import DetectorName, StopReason


@dataclass(frozen=True)
class StopDecision:
    should_stop: bool
    reason: StopReason = StopReason.NATURAL
    detail: str = ""


class ReasoningDetector(Protocol):
    name: DetectorName

    def update(self, piece: str, total_chars: int) -> StopDecision:
        """Process a reasoning chunk and optionally stop streaming."""


class NoStopDetector:
    """Monitor the stream without stopping.

    This is useful for baseline experiments where we still need to measure
    reasoning length and latency.
    """

    name: DetectorName = "none"

    def update(self, piece: str, total_chars: int) -> StopDecision:
        return StopDecision(should_stop=False)


class BudgetDetector:
    """Stop at a hard limit or at a sentence boundary after the soft budget."""

    name: DetectorName = "budget"

    def __init__(self, config: EarlyStopConfig) -> None:
        self.config = config
        self.sentence_re = re.compile(config.sentence_end_pattern)

    def update(self, piece: str, total_chars: int) -> StopDecision:
        if total_chars >= self.config.hard_limit:
            return StopDecision(
                should_stop=True,
                reason=StopReason.HARD,
                detail=f"hard_limit={self.config.hard_limit}",
            )

        if total_chars >= self.config.soft_budget and self.sentence_re.search(piece):
            return StopDecision(
                should_stop=True,
                reason=StopReason.SOFT,
                detail=f"soft_budget={self.config.soft_budget}",
            )

        return StopDecision(should_stop=False)


def compression_ratio(text: str) -> float:
    """Return gzip-compressed bytes / raw bytes for a short text window."""
    if not text:
        return 1.0
    data = text.encode("utf-8")
    return len(gzip.compress(data, compresslevel=6)) / len(data)


def lz77_factor_count(text: str) -> int:
    """Count LZ77-style factors with a simple O(n^2) parser.

    The detector uses short windows by default, so this implementation keeps the
    dependency surface small. Larger windows should use a faster rolling matcher.
    """
    n = len(text)
    i = 0
    factors = 0
    while i < n:
        best = 0
        for j in range(i):
            length = 0
            while i + length < n and j + length < i and text[j + length] == text[i + length]:
                length += 1
            best = max(best, length)
        i += max(1, best)
        factors += 1
    return factors


class CompressionDetector:
    """Layer 1 compression detector using CRD and relative LZ factor rate."""

    name: DetectorName = "compression"

    def __init__(self, config: EarlyStopConfig) -> None:
        self.config = config
        self._buf: list[str] = []
        self._baseline_rho: float | None = None
        self._baseline_lz_rate: float | None = None
        self._trigger_count = 0

    def update(self, piece: str, total_chars: int) -> StopDecision:
        if total_chars >= self.config.hard_limit:
            return StopDecision(
                should_stop=True,
                reason=StopReason.HARD,
                detail=f"hard_limit={self.config.hard_limit}",
            )

        self._buf.append(piece)
        text = "".join(self._buf)
        baseline_chars = self.config.compression_baseline_chars
        recent_chars = self.config.compression_recent_chars
        if len(text) < baseline_chars + recent_chars:
            return StopDecision(should_stop=False, detail="compression_warmup")

        if self._baseline_rho is None or self._baseline_lz_rate is None:
            baseline = text[:baseline_chars]
            self._baseline_rho = compression_ratio(baseline)
            self._baseline_lz_rate = lz77_factor_count(baseline) / max(1, len(baseline))

        recent = text[-recent_chars:]
        crd = compression_ratio(recent) / max(self._baseline_rho, 1e-9)
        lz_rate = lz77_factor_count(recent) / max(1, len(recent))
        lz_ratio = lz_rate / max(self._baseline_lz_rate, 1e-9)

        if crd < self.config.compression_theta_crd or lz_ratio < self.config.compression_theta_lz:
            self._trigger_count += 1
        else:
            self._trigger_count = 0

        if self._trigger_count >= self.config.compression_consecutive_windows:
            return StopDecision(
                should_stop=True,
                reason=StopReason.SOFT,
                detail=f"compression crd={crd:.3f} lz={lz_ratio:.3f}",
            )

        return StopDecision(
            should_stop=False,
            detail=f"compression crd={crd:.3f} lz={lz_ratio:.3f}",
        )


def build_detector(config: EarlyStopConfig) -> ReasoningDetector:
    if config.detector == "none":
        return NoStopDetector()
    if config.detector == "budget":
        return BudgetDetector(config)
    if config.detector == "compression":
        return CompressionDetector(config)
    raise ValueError(f"Unknown detector: {config.detector}")
