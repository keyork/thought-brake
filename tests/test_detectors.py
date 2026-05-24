"""Tests for streaming reasoning detectors."""

from thought_brake.config import EarlyStopConfig
from thought_brake.detectors import (
    BudgetDetector,
    CompressionDetector,
    NoStopDetector,
    build_detector,
    compression_ratio,
    lz77_factor_count,
)
from thought_brake.types import StopReason


def test_budget_detector_soft_stop_at_sentence_boundary() -> None:
    detector = BudgetDetector(EarlyStopConfig(soft_budget=5, hard_limit=100))

    first = detector.update("12345", 5)
    second = detector.update(" done。", 11)

    assert not first.should_stop
    assert second.should_stop
    assert second.reason == StopReason.SOFT


def test_budget_detector_hard_stop_without_sentence_boundary() -> None:
    detector = BudgetDetector(EarlyStopConfig(soft_budget=5, hard_limit=10))

    decision = detector.update("678901", 11)

    assert decision.should_stop
    assert decision.reason == StopReason.HARD


def test_no_stop_detector_never_stops() -> None:
    detector = NoStopDetector()

    decision = detector.update("1234567890。", 10_000)

    assert not decision.should_stop


def test_build_detector_from_config() -> None:
    assert isinstance(build_detector(EarlyStopConfig(detector="none")), NoStopDetector)
    assert isinstance(build_detector(EarlyStopConfig(detector="budget")), BudgetDetector)
    assert isinstance(build_detector(EarlyStopConfig(detector="compression")), CompressionDetector)


def test_compression_signals_drop_for_repetitive_text() -> None:
    healthy = "先分析条件，再建立方程，最后检查答案是否满足题意。"
    repetitive = "等等，我再想想。" * 20

    assert compression_ratio(repetitive) < compression_ratio(healthy)
    assert lz77_factor_count(repetitive) / len(repetitive) < lz77_factor_count(healthy) / len(
        healthy
    )


def test_compression_detector_stops_on_repetitive_tail() -> None:
    cfg = EarlyStopConfig(
        detector="compression",
        hard_limit=10_000,
        compression_baseline_chars=20,
        compression_recent_chars=20,
        compression_theta_crd=0.95,
        compression_theta_lz=0.95,
        compression_consecutive_windows=1,
    )
    detector = CompressionDetector(cfg)

    total = 0
    stopped = False
    for piece in [
        "先分析条件再建立方程然后检查。",
        "根据题意列出变量关系并计算。",
        "等等等等等等等等等等等等等等等等等等。",
        "等等等等等等等等等等等等等等等等等等。",
    ]:
        total += len(piece)
        stopped = detector.update(piece, total).should_stop

    assert stopped
