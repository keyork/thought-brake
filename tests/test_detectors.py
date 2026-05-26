"""Tests for streaming reasoning detectors."""

from thought_brake.config import EarlyStopConfig
from thought_brake.detectors import (
    BOCPDDetector,
    BudgetDetector,
    CompressionDetector,
    KeywordDetector,
    NGramDetector,
    NoStopDetector,
    OnlineChangePoint,
    SemanticDetector,
    bocpd_features,
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
    assert isinstance(build_detector(EarlyStopConfig(detector="ngram")), NGramDetector)
    assert isinstance(build_detector(EarlyStopConfig(detector="keyword")), KeywordDetector)
    assert isinstance(build_detector(EarlyStopConfig(detector="semantic")), SemanticDetector)
    assert isinstance(build_detector(EarlyStopConfig(detector="bocpd")), BOCPDDetector)


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


def test_ngram_detector_does_not_stop_on_fresh_text() -> None:
    cfg = EarlyStopConfig(
        detector="ngram",
        hard_limit=10_000,
        ngram_size=3,
        ngram_window_chars=30,
        ngram_threshold=0.5,
        ngram_consecutive_windows=1,
    )
    detector = NGramDetector(cfg)

    total = 0
    stopped = False
    for piece in [
        "今天天气不错适合出门散步。",
        "我想去买一本关于历史的书籍。",
        "这个问题需要仔细分析才能得出结论。",
        "根据物理学原理物体运动的规律。",
    ]:
        total += len(piece)
        stopped = detector.update(piece, total).should_stop

    assert not stopped


def test_ngram_detector_stops_on_repetitive_text() -> None:
    cfg = EarlyStopConfig(
        detector="ngram",
        hard_limit=10_000,
        ngram_size=3,
        ngram_window_chars=20,
        ngram_threshold=0.5,
        ngram_consecutive_windows=1,
    )
    detector = NGramDetector(cfg)

    total = 0
    stopped = False
    for piece in [
        "让我再想想这个问题让我再想想这个问题让我再想想。",
        "让我再想想这个问题让我再想想这个问题让我再想想。",
        "让我再想想这个问题让我再想想这个问题让我再想想。",
    ]:
        total += len(piece)
        stopped = detector.update(piece, total).should_stop

    assert stopped


def test_ngram_detector_hard_limit() -> None:
    cfg = EarlyStopConfig(
        detector="ngram",
        hard_limit=20,
        ngram_size=3,
        ngram_window_chars=10,
        ngram_threshold=0.9,
        ngram_consecutive_windows=5,
    )
    detector = NGramDetector(cfg)

    decision = detector.update("a" * 25, 25)
    assert decision.should_stop
    assert decision.reason == StopReason.HARD


def test_keyword_detector_does_not_stop_before_conclusion() -> None:
    cfg = EarlyStopConfig(
        detector="keyword",
        hard_limit=10_000,
        keyword_window_chars=30,
        keyword_trigger_threshold=0.01,
        keyword_consecutive_windows=1,
    )
    detector = KeywordDetector(cfg)

    total = 0
    stopped = False
    for piece in [
        "分析问题的条件和约束。",
        "根据条件建立方程求解。",
        "计算得到结果验证正确。",
        "继续考虑其他可能性。",
    ]:
        total += len(piece)
        stopped = detector.update(piece, total).should_stop

    assert not stopped


def test_keyword_detector_stops_after_conclusion_with_hedging() -> None:
    cfg = EarlyStopConfig(
        detector="keyword",
        hard_limit=10_000,
        keyword_window_chars=50,
        keyword_trigger_threshold=0.01,
        keyword_consecutive_windows=1,
    )
    detector = KeywordDetector(cfg)

    total = 0
    stopped = False
    for piece in [
        "分析条件后得出答案是水。因此答案是水。",
        "水因为能把污垢冲洗掉所以越洗越脏。",
        "等等，让我再想想这个问题。不过换个角度考虑。",
        "但是重新思考一下，还有没有其他可能。让我再考虑考虑。",
    ]:
        total += len(piece)
        stopped = detector.update(piece, total).should_stop

    assert stopped


def test_keyword_detector_hard_limit() -> None:
    cfg = EarlyStopConfig(
        detector="keyword",
        hard_limit=20,
        keyword_window_chars=10,
        keyword_trigger_threshold=0.9,
        keyword_consecutive_windows=5,
    )
    detector = KeywordDetector(cfg)

    decision = detector.update("a" * 25, 25)
    assert decision.should_stop
    assert decision.reason == StopReason.HARD


def test_semantic_detector_does_not_stop_on_diverse_text() -> None:
    cfg = EarlyStopConfig(
        detector="semantic",
        hard_limit=10_000,
        semantic_window_chars=30,
        semantic_jaccard_threshold=0.4,
        semantic_consecutive_windows=1,
        semantic_min_words=3,
    )
    detector = SemanticDetector(cfg)

    total = 0
    stopped = False
    for piece in [
        "分析条件建立方程求解计算。",
        "根据物理学原理推导结论。",
        "利用历史知识判断事件因果。",
        "通过化学反应方程配平验证。",
    ]:
        total += len(piece)
        stopped = detector.update(piece, total).should_stop

    assert not stopped


def test_semantic_detector_stops_on_semantic_repetition() -> None:
    cfg = EarlyStopConfig(
        detector="semantic",
        hard_limit=10_000,
        semantic_window_chars=20,
        semantic_jaccard_threshold=0.25,
        semantic_consecutive_windows=1,
        semantic_min_words=2,
    )
    detector = SemanticDetector(cfg)

    total = 0
    stopped = False
    for piece in [
        "分析问题条件建立方程求解。",
        "答案是水因为洗东西水变脏。",
        "水洗东西污垢进入水中变脏。",
        "因为用水清洗污垢融入水中。",
    ]:
        total += len(piece)
        stopped = detector.update(piece, total).should_stop

    assert stopped


def test_bocpd_features_score_repetitive_text_higher_than_diverse_text() -> None:
    cfg = EarlyStopConfig(detector="bocpd")
    history = "分析题目条件，建立关系，逐步推导，检查候选答案。"
    diverse = "接着根据新的约束计算比例，并比较不同候选项。"
    repetitive = "答案是水。等等，让我再想想。答案是水。等等，让我再想想。"

    diverse_features = bocpd_features(diverse, history, conclusion_seen=False, config=cfg)
    repetitive_features = bocpd_features(
        repetitive,
        history + repetitive,
        conclusion_seen=True,
        config=cfg,
    )

    assert repetitive_features.low_value_score > diverse_features.low_value_score
    assert repetitive_features.conclusion_signal == 1.0


def test_online_change_point_reacts_to_mean_shift() -> None:
    detector = OnlineChangePoint(
        hazard_lambda=20,
        max_run_length=16,
        observation_sigma=0.08,
        prior_mean=0.5,
        prior_sigma=0.35,
    )

    steady_states = [detector.update(0.15) for _ in range(8)]
    shifted = detector.update(0.85)

    assert steady_states[-1].map_run_length > 1
    assert shifted.change_prob > steady_states[-1].change_prob
    assert abs(sum(shifted.run_length_probs) - 1.0) < 1e-9


def test_bocpd_detector_stops_after_conclusion_and_low_value_shift() -> None:
    cfg = EarlyStopConfig(
        detector="bocpd",
        soft_budget=40,
        hard_limit=10_000,
        bocpd_window_chars=20,
        bocpd_min_windows=2,
        bocpd_hazard_lambda=20,
        bocpd_stop_prob=0.2,
        bocpd_low_value_threshold=0.25,
        bocpd_observation_sigma=0.08,
    )
    detector = BOCPDDetector(cfg)

    total = 0
    stopped = False
    for piece in [
        "先分析题目条件然后建立约束关系。",
        "根据约束推导候选答案并验证。",
        "因此答案是水。等等让我再想想。",
        "答案是水等等让我再想想答案是水。",
    ]:
        total += len(piece)
        stopped = stopped or detector.update(piece, total).should_stop

    assert stopped


def test_bocpd_detector_reports_blockers_before_soft_stop() -> None:
    cfg = EarlyStopConfig(
        detector="bocpd",
        soft_budget=40,
        hard_limit=10_000,
        bocpd_window_chars=20,
        bocpd_min_windows=1,
        bocpd_stop_prob=0.99,
        bocpd_low_value_threshold=0.99,
    )
    detector = BOCPDDetector(cfg)

    piece = "先分析条件然后建立约束关系继续推导并检查候选答案。"
    decision = detector.update(piece, len(piece))

    assert not decision.should_stop
    assert "blocked=" in decision.detail
    assert "no_conclusion" in decision.detail


def test_bocpd_detector_hard_limit() -> None:
    detector = BOCPDDetector(EarlyStopConfig(detector="bocpd", hard_limit=20))

    decision = detector.update("a" * 25, 25)

    assert decision.should_stop
    assert decision.reason == StopReason.HARD
    assert decision.detail.startswith("hard_limit=20")
    assert "last=bocpd" in decision.detail
