"""Streaming reasoning detectors.

Detectors are deliberately small and stateful. They receive reasoning text as it
streams in and decide whether Phase 1 should stop.
"""

import gzip
import math
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


@dataclass(frozen=True)
class BOCPDFeatures:
    compression_signal: float
    lz_signal: float
    ngram_overlap: float
    keyword_signal: float
    conclusion_signal: float
    low_value_score: float


@dataclass(frozen=True)
class ChangePointState:
    change_prob: float
    map_run_length: int
    run_length_probs: tuple[float, ...]


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


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _extract_char_ngrams(text: str, ngram_size: int) -> set[str]:
    if len(text) < ngram_size:
        return set()
    return {text[i : i + ngram_size] for i in range(len(text) - ngram_size + 1)}


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


class NGramDetector:
    """Stop when recent n-grams overlap history above threshold for k consecutive windows."""

    name: DetectorName = "ngram"

    def __init__(self, config: EarlyStopConfig) -> None:
        self.config = config
        self._buf: list[str] = []
        self._history_ngrams: set[str] = set()
        self._ngram_size = config.ngram_size
        self._window_chars = config.ngram_window_chars
        self._trigger_count = 0

    def _extract_ngrams(self, text: str) -> set[str]:
        return _extract_char_ngrams(text, self._ngram_size)

    def update(self, piece: str, total_chars: int) -> StopDecision:
        if total_chars >= self.config.hard_limit:
            return StopDecision(
                should_stop=True,
                reason=StopReason.HARD,
                detail=f"hard_limit={self.config.hard_limit}",
            )

        self._buf.append(piece)
        text = "".join(self._buf)

        window_chars = self._window_chars
        if len(text) < window_chars * 2:
            return StopDecision(should_stop=False, detail="ngram_warmup")

        history = text[: len(text) - window_chars]
        recent = text[-window_chars:]

        history_ngrams = self._extract_ngrams(history)
        recent_ngrams = self._extract_ngrams(recent)

        if not recent_ngrams:
            return StopDecision(should_stop=False, detail="ngram_window_too_short")

        overlap = len(recent_ngrams & history_ngrams) / len(recent_ngrams)

        if overlap >= self.config.ngram_threshold:
            self._trigger_count += 1
        else:
            self._trigger_count = 0

        if self._trigger_count >= self.config.ngram_consecutive_windows:
            return StopDecision(
                should_stop=True,
                reason=StopReason.SOFT,
                detail=f"ngram overlap={overlap:.3f}",
            )

        return StopDecision(
            should_stop=False,
            detail=f"ngram overlap={overlap:.3f}",
        )


_HEDGE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p)
    for p in [
        r"等等[，。]?\s*",
        r"让我再",
        r"不对[，。]?\s*",
        r"不过[，。]?\s*",
        r"但是[，。]?\s*",
        r"然而[，。]?\s*",
        r"重新(?:考虑|思考|审视)",
        r"再(?:验证|检查|确认|想想|思考)一?下?",
        r"换个角度",
        r"另一种(?:可能|思路|解释)",
        r"会不会(?:是|有|就)",
        r"话说回来",
        r"其实还可以",
        r"wait[,.]?\s*",
        r"let me (?:reconsider|rethink|check|verify)",
        r"but (?:wait|actually|hold on)",
        r"on second thought",
        r"(?:I'm|I am) not (?:sure|certain)",
        r"let's see",
    ]
]

_CONCLUSION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p)
    for p in [
        r"因此[，]?\s*(?:答案|最终|结论|结果)",
        r"所以[，]?\s*(?:答案|最终|结论|结果|.*是)",
        r"答案是[：:]",
        r"最终答案[：:]",
        r"the answer is",
        r"therefore[,]\s*",
    ]
]


class KeywordDetector:
    """Stop when hedge/transition phrase density exceeds threshold after a conclusion."""

    name: DetectorName = "keyword"

    def __init__(self, config: EarlyStopConfig) -> None:
        self.config = config
        self._buf: list[str] = []
        self._found_conclusion = False
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

        if not self._found_conclusion:
            for pat in _CONCLUSION_PATTERNS:
                if pat.search(piece):
                    self._found_conclusion = True
                    break

        if not self._found_conclusion:
            return StopDecision(should_stop=False, detail="keyword_no_conclusion_yet")

        window_chars = self.config.keyword_window_chars
        if len(text) < window_chars:
            return StopDecision(should_stop=False, detail="keyword_warmup")

        recent = text[-window_chars:]
        hedge_count = sum(1 for pat in _HEDGE_PATTERNS if pat.search(recent))
        density = hedge_count / max(1, len(recent) / 50)

        if density >= self.config.keyword_trigger_threshold:
            self._trigger_count += 1
        else:
            self._trigger_count = 0

        if self._trigger_count >= self.config.keyword_consecutive_windows:
            return StopDecision(
                should_stop=True,
                reason=StopReason.SOFT,
                detail=f"keyword density={density:.3f} hedges={hedge_count}",
            )

        return StopDecision(
            should_stop=False,
            detail=f"keyword density={density:.3f} hedges={hedge_count}",
        )


_CN_STOPWORDS: frozenset[str] = frozenset(
    "的 了 是 在 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好 "
    "自己 这 那 他 她 它 们 把 被 让 给 从 对 与 而 或 但 如果 因为 所以 什么 怎么 "
    "哪 里 多 少 几 第 个 只 还 又 再 已 吧 呢 吗 啊 呀 嗯 哈 哦".split()
)

_EN_STOPWORDS: frozenset[str] = frozenset(
    "the a an is are was were be been being have has had do does did will would "
    "shall should may might must can could of in to for with on at from by about "
    "as into through during before after above below between out off over under "
    "again further then once here there when where why how all both each few more "
    "most other some such no nor not only own same so than too very and but or if "
    "it its this that these those i me my we our you your he him his she her they "
    "them their what which who whom".split()
)


def _content_words(text: str) -> set[str]:
    tokens = re.findall(r"[\u4e00-\u9fff]|[a-zA-Z]+", text)
    return {t.lower() for t in tokens if t not in _CN_STOPWORDS and t not in _EN_STOPWORDS}


class SemanticDetector:
    """Stop when content-word Jaccard similarity between recent and history exceeds threshold."""

    name: DetectorName = "semantic"

    def __init__(self, config: EarlyStopConfig) -> None:
        self.config = config
        self._buf: list[str] = []
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

        window_chars = self.config.semantic_window_chars
        if len(text) < window_chars * 2:
            return StopDecision(should_stop=False, detail="semantic_warmup")

        history = text[: len(text) - window_chars]
        recent = text[-window_chars:]

        history_words = _content_words(history)
        recent_words = _content_words(recent)

        if len(recent_words) < self.config.semantic_min_words:
            return StopDecision(
                should_stop=False,
                detail=f"semantic_few_words={len(recent_words)}",
            )

        union = history_words | recent_words
        if not union:
            return StopDecision(should_stop=False, detail="semantic_empty")

        jaccard = len(history_words & recent_words) / len(union)

        if jaccard >= self.config.semantic_jaccard_threshold:
            self._trigger_count += 1
        else:
            self._trigger_count = 0

        if self._trigger_count >= self.config.semantic_consecutive_windows:
            return StopDecision(
                should_stop=True,
                reason=StopReason.SOFT,
                detail=f"semantic jaccard={jaccard:.3f}",
            )

        return StopDecision(
            should_stop=False,
            detail=f"semantic jaccard={jaccard:.3f}",
        )


def bocpd_features(
    window: str,
    history: str,
    conclusion_seen: bool,
    config: EarlyStopConfig,
) -> BOCPDFeatures:
    """Return low-value reasoning features for one BOCPD observation window."""
    compression_signal = _clamp01(1.0 - compression_ratio(window))
    lz_rate = lz77_factor_count(window) / max(1, len(window))
    lz_signal = _clamp01(1.0 - (lz_rate / 0.5))

    recent_ngrams = _extract_char_ngrams(window, config.ngram_size)
    history_ngrams = _extract_char_ngrams(history, config.ngram_size)
    if recent_ngrams and history_ngrams:
        ngram_overlap = len(recent_ngrams & history_ngrams) / len(recent_ngrams)
    else:
        ngram_overlap = 0.0

    hedge_count = sum(1 for pat in _HEDGE_PATTERNS if pat.search(window))
    keyword_signal = _clamp01(hedge_count / max(1.0, len(window) / 50))
    conclusion_signal = 1.0 if conclusion_seen else 0.0

    weighted_sum = (
        config.bocpd_compression_weight * compression_signal
        + config.bocpd_lz_weight * lz_signal
        + config.bocpd_ngram_weight * ngram_overlap
        + config.bocpd_keyword_weight * keyword_signal
        + config.bocpd_conclusion_weight * conclusion_signal
    )
    weight_total = (
        config.bocpd_compression_weight
        + config.bocpd_lz_weight
        + config.bocpd_ngram_weight
        + config.bocpd_keyword_weight
        + config.bocpd_conclusion_weight
    )
    low_value_score = weighted_sum / max(weight_total, 1e-9)

    return BOCPDFeatures(
        compression_signal=compression_signal,
        lz_signal=lz_signal,
        ngram_overlap=ngram_overlap,
        keyword_signal=keyword_signal,
        conclusion_signal=conclusion_signal,
        low_value_score=low_value_score,
    )


@dataclass(frozen=True)
class _RunHypothesis:
    run_length: int
    probability: float
    mean: float
    count: int


class OnlineChangePoint:
    """Small online change-point core used by BOCPDDetector.

    This is a lightweight BOCPD-style approximation for bounded run lengths. It
    keeps a distribution over run lengths and uses a broad prior predictive for
    newly started segments.
    """

    def __init__(
        self,
        *,
        hazard_lambda: float,
        max_run_length: int,
        observation_sigma: float,
        prior_mean: float,
        prior_sigma: float,
    ) -> None:
        self.hazard = _clamp01(1.0 / max(hazard_lambda, 1e-9))
        self.max_run_length = max_run_length
        self.observation_sigma = max(observation_sigma, 1e-6)
        self.prior_mean = prior_mean
        self.prior_sigma = max(prior_sigma, self.observation_sigma)
        self._hypotheses: list[_RunHypothesis] = [
            _RunHypothesis(run_length=0, probability=1.0, mean=prior_mean, count=0)
        ]

    def update(self, value: float) -> ChangePointState:
        value = _clamp01(value)
        candidates: dict[int, _RunHypothesis] = {}
        change_probability = 0.0

        for hypothesis in self._hypotheses:
            growth_likelihood = self._normal_pdf(
                value,
                hypothesis.mean,
                self.observation_sigma,
            )
            change_likelihood = self._normal_pdf(value, self.prior_mean, self.prior_sigma)

            growth_prob = hypothesis.probability * (1.0 - self.hazard) * growth_likelihood
            growth_run = min(hypothesis.run_length + 1, self.max_run_length)
            growth_mean = (
                value
                if hypothesis.count == 0
                else (hypothesis.mean * hypothesis.count + value) / (hypothesis.count + 1)
            )
            self._merge_candidate(
                candidates,
                _RunHypothesis(
                    run_length=growth_run,
                    probability=growth_prob,
                    mean=growth_mean,
                    count=min(hypothesis.count + 1, self.max_run_length),
                ),
            )

            change_prob = hypothesis.probability * self.hazard * change_likelihood
            change_probability += change_prob
            self._merge_candidate(
                candidates,
                _RunHypothesis(
                    run_length=0,
                    probability=change_prob,
                    mean=value,
                    count=1,
                ),
            )

        total = sum(h.probability for h in candidates.values())
        if total <= 0:
            self._hypotheses = [
                _RunHypothesis(run_length=0, probability=1.0, mean=value, count=1)
            ]
            return ChangePointState(
                change_prob=1.0,
                map_run_length=0,
                run_length_probs=(1.0,),
            )

        self._hypotheses = sorted(
            (
                _RunHypothesis(
                    run_length=h.run_length,
                    probability=h.probability / total,
                    mean=h.mean,
                    count=h.count,
                )
                for h in candidates.values()
            ),
            key=lambda h: h.probability,
            reverse=True,
        )[: self.max_run_length]

        normalized_total = sum(h.probability for h in self._hypotheses)
        self._hypotheses = [
            _RunHypothesis(
                run_length=h.run_length,
                probability=h.probability / normalized_total,
                mean=h.mean,
                count=h.count,
            )
            for h in self._hypotheses
        ]
        map_run_length = max(self._hypotheses, key=lambda h: h.probability).run_length
        normalized_change_prob = change_probability / total

        return ChangePointState(
            change_prob=_clamp01(normalized_change_prob),
            map_run_length=map_run_length,
            run_length_probs=tuple(h.probability for h in self._hypotheses),
        )

    @staticmethod
    def _normal_pdf(value: float, mean: float, sigma: float) -> float:
        z = (value - mean) / sigma
        return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))

    @staticmethod
    def _merge_candidate(
        candidates: dict[int, _RunHypothesis], candidate: _RunHypothesis
    ) -> None:
        existing = candidates.get(candidate.run_length)
        if existing is None:
            candidates[candidate.run_length] = candidate
            return

        probability = existing.probability + candidate.probability
        if probability <= 0:
            return
        mean = (
            existing.mean * existing.probability
            + candidate.mean * candidate.probability
        ) / probability
        candidates[candidate.run_length] = _RunHypothesis(
            run_length=candidate.run_length,
            probability=probability,
            mean=mean,
            count=max(existing.count, candidate.count),
        )


class BOCPDDetector:
    """Layer 2 detector using online change-point detection over text features."""

    name: DetectorName = "bocpd"

    def __init__(self, config: EarlyStopConfig) -> None:
        self.config = config
        self._buf: list[str] = []
        self._processed_chars = 0
        self._windows_seen = 0
        self._found_conclusion = False
        self._state = ChangePointState(
            change_prob=0.0,
            map_run_length=0,
            run_length_probs=(1.0,),
        )
        self._change_point = OnlineChangePoint(
            hazard_lambda=config.bocpd_hazard_lambda,
            max_run_length=config.bocpd_max_run_length,
            observation_sigma=config.bocpd_observation_sigma,
            prior_mean=config.bocpd_prior_mean,
            prior_sigma=config.bocpd_prior_sigma,
        )

    def update(self, piece: str, total_chars: int) -> StopDecision:
        if total_chars >= self.config.hard_limit:
            return StopDecision(
                should_stop=True,
                reason=StopReason.HARD,
                detail=f"hard_limit={self.config.hard_limit}",
            )

        self._buf.append(piece)
        if not self._found_conclusion and any(pat.search(piece) for pat in _CONCLUSION_PATTERNS):
            self._found_conclusion = True

        text = "".join(self._buf)
        window_chars = self.config.bocpd_window_chars
        last_features: BOCPDFeatures | None = None
        while len(text) - self._processed_chars >= window_chars:
            window = text[self._processed_chars : self._processed_chars + window_chars]
            history = text[: self._processed_chars]
            last_features = bocpd_features(
                window,
                history,
                self._found_conclusion,
                self.config,
            )
            self._state = self._change_point.update(last_features.low_value_score)
            self._processed_chars += window_chars
            self._windows_seen += 1

        if last_features is None:
            return StopDecision(
                should_stop=False,
                detail=f"bocpd_collecting chars={len(text) - self._processed_chars}",
            )

        detail = (
            f"bocpd p_change={self._state.change_prob:.3f} "
            f"z={last_features.low_value_score:.3f} r_map={self._state.map_run_length}"
        )
        if self._windows_seen < self.config.bocpd_min_windows:
            return StopDecision(should_stop=False, detail=f"{detail} warmup")

        recent_change = self._state.map_run_length <= self.config.bocpd_recent_run_threshold
        should_stop = (
            total_chars >= self.config.soft_budget
            and self._found_conclusion
            and recent_change
            and self._state.change_prob >= self.config.bocpd_stop_prob
            and last_features.low_value_score >= self.config.bocpd_low_value_threshold
        )
        if should_stop:
            return StopDecision(
                should_stop=True,
                reason=StopReason.SOFT,
                detail=detail,
            )

        return StopDecision(should_stop=False, detail=detail)


def build_detector(config: EarlyStopConfig) -> ReasoningDetector:
    if config.detector == "none":
        return NoStopDetector()
    if config.detector == "budget":
        return BudgetDetector(config)
    if config.detector == "compression":
        return CompressionDetector(config)
    if config.detector == "ngram":
        return NGramDetector(config)
    if config.detector == "keyword":
        return KeywordDetector(config)
    if config.detector == "semantic":
        return SemanticDetector(config)
    if config.detector == "bocpd":
        return BOCPDDetector(config)
    raise ValueError(f"Unknown detector: {config.detector}")
