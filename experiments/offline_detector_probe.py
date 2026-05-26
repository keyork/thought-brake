"""Replay local reasoning traces through detectors without calling an LLM API.

Usage:
    uv run python experiments/offline_detector_probe.py
    uv run python experiments/offline_detector_probe.py --detectors compression,keyword,bocpd
    uv run python experiments/offline_detector_probe.py --input tmp/reasoning.txt
"""

import argparse
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

if __package__ in {None, ""}:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    sys.path = [p for p in sys.path if Path(p or ".").resolve() != script_dir]
    sys.path.insert(0, str(repo_root))

from experiments.config import DEFAULT_ENCODING  # noqa: E402
from thought_brake import EarlyStopConfig  # noqa: E402
from thought_brake.detectors import build_detector  # noqa: E402
from thought_brake.types import DetectorName, StopReason  # noqa: E402


@dataclass(frozen=True)
class SyntheticTrace:
    name: str
    text: str
    answer_char: int
    overthink_char: int


@dataclass(frozen=True)
class ReplayResult:
    trace: str
    detector: str
    stopped: bool
    reason: str
    stop_chars: int
    detail: str
    verdict: str


def _synthetic_traces() -> list[SyntheticTrace]:
    productive_prefix = (
        "先读题：小明的妈妈有三个孩子，老大叫大毛，老二叫二毛，问老三叫什么。\n"
        "关键条件在题干开头，主语是小明的妈妈，所以第三个孩子就是小明。\n"
        "答案是小明。\n"
    )
    overthinking_tail = (
        "不过我再确认一下：如果老大叫大毛，老二叫二毛，很多人会顺着模式猜三毛。\n"
        "但是题目问的是小明的妈妈的三个孩子，所以小明一定是其中一个孩子。\n"
        "再换个角度看，题目没有说老三叫三毛，只给了老大老二的名字。\n"
        "所以答案仍然是小明。让我再检查一遍，答案是小明，不是三毛。\n"
        "继续验证也不会改变结论，因为题干已经直接给出母亲是小明的妈妈。\n"
    )

    math_prefix = (
        "要求 17 加 28。先拆开计算：17 + 20 = 37，37 + 8 = 45。\n"
        "因此结果是 45。\n"
    )
    math_overthinking_tail = (
        "再检查一次：28 + 17 也可以算成 30 + 15 = 45。\n"
        "不过还可以从个位十位看，7 + 8 = 15，写 5 进 1，1 + 2 + 1 = 4，所以是 45。\n"
        "让我再确认，没有进位错误，仍然是 45。\n"
        "换个方式验证，45 - 28 = 17，所以答案不变。\n"
    )

    productive = (
        "分析题目：需要判断哪种做法符合学校安全最佳实践。\n"
        "零容忍和直接开除通常不是推荐方式，因为它们不能系统性改善学校安全。\n"
        "更合理的是建立学校安全与响应团队，负责预防、干预和危机响应。\n"
        "因此答案是：建立学校安全与响应团队。\n"
    )

    return [
        SyntheticTrace(
            name="riddle_overthinking_cn",
            text=productive_prefix + overthinking_tail,
            answer_char=len(productive_prefix),
            overthink_char=len(productive_prefix),
        ),
        SyntheticTrace(
            name="math_overthinking_cn",
            text=math_prefix + math_overthinking_tail,
            answer_char=len(math_prefix),
            overthink_char=len(math_prefix),
        ),
        SyntheticTrace(
            name="productive_cn",
            text=productive,
            answer_char=len(productive),
            overthink_char=len(productive) + 1,
        ),
    ]


def _chunks(text: str, chunk_chars: int) -> Iterable[str]:
    for start in range(0, len(text), chunk_chars):
        yield text[start : start + chunk_chars]


def _verdict(trace: SyntheticTrace, stopped: bool, stop_chars: int) -> str:
    if not stopped:
        return "missed" if trace.overthink_char <= len(trace.text) else "ok_no_stop"
    if stop_chars < trace.answer_char:
        return "too_early"
    if stop_chars <= len(trace.text):
        return "ok_stop"
    return "late"


def _replay(
    trace: SyntheticTrace,
    detector_name: DetectorName,
    *,
    chunk_chars: int,
    soft_budget: int,
    hard_limit: int,
) -> ReplayResult:
    cfg = EarlyStopConfig(
        detector=detector_name,
        soft_budget=soft_budget,
        hard_limit=hard_limit,
        compression_baseline_chars=80,
        compression_recent_chars=80,
        compression_consecutive_windows=1,
        ngram_window_chars=80,
        ngram_consecutive_windows=1,
        keyword_window_chars=120,
        keyword_consecutive_windows=1,
        semantic_window_chars=80,
        semantic_consecutive_windows=1,
        semantic_min_words=2,
        bocpd_window_chars=80,
        bocpd_min_windows=2,
    )
    detector = build_detector(cfg)
    total_chars = 0
    last_detail = ""

    for piece in _chunks(trace.text, chunk_chars):
        total_chars += len(piece)
        decision = detector.update(piece, total_chars)
        last_detail = decision.detail
        if decision.should_stop:
            return ReplayResult(
                trace=trace.name,
                detector=detector_name,
                stopped=True,
                reason=decision.reason.value,
                stop_chars=total_chars,
                detail=decision.detail,
                verdict=_verdict(trace, True, total_chars),
            )

    return ReplayResult(
        trace=trace.name,
        detector=detector_name,
        stopped=False,
        reason=StopReason.NATURAL.value,
        stop_chars=total_chars,
        detail=last_detail,
        verdict=_verdict(trace, False, total_chars),
    )


def _format_table(rows: list[ReplayResult]) -> str:
    headers = ["trace", "detector", "reason", "chars", "verdict", "detail"]
    table = [headers]
    for row in rows:
        detail = row.detail.replace("\n", " ")
        if len(detail) > 72:
            detail = detail[:69] + "..."
        table.append(
            [
                row.trace,
                row.detector,
                row.reason,
                str(row.stop_chars),
                row.verdict,
                detail,
            ]
        )

    widths = [max(len(r[i]) for r in table) for i in range(len(headers))]
    lines = []
    for idx, row in enumerate(table):
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
        if idx == 0:
            lines.append("  ".join("-" * width for width in widths))
    return "\n".join(lines)


def _parse_detectors(raw: str) -> list[DetectorName]:
    allowed: set[str] = {
        "budget",
        "compression",
        "keyword",
        "ngram",
        "semantic",
        "bocpd",
    }
    values = [item.strip() for item in raw.split(",") if item.strip()]
    invalid = sorted(set(values) - allowed)
    if invalid:
        raise ValueError(f"Unknown detector(s): {', '.join(invalid)}")
    return [value for value in values]  # type: ignore[list-item]


def _load_traces(input_path: Path | None) -> list[SyntheticTrace]:
    if input_path is None:
        return _synthetic_traces()

    text = input_path.read_text(encoding=DEFAULT_ENCODING)
    return [
        SyntheticTrace(
            name=input_path.stem,
            text=text,
            answer_char=len(text),
            overthink_char=0,
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline detector replay probe")
    parser.add_argument(
        "--detectors",
        default="compression,keyword,ngram,semantic,bocpd",
        help="Comma-separated detectors to replay",
    )
    parser.add_argument("--input", default=None, help="Optional raw reasoning text file")
    parser.add_argument("--chunk-chars", type=int, default=40)
    parser.add_argument("--soft-budget", type=int, default=120)
    parser.add_argument("--hard-limit", type=int, default=320)
    args = parser.parse_args()

    detectors = _parse_detectors(args.detectors)
    traces = _load_traces(Path(args.input) if args.input else None)
    rows = [
        _replay(
            trace,
            detector,
            chunk_chars=args.chunk_chars,
            soft_budget=args.soft_budget,
            hard_limit=args.hard_limit,
        )
        for trace in traces
        for detector in detectors
    ]

    print(_format_table(rows))


if __name__ == "__main__":
    main()
