"""Experiment runner: sweep soft_budget × question set, write JSONL results.

Usage:
    uv run python experiments/runner.py --dataset riddles --budgets 0,100,200,500,1000
    uv run python experiments/runner.py --dataset gsm8k --n 100 --budgets 0,200,500,1000,2000
    uv run python experiments/runner.py --dataset all --budgets 0,100,200,500,1000,2000
"""

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

if __package__ in {None, ""}:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    sys.path = [p for p in sys.path if Path(p or ".").resolve() != script_dir]
    sys.path.insert(0, str(repo_root))

from tqdm import tqdm  # noqa: E402

from experiments.config import (  # noqa: E402
    BASELINE_BUDGET,
    DEFAULT_BUDGETS,
    DEFAULT_ENCODING,
    DEFAULT_EXPERIMENT_N,
    DEFAULT_WORKERS,
    RESULT_SCHEMA_VERSION,
    RESULTS_DIR,
)
from experiments.datasets import gsm8k, mmlu, riddles  # noqa: E402
from experiments.datasets.base import Question  # noqa: E402
from experiments.evaluate import exact_match, llm_judge  # noqa: E402
from thought_brake import EarlyStopConfig, ThoughtBrakeClient  # noqa: E402
from thought_brake.types import DetectorName, Phase2Mode, StopReason  # noqa: E402

_WORKER_STATE = threading.local()


@dataclass
class Record:
    schema_version: int
    question_id: str
    difficulty: str
    category: str
    eval_mode: str
    budget: int               # soft_budget; 0 = baseline (no stopping)
    detector: str
    reasoning_chars: int
    truncated: bool
    stop_reason: str
    answer: str
    answer_chars: int
    phase1_prompt_tokens: int | None
    phase1_completion_tokens: int | None
    phase1_total_tokens: int | None
    phase1_reasoning_tokens: int | None
    phase2_prompt_tokens: int | None
    phase2_completion_tokens: int | None
    phase2_total_tokens: int | None
    phase2_reasoning_tokens: int | None
    total_prompt_tokens: int | None
    total_completion_tokens: int | None
    total_tokens: int | None
    total_reasoning_tokens: int | None
    estimated_phase1_prompt_tokens: int | None
    estimated_phase1_completion_tokens: int | None
    estimated_phase1_total_tokens: int | None
    estimated_phase2_prompt_tokens: int | None
    estimated_phase2_completion_tokens: int | None
    estimated_phase2_total_tokens: int | None
    estimated_total_tokens: int | None
    token_usage_source: str
    quality_score: float      # vs ground truth; -1 if evaluation skipped
    latency_ms: float
    phase2_used: bool
    phase2_failed: bool


def _get_worker_client() -> ThoughtBrakeClient:
    client = getattr(_WORKER_STATE, "client", None)
    if client is None:
        client = ThoughtBrakeClient()
        _WORKER_STATE.client = client
    return client


def _run_one(
    client: ThoughtBrakeClient,
    question: Question,
    budget: int,
    detector: DetectorName,
    phase2_mode: Phase2Mode = "prefill",
    track_usage: bool = False,
) -> Record:
    cfg = (
        EarlyStopConfig(detector="none")
        if budget == BASELINE_BUDGET
        else EarlyStopConfig(
            detector=detector,
            soft_budget=budget,
            hard_limit=budget * 2,
            phase2_mode=phase2_mode,
            track_token_usage=track_usage,
        )
    )
    if budget == BASELINE_BUDGET:
        cfg.track_token_usage = track_usage

    t0 = time.monotonic()
    resp = client.chat(
        messages=[{"role": "user", "content": question.question}],
        config=cfg,
    )
    latency_ms = (time.monotonic() - t0) * 1000

    return Record(
        schema_version=RESULT_SCHEMA_VERSION,
        question_id=question.id,
        difficulty=question.difficulty,
        category=question.category,
        eval_mode=question.eval_mode,
        budget=budget,
        detector=cfg.detector,
        reasoning_chars=resp.metrics.reasoning_chars,
        truncated=resp.metrics.stop_reason != StopReason.NATURAL,
        stop_reason=resp.metrics.stop_reason.value,
        answer=resp.content,
        answer_chars=len(resp.content),
        phase1_prompt_tokens=resp.metrics.phase1_prompt_tokens,
        phase1_completion_tokens=resp.metrics.phase1_completion_tokens,
        phase1_total_tokens=resp.metrics.phase1_total_tokens,
        phase1_reasoning_tokens=resp.metrics.phase1_reasoning_tokens,
        phase2_prompt_tokens=resp.metrics.phase2_prompt_tokens,
        phase2_completion_tokens=resp.metrics.phase2_completion_tokens,
        phase2_total_tokens=resp.metrics.phase2_total_tokens,
        phase2_reasoning_tokens=resp.metrics.phase2_reasoning_tokens,
        total_prompt_tokens=resp.metrics.total_prompt_tokens,
        total_completion_tokens=resp.metrics.total_completion_tokens,
        total_tokens=resp.metrics.total_tokens,
        total_reasoning_tokens=resp.metrics.total_reasoning_tokens,
        estimated_phase1_prompt_tokens=resp.metrics.estimated_phase1_prompt_tokens,
        estimated_phase1_completion_tokens=resp.metrics.estimated_phase1_completion_tokens,
        estimated_phase1_total_tokens=resp.metrics.estimated_phase1_total_tokens,
        estimated_phase2_prompt_tokens=resp.metrics.estimated_phase2_prompt_tokens,
        estimated_phase2_completion_tokens=resp.metrics.estimated_phase2_completion_tokens,
        estimated_phase2_total_tokens=resp.metrics.estimated_phase2_total_tokens,
        estimated_total_tokens=resp.metrics.estimated_total_tokens,
        token_usage_source=resp.metrics.token_usage_source,
        quality_score=-1.0,
        latency_ms=latency_ms,
        phase2_used=resp.metrics.phase2_used,
        phase2_failed=resp.metrics.phase2_failed,
    )


def _evaluate(
    client: ThoughtBrakeClient,
    question: Question,
    record: Record,
) -> float:
    if question.eval_mode == "exact_match":
        return exact_match.score(record.answer, question.ground_truth)
    else:
        return llm_judge.score(
            client._openai,
            client.model,
            question.question,
            question.ground_truth,
            record.answer,
        )


def _run_and_evaluate(
    question: Question,
    budget: int,
    detector: DetectorName,
    skip_eval: bool,
    phase2_mode: Phase2Mode = "prefill",
    track_usage: bool = False,
) -> Record:
    client = _get_worker_client()
    record = _run_one(client, question, budget, detector, phase2_mode, track_usage)
    if not skip_eval:
        record.quality_score = _evaluate(client, question, record)
    return record


def _load_done(results_path: Path) -> set[tuple[str, int, str]]:
    if not results_path.exists():
        return set()

    done: set[tuple[str, int, str]] = set()
    with results_path.open(encoding=DEFAULT_ENCODING) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if int(r.get("schema_version", 1)) != RESULT_SCHEMA_VERSION:
                continue
            done.add((r["question_id"], int(r["budget"]), str(r.get("detector", "budget"))))
    return done


def run_experiment(
    questions: list[Question],
    budgets: list[int],
    output_path: Path,
    skip_eval: bool = False,
    workers: int = DEFAULT_WORKERS,
    detector: DetectorName = "budget",
    phase2_mode: Phase2Mode = "prefill",
    track_usage: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pairs = [(q, b) for q in questions for b in budgets]
    done = _load_done(output_path)
    pending = [
        (q, b)
        for q, b in pairs
        if (q.id, b, "none" if b == BASELINE_BUDGET else detector) not in done
    ]
    skipped = len(pairs) - len(pending)
    max_workers = max(1, workers)

    with tqdm(total=len(pairs), initial=skipped, desc="Running") as pbar:
        with output_path.open("a", encoding=DEFAULT_ENCODING) as out:
            if max_workers == 1:
                for question, budget in pending:
                    pbar.set_postfix(q=question.id, budget=budget)
                    record = _run_and_evaluate(
                        question,
                        budget,
                        detector,
                        skip_eval,
                        phase2_mode,
                        track_usage,
                    )
                    out.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
                    out.flush()
                    pbar.update(1)
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(
                            _run_and_evaluate,
                            question,
                            budget,
                            detector,
                            skip_eval,
                            phase2_mode,
                            track_usage,
                        ): (question, budget)
                        for question, budget in pending
                    }
                    for future in as_completed(futures):
                        question, budget = futures[future]
                        pbar.set_postfix(q=question.id, budget=budget)
                        record = future.result()
                        out.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
                        out.flush()
                        pbar.update(1)

    print(
        f"\nResults saved to {output_path} "
        f"({max_workers} workers, detector={detector}, phase2={phase2_mode}, "
        f"track_usage={track_usage})"
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="thought-brake experiment runner")
    p.add_argument(
        "--dataset",
        choices=["riddles", "gsm8k", "mmlu", "all"],
        default="riddles",
    )
    p.add_argument(
        "--budgets",
        default=DEFAULT_BUDGETS,
        help="Comma-separated soft_budget values; 0 = baseline",
    )
    p.add_argument("--n", type=int, default=DEFAULT_EXPERIMENT_N, help="Max questions per dataset")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Parallel API workers")
    p.add_argument(
        "--detector",
        choices=["budget", "compression", "keyword", "ngram", "semantic"],
        default="budget",
        help="Early-stop detector for non-baseline budgets",
    )
    p.add_argument("--difficulties", default=None, help="e.g. easy,medium")
    p.add_argument(
        "--output",
        default=None,
        help="Output JSONL path (default: experiments/results/<dataset>.jsonl)",
    )
    p.add_argument("--skip-eval", action="store_true", help="Skip quality evaluation")
    p.add_argument(
        "--track-usage",
        action="store_true",
        help="Request streaming token usage from the LLM API and write token fields",
    )
    p.add_argument(
        "--phase2",
        choices=["prefill", "direct"],
        default="prefill",
        help="Phase 2 mode: prefill (assistant prefill) or direct (user prompt + disable thinking)",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    budgets = [int(b) for b in args.budgets.split(",")]
    difficulties = args.difficulties.split(",") if args.difficulties else None

    questions: list[Question] = []
    if args.dataset in ("riddles", "all"):
        questions += riddles.load(difficulties=difficulties)
    if args.dataset in ("gsm8k", "all"):
        questions += gsm8k.load(n=args.n)
    if args.dataset in ("mmlu", "all"):
        questions += mmlu.load(n=args.n)

    output = Path(args.output) if args.output else RESULTS_DIR / f"{args.dataset}.jsonl"
    run_experiment(
        questions,
        budgets,
        output,
        skip_eval=args.skip_eval,
        workers=args.workers,
        detector=args.detector,
        phase2_mode=args.phase2,
        track_usage=args.track_usage,
    )


if __name__ == "__main__":
    main()
