"""Load a subset of MMLU from HuggingFace datasets.

MMLU is multiple-choice: question + 4 choices, answer is 0-3 index.
We format the question with choices, extract the letter as ground truth,
and use exact_match evaluation.
"""

import random

from experiments.config import DEFAULT_EXPERIMENT_N

from .base import Question

_LABELS = ["A", "B", "C", "D"]


def _format_question(question: str, choices: list[str]) -> str:
    lines = [question]
    for label, choice in zip(_LABELS, choices):
        lines.append(f"{label}. {choice}")
    return "\n".join(lines)


def load(
    n: int = DEFAULT_EXPERIMENT_N,
    subjects: list[str] | None = None,
    seed: int = 42,
    split: str = "test",
) -> list[Question]:
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError("Run: uv sync --group experiments") from e

    ds = load_dataset("cais/mmlu", "all", split=split)

    rows = list(ds)
    if subjects:
        rows = [r for r in rows if r["subject"] in subjects]

    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:n]

    questions = []
    for row in rows:
        answer_letter = _LABELS[row["answer"]]
        q_hash = abs(hash(row["question"])) % 100000
        questions.append(
            Question(
                id=f"mmlu_{row['subject']}_{q_hash:05d}",
                difficulty="medium",
                category=row["subject"],
                question=_format_question(row["question"], row["choices"]),
                ground_truth=answer_letter,
                eval_mode="exact_match",
            )
        )
    return questions
