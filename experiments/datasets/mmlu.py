"""Load a subset of MMLU from HuggingFace datasets.

MMLU is multiple-choice: question + 4 choices, answer is 0-3 index.
We format the question with choices, extract the letter as ground truth,
and use exact_match evaluation.
"""

import random
from hashlib import sha1

from experiments.config import DEFAULT_ENCODING, DEFAULT_EXPERIMENT_N

from .base import Question

_LABELS = ["A", "B", "C", "D"]


def _format_question(question: str, choices: list[str]) -> str:
    lines = [question]
    for label, choice in zip(_LABELS, choices):
        lines.append(f"{label}. {choice}")
    return "\n".join(lines)


def _stable_question_id(subject: str, question: str) -> str:
    digest = sha1(question.encode(DEFAULT_ENCODING)).hexdigest()[:10]
    return f"mmlu_{subject}_{digest}"


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
        questions.append(
            Question(
                id=_stable_question_id(row["subject"], row["question"]),
                difficulty="medium",
                category=row["subject"],
                question=_format_question(row["question"], row["choices"]),
                ground_truth=answer_letter,
                eval_mode="exact_match",
            )
        )
    return questions
