"""Load a subset of GSM8K from HuggingFace datasets.

GSM8K answers are in the format "...#### <number>".
We extract the number as ground truth and use exact_match evaluation.
"""

import re

from experiments.config import DEFAULT_EXPERIMENT_N, DEFAULT_GSM8K_SPLIT

from .base import Question

_ANSWER_RE = re.compile(r"####\s*([\d,]+)")


def _extract_number(text: str) -> str:
    m = _ANSWER_RE.search(text)
    return m.group(1).replace(",", "") if m else text.strip()


def load(n: int = DEFAULT_EXPERIMENT_N, split: str = DEFAULT_GSM8K_SPLIT) -> list[Question]:
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError("Run: uv sync --group experiments") from e

    ds = load_dataset("openai/gsm8k", "main", split=split)
    questions = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        difficulty = "easy" if i < n // 3 else ("medium" if i < 2 * n // 3 else "hard")
        questions.append(
            Question(
                id=f"gsm8k_{i:04d}",
                difficulty=difficulty,
                category="math",
                question=row["question"],
                ground_truth=_extract_number(row["answer"]),
                eval_mode="exact_match",
            )
        )
    return questions
