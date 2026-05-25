import json

from experiments.config import DATA_DIR, DEFAULT_ENCODING

from .base import Question

_DATA_FILE = DATA_DIR / "riddles.jsonl"


def load(difficulties: list[str] | None = None, n: int | None = None) -> list[Question]:
    questions = []
    with _DATA_FILE.open(encoding=DEFAULT_ENCODING) as f:
        for line in f:
            row = json.loads(line)
            if difficulties and row["difficulty"] not in difficulties:
                continue
            questions.append(
                Question(
                    id=row["id"],
                    difficulty=row["difficulty"],
                    category="riddle",
                    question=row["question"],
                    ground_truth=row["answer"],
                    eval_mode=row.get("eval_mode", "llm_judge"),
                )
            )
            if n is not None and len(questions) >= n:
                break
    return questions
