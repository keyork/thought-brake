from dataclasses import dataclass
from typing import Literal


@dataclass
class Question:
    id: str
    difficulty: Literal["easy", "medium", "hard"]
    category: str           # "math" | "riddle" | "factual"
    question: str
    ground_truth: str       # canonical answer string
    eval_mode: Literal["exact_match", "llm_judge"]
