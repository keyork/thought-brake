"""LLM-as-judge evaluator for open-ended answers."""

import json
import re

from openai import OpenAI

from experiments.config import (
    LLM_JUDGE_FALLBACK_SCORE,
    LLM_JUDGE_SYSTEM_PROMPT,
    LLM_JUDGE_TEMPERATURE,
    LLM_JUDGE_USER_TEMPLATE,
)

_SCORE_RE = re.compile(r'"score"\s*:\s*([0-9.]+)')


def score(
    client: OpenAI,
    model: str,
    question: str,
    reference: str,
    prediction: str,
) -> float:
    """Return a quality score in [0, 1] judged by the LLM."""
    user_msg = LLM_JUDGE_USER_TEMPLATE.format(
        question=question,
        reference=reference,
        prediction=prediction,
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": LLM_JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=LLM_JUDGE_TEMPERATURE,
        )
        raw = resp.choices[0].message.content or ""
        # Try JSON parse first, fall back to regex
        try:
            data = json.loads(raw)
            return float(data["score"])
        except (json.JSONDecodeError, KeyError):
            m = _SCORE_RE.search(raw)
            return float(m.group(1)) if m else LLM_JUDGE_FALLBACK_SCORE
    except Exception:
        return LLM_JUDGE_FALLBACK_SCORE
