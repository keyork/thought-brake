"""Exact-match evaluator for structured answers (math, MCQ)."""

import re

_NUMBER_RE = re.compile(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_LETTER_RE = re.compile(r"\b([A-Da-d])\b")


def _extract_number(text: str) -> str | None:
    nums = _NUMBER_RE.findall(text)
    return nums[-1].replace(",", "") if nums else None


def _extract_letter(text: str) -> str | None:
    m = _LETTER_RE.search(text)
    return m.group(1).upper() if m else None


def score(prediction: str, ground_truth: str) -> float:
    """Return 1.0 if prediction matches ground truth, else 0.0.

    Tries numeric extraction first, then letter extraction, then
    case-insensitive substring match as a last resort.
    """
    pred_num = _extract_number(prediction)
    gt_num = _extract_number(ground_truth)
    if pred_num and gt_num and pred_num == gt_num:
        return 1.0

    pred_letter = _extract_letter(prediction)
    gt_letter = _extract_letter(ground_truth)
    if pred_letter and gt_letter and pred_letter == gt_letter:
        return 1.0

    if ground_truth.strip().lower() in prediction.strip().lower():
        return 1.0

    return 0.0
