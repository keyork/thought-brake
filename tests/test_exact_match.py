"""Tests for experiment exact-match scoring."""

from experiments.evaluate import exact_match


def test_exact_match_handles_comma_separated_numbers() -> None:
    prediction = "Josh made a profit of **$70,000**."

    assert exact_match.score(prediction, "70000") == 1.0


def test_exact_match_uses_last_number_for_final_answer() -> None:
    prediction = "40 + 10 + 3 + 4 = **57.00**"

    assert exact_match.score(prediction, "57") == 1.0
