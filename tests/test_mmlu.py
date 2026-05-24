from experiments.datasets.mmlu import _stable_question_id


def test_stable_question_id_is_deterministic() -> None:
    question = "What is the capital of France?"

    assert _stable_question_id("geography", question) == _stable_question_id(
        "geography", question
    )
    assert _stable_question_id("geography", question).startswith("mmlu_geography_")
