from experiments.datasets import riddles


def test_riddles_load_can_limit_count() -> None:
    questions = riddles.load(n=10)

    assert len(questions) == 10
    assert questions[0].id == "r001"
    assert questions[-1].id == "r010"


def test_riddles_dataset_has_expanded_sample() -> None:
    questions = riddles.load()

    assert len(questions) >= 100
    assert len({question.id for question in questions}) == len(questions)
