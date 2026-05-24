from unittest.mock import patch

import pytest

from thought_brake.config import EarlyStopConfig


def test_defaults() -> None:
    cfg = EarlyStopConfig()
    assert cfg.detector == "budget"
    assert cfg.soft_budget == 300
    assert cfg.hard_limit == 600
    assert cfg.compression_baseline_chars == 200
    assert cfg.compression_recent_chars == 200
    assert cfg.compression_theta_crd == 0.7
    assert cfg.compression_theta_lz == 0.5
    assert cfg.compression_consecutive_windows == 2
    assert cfg.final_answer_prompt
    assert cfg.clean_phase2_answer is True
    assert cfg.phase2_mode == "direct"
    assert cfg.fallback_excerpt_chars == 300
    assert cfg.enable is True
    assert cfg.fallback_on_phase2_fail is True


def test_for_task_chat() -> None:
    cfg = EarlyStopConfig.for_task("chat")
    assert cfg.soft_budget == 200
    assert cfg.hard_limit == 400


def test_for_task_math() -> None:
    cfg = EarlyStopConfig.for_task("math")
    assert cfg.soft_budget == 1500
    assert cfg.hard_limit == 3000


def test_for_task_unknown_falls_back_to_defaults() -> None:
    cfg = EarlyStopConfig.for_task("unknown_task_type")
    assert cfg.soft_budget == 300
    assert cfg.hard_limit == 600


def test_from_env_overrides_runtime_config() -> None:
    with patch.dict(
        "os.environ",
        {
            "THOUGHT_BRAKE_SOFT_BUDGET": "123",
            "THOUGHT_BRAKE_DETECTOR": "semantic",
            "THOUGHT_BRAKE_HARD_LIMIT": "456",
            "THOUGHT_BRAKE_COMPRESSION_BASELINE_CHARS": "50",
            "THOUGHT_BRAKE_COMPRESSION_RECENT_CHARS": "75",
            "THOUGHT_BRAKE_COMPRESSION_THETA_CRD": "0.8",
            "THOUGHT_BRAKE_COMPRESSION_THETA_LZ": "0.6",
            "THOUGHT_BRAKE_COMPRESSION_CONSECUTIVE_WINDOWS": "3",
            "THOUGHT_BRAKE_NGRAM_SIZE": "5",
            "THOUGHT_BRAKE_NGRAM_WINDOW_CHARS": "150",
            "THOUGHT_BRAKE_NGRAM_THRESHOLD": "0.55",
            "THOUGHT_BRAKE_NGRAM_CONSECUTIVE_WINDOWS": "4",
            "THOUGHT_BRAKE_KEYWORD_WINDOW_CHARS": "250",
            "THOUGHT_BRAKE_KEYWORD_TRIGGER_THRESHOLD": "0.04",
            "THOUGHT_BRAKE_KEYWORD_CONSECUTIVE_WINDOWS": "3",
            "THOUGHT_BRAKE_SEMANTIC_WINDOW_CHARS": "180",
            "THOUGHT_BRAKE_SEMANTIC_JACCARD_THRESHOLD": "0.45",
            "THOUGHT_BRAKE_SEMANTIC_CONSECUTIVE_WINDOWS": "4",
            "THOUGHT_BRAKE_SEMANTIC_MIN_WORDS": "6",
            "THOUGHT_BRAKE_FINALIZE_HINT": "\nfinalize now",
            "THOUGHT_BRAKE_REASONING_START_TAG": "<reasoning>",
            "THOUGHT_BRAKE_REASONING_END_TAG": "</reasoning>",
            "THOUGHT_BRAKE_FINAL_ANSWER_PROMPT": "final only",
            "THOUGHT_BRAKE_CLEAN_PHASE2_ANSWER": "false",
            "THOUGHT_BRAKE_PHASE2_MODE": "prefill",
            "THOUGHT_BRAKE_PHASE2_DIRECT_TEMPLATE": "q={question} r={reasoning}",
            "THOUGHT_BRAKE_PHASE2_DISABLE_THINKING": "false",
            "THOUGHT_BRAKE_PHASE2_EXTRA_BODY": '{"thinking": {"type": "disabled"}}',
            "THOUGHT_BRAKE_PHASE2_DIRECT_CONVERSATION_CHARS": "900",
            "THOUGHT_BRAKE_PHASE2_DIRECT_HEAD_CHARS": "80",
            "THOUGHT_BRAKE_PHASE2_DIRECT_TAIL_CHARS": "90",
            "THOUGHT_BRAKE_PHASE2_ANSWER_PREFIX": "ANSWER:",
            "THOUGHT_BRAKE_PHASE2_SKIP_USER_PROMPT": "true",
            "THOUGHT_BRAKE_PHASE2_TEMPERATURE": "0.0",
            "THOUGHT_BRAKE_PHASE2_MAX_TOKENS": "128",
            "THOUGHT_BRAKE_SENTENCE_END_PATTERN": r"[.!?]",
            "THOUGHT_BRAKE_FALLBACK_EXCERPT_CHARS": "25",
            "THOUGHT_BRAKE_FALLBACK_ASSISTANT_TEMPLATE": "partial={reasoning}",
            "THOUGHT_BRAKE_FALLBACK_USER_PROMPT": "answer only",
            "THOUGHT_BRAKE_ENABLE": "false",
            "THOUGHT_BRAKE_FALLBACK_ON_PHASE2_FAIL": "0",
        },
        clear=True,
    ):
        cfg = EarlyStopConfig.from_env()

    assert cfg.soft_budget == 123
    assert cfg.detector == "semantic"
    assert cfg.hard_limit == 456
    assert cfg.compression_baseline_chars == 50
    assert cfg.compression_recent_chars == 75
    assert cfg.compression_theta_crd == 0.8
    assert cfg.compression_theta_lz == 0.6
    assert cfg.compression_consecutive_windows == 3
    assert cfg.ngram_size == 5
    assert cfg.ngram_window_chars == 150
    assert cfg.ngram_threshold == 0.55
    assert cfg.ngram_consecutive_windows == 4
    assert cfg.keyword_window_chars == 250
    assert cfg.keyword_trigger_threshold == 0.04
    assert cfg.keyword_consecutive_windows == 3
    assert cfg.semantic_window_chars == 180
    assert cfg.semantic_jaccard_threshold == 0.45
    assert cfg.semantic_consecutive_windows == 4
    assert cfg.semantic_min_words == 6
    assert cfg.finalize_hint == "\nfinalize now"
    assert cfg.reasoning_start_tag == "<reasoning>"
    assert cfg.reasoning_end_tag == "</reasoning>"
    assert cfg.final_answer_prompt == "final only"
    assert cfg.clean_phase2_answer is False
    assert cfg.phase2_mode == "prefill"
    assert cfg.phase2_direct_template == "q={question} r={reasoning}"
    assert cfg.phase2_disable_thinking is False
    assert cfg.phase2_extra_body == {"thinking": {"type": "disabled"}}
    assert cfg.phase2_direct_conversation_chars == 900
    assert cfg.phase2_direct_head_chars == 80
    assert cfg.phase2_direct_tail_chars == 90
    assert cfg.phase2_answer_prefix == "ANSWER:"
    assert cfg.phase2_skip_user_prompt is True
    assert cfg.phase2_temperature == 0.0
    assert cfg.phase2_max_tokens == 128
    assert cfg.sentence_end_pattern == r"[.!?]"
    assert cfg.fallback_excerpt_chars == 25
    assert cfg.fallback_assistant_template == "partial={reasoning}"
    assert cfg.fallback_user_prompt == "answer only"
    assert cfg.enable is False
    assert cfg.fallback_on_phase2_fail is False


def test_from_env_invalid_bool_raises() -> None:
    with patch.dict("os.environ", {"THOUGHT_BRAKE_ENABLE": "maybe"}, clear=True):
        with pytest.raises(ValueError, match="THOUGHT_BRAKE_ENABLE"):
            EarlyStopConfig.from_env()


def test_from_env_invalid_detector_raises() -> None:
    with patch.dict("os.environ", {"THOUGHT_BRAKE_DETECTOR": "compression"}, clear=True):
        assert EarlyStopConfig.from_env().detector == "compression"

    with patch.dict("os.environ", {"THOUGHT_BRAKE_DETECTOR": "keyword"}, clear=True):
        assert EarlyStopConfig.from_env().detector == "keyword"

    with patch.dict("os.environ", {"THOUGHT_BRAKE_DETECTOR": "unknown"}, clear=True):
        with pytest.raises(ValueError, match="THOUGHT_BRAKE_DETECTOR"):
            EarlyStopConfig.from_env()


def test_from_env_invalid_phase2_extra_body_raises() -> None:
    with patch.dict("os.environ", {"THOUGHT_BRAKE_PHASE2_EXTRA_BODY": "[]"}, clear=True):
        with pytest.raises(ValueError, match="THOUGHT_BRAKE_PHASE2_EXTRA_BODY"):
            EarlyStopConfig.from_env()
