"""Tests for Phase 2 prefill."""

from unittest.mock import MagicMock

from thought_brake._prefill import (
    _compose_reasoning_excerpt,
    build_direct_messages,
    build_prefill_messages,
    clean_final_answer,
    run_prefill,
)
from thought_brake.config import EarlyStopConfig

from .conftest import MockStream, content_chunk, reasoning_chunk


def _mock_client(chunks):
    client = MagicMock()
    client.chat.completions.create.return_value = MockStream(chunks)
    return client


def test_build_prefill_messages_default_includes_user_prompt() -> None:
    cfg = EarlyStopConfig()
    original = [{"role": "user", "content": "question"}]
    msgs = build_prefill_messages(original, "partial reasoning", cfg)

    assert len(msgs) == 3
    assert msgs[0] == {"role": "user", "content": "question"}
    assert msgs[1]["role"] == "assistant"
    body = msgs[1]["content"]
    assert body.startswith(cfg.reasoning_start_tag)
    assert "partial reasoning" in body
    assert cfg.finalize_hint in body
    assert cfg.reasoning_end_tag in body
    assert body.endswith(cfg.phase2_answer_prefix)
    assert msgs[2] == {"role": "user", "content": cfg.final_answer_prompt}


def test_build_prefill_messages_skips_user_prompt_when_enabled() -> None:
    cfg = EarlyStopConfig(phase2_skip_user_prompt=True)
    msgs = build_prefill_messages([{"role": "user", "content": "q"}], "partial", cfg)

    assert len(msgs) == 2
    assert msgs[1]["role"] == "assistant"


def test_build_prefill_messages_uses_configured_tags() -> None:
    cfg = EarlyStopConfig(
        finalize_hint="\ndone",
        reasoning_start_tag="<reasoning>",
        reasoning_end_tag="</reasoning>",
        phase2_answer_prefix="ANSWER:",
    )
    msgs = build_prefill_messages([{"role": "user", "content": "question"}], "partial", cfg)

    assert msgs[1]["content"] == "<reasoning>partial\ndone</reasoning>ANSWER:"


def test_build_direct_messages_structure() -> None:
    cfg = EarlyStopConfig()
    original = [{"role": "user", "content": "盲人买剪刀"}]
    msgs = build_direct_messages(original, "盲人可以说话...", cfg)

    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert "盲人买剪刀" in msgs[0]["content"]
    assert "盲人可以说话" in msgs[0]["content"]


def test_build_direct_messages_extracts_last_user_message() -> None:
    cfg = EarlyStopConfig()
    original = [
        {"role": "system", "content": "你是一个助手"},
        {"role": "user", "content": "第一个问题"},
        {"role": "assistant", "content": "第一轮回答"},
        {"role": "user", "content": "第二个问题"},
    ]
    msgs = build_direct_messages(original, "some reasoning", cfg)
    assert "第二个问题" in msgs[0]["content"]
    assert "第一个问题" not in msgs[0]["content"]


def test_run_prefill_collects_content() -> None:
    chunks = [content_chunk("Final answer here.")]
    result = run_prefill(_mock_client(chunks), "model", [], "reasoning", EarlyStopConfig())
    assert result == "Final answer here."


def test_run_prefill_treats_leftover_reasoning_as_content() -> None:
    chunks = [reasoning_chunk("still thinking but accept as answer")]
    result = run_prefill(_mock_client(chunks), "model", [], "reasoning", EarlyStopConfig())
    assert "still thinking" in result


def test_run_prefill_original_messages_preserved() -> None:
    cfg = EarlyStopConfig(phase2_skip_user_prompt=False, phase2_mode="prefill")
    client = _mock_client([content_chunk("ok")])
    original = [{"role": "user", "content": "hi"}]
    run_prefill(client, "model", original, "reasoning text", cfg)

    call_args = client.chat.completions.create.call_args
    sent_messages = call_args.kwargs["messages"]
    assert sent_messages[0] == {"role": "user", "content": "hi"}
    assert sent_messages[1]["role"] == "assistant"
    assert sent_messages[2]["role"] == "user"


def test_run_prefill_direct_mode_sends_single_user_message() -> None:
    client = _mock_client([content_chunk("answer")])
    cfg = EarlyStopConfig(phase2_mode="direct")
    run_prefill(client, "model", [{"role": "user", "content": "q"}], "reasoning", cfg)

    call_kwargs = client.chat.completions.create.call_args.kwargs
    sent_messages = call_kwargs["messages"]
    assert len(sent_messages) == 1
    assert sent_messages[0]["role"] == "user"


def test_run_prefill_direct_mode_passes_enable_thinking_false() -> None:
    client = _mock_client([content_chunk("answer")])
    cfg = EarlyStopConfig(phase2_mode="direct")
    run_prefill(client, "model", [{"role": "user", "content": "q"}], "reasoning", cfg)

    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert call_kwargs["extra_body"]["enable_thinking"] is False


def test_run_prefill_direct_mode_no_thinking_flag_when_disabled() -> None:
    client = _mock_client([content_chunk("answer")])
    cfg = EarlyStopConfig(phase2_mode="direct", phase2_disable_thinking=False)
    run_prefill(client, "model", [{"role": "user", "content": "q"}], "reasoning", cfg)

    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert "extra_body" not in call_kwargs or "enable_thinking" not in call_kwargs.get(
        "extra_body", {}
    )


def test_run_prefill_default_does_not_set_temperature() -> None:
    client = _mock_client([content_chunk("ok")])
    run_prefill(client, "model", [], "reasoning", EarlyStopConfig())

    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert "temperature" not in call_kwargs


def test_run_prefill_default_does_not_set_max_tokens() -> None:
    client = _mock_client([content_chunk("ok")])
    run_prefill(client, "model", [], "reasoning", EarlyStopConfig())

    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert "max_tokens" not in call_kwargs


def test_run_prefill_sets_temperature_when_configured() -> None:
    client = _mock_client([content_chunk("ok")])
    cfg = EarlyStopConfig(phase2_temperature=0.0)
    run_prefill(client, "model", [], "reasoning", cfg)

    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert call_kwargs["temperature"] == 0.0


def test_run_prefill_sets_max_tokens_when_configured() -> None:
    client = _mock_client([content_chunk("ok")])
    cfg = EarlyStopConfig(phase2_max_tokens=200)
    run_prefill(client, "model", [], "reasoning", cfg)

    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert call_kwargs["max_tokens"] == 200


def test_clean_final_answer_keeps_plain_answer() -> None:
    assert clean_final_answer("答案是：水。") == "答案是：水。"


def test_clean_final_answer_strips_obvious_reasoning_draft() -> None:
    raw = (
        "1. **分析请求：** 用户问什么东西越洗越脏。\n\n"
        "2. **识别问题：** 这是脑筋急转弯。\n\n"
        "答案是：水。因为污垢会进入水里。"
    )

    assert clean_final_answer(raw) == "答案是：水。因为污垢会进入水里。"


def test_clean_final_answer_strips_meta_step_wrapper() -> None:
    raw = "4. **生成最终输出：**\n    * 白色白色"
    assert clean_final_answer(raw) == "白色"


def test_clean_final_answer_strips_verification_wrapper() -> None:
    raw = "5.  **最终确认：** 4小时4小时"
    assert clean_final_answer(raw) == "4小时"


def test_clean_final_answer_deduplicates_tail() -> None:
    assert clean_final_answer("火柴。火柴。") == "火柴。"
    assert clean_final_answer("白色白色") == "白色"
    assert clean_final_answer("4小时4小时") == "4小时"


def test_clean_final_answer_no_deduplicate_different_halves() -> None:
    assert clean_final_answer("猫和狗。狗和猫。") == "猫和狗。狗和猫。"


def test_clean_final_answer_deduplicate_long_repeat() -> None:
    assert (
        clean_final_answer("因为他是个光头。因为他是个光头。") == "因为他是个光头。"
    )


def test_compose_reasoning_excerpt_short_text_returns_full() -> None:
    text = "短推理内容"
    assert _compose_reasoning_excerpt(text, head_chars=150, tail_chars=200) == text


def test_compose_reasoning_excerpt_below_threshold_returns_head_only() -> None:
    text = "a" * 200
    result = _compose_reasoning_excerpt(text, head_chars=100, tail_chars=100)
    assert "……" not in result
    assert result == text[:100]


def test_compose_reasoning_excerpt_above_threshold_returns_head_tail() -> None:
    text = "a" * 600
    result = _compose_reasoning_excerpt(text, head_chars=100, tail_chars=100)
    assert "……" in result
    assert result.startswith("a")
    assert result.endswith("a")


def test_compose_reasoning_excerpt_zero_tail_returns_head() -> None:
    text = "a" * 600
    result = _compose_reasoning_excerpt(text, head_chars=100, tail_chars=0)
    assert "……" not in result
    assert len(result) == 100
