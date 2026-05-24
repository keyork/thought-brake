from dataclasses import dataclass
from typing import cast

from .types import DetectorName, Phase2Mode

DEFAULT_ENCODING = "utf-8"

# (soft_budget, hard_limit) in characters
_PRESETS: dict[str, tuple[int, int]] = {
    "chat": (200, 400),
    "qa": (500, 1000),
    "math": (1500, 3000),
    "complex": (3000, 6000),
}


def _env_str(name: str, default: str) -> str:
    import os

    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    import os

    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_bool(name: str, default: bool) -> bool:
    import os

    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default

    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _env_detector(name: str, default: DetectorName) -> DetectorName:
    raw = _env_str(name, default)
    if raw in {"budget", "compression", "none"}:
        return cast(DetectorName, raw)
    raise ValueError(f"{name} must be one of: budget, compression, none")


@dataclass
class EarlyStopConfig:
    detector: DetectorName = "budget"
    soft_budget: int = 300
    hard_limit: int = 600
    compression_baseline_chars: int = 200
    compression_recent_chars: int = 200
    compression_theta_crd: float = 0.7
    compression_theta_lz: float = 0.5
    compression_consecutive_windows: int = 2
    finalize_hint: str = "\n\n好，已经想清楚了，直接给出最终答案。"
    reasoning_start_tag: str = "hã\n"
    reasoning_end_tag: str = "\n boxed\n\n"
    final_answer_prompt: str = (
        "只输出最终答案。不要复述推理过程，不要列分析步骤，不要提到你已经思考。"
    )
    phase2_mode: Phase2Mode = "direct"
    phase2_direct_template: str = (
        "以下是问题和已完成的思考过程。请直接给出最终答案，不需要再分析。\n\n"
        "问题：{question}\n\n"
        "已完成的思考：\n{reasoning}\n\n"
        "最终答案："
    )
    phase2_disable_thinking: bool = True
    phase2_direct_head_chars: int = 150
    phase2_direct_tail_chars: int = 200
    phase2_answer_prefix: str = ""
    phase2_skip_user_prompt: bool = False
    phase2_temperature: float = -1
    phase2_max_tokens: int = 0
    clean_phase2_answer: bool = True
    sentence_end_pattern: str = r"[。！？!?\n]"
    fallback_excerpt_chars: int = 300
    fallback_assistant_template: str = "[已思考]: {reasoning}..."
    fallback_user_prompt: str = "请直接给出最终答案，简明扼要。"
    enable: bool = True
    fallback_on_phase2_fail: bool = True

    @classmethod
    def for_task(cls, task: str) -> "EarlyStopConfig":
        """Return a config with budgets tuned for the given task type.

        task: one of 'chat', 'qa', 'math', 'complex'
        """
        soft, hard = _PRESETS.get(task, (300, 600))
        return cls(soft_budget=soft, hard_limit=hard)

    @classmethod
    def from_env(cls, task: str | None = None) -> "EarlyStopConfig":
        """Return config using THOUGHT_BRAKE_* environment overrides.

        Call load_dotenv before this when .env values should be considered.
        """
        base = cls.for_task(task) if task else cls()
        return cls(
            detector=_env_detector("THOUGHT_BRAKE_DETECTOR", base.detector),
            soft_budget=_env_int("THOUGHT_BRAKE_SOFT_BUDGET", base.soft_budget),
            hard_limit=_env_int("THOUGHT_BRAKE_HARD_LIMIT", base.hard_limit),
            compression_baseline_chars=_env_int(
                "THOUGHT_BRAKE_COMPRESSION_BASELINE_CHARS",
                base.compression_baseline_chars,
            ),
            compression_recent_chars=_env_int(
                "THOUGHT_BRAKE_COMPRESSION_RECENT_CHARS",
                base.compression_recent_chars,
            ),
            compression_theta_crd=float(
                _env_str("THOUGHT_BRAKE_COMPRESSION_THETA_CRD", str(base.compression_theta_crd))
            ),
            compression_theta_lz=float(
                _env_str("THOUGHT_BRAKE_COMPRESSION_THETA_LZ", str(base.compression_theta_lz))
            ),
            compression_consecutive_windows=_env_int(
                "THOUGHT_BRAKE_COMPRESSION_CONSECUTIVE_WINDOWS",
                base.compression_consecutive_windows,
            ),
            finalize_hint=_env_str("THOUGHT_BRAKE_FINALIZE_HINT", base.finalize_hint),
            reasoning_start_tag=_env_str(
                "THOUGHT_BRAKE_REASONING_START_TAG", base.reasoning_start_tag
            ),
            reasoning_end_tag=_env_str(
                "THOUGHT_BRAKE_REASONING_END_TAG", base.reasoning_end_tag
            ),
            final_answer_prompt=_env_str(
                "THOUGHT_BRAKE_FINAL_ANSWER_PROMPT", base.final_answer_prompt
            ),
            clean_phase2_answer=_env_bool(
                "THOUGHT_BRAKE_CLEAN_PHASE2_ANSWER", base.clean_phase2_answer
            ),
            sentence_end_pattern=_env_str(
                "THOUGHT_BRAKE_SENTENCE_END_PATTERN", base.sentence_end_pattern
            ),
            fallback_excerpt_chars=_env_int(
                "THOUGHT_BRAKE_FALLBACK_EXCERPT_CHARS", base.fallback_excerpt_chars
            ),
            fallback_assistant_template=_env_str(
                "THOUGHT_BRAKE_FALLBACK_ASSISTANT_TEMPLATE",
                base.fallback_assistant_template,
            ),
            fallback_user_prompt=_env_str(
                "THOUGHT_BRAKE_FALLBACK_USER_PROMPT", base.fallback_user_prompt
            ),
            enable=_env_bool("THOUGHT_BRAKE_ENABLE", base.enable),
            fallback_on_phase2_fail=_env_bool(
                "THOUGHT_BRAKE_FALLBACK_ON_PHASE2_FAIL",
                base.fallback_on_phase2_fail,
            ),
        )
