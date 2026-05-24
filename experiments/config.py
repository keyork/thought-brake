"""Shared configuration for experiment scripts."""

from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
REPORT_DIR = BASE_DIR / "report"

DEFAULT_ENCODING = "utf-8"
DEFAULT_BUDGETS = "0,100,200,500,1000"
DEFAULT_EXPERIMENT_N = 100
DEFAULT_GSM8K_SPLIT = "test"
DEFAULT_WORKERS = 10
BASELINE_BUDGET = 0
RESULT_SCHEMA_VERSION = 2

LLM_JUDGE_TEMPERATURE = 0.0
LLM_JUDGE_FALLBACK_SCORE = 0.5
LLM_JUDGE_SYSTEM_PROMPT = """\
你是一个严格的答案质量评判者。
你会收到一道题、一个参考答案（可能是标准答案或基线模型的回答），以及一个候选答案。
请判断候选答案的质量，输出 JSON：{"score": <0.0到1.0>, "reason": "<一句话说明>"}
评分标准：
  1.0 = 完全正确，包含关键信息
  0.5 = 部分正确，有核心要点但不完整或有小错
  0.0 = 错误或缺失关键答案
只输出 JSON，不要其他内容。"""
LLM_JUDGE_USER_TEMPLATE = "题目：{question}\n参考答案：{reference}\n候选答案：{prediction}"
