"""Minimal runnable example — requires a .env file with credentials."""

from thought_brake import EarlyStopConfig, ThoughtBrakeClient

client = ThoughtBrakeClient()

question = "一个盲人去买剪刀，一个聋哑人去买锤子，请问谁先买到？"

resp = client.chat(
    messages=[{"role": "user", "content": question}],
    config=EarlyStopConfig.for_task("chat"),
)

print(f"Answer:          {resp.content}")
print(f"Reasoning chars: {resp.metrics.reasoning_chars}")
print(f"Stop reason:     {resp.metrics.stop_reason}")
print(f"Phase 2 used:    {resp.metrics.phase2_used}")
