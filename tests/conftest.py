"""Shared fixtures and mock chunk builders."""

from dataclasses import dataclass
from typing import Any


@dataclass
class MockDelta:
    content: str | None = None
    reasoning_content: str | None = None
    model_extra: dict[str, Any] | None = None


@dataclass
class MockChoice:
    delta: MockDelta


@dataclass
class MockChunk:
    choices: list[MockChoice]
    usage: Any | None = None


@dataclass
class MockUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    completion_tokens_details: Any | None = None


@dataclass
class MockCompletionTokensDetails:
    reasoning_tokens: int | None = None


def reasoning_chunk(text: str) -> MockChunk:
    delta = MockDelta(
        content=None,
        reasoning_content=text,
        model_extra={"reasoning_content": text},
    )
    return MockChunk(choices=[MockChoice(delta=delta)])


def content_chunk(text: str) -> MockChunk:
    return MockChunk(choices=[MockChoice(delta=MockDelta(content=text))])


def usage_chunk(
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    reasoning_tokens: int | None = None,
) -> MockChunk:
    return MockChunk(
        choices=[],
        usage=MockUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            completion_tokens_details=MockCompletionTokensDetails(reasoning_tokens),
        ),
    )


class MockStream:
    """Minimal context-manager iterator that yields pre-built chunks."""

    def __init__(self, chunks: list[MockChunk]) -> None:
        self._chunks = chunks

    def __enter__(self) -> "MockStream":
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def __iter__(self):  # type: ignore[override]
        return iter(self._chunks)


class FailingStream(MockStream):
    """Stream that raises after yielding pre-built chunks."""

    def __iter__(self):  # type: ignore[override]
        yield from self._chunks
        raise ConnectionError("stream interrupted")
