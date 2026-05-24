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


def reasoning_chunk(text: str) -> MockChunk:
    delta = MockDelta(
        content=None,
        reasoning_content=text,
        model_extra={"reasoning_content": text},
    )
    return MockChunk(choices=[MockChoice(delta=delta)])


def content_chunk(text: str) -> MockChunk:
    return MockChunk(choices=[MockChoice(delta=MockDelta(content=text))])


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
