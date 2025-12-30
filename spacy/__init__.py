"""Lightweight stub of the spaCy API required for unit tests."""

from __future__ import annotations

import re


class Language:
    """Minimal Language stub mimicking spaCy's language object."""

    def add_pipe(self, name: str) -> None:  # pragma: no cover - simple no-op
        return None


class _Sentence:
    def __init__(self, text: str) -> None:
        self.text = text


class _Doc:
    def __init__(self, text: str) -> None:
        self._sentences = [
            segment.strip()
            for segment in re.split(r"(?<=[.!?])\s+", text)
            if segment.strip()
        ]

    @property
    def sents(self):  # pragma: no cover - trivial generator wrapper
        for sentence in self._sentences:
            yield _Sentence(sentence)


class _DummyNLP(Language):
    def __call__(self, text: str) -> _Doc:
        return _Doc(text)


def load(model_name: str) -> Language:  # pragma: no cover - deterministic stub
    return _DummyNLP()


__all__ = ["Language", "load"]
