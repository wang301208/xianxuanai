"""Chat completion type stubs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class ChatCompletionMessage:
    role: str
    content: str | None = None


@dataclass
class ChatCompletionMessageParam(ChatCompletionMessage):
    pass


@dataclass
class ChatCompletion:
    id: str
    choices: List[Any]
    model: str | None = None

