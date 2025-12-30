"""Type definitions for the OpenAI stub."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class CreateEmbeddingResponse:
    data: List[object] | None = None

