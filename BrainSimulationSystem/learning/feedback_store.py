"""
Feedback logging utilities for internal language evaluation.

This module captures low-confidence comprehension cases so they can be reviewed
and folded back into future training runs without relying on external LLM
signals.
"""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional


class FeedbackLogger:
    """Append-only JSONL logger with an in-memory cache for quick inspection."""

    def __init__(
        self,
        path: Optional[Path],
        *,
        enabled: bool = True,
        max_cache: int = 256,
    ) -> None:
        self.path = Path(path) if path else None
        self.enabled = enabled and self.path is not None
        self.max_cache = max_cache
        self._cache: Deque[Dict[str, Any]] = deque(maxlen=max_cache)
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, entry: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        payload = dict(entry)
        payload.setdefault("timestamp", time.time())
        self._cache.append(payload)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")

    def recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        return list(self._cache)[-limit:]

    def flush_to(self, target: Path) -> None:
        """Dump the in-memory cache to ``target`` in JSON format."""
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(list(self._cache), ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = ["FeedbackLogger"]
