from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ActionLogger:
    """Append-only JSONL logger for agent actions and diagnostics."""

    path: str

    def __post_init__(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload: Dict[str, Any], *, flush: bool = True) -> None:
        record = dict(payload)
        line = json.dumps(record, ensure_ascii=False)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            if flush:
                handle.flush()

