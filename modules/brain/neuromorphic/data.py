from __future__ import annotations

import json
from pathlib import Path
from typing import List


class DatasetLoader:
    """Load neuromorphic experiment assets from disk."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def load_config(self) -> str:
        return str(self.root / "config.json")

    def load_signal(self) -> str:
        return str(self.root / "signal.json")

    def load_target(self) -> str:
        return str(self.root / "target.json")

    def read_json(self, name: str) -> List[list[float]]:
        path = self.root / name
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError(f"{name} must be a JSON array")
        return [list(map(float, row)) for row in data]


__all__ = ["DatasetLoader"]
