from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

REGISTRY_FILE = Path("artifacts/registry.json")


class ModelRegistry:
    """Simple JSON-backed registry for trained models.

    The registry keeps track of all trained versions along with their
    evaluation metrics and applied compression level. It also tracks the
    currently deployed version and a history of previous deployments for
    rollback purposes.
    """

    def __init__(self, path: Path = REGISTRY_FILE) -> None:
        self.path = path
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        else:
            self.data = {"current": None, "history": [], "models": {}}

    def _save(self) -> None:
        self.path.parent.mkdir(exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    def register(
        self,
        version: str,
        metrics: Dict[str, float],
        compression: int | None,
    ) -> None:
        """Register a model version with associated metadata."""
        self.data["models"][version] = {
            "version": version,
            "metrics": metrics,
            "compression": compression,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._save()

    def set_current(self, version: str) -> None:
        """Mark ``version`` as the currently deployed model."""
        current = self.data.get("current")
        if current:
            self.data.setdefault("history", []).append(current)
        self.data["current"] = version
        self._save()

    def current(self) -> Optional[Dict]:
        """Return metadata for the currently deployed model."""
        cur = self.data.get("current")
        if cur is None:
            return None
        return self.data["models"].get(cur)

    def rollback(self) -> Optional[str]:
        """Rollback to the previous deployed model, if any.

        Returns the version rolled back to or ``None`` if no previous
        deployment exists.
        """
        history = self.data.get("history", [])
        if not history:
            return None
        prev = history.pop()
        self.data["current"] = prev
        self._save()
        return prev

    def get(self, version: str) -> Optional[Dict]:
        """Return metadata for ``version`` if present."""
        return self.data["models"].get(version)
