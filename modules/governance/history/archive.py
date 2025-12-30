"""Archive proposals, approvals and outcomes for meta-analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class HistoryArchive:
    """Simple JSONL based archive."""

    def __init__(self, base_path: Path | str = "governance_history") -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _append(self, filename: str, record: Dict[str, Any]) -> None:
        path = self.base_path / filename
        with path.open("a", encoding="utf-8") as f:
            json.dump(record, f)
            f.write("\n")

    def _load(self, filename: str) -> List[Dict[str, Any]]:
        path = self.base_path / filename
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    # ------------------------------------------------------------------
    # record helpers
    # ------------------------------------------------------------------
    def record_proposal(self, proposal: Dict[str, Any]) -> None:
        self._append("proposals.jsonl", proposal)

    def record_approval(self, approval: Dict[str, Any]) -> None:
        self._append("approvals.jsonl", approval)

    def record_outcome(self, outcome: Dict[str, Any]) -> None:
        self._append("outcomes.jsonl", outcome)

    # ------------------------------------------------------------------
    # retrieval helpers
    # ------------------------------------------------------------------
    def proposals(self) -> List[Dict[str, Any]]:
        return self._load("proposals.jsonl")

    def approvals(self) -> List[Dict[str, Any]]:
        return self._load("approvals.jsonl")

    def outcomes(self) -> List[Dict[str, Any]]:
        return self._load("outcomes.jsonl")
