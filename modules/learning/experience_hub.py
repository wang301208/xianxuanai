from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List


@dataclass
class EpisodeRecord:
    task_id: str
    policy_version: str
    total_reward: float
    steps: int
    success: bool
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    metadata: Dict[str, Any] = field(default_factory=dict)
    trajectory_path: str | None = None


@dataclass
class DemonstrationRecord:
    """Human/expert demonstration trajectory metadata."""

    task_id: str
    source: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    trajectory_path: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExperienceHub:
    """Lightweight episode store for RL training loops."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "episodes.jsonl"
        self.demo_path = self.root / "demonstrations.jsonl"

    def append(self, record: EpisodeRecord) -> None:
        with self.index_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def append_demonstration(self, demo: DemonstrationRecord) -> None:
        with self.demo_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(demo), ensure_ascii=False) + "\n")

    def latest(self, limit: int = 50) -> List[EpisodeRecord]:
        if not self.index_path.exists():
            return []
        lines = self.index_path.read_text(encoding="utf-8").strip().splitlines()
        records: List[EpisodeRecord] = []
        for line in reversed(lines):
            if not line:
                continue
            data = json.loads(line)
            records.append(EpisodeRecord(**data))
            if len(records) >= limit:
                break
        return list(reversed(records))

    def load_for_training(self, min_reward: float | None = None) -> Iterable[Dict[str, Any]]:
        for record in self.latest():
            if min_reward is not None and record.total_reward < min_reward:
                continue
            if record.trajectory_path:
                path = Path(record.trajectory_path)
                if path.exists():
                    yield json.loads(path.read_text(encoding="utf-8"))

    def load_demonstrations(self, limit: int = 20) -> Iterable[Dict[str, Any]]:
        if not self.demo_path.exists():
            return []
        lines = self.demo_path.read_text(encoding="utf-8").strip().splitlines()
        demos: List[DemonstrationRecord] = []
        for line in reversed(lines):
            if not line:
                continue
            data = json.loads(line)
            demos.append(DemonstrationRecord(**data))
            if len(demos) >= limit:
                break
        demos = list(reversed(demos))
        for demo in demos:
            if demo.trajectory_path:
                path = Path(demo.trajectory_path)
                if path.exists():
                    yield json.loads(path.read_text(encoding="utf-8"))
