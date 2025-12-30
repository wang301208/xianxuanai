"""Strategist agent that extracts strategic principles from task logs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple, Optional
from datetime import datetime
from threading import Timer
from heapq import heappop, heappush
import json

from capability.meta_skill import META_SKILL_STRATEGY_EVOLUTION
from capability.skill_library import SkillLibrary
from common.async_utils import run_async

from . import Agent


class Strategist(Agent):
    """Aggregates logs, derives principles, and writes them to the charter."""

    def __init__(
        self,
        charter_dir: Path | str = Path("governance/charter"),
        metrics_dir: Path | str = Path("governance/metrics/strategist"),
        skill_repo: Path | str = Path("."),
    ) -> None:
        self.charter_dir = Path(charter_dir)
        self.charter_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        # Local library of meta-skills consulted during analysis
        self.library = SkillLibrary(skill_repo)

    def perform(self, logs: Iterable[Path] | None = None) -> Path:
        # Load strategist reasoning template to guide log analysis
        template, _meta = run_async(
            self.library.get_skill(META_SKILL_STRATEGY_EVOLUTION)
        )
        lines: List[str] = template.splitlines()
        for log in logs or []:
            path = Path(log)
            if path.exists():
                lines.extend(path.read_text().splitlines())

        principles = self._extract_principles(lines)
        charter_path = self.charter_dir / "strategic_principles.md"
        charter_path.write_text("\n".join(principles) + "\n")

        # Determine an executable action sequence using A* planning
        actions = self._plan_actions(principles)
        plan_path = self.charter_dir / "strategic_plan.md"
        plan_path.write_text("\n".join(f"- {step}" for step in actions) + "\n")

        # Reflect on recent performance metrics
        self.reflect(logs=logs, principles=principles)
        return plan_path

    def reflect(
        self,
        logs: Iterable[Path] | None = None,
        principles: List[str] | None = None,
    ) -> Path:
        """Compute performance metrics from historical task logs."""
        lines: List[str] = []
        for log in logs or []:
            path = Path(log)
            if path.exists():
                lines.extend(path.read_text().splitlines())

        successes = sum(1 for line in lines if "success" in line.lower())
        failures = sum(1 for line in lines if "fail" in line.lower())
        total = successes + failures
        success_rate = successes / total if total else 0.0

        existing = sorted(self.metrics_dir.glob("*.json"))
        prev_success = 0.0
        prev_principles = 0
        if existing:
            prev_data = json.loads(existing[-1].read_text())
            prev_success = prev_data.get("success_rate", 0.0)
            prev_principles = prev_data.get("principles_count", 0)

        success_rate_change = success_rate - prev_success
        principles_count = len(principles or [])
        principle_derivation_velocity = principles_count - prev_principles

        data = {
            "successes": successes,
            "failures": failures,
            "success_rate": success_rate,
            "success_rate_change": success_rate_change,
            "principles_count": principles_count,
            "principle_derivation_velocity": principle_derivation_velocity,
        }

        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        metrics_path = self.metrics_dir / f"{timestamp}.json"
        metrics_path.write_text(json.dumps(data, indent=2))

        history_path = self.metrics_dir / "history.csv"
        is_new = not history_path.exists()
        with history_path.open("a") as f:
            if is_new:
                f.write("timestamp,principles_count,success_rate\n")
            f.write(f"{timestamp},{principles_count},{success_rate}\n")

        self._update_model(principles_count, success_rate)

        # Determine if performance has stagnated and a meta-upgrade is needed
        if existing and (
            success_rate_change <= 0 or principle_derivation_velocity <= 0
        ):
            reasons: List[str] = []
            if success_rate_change <= 0:
                reasons.append(
                    f"Success rate stagnated at {success_rate:.2%}"
                )
            if principle_derivation_velocity <= 0:
                reasons.append(
                    "No new strategic principles were derived"
                )
            self.request_meta_upgrade("; ".join(reasons))

        return metrics_path

    def schedule_reflection(
        self, interval_hours: int = 24, logs: Iterable[Path] | None = None
    ) -> Timer:
        """Schedule recurring reflection runs via a simple timer."""

        def _run() -> None:
            self.reflect(logs=logs)
            self.schedule_reflection(interval_hours, logs)

        timer = Timer(interval_hours * 3600, _run)
        timer.daemon = True
        timer.start()
        return timer

    def request_meta_upgrade(self, reason: str) -> Path:
        """Create a meta-upgrade ticket summarizing current limitations."""
        tickets_dir = Path("evolution/meta_tickets")
        tickets_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        ticket_path = tickets_dir / f"{timestamp}.md"
        content = (
            f"# Meta Ticket\n\n"
            f"- Created: {timestamp}\n\n"
            f"## Limitation\n{reason}\n\n"
            "## Desired Improvements\n\n"
            "- [ ] Detail proposed improvement\n"
        )
        ticket_path.write_text(content)
        return ticket_path

    def _extract_principles(self, lines: List[str]) -> List[str]:
        unique: List[str] = []
        for line in lines:
            line = line.strip()
            if line and line not in unique:
                unique.append(f"- {line}")
        if not unique:
            unique.append("- No actionable principles found.")
            return unique

        model = self._load_model()
        if model is None:
            return unique
        w, b = model
        best_idx = len(unique)
        best_pred = float("-inf")
        for n in range(1, len(unique) + 1):
            pred = w * n + b
            if pred > best_pred:
                best_pred = pred
                best_idx = n
        return unique[:best_idx]

    def _load_model(self) -> Optional[Tuple[float, float]]:
        model_path = self.metrics_dir / "model.json"
        if model_path.exists():
            data = json.loads(model_path.read_text())
            return data.get("w", 0.0), data.get("b", 0.0)
        return None

    def _save_model(self, w: float, b: float) -> None:
        model_path = self.metrics_dir / "model.json"
        model_path.write_text(json.dumps({"w": w, "b": b}, indent=2))

    def _update_model(self, x: float, y: float, lr: float = 0.01) -> None:
        model = self._load_model()
        if model is None:
            w, b = 0.0, 0.0
        else:
            w, b = model
        pred = w * x + b
        error = pred - y
        w -= lr * error * x
        b -= lr * error
        self._save_model(w, b)

    def _plan_actions(self, principles: List[str]) -> List[str]:
        """Generate an ordered action plan from principles using A* search."""

        def heuristic(idx: int) -> int:
            # Remaining principles act as a simple distance heuristic
            return len(principles) - idx

        frontier: List[Tuple[int, int, List[str]]] = []
        heappush(frontier, (heuristic(0), 0, []))
        visited = set()

        while frontier:
            _score, idx, path = heappop(frontier)
            if idx == len(principles):
                return path
            if idx in visited:
                continue
            visited.add(idx)

            principle_text = principles[idx].lstrip("- ").strip()
            step = f"Act on: {principle_text}"
            new_path = path + [step]
            cost = len(new_path)
            heappush(frontier, (cost + heuristic(idx + 1), idx + 1, new_path))

        return []
