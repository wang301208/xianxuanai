from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any, Iterable, Optional

from ...core.agent.simple import PerformanceEvaluator
from autogpt.core.agent.layered import LayeredAgent
from autogpt.core.memory import Memory
from autogpt.core.planning import SimplePlanner
from ml.experience_collector import log_interaction
from evolution.policy_trainer import PolicyTrainer


class EvolutionAgent(LayeredAgent):
    """Layer that adapts tasks and abilities based on past outcomes.

    This agent sits between planning and ability execution layers. It inspects
    historical task results stored in memory or planning subsystems and filters
    out tasks or abilities that have repeatedly failed. The remaining tasks are
    delegated to the next layer for ability selection or execution.
    """

    def __init__(
        self,
        memory: Optional[Memory] = None,
        planning: Optional[SimplePlanner] = None,
        next_layer: Optional[LayeredAgent] = None,
    ) -> None:
        super().__init__(next_layer=next_layer)
        self._memory = memory
        self._planning = planning

        data_dir = Path(__file__).resolve().parents[3] / "data"
        data_dir.mkdir(exist_ok=True)
        skills_dir = Path(__file__).resolve().parents[3] / "skills" / "MetaSkill_StrategyEvolution"
        skills_dir.mkdir(parents=True, exist_ok=True)
        self._policy_path = skills_dir / "policy.json"
        self._log_path = skills_dir / "training_log.json"
        self._policy: dict[str, dict[str, float]] = {}
        self._learning_rate = float(os.getenv("EVOLUTION_LEARNING_RATE", 0.1))
        self._generations = int(os.getenv("EVOLUTION_GENERATIONS", 10))
        self._fitness_fn = os.getenv("EVOLUTION_FITNESS_FUNCTION", "reward")
        self._performance = PerformanceEvaluator()
        self._last_state: Optional[str] = None
        self._last_action: Optional[str] = None
        self._last_prob: dict[str, float] | None = None
        self._generation_count = 0
        self._experience_batch: list[dict[str, Any]] = []
        self._policy_mtime: float | None = None

        self._load_policy()

        dataset = data_dir / "dataset.csv"
        self._trainer = PolicyTrainer(dataset_path=dataset, learning_rate=self._learning_rate, policy=self._policy)

    # ------------------------------------------------------------------
    # Policy management helpers
    # ------------------------------------------------------------------
    def _load_policy(self) -> None:
        if self._policy_path.exists():
            try:
                self._policy = json.loads(self._policy_path.read_text())
                self._policy_mtime = self._policy_path.stat().st_mtime
            except Exception:
                self._policy = {}
                self._policy_mtime = None

    def _save_policy(self) -> None:
        try:
            self._policy_path.write_text(json.dumps(self._policy, indent=2))
            self._policy_mtime = self._policy_path.stat().st_mtime
        except Exception:
            pass

    def _log_training(self, batch: list[dict[str, Any]]) -> None:
        log: list[Any] = []
        if self._log_path.exists():
            try:
                log = json.loads(self._log_path.read_text())
            except Exception:
                log = []
        log.extend(batch)
        try:
            self._log_path.write_text(json.dumps(log, indent=2))
        except Exception:
            pass

    def _maybe_reload_policy(self) -> None:
        if not self._policy_path.exists():
            return
        try:
            mtime = self._policy_path.stat().st_mtime
        except Exception:
            return
        if self._policy_mtime is None or mtime != self._policy_mtime:
            self._load_policy()

    # ------------------------------------------------------------------
    # Policy gradient ability selection
    # ------------------------------------------------------------------
    def _softmax(self, values: list[float]) -> list[float]:
        max_v = max(values) if values else 0.0
        exps = [math.exp(v - max_v) for v in values]
        total = sum(exps) or 1.0
        return [e / total for e in exps]

    def _select_ability(self, state: str, abilities: list[Any]) -> Any:
        self._maybe_reload_policy()
        ability_names = [getattr(a, "name", str(a)) for a in abilities]
        prefs = self._policy.setdefault(state, {})
        for name in ability_names:
            prefs.setdefault(name, 0.0)
        probs_list = self._softmax([prefs[n] for n in ability_names])
        chosen_name = random.choices(ability_names, weights=probs_list, k=1)[0]
        for ability in abilities:
            if getattr(ability, "name", str(ability)) == chosen_name:
                self._last_state = state
                self._last_action = chosen_name
                self._last_prob = {n: p for n, p in zip(ability_names, probs_list)}
                return ability
        return abilities[0]

    def record_feedback(self, ability_name: str, result: Any) -> None:
        if self._last_state is None or self._last_action is None:
            return
        reward = self._performance.score(result, cost=0.0, duration=0.0)

        self._trainer.push_experience(self._last_state, self._last_action, reward)
        self._experience_batch.append(
            {"state": self._last_state, "action": self._last_action, "reward": reward}
        )
        try:
            log_interaction(self._last_state, ability_name, result, reward)
        except Exception:
            pass
        self._generation_count += 1
        if self._generation_count >= self._generations:
            if self._experience_batch:
                self._log_training(self._experience_batch)
                self._experience_batch.clear()
            self._policy = self._trainer.update_policy()
            self._save_policy()
            self._generation_count = 0

    async def determine_next_ability(
        self,
        task_queue: Iterable[Any],
        ability_list: Iterable[Any],
        *args: Any,
        **kwargs: Any,
    ):
        """Evaluate history and forward filtered tasks to the next layer."""

        history: list[Any] = []
        if self._memory is not None:
            try:
                history.extend(self._memory.get() or [])
            except Exception:
                pass
        if self._planning is not None and hasattr(
            self._planning, "get_completed_tasks"
        ):
            try:
                history.extend(self._planning.get_completed_tasks())
            except Exception:
                pass

        task_queue = list(task_queue)
        ability_list = list(ability_list)

        def task_failed(task: Any) -> bool:
            identifier = getattr(task, "id", getattr(task, "name", str(task)))
            return any(
                identifier in str(item) and "FAIL" in str(item)
                for item in history
            )

        def ability_failed(name: str) -> bool:
            return any(
                name in str(item) and "ABILITY_FAIL" in str(item)
                for item in history
            )

        filtered_tasks = [t for t in task_queue if not task_failed(t)]
        filtered_abilities = [
            a for a in ability_list if not ability_failed(getattr(a, "name", str(a)))
        ]

        chosen_task = filtered_tasks[0] if filtered_tasks else None
        if chosen_task and filtered_abilities:
            state = getattr(chosen_task, "id", getattr(chosen_task, "name", str(chosen_task)))
            ability = self._select_ability(state, filtered_abilities)
            filtered_tasks = [chosen_task]
            filtered_abilities = [ability]

        if self._planning is not None and hasattr(self._planning, "update_tasks"):
            try:
                self._planning.update_tasks(filtered_tasks)
            except Exception:
                pass

        if self._memory is not None and hasattr(self._memory, "add"):
            if len(filtered_tasks) < len(task_queue):
                self._memory.add("EvolutionAgent: removed failed tasks from queue.")

        if self.next_layer is not None:
            outcome = await self.next_layer.determine_next_ability(
                filtered_tasks, filtered_abilities, *args, **kwargs
            )
            if (
                isinstance(outcome, tuple)
                and len(outcome) == 2
                and isinstance(outcome[0], dict)
            ):
                ability_name = outcome[0].get("next_ability")
                self.record_feedback(ability_name, outcome[1])
            return outcome
        return filtered_tasks, filtered_abilities
