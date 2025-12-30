"""Continuous learning loop orchestration tying planning, execution, and training."""

from __future__ import annotations

import json
import logging
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence

from BrainSimulationSystem.models.meta_reasoner import MetaReasoner

from .experience_collector import log_interaction
from .retraining_pipeline import DATASET, REFLECTION_CATEGORY, TRAINING_INTERACTION_CATEGORY

if TYPE_CHECKING:
    from ..memory.long_term import LongTermMemory
    from .workspace_bridge import WorkspaceBridge
logger = logging.getLogger(__name__)


class PlannerFn(Protocol):
    def __call__(self, task: Mapping[str, Any], *, strategy: Optional[str] = None) -> Dict[str, Any]:
        ...


class ExecutorFn(Protocol):
    def __call__(self, plan: Mapping[str, Any], *, strategy: str, context: Mapping[str, Any]) -> "ExecutionFeedback":
        ...


class ReflectionFn(Protocol):
    def __call__(self, plan: Mapping[str, Any], execution: "ExecutionFeedback") -> Dict[str, Any]:
        ...


@dataclass
class ExecutionFeedback:
    """Container describing execution phase output."""

    result: Mapping[str, Any]
    reward: Optional[float] = None
    success: bool = False
    logs: Sequence[str] = field(default_factory=list)
    confidence: Optional[float] = None
    metrics: Optional[Mapping[str, float]] = None


@dataclass
class LearningCycleConfig:
    """Configuration for automated training triggers."""

    min_samples_supervised: int = 120
    min_samples_meta: int = 160
    min_samples_self_sup: int = 200
    min_samples_rl: int = 80
    enable_supervised: bool = True
    enable_rl: bool = True
    enable_self_supervised: bool = True
    enable_meta: bool = True
    cool_down_seconds: int = 1800
    exploration_probability: float = 0.0
    exploration_attempts: int = 1
    exploration_requires_success: bool = False


@dataclass
class ReplayConfig:
    """Configuration for replaying prior interactions during idle periods."""

    enable_replay: bool = True
    replay_interval_seconds: int = 900
    replay_batch_size: int = 3
    sampling_strategy: str = "uniform"  # "uniform", "low_confidence", "stale"
    categories: Sequence[str] = (TRAINING_INTERACTION_CATEGORY, REFLECTION_CATEGORY)
    mixed_curriculum: bool = True
    candidate_search_limit: int = 50
    low_confidence_threshold: float = 0.4


@dataclass
class ReplaySample:
    entry_id: int
    category: str
    payload: Any
    confidence: float
    timestamp: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


class TrainingScheduler:
    """Trigger retraining runs when sufficient new data has been collected."""

    def __init__(self, config: LearningCycleConfig) -> None:
        self.config = config
        self._last_row_count: int = 0
        self._last_run: Dict[str, float] = {}
        self._rows_at_last_run: Dict[str, int] = {}

    def _row_count(self) -> int:
        if not DATASET.exists():
            return 0
        count = 0
        with DATASET.open("r", encoding="utf-8") as f:
            for count, _ in enumerate(f, 1):
                pass
        return max(count - 1, 0)

    def maybe_trigger(self) -> Dict[str, bool]:
        triggered: Dict[str, bool] = {}
        rows = self._row_count()
        delta = rows - self._last_row_count
        self._last_row_count = rows
        if rows == 0 or delta <= 0:
            return triggered

        now = time.time()
        thresholds = {
            "llm": self.config.min_samples_supervised,
            "self_supervised": self.config.min_samples_self_sup,
            "meta": self.config.min_samples_meta,
            "rl": self.config.min_samples_rl,
        }
        enabled = {
            "llm": self.config.enable_supervised,
            "self_supervised": self.config.enable_self_supervised,
            "meta": self.config.enable_meta,
            "rl": self.config.enable_rl,
        }

        for model, threshold in thresholds.items():
            if not enabled.get(model, False):
                continue
            last_run = self._last_run.get(model, 0.0)
            if now - last_run < self.config.cool_down_seconds:
                continue
            previous_rows = self._rows_at_last_run.get(model, 0)
            gained_since_last = rows - previous_rows
            if gained_since_last < threshold and rows < threshold:
                continue
            if self._run_pipeline(model):
                triggered[model] = True
                self._last_run[model] = now
                self._rows_at_last_run[model] = rows
        return triggered

    def _run_pipeline(self, model: str) -> bool:
        script = Path("backend") / "ml" / "retraining_pipeline.py"
        if not script.exists():
            logger.warning("Retraining pipeline not found at %s", script)
            return False
        logger.info("Triggering retraining pipeline for model=%s", model)
        try:
            subprocess.run(
                [sys.executable, str(script), "--model", model],
                check=True,
            )
            return True
        except subprocess.CalledProcessError as exc:
            logger.exception("Retraining pipeline failed for %s: %s", model, exc)
            return False


class LearningLoopOrchestrator:
    """Coordinate planning, execution, feedback logging, and retraining."""

    def __init__(
        self,
        planner: PlannerFn,
        executor: ExecutorFn,
        *,
        meta_reasoner: Optional[MetaReasoner] = None,
        reflector: Optional[ReflectionFn] = None,
        config: Optional[LearningCycleConfig] = None,
        replay_config: Optional[ReplayConfig] = None,
        workspace_bridge: Optional["WorkspaceBridge"] = None,
        memory: Optional["LongTermMemory"] = None,
        exploration_rng: Optional[random.Random] = None,
    ) -> None:
        self.planner = planner
        self.executor = executor
        self.meta_reasoner = meta_reasoner or MetaReasoner()
        self.reflector = reflector
        self.config = config or LearningCycleConfig()
        self.replay_config = replay_config or ReplayConfig()
        self.scheduler = TrainingScheduler(self.config)
        self.workspace_bridge = workspace_bridge
        self.memory = memory
        self._rng = exploration_rng or random.Random()
        self._last_replay: float = 0.0

    def run_cycle(self, task: Mapping[str, Any]) -> Dict[str, Any]:
        """Execute a complete plan-act-learn loop for ``task``."""

        task_id = str(task.get("id", task.get("name", "task")))
        ability = str(task.get("ability", "default"))
        context = task.get("context", {})

        plan = self.planner(task, strategy=None)
        strategies = plan.get("strategies") or plan.get("candidates") or ["default"]
        chosen_strategy = self.meta_reasoner.recommend_strategy(task_id, strategies)

        decision_confidence = float(plan.get("confidence", 0.5) or 0.5)
        decision = {"decision": chosen_strategy, "confidence": decision_confidence}
        analysis = self.meta_reasoner.evaluate(plan, decision, context)

        feedback, reward_value, success_flag, reflection = self._execute_strategy(
            plan, chosen_strategy, context, task_id
        )

        primary_record = self._record_learning_signal(
            task_id=task_id,
            ability=ability,
            plan=plan,
            strategy=chosen_strategy,
            feedback=feedback,
            reward=reward_value,
            success=success_flag,
            analysis=analysis,
            reflection=reflection,
            confidence_hint=decision_confidence,
        )

        exploration_runs = self._run_exploration(
            task_id=task_id,
            ability=ability,
            plan=plan,
            context=context,
            strategies=strategies,
            chosen=chosen_strategy,
            primary_success=success_flag,
        )

        triggered = self.scheduler.maybe_trigger()

        summary = {
            "plan": plan,
            "analysis": analysis,
            "execution": {
                "result": primary_record["result"],
                "success": primary_record["success"],
                "reward": primary_record["reward"],
                "logs": primary_record["logs"],
                "metrics": primary_record["metrics"],
            },
            "reflection": reflection,
            "strategy": chosen_strategy,
            "training_triggered": triggered,
            "strategy_summary": self.meta_reasoner.summarise_strategies(task_id),
            "exploration_runs": exploration_runs,
            "exploration_triggered": bool(exploration_runs),
        }

        if self.workspace_bridge is not None:
            try:
                self.workspace_bridge.handle_cycle(
                    task_id=task_id,
                    ability=ability,
                    summary=summary,
                )
            except Exception:  # pragma: no cover - defensive
                logger.exception("Workspace bridge failed for task %s", task_id)

        return summary

    # ------------------------------------------------------------------
    def _execute_strategy(
        self,
        plan: Mapping[str, Any],
        strategy: str,
        context: Mapping[str, Any],
        task_id: str,
    ) -> tuple[ExecutionFeedback, float, bool, Dict[str, Any]]:
        feedback = self.executor(plan, strategy=strategy, context=context)
        reward_value = float(feedback.reward) if feedback.reward is not None else 0.0
        success_flag = bool(feedback.success)
        reflection: Dict[str, Any] = {}
        if self.reflector is not None:
            try:
                reflection = self.reflector(plan, feedback)
            except Exception:  # pragma: no cover - defensive
                logger.exception("Reflection callable failed for task %s", task_id)
                reflection = {}
        return feedback, reward_value, success_flag, reflection

    def _record_learning_signal(
        self,
        *,
        task_id: str,
        ability: str,
        plan: Mapping[str, Any],
        strategy: str,
        feedback: ExecutionFeedback,
        reward: float,
        success: bool,
        analysis: Any,
        reflection: Mapping[str, Any],
        confidence_hint: float,
    ) -> Dict[str, Any]:
        ability_key = f"{ability}:{strategy}"
        log_payload = {
            "input": plan,
            "output": feedback.result,
            "analysis": analysis,
            "reflection": reflection,
            "strategy": strategy,
        }
        log_interaction(task=task_id, ability=ability_key, result=log_payload, reward=reward)

        confidence = (
            float(feedback.confidence)
            if feedback.confidence is not None
            else float(confidence_hint)
        )
        self.meta_reasoner.record_strategy_outcome(
            task_signature=task_id,
            strategy=strategy,
            reward=reward,
            success=success,
            confidence=confidence,
            metadata={"ability": ability},
        )

        self._persist_training_sample(
            task_id=task_id,
            ability=ability,
            strategy=strategy,
            plan=plan,
            analysis=analysis,
            reflection=reflection,
            feedback=feedback,
            reward=reward,
            success=success,
        )
        self._persist_reflection_history()

        return {
            "result": feedback.result,
            "success": success,
            "reward": reward,
            "logs": list(feedback.logs),
            "metrics": dict(feedback.metrics or {}),
            "strategy": strategy,
            "ability": ability,
            "reflection": reflection,
            "analysis": analysis,
        }

    def _persist_training_sample(
        self,
        *,
        task_id: str,
        ability: str,
        strategy: str,
        plan: Mapping[str, Any],
        analysis: Any,
        reflection: Mapping[str, Any],
        feedback: ExecutionFeedback,
        reward: float,
        success: bool,
    ) -> None:
        if not self.memory:
            return
        if isinstance(feedback.result, Mapping):
            result_payload: Mapping[str, Any] = dict(feedback.result)
        else:
            result_payload = {"value": feedback.result}
        record = {
            "task": task_id,
            "ability": ability,
            "strategy": strategy,
            "plan": plan,
            "analysis": analysis,
            "reflection": reflection,
            "result": result_payload,
            "logs": list(feedback.logs),
            "metrics": dict(feedback.metrics or {}),
            "reward": reward,
            "success": success,
        }
        try:
            payload = json.dumps(record, ensure_ascii=True, sort_keys=True, default=str)
            self.memory.add(TRAINING_INTERACTION_CATEGORY, payload)
        except Exception:  # pragma: no cover - defensive persistence
            logger.debug("failed to persist training sample", exc_info=True)

    def _persist_reflection_history(self) -> None:
        if not self.memory or not self.reflector:
            return
        source = getattr(self.reflector, "__self__", self.reflector)
        history = getattr(source, "history", None)
        if not history:
            return
        try:
            from ..reflection.reflection import save_history

            save_history(self.memory, history, category=REFLECTION_CATEGORY)
        except Exception:  # pragma: no cover - defensive persistence
            logger.debug("skipped saving reflection history", exc_info=True)

    def _should_explore(
        self,
        success: bool,
        strategies: Sequence[str],
        chosen: str,
    ) -> bool:
        if self.config.exploration_probability <= 0.0:
            return False
        if self.config.exploration_requires_success and not success:
            return False
        available = [s for s in strategies if s != chosen]
        if not available:
            return False
        return self._rng.random() < self.config.exploration_probability

    def _run_exploration(
        self,
        *,
        task_id: str,
        ability: str,
        plan: Mapping[str, Any],
        context: Mapping[str, Any],
        strategies: Sequence[str],
        chosen: str,
        primary_success: bool,
    ) -> List[Dict[str, Any]]:
        runs: List[Dict[str, Any]] = []
        if not self._should_explore(
            success=primary_success,
            strategies=strategies,
            chosen=chosen,
        ):
            return runs
        candidates = [s for s in strategies if s != chosen]
        self._rng.shuffle(candidates)
        attempts = min(len(candidates), max(0, self.config.exploration_attempts))
        for strategy in candidates[:attempts]:
            analysis = self.meta_reasoner.evaluate(plan, {"decision": strategy}, context)
            feedback, reward_value, success_flag, reflection = self._execute_strategy(
                plan, strategy, context, task_id
            )
            record = self._record_learning_signal(
                task_id=task_id,
                ability=ability,
                plan=plan,
                strategy=strategy,
                feedback=feedback,
                reward=reward_value,
                success=success_flag,
                analysis=analysis,
                reflection=reflection,
                confidence_hint=self._confidence_hint(analysis),
            )
            runs.append(record)
        return runs

    @staticmethod
    def _confidence_hint(analysis: Any, fallback: float = 0.5) -> float:
        if isinstance(analysis, Mapping):
            value = analysis.get("confidence")
            if isinstance(value, (int, float)):
                return float(value)
        return fallback

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------
    def run_replay_if_idle(self, *, now: Optional[float] = None) -> List[Dict[str, Any]]:
        """Sample past interactions from memory and refresh them through the loop."""

        if not self.memory or not self.replay_config.enable_replay:
            return []

        current_time = float(now if now is not None else time.time())
        if current_time - self._last_replay < self.replay_config.replay_interval_seconds:
            return []

        samples = self._sample_replay_entries()
        results: List[Dict[str, Any]] = []
        for sample in samples:
            try:
                if sample.category == TRAINING_INTERACTION_CATEGORY:
                    record = self._replay_training_interaction(sample)
                    if record:
                        results.append(record)
                elif sample.category == REFLECTION_CATEGORY:
                    results.extend(self._replay_reflection_history(sample))
            except Exception:  # pragma: no cover - defensive
                logger.exception("Replay failed for entry %s", sample.entry_id)

        self._last_replay = current_time
        return results

    def _sample_replay_entries(self) -> List[ReplaySample]:
        if not self.memory:
            return []

        candidates_by_category: Dict[str, List[ReplaySample]] = {}
        strategy = (self.replay_config.sampling_strategy or "uniform").lower()
        for category in self.replay_config.categories:
            entries: List[ReplaySample] = []
            for entry in self.memory.get(
                category=category,
                include_metadata=True,
                limit=self.replay_config.candidate_search_limit,
                newest_first=strategy != "stale",
            ):
                payload = self._decode_payload(entry.get("content"))
                entries.append(
                    ReplaySample(
                        entry_id=int(entry.get("id", 0)),
                        category=category,
                        payload=payload,
                        confidence=float(entry.get("confidence", 1.0) or 1.0),
                        timestamp=float(entry.get("timestamp", 0.0) or 0.0),
                        metadata=entry.get("metadata", {}) or {},
                    )
                )

            if strategy == "low_confidence":
                entries.sort(key=lambda item: item.confidence)
            elif strategy == "stale":
                entries.sort(key=lambda item: item.timestamp)
            else:
                self._rng.shuffle(entries)
            candidates_by_category[category] = entries

        batch_size = max(0, int(self.replay_config.replay_batch_size))
        selected: List[ReplaySample] = []
        if self.replay_config.mixed_curriculum:
            while len(selected) < batch_size:
                progressed = False
                for category in self.replay_config.categories:
                    pool = candidates_by_category.get(category, [])
                    if not pool:
                        continue
                    selected.append(pool.pop(0))
                    progressed = True
                    if len(selected) >= batch_size:
                        break
                if not progressed:
                    break
        else:
            merged: List[ReplaySample] = []
            for category in self.replay_config.categories:
                merged.extend(candidates_by_category.get(category, []))
            selected = merged[:batch_size]

        return selected

    @staticmethod
    def _decode_payload(raw: Any) -> Any:
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw
        return raw

    def _replay_training_interaction(self, sample: ReplaySample) -> Optional[Dict[str, Any]]:
        payload = sample.payload if isinstance(sample.payload, Mapping) else {}
        plan = payload.get("plan") if isinstance(payload.get("plan"), Mapping) else {}
        analysis = payload.get("analysis") if isinstance(payload.get("analysis"), Mapping) else {}
        strategy = str(payload.get("strategy", "default"))
        task_id = str(payload.get("task", f"replay-{sample.entry_id}"))
        ability = str(payload.get("ability", "replay"))
        context = payload.get("context") if isinstance(payload.get("context"), Mapping) else {}

        feedback, reward_value, success_flag, reflection = self._execute_strategy(
            plan,
            strategy,
            context,
            task_id,
        )
        confidence_hint = self._confidence_hint(analysis, fallback=sample.confidence)
        record = self._record_learning_signal(
            task_id=task_id,
            ability=ability,
            plan=plan,
            strategy=strategy,
            feedback=feedback,
            reward=reward_value,
            success=success_flag,
            analysis=analysis,
            reflection=reflection,
            confidence_hint=confidence_hint,
        )

        try:
            new_metadata = dict(sample.metadata)
            new_metadata["last_replayed_at"] = time.time()
            self.memory.update_entry(
                sample.entry_id,
                confidence=max(sample.confidence, feedback.confidence or confidence_hint),
                metadata=new_metadata,
            )
        except Exception:  # pragma: no cover - defensive persistence
            logger.debug("Failed to update replayed memory entry", exc_info=True)

        return record

    def _replay_reflection_history(self, sample: ReplaySample) -> List[Dict[str, Any]]:
        payload = sample.payload if isinstance(sample.payload, list) else []
        results: List[Dict[str, Any]] = []
        for idx, entry in enumerate(payload):
            if not isinstance(entry, Mapping):
                continue
            evaluation = entry.get("evaluation", {})
            confidence = (
                float(evaluation.get("confidence", sample.confidence) or sample.confidence)
                if isinstance(evaluation, Mapping)
                else sample.confidence
            )
            reflection_payload = entry.get("revision", entry)
            feedback = ExecutionFeedback(
                result={"reflection": reflection_payload},
                reward=confidence,
                success=True,
                confidence=confidence,
            )
            analysis = {"source": "reflection_replay", "confidence": confidence}
            record = self._record_learning_signal(
                task_id=f"reflection-replay-{sample.entry_id}-{idx}",
                ability="reflection",
                plan={"reflection": entry},
                strategy="replay",
                feedback=feedback,
                reward=confidence,
                success=True,
                analysis=analysis,
                reflection=entry,
                confidence_hint=confidence,
            )
            results.append(record)

        if results:
            try:
                new_metadata = dict(sample.metadata)
                new_metadata["last_replayed_at"] = time.time()
                self.memory.update_entry(
                    sample.entry_id,
                    confidence=max(sample.confidence, max(r["reward"] for r in results)),
                    metadata=new_metadata,
                )
            except Exception:  # pragma: no cover - defensive persistence
                logger.debug("Failed to update reflection replay entry", exc_info=True)
        return results
