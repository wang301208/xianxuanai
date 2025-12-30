"""Coordinate meta-learning updates driven by benchmark feedback."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover - torch is optional
    torch = None  # type: ignore[assignment]

from BrainSimulationSystem.learning.meta_learning import MetaTask
from .core import FewShotTask, MAMLEngine

logger = logging.getLogger(__name__)


class BenchmarkMetaLearningCoordinator:
    """Bridge benchmark completions to meta-learning updates.

    The coordinator listens for benchmark results containing support/query
    splits, converts them into :class:`MetaTask` objects for gradient-based
    meta-learning or :class:`FewShotTask` objects for the lightweight MAML
    engine, and persists the adapted weights to a shared store that can be
    consumed by subsequent tasks.
    """

    def __init__(
        self,
        *,
        meta_learner: Any | None = None,
        maml_engine: MAMLEngine | None = None,
        checkpoint_dir: str | Path = "benchmarks/results/meta_checkpoints",
        agent_loader: Optional[Callable[[Path], None]] = None,
    ) -> None:
        self.meta_learner = meta_learner
        self.maml_engine = maml_engine
        self.shared_store = Path(checkpoint_dir)
        self.shared_store.mkdir(parents=True, exist_ok=True)
        self.agent_loader = agent_loader
        self.last_checkpoint: Path | None = None

    # ------------------------------------------------------------------
    def handle_benchmark_result(self, result: Mapping[str, Any]) -> Optional[Path]:
        """Trigger meta-updates when benchmark data contains meta splits."""

        raw_data = result.get("raw_data") if isinstance(result, Mapping) else None
        if not raw_data:
            return None

        payloads = self._extract_tasks(raw_data)
        if not payloads:
            return None

        checkpoint: Path | None = None
        if self.meta_learner is not None:
            tasks = self._build_meta_tasks(payloads)
            if tasks:
                logger.info("Running gradient-based meta-train on %d tasks", len(tasks))
                self.meta_learner.meta_train(tasks, iterations=1)
                checkpoint = self._persist_meta_learner(payloads)
        elif self.maml_engine is not None:
            few_shot = self._build_few_shot_tasks(payloads)
            if few_shot:
                logger.info("Running lightweight MAML meta-update on %d tasks", len(few_shot))
                self.maml_engine.learn_to_learn(few_shot, epochs=1)
                checkpoint = self._persist_maml_engine(payloads)

        if checkpoint is not None:
            self.last_checkpoint = checkpoint
            self.apply_latest_checkpoint()
        return checkpoint

    # ------------------------------------------------------------------
    def apply_latest_checkpoint(self) -> None:
        """Load the most recent checkpoint into the agent if requested."""

        if self.agent_loader is None:
            return
        if self.last_checkpoint is None or not self.last_checkpoint.exists():
            return
        try:
            self.agent_loader(self.last_checkpoint)
            logger.info("Applied meta-learning checkpoint %s", self.last_checkpoint)
        except Exception:  # pragma: no cover - defensive logging
            logger.debug("Failed to apply meta-learning checkpoint", exc_info=True)

    # ------------------------------------------------------------------
    def _extract_tasks(self, raw_data: MutableMapping[str, Any]) -> List[Mapping[str, Any]]:
        payload = raw_data.get("meta_task")
        if payload is None:
            return []
        if isinstance(payload, Iterable) and not isinstance(payload, Mapping):
            return [p for p in payload if isinstance(p, Mapping)]
        if isinstance(payload, Mapping):
            return [payload]
        return []

    # ------------------------------------------------------------------
    def _build_meta_tasks(self, payloads: Iterable[Mapping[str, Any]]) -> List[MetaTask]:
        if torch is None:
            logger.debug("Torch not available; skipping MetaTask construction")
            return []

        tasks: List[MetaTask] = []
        for payload in payloads:
            try:
                support_x = torch.tensor(payload["support_x"], dtype=torch.float32)
                support_y = torch.tensor(payload["support_y"], dtype=torch.float32)
                query_x = torch.tensor(payload["query_x"], dtype=torch.float32)
                query_y = torch.tensor(payload["query_y"], dtype=torch.float32)
            except Exception:  # pragma: no cover - malformed payloads are skipped
                logger.debug("Invalid meta-learning payload", exc_info=True)
                continue

            def _sampler(x: "torch.Tensor", y: "torch.Tensor"):
                def _sample() -> tuple["torch.Tensor", "torch.Tensor"]:
                    return x.clone(), y.clone()

                return _sample

            task = MetaTask(
                task_id=str(payload.get("task_id", "benchmark-task")),
                support_sampler=_sampler(support_x, support_y),
                query_sampler=_sampler(query_x, query_y),
                metadata=dict(payload.get("metadata", {})),
            )
            tasks.append(task)
        return tasks

    # ------------------------------------------------------------------
    def _build_few_shot_tasks(self, payloads: Iterable[Mapping[str, Any]]) -> List[FewShotTask]:
        tasks: List[FewShotTask] = []
        for payload in payloads:
            try:
                support_x = np.array(payload["support_x"], dtype=float)
                support_y = np.array(payload["support_y"], dtype=float)
                query_x = np.array(payload["query_x"], dtype=float)
                query_y = np.array(payload["query_y"], dtype=float)
            except Exception:  # pragma: no cover - malformed payloads are skipped
                logger.debug("Invalid few-shot payload", exc_info=True)
                continue
            tasks.append(FewShotTask(support_x, support_y, query_x, query_y))
        return tasks

    # ------------------------------------------------------------------
    def _persist_meta_learner(self, payloads: Iterable[Mapping[str, Any]]) -> Path:
        assert self.meta_learner is not None
        checkpoint = self.shared_store / f"meta_learner_{int(time.time())}.pt"
        try:
            torch.save(  # type: ignore[arg-type]
                {
                    "state_dict": self.meta_learner.model.state_dict(),
                    "config": getattr(self.meta_learner, "config", None),
                    "payload": json.loads(json.dumps(list(payloads))),
                },
                checkpoint,
            )
        except Exception:  # pragma: no cover - defensive logging
            logger.debug("Failed to persist meta-learner checkpoint", exc_info=True)
        return checkpoint

    # ------------------------------------------------------------------
    def _persist_maml_engine(self, payloads: Iterable[Mapping[str, Any]]) -> Path:
        assert self.maml_engine is not None
        checkpoint = self.shared_store / f"maml_engine_{int(time.time())}.npy"
        try:
            np.save(checkpoint, self.maml_engine.weights)
            meta_path = checkpoint.with_suffix(".json")
            meta_path.write_text(json.dumps(list(payloads)), encoding="utf-8")
        except Exception:  # pragma: no cover - defensive logging
            logger.debug("Failed to persist MAML engine checkpoint", exc_info=True)
        return checkpoint


__all__ = ["BenchmarkMetaLearningCoordinator"]
