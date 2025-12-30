"""Cross-module fusion utilities for multi-modal coordination."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

try:  # pragma: no cover - torch is optional at runtime
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from backend.world_model.vision import CrossModalAttention, VisionStore


def _to_array(value: Any) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def _infer_dim(*values: Any) -> int:
    for value in values:
        if value is None:
            continue
        arr = _to_array(value)
        if arr.ndim == 0:
            return 1
        return int(arr.shape[-1])
    return 1


def _normalise(weights: Sequence[float]) -> List[float]:
    positives = [max(float(w), 0.0) for w in weights]
    total = sum(positives)
    if total <= 0.0:
        n = len(positives)
        return [1.0 / n for _ in positives] if n else []
    return [w / total for w in positives]


@dataclass
class FusionResult:
    steps: List[str]
    attention: List[float]
    raw_scores: List[float]
    annotations: List[Dict[str, Any]]


class CrossModuleFusion:
    """Provide cross-modal and cross-module attention utilities."""

    def __init__(self, vision_store: Optional[VisionStore] = None) -> None:
        self.vision_store = vision_store or VisionStore()

    # ------------------------------------------------------------------
    # Vision-language fusion
    # ------------------------------------------------------------------
    def fuse_vision_language(
        self,
        agent_id: str,
        *,
        vision_features: Any,
        text_features: Any,
    ) -> Optional[np.ndarray]:
        if vision_features is None or text_features is None:
            return None
        unified = self.vision_store.ingest(
            agent_id,
            features=vision_features,
            text=text_features,
        )
        if unified is not None:
            return _to_array(unified)
        embed_dim = _infer_dim(vision_features, text_features)
        attention = CrossModalAttention(embed_dim)
        fused = attention(vision_features, text_features)
        return _to_array(fused)

    def fuse_visual_language_from_summary(
        self,
        summary: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        plan = summary.get("plan", {})
        execution = summary.get("execution", {})
        result = execution.get("result", {}) if isinstance(execution, Mapping) else {}

        agent_id = (
            result.get("agent_id")
            or plan.get("agent_id")
            or summary.get("analysis", {}).get("agent_id")
        )
        vision_features = (
            result.get("vision_features")
            or result.get("features")
            or result.get("image_features")
        )
        text_features = (
            plan.get("text_embedding")
            or result.get("text_embedding")
            or summary.get("analysis", {}).get("language_embedding")
        )
        if agent_id is None or vision_features is None or text_features is None:
            return None

        fused = self.fuse_vision_language(
            str(agent_id),
            vision_features=vision_features,
            text_features=text_features,
        )
        if fused is None:
            return None

        norm = float(np.linalg.norm(fused))
        return {
            "agent_id": str(agent_id),
            "embedding": fused.tolist(),
            "norm": norm,
            "attention": [norm] if norm > 0 else [0.5],
        }

    # ------------------------------------------------------------------
    # Plan / execution cross-attention
    # ------------------------------------------------------------------
    def align_plan_execution(
        self,
        plan: Any,
        execution: Mapping[str, Any],
    ) -> Optional[FusionResult]:
        steps = self._extract_steps(plan)
        if not steps:
            return None

        metrics = execution.get("metrics", {}) if isinstance(execution, Mapping) else {}

        scores: List[float] = []
        annotations: List[Dict[str, Any]] = []
        for idx, step in enumerate(steps):
            raw_components: Dict[str, float] = {}
            value = self._collect_metric(metrics, idx, step, raw_components)
            if value is None:
                value = 1.0
                raw_components["default"] = value
            scores.append(float(value))
            annotations.append({"step": step, "components": raw_components})

        attention = _normalise(scores)
        return FusionResult(
            steps=steps,
            attention=attention,
            raw_scores=scores,
            annotations=annotations,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _extract_steps(self, plan: Any) -> List[str]:

        if isinstance(plan, Mapping):
            for key in ("steps", "plan", "actions", "sequence"):
                value = plan.get(key)
                if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                    return [str(item) for item in value]
            if isinstance(plan.get("outline"), Sequence):
                return [str(item) for item in plan.get("outline")]
            if isinstance(plan.get("candidates"), Sequence):
                collected: List[str] = []
                for candidate in plan.get("candidates", []):
                    if isinstance(candidate, Mapping):
                        action = candidate.get("action")
                    else:
                        action = candidate
                    if action is not None:
                        collected.append(str(action))
                if collected:
                    return collected
        elif isinstance(plan, Sequence) and not isinstance(plan, (str, bytes)):
            return [str(item) for item in plan]
        return []

    def _collect_metric(
        self,
        metrics: Mapping[str, Any],
        idx: int,
        step: str,
        raw_components: Dict[str, float],
    ) -> Optional[float]:
        if not isinstance(metrics, Mapping):
            return None

        def _lookup(container: Any) -> Optional[float]:
            if isinstance(container, Sequence) and not isinstance(container, (str, bytes)):
                if idx < len(container):
                    return float(container[idx])
            if isinstance(container, Mapping):
                for key in (str(idx), step, step.lower()):
                    if key in container:
                        return float(container[key])
            return None

        score = 0.0
        found = False
        for name in metrics:
            value = metrics[name]
            candidate = _lookup(value)
            if candidate is None:
                continue
            found = True
            if any(token in name for token in ("confidence", "success", "accuracy")):
                contribution = max(0.0, 1.0 - float(candidate))
            elif any(token in name for token in ("reward", "gain", "score")):
                contribution = max(0.0, -float(candidate))
            else:
                contribution = abs(float(candidate))
            raw_components[name] = contribution
            score += contribution

        return score if found else None


__all__ = ["CrossModuleFusion", "FusionResult"]
