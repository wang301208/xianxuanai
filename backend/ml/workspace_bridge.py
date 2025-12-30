"""Bridge between local module workspaces and the shared global workspace."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from backend.monitoring.global_workspace import (
    GlobalWorkspace,
    global_workspace as DEFAULT_GLOBAL_WORKSPACE,
)
from modules.brain.consciousness.hierarchical_model import (
    HierarchicalConsciousnessModel,
)
from BrainSimulationSystem.models.attention_manager import AttentionManager
from .fusion import CrossModuleFusion, FusionResult


def _ensure_list(value: Optional[Sequence[float] | float]) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        return [float(value)]  # pragma: no cover - defensive fallback
    if isinstance(value, Sequence):
        return [float(v) for v in value]
    return [float(value)]


@dataclass
class WorkspaceEvent:
    """Canonical representation of a broadcastable workspace message."""

    module: str
    topic: str
    payload: Mapping[str, Any]
    score: float = 0.0
    attention: Optional[Sequence[float] | float] = None
    strategy: str = "full"
    targets: Optional[Sequence[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkspaceBridge:
    """Coordinate hierarchical consciousness with the global workspace."""

    def __init__(
        self,
        *,
        workspace: Optional[GlobalWorkspace] = None,
        consciousness: Optional[HierarchicalConsciousnessModel] = None,
        attention_manager: Optional[AttentionManager] = None,
        fusion: Optional[CrossModuleFusion] = None,
        base_threshold: float = 0.25,
        auto_register: Optional[Iterable[str]] = None,
    ) -> None:
        self.global_workspace = workspace or DEFAULT_GLOBAL_WORKSPACE
        self.consciousness = consciousness or HierarchicalConsciousnessModel(
            base_threshold=base_threshold,
        )
        self.attention_manager = attention_manager
        self.fusion = fusion
        self._registered_modules: set[str] = set()
        self._last_focus_digest: Optional[str] = None
        if auto_register:
            for name in auto_register:
                self.register_module(name)

    # ------------------------------------------------------------------
    def register_module(self, name: str, module: Optional[Any] = None) -> None:
        """Register *module* (if provided) with both local and global workspaces."""

        if name not in self._registered_modules:
            # Ensure thresholds exist for the module.
            self.consciousness.thresholds[name]  # lazily initialises via defaultdict
            self._registered_modules.add(name)
        if module is not None:
            self.global_workspace.register_module(name, module)

    # ------------------------------------------------------------------
    def publish(self, event: WorkspaceEvent) -> bool:
        """Publish *event* using the hierarchical attention gate."""

        self.register_module(event.module)
        info = {
            "topic": event.topic,
            "payload": dict(event.payload),
            "score": float(event.score),
            "metadata": dict(event.metadata),
            "timestamp": time.time(),
        }
        attention = _ensure_list(event.attention)

        self._ingest_attention(
            source=event.module,
            topic=event.topic,
            payload=info,
            attention=attention,
        )

        salient = self.consciousness.focus_attention(event.module, info)
        if not salient:
            return False

        if attention is None:
            attention = [float(event.score)]
        targets = list(event.targets) if event.targets is not None else None
        self.global_workspace.broadcast(
            sender=event.module,
            state=info,
            attention=attention,
            strategy=event.strategy,
            targets=targets,
        )
        return True

    # ------------------------------------------------------------------
    def handle_cycle(
        self,
        *,
        task_id: str,
        ability: str,
        summary: Mapping[str, Any],
    ) -> None:
        """Publish a structured set of events from a learning cycle."""

        plan = summary.get("plan", {})
        analysis = summary.get("analysis", {})
        execution = summary.get("execution", {})
        reflection = summary.get("reflection", {})
        triggered = summary.get("training_triggered", {})

        fusion_alignment: Optional[FusionResult] = None
        fused_visual: Optional[Dict[str, Any]] = None
        if self.fusion is not None:
            fusion_alignment = self.fusion.align_plan_execution(plan, execution)
            fused_visual = self.fusion.fuse_visual_language_from_summary(summary)

        plan_score = float(analysis.get("reliability_score", 0.0))
        plan_attention = fusion_alignment.attention if fusion_alignment else None
        plan_metadata: Dict[str, Any] = {"ability": ability}
        if fusion_alignment:
            plan_metadata["step_attention"] = {
                "steps": fusion_alignment.steps,
                "attention": fusion_alignment.attention,
                "raw_scores": fusion_alignment.raw_scores,
                "annotations": fusion_alignment.annotations,
            }
        if isinstance(plan, dict) and plan.get("constraint_summary"):
            plan_metadata["constraint_summary"] = plan.get("constraint_summary")
        self.publish(
            WorkspaceEvent(
                module=f"planner:{ability}",
                topic="plan_completed",
                score=plan_score,
                attention=plan_attention,
                payload={
                    "task_id": task_id,
                    "plan": plan,
                    "analysis": analysis,
                },
                metadata=plan_metadata,
            )
        )

        reward_value = float(execution.get("reward", 0.0))
        success = bool(execution.get("success", False))
        exec_score = abs(reward_value) + (0.3 if success else 0.0)
        exec_metadata: Dict[str, Any] = {"logs": execution.get("logs", [])}
        if fusion_alignment:
            exec_metadata["plan_alignment"] = fusion_alignment.attention
        self.publish(
            WorkspaceEvent(
                module=f"executor:{ability}",
                topic="execution_feedback",
                score=exec_score,
                attention=[exec_score, plan_score],
                payload={
                    "task_id": task_id,
                    "result": execution.get("result"),
                    "reward": reward_value,
                    "success": success,
                    "metrics": execution.get("metrics", {}),
                },
                metadata=exec_metadata,
            )
        )

        if reflection:
            reflection_score = float(reflection.get("confidence", 0.5))
            self.publish(
                WorkspaceEvent(
                    module="reflection",
                    topic="self_reflection",
                    score=reflection_score,
                    payload={
                        "task_id": task_id,
                        "insights": reflection,
                    },
                )
            )

        triggered_models = [name for name, active in triggered.items() if active]
        if triggered_models:
            self.publish(
                WorkspaceEvent(
                    module="learning",
                    topic="retraining_triggered",
                    score=0.6 + 0.1 * len(triggered_models),
                    payload={
                        "task_id": task_id,
                        "models": triggered_models,
                    },
                )
            )

        if fused_visual:
            self.publish(
                WorkspaceEvent(
                    module="crossmodal",
                    topic="vision_language_fusion",
                    score=0.4 + 0.1 * fused_visual.get("norm", 0.0),
                    attention=fused_visual.get("attention"),
                    payload={
                        "task_id": task_id,
                        "agent_id": fused_visual.get("agent_id"),
                        "embedding": fused_visual.get("embedding"),
                        "norm": fused_visual.get("norm"),
                    },
                )
            )

        self.update_focus()

    # ------------------------------------------------------------------
    def update_focus(
        self,
        *,
        context: Optional[Dict[str, Any]] = None,
        motivation: Optional[Dict[str, float]] = None,
        broadcast: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Refresh spotlight selection using the attention manager."""

        if self.attention_manager is None:
            return None
        snapshot = self.attention_manager.select_focus(
            context=context,
            motivation=motivation,
        )
        if not broadcast:
            return snapshot

        digest = self._focus_digest(snapshot.get("focus", []))
        if digest == self._last_focus_digest:
            return snapshot
        self._last_focus_digest = digest

        focus_payload = {
            "focus": snapshot.get("focus", []),
            "suppressed": snapshot.get("suppressed", []),
            "scores": snapshot.get("scores", []),
            "strategy": snapshot.get("strategy"),
            "timestamp": snapshot.get("timestamp"),
        }
        combined_attention = [
            float(item.get("score", 0.0)) for item in snapshot.get("focus", [])
        ] or [0.0]
        self.global_workspace.broadcast(
            sender="attention-manager",
            state=focus_payload,
            attention=combined_attention,
            strategy="full",
        )
        return snapshot

    # ------------------------------------------------------------------
    def _ingest_attention(
        self,
        *,
        source: str,
        topic: str,
        payload: Mapping[str, Any],
        attention: Optional[List[float]],
    ) -> None:
        if self.attention_manager is None:
            return
        if source == "attention-manager":
            return

        salience = max(0.0, min(1.0, float(payload.get("score", 0.5))))
        confidence = float(sum(attention) / len(attention)) if attention else salience
        novelty = float(payload.get("metadata", {}).get("novelty", 0.5))

        entry_payload = {
            "topic": topic,
            "task_id": payload.get("payload", {}).get("task_id"),
            "metadata": payload.get("metadata", {}),
        }
        self.attention_manager.add(
            source=source,
            payload=entry_payload,
            salience=salience,
            confidence=max(0.0, min(1.0, confidence)),
            novelty=max(0.0, min(1.0, novelty)),
        )

    @staticmethod
    def _focus_digest(focus: Sequence[Mapping[str, Any]]) -> str:
        items = []
        for entry in focus:
            source = str(entry.get("source"))
            score = float(entry.get("score", 0.0))
            topic = str(entry.get("payload", {}).get("topic", ""))
            items.append(f"{source}:{topic}:{score:.3f}")
        return "|".join(items)
            )
