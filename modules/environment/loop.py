"""Event-driven action-perception loop hooking agents to simulated environments."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from modules.brain.state import PerceptionSnapshot
from modules.environment.simulator import BaseEnvironment
from modules.perception.semantic_bridge import SemanticBridge, SemanticBridgeOutput
from backend.monitoring.global_workspace import WorkspaceMessage, global_workspace

LOGGER = logging.getLogger(__name__)


class ActionPerceptionLoop:
    """Listen for agent actions, forward them to an environment, and integrate feedback."""

    def __init__(
        self,
        event_bus: Any,
        environment: BaseEnvironment,
        *,
        knowledge_pipeline: Any | None = None,
        semantic_bridge: SemanticBridge | None = None,
    ) -> None:
        self._bus = event_bus
        self._environment = environment
        self._knowledge_pipeline = knowledge_pipeline
        self._semantic_bridge = semantic_bridge or SemanticBridge()
        self._bus.subscribe("agent.action.executed", self._on_action_event)

    async def _on_action_event(self, event: Dict[str, Any]) -> None:
        agent_id = event.get("agent")
        command = event.get("command")
        arguments = event.get("arguments") or {}
        if not command:
            return

        self.step_and_process(
            agent_id=agent_id,
            task_id=event.get("task_id") or f"env:{agent_id}",
            cycle=event.get("cycle"),
            command=str(command),
            arguments=dict(arguments) if isinstance(arguments, dict) else {},
            ingest=False,
            publish=True,
            status="environment_feedback",
            auto_reset=True,
        )

    def reset_environment(self) -> Dict[str, Any]:
        return self._environment.reset()

    def process_reset(
        self,
        *,
        agent_id: Optional[str] = None,
        task_id: Optional[str] = None,
        cycle: Optional[int] = None,
        ingest: bool = False,
        publish: bool = True,
        broadcast_workspace: bool = True,
        status: str = "environment_reset",
    ) -> Optional[Dict[str, Any]]:
        """Reset the environment and return a processed ``environment.perception`` payload."""

        try:
            step_result = self._environment.reset()
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.debug("Environment reset failed: %s", exc, exc_info=True)
            return None
        return self._process_step_result(
            agent_id=agent_id,
            task_id=task_id,
            cycle=cycle,
            step_result=step_result,
            ingest=ingest,
            publish=publish,
            broadcast_workspace=broadcast_workspace,
            status=status,
        )

    def step_and_process(
        self,
        *,
        agent_id: Optional[str],
        task_id: Optional[str],
        cycle: Optional[int],
        command: str,
        arguments: Dict[str, Any] | None = None,
        ingest: bool = False,
        publish: bool = True,
        broadcast_workspace: bool = True,
        status: str = "environment_feedback",
        auto_reset: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Execute one environment step and return a processed perception payload.

        This is the synchronous equivalent of handling the ``agent.action.executed`` event.
        """

        try:
            step_result = self._environment.step(command, arguments or {})
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.debug("Environment step failed: %s", exc, exc_info=True)
            return None

        perception_event = self._process_step_result(
            agent_id=agent_id,
            task_id=task_id,
            cycle=cycle,
            step_result=step_result,
            ingest=ingest,
            publish=publish,
            broadcast_workspace=broadcast_workspace,
            status=status,
        )

        if auto_reset and step_result.get("done"):
            try:
                self._environment.reset()
            except Exception:
                LOGGER.debug("Environment reset failed after termination.", exc_info=True)

        return perception_event

    def _process_step_result(
        self,
        *,
        agent_id: Optional[str],
        task_id: Optional[str],
        cycle: Optional[int],
        step_result: Dict[str, Any],
        ingest: bool,
        publish: bool,
        broadcast_workspace: bool,
        status: str,
    ) -> Optional[Dict[str, Any]]:
        observation = step_result.get("observation") or {}
        if not observation:
            return None

        snapshot = PerceptionSnapshot(modalities=dict(observation))
        bridge_output = self._semantic_bridge.process(
            snapshot,
            agent_id=agent_id,
            cycle_index=cycle,
            ingest=bool(ingest),
        )

        knowledge_statements = _build_statements(bridge_output.semantic_annotations)
        perception_event = {
            "agent_id": agent_id,
            "task_id": task_id or (f"env:{agent_id}" if agent_id else "env:unknown"),
            "status": status,
            "detail": step_result.get("info"),
            "summary": "; ".join(knowledge_statements) if knowledge_statements else None,
            "knowledge_statements": knowledge_statements,
            "knowledge_facts": bridge_output.knowledge_facts,
            "fused_embedding": bridge_output.fused_embedding,
            "modality_embeddings": bridge_output.modality_embeddings,
            "metadata": {
                "reward": step_result.get("reward"),
                "done": step_result.get("done"),
            },
        }

        if self._knowledge_pipeline is not None:
            try:
                self._knowledge_pipeline.process_task_event(perception_event)
            except Exception:  # pragma: no cover - defensive guard
                LOGGER.debug("Knowledge pipeline failed for environment event.", exc_info=True)

        if publish:
            try:
                self._bus.publish("environment.perception", perception_event)
            except Exception:  # pragma: no cover - event bus optional
                LOGGER.debug("Failed to publish environment perception event.", exc_info=True)

        if broadcast_workspace:
            self._broadcast_workspace_feedback(
                agent_id=agent_id,
                cycle=cycle,
                observation=observation,
                bridge_output=bridge_output,
                perception_event=perception_event,
                step_result=step_result,
            )

        return perception_event

    def _broadcast_workspace_feedback(
        self,
        *,
        agent_id: Optional[str],
        cycle: Optional[int],
        observation: Dict[str, Any],
        bridge_output: SemanticBridgeOutput,
        perception_event: Dict[str, Any],
        step_result: Dict[str, Any],
    ) -> None:
        source_name = getattr(self._environment, "name", None) or self._environment.__class__.__name__
        summary = perception_event.get("summary")
        if not summary:
            statements = perception_event.get("knowledge_statements") or []
            if statements:
                summary = "; ".join(statements)

        payload = {
            "agent_id": agent_id,
            "cycle": cycle,
            "environment": source_name,
            "observation": observation,
            "annotations": bridge_output.semantic_annotations,
            "knowledge_facts": bridge_output.knowledge_facts,
            "fused_embedding": bridge_output.fused_embedding,
            "modality_embeddings": bridge_output.modality_embeddings,
            "metadata": perception_event.get("metadata", {}),
        }

        reward = step_result.get("reward")
        try:
            importance = float(reward) if reward is not None else 0.0
        except (TypeError, ValueError):
            importance = 0.0

        try:
            global_workspace.publish_message(
                WorkspaceMessage(
                    type="perception.observation",
                    source=f"environment:{source_name}",
                    payload=payload,
                    summary=summary,
                    tags=("environment", "perception"),
                    importance=max(0.0, abs(importance)),
                ),
                attention=reward,
                propagate=True,
            )
        except Exception:  # pragma: no cover - defensive guard
            LOGGER.debug(
                "Failed to publish environment feedback to workspace.",
                exc_info=True,
            )


def _build_statements(annotations: Dict[str, Dict[str, Any]]) -> list[str]:
    statements: list[str] = []
    for modality, data in annotations.items():
        summary = data.get("summary")
        labels = ", ".join(data.get("labels", []))
        if summary:
            statements.append(f"{modality}: {summary}")
        elif labels:
            statements.append(f"{modality}: {labels}")
    return statements
