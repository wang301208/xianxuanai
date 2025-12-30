"""World model module.

Provides a structured, multimodal representation of the environment together
with lightweight predictive and introspective utilities. Beyond tracking
current state, the model can simulate hypothetical futures, surface knowledge
gaps, and highlight emerging opportunities so higher-level planners can set
autonomous goals.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Sequence

from .multimodal import MultimodalStore, SensorEncoder
from .vision import VisionStore


DynamicsModel = Callable[[Dict[str, Any], Dict[str, Any], "WorldModel"], Dict[str, Any]]


class WorldModel:
    """Maintain a structured state of the environment.

    The model tracks tasks, resource usage of agents and their recent actions.
    A lightweight learning component keeps exponentially-weighted moving
    averages of resource usage, which serve as predictions for future usage.
    """

    def __init__(self, alpha: float = 0.5) -> None:
        """Create a new world model.

        Args:
            alpha: Smoothing factor for the moving average used in learning
                resource usage patterns. ``0 < alpha <= 1``.
        """

        self.alpha = alpha
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.resources: Dict[str, Dict[str, float]] = {}
        self.actions: List[Dict[str, str]] = []
        self._predictions: Dict[str, Dict[str, float]] = {}
        self.vision = VisionStore()
        self._multimodal = MultimodalStore()
        self._multimodal.register_modality("vision")
        self._multimodal.register_modality("text")
        self._dynamics_models: Dict[str, DynamicsModel] = {}
        self._competence: Dict[str, Dict[str, Any]] = {}
        self._opportunities: Dict[str, float] = {}
        self._opportunity_meta: Dict[str, Dict[str, Any]] = {}

    @property
    def multimodal(self) -> Dict[str, Any]:
        """Map of agents to their latest fused multimodal embeddings."""

        fused: Dict[str, Any] = {}
        for agent_id in self._multimodal.agents():
            unified = self._multimodal.unified(agent_id)
            if unified is not None:
                if hasattr(unified, "tolist"):
                    fused[agent_id] = list(unified.tolist())
                else:
                    fused[agent_id] = list(unified)
        return fused

    # ------------------------------------------------------------------
    # State management APIs
    # ------------------------------------------------------------------
    def add_task(self, task_id: str, metadata: Dict[str, Any]) -> None:
        """Add or update a task in the world model."""

        self.tasks[task_id] = metadata

    def update_resources(self, agent_id: str, usage: Dict[str, float]) -> None:
        """Update the current resource usage for an agent.

        This method also updates the internal prediction for the agent using an
        exponentially-weighted moving average (EWMA) based on past interactions.
        """

        self.resources[agent_id] = usage

        prev = self._predictions.get(agent_id)
        if prev is None:
            self._predictions[agent_id] = {
                "cpu": usage.get("cpu", 0.0),
                "memory": usage.get("memory", 0.0),
            }
        else:
            self._predictions[agent_id] = {
                "cpu": self.alpha * usage.get("cpu", 0.0)
                + (1 - self.alpha) * prev.get("cpu", 0.0),
                "memory": self.alpha * usage.get("memory", 0.0)
                + (1 - self.alpha) * prev.get("memory", 0.0),
            }

    def get_state(self) -> Dict[str, Any]:
        """Return a snapshot of the current world state."""

        return {
            "tasks": dict(self.tasks),
            "resources": {k: dict(v) for k, v in self.resources.items()},
            "actions": list(self.actions),
            "vision": self.vision.all(),
            "multimodal": self._multimodal.snapshot(),
            "competence": self._competence_snapshot(),
            "opportunities": self._opportunity_snapshot(),
        }

    def add_visual_observation(
        self,
        agent_id: str,
        image: Any | None = None,
        features: Any | None = None,
        vit_features: Any | None = None,
        text: Any | None = None,
    ) -> None:
        """Store visual data for ``agent_id``.

        Parameters
        ----------
        agent_id:
            Identifier of the agent that produced the observation.
        image:
            Optional raw image tensor or array.
        features:
            Optional feature vector representing the image.
        vit_features:
            Optional feature vector produced by a ViT model.
        text:
            Optional textual embedding associated with the observation.
        """

        unified = self.vision.ingest(
            agent_id,
            image=image,
            features=features,
            vit_features=vit_features,
            text=text,
        )
        vision_features = features or vit_features or unified
        if image is not None or vision_features is not None:
            self._multimodal.ingest(
                agent_id,
                "vision",
                data=image,
                features=vision_features,
            )
        if text is not None:
            self._multimodal.ingest(
                agent_id,
                "text",
                data=text,
                features=text,
            )

    def get_visual_observation(self, agent_id: str) -> Dict[str, Any]:
        """Retrieve the latest visual observation for ``agent_id``."""

        return self.vision.get(agent_id)

    def get_unified_representation(self, agent_id: str) -> Any:
        """Retrieve the unified multimodal representation for ``agent_id``."""

        return self._multimodal.unified(agent_id)

    def register_modality(self, name: str, encoder: Optional[SensorEncoder] = None) -> None:
        """Register a new sensory modality and optional encoder."""

        self._multimodal.register_modality(name, encoder)

    def add_multimodal_observation(
        self,
        agent_id: str,
        modality: str,
        *,
        data: Any | None = None,
        features: Any | None = None,
    ) -> Any:
        """Record an observation for an arbitrary modality."""

        return self._multimodal.ingest(agent_id, modality, data=data, features=features)

    def get_multimodal_observation(self, agent_id: str) -> Dict[str, Any]:
        """Return all modality records stored for ``agent_id``."""

        modalities = self._multimodal.modalities(agent_id)
        return {
            name: {
                "raw": record.raw,
                "features": record.features,
            }
            for name, record in modalities.items()
        }

    # ------------------------------------------------------------------
    # Prediction APIs
    # ------------------------------------------------------------------
    def predict(
        self,
        agent_id: Optional[str] = None,
        resources: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, float]:
        """Predict resource usage.

        Parameters
        ----------
        agent_id:
            When provided, return predictions specific to this agent.
        resources:
            Optional snapshot of current resource usage for multiple agents.
            When supplied, the snapshot is incorporated before computing
            predictions so callers without prior calls to :meth:`update_resources`
            can still obtain meaningful averages.  For backwards compatibility a
            mapping passed as the first argument is also accepted.
        """

        if isinstance(agent_id, dict) and resources is None:
            resources = agent_id
            agent_id = None

        if resources:
            for res_agent, usage in resources.items():
                if not isinstance(usage, dict):
                    continue
                self.update_resources(str(res_agent), usage)

        if agent_id is not None:
            return self._predictions.get(agent_id, {"cpu": 0.0, "memory": 0.0})

        if not self._predictions:
            return {"avg_cpu": 0.0, "avg_memory": 0.0}

        total_cpu = sum(p.get("cpu", 0.0) for p in self._predictions.values())
        total_mem = sum(p.get("memory", 0.0) for p in self._predictions.values())
        count = len(self._predictions) or 1
        return {"avg_cpu": total_cpu / count, "avg_memory": total_mem / count}

    # ------------------------------------------------------------------
    # Predictive planning and self-assessment APIs
    # ------------------------------------------------------------------
    def register_dynamics(self, domain: str, model: DynamicsModel) -> None:
        """Register a simulation model for ``domain`` specific actions."""

        if not domain:
            raise ValueError("domain must be a non-empty string")
        self._dynamics_models[domain] = model

    def simulate(
        self,
        plan: Sequence[Dict[str, Any]] | None = None,
        *,
        context: Optional[Dict[str, Any]] = None,
        horizon: int = 3,
    ) -> List[Dict[str, Any]]:
        """Project future states under a hypothetical ``plan``.

        The returned trajectory contains shallow copies of the simulated world
        state at each step so callers can inspect competence projections and
        resource demand without mutating the live world model.
        """

        if horizon <= 0:
            return []

        snapshot = context or self.get_state()
        state = deepcopy(snapshot)
        steps = max(1, horizon)
        actions = list(plan or [])
        trajectory: List[Dict[str, Any]] = []

        for idx in range(steps):
            action = deepcopy(actions[idx] if idx < len(actions) else {})
            domain = str(
                action.get("domain")
                or action.get("capability")
                or action.get("skill")
                or "generic"
            )
            dynamics = self._dynamics_models.get(domain, self._generic_dynamics)
            state = dynamics(state, action, self)
            trajectory.append(deepcopy(state))

        return trajectory

    def project_goal_outcomes(
        self,
        goal: str,
        *,
        candidates: Optional[Sequence[Dict[str, Any]]] = None,
        horizon: int = 3,
    ) -> List[Dict[str, Any]]:
        """Helper that converts a text ``goal`` into a simulated trajectory."""

        domain = self._infer_domain_from_goal(goal)
        actions: List[Dict[str, Any]]
        if candidates:
            actions = [dict(action) for action in candidates]
        else:
            actions = [
                {
                    "description": goal,
                    "domain": domain,
                    "agent_id": "self",
                    "estimated_load": 5.0,
                    "learning_rate": 0.08,
                }
            ]
        return self.simulate(actions, horizon=horizon)

    def update_competence(
        self,
        domain: str,
        confidence: float,
        *,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Blend a new competence estimate for ``domain`` into the model."""

        domain = domain.strip()
        if not domain:
            return
        score = max(0.0, min(1.0, float(confidence)))
        entry = self._competence.setdefault(domain, {"score": 0.0, "samples": 0})
        samples = max(0, int(entry.get("samples", 0)))
        entry["score"] = (entry.get("score", 0.0) * samples + score) / (samples + 1)
        entry["samples"] = samples + 1
        if source:
            entry["last_source"] = source
        if metadata:
            existing = entry.setdefault("metadata", {})
            existing.update(metadata)

    def knowledge_gaps(self, threshold: float = 0.6) -> List[str]:
        """Return domains with competence below ``threshold``."""

        return [
            domain
            for domain, entry in self._competence.items()
            if entry.get("score", 0.0) < threshold
        ]

    def suggest_learning_targets(
        self,
        limit: int = 3,
        threshold: float = 0.6,
    ) -> List[str]:
        """Rank domains that would benefit most from deliberate practice."""

        gaps = [
            (domain, entry.get("score", 0.0))
            for domain, entry in self._competence.items()
            if entry.get("score", 0.0) < threshold
        ]
        gaps.sort(key=lambda item: item[1])
        return [domain for domain, _ in gaps[:limit]]

    def record_opportunity(
        self,
        topic: str,
        *,
        weight: float = 0.2,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an intrinsic opportunity or curiosity driver."""

        topic = topic.strip()
        if not topic:
            return
        updated = max(0.0, self._opportunities.get(topic, 0.0) + max(0.0, float(weight)))
        self._opportunities[topic] = min(1.0, updated)
        if metadata:
            store = self._opportunity_meta.setdefault(topic, {})
            store.update(metadata)

    def suggest_opportunities(
        self,
        limit: int = 3,
        threshold: float = 0.4,
    ) -> List[Dict[str, Any]]:
        """Return the highest-weighted intrinsic opportunities."""

        items = [
            {
                "topic": topic,
                "weight": weight,
                "metadata": dict(self._opportunity_meta.get(topic, {})),
            }
            for topic, weight in self._opportunities.items()
            if weight >= threshold
        ]
        items.sort(key=lambda item: item["weight"], reverse=True)
        return items[:limit]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _competence_snapshot(self) -> Dict[str, Dict[str, Any]]:
        return {key: dict(value) for key, value in self._competence.items()}

    def _opportunity_snapshot(self) -> List[Dict[str, Any]]:
        if not self._opportunities:
            return []
        return [
            {
                "topic": topic,
                "weight": weight,
                "metadata": dict(self._opportunity_meta.get(topic, {})),
            }
            for topic, weight in self._opportunities.items()
        ]

    @staticmethod
    def _infer_domain_from_goal(goal: str) -> str:
        lowered = goal.lower()
        if "research" in lowered or "learn" in lowered:
            return "research"
        if "optimiz" in lowered:
            return "optimization"
        if "strategy" in lowered:
            return "strategy"
        return "generic"

    def _generic_dynamics(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        model: "WorldModel",
    ) -> Dict[str, Any]:
        """Fallback transition model that approximates resource and competence changes."""

        next_state = deepcopy(state)

        agent_id = str(action.get("agent_id") or action.get("actor") or "agent")
        load = float(action.get("estimated_load") or action.get("load") or 0.0)
        learning_rate = float(action.get("learning_rate") or 0.05)
        domain = str(
            action.get("domain")
            or action.get("capability")
            or action.get("skill")
            or action.get("goal")
            or "generic"
        )

        resources = next_state.setdefault("resources", {})
        snapshot = dict(resources.get(agent_id, {"cpu": 0.0, "memory": 0.0}))
        if load:
            snapshot["cpu"] = max(0.0, snapshot.get("cpu", 0.0) + load * 0.5)
            snapshot["memory"] = max(0.0, snapshot.get("memory", 0.0) + load)
        resources[agent_id] = snapshot
        next_state["resources"] = resources

        competence_state = next_state.setdefault("competence", {})
        baseline = model._competence.get(domain, {"score": 0.0, "samples": 0})
        projected = dict(baseline)
        projected_score = min(1.0, max(0.0, projected.get("score", 0.0) + learning_rate))
        projected["score"] = projected_score
        projected["samples"] = projected.get("samples", 0) + 1
        projected["last_source"] = "simulation"
        competence_state[domain] = projected
        next_state["competence"] = competence_state

        curiosity = action.get("discover") or action.get("topic")
        if curiosity:
            topic = str(curiosity)
            weight = float(action.get("opportunity_weight") or learning_rate)
            current = next_state.setdefault("opportunities", {})
            current[topic] = min(1.0, max(0.0, current.get(topic, 0.0) + weight))
            next_state["opportunities"] = current

        return next_state

    def _update_competence_from_record(self, record: Dict[str, Any]) -> None:
        metadata = record.get("metadata") or {}
        domain = (
            metadata.get("domain")
            or metadata.get("capability")
            or metadata.get("skill")
            or metadata.get("topic")
            or self._infer_domain_from_goal(str(metadata.get("goal") or record.get("action") or ""))
        )
        if not domain:
            return

        status = str(record.get("status") or "").lower()
        confidence = metadata.get("confidence")
        if confidence is None:
            if status in {"failed", "error", "timeout"}:
                confidence = 0.2
            elif status in {"completed", "success", "done"}:
                confidence = 0.9
            else:
                confidence = 0.5

        self.update_competence(
            str(domain),
            float(confidence),
            source="action",
            metadata={"last_action": record.get("action"), "status": status},
        )

        if status in {"failed", "error", "timeout"}:
            self.record_opportunity(
                f"Investigate weaknesses in {domain}",
                weight=0.3,
                metadata={"origin": "action_failure", "domain": domain},
            )

    # Override to hook competence updates after logging actions
    def record_action(
        self,
        agent_id: str,
        action: str,
        *,
        status: str | None = None,
        result: str | None = None,
        error: str | None = None,
        metrics: Optional[Dict[str, float]] = None,
        retries: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an action performed by an agent along with outcome details."""

        record: Dict[str, Any] = {"agent_id": agent_id, "action": action}
        if status is not None:
            record["status"] = status
        if result is not None:
            record["result"] = result
        if error is not None:
            record["error"] = error
        if retries:
            record["retries"] = int(retries)
        if metrics:
            clean_metrics: Dict[str, float] = {}
            for key, value in metrics.items():
                try:
                    clean_metrics[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
            if clean_metrics:
                record["metrics"] = clean_metrics
        if metadata:
            record["metadata"] = dict(metadata)
        self.actions.append(record)
        self._update_competence_from_record(record)


__all__ = ["WorldModel", "VisionStore", "MultimodalStore"]

