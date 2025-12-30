"""
Cognitive controller orchestrating high-level cognitive components.

The BrainSimulationSystem test-suite expects this controller to:
- expose `.network`, `.params`, `.components`
- maintain a mutable `workspace` (attention traces)
- generate metacognitive self-evaluation reports
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import time

from .attention import AttentionSystem
from .self_model import SelfAwarenessModule
from .working_memory import WorkingMemory


class CognitiveController:
    """Orchestrate attention, working memory, and self-monitoring."""

    def __init__(
        self,
        network: Any = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if params is None and "params" in kwargs:
            params = kwargs["params"]
        if isinstance(network, dict) and params is None:
            params = network
            network = None

        self.network = network
        self.params: Dict[str, Any] = params or {}

        self.workspace: Dict[str, Any] = {
            "_attention": {},
            "_attention_focus": [],
        }
        self.control_signals: Dict[str, Dict[str, float]] = {}
        self.meta_context: Dict[str, Any] = {"reflections": []}
        self._time = 0.0

        attention_params = self.params.get("attention", {}) if isinstance(self.params, dict) else {}
        wm_params = self.params.get("working_memory", {}) if isinstance(self.params, dict) else {}
        self_model_cfg = (self.params.get("self_model", {}) if isinstance(self.params, dict) else {}).get(
            "module", {}
        )

        # Components are optional; keep lightweight defaults for unit tests.
        self.components: Dict[str, Any] = {
            "attention": AttentionSystem(network, attention_params),
            "working_memory": WorkingMemory(network, wm_params),
            "self_model": SelfAwarenessModule(self_model_cfg),
        }

        # Backwards-compatible configuration fields (used by older dialogue-oriented code paths).
        self.urgency_threshold = float(self.params.get("urgency_threshold", 0.6))
        self.acknowledge_tones = set(self.params.get("acknowledge_tones", ["polite", "uncertain"]))
        self.request_clarification_tones = set(self.params.get("clarify_tones", ["uncertain", "confused"]))
        self.minimum_confidence = float(self.params.get("minimum_confidence", 0.45))
        self.retrieve_on_unknown = bool(self.params.get("retrieve_on_unknown", True))

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run one controller step and return a combined cognitive snapshot."""

        sensory_input = inputs.get("sensory_input") or {}
        task_goal = inputs.get("task_goal")
        incoming_controls = inputs.get("control_signals") or {}
        if isinstance(incoming_controls, dict):
            for key, payload in incoming_controls.items():
                if isinstance(payload, dict):
                    self.control_signals.setdefault(str(key), {}).update(
                        {str(k): float(v) for k, v in payload.items() if isinstance(v, (int, float))}
                    )

        allocation = self._allocate_attention(
            sensory_input=sensory_input,
            task_goal=str(task_goal) if task_goal is not None else "",
            control_signals=incoming_controls if isinstance(incoming_controls, dict) else {},
        )

        self.workspace["_attention"] = allocation["weights"]
        self.workspace["_attention_focus"] = allocation["focus"]

        self_evaluation = self._self_monitor(
            task_goal=task_goal,
            attention_weights=allocation["weights"],
        )

        return {
            "attention_allocation": allocation,
            "self_evaluation": self_evaluation,
        }

    def decide(
        self,
        intent: str,
        action_plan: Any,
        affect_info: Any,
        memory_state: Dict[str, Any],
        semantic_info: Any,
    ) -> Dict[str, Any]:
        tone = self._extract_field(affect_info, "tone", [])
        polarity = self._extract_field(affect_info, "polarity", "neutral")
        actions = self._extract_field(action_plan, "actions", [])
        action_confidence = self._extract_field(action_plan, "confidence", 0.5)
        semantic_confidence = getattr(semantic_info, "confidence", None)
        key_terms = getattr(semantic_info, "key_terms", []) if semantic_info else []

        template_hint = self._suggest_template(intent, actions, tone, polarity, memory_state)
        priority = "normal"
        if "urgent" in tone or action_confidence >= self.urgency_threshold:
            priority = "high"

        should_ack = bool(self.acknowledge_tones.intersection(tone)) or intent == "statement"
        should_clarify = (
            "uncertain" in tone
            or intent == "question"
            and (semantic_confidence is not None and semantic_confidence < self.minimum_confidence)
        )

        retrieval_needed = False
        if self.retrieve_on_unknown and intent == "question":
            if semantic_confidence is not None and semantic_confidence < self.minimum_confidence:
                retrieval_needed = True
        if intent == "command" and not actions:
            retrieval_needed = True

        return {
            "template_hint": template_hint,
            "priority": priority,
            "should_acknowledge": should_ack,
            "tones": tone,
            "polarity": polarity,
            "request_clarification": should_clarify,
            "trigger_retrieval": retrieval_needed,
            "focus_terms": key_terms[:4],
        }

    def _suggest_template(
        self,
        intent: str,
        actions: Sequence[str],
        tone: Sequence[str],
        polarity: str,
        memory_state: Dict[str, Any],
    ) -> Optional[str]:
        if "urgent" in tone:
            return "confirm_urgent"
        if intent == "question":
            return "answer"
        if intent == "command":
            return "confirm"
        if intent == "greeting":
            return "greet_back"
        if intent == "statement" and polarity == "negative":
            return "acknowledge"
        if memory_state.get("recent_intent") == "command" and actions:
            return "confirm"
        return None

    @staticmethod
    def _extract_field(obj: Any, field: str, default: Any) -> Any:
        if obj is None:
            return default
        if hasattr(obj, field):
            return getattr(obj, field)
        if isinstance(obj, dict):
            return obj.get(field, default)
        return default

    def _allocate_attention(
        self,
        *,
        sensory_input: Dict[str, Any],
        task_goal: str,
        control_signals: Dict[str, Any],
    ) -> Dict[str, Any]:
        attention_params = self.params.get("attention", {}) if isinstance(self.params, dict) else {}
        bottom_up_weight = float(attention_params.get("bottom_up_weight", 0.5))
        top_down_weight = float(attention_params.get("top_down_weight", 0.5))
        span = int(attention_params.get("attention_span", 3))

        raw: Dict[str, float] = {}
        channels = set(str(k) for k in (sensory_input or {}).keys()) | set(str(k) for k in (control_signals or {}).keys())
        goal_text = (task_goal or "").lower()

        for channel in channels:
            sensory_val = sensory_input.get(channel, 0.0)
            try:
                salience = abs(float(sensory_val))
            except (TypeError, ValueError):
                salience = 0.0

            ctrl_val = 0.0
            ctrl_payload = control_signals.get(channel)
            if isinstance(ctrl_payload, dict):
                ctrl_val = float(ctrl_payload.get("attention", 0.0) or 0.0)

            goal_bias = 0.35 if channel.lower() and channel.lower() in goal_text else 0.0
            score = bottom_up_weight * salience + top_down_weight * (max(0.0, ctrl_val) + goal_bias)
            raw[channel] = max(0.0, float(score))

        total = sum(raw.values())
        if total <= 0:
            if not channels:
                return {"weights": {}, "focus": []}
            uniform = 1.0 / len(channels)
            weights = {ch: uniform for ch in channels}
        else:
            weights = {ch: val / total for ch, val in raw.items()}

        focus = [k for k, _ in sorted(weights.items(), key=lambda item: item[1], reverse=True)[: max(1, span)]]
        return {"weights": weights, "focus": focus}

    def _self_monitor(self, *, task_goal: Any, attention_weights: Dict[str, float]) -> Dict[str, Any]:
        self._time += 1.0
        module: SelfAwarenessModule = self.components.get("self_model")  # type: ignore[assignment]

        focus = [
            {"source": name, "score": float(attention_weights.get(name, 0.0))}
            for name in list(self.workspace.get("_attention_focus", []))[:3]
        ]
        report = module.observe(
            time_point=self._time,
            goals=[task_goal] if task_goal else [],
            decision_result={"confidence": float(self.params.get("decision_confidence", 0.0))},
            attention_focus=focus,
            attention_scores=[{"source": k, "score": v} for k, v in attention_weights.items()],
            working_memory={"items": list(self.components.get("working_memory").to_list()) if self.components.get("working_memory") else []},  # type: ignore[union-attr]
        )

        alerts = list(report.get("alerts", [])) if isinstance(report, dict) else []
        summary = None
        if isinstance(report, dict):
            summary = report.get("insight") or report.get("summary")
        if not summary:
            summary = f"Self-evaluation generated at {time.time():.0f}"

        reflection_entry = {
            "time": report.get("time", self._time) if isinstance(report, dict) else self._time,
            "alerts": alerts,
            "summary": summary,
        }
        self.meta_context.setdefault("reflections", []).append(reflection_entry)

        if "high_uncertainty" in alerts:
            attention_ctrl = self.control_signals.setdefault("attention", {})
            attention_ctrl["attention"] = max(float(attention_ctrl.get("attention", 0.0) or 0.0), 0.75)

        return {"summary": summary, "alerts": alerts, "beliefs": report.get("beliefs") if isinstance(report, dict) else {}}
