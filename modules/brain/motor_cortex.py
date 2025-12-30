from __future__ import annotations

"""Motor cortex integrating cortical planning with actuators."""

import logging

from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Dict, Optional, Tuple, Sequence

from .motor.precision import PrecisionMotorSystem
from .neuromorphic.spiking_network import NeuromorphicRunResult

from .motor.actions import (
    ActionMapping,
    ActuatorInterface,
    MotorCommand,
    MotorExecutionResult,
    MotorPlan,
)


class PrimaryMotor:
    """Primary motor cortex responsible for executing motor commands."""

    def execute(self, motor_command: str) -> str:
        return f"executed {motor_command}"


class PreMotorArea:
    """Pre-motor area that plans movements based on intention."""

    def plan(self, intention: str) -> str:
        return f"plan for {intention}"


class SupplementaryMotor:
    """Supplementary motor area that coordinates complex movements."""

    def organize(self, plan: str) -> str:
        return f"organized {plan}"


@dataclass
class MotorCortex:
    """Motor cortex orchestrating planning, ethics, and actuator dispatch."""

    basal_ganglia: Any = None
    cerebellum: Any = None
    spiking_backend: Any = None
    ethics: Any = None
    actuator: ActuatorInterface | None = None
    action_map: Dict[str, ActionMapping] = field(default_factory=dict)
    precision_system: PrecisionMotorSystem | None = None

    def __post_init__(self) -> None:
        self.primary_motor = PrimaryMotor()
        self.premotor_area = PreMotorArea()
        self.supplementary_motor = SupplementaryMotor()
        self.feedback_history: list[MotorExecutionResult] = []
        self._logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Planning ---------------------------------------------------------
    # ------------------------------------------------------------------
    def plan_movement(self, intention: str, parameters: Optional[Dict[str, Any]] = None) -> MotorPlan:
        """Plan a movement and generate a structured `MotorPlan`."""

        parameters = dict(parameters or {})
        neuromorphic_result = parameters.pop("neuromorphic_result", None)
        spike_sequence = parameters.pop("spike_sequence", None)
        modulators = {key: float(value) for key, value in dict(parameters.get("modulators", {})).items()}
        weights = {key: float(value) for key, value in dict(parameters.get("weights", {})).items()}
        if modulators:
            parameters["modulators"] = modulators
        if weights:
            parameters["weights"] = weights

        first_stage = self.premotor_area.plan(intention)
        second_stage = self.supplementary_motor.organize(first_stage)
        stages = [first_stage, second_stage]
        final_stage = stages[-1]
        if self.basal_ganglia:
            final_stage = self._apply_basal_ganglia(final_stage)
            stages.append(final_stage)

        if self.precision_system:
            precision_plan = self.precision_system.plan_movement(intention)
            stages.append(precision_plan)
            final_stage = precision_plan
            parameters.setdefault("precision_plan", precision_plan)
            parameters.setdefault(
                "precision_gating",
                list(getattr(self.precision_system.basal_ganglia, "gating_history", [])),
            )

        plan = MotorPlan(intention=intention, stages=stages, parameters=parameters)
        plan.metadata["plan_summary"] = final_stage
        if modulators:
            plan.metadata["modulators"] = modulators
        plan.command = self._augment_command_with_neuromorphic(
            plan,
            neuromorphic_result=neuromorphic_result,
            spike_sequence=spike_sequence,
            weights=weights,
        )
        return plan

    def register_action(self, intention: str, mapping: ActionMapping) -> None:
        """Register or override an action mapping for a given intention."""

        self.action_map[intention] = mapping

    # ------------------------------------------------------------------
    # Execution --------------------------------------------------------
    # ------------------------------------------------------------------
    def execute_action(self, motor_instruction: Any):
        """Execute motor instructions via the actuator or primary motor."""

        if isinstance(motor_instruction, NeuromorphicRunResult):
            meta_intention = motor_instruction.metadata.get("intention", "neuromorphic")
            motor_instruction = self.plan_movement(
                str(meta_intention),
                parameters={"neuromorphic_result": motor_instruction},
            )
        elif self._looks_like_spike_sequence(motor_instruction):
            if self.spiking_backend is not None:
                try:
                    run_result = None
                    effective_counts: list[float] = []
                    if hasattr(self.spiking_backend, "run_sequence"):
                        run_result = self.spiking_backend.run_sequence(
                            motor_instruction, reset=True
                        )
                    elif hasattr(self.spiking_backend, "learn"):
                        run_result = self.spiking_backend.learn(motor_instruction)
                    if run_result is not None:
                        effective_counts = list(
                            getattr(run_result, "spike_counts", []) or []
                        )
                    if not effective_counts and isinstance(motor_instruction, Sequence):
                        temp_counts: list[float] = []
                        for frame in motor_instruction:
                            if isinstance(frame, Sequence) and not isinstance(frame, (str, bytes)):
                                for idx, value in enumerate(frame):
                                    while idx >= len(temp_counts):
                                        temp_counts.append(0.0)
                                    try:
                                        temp_counts[idx] += float(value)
                                    except (TypeError, ValueError):
                                        continue
                        effective_counts = temp_counts
                    synapses = getattr(self.spiking_backend, "synapses", None)
                    weights = getattr(synapses, "weights", None)
                    if weights is not None and effective_counts:
                        try:
                            max_count = max(abs(float(c)) for c in effective_counts) or 1.0
                            for idx, count in enumerate(effective_counts):
                                delta = float(count) / (max_count * 100.0)
                                if hasattr(weights, "__getitem__") and hasattr(weights, "__setitem__"):
                                    row = weights[idx]
                                    try:
                                        length = len(row)
                                    except TypeError:
                                        length = None
                                    if length is not None:
                                        for col in range(length):
                                            row[col] += delta
                                    else:
                                        weights[idx] = row + delta
                        except Exception as adjust_exc:  # pragma: no cover - defensive
                            self._logger.debug("Spiking weight adaptation failed: %s", adjust_exc)
                except Exception as exc:  # pragma: no cover - defensive safeguard
                    self._logger.debug("Spiking backend execution failed: %s", exc)
            motor_instruction = self.plan_movement(
                "spike_sequence",
                parameters={"spike_sequence": motor_instruction},
            )

        plan, command = self._ensure_command(motor_instruction)

        if self.ethics:
            report = self.ethics.evaluate_action(command.operation)
            if not report.get("compliant", True):
                return report

        tuned_command, textual_hint = self._apply_cerebellum(command, plan)

        if self.precision_system and plan:
            precision_trace = plan.metadata.get("plan_summary", plan.describe())
            precision_feedback = self.precision_system.execute_action(str(precision_trace))
            tuned_command = self._merge_precision_feedback(tuned_command, precision_feedback)
            textual_hint = precision_feedback

        if self.actuator:
            result = self.actuator.execute(tuned_command)
            self._post_execute_feedback(result)
            return result

        command_text = textual_hint or tuned_command.metadata.get("plan_summary", tuned_command.operation)
        return self.primary_motor.execute(str(command_text))

    # ------------------------------------------------------------------
    # Learning ---------------------------------------------------------
    # ------------------------------------------------------------------
    def train(self, feedback: MotorExecutionResult | str | Dict[str, Any]) -> str:
        """Update cerebellar model with rich feedback information."""

        metrics = self.parse_feedback_metrics(feedback)
        if metrics:
            if self.precision_system and hasattr(self.precision_system, "update_feedback"):
                self.precision_system.update_feedback(metrics)
            if self.cerebellum and hasattr(self.cerebellum, "update_feedback"):
                self.cerebellum.update_feedback(metrics)
        if isinstance(feedback, MotorExecutionResult):
            self.feedback_history.append(feedback)
        if not self.cerebellum and not self.precision_system:
            return f"no cerebellum to learn from {feedback}"
        if self.cerebellum and hasattr(self.cerebellum, "motor_learning"):
            result = self.cerebellum.motor_learning(feedback)
        elif self.precision_system and hasattr(self.precision_system.cerebellum, "learn"):
            payload: Any = metrics or str(feedback)
            result = self.precision_system.cerebellum.learn(payload)
        else:
            result = f"no cerebellum to learn from {feedback}"
        return result

    # ------------------------------------------------------------------
    # Internal helpers -------------------------------------------------
    # ------------------------------------------------------------------
    def _map_plan_to_command(self, plan: MotorPlan) -> MotorCommand:
        mapping = self.action_map.get(plan.intention)
        if mapping:
            command = mapping.to_command(plan)
        else:
            metadata = {
                "plan_summary": plan.describe(),
                "stages": list(plan.stages),
                "intention": plan.intention,
            }
            command = MotorCommand("motor", plan.intention, dict(plan.parameters), metadata)
        command.metadata.setdefault("plan_summary", plan.describe())
        return command

    def _ensure_command(self, motor_instruction: Any) -> Tuple[MotorPlan | None, MotorCommand]:
        if isinstance(motor_instruction, MotorPlan):
            command = motor_instruction.command or self._map_plan_to_command(motor_instruction)
            return motor_instruction, command
        if isinstance(motor_instruction, MotorCommand):
            return None, motor_instruction
        if isinstance(motor_instruction, NeuromorphicRunResult):
            intention = motor_instruction.metadata.get("intention", "neuromorphic")
            plan = MotorPlan(
                intention=str(intention),
                stages=["neuromorphic"],
                parameters={"neuromorphic_result": motor_instruction},
                metadata={"plan_summary": "neuromorphic"},
            )
            plan.command = self._augment_command_with_neuromorphic(plan, neuromorphic_result=motor_instruction)
            return plan, plan.command
        if isinstance(motor_instruction, dict) and "operation" in motor_instruction:
            return None, MotorCommand(
                tool=motor_instruction.get("tool", "motor"),
                operation=motor_instruction["operation"],
                arguments=dict(motor_instruction.get("arguments", {})),
                metadata=dict(motor_instruction.get("metadata", {})),
            )
        if isinstance(motor_instruction, str):
            plan = self.plan_movement(motor_instruction)
            return plan, plan.command  # type: ignore[return-value]
        raise TypeError(f"Unsupported motor instruction type: {type(motor_instruction)!r}")

    def _apply_cerebellum(
        self, command: MotorCommand, plan: MotorPlan | None
    ) -> Tuple[MotorCommand, Optional[str]]:
        tuned_command = command
        textual_hint = plan.describe() if plan else None

        if self.cerebellum and hasattr(self.cerebellum, "fine_tune"):
            candidate = self.cerebellum.fine_tune(command)
            if isinstance(candidate, MotorCommand):
                tuned_command = candidate
            if plan:
                textual_hint = self.cerebellum.fine_tune(plan.describe())  # type: ignore[arg-type]
                if isinstance(textual_hint, MotorCommand):
                    textual_hint = textual_hint.metadata.get("plan_summary", plan.describe())

        if self.precision_system and hasattr(self.precision_system.cerebellum, "fine_tune"):
            precision_hint = self.precision_system.cerebellum.fine_tune(
                textual_hint or tuned_command.metadata.get("plan_summary", tuned_command.operation)
            )
            textual_hint = str(precision_hint)
            if isinstance(tuned_command, MotorCommand):
                tuned_command = tuned_command.with_updates(
                    metadata={"precision_hint": textual_hint}
                )

        return tuned_command, textual_hint

    def _apply_basal_ganglia(self, stage: str) -> str:
        if hasattr(self.basal_ganglia, "modulate"):
            return self.basal_ganglia.modulate(stage)
        if hasattr(self.basal_ganglia, "gate"):
            return self.basal_ganglia.gate(stage)
        return stage

    def _augment_command_with_neuromorphic(
        self,
        plan: MotorPlan,
        *,
        neuromorphic_result: NeuromorphicRunResult | None = None,
        spike_sequence: Any = None,
        weights: Dict[str, float] | None = None,
    ) -> MotorCommand:
        base_command = self._map_plan_to_command(plan)
        metadata_updates: Dict[str, Any] = {}
        argument_updates: Dict[str, Any] = {}

        if weights:
            argument_updates["weights"] = weights
        if neuromorphic_result:
            metadata_updates["neuromorphic"] = neuromorphic_result.to_dict()
            channels = neuromorphic_result.metadata.get("channels", [])
            if channels:
                metadata_updates.setdefault("channels", list(channels))
            if neuromorphic_result.average_rate:
                argument_updates["average_rate"] = list(neuromorphic_result.average_rate)
            if neuromorphic_result.spike_counts:
                argument_updates["spike_counts"] = list(neuromorphic_result.spike_counts)
            argument_updates.setdefault("energy_used", neuromorphic_result.energy_used)
            argument_updates.setdefault("idle_skipped", neuromorphic_result.idle_skipped)
            if "modulators" in plan.metadata:
                metadata_updates["modulators"] = plan.metadata["modulators"]
        elif spike_sequence is not None:
            metadata_updates["spike_sequence"] = self._summarise_spike_sequence(spike_sequence)
            if "modulators" in plan.metadata:
                metadata_updates["modulators"] = plan.metadata["modulators"]

        if not argument_updates and not metadata_updates:
            return base_command
        return base_command.with_updates(arguments=argument_updates, metadata=metadata_updates)

    def _summarise_spike_sequence(self, spike_sequence: Any) -> Dict[str, Any]:
        if isinstance(spike_sequence, NeuromorphicRunResult):
            return spike_sequence.to_dict()
        if isinstance(spike_sequence, Sequence):
            flattened = [float(v) for v in self._flatten_sequence(spike_sequence)]
            return {
                "length": len(flattened),
                "sum": float(sum(flattened)),
                "max": max(flattened) if flattened else 0.0,
            }
        return {"length": 0, "sum": 0.0, "max": 0.0}

    def _flatten_sequence(self, sequence: Sequence[Any]) -> list[float]:
        flattened: list[float] = []
        for item in sequence:
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                flattened.extend(self._flatten_sequence(item))
            else:
                try:
                    flattened.append(float(item))
                except (TypeError, ValueError):
                    continue
        return flattened

    def _post_execute_feedback(self, result: MotorExecutionResult) -> None:
        self.feedback_history.append(result)
        metrics = self.parse_feedback_metrics(result)
        if metrics:
            if self.precision_system and hasattr(self.precision_system, "update_feedback"):
                self.precision_system.update_feedback(metrics)
            if self.cerebellum and hasattr(self.cerebellum, "update_feedback"):
                self.cerebellum.update_feedback(metrics)
        if self.cerebellum:
            self.cerebellum.motor_learning(result)

    def _looks_like_spike_sequence(self, value: Any) -> bool:
        if isinstance(value, NeuromorphicRunResult):
            return True
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, MotorPlan, MotorCommand)):
            return any(isinstance(item, (int, float)) or self._looks_like_spike_sequence(item) for item in value)
        return False

    def _merge_precision_feedback(
        self, command: MotorCommand, precision_feedback: Any
    ) -> MotorCommand:
        if isinstance(command, MotorCommand):
            metadata = {"precision_feedback": str(precision_feedback)}
            return command.with_updates(metadata=metadata)
        return command

    # ------------------------------------------------------------------
    # Feedback processing ----------------------------------------------
    # ------------------------------------------------------------------
    def parse_feedback_metrics(
        self,
        feedback: MotorExecutionResult | Dict[str, Any] | str | None,
        *,
        base_reward: float | None = None,
    ) -> Dict[str, float]:
        raw = self._coerce_feedback_metrics(feedback)
        if base_reward is not None and isinstance(base_reward, Real):
            raw.setdefault("reward", float(base_reward))
        return self._structure_feedback_metrics(raw)

    def _coerce_feedback_metrics(
        self, feedback: MotorExecutionResult | Dict[str, Any] | str | None
    ) -> Dict[str, float]:
        if feedback is None:
            return {}
        numeric: Dict[str, float] = {}
        if isinstance(feedback, MotorExecutionResult):
            numeric.update(
                {
                    key: float(value)
                    for key, value in feedback.telemetry.items()
                    if isinstance(value, Real)
                }
            )
            numeric.setdefault("success_rate", 1.0 if feedback.success else 0.0)
            numeric.setdefault("overall_error", 0.0 if feedback.success else 1.0)
            if feedback.error:
                numeric.setdefault("error_flag", 1.0)
        elif isinstance(feedback, dict):
            expanded = dict(feedback)
            embedded = expanded.pop("metrics", None)
            if isinstance(embedded, dict):
                for key, value in embedded.items():
                    if isinstance(value, Real):
                        numeric[key] = float(value)
            for key, value in expanded.items():
                if isinstance(value, Real):
                    numeric[key] = float(value)
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, Real):
                            numeric[f"{key}_{sub_key}"] = float(sub_value)
        return numeric

    def _structure_feedback_metrics(self, raw: Dict[str, float]) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        def _pop(*names: str) -> float | None:
            for name in names:
                if name in raw:
                    return float(raw.pop(name))
            return None

        velocity = _pop("velocity_error", "speed_error", "velocity_delta", "speed_delta", "velocity", "speed")
        stability = _pop("stability_error", "balance_error", "oscillation_error", "stability")
        accuracy = _pop("accuracy_error", "precision_error", "trajectory_error", "accuracy")
        reward = _pop("reward", "reward_signal", "score", "utility")
        energy = _pop("energy_cost", "energy", "power", "energy_used")
        latency = _pop("latency", "response_time", "latency_ms")
        success = _pop("success_rate", "success")
        overall_error = _pop("overall_error", "error", "tracking_error")

        provided_components = {
            "velocity_error": velocity,
            "stability_error": stability,
            "accuracy_error": accuracy,
        }
        provided_total = sum(abs(value) for value in provided_components.values() if value is not None)
        if overall_error is None:
            overall_error = provided_total
        if overall_error is None:
            overall_error = 0.0
        missing = [key for key, value in provided_components.items() if value is None]
        remainder = max(0.0, overall_error - provided_total)
        share = remainder / len(missing) if missing else 0.0
        for key, value in provided_components.items():
            component_value = value if value is not None else share
            metrics[key] = float(max(0.0, component_value))

        metrics["overall_error"] = float(max(0.0, overall_error))
        metrics["reward"] = float(reward) if reward is not None else metrics.get("reward", 0.0)
        if energy is not None:
            metrics["energy_cost"] = float(max(0.0, energy))
        if latency is not None:
            metrics["latency"] = float(max(0.0, latency))
        if success is None:
            success = max(0.0, min(1.0, 1.0 - metrics["overall_error"]))
        else:
            success = max(0.0, min(1.0, success))
        metrics["success_rate"] = float(success)
        for key, value in list(raw.items()):
            metrics[f"extra_{key}"] = float(value)
        return metrics


__all__ = ["MotorCortex", "PrimaryMotor", "PreMotorArea", "SupplementaryMotor"]
