"""Motor control and learning components inspired by the cerebellum."""

from __future__ import annotations

from numbers import Real
from typing import Any, Dict, Iterable, List, Mapping

from .motor.actions import MotorCommand, MotorExecutionResult


class PurkinjeNetwork:
    """Network of Purkinje cells for refining motor signals."""

    def __init__(self, *, kp: float = 0.6, ki: float = 0.12, kd: float = 0.08) -> None:
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self._integral: Dict[str, float] = {}
        self._previous_error: Dict[str, float] = {}

    def refine(self, signal: Mapping[str, Real], *, update_state: bool = True) -> Dict[str, float]:
        """Compute PID-style corrections for the provided error signal."""

        corrections: Dict[str, float] = {}
        for key, value in signal.items():
            try:
                error = float(value)
            except (TypeError, ValueError):
                continue

            last_integral = self._integral.get(key, 0.0)
            last_error = self._previous_error.get(key, 0.0)

            integral = last_integral + error
            derivative = error - last_error

            if update_state:
                self._integral[key] = integral
                self._previous_error[key] = error

            correction = self.kp * error + self.ki * integral + self.kd * derivative
            corrections[key] = correction

        return corrections


class GranuleNetwork:
    """Network of granule cells preprocessing inputs."""

    def __init__(self, *, smoothing: float = 0.25) -> None:
        self.smoothing = float(smoothing)
        self._running_mean: Dict[str, float] = {}

    def process(
        self,
        input_signal: Mapping[str, Real] | MotorExecutionResult | MotorCommand | Iterable[tuple[str, Real]] | None,
    ) -> Dict[str, float]:
        """Filter incoming signals and return normalised residual errors."""

        telemetry = self._extract_numeric(input_signal)
        processed: Dict[str, float] = {}

        for key, value in telemetry.items():
            baseline = self._running_mean.get(key, 0.0)
            baseline += self.smoothing * (float(value) - baseline)
            self._running_mean[key] = baseline
            processed[key] = float(value) - baseline

        return processed

    def _extract_numeric(
        self, signal: Mapping[str, Real] | MotorExecutionResult | MotorCommand | Iterable[tuple[str, Real]] | None
    ) -> Dict[str, float]:
        if signal is None:
            return {}
        if isinstance(signal, MotorExecutionResult):
            return {key: float(value) for key, value in signal.telemetry.items() if isinstance(value, Real)}
        if isinstance(signal, MotorCommand):
            return {key: float(value) for key, value in signal.arguments.items() if isinstance(value, Real)}
        if isinstance(signal, Mapping):
            return {key: float(value) for key, value in signal.items() if isinstance(value, Real)}
        telemetry: Dict[str, float] = {}
        for item in signal:
            try:
                key, value = item
            except (TypeError, ValueError):
                continue
            if isinstance(key, str) and isinstance(value, Real):
                telemetry[key] = float(value)
        return telemetry


class Cerebellum:
    """Simplified cerebellum coordinating motor control and learning."""

    def __init__(self) -> None:
        self.purkinje = PurkinjeNetwork()
        self.granule = GranuleNetwork()
        self.learned_signals: List[Dict[str, Any]] = []
        self.metric_history: List[Dict[str, float]] = []
        self._latest_error_signal: Dict[str, float] = {}
        self._latest_corrections: Dict[str, float] = {}
        self._balance_state: Dict[str, float] = {}

    def fine_tune(self, motor_command: str | MotorCommand) -> str | MotorCommand:
        """Refine motor commands or command objects for smoother execution."""

        if isinstance(motor_command, MotorCommand):
            return self._fine_tune_command(motor_command)
        return self._summarise_state()

    def motor_learning(self, feedback: str | Dict[str, Any] | MotorExecutionResult) -> Dict[str, Any]:
        """Update internal state using structured feedback signals."""

        record = self._normalise_feedback(feedback)
        telemetry = record.get("telemetry", {})
        numeric_telemetry = {
            key: float(value)
            for key, value in dict(telemetry).items()
            if isinstance(value, Real)
        }

        if record.get("success") is not None:
            numeric_telemetry.setdefault("success_rate", 1.0 if record["success"] else 0.0)

        if numeric_telemetry:
            self.metric_history.append(dict(numeric_telemetry))

        filtered_error = self.granule.process(numeric_telemetry) if numeric_telemetry else {}
        self._latest_error_signal = dict(filtered_error)
        corrections = self.purkinje.refine(filtered_error) if filtered_error else {}
        self._latest_corrections = dict(corrections)

        sample = {
            "telemetry": numeric_telemetry,
            "filtered_error": filtered_error,
            "corrections": corrections,
        }
        if "success" in record:
            sample["success"] = bool(record["success"])
        if record.get("label"):
            sample["label"] = record["label"]

        self.learned_signals.append(sample)
        return sample

    def balance_control(self, sensory_input: Mapping[str, Real] | str) -> Dict[str, Any]:
        """Update balance estimates using structured sensory feedback."""

        metrics = self._parse_structured_input(sensory_input)
        if not metrics:
            return {"adjustments": {}, "filtered": {}, "status": "insufficient_data"}

        filtered = self.granule.process(metrics)
        adjustments = self.purkinje.refine(filtered, update_state=False)
        self._balance_state = dict(adjustments)
        return {"adjustments": adjustments, "filtered": filtered}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _fine_tune_command(self, command: MotorCommand) -> MotorCommand:
        corrections = dict(self._latest_corrections)
        if not corrections and self._latest_error_signal:
            corrections = self.purkinje.refine(self._latest_error_signal)
            self._latest_corrections = dict(corrections)

        argument_updates = self._compute_argument_updates(command.arguments, corrections)
        metadata = {
            "cerebellum": {
                "training_samples": len(self.learned_signals),
                "metric_samples": len(self.metric_history),
                "filtered_error": dict(self._latest_error_signal),
                "applied_corrections": dict(corrections),
            }
        }
        return command.with_updates(arguments=argument_updates, metadata=metadata)

    def _compute_argument_updates(
        self, arguments: Mapping[str, Any], corrections: Mapping[str, float]
    ) -> Dict[str, Any]:
        updates: Dict[str, Any] = {}
        for key, value in arguments.items():
            correction_key = None
            if isinstance(value, Real):
                if key in corrections:
                    correction_key = key
                elif f"{key}_error" in corrections:
                    correction_key = f"{key}_error"
                elif f"{key}_delta" in corrections:
                    correction_key = f"{key}_delta"
                elif f"{key}_deviation" in corrections:
                    correction_key = f"{key}_deviation"

            if correction_key is None:
                continue

            correction = float(corrections[correction_key])
            if isinstance(value, Real):
                updates[key] = float(value) - correction
        return updates

    def _summarise_state(self) -> str:
        if not self.metric_history:
            return "cerebellum: no telemetry"
        latest = self.metric_history[-1]
        parts = [f"{key}:{latest[key]:.3f}" for key in sorted(latest)[:4]]
        return " | ".join(parts)

    def _normalise_feedback(self, feedback: str | Dict[str, Any] | MotorExecutionResult) -> Dict[str, Any]:
        if isinstance(feedback, str):
            telemetry = self._parse_structured_input(feedback)
            return {"signal": feedback, "label": feedback, "success": False, "telemetry": telemetry}
        if isinstance(feedback, dict):
            data = dict(feedback)
            data.setdefault("label", str(data.get("signal", "feedback")))
            return data
        return {
            "signal": feedback.error or "execution",
            "label": feedback.error or ("success" if feedback.success else "execution"),
            "success": feedback.success,
            "telemetry": dict(feedback.telemetry),
        }

    def _parse_structured_input(self, sensory_input: Mapping[str, Real] | str) -> Dict[str, float]:
        if isinstance(sensory_input, Mapping):
            return {
                key: float(value)
                for key, value in sensory_input.items()
                if isinstance(value, Real)
            }
        metrics: Dict[str, float] = {}
        tokens = sensory_input.replace(",", " ").split()
        for token in tokens:
            if ":" not in token:
                continue
            key, value = token.split(":", 1)
            try:
                metrics[key] = float(value)
            except ValueError:
                continue
        return metrics

    def update_feedback(self, metrics: Dict[str, Any]) -> None:
        """Integrate structured metric feedback for multi-channel learning."""

        numeric: Dict[str, float] = {}
        for key, value in metrics.items():
            if isinstance(value, Real):
                numeric[key] = float(value)
        if not numeric:
            return
        summary = {key: numeric[key] for key in sorted(numeric)}
        self.metric_history.append(summary)
        self.learned_signals.append({"label": "metrics", "metrics": summary, "success": summary.get("success_rate", 1.0) >= 0.5})


__all__ = ["Cerebellum", "GranuleNetwork", "PurkinjeNetwork"]
