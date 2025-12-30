"""
Motor control system combining cortical command generation and cerebellar coordination.

Provides optional integration with reinforcement-learning policies (Stable-Baselines3)
while offering heuristic fallbacks when such dependencies or trained models are not
available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import math
import numpy as np

try:  # pragma: no cover - optional UI automation support
    from BrainSimulationSystem.environment.ui_automation import (
        UIAutomationController,
        UIAutomationError,
    )
except Exception:  # pragma: no cover - keep motor control usable without UI layer
    UIAutomationController = None  # type: ignore[assignment]
    UIAutomationError = Exception  # type: ignore[misc,assignment]

try:  # pragma: no cover - optional dependency
    from stable_baselines3 import PPO
except Exception:  # pragma: no cover - RL framework might be absent
    PPO = None  # type: ignore[assignment]


class MotorControlUnavailable(RuntimeError):
    """Raised when a requested motor control backend cannot be used."""


@dataclass
class MotorControlConfig:
    """Configuration for the motor control system."""

    backend: str = "auto"  # auto | rl | heuristic
    action_dim: int = 3
    max_force: float = 1.0
    smoothing_factor: float = 0.25
    rl_model_path: Optional[str] = None
    heuristic_gain: float = 0.5
    ui: Optional[Dict[str, Any]] = None


class HeuristicMotorPolicy:
    """Simple rule-based motor policy used as a fall-back option."""

    def __init__(self, config: MotorControlConfig) -> None:
        self.config = config
        self._command_map: Dict[str, np.ndarray] = {
            "idle": np.zeros(self.config.action_dim, dtype=np.float32),
            "move_forward": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "move_backward": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            "turn_left": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "turn_right": np.array([0.0, 0.0, -1.0], dtype=np.float32),
            "lift": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "lower": np.array([0.0, -1.0, 0.0], dtype=np.float32),
        }

    def compute(self, intention: Any, feedback: Optional[Dict[str, Any]] = None) -> np.ndarray:
        feedback = feedback or {}

        vector = self._intent_to_vector(intention, feedback)
        norm = np.linalg.norm(vector)
        if norm > 0:
            scaled = (vector / norm) * min(norm, self.config.max_force)
        else:
            scaled = vector
        return scaled.astype(np.float32)

    def _intent_to_vector(self, intention: Any, feedback: Dict[str, Any]) -> np.ndarray:
        if isinstance(intention, np.ndarray):
            return intention.astype(np.float32)

        if isinstance(intention, Sequence) and not isinstance(intention, (str, bytes)):
            arr = np.asarray(intention, dtype=np.float32)
            if arr.size == self.config.action_dim:
                return arr
            return np.pad(arr, (0, max(0, self.config.action_dim - arr.size)))[: self.config.action_dim]

        if isinstance(intention, dict):
            target = intention.get("target_position")
            current = intention.get("current_position", np.zeros(self.config.action_dim, dtype=np.float32))
            if target is not None:
                target_vec = np.asarray(target, dtype=np.float32)
                current_vec = np.asarray(current, dtype=np.float32)
                diff = target_vec - current_vec
                gain = intention.get("gain", self.config.heuristic_gain)
                return diff * float(gain)
            if "command" in intention:
                return self._intent_to_vector(intention["command"], feedback)

        if isinstance(intention, str):
            return self._command_map.get(intention.lower(), np.zeros(self.config.action_dim, dtype=np.float32))

        # Default: use feedback error if available
        error = feedback.get("error_vector")
        if error is not None:
            return np.asarray(error, dtype=np.float32)

        return np.zeros(self.config.action_dim, dtype=np.float32)


class RLMotorPolicy:
    """Wrapper around a Stable-Baselines PPO policy."""

    def __init__(self, config: MotorControlConfig) -> None:
        if PPO is None:  # pragma: no cover - dependency missing
            raise MotorControlUnavailable("Stable-Baselines3 is required for the RL backend.")
        if not config.rl_model_path:  # pragma: no cover - requires trained model
            raise MotorControlUnavailable("rl_model_path must be provided for RL backend.")

        self._model = PPO.load(config.rl_model_path)
        self.config = config

    def compute(self, intention: Any, feedback: Optional[Dict[str, Any]] = None) -> np.ndarray:  # pragma: no cover - depends on external model
        obs = self._intention_to_obs(intention, feedback)
        action, _ = self._model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    def _intention_to_obs(self, intention: Any, feedback: Optional[Dict[str, Any]]) -> np.ndarray:
        vector = HeuristicMotorPolicy(self.config)._intent_to_vector(intention, feedback or {})
        return vector


class CerebellarCoordinator:
    """Applies smoothing/feedback corrections akin to cerebellar adjustments."""

    def __init__(self, smoothing_factor: float, action_dim: int) -> None:
        self.smoothing_factor = float(np.clip(smoothing_factor, 0.0, 1.0))
        self.last_command = np.zeros(action_dim, dtype=np.float32)

    def smooth(self, command: np.ndarray, feedback: Optional[Dict[str, Any]] = None) -> np.ndarray:
        feedback = feedback or {}
        error = np.asarray(feedback.get("error_vector", np.zeros_like(command)), dtype=np.float32)
        correction_gain = float(feedback.get("correction_gain", 0.1))

        corrected = command - correction_gain * error
        smoothed = (1 - self.smoothing_factor) * corrected + self.smoothing_factor * self.last_command
        self.last_command = smoothed.astype(np.float32)
        return self.last_command


class MotorControlSystem:
    """High level orchestrator for motor command generation and adaptation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = dict(config or {})
        self.config = MotorControlConfig(**config)

        backend = self.config.backend
        if backend == "auto":
            backend = "rl" if (PPO is not None and self.config.rl_model_path) else "heuristic"

        if backend == "rl":
            self.policy = RLMotorPolicy(self.config)
        else:
            self.policy = HeuristicMotorPolicy(self.config)
            backend = "heuristic"

        self.backend = backend
        self.cerebellum = CerebellarCoordinator(self.config.smoothing_factor, self.config.action_dim)

        self.ui_controller = None
        self.ui_status: Dict[str, Any] = {}
        ui_cfg = None
        if isinstance(self.config.ui, dict):
            ui_cfg = dict(self.config.ui)
        elif isinstance(config.get("ui"), dict):
            ui_cfg = dict(config.get("ui") or {})

        if ui_cfg is not None and UIAutomationController is not None:
            try:
                self.ui_controller = UIAutomationController(ui_cfg)
                self.ui_status = {
                    "backend": getattr(self.ui_controller, "backend_name", None),
                    "enabled": bool(getattr(self.ui_controller, "config", {}).enabled) if hasattr(self.ui_controller, "config") else None,
                    "dry_run": bool(getattr(self.ui_controller, "config", {}).dry_run) if hasattr(self.ui_controller, "config") else None,
                }
            except Exception as exc:  # pragma: no cover - defensive optional path
                self.ui_controller = None
                self.ui_status = {"error": str(exc)}

    def compute(
        self,
        intention: Any,
        feedback: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        feedback = feedback or {}

        raw_command = self.policy.compute(intention, feedback)
        clipped = np.clip(raw_command, -self.config.max_force, self.config.max_force)
        smoothed = self.cerebellum.smooth(clipped, feedback)

        energy = float(np.linalg.norm(smoothed))
        stability = 1.0 / (1.0 + energy)

        output: Dict[str, Any] = {
            "commands": smoothed.astype(np.float32).tolist(),
            "energy": energy,
            "stability": stability,
            "backend": self.backend,
            "raw_command": clipped.astype(np.float32).tolist(),
        }

        ui_result = self._maybe_execute_ui(intention)
        if ui_result is not None:
            output["ui"] = ui_result

        return output

    def _maybe_execute_ui(self, intention: Any) -> Optional[Dict[str, Any]]:
        """Optionally translate intentions into UI automation actions."""

        if self.ui_controller is None:
            return None

        actions: List[Dict[str, Any]] = []

        if isinstance(intention, dict):
            ui_payload = intention.get("ui")
            if isinstance(ui_payload, list):
                actions.extend([a for a in ui_payload if isinstance(a, dict)])
            elif isinstance(ui_payload, dict):
                if isinstance(ui_payload.get("actions"), list):
                    actions.extend([a for a in ui_payload.get("actions", []) if isinstance(a, dict)])
                else:
                    actions.append(dict(ui_payload))
            if isinstance(intention.get("ui_actions"), list):
                actions.extend([a for a in intention.get("ui_actions", []) if isinstance(a, dict)])
            if "ui_action" in intention and "action" not in intention:
                action_name = intention.get("ui_action")
                if isinstance(action_name, str):
                    payload = {k: v for k, v in intention.items() if k not in {"ui_action"}}
                    payload["action"] = action_name
                    actions.append(payload)

        if not actions and isinstance(intention, str):
            # Optional mapping for legacy motor commands -> UI actions.
            command_map: Any = {}
            if isinstance(self.config.ui, dict):
                command_map = self.config.ui.get("command_map", {})
            if isinstance(command_map, dict):
                mapped = command_map.get(intention) or command_map.get(intention.lower())
                if isinstance(mapped, dict):
                    actions.append(dict(mapped))
                elif isinstance(mapped, list):
                    actions.extend([a for a in mapped if isinstance(a, dict)])

        if not actions:
            return None

        try:
            results = self.ui_controller.execute_actions(actions)
            return {
                "backend": getattr(self.ui_controller, "backend_name", None),
                "results": results,
            }
        except UIAutomationError as exc:
            return {"error": str(exc), "backend": getattr(self.ui_controller, "backend_name", None)}
        except Exception as exc:  # pragma: no cover - defensive
            return {"error": str(exc), "backend": getattr(self.ui_controller, "backend_name", None)}
