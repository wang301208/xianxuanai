"""Service objects that manage background brain simulation runs."""
from __future__ import annotations

import math
import random
import threading
import time
from typing import Any, Dict, List, Optional

try:  # pragma: no cover - optional dependency
    import numpy as np
except ImportError:  # pragma: no cover - minimal environments without numpy
    np = None  # type: ignore


class SimulationOrchestrator:
    """Manage long-running simulation loops for the Brain API."""

    def __init__(self, controller: Any) -> None:
        self.controller = controller
        self.max_steps = 100
        self.step_interval = 0.5
        self.current_step = 0
        self.results: List[Any] = []
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self, *, steps: Optional[int] = None, interval: Optional[float] = None) -> Dict[str, Any]:
        """Begin executing the simulation loop in a background thread."""
        with self._lock:
            if self._running:
                raise RuntimeError("simulation already running")
            if steps is not None:
                self.max_steps = int(steps)
            if interval is not None:
                self.step_interval = float(interval)
            self.results.clear()
            self.current_step = 0
            self._running = True
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
        return {
            "status": "started",
            "max_steps": self.max_steps,
            "step_interval": self.step_interval,
        }

    def stop(self, *, timeout: float = 2.0) -> Dict[str, Any]:
        """Request the loop to stop and wait for the thread."""
        with self._lock:
            self._running = False
            thread = self._thread
            self._thread = None
        if thread and thread.is_alive():
            thread.join(timeout=timeout)
        return {
            "status": "stopped",
            "completed_steps": self.current_step,
            "results_count": len(self.results),
        }

    def status(self) -> Dict[str, Any]:
        """Return a snapshot of the current simulation status."""
        return {
            "running": self._running,
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "progress": (self.current_step / self.max_steps) * 100 if self.max_steps > 0 else 0,
            "results_count": len(self.results),
        }

    def paginated_results(self, page: int, page_size: int) -> Dict[str, Any]:
        """Return paginated simulation outputs."""
        start = page * page_size
        end = start + page_size
        results_subset = self.results[start:end] if start < len(self.results) else []
        return {
            "total": len(self.results),
            "page": page,
            "page_size": page_size,
            "results": results_subset,
        }

    def join(self, timeout: Optional[float] = None) -> None:
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)

    def _run_loop(self) -> None:
        try:
            for step in range(self.max_steps):
                with self._lock:
                    if not self._running:
                        break
                    self.current_step = step
                controller_input = {
                    "sensory_input": self._generate_sensory_input(step),
                    "neuromodulators": self._modulate_neuromodulators(step),
                    "task_goal": "simulation_task",
                    "decision_required": step % 10 == 0,
                    "response_required": step % 10 == 5,
                    "response_complete": step % 10 == 6,
                }
                result = self.controller.process(controller_input)
                self.results.append(result)
                time.sleep(self.step_interval)
        except Exception:
            # Keep the orchestrator in a consistent state regardless of controller errors.
            pass
        finally:
            with self._lock:
                self._running = False

    @staticmethod
    def _generate_sensory_input(step: int) -> Dict[str, Any]:
        sensory_input: Dict[str, Any] = {
            "visual_object_1": {"shape": "circle", "color": "red", "size": 0.8},
            "visual_object_2": {"shape": "square", "color": "blue", "size": 0.5},
            "auditory_input": {"frequency": 440, "volume": 0.7},
        }
        if step % 3 == 0:
            colors = ["red", "blue", "green", "yellow", "purple"]
            sensory_input["visual_object_1"]["color"] = colors[step % len(colors)]
        if step % 5 == 0:
            sensory_input[f"new_object_{step}"] = {
                "shape": "star",
                "color": "yellow",
                "size": 0.9,
                "novelty": 1.0,
            }
        return sensory_input

    @staticmethod
    def _modulate_neuromodulators(step: int) -> Dict[str, float]:
        base_level = 0.5
        trig_sin = math.sin
        ach_level = base_level + 0.3 * trig_sin(step / 10)
        dopa_level = base_level + (0.4 if step % 15 == 0 else 0.0)
        if np is not None:
            ne_noise = float(np.random.random())
            serotonin_level = base_level + 0.1 * np.sin(step / 20)
        else:
            ne_noise = random.random()
            serotonin_level = base_level + 0.1 * trig_sin(step / 20)
        ne_level = base_level + 0.2 * ne_noise
        clamp = lambda value: max(0.0, min(1.0, value))
        return {
            "acetylcholine": clamp(ach_level),
            "dopamine": clamp(dopa_level),
            "norepinephrine": clamp(ne_level),
            "serotonin": clamp(serotonin_level),
        }


__all__ = ["SimulationOrchestrator"]
