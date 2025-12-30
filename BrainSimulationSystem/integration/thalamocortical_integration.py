"""High-level utilities for wiring thalamocortical components together.

This module provides a small façade used by the BrainSimulationSystem tests.
It is intentionally lightweight and avoids requiring a full simulator runtime.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np

from BrainSimulationSystem.core.enhanced_cortical_column import EnhancedCorticalColumnWithLoop
from BrainSimulationSystem.core.enhanced_thalamocortical_loop import ThalamicNucleusType, ThalamocorticalLoop


@dataclass
class ThalamocorticalConfig:
    """Configuration bundle driving thalamocortical integration."""

    enabled: bool = True
    num_cortical_columns: int = 1
    oscillation_enabled: bool = True
    plasticity_enabled: bool = True
    loop_config: Optional[Dict[str, Any]] = None
    cortical_config: Optional[Dict[str, Any]] = None


class ThalamocorticalIntegration:
    """Facade exposing a ready-to-use thalamocortical processing pipeline."""

    def __init__(self, config: ThalamocorticalConfig | None = None):
        self.logger = logging.getLogger("ThalamocorticalIntegration")
        self.config = config or ThalamocorticalConfig()
        self.is_initialized = False
        self.thalamocortical_loop: ThalamocorticalLoop | None = None
        self.cortical_columns: Dict[int, EnhancedCorticalColumnWithLoop] = {}
        self._sensory_inputs: Dict[ThalamicNucleusType, np.ndarray] = {}
        self._step_count = 0
        self._last_update_time = 0.0
        self.current_time = 0.0
        self.performance_metrics: Dict[str, Any] = {"update_times": []}

        if not bool(getattr(self.config, "enabled", True)):
            return

        loop_cfg = dict(self.config.loop_config or {})
        self.thalamocortical_loop = ThalamocorticalLoop(loop_cfg)

        base_cortical_cfg = dict(self.config.cortical_config or {})
        base_cortical_cfg.setdefault("oscillation_enabled", bool(self.config.oscillation_enabled))
        base_cortical_cfg.setdefault("plasticity_enabled", bool(self.config.plasticity_enabled))
        base_cortical_cfg.setdefault("total_neurons", 500)

        num = max(1, int(self.config.num_cortical_columns or 1))
        for column_id in range(num):
            column = EnhancedCorticalColumnWithLoop(dict(base_cortical_cfg), self.thalamocortical_loop)
            self.thalamocortical_loop.add_cortical_column(column_id, column)
            self.cortical_columns[column_id] = column

        self.is_initialized = True

    def set_sensory_input(self, modality: str, input_data: Any) -> None:
        """Set sensory input for a modality (visual/somatosensory/auditory/etc)."""
        if not self.is_initialized or self.thalamocortical_loop is None:
            return
        modality = str(modality or "").strip().lower()
        mapping = {
            "visual": ThalamicNucleusType.LGN,
            "somatosensory": ThalamicNucleusType.VPL,
            "auditory": ThalamicNucleusType.MGN,
            "cognitive": ThalamicNucleusType.MD,
            "attention": ThalamicNucleusType.PULVINAR,
        }
        nucleus_type = mapping.get(modality)
        if nucleus_type is None:
            return
        arr = np.asarray(input_data, dtype=float).reshape(-1)
        self._sensory_inputs[nucleus_type] = arr
        self.thalamocortical_loop.set_sensory_input(nucleus_type, arr)

    def update_attention_focus(self, region: str, focus_level: float) -> None:
        if self.thalamocortical_loop is None:
            return
        try:
            self.thalamocortical_loop.update_attention_focus(str(region or "").strip().lower(), float(focus_level))
        except Exception:
            return

    def update_arousal_level(self, arousal_level: float) -> None:
        if self.thalamocortical_loop is None:
            return
        try:
            self.thalamocortical_loop.update_global_arousal(float(arousal_level))
        except Exception:
            return

    def simulate_sleep_transition(self, target_stage: int) -> None:
        if self.thalamocortical_loop is None:
            return
        try:
            self.thalamocortical_loop.simulate_sleep_transition(int(target_stage))
        except Exception:
            return

    def step(self, dt: float = 1.0) -> Dict[str, Any]:
        """Advance the integrated thalamocortical system by one step."""
        start = time.time()
        if not self.is_initialized or self.thalamocortical_loop is None:
            return {
                "thalamic_result": {},
                "cortical_results": {},
                "synchronization_indices": {},
                "total_spikes": 0,
                "update_time": 0.0,
            }

        for nucleus_type, arr in self._sensory_inputs.items():
            self.thalamocortical_loop.set_sensory_input(nucleus_type, arr)

        thalamic_inputs: Dict[ThalamicNucleusType, np.ndarray] = {}
        for nucleus_type, nucleus in self.thalamocortical_loop.thalamic_nuclei.items():
            sensory = getattr(nucleus, "sensory_input", None)
            if sensory is not None:
                thalamic_inputs[nucleus_type] = sensory

        cortical_results: Dict[int, Any] = {}
        for column_id, column in self.cortical_columns.items():
            try:
                column.process_thalamic_input(thalamic_inputs)
            except Exception:
                pass
            cortical_results[column_id] = column.step(dt)

        thalamic_result = self.thalamocortical_loop.step(dt)
        sync = self.thalamocortical_loop.get_synchronization_index()

        total_spikes = 0
        activity: Mapping[str, Any] = thalamic_result.get("thalamic_activity", {}) if isinstance(thalamic_result, Mapping) else {}
        for nucleus_payload in activity.values():
            if not isinstance(nucleus_payload, Mapping):
                continue
            total_spikes += len(nucleus_payload.get("relay_spikes", []) or [])
            total_spikes += len(nucleus_payload.get("interneuron_spikes", []) or [])

        self._step_count += 1
        self._last_update_time = time.time() - start
        self.current_time += float(dt)
        try:
            self.performance_metrics.setdefault("update_times", []).append(float(self._last_update_time))
        except Exception:
            pass

        return {
            "thalamic_result": thalamic_result,
            "cortical_results": cortical_results,
            "synchronization_indices": sync,
            "total_spikes": int(total_spikes),
            "update_time": float(self._last_update_time),
        }

    def get_system_state(self) -> Dict[str, Any]:
        """Return a small monitoring snapshot for tests/diagnostics."""
        global_arousal = None
        if self.thalamocortical_loop is not None:
            global_arousal = float(getattr(self.thalamocortical_loop, "global_arousal", 0.0) or 0.0)
        return {
            "initialized": bool(self.is_initialized),
            "num_cortical_columns": int(len(self.cortical_columns)),
            "current_time": float(self.current_time),
            "global_arousal": global_arousal,
            "performance": {
                "steps": int(self._step_count),
                "last_update_time": float(self._last_update_time),
            },
        }

    def reset(self) -> None:
        """Reset integration runtime state (for tests)."""
        self.current_time = 0.0
        self._step_count = 0
        self._last_update_time = 0.0
        self.performance_metrics = {"update_times": []}


__all__ = ["ThalamocorticalIntegration", "ThalamocorticalConfig"]
