"""Microcircuit abstractions for hierarchical hybrid modeling.

High-level cognition can remain modular (vision/memory/decision/etc.) while each
region/module optionally owns a *microcircuit* implementation that produces
neural dynamics and readouts (rates, spikes, state variables).

This keeps the cognitive layer decoupled from the simulation kernel: cognition
reads/writes microcircuit state through a small, explicit interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .enums import BrainRegion
import threading


@dataclass
class MicrocircuitReadout:
    activation: float
    rate_hz: float
    rate_hz_smooth: float
    region_rates_hz: Dict[str, float]
    state: Dict[str, Any]


class Microcircuit(ABC):
    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def step(
        self, dt_ms: float, inputs: Dict[str, float], neuromodulators: Dict[str, float]
    ) -> MicrocircuitReadout: ...

    def apply_control(self, control: Dict[str, Any]) -> Dict[str, Any]:
        """Apply optional top-down control updates.

        The cognitive layer may call this to adjust coarse operating points
        (e.g., input gain, target rate) without depending on backend-specific APIs.
        Microcircuit implementations may override this; the default is a no-op.
        """

        del control
        return {}


class ShadowMicrocircuit(Microcircuit):
    """Run a *primary* microcircuit plus a *shadow* microcircuit in lockstep.

    This is intended for **progressive replacement validation**: keep the system
    behaviour driven by the primary implementation, while executing a candidate
    implementation in parallel and attaching diff telemetry into the returned
    readout state.
    """

    def __init__(
        self,
        primary: Microcircuit,
        shadow: Microcircuit,
        *,
        compare: Optional[Dict[str, Any]] = None,
        label_primary: str = "primary",
        label_shadow: str = "shadow",
    ) -> None:
        self.primary = primary
        self.shadow = shadow
        self.compare_cfg = dict(compare or {}) if isinstance(compare, dict) else {}
        self.label_primary = str(label_primary or "primary")
        self.label_shadow = str(label_shadow or "shadow")

        self.requires_global_step = bool(
            getattr(primary, "requires_global_step", False)
            or getattr(shadow, "requires_global_step", False)
        )
        self.force_update_each_cycle = bool(
            getattr(primary, "force_update_each_cycle", False)
            or getattr(shadow, "force_update_each_cycle", False)
        )

        self._engines: list[Any] = []
        for micro in (primary, shadow):
            engine = getattr(micro, "engine", None)
            if engine is None:
                engine = getattr(micro, "_engine", None)
            if engine is not None:
                self._engines.append(engine)

        seen: set[int] = set()
        unique_engines: list[Any] = []
        for candidate in self._engines:
            key = id(candidate)
            if key in seen:
                continue
            seen.add(key)
            unique_engines.append(candidate)
        self._engines = unique_engines

        self._engine = self._engines[0] if self._engines else None

    @property
    def engine(self):
        return self._engine

    @property
    def engines(self):
        return list(self._engines)

    def prepare_step_inputs(self, dt_ms: float, inputs: Dict[str, float], neuromodulators: Dict[str, float]) -> None:
        for micro in (self.primary, self.shadow):
            prepare = getattr(micro, "prepare_step_inputs", None)
            if callable(prepare):
                prepare(dt_ms, inputs or {}, neuromodulators or {})

    def reset(self) -> None:
        for micro in (self.primary, self.shadow):
            try:
                micro.reset()
            except Exception:
                continue

    def apply_control(self, control: Dict[str, Any]) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        for label, micro in ((self.label_primary, self.primary), (self.label_shadow, self.shadow)):
            applier = getattr(micro, "apply_control", None)
            if not callable(applier):
                continue
            try:
                results[label] = applier(control or {})
            except Exception as exc:
                results[label] = {"error": str(exc)}
        return results

    def scale_synapses(self, factor: float, *, exc_only: bool = True, inh_only: bool = False) -> Dict[str, Any]:
        summaries: Dict[str, Any] = {}
        total_scaled = 0
        for label, micro in ((self.label_primary, self.primary), (self.label_shadow, self.shadow)):
            scaler = getattr(micro, "scale_synapses", None)
            if not callable(scaler):
                continue
            try:
                summary = scaler(float(factor), exc_only=bool(exc_only), inh_only=bool(inh_only)) or {}
            except Exception as exc:
                summary = {"error": str(exc)}
            summaries[label] = dict(summary) if isinstance(summary, dict) else {"state": summary}
            if isinstance(summary, dict):
                try:
                    total_scaled += int(summary.get("scaled", 0) or 0)
                except Exception:
                    pass

        return {"factor": float(factor), "scaled": int(total_scaled), "details": summaries}

    def step(self, dt_ms: float, inputs: Dict[str, float], neuromodulators: Dict[str, float]) -> MicrocircuitReadout:
        primary = self.primary.step(dt_ms, inputs or {}, neuromodulators or {})

        shadow_readout: Optional[MicrocircuitReadout] = None
        shadow_error: Optional[str] = None
        try:
            shadow_readout = self.shadow.step(dt_ms, inputs or {}, neuromodulators or {})
        except Exception as exc:
            shadow_error = str(exc)

        state = dict(primary.state) if isinstance(primary.state, dict) else {"state": primary.state}

        compare_payload: Dict[str, Any] = {
            "label_primary": self.label_primary,
            "label_shadow": self.label_shadow,
            "primary": {
                "activation": float(primary.activation),
                "rate_hz": float(primary.rate_hz),
                "rate_hz_smooth": float(primary.rate_hz_smooth),
                "region_rates_hz": dict(primary.region_rates_hz or {}),
                "framework": (primary.state or {}).get("framework") if isinstance(primary.state, dict) else None,
            },
            "shadow": None,
            "diff": None,
            "ok": False,
            "errors": {},
            "warnings": [],
        }

        if len(self._engines) > 1:
            compare_payload["warnings"].append("multiple_engines_detected")

        if shadow_error is not None:
            compare_payload["errors"][self.label_shadow] = shadow_error
        elif shadow_readout is not None:
            compare_payload["shadow"] = {
                "activation": float(shadow_readout.activation),
                "rate_hz": float(shadow_readout.rate_hz),
                "rate_hz_smooth": float(shadow_readout.rate_hz_smooth),
                "region_rates_hz": dict(shadow_readout.region_rates_hz or {}),
                "framework": (shadow_readout.state or {}).get("framework")
                if isinstance(shadow_readout.state, dict)
                else None,
            }

            region_keys = set(primary.region_rates_hz or {}).union(set(shadow_readout.region_rates_hz or {}))
            region_l1 = 0.0
            region_max = 0.0
            region_diff: Dict[str, float] = {}
            for key in sorted(region_keys):
                try:
                    primary_rate = float((primary.region_rates_hz or {}).get(key, 0.0) or 0.0)
                except Exception:
                    primary_rate = 0.0
                try:
                    shadow_rate = float((shadow_readout.region_rates_hz or {}).get(key, 0.0) or 0.0)
                except Exception:
                    shadow_rate = 0.0
                delta_rate = float(shadow_rate - primary_rate)
                region_diff[str(key)] = delta_rate
                region_l1 += abs(delta_rate)
                region_max = max(region_max, abs(delta_rate))

            diff = {
                "activation": float(shadow_readout.activation) - float(primary.activation),
                "rate_hz": float(shadow_readout.rate_hz) - float(primary.rate_hz),
                "rate_hz_smooth": float(shadow_readout.rate_hz_smooth) - float(primary.rate_hz_smooth),
                "region_rates_l1": float(region_l1),
                "region_rates_max": float(region_max),
                "region_rates": region_diff,
            }
            compare_payload["diff"] = diff

            violations: Dict[str, Any] = {}
            tol = dict(self.compare_cfg or {}) if isinstance(self.compare_cfg, dict) else {}
            for name, key in (
                ("activation_abs", "activation"),
                ("rate_hz_abs", "rate_hz"),
                ("rate_hz_smooth_abs", "rate_hz_smooth"),
                ("region_rates_l1_abs", "region_rates_l1"),
                ("region_rates_max_abs", "region_rates_max"),
            ):
                if name not in tol:
                    continue
                try:
                    limit = float(tol.get(name))
                except Exception:
                    continue
                if not np.isfinite(limit) or limit < 0.0:
                    continue
                try:
                    value = float(abs(diff.get(key, 0.0) or 0.0))
                except Exception:
                    value = 0.0
                if value > limit:
                    violations[name] = {"limit": float(limit), "value": float(value), "key": key}

            if violations:
                compare_payload["errors"]["tolerance"] = violations
            compare_payload["ok"] = not bool(compare_payload["errors"])

        state["shadow_compare"] = compare_payload

        return MicrocircuitReadout(
            activation=float(primary.activation),
            rate_hz=float(primary.rate_hz),
            rate_hz_smooth=float(primary.rate_hz_smooth),
            region_rates_hz=dict(primary.region_rates_hz or {}),
            state=state,
        )


def _coerce_finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return float(out)


def _apply_common_microcircuit_control(micro: Any, control: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(control, dict):
        return {}

    applied: Dict[str, Any] = {}

    try:
        if bool(control.get("reset", False)):
            reset = getattr(micro, "reset", None)
            if callable(reset):
                reset()
                applied["reset"] = True
    except Exception:
        pass

    def _set_number(attr: str, value: float) -> None:
        setattr(micro, attr, float(value))
        cfg = getattr(micro, "cfg", None)
        if isinstance(cfg, dict):
            cfg[attr.lstrip("_")] = float(value)

    input_gain = _coerce_finite_float(control.get("input_gain")) if "input_gain" in control else None
    input_scale = _coerce_finite_float(control.get("input_gain_scale")) if "input_gain_scale" in control else None
    if hasattr(micro, "_input_gain"):
        if input_gain is not None and input_gain > 0.0:
            _set_number("_input_gain", float(input_gain))
            applied["input_gain"] = float(input_gain)
        if input_scale is not None and input_scale > 0.0:
            try:
                current = float(getattr(micro, "_input_gain", 25.0) or 25.0)
            except Exception:
                current = 25.0
            updated = float(current) * float(input_scale)
            if np.isfinite(updated) and updated > 0.0:
                _set_number("_input_gain", float(updated))
                applied["input_gain"] = float(updated)
                applied["input_gain_scale"] = float(input_scale)

    target_rate = _coerce_finite_float(control.get("target_rate_hz")) if "target_rate_hz" in control else None
    target_scale = _coerce_finite_float(control.get("target_rate_scale")) if "target_rate_scale" in control else None
    if hasattr(micro, "_target_rate_hz"):
        if target_rate is not None and target_rate > 0.0:
            _set_number("_target_rate_hz", float(target_rate))
            applied["target_rate_hz"] = float(target_rate)
        if target_scale is not None and target_scale > 0.0:
            try:
                current = float(getattr(micro, "_target_rate_hz", 20.0) or 20.0)
            except Exception:
                current = 20.0
            updated = float(current) * float(target_scale)
            if np.isfinite(updated) and updated > 0.0:
                _set_number("_target_rate_hz", float(updated))
                applied["target_rate_hz"] = float(updated)
                applied["target_rate_scale"] = float(target_scale)

    smooth_tau = _coerce_finite_float(control.get("smooth_tau_ms")) if "smooth_tau_ms" in control else None
    if hasattr(micro, "_smooth_tau_ms") and smooth_tau is not None and smooth_tau > 0.0:
        _set_number("_smooth_tau_ms", float(smooth_tau))
        applied["smooth_tau_ms"] = float(smooth_tau)

    if hasattr(micro, "_dopamine_input_gain") and "dopamine_input_gain" in control:
        dopamine_gain = _coerce_finite_float(control.get("dopamine_input_gain"))
        if dopamine_gain is not None and dopamine_gain >= 0.0:
            try:
                setattr(micro, "_dopamine_input_gain", float(dopamine_gain))
                applied["dopamine_input_gain"] = float(dopamine_gain)
            except Exception:
                pass

    return applied


class BiophysicalMicrocircuit(Microcircuit):
    """Wrap the `biophysical` spiking backend as an embeddable microcircuit."""

    def __init__(self, *, params: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = dict(cfg or {})
        self.params = dict(params or {})
        self._network = None
        self._last_readout: Optional[MicrocircuitReadout] = None
        self._rate_smooth = 0.0

        # Lazy import to avoid pulling heavy backend dependencies in unrelated unit tests.
        from BrainSimulationSystem.core.network.biophysical import BiophysicalSpikingNetwork

        self._network = BiophysicalSpikingNetwork({}, params=self.params)

        self._input_gain = float(self.cfg.get("input_gain", 25.0))
        self._target_rate_hz = float(self.cfg.get("target_rate_hz", 20.0))
        self._smooth_tau_ms = float(self.cfg.get("smooth_tau_ms", 50.0))

        if not np.isfinite(self._input_gain):
            self._input_gain = 25.0
        if not np.isfinite(self._target_rate_hz) or self._target_rate_hz <= 0.0:
            self._target_rate_hz = 20.0
        if not np.isfinite(self._smooth_tau_ms) or self._smooth_tau_ms <= 0.0:
            self._smooth_tau_ms = 50.0

    @property
    def network(self):
        return self._network

    def reset(self) -> None:
        if self._network is not None and hasattr(self._network, "reset"):
            self._network.reset()
        self._last_readout = None
        self._rate_smooth = 0.0

    def apply_control(self, control: Dict[str, Any]) -> Dict[str, Any]:
        return _apply_common_microcircuit_control(self, control or {})

    def scale_synapses(self, factor: float, *, exc_only: bool = True, inh_only: bool = False) -> Dict[str, Any]:
        """Apply global synaptic scaling (e.g., sleep downscaling) if supported by the backend."""

        network = self._network
        if network is None:
            return {"scaled": 0, "factor": float(factor)}

        scaler = getattr(network, "scale_synaptic_weights", None)
        if callable(scaler):
            try:
                return dict(scaler(float(factor), exc_only=bool(exc_only), inh_only=bool(inh_only)) or {})
            except Exception:
                return {"scaled": 0, "factor": float(factor)}
        return {"scaled": 0, "factor": float(factor)}

    def _map_inputs_to_drive(self, inputs: Dict[str, float]) -> list[float]:
        assert self._network is not None

        total = 0.0
        if isinstance(inputs, dict):
            for _, value in inputs.items():
                try:
                    total += float(value)
                except Exception:
                    continue

        drive = float(total) * float(self._input_gain)
        if not np.isfinite(drive):
            drive = 0.0

        names = [str(r) for r in getattr(self._network, "regions", [])]
        if names:
            # If keys match region names, allow per-region drive.
            lowered = {str(k).strip().lower(): v for k, v in (inputs or {}).items()}
            per_region = []
            direct_hits = 0
            for name in names:
                key = str(name).strip().lower()
                if key in lowered:
                    try:
                        per_region.append(float(lowered[key]) * float(self._input_gain))
                        direct_hits += 1
                    except Exception:
                        per_region.append(drive)
                else:
                    per_region.append(drive)
            if direct_hits:
                return per_region
            return [drive] * int(getattr(self._network, "region_count", len(names)) or len(names))

        return [drive]

    def step(
        self, dt_ms: float, inputs: Dict[str, float], neuromodulators: Dict[str, float]
    ) -> MicrocircuitReadout:
        if self._network is None:
            raise RuntimeError("Microcircuit network is not initialized")

        dt_ms = float(dt_ms)
        if not np.isfinite(dt_ms) or dt_ms <= 0.0:
            raise ValueError("dt_ms must be positive")

        drive = self._map_inputs_to_drive(inputs or {})
        try:
            self._network.set_input(drive)
        except Exception:
            pass

        if isinstance(neuromodulators, dict) and neuromodulators:
            setter = getattr(self._network, "set_neuromodulators", None)
            if callable(setter):
                try:
                    setter(neuromodulators)
                except Exception:
                    pass

        state = self._network.step(dt_ms)

        spike_count = int(state.get("spike_count", len(state.get("spikes") or [])) or 0)
        n_neurons = int(getattr(self._network, "n_neurons", 0) or 0)
        dt_s = dt_ms / 1000.0
        rate_hz = float(spike_count) / max(float(n_neurons) * max(float(dt_s), 1e-9), 1e-9)

        alpha = float(np.clip(dt_ms / self._smooth_tau_ms, 0.0, 1.0))
        self._rate_smooth = (1.0 - alpha) * float(self._rate_smooth) + alpha * float(rate_hz)

        region_rates: Dict[str, float] = {}
        region_counts = state.get("region_spike_counts")
        if isinstance(region_counts, dict) and getattr(self._network, "region_count", 0):
            per_region = int(getattr(self._network, "bio").neurons_per_region) if hasattr(self._network, "bio") else 0
            for region, count in region_counts.items():
                region_rates[str(region)] = float(count or 0) / max(float(per_region) * max(dt_s, 1e-9), 1e-9)

        activation = 1.0 - float(np.exp(-float(self._rate_smooth) / float(self._target_rate_hz)))
        activation = float(np.clip(activation, 0.0, 1.0))

        readout = MicrocircuitReadout(
            activation=activation,
            rate_hz=float(rate_hz),
            rate_hz_smooth=float(self._rate_smooth),
            region_rates_hz=region_rates,
            state=dict(state) if isinstance(state, dict) else {"state": state},
        )
        self._last_readout = readout
        return readout


class _NestEngineUnavailable(RuntimeError):
    pass


class NestSimulationEngine:
    """Process-global NEST simulation engine (single kernel).

    NEST advances *all* nodes in a process when ``nest.Simulate()`` is called.
    To prevent accidental multi-stepping (e.g., one call per region), we
    centralize stepping here and keep ``NestMicrocircuit`` as a lightweight
    view that only queues inputs and consumes readouts.
    """

    _instance: Optional["NestSimulationEngine"] = None
    _instance_lock = threading.Lock()

    def __init__(self, *, resolution_ms: float = 0.1, threads: int = 1, print_time: bool = False) -> None:
        try:
            import nest  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise _NestEngineUnavailable(
                "PyNEST (nest) is not available. Install NEST Simulator + PyNEST to use the 'nest' microcircuit model."
            ) from exc

        self.nest = nest
        self.microcircuits: list["NestMicrocircuit"] = []

        # One-time kernel init for this process.
        nest.ResetKernel()
        try:
            nest.SetKernelStatus(
                {
                    "resolution": float(resolution_ms),
                    "local_num_threads": int(max(1, int(threads))),
                    "print_time": bool(print_time),
                    "overwrite_files": True,
                }
            )
        except Exception:
            pass

    @classmethod
    def get_or_create(
        cls, *, resolution_ms: float = 0.1, threads: int = 1, print_time: bool = False
    ) -> "NestSimulationEngine":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = NestSimulationEngine(
                    resolution_ms=float(resolution_ms),
                    threads=int(threads),
                    print_time=bool(print_time),
                )
            return cls._instance

    def register(self, microcircuit: "NestMicrocircuit") -> None:
        if microcircuit not in self.microcircuits:
            self.microcircuits.append(microcircuit)

    def step(self, dt_ms: float) -> Dict[str, Any]:
        dt_ms = float(dt_ms)
        if not np.isfinite(dt_ms) or dt_ms <= 0.0:
            raise ValueError("dt_ms must be positive")

        # Serialize engine stepping: even if regions update in Python threads,
        # NEST stepping must remain single-threaded at the Python API level.
        with self._instance_lock:
            for micro in list(self.microcircuits):
                try:
                    micro._apply_pending_drive()
                except Exception:
                    continue

            self.nest.Simulate(float(dt_ms))

            summaries: list[Dict[str, Any]] = []
            total_spikes = 0
            for micro in list(self.microcircuits):
                try:
                    info = micro._collect_after_step(float(dt_ms)) or {}
                except Exception:
                    info = {}
                summaries.append(dict(info) if isinstance(info, dict) else {"state": info})
                try:
                    total_spikes += int(info.get("spike_count", 0) or 0)
                except Exception:
                    pass

            try:
                time_ms = float(self.nest.GetKernelStatus("time"))
            except Exception:
                time_ms = float("nan")

        return {
            "framework": "nest",
            "time_ms": time_ms,
            "dt_ms": float(dt_ms),
            "microcircuits": summaries,
            "spike_count": int(total_spikes),
        }


class NestMicrocircuit(Microcircuit):
    """NEST-backed spiking microcircuit (stepped by ``NestSimulationEngine``)."""

    # Duck-typed by the cognitive scheduler to run global engines before per-region updates.
    requires_global_step = True
    # Ensure event-driven mode still updates physiology/activation for NEST regions.
    force_update_each_cycle = True

    def __init__(self, *, params: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = dict(cfg or {})
        self.params = dict(params or {})
        self._last_readout: Optional[MicrocircuitReadout] = None
        self._rate_smooth = 0.0

        # Readout parameters (kept consistent with BiophysicalMicrocircuit).
        self._input_gain = float(self.cfg.get("input_gain", 25.0))
        self._target_rate_hz = float(self.cfg.get("target_rate_hz", 20.0))
        self._smooth_tau_ms = float(self.cfg.get("smooth_tau_ms", 50.0))

        if not np.isfinite(self._input_gain):
            self._input_gain = 25.0
        if not np.isfinite(self._target_rate_hz) or self._target_rate_hz <= 0.0:
            self._target_rate_hz = 20.0
        if not np.isfinite(self._smooth_tau_ms) or self._smooth_tau_ms <= 0.0:
            self._smooth_tau_ms = 50.0

        nest_cfg = self.cfg.get("nest", {})
        if not isinstance(nest_cfg, dict):
            nest_cfg = {}
        resolution_ms = float(nest_cfg.get("resolution_ms", 0.1))
        threads = int(nest_cfg.get("threads", 1))
        print_time = bool(nest_cfg.get("print_time", False))

        self._engine = NestSimulationEngine.get_or_create(
            resolution_ms=resolution_ms, threads=threads, print_time=print_time
        )
        self.nest = self._engine.nest

        self._regions: list[str] = [str(r) for r in (self.params.get("regions") or [])]
        if not self._regions:
            self._regions = ["REGION"]

        try:
            self._neurons_per_region = int(self.params.get("neurons_per_region", 80))
        except Exception:
            self._neurons_per_region = 80
        if self._neurons_per_region <= 0:
            self._neurons_per_region = 80

        try:
            self._excitatory_ratio = float(self.params.get("excitatory_ratio", 0.8))
        except Exception:
            self._excitatory_ratio = 0.8
        self._excitatory_ratio = float(np.clip(self._excitatory_ratio, 0.0, 1.0))

        self._neuron_model = str(self.params.get("neuron_model", "lif") or "lif").strip().lower()
        self._stdp_enabled = bool(self.params.get("stdp_enabled", False))

        self._connectome_weights = np.asarray(self.params.get("connectome_weights", []), dtype=np.float32)
        self._connectome_delays_ms = np.asarray(self.params.get("connectome_delays_ms", []), dtype=np.float32)

        self._pop_exc: list[Any] = []
        self._pop_inh: list[Any] = []
        self._input_generators: list[Any] = []
        self._spike_recorders: list[Any] = []
        self._last_event_counts: list[int] = []
        self._pending_drive: list[float] = [0.0 for _ in self._regions]

        self._build_network()
        self.reset()
        self._engine.register(self)

    @property
    def engine(self) -> NestSimulationEngine:
        return self._engine

    def reset(self) -> None:
        self._last_readout = None
        self._rate_smooth = 0.0
        self._pending_drive = [0.0 for _ in self._regions]
        self._last_event_counts = [0 for _ in self._spike_recorders]

        for idx, rec in enumerate(self._spike_recorders):
            try:
                events = self.nest.GetStatus(rec, "events")[0]
                self._last_event_counts[idx] = int(len(events.get("times", [])))
            except Exception:
                self._last_event_counts[idx] = 0

    def apply_control(self, control: Dict[str, Any]) -> Dict[str, Any]:
        return _apply_common_microcircuit_control(self, control or {})

    def prepare_step_inputs(self, dt_ms: float, inputs: Dict[str, float], neuromodulators: Dict[str, float]) -> None:
        del dt_ms
        del neuromodulators
        self._pending_drive = self._map_inputs_to_drive(inputs or {})

    def _apply_pending_drive(self) -> None:
        for idx, gen in enumerate(self._input_generators):
            amplitude = float(self._pending_drive[idx]) if idx < len(self._pending_drive) else 0.0
            if not np.isfinite(amplitude):
                amplitude = 0.0
            try:
                self.nest.SetStatus(gen, {"amplitude": float(amplitude)})
            except Exception:
                continue

    def _collect_after_step(self, dt_ms: float) -> Dict[str, Any]:
        dt_s = float(dt_ms) / 1000.0
        spike_count_total = 0
        region_counts: Dict[str, int] = {}

        for ridx, rec in enumerate(self._spike_recorders):
            try:
                events = self.nest.GetStatus(rec, "events")[0]
                count = int(len(events.get("times", [])))
            except Exception:
                count = 0
            last = int(self._last_event_counts[ridx]) if ridx < len(self._last_event_counts) else 0
            new_count = max(0, count - last)
            self._last_event_counts[ridx] = count
            spike_count_total += new_count
            region_counts[self._regions[ridx]] = new_count

        region_rates: Dict[str, float] = {}
        for ridx, name in enumerate(self._regions):
            n_neurons = self._neurons_per_region
            region_rates[name] = float(region_counts.get(name, 0)) / max(float(n_neurons) * max(dt_s, 1e-9), 1e-9)

        total_neurons = int(self._neurons_per_region) * int(len(self._regions))
        rate_hz = float(spike_count_total) / max(float(total_neurons) * max(float(dt_s), 1e-9), 1e-9)

        alpha = float(np.clip(float(dt_ms) / float(self._smooth_tau_ms), 0.0, 1.0))
        self._rate_smooth = (1.0 - alpha) * float(self._rate_smooth) + alpha * float(rate_hz)

        activation = 1.0 - float(np.exp(-float(self._rate_smooth) / float(self._target_rate_hz)))
        activation = float(np.clip(activation, 0.0, 1.0))

        readout = MicrocircuitReadout(
            activation=activation,
            rate_hz=float(rate_hz),
            rate_hz_smooth=float(self._rate_smooth),
            region_rates_hz=region_rates,
            state={
                "framework": "nest",
                "spike_count": int(spike_count_total),
                "region_spike_counts": dict(region_counts),
            },
        )
        self._last_readout = readout
        return readout.state

    def step(self, dt_ms: float, inputs: Dict[str, float], neuromodulators: Dict[str, float]) -> MicrocircuitReadout:
        # The cognitive scheduler should call `prepare_step_inputs()` for all NEST microcircuits,
        # then step the engine *once* per cognitive cycle. For standalone usage you can set
        # `cfg.external_step=False` to let this microcircuit drive the engine.
        self.prepare_step_inputs(dt_ms, inputs or {}, neuromodulators or {})
        if not bool(self.cfg.get("external_step", True)):
            self._engine.step(float(dt_ms))

        if self._last_readout is None:
            return MicrocircuitReadout(
                activation=0.0,
                rate_hz=0.0,
                rate_hz_smooth=float(self._rate_smooth),
                region_rates_hz={name: 0.0 for name in self._regions},
                state={"framework": "nest", "spike_count": 0, "region_spike_counts": {}},
            )
        return self._last_readout

    def _map_inputs_to_drive(self, inputs: Dict[str, float]) -> list[float]:
        total = 0.0
        if isinstance(inputs, dict):
            for _, value in inputs.items():
                try:
                    total += float(value)
                except Exception:
                    continue

        drive = float(total) * float(self._input_gain)
        if not np.isfinite(drive):
            drive = 0.0

        lowered = {str(k).strip().lower(): v for k, v in (inputs or {}).items()}
        per_region: list[float] = []
        direct_hits = 0
        for name in self._regions:
            key = str(name).strip().lower()
            if key in lowered:
                try:
                    per_region.append(float(lowered[key]) * float(self._input_gain))
                    direct_hits += 1
                except Exception:
                    per_region.append(drive)
            else:
                per_region.append(drive)

        if direct_hits:
            return per_region
        return [drive for _ in self._regions]

    @staticmethod
    def _nest_neuron_model_for(kind: str) -> str:
        kind = str(kind or "lif").strip().lower()
        mapping = {
            "lif": "iaf_psc_alpha",
            "iaf": "iaf_psc_alpha",
            "hh": "hh_psc_alpha",
            "adex": "aeif_cond_exp",
            "aeif": "aeif_cond_exp",
            "izh": "izhikevich",
            "izhikevich": "izhikevich",
        }
        return mapping.get(kind, "iaf_psc_alpha")

    def _build_network(self) -> None:
        nest = self.nest

        model_exc = self._nest_neuron_model_for("adex" if self._neuron_model == "hybrid" else self._neuron_model)
        model_inh = self._nest_neuron_model_for("izhikevich" if self._neuron_model == "hybrid" else self._neuron_model)

        n_total = int(self._neurons_per_region)
        n_exc = int(round(float(n_total) * float(self._excitatory_ratio)))
        n_exc = int(np.clip(n_exc, 1, max(1, n_total)))
        n_inh = max(0, int(n_total) - int(n_exc))

        for _ in self._regions:
            exc = nest.Create(model_exc, n_exc)
            inh = nest.Create(model_inh, n_inh) if n_inh > 0 else None
            self._pop_exc.append(exc)
            self._pop_inh.append(inh)

            gen = nest.Create("dc_generator", 1, {"amplitude": 0.0})
            self._input_generators.append(gen)
            try:
                nest.Connect(gen, exc)
            except Exception:
                pass
            if inh is not None:
                try:
                    nest.Connect(gen, inh)
                except Exception:
                    pass

            rec = nest.Create("spike_recorder")
            self._spike_recorders.append(rec)
            self._last_event_counts.append(0)
            try:
                nest.Connect(exc, rec)
            except Exception:
                pass
            if inh is not None:
                try:
                    nest.Connect(inh, rec)
                except Exception:
                    pass

        intra_p = float(self.params.get("intra_connection_prob", 0.05))
        inter_p = float(self.params.get("inter_connection_prob", 0.02))
        if not np.isfinite(intra_p) or intra_p < 0.0:
            intra_p = 0.0
        if not np.isfinite(inter_p) or inter_p < 0.0:
            inter_p = 0.0
        intra_p = float(np.clip(intra_p, 0.0, 1.0))
        inter_p = float(np.clip(inter_p, 0.0, 1.0))

        base_delay = float(self.params.get("synaptic_delay_ms", 1.0))
        if not np.isfinite(base_delay) or base_delay <= 0.0:
            base_delay = 1.0

        def _syn_spec(weight: float, delay: float, *, stdp: bool) -> Dict[str, Any]:
            spec: Dict[str, Any] = {"weight": float(weight), "delay": float(delay)}
            if stdp:
                spec["model"] = "stdp_synapse"
            return spec

        for ridx in range(len(self._regions)):
            exc = self._pop_exc[ridx]
            inh = self._pop_inh[ridx]

            if intra_p > 0.0:
                try:
                    nest.Connect(
                        exc,
                        exc,
                        {"rule": "pairwise_bernoulli", "p": intra_p},
                        _syn_spec(weight=1.0, delay=base_delay, stdp=bool(self._stdp_enabled)),
                    )
                except Exception:
                    pass

            if inh is not None and intra_p > 0.0:
                try:
                    nest.Connect(
                        inh,
                        exc,
                        {"rule": "pairwise_bernoulli", "p": intra_p},
                        _syn_spec(weight=-1.0, delay=base_delay, stdp=False),
                    )
                except Exception:
                    pass

        if self._connectome_weights.ndim == 2 and self._connectome_weights.shape[0] == len(self._regions):
            for src in range(len(self._regions)):
                for dst in range(len(self._regions)):
                    if src == dst:
                        continue
                    w = float(self._connectome_weights[src, dst])
                    if abs(w) < 1e-9:
                        continue

                    delay = base_delay
                    if (
                        self._connectome_delays_ms.ndim == 2
                        and self._connectome_delays_ms.shape == self._connectome_weights.shape
                    ):
                        try:
                            delay = float(self._connectome_delays_ms[src, dst])
                        except Exception:
                            delay = base_delay
                    if not np.isfinite(delay) or delay <= 0.0:
                        delay = base_delay

                    if w >= 0.0:
                        source = self._pop_exc[src]
                        spec = _syn_spec(weight=w, delay=delay, stdp=bool(self._stdp_enabled))
                    else:
                        source = self._pop_inh[src]
                        spec = _syn_spec(weight=w, delay=delay, stdp=False)
                        if source is None:
                            continue

                    target_exc = self._pop_exc[dst]
                    target_inh = self._pop_inh[dst]

                    if inter_p > 0.0:
                        try:
                            nest.Connect(
                                source,
                                target_exc,
                                {"rule": "pairwise_bernoulli", "p": inter_p},
                                spec,
                            )
                        except Exception:
                            pass
                        if target_inh is not None:
                            try:
                                nest.Connect(
                                    source,
                                    target_inh,
                                    {"rule": "pairwise_bernoulli", "p": inter_p},
                                    spec,
                                )
                            except Exception:
                                    pass


class _Brian2Unavailable(RuntimeError):
    pass


class _SpiNNakerUnavailable(RuntimeError):
    pass


class SpiNNakerSimulationEngine:
    """Process-global sPyNNaker/PyNN simulation engine (single kernel).

    Like NEST, sPyNNaker advances the whole simulation when ``sim.run()`` is
    called. To avoid accidental multi-stepping (e.g., once per region), we
    centralize stepping here and keep ``SpiNNakerMicrocircuit`` as a view that
    queues inputs and consumes readouts.
    """

    _instance: Optional["SpiNNakerSimulationEngine"] = None
    _instance_lock = threading.Lock()

    def __init__(
        self,
        *,
        timestep_ms: float = 0.1,
        min_delay_ms: float = 0.1,
        max_delay_ms: float = 10.0,
    ) -> None:
        try:
            import spynnaker8 as sim  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise _SpiNNakerUnavailable(
                "spynnaker8 is not available. Install sPyNNaker/spynnaker8 to use the 'spinnaker' microcircuit model."
            ) from exc

        self.sim = sim
        self.microcircuits: list["SpiNNakerMicrocircuit"] = []

        # One-time init for this process.
        timestep_ms = float(timestep_ms)
        min_delay_ms = float(min_delay_ms)
        max_delay_ms = float(max_delay_ms)
        if not np.isfinite(timestep_ms) or timestep_ms <= 0.0:
            timestep_ms = 0.1
        if not np.isfinite(min_delay_ms) or min_delay_ms <= 0.0:
            min_delay_ms = timestep_ms
        if not np.isfinite(max_delay_ms) or max_delay_ms <= 0.0:
            max_delay_ms = max(10.0, float(min_delay_ms))
        if max_delay_ms < min_delay_ms:
            max_delay_ms = float(min_delay_ms)

        try:
            self.sim.setup(timestep=float(timestep_ms), min_delay=float(min_delay_ms), max_delay=float(max_delay_ms))
        except TypeError:  # pragma: no cover - depends on sPyNNaker version
            self.sim.setup(timestep=float(timestep_ms))

    @classmethod
    def get_or_create(
        cls,
        *,
        timestep_ms: float = 0.1,
        min_delay_ms: float = 0.1,
        max_delay_ms: float = 10.0,
    ) -> "SpiNNakerSimulationEngine":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = SpiNNakerSimulationEngine(
                    timestep_ms=float(timestep_ms),
                    min_delay_ms=float(min_delay_ms),
                    max_delay_ms=float(max_delay_ms),
                )
            return cls._instance

    def register(self, microcircuit: "SpiNNakerMicrocircuit") -> None:
        if microcircuit not in self.microcircuits:
            self.microcircuits.append(microcircuit)

    def step(self, dt_ms: float) -> Dict[str, Any]:
        dt_ms = float(dt_ms)
        if not np.isfinite(dt_ms) or dt_ms <= 0.0:
            raise ValueError("dt_ms must be positive")

        with self._instance_lock:
            for micro in list(self.microcircuits):
                try:
                    micro._apply_pending_drive()
                except Exception:
                    continue

            self.sim.run(float(dt_ms))

            summaries: list[Dict[str, Any]] = []
            total_spikes = 0
            for micro in list(self.microcircuits):
                try:
                    info = micro._collect_after_step(float(dt_ms)) or {}
                except Exception:
                    info = {}
                summaries.append(dict(info) if isinstance(info, dict) else {"state": info})
                try:
                    total_spikes += int(info.get("spike_count", 0) or 0)
                except Exception:
                    pass

            try:
                time_ms = float(self.sim.get_current_time())
            except Exception:
                time_ms = float("nan")

        return {
            "framework": "spinnaker",
            "time_ms": time_ms,
            "dt_ms": float(dt_ms),
            "microcircuits": summaries,
            "spike_count": int(total_spikes),
        }


class SpiNNakerMicrocircuit(Microcircuit):
    """sPyNNaker/PyNN-backed spiking microcircuit (stepped by ``SpiNNakerSimulationEngine``)."""

    requires_global_step = True
    force_update_each_cycle = True

    def __init__(self, *, params: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = dict(cfg or {})
        self.params = dict(params or {})
        self._last_readout: Optional[MicrocircuitReadout] = None
        self._rate_smooth = 0.0

        self._input_gain = float(self.cfg.get("input_gain", 25.0))
        self._target_rate_hz = float(self.cfg.get("target_rate_hz", 20.0))
        self._smooth_tau_ms = float(self.cfg.get("smooth_tau_ms", 50.0))

        if not np.isfinite(self._input_gain):
            self._input_gain = 25.0
        if not np.isfinite(self._target_rate_hz) or self._target_rate_hz <= 0.0:
            self._target_rate_hz = 20.0
        if not np.isfinite(self._smooth_tau_ms) or self._smooth_tau_ms <= 0.0:
            self._smooth_tau_ms = 50.0

        spinnaker_cfg = self.cfg.get("spinnaker", {})
        if not isinstance(spinnaker_cfg, dict):
            spinnaker_cfg = {}

        try:
            timestep_ms = float(spinnaker_cfg.get("timestep_ms", 0.1))
        except Exception:
            timestep_ms = 0.1
        try:
            min_delay_ms = float(spinnaker_cfg.get("min_delay_ms", 0.1))
        except Exception:
            min_delay_ms = 0.1
        try:
            max_delay_ms = float(spinnaker_cfg.get("max_delay_ms", 10.0))
        except Exception:
            max_delay_ms = 10.0

        self._engine = SpiNNakerSimulationEngine.get_or_create(
            timestep_ms=timestep_ms, min_delay_ms=min_delay_ms, max_delay_ms=max_delay_ms
        )
        self.sim = self._engine.sim

        self._regions: list[str] = [str(r) for r in (self.params.get("regions") or [])]
        if not self._regions:
            self._regions = ["REGION"]

        try:
            self._neurons_per_region = int(self.params.get("neurons_per_region", 80))
        except Exception:
            self._neurons_per_region = 80
        if self._neurons_per_region <= 0:
            self._neurons_per_region = 80

        try:
            self._excitatory_ratio = float(self.params.get("excitatory_ratio", 0.8))
        except Exception:
            self._excitatory_ratio = 0.8
        self._excitatory_ratio = float(np.clip(self._excitatory_ratio, 0.0, 1.0))

        self._neuron_model = str(self.params.get("neuron_model", "lif") or "lif").strip().lower()

        self._connectome_weights = np.asarray(self.params.get("connectome_weights", []), dtype=np.float32)
        self._connectome_delays_ms = np.asarray(self.params.get("connectome_delays_ms", []), dtype=np.float32)

        self._pop_exc: list[Any] = []
        self._pop_inh: list[Any] = []
        self._projections: list[Any] = []
        self._last_spike_totals: list[int] = []
        self._pending_drive: list[float] = [0.0 for _ in self._regions]

        self._build_network()
        self.reset()
        self._engine.register(self)

    @property
    def engine(self) -> SpiNNakerSimulationEngine:
        return self._engine

    def reset(self) -> None:
        self._last_readout = None
        self._rate_smooth = 0.0
        self._pending_drive = [0.0 for _ in self._regions]
        self._last_spike_totals = [0 for _ in self._regions]

        for ridx in range(len(self._regions)):
            total = 0
            try:
                total += self._population_spike_total(self._pop_exc[ridx])
            except Exception:
                pass
            inh = self._pop_inh[ridx] if ridx < len(self._pop_inh) else None
            if inh is not None:
                try:
                    total += self._population_spike_total(inh)
                except Exception:
                    pass
            self._last_spike_totals[ridx] = int(total)

    def apply_control(self, control: Dict[str, Any]) -> Dict[str, Any]:
        return _apply_common_microcircuit_control(self, control or {})

    def prepare_step_inputs(self, dt_ms: float, inputs: Dict[str, float], neuromodulators: Dict[str, float]) -> None:
        del dt_ms
        del neuromodulators
        self._pending_drive = self._map_inputs_to_drive(inputs or {})

    def _apply_pending_drive(self) -> None:
        for ridx in range(len(self._regions)):
            amplitude = float(self._pending_drive[ridx]) if ridx < len(self._pending_drive) else 0.0
            if not np.isfinite(amplitude):
                amplitude = 0.0

            exc = self._pop_exc[ridx]
            try:
                exc.set(i_offset=float(amplitude))
            except Exception:
                pass

            inh = self._pop_inh[ridx] if ridx < len(self._pop_inh) else None
            if inh is not None:
                try:
                    inh.set(i_offset=float(amplitude))
                except Exception:
                    pass

    @staticmethod
    def _population_spike_total(population: Any) -> int:
        try:
            counts = population.get_spike_counts()
        except Exception:
            counts = None
        if isinstance(counts, dict):
            total = 0
            for _, value in counts.items():
                try:
                    total += int(value)
                except Exception:
                    continue
            return int(total)

        try:
            data = population.get_data("spikes")
        except Exception:
            return 0

        total = 0
        try:
            segments = getattr(data, "segments", []) or []
        except Exception:
            segments = []
        for seg in segments:
            try:
                trains = getattr(seg, "spiketrains", []) or []
            except Exception:
                trains = []
            for st in trains:
                try:
                    total += int(len(st))
                except Exception:
                    continue
        return int(total)

    def _collect_after_step(self, dt_ms: float) -> Dict[str, Any]:
        dt_s = float(dt_ms) / 1000.0
        spike_count_total = 0
        region_counts: Dict[str, int] = {}

        for ridx, name in enumerate(self._regions):
            total = 0
            try:
                total += self._population_spike_total(self._pop_exc[ridx])
            except Exception:
                pass
            inh = self._pop_inh[ridx] if ridx < len(self._pop_inh) else None
            if inh is not None:
                try:
                    total += self._population_spike_total(inh)
                except Exception:
                    pass

            last = int(self._last_spike_totals[ridx]) if ridx < len(self._last_spike_totals) else 0
            new_count = max(0, int(total) - int(last))
            if ridx < len(self._last_spike_totals):
                self._last_spike_totals[ridx] = int(total)

            region_counts[str(name)] = int(new_count)
            spike_count_total += int(new_count)

        region_rates: Dict[str, float] = {}
        for ridx, name in enumerate(self._regions):
            n_neurons = int(self._neurons_per_region)
            region_rates[str(name)] = float(region_counts.get(str(name), 0)) / max(
                float(n_neurons) * max(dt_s, 1e-9), 1e-9
            )

        total_neurons = int(self._neurons_per_region) * int(len(self._regions))
        rate_hz = float(spike_count_total) / max(float(total_neurons) * max(float(dt_s), 1e-9), 1e-9)

        alpha = float(np.clip(float(dt_ms) / float(self._smooth_tau_ms), 0.0, 1.0))
        self._rate_smooth = (1.0 - alpha) * float(self._rate_smooth) + alpha * float(rate_hz)

        activation = 1.0 - float(np.exp(-float(self._rate_smooth) / float(self._target_rate_hz)))
        activation = float(np.clip(activation, 0.0, 1.0))

        readout = MicrocircuitReadout(
            activation=activation,
            rate_hz=float(rate_hz),
            rate_hz_smooth=float(self._rate_smooth),
            region_rates_hz=region_rates,
            state={
                "framework": "spinnaker",
                "spike_count": int(spike_count_total),
                "region_spike_counts": dict(region_counts),
            },
        )
        self._last_readout = readout
        return readout.state

    def step(self, dt_ms: float, inputs: Dict[str, float], neuromodulators: Dict[str, float]) -> MicrocircuitReadout:
        # The cognitive scheduler should call `prepare_step_inputs()` for all sPyNNaker microcircuits,
        # then step the engine *once* per cognitive cycle. For standalone usage you can set
        # `cfg.external_step=False` to let this microcircuit drive the engine.
        self.prepare_step_inputs(dt_ms, inputs or {}, neuromodulators or {})
        if not bool(self.cfg.get("external_step", True)):
            self._engine.step(float(dt_ms))

        if self._last_readout is None:
            return MicrocircuitReadout(
                activation=0.0,
                rate_hz=0.0,
                rate_hz_smooth=float(self._rate_smooth),
                region_rates_hz={name: 0.0 for name in self._regions},
                state={"framework": "spinnaker", "spike_count": 0, "region_spike_counts": {}},
            )
        return self._last_readout

    def _map_inputs_to_drive(self, inputs: Dict[str, float]) -> list[float]:
        total = 0.0
        if isinstance(inputs, dict):
            for _, value in inputs.items():
                try:
                    total += float(value)
                except Exception:
                    continue

        drive = float(total) * float(self._input_gain)
        if not np.isfinite(drive):
            drive = 0.0

        lowered = {str(k).strip().lower(): v for k, v in (inputs or {}).items()}
        per_region: list[float] = []
        direct_hits = 0
        for name in self._regions:
            key = str(name).strip().lower()
            if key in lowered:
                try:
                    per_region.append(float(lowered[key]) * float(self._input_gain))
                    direct_hits += 1
                except Exception:
                    per_region.append(drive)
            else:
                per_region.append(drive)

        if direct_hits:
            return per_region
        return [drive for _ in self._regions]

    def _pynn_neuron_model_for(self, kind: str) -> Any:
        kind = str(kind or "lif").strip().lower()
        if kind in {"izh", "izhikevich"}:
            try:
                return self.sim.Izhikevich()
            except Exception:
                return self.sim.IF_curr_exp()
        return self.sim.IF_curr_exp()

    def _build_network(self) -> None:
        sim = self.sim

        model_exc = self._pynn_neuron_model_for("lif" if self._neuron_model == "hybrid" else self._neuron_model)
        model_inh = self._pynn_neuron_model_for("izhikevich" if self._neuron_model == "hybrid" else self._neuron_model)

        n_total = int(self._neurons_per_region)
        n_exc = int(round(float(n_total) * float(self._excitatory_ratio)))
        n_exc = int(np.clip(n_exc, 1, max(1, n_total)))
        n_inh = max(0, int(n_total) - int(n_exc))

        for name in self._regions:
            exc = sim.Population(n_exc, model_exc, label=f"{name}_exc")
            inh = sim.Population(n_inh, model_inh, label=f"{name}_inh") if n_inh > 0 else None
            self._pop_exc.append(exc)
            self._pop_inh.append(inh)

            try:
                exc.record(["spikes"])
            except Exception:
                pass
            if inh is not None:
                try:
                    inh.record(["spikes"])
                except Exception:
                    pass

        try:
            intra_p = float(self.params.get("intra_connection_prob", 0.05))
        except Exception:
            intra_p = 0.05
        try:
            inter_p = float(self.params.get("inter_connection_prob", 0.02))
        except Exception:
            inter_p = 0.02
        if not np.isfinite(intra_p) or intra_p < 0.0:
            intra_p = 0.0
        if not np.isfinite(inter_p) or inter_p < 0.0:
            inter_p = 0.0
        intra_p = float(np.clip(intra_p, 0.0, 1.0))
        inter_p = float(np.clip(inter_p, 0.0, 1.0))

        try:
            base_delay = float(self.params.get("synaptic_delay_ms", 1.0))
        except Exception:
            base_delay = 1.0
        if not np.isfinite(base_delay) or base_delay <= 0.0:
            base_delay = 1.0

        try:
            w_exc = float(self.params.get("w_exc", 1.0))
        except Exception:
            w_exc = 1.0
        try:
            w_inh = float(self.params.get("w_inh", 1.0))
        except Exception:
            w_inh = 1.0
        if not np.isfinite(w_exc):
            w_exc = 1.0
        if not np.isfinite(w_inh):
            w_inh = 1.0

        # Local intra-(sub)region connections.
        for ridx in range(len(self._regions)):
            exc = self._pop_exc[ridx]
            inh = self._pop_inh[ridx]

            if intra_p > 0.0:
                conn = sim.FixedProbabilityConnector(p_connect=float(intra_p))
                try:
                    self._projections.append(
                        sim.Projection(
                            exc,
                            exc,
                            conn,
                            synapse_type=sim.StaticSynapse(weight=float(w_exc), delay=float(base_delay)),
                            receptor_type="excitatory",
                        )
                    )
                except Exception:
                    pass

                if inh is not None:
                    try:
                        self._projections.append(
                            sim.Projection(
                                exc,
                                inh,
                                conn,
                                synapse_type=sim.StaticSynapse(weight=float(w_exc), delay=float(base_delay)),
                                receptor_type="excitatory",
                            )
                        )
                    except Exception:
                        pass
                    try:
                        self._projections.append(
                            sim.Projection(
                                inh,
                                exc,
                                conn,
                                synapse_type=sim.StaticSynapse(weight=float(abs(w_inh)), delay=float(base_delay)),
                                receptor_type="inhibitory",
                            )
                        )
                    except Exception:
                        pass
                    try:
                        self._projections.append(
                            sim.Projection(
                                inh,
                                inh,
                                conn,
                                synapse_type=sim.StaticSynapse(weight=float(abs(w_inh)), delay=float(base_delay)),
                                receptor_type="inhibitory",
                            )
                        )
                    except Exception:
                        pass

        # Inter-(sub)region connections via connectome matrices (optional).
        if self._connectome_weights.ndim == 2 and self._connectome_weights.shape[0] == len(self._regions):
            for src in range(len(self._regions)):
                for dst in range(len(self._regions)):
                    if src == dst:
                        continue
                    w = float(self._connectome_weights[src, dst])
                    if abs(w) < 1e-9:
                        continue

                    delay = float(base_delay)
                    if (
                        self._connectome_delays_ms.ndim == 2
                        and self._connectome_delays_ms.shape == self._connectome_weights.shape
                    ):
                        try:
                            delay = float(self._connectome_delays_ms[src, dst])
                        except Exception:
                            delay = float(base_delay)
                    if not np.isfinite(delay) or delay <= 0.0:
                        delay = float(base_delay)

                    if w >= 0.0:
                        source = self._pop_exc[src]
                        receptor = "excitatory"
                        weight = float(w_exc) * float(w)
                    else:
                        source = self._pop_inh[src]
                        if source is None:
                            continue
                        receptor = "inhibitory"
                        weight = float(abs(w_inh)) * float(abs(w))

                    target_exc = self._pop_exc[dst]
                    target_inh = self._pop_inh[dst]

                    if inter_p > 0.0:
                        conn = sim.FixedProbabilityConnector(p_connect=float(inter_p))
                        try:
                            self._projections.append(
                                sim.Projection(
                                    source,
                                    target_exc,
                                    conn,
                                    synapse_type=sim.StaticSynapse(weight=float(abs(weight)), delay=float(delay)),
                                    receptor_type=str(receptor),
                                )
                            )
                        except Exception:
                            pass
                        if target_inh is not None:
                            try:
                                self._projections.append(
                                    sim.Projection(
                                        source,
                                        target_inh,
                                        conn,
                                        synapse_type=sim.StaticSynapse(weight=float(abs(weight)), delay=float(delay)),
                                        receptor_type=str(receptor),
                                    )
                                )
                            except Exception:
                                pass


class _PyNNEngineUnavailable(RuntimeError):
    pass


class PyNNSimulationEngine:
    """Process-global PyNN simulation engine (one per backend).

    PyNN backends (pyNN.nest / pyNN.brian2 / pyNN.neuron / spynnaker8) are
    process-global: calling ``sim.run()`` advances the entire simulation. This
    engine centralizes stepping so multiple region microcircuits can share the
    same backend and be advanced exactly once per cognitive cycle.
    """

    _instances: dict[str, "PyNNSimulationEngine"] = {}
    _instance_lock = threading.Lock()

    def __init__(
        self,
        *,
        backend: str = "nest",
        timestep_ms: float = 0.1,
        min_delay_ms: float = 0.1,
        max_delay_ms: float = 10.0,
        setup: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.backend = self._normalize_backend(backend)
        self.microcircuits: list["PyNNMicrocircuit"] = []
        self._lock = threading.Lock()

        self.sim = self._import_backend(self.backend)

        timestep_ms = float(timestep_ms)
        min_delay_ms = float(min_delay_ms)
        max_delay_ms = float(max_delay_ms)
        if not np.isfinite(timestep_ms) or timestep_ms <= 0.0:
            timestep_ms = 0.1
        if not np.isfinite(min_delay_ms) or min_delay_ms <= 0.0:
            min_delay_ms = timestep_ms
        if not np.isfinite(max_delay_ms) or max_delay_ms <= 0.0:
            max_delay_ms = max(10.0, float(min_delay_ms))
        if max_delay_ms < min_delay_ms:
            max_delay_ms = float(min_delay_ms)

        setup_kwargs = dict(setup or {}) if isinstance(setup, dict) else {}
        for key in ("timestep", "min_delay", "max_delay"):
            setup_kwargs.pop(key, None)

        try:
            self.sim.setup(
                timestep=float(timestep_ms),
                min_delay=float(min_delay_ms),
                max_delay=float(max_delay_ms),
                **setup_kwargs,
            )
        except TypeError:  # pragma: no cover - backend specific signature
            try:
                self.sim.setup(timestep=float(timestep_ms), **setup_kwargs)
            except TypeError:  # pragma: no cover - backend specific signature
                self.sim.setup(timestep=float(timestep_ms))

    @staticmethod
    def _normalize_backend(value: str) -> str:
        value = str(value or "nest").strip().lower()
        aliases = {
            "pynn_nest": "nest",
            "pynn.brian2": "brian2",
            "brian": "brian2",
            "pynn_brian2": "brian2",
            "pynn.neuron": "neuron",
            "nrn": "neuron",
            "pynn_neuron": "neuron",
            "spynnaker": "spinnaker",
            "spinnaker8": "spinnaker",
            "spiNNaker": "spinnaker",
        }
        return aliases.get(value, value)

    @staticmethod
    def _import_backend(backend: str):
        backend = PyNNSimulationEngine._normalize_backend(backend)

        if backend == "spinnaker":
            try:
                import spynnaker8 as sim  # type: ignore
            except Exception as exc:  # pragma: no cover - optional dependency
                raise _PyNNEngineUnavailable(
                    "sPyNNaker (spynnaker8) is not available. Install spynnaker8 to use the PyNN 'spinnaker' backend."
                ) from exc
            return sim

        try:
            if backend == "nest":
                import pyNN.nest as sim  # type: ignore
                return sim
            if backend == "brian2":
                import pyNN.brian2 as sim  # type: ignore
                return sim
            if backend == "neuron":
                import pyNN.neuron as sim  # type: ignore
                return sim
        except Exception as exc:  # pragma: no cover - optional dependency
            raise _PyNNEngineUnavailable(
                f"PyNN backend '{backend}' is not available. Install pyNN (+ the backend runtime) to use model='pynn'."
            ) from exc

        raise ValueError(f"Unsupported PyNN backend: {backend}")

    @classmethod
    def get_or_create(
        cls,
        *,
        backend: str = "nest",
        timestep_ms: float = 0.1,
        min_delay_ms: float = 0.1,
        max_delay_ms: float = 10.0,
        setup: Optional[Dict[str, Any]] = None,
    ) -> "PyNNSimulationEngine":
        backend_norm = cls._normalize_backend(backend)
        with cls._instance_lock:
            if backend_norm not in cls._instances:
                cls._instances[backend_norm] = PyNNSimulationEngine(
                    backend=backend_norm,
                    timestep_ms=float(timestep_ms),
                    min_delay_ms=float(min_delay_ms),
                    max_delay_ms=float(max_delay_ms),
                    setup=setup,
                )
            return cls._instances[backend_norm]

    def register(self, microcircuit: "PyNNMicrocircuit") -> None:
        if microcircuit not in self.microcircuits:
            self.microcircuits.append(microcircuit)

    def step(self, dt_ms: float) -> Dict[str, Any]:
        dt_ms = float(dt_ms)
        if not np.isfinite(dt_ms) or dt_ms <= 0.0:
            raise ValueError("dt_ms must be positive")

        with self._lock:
            for micro in list(self.microcircuits):
                try:
                    micro._apply_pending_drive()
                except Exception:
                    continue

            self.sim.run(float(dt_ms))

            summaries: list[Dict[str, Any]] = []
            total_spikes = 0
            for micro in list(self.microcircuits):
                try:
                    info = micro._collect_after_step(float(dt_ms)) or {}
                except Exception:
                    info = {}
                summaries.append(dict(info) if isinstance(info, dict) else {"state": info})
                try:
                    total_spikes += int(info.get("spike_count", 0) or 0)
                except Exception:
                    pass

            try:
                time_ms = float(self.sim.get_current_time())
            except Exception:
                time_ms = float("nan")

        return {
            "framework": "pynn",
            "backend": str(self.backend),
            "time_ms": time_ms,
            "dt_ms": float(dt_ms),
            "microcircuits": summaries,
            "spike_count": int(total_spikes),
        }


class PyNNMicrocircuit(Microcircuit):
    """PyNN-backed spiking microcircuit (stepped by ``PyNNSimulationEngine``)."""

    requires_global_step = True
    force_update_each_cycle = True

    def __init__(self, *, params: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = dict(cfg or {})
        self.params = dict(params or {})
        self._last_readout: Optional[MicrocircuitReadout] = None
        self._rate_smooth = 0.0

        self._input_gain = float(self.cfg.get("input_gain", 25.0))
        self._target_rate_hz = float(self.cfg.get("target_rate_hz", 20.0))
        self._smooth_tau_ms = float(self.cfg.get("smooth_tau_ms", 50.0))

        if not np.isfinite(self._input_gain):
            self._input_gain = 25.0
        if not np.isfinite(self._target_rate_hz) or self._target_rate_hz <= 0.0:
            self._target_rate_hz = 20.0
        if not np.isfinite(self._smooth_tau_ms) or self._smooth_tau_ms <= 0.0:
            self._smooth_tau_ms = 50.0

        pynn_cfg = self.cfg.get("pynn", {})
        if not isinstance(pynn_cfg, dict):
            pynn_cfg = {}

        backend = str(pynn_cfg.get("backend", "nest") or "nest").strip().lower()
        try:
            timestep_ms = float(pynn_cfg.get("timestep_ms", pynn_cfg.get("resolution_ms", 0.1)))
        except Exception:
            timestep_ms = 0.1
        try:
            min_delay_ms = float(pynn_cfg.get("min_delay_ms", timestep_ms))
        except Exception:
            min_delay_ms = float(timestep_ms)
        try:
            max_delay_ms = float(pynn_cfg.get("max_delay_ms", self.params.get("max_delay_ms", 10.0)))
        except Exception:
            max_delay_ms = 10.0

        setup_kwargs = pynn_cfg.get("setup", {})
        if not isinstance(setup_kwargs, dict):
            setup_kwargs = {}

        self._engine = PyNNSimulationEngine.get_or_create(
            backend=backend,
            timestep_ms=float(timestep_ms),
            min_delay_ms=float(min_delay_ms),
            max_delay_ms=float(max_delay_ms),
            setup=setup_kwargs,
        )
        self.sim = self._engine.sim
        self._backend = str(self._engine.backend)

        self._regions: list[str] = [str(r) for r in (self.params.get("regions") or [])]
        if not self._regions:
            self._regions = ["REGION"]

        try:
            self._neurons_per_region = int(self.params.get("neurons_per_region", 80))
        except Exception:
            self._neurons_per_region = 80
        if self._neurons_per_region <= 0:
            self._neurons_per_region = 80

        try:
            self._excitatory_ratio = float(self.params.get("excitatory_ratio", 0.8))
        except Exception:
            self._excitatory_ratio = 0.8
        self._excitatory_ratio = float(np.clip(self._excitatory_ratio, 0.0, 1.0))

        self._neuron_model = str(self.params.get("neuron_model", "lif") or "lif").strip().lower()
        self._stdp_enabled = bool(self.params.get("stdp_enabled", False))

        self._connectome_weights = np.asarray(self.params.get("connectome_weights", []), dtype=np.float32)
        self._connectome_delays_ms = np.asarray(self.params.get("connectome_delays_ms", []), dtype=np.float32)

        self._pop_exc: list[Any] = []
        self._pop_inh: list[Any] = []
        self._projections: list[Any] = []
        self._pending_drive: list[float] = [0.0 for _ in self._regions]
        self._last_spike_totals: list[int] = [0 for _ in self._regions]

        self._build_network()
        self.reset()
        self._engine.register(self)

    @property
    def engine(self) -> PyNNSimulationEngine:
        return self._engine

    def reset(self) -> None:
        self._last_readout = None
        self._rate_smooth = 0.0
        self._pending_drive = [0.0 for _ in self._regions]

        for ridx in range(len(self._regions)):
            total = 0
            try:
                total += self._population_spike_total(self._pop_exc[ridx])
            except Exception:
                pass
            inh = self._pop_inh[ridx] if ridx < len(self._pop_inh) else None
            if inh is not None:
                try:
                    total += self._population_spike_total(inh)
                except Exception:
                    pass
            if ridx < len(self._last_spike_totals):
                self._last_spike_totals[ridx] = int(total)

    def apply_control(self, control: Dict[str, Any]) -> Dict[str, Any]:
        return _apply_common_microcircuit_control(self, control or {})

    def prepare_step_inputs(self, dt_ms: float, inputs: Dict[str, float], neuromodulators: Dict[str, float]) -> None:
        del dt_ms
        del neuromodulators
        self._pending_drive = self._map_inputs_to_drive(inputs or {})

    def _apply_pending_drive(self) -> None:
        for ridx in range(len(self._regions)):
            amplitude = float(self._pending_drive[ridx]) if ridx < len(self._pending_drive) else 0.0
            if not np.isfinite(amplitude):
                amplitude = 0.0

            exc = self._pop_exc[ridx]
            try:
                exc.set(i_offset=float(amplitude))
            except Exception:
                pass

            inh = self._pop_inh[ridx] if ridx < len(self._pop_inh) else None
            if inh is not None:
                try:
                    inh.set(i_offset=float(amplitude))
                except Exception:
                    pass

    @staticmethod
    def _population_spike_total(population: Any) -> int:
        try:
            counts = population.get_spike_counts()
        except Exception:
            counts = None
        if isinstance(counts, dict):
            total = 0
            for _, value in counts.items():
                try:
                    total += int(value)
                except Exception:
                    continue
            return int(total)

        try:
            data = population.get_data("spikes")
        except Exception:
            return 0

        total = 0
        try:
            segments = getattr(data, "segments", []) or []
        except Exception:
            segments = []
        for seg in segments:
            try:
                trains = getattr(seg, "spiketrains", []) or []
            except Exception:
                trains = []
            for st in trains:
                try:
                    total += int(len(st))
                except Exception:
                    continue
        return int(total)

    def _collect_after_step(self, dt_ms: float) -> Dict[str, Any]:
        dt_s = float(dt_ms) / 1000.0
        spike_count_total = 0
        region_counts: Dict[str, int] = {}

        for ridx, name in enumerate(self._regions):
            total = 0
            try:
                total += self._population_spike_total(self._pop_exc[ridx])
            except Exception:
                pass
            inh = self._pop_inh[ridx] if ridx < len(self._pop_inh) else None
            if inh is not None:
                try:
                    total += self._population_spike_total(inh)
                except Exception:
                    pass

            last = int(self._last_spike_totals[ridx]) if ridx < len(self._last_spike_totals) else 0
            new_count = max(0, int(total) - int(last))
            if ridx < len(self._last_spike_totals):
                self._last_spike_totals[ridx] = int(total)

            region_counts[str(name)] = int(new_count)
            spike_count_total += int(new_count)

        region_rates: Dict[str, float] = {}
        for ridx, name in enumerate(self._regions):
            n_neurons = int(self._neurons_per_region)
            region_rates[str(name)] = float(region_counts.get(str(name), 0)) / max(
                float(n_neurons) * max(dt_s, 1e-9), 1e-9
            )

        total_neurons = int(self._neurons_per_region) * int(len(self._regions))
        rate_hz = float(spike_count_total) / max(float(total_neurons) * max(float(dt_s), 1e-9), 1e-9)

        alpha = float(np.clip(float(dt_ms) / float(self._smooth_tau_ms), 0.0, 1.0))
        self._rate_smooth = (1.0 - alpha) * float(self._rate_smooth) + alpha * float(rate_hz)

        activation = 1.0 - float(np.exp(-float(self._rate_smooth) / float(self._target_rate_hz)))
        activation = float(np.clip(activation, 0.0, 1.0))

        readout = MicrocircuitReadout(
            activation=activation,
            rate_hz=float(rate_hz),
            rate_hz_smooth=float(self._rate_smooth),
            region_rates_hz=region_rates,
            state={
                "framework": "pynn",
                "backend": str(self._backend),
                "stdp_enabled": bool(self._stdp_enabled),
                "spike_count": int(spike_count_total),
                "region_spike_counts": dict(region_counts),
            },
        )
        self._last_readout = readout
        return readout.state

    def step(self, dt_ms: float, inputs: Dict[str, float], neuromodulators: Dict[str, float]) -> MicrocircuitReadout:
        self.prepare_step_inputs(dt_ms, inputs or {}, neuromodulators or {})
        if not bool(self.cfg.get("external_step", True)):
            self._engine.step(float(dt_ms))

        if self._last_readout is None:
            return MicrocircuitReadout(
                activation=0.0,
                rate_hz=0.0,
                rate_hz_smooth=float(self._rate_smooth),
                region_rates_hz={name: 0.0 for name in self._regions},
                state={"framework": "pynn", "backend": str(self._backend), "spike_count": 0, "region_spike_counts": {}},
            )
        return self._last_readout

    def _map_inputs_to_drive(self, inputs: Dict[str, float]) -> list[float]:
        total = 0.0
        if isinstance(inputs, dict):
            for _, value in inputs.items():
                try:
                    total += float(value)
                except Exception:
                    continue

        drive = float(total) * float(self._input_gain)
        if not np.isfinite(drive):
            drive = 0.0

        lowered = {str(k).strip().lower(): v for k, v in (inputs or {}).items()}
        per_region: list[float] = []
        direct_hits = 0
        for name in self._regions:
            key = str(name).strip().lower()
            if key in lowered:
                try:
                    per_region.append(float(lowered[key]) * float(self._input_gain))
                    direct_hits += 1
                except Exception:
                    per_region.append(drive)
            else:
                per_region.append(drive)

        if direct_hits:
            return per_region
        return [drive for _ in self._regions]

    def _pynn_neuron_model_for(self, kind: str) -> Any:
        kind = str(kind or "lif").strip().lower()
        if kind in {"izh", "izhikevich"}:
            try:
                return self.sim.Izhikevich()
            except Exception:
                return self.sim.IF_curr_exp()
        return self.sim.IF_curr_exp()

    def _build_network(self) -> None:
        sim = self.sim

        model_exc = self._pynn_neuron_model_for("lif" if self._neuron_model == "hybrid" else self._neuron_model)
        model_inh = self._pynn_neuron_model_for("izhikevich" if self._neuron_model == "hybrid" else self._neuron_model)

        n_total = int(self._neurons_per_region)
        n_exc = int(round(float(n_total) * float(self._excitatory_ratio)))
        n_exc = int(np.clip(n_exc, 1, max(1, n_total)))
        n_inh = max(0, int(n_total) - int(n_exc))

        for name in self._regions:
            exc = sim.Population(n_exc, model_exc, label=f"{name}_exc")
            inh = sim.Population(n_inh, model_inh, label=f"{name}_inh") if n_inh > 0 else None
            self._pop_exc.append(exc)
            self._pop_inh.append(inh)

            try:
                exc.record(["spikes"])
            except Exception:
                pass
            if inh is not None:
                try:
                    inh.record(["spikes"])
                except Exception:
                    pass

        try:
            intra_p = float(self.params.get("intra_connection_prob", 0.05))
        except Exception:
            intra_p = 0.05
        try:
            inter_p = float(self.params.get("inter_connection_prob", 0.02))
        except Exception:
            inter_p = 0.02
        if not np.isfinite(intra_p) or intra_p < 0.0:
            intra_p = 0.0
        if not np.isfinite(inter_p) or inter_p < 0.0:
            inter_p = 0.0
        intra_p = float(np.clip(intra_p, 0.0, 1.0))
        inter_p = float(np.clip(inter_p, 0.0, 1.0))

        try:
            base_delay = float(self.params.get("synaptic_delay_ms", 1.0))
        except Exception:
            base_delay = 1.0
        if not np.isfinite(base_delay) or base_delay <= 0.0:
            base_delay = 1.0

        try:
            w_exc = float(self.params.get("w_exc", 1.0))
        except Exception:
            w_exc = 1.0
        try:
            w_inh = float(self.params.get("w_inh", 1.0))
        except Exception:
            w_inh = 1.0
        if not np.isfinite(w_exc):
            w_exc = 1.0
        if not np.isfinite(w_inh):
            w_inh = 1.0

        # Local intra-(sub)region connections.
        for ridx in range(len(self._regions)):
            exc = self._pop_exc[ridx]
            inh = self._pop_inh[ridx]

            if intra_p > 0.0:
                conn = sim.FixedProbabilityConnector(p_connect=float(intra_p))
                try:
                    self._projections.append(
                        sim.Projection(
                            exc,
                            exc,
                            conn,
                            synapse_type=sim.StaticSynapse(weight=float(w_exc), delay=float(base_delay)),
                            receptor_type="excitatory",
                        )
                    )
                except TypeError:
                    try:
                        self._projections.append(
                            sim.Projection(
                                exc,
                                exc,
                                conn,
                                synapse_type=sim.StaticSynapse(weight=float(w_exc), delay=float(base_delay)),
                            )
                        )
                    except Exception:
                        pass
                except Exception:
                    pass

                if inh is not None:
                    try:
                        self._projections.append(
                            sim.Projection(
                                exc,
                                inh,
                                conn,
                                synapse_type=sim.StaticSynapse(weight=float(w_exc), delay=float(base_delay)),
                                receptor_type="excitatory",
                            )
                        )
                    except TypeError:
                        try:
                            self._projections.append(
                                sim.Projection(
                                    exc,
                                    inh,
                                    conn,
                                    synapse_type=sim.StaticSynapse(weight=float(w_exc), delay=float(base_delay)),
                                )
                            )
                        except Exception:
                            pass
                    except Exception:
                        pass

                    try:
                        self._projections.append(
                            sim.Projection(
                                inh,
                                exc,
                                conn,
                                synapse_type=sim.StaticSynapse(weight=float(abs(w_inh)), delay=float(base_delay)),
                                receptor_type="inhibitory",
                            )
                        )
                    except TypeError:
                        try:
                            self._projections.append(
                                sim.Projection(
                                    inh,
                                    exc,
                                    conn,
                                    synapse_type=sim.StaticSynapse(weight=float(abs(w_inh)), delay=float(base_delay)),
                                )
                            )
                        except Exception:
                            pass
                    except Exception:
                        pass

                    try:
                        self._projections.append(
                            sim.Projection(
                                inh,
                                inh,
                                conn,
                                synapse_type=sim.StaticSynapse(weight=float(abs(w_inh)), delay=float(base_delay)),
                                receptor_type="inhibitory",
                            )
                        )
                    except TypeError:
                        try:
                            self._projections.append(
                                sim.Projection(
                                    inh,
                                    inh,
                                    conn,
                                    synapse_type=sim.StaticSynapse(weight=float(abs(w_inh)), delay=float(base_delay)),
                                )
                            )
                        except Exception:
                            pass
                    except Exception:
                        pass

        # Inter-(sub)region connections via connectome matrices (optional).
        if self._connectome_weights.ndim == 2 and self._connectome_weights.shape[0] == len(self._regions):
            for src in range(len(self._regions)):
                for dst in range(len(self._regions)):
                    if src == dst:
                        continue
                    w = float(self._connectome_weights[src, dst])
                    if abs(w) < 1e-9:
                        continue

                    delay = float(base_delay)
                    if (
                        self._connectome_delays_ms.ndim == 2
                        and self._connectome_delays_ms.shape == self._connectome_weights.shape
                    ):
                        try:
                            delay = float(self._connectome_delays_ms[src, dst])
                        except Exception:
                            delay = float(base_delay)
                    if not np.isfinite(delay) or delay <= 0.0:
                        delay = float(base_delay)

                    if w >= 0.0:
                        source = self._pop_exc[src]
                        receptor = "excitatory"
                        weight = float(w_exc) * float(w)
                    else:
                        source = self._pop_inh[src]
                        if source is None:
                            continue
                        receptor = "inhibitory"
                        weight = float(abs(w_inh)) * float(abs(w))

                    target_exc = self._pop_exc[dst]
                    target_inh = self._pop_inh[dst]

                    if inter_p > 0.0:
                        conn = sim.FixedProbabilityConnector(p_connect=float(inter_p))
                        try:
                            self._projections.append(
                                sim.Projection(
                                    source,
                                    target_exc,
                                    conn,
                                    synapse_type=sim.StaticSynapse(weight=float(abs(weight)), delay=float(delay)),
                                    receptor_type=str(receptor),
                                )
                            )
                        except TypeError:
                            try:
                                self._projections.append(
                                    sim.Projection(
                                        source,
                                        target_exc,
                                        conn,
                                        synapse_type=sim.StaticSynapse(weight=float(abs(weight)), delay=float(delay)),
                                    )
                                )
                            except Exception:
                                pass
                        except Exception:
                            pass
                        if target_inh is not None:
                            try:
                                self._projections.append(
                                    sim.Projection(
                                        source,
                                        target_inh,
                                        conn,
                                        synapse_type=sim.StaticSynapse(weight=float(abs(weight)), delay=float(delay)),
                                        receptor_type=str(receptor),
                                    )
                                )
                            except TypeError:
                                try:
                                    self._projections.append(
                                        sim.Projection(
                                            source,
                                            target_inh,
                                            conn,
                                            synapse_type=sim.StaticSynapse(weight=float(abs(weight)), delay=float(delay)),
                                        )
                                    )
                                except Exception:
                                    pass
                            except Exception:
                                pass

class Brian2Microcircuit(Microcircuit):
    """Brian2-backed microcircuit for rapid prototyping of neuron/synapse equations.

    Notes
    - Brian2 runs in-process and supports custom differential equations; this wrapper keeps the
      project-facing interface consistent (`Microcircuit.step/reset`).
    - For portability, this defaults to the `numpy` codegen target and a simple unitless LIF-like model.
    - If Brian2 is not installed, instantiation raises and the caller can fall back gracefully.
    """

    _global_lock = threading.Lock()

    def __init__(self, *, params: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = dict(cfg or {})
        self.params = dict(params or {})
        self._last_readout: Optional[MicrocircuitReadout] = None
        self._rate_smooth = 0.0

        # Lazy import: keep Brian2 optional.
        try:
            import brian2 as b2  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise _Brian2Unavailable(
                "Brian2 is not available. Install brian2 to use the 'brian2' microcircuit model."
            ) from exc

        self.b2 = b2

        brian_cfg = self.cfg.get("brian2", {})
        if not isinstance(brian_cfg, dict):
            brian_cfg = {}

        # Prefer safe defaults (no compilation toolchain required).
        try:
            codegen_target = str(brian_cfg.get("codegen_target", "numpy") or "numpy").strip().lower()
        except Exception:
            codegen_target = "numpy"
        if codegen_target not in {"numpy", "cython"}:
            codegen_target = "numpy"
        try:
            self.b2.prefs.codegen.target = codegen_target
        except Exception:
            pass
        self._codegen_target = codegen_target

        try:
            resolution_ms = float(brian_cfg.get("resolution_ms", 0.1))
        except Exception:
            resolution_ms = 0.1
        if not np.isfinite(resolution_ms) or resolution_ms <= 0.0:
            resolution_ms = 0.1

        try:
            seed = int(self.params.get("seed", brian_cfg.get("seed", 0)) or 0)
        except Exception:
            seed = 0
        if seed:
            try:
                self.b2.seed(seed)
            except Exception:
                pass

        self._clock = self.b2.Clock(dt=float(resolution_ms) * self.b2.ms)

        self._input_gain = float(self.cfg.get("input_gain", 25.0))
        self._target_rate_hz = float(self.cfg.get("target_rate_hz", 20.0))
        self._smooth_tau_ms = float(self.cfg.get("smooth_tau_ms", 50.0))
        if not np.isfinite(self._input_gain):
            self._input_gain = 25.0
        if not np.isfinite(self._target_rate_hz) or self._target_rate_hz <= 0.0:
            self._target_rate_hz = 20.0
        if not np.isfinite(self._smooth_tau_ms) or self._smooth_tau_ms <= 0.0:
            self._smooth_tau_ms = 50.0

        self._regions: list[str] = [str(r) for r in (self.params.get("regions") or [])]
        if not self._regions:
            self._regions = ["REGION"]

        try:
            self._neurons_per_region = int(self.params.get("neurons_per_region", 80))
        except Exception:
            self._neurons_per_region = 80
        if self._neurons_per_region <= 0:
            self._neurons_per_region = 80

        try:
            self._excitatory_ratio = float(self.params.get("excitatory_ratio", 0.8))
        except Exception:
            self._excitatory_ratio = 0.8
        self._excitatory_ratio = float(np.clip(self._excitatory_ratio, 0.0, 1.0))

        self._connectome_weights = np.asarray(self.params.get("connectome_weights", []), dtype=np.float32)
        self._connectome_delays_ms = np.asarray(self.params.get("connectome_delays_ms", []), dtype=np.float32)

        # Default unitless LIF-like model (easy to replace via cfg["brian2"]["neuron"]).
        neuron_cfg = brian_cfg.get("neuron", {})
        if not isinstance(neuron_cfg, dict):
            neuron_cfg = {}

        eqs = neuron_cfg.get(
            "equations",
            """
            dv/dt = (-v + I_ext + I_syn) / tau : 1
            dI_syn/dt = -I_syn / tau_syn : 1
            I_ext : 1
            tau : second (constant)
            tau_syn : second (constant)
            """,
        )
        threshold = str(neuron_cfg.get("threshold", "v > v_thresh") or "v > v_thresh")
        reset = str(neuron_cfg.get("reset", "v = v_reset") or "v = v_reset")
        method = str(neuron_cfg.get("method", "euler") or "euler")

        try:
            tau_ms = float(neuron_cfg.get("tau_ms", self.params.get("lif_tau_m_ms", 20.0)))
        except Exception:
            tau_ms = 20.0
        if not np.isfinite(tau_ms) or tau_ms <= 0.0:
            tau_ms = 20.0

        try:
            tau_syn_ms = float(neuron_cfg.get("tau_syn_ms", 5.0))
        except Exception:
            tau_syn_ms = 5.0
        if not np.isfinite(tau_syn_ms) or tau_syn_ms <= 0.0:
            tau_syn_ms = 5.0

        try:
            v_thresh = float(neuron_cfg.get("v_thresh", 1.0))
        except Exception:
            v_thresh = 1.0
        if not np.isfinite(v_thresh):
            v_thresh = 1.0

        try:
            v_reset = float(neuron_cfg.get("v_reset", 0.0))
        except Exception:
            v_reset = 0.0
        if not np.isfinite(v_reset):
            v_reset = 0.0

        syn_cfg = brian_cfg.get("synapse", {})
        if not isinstance(syn_cfg, dict):
            syn_cfg = {}
        on_pre = str(syn_cfg.get("on_pre", "I_syn_post += w") or "I_syn_post += w")
        syn_model = str(syn_cfg.get("model", "w : 1") or "w : 1")

        try:
            base_delay_ms = float(self.params.get("synaptic_delay_ms", 1.0))
        except Exception:
            base_delay_ms = 1.0
        if not np.isfinite(base_delay_ms) or base_delay_ms <= 0.0:
            base_delay_ms = 1.0

        try:
            w_exc = float(syn_cfg.get("w_exc", 0.6))
        except Exception:
            w_exc = 0.6
        if not np.isfinite(w_exc):
            w_exc = 0.6

        try:
            w_inh = float(syn_cfg.get("w_inh", -0.8))
        except Exception:
            w_inh = -0.8
        if not np.isfinite(w_inh):
            w_inh = -0.8
        if w_inh > 0.0:
            w_inh = -abs(w_inh)

        try:
            intra_p = float(self.params.get("intra_connection_prob", 0.05))
        except Exception:
            intra_p = 0.05
        try:
            inter_p = float(self.params.get("inter_connection_prob", 0.02))
        except Exception:
            inter_p = 0.02
        if not np.isfinite(intra_p) or intra_p < 0.0:
            intra_p = 0.0
        if not np.isfinite(inter_p) or inter_p < 0.0:
            inter_p = 0.0
        intra_p = float(np.clip(intra_p, 0.0, 1.0))
        inter_p = float(np.clip(inter_p, 0.0, 1.0))

        # Build per-(sub)region E/I populations.
        self._groups_exc: list[Any] = []
        self._groups_inh: list[Any] = []
        self._mon_exc: list[Any] = []
        self._mon_inh: list[Any] = []
        self._synapses: list[Any] = []
        self._last_spike_counts: list[int] = []
        self._pending_drive: list[float] = [0.0 for _ in self._regions]

        for _ in self._regions:
            n_total = int(self._neurons_per_region)
            n_exc = int(round(float(n_total) * float(self._excitatory_ratio)))
            n_exc = int(np.clip(n_exc, 1, max(1, n_total)))
            n_inh = max(0, int(n_total) - int(n_exc))

            exc = self.b2.NeuronGroup(
                n_exc,
                model=eqs,
                threshold=threshold,
                reset=reset,
                method=method,
                clock=self._clock,
                namespace={"v_thresh": v_thresh, "v_reset": v_reset},
            )
            exc.v = float(v_reset)
            exc.I_ext = 0.0
            exc.I_syn = 0.0
            exc.tau = float(tau_ms) * self.b2.ms
            exc.tau_syn = float(tau_syn_ms) * self.b2.ms

            inh = None
            if n_inh > 0:
                inh = self.b2.NeuronGroup(
                    n_inh,
                    model=eqs,
                    threshold=threshold,
                    reset=reset,
                    method=method,
                    clock=self._clock,
                    namespace={"v_thresh": v_thresh, "v_reset": v_reset},
                )
                inh.v = float(v_reset)
                inh.I_ext = 0.0
                inh.I_syn = 0.0
                inh.tau = float(tau_ms) * self.b2.ms
                inh.tau_syn = float(tau_syn_ms) * self.b2.ms

            self._groups_exc.append(exc)
            self._groups_inh.append(inh)

            mon_exc = self.b2.SpikeMonitor(exc)
            self._mon_exc.append(mon_exc)
            mon_inh = self.b2.SpikeMonitor(inh) if inh is not None else None
            self._mon_inh.append(mon_inh)
            self._last_spike_counts.append(0)

        # Local intra-(sub)region connections.
        for ridx in range(len(self._regions)):
            exc = self._groups_exc[ridx]
            inh = self._groups_inh[ridx]

            if intra_p > 0.0:
                syn_ee = self.b2.Synapses(exc, exc, model=syn_model, on_pre=on_pre, method=method)
                syn_ee.connect(p=intra_p)
                syn_ee.w = float(w_exc)
                syn_ee.delay = float(base_delay_ms) * self.b2.ms
                self._synapses.append(syn_ee)

                if inh is not None:
                    syn_ei = self.b2.Synapses(exc, inh, model=syn_model, on_pre=on_pre, method=method)
                    syn_ei.connect(p=intra_p)
                    syn_ei.w = float(w_exc)
                    syn_ei.delay = float(base_delay_ms) * self.b2.ms
                    self._synapses.append(syn_ei)

                    syn_ie = self.b2.Synapses(inh, exc, model=syn_model, on_pre=on_pre, method=method)
                    syn_ie.connect(p=intra_p)
                    syn_ie.w = float(w_inh)
                    syn_ie.delay = float(base_delay_ms) * self.b2.ms
                    self._synapses.append(syn_ie)

                    syn_ii = self.b2.Synapses(inh, inh, model=syn_model, on_pre=on_pre, method=method)
                    syn_ii.connect(p=intra_p)
                    syn_ii.w = float(w_inh)
                    syn_ii.delay = float(base_delay_ms) * self.b2.ms
                    self._synapses.append(syn_ii)

        # Inter-(sub)region connections via connectome matrices (optional).
        if self._connectome_weights.ndim == 2 and self._connectome_weights.shape[0] == len(self._regions):
            for src in range(len(self._regions)):
                for dst in range(len(self._regions)):
                    if src == dst:
                        continue
                    w = float(self._connectome_weights[src, dst])
                    if abs(w) < 1e-9:
                        continue

                    delay_ms = float(base_delay_ms)
                    if (
                        self._connectome_delays_ms.ndim == 2
                        and self._connectome_delays_ms.shape == self._connectome_weights.shape
                    ):
                        try:
                            delay_ms = float(self._connectome_delays_ms[src, dst])
                        except Exception:
                            delay_ms = float(base_delay_ms)
                    if not np.isfinite(delay_ms) or delay_ms <= 0.0:
                        delay_ms = float(base_delay_ms)

                    if w >= 0.0:
                        source = self._groups_exc[src]
                        weight_val = float(w_exc) * float(w)
                    else:
                        source = self._groups_inh[src] if self._groups_inh[src] is not None else self._groups_exc[src]
                        weight_val = float(w_inh) * float(abs(w))

                    target_exc = self._groups_exc[dst]
                    target_inh = self._groups_inh[dst]

                    if inter_p > 0.0:
                        syn_xe = self.b2.Synapses(source, target_exc, model=syn_model, on_pre=on_pre, method=method)
                        syn_xe.connect(p=inter_p)
                        syn_xe.w = float(weight_val)
                        syn_xe.delay = float(delay_ms) * self.b2.ms
                        self._synapses.append(syn_xe)

                        if target_inh is not None:
                            syn_xi = self.b2.Synapses(
                                source, target_inh, model=syn_model, on_pre=on_pre, method=method
                            )
                            syn_xi.connect(p=inter_p)
                            syn_xi.w = float(weight_val)
                            syn_xi.delay = float(delay_ms) * self.b2.ms
                            self._synapses.append(syn_xi)

        # Build an explicit Network (avoid Brian2 magic network side effects).
        objects: list[Any] = []
        for ridx in range(len(self._regions)):
            objects.append(self._groups_exc[ridx])
            if self._groups_inh[ridx] is not None:
                objects.append(self._groups_inh[ridx])
            objects.append(self._mon_exc[ridx])
            if self._mon_inh[ridx] is not None:
                objects.append(self._mon_inh[ridx])
        objects.extend(self._synapses)
        self._network = self.b2.Network(*objects)

        # Prime the network so monitors have an initial baseline count.
        self.reset()

    def reset(self) -> None:
        self._last_readout = None
        self._rate_smooth = 0.0
        self._pending_drive = [0.0 for _ in self._regions]

        # Capture baseline spike counts for delta accounting.
        for ridx in range(len(self._regions)):
            total = 0
            try:
                total += int(getattr(self._mon_exc[ridx], "num_spikes", 0) or 0)
            except Exception:
                pass
            mon_inh = self._mon_inh[ridx]
            if mon_inh is not None:
                try:
                    total += int(getattr(mon_inh, "num_spikes", 0) or 0)
                except Exception:
                    pass
            self._last_spike_counts[ridx] = int(total)

    def apply_control(self, control: Dict[str, Any]) -> Dict[str, Any]:
        return _apply_common_microcircuit_control(self, control or {})

    def scale_synapses(self, factor: float, *, exc_only: bool = True, inh_only: bool = False) -> Dict[str, Any]:
        try:
            factor_f = float(factor)
        except Exception:
            factor_f = 1.0
        if not np.isfinite(factor_f) or factor_f < 0.0:
            factor_f = 1.0

        scaled = 0
        for syn in list(self._synapses):
            try:
                w = np.asarray(syn.w[:], dtype=float)
            except Exception:
                continue

            if bool(exc_only) and not bool(inh_only):
                mask = w > 0.0
            elif bool(inh_only) and not bool(exc_only):
                mask = w < 0.0
            else:
                mask = np.ones_like(w, dtype=bool)

            if not np.any(mask):
                continue

            w[mask] = w[mask] * float(factor_f)
            try:
                syn.w = w
                scaled += int(np.count_nonzero(mask))
            except Exception:
                continue

        return {"scaled": int(scaled), "factor": float(factor_f)}

    def _map_inputs_to_drive(self, inputs: Dict[str, float]) -> list[float]:
        total = 0.0
        if isinstance(inputs, dict):
            for _, value in inputs.items():
                try:
                    total += float(value)
                except Exception:
                    continue

        drive = float(total) * float(self._input_gain)
        if not np.isfinite(drive):
            drive = 0.0

        lowered = {str(k).strip().lower(): v for k, v in (inputs or {}).items()}
        per_region: list[float] = []
        direct_hits = 0
        for name in self._regions:
            key = str(name).strip().lower()
            if key in lowered:
                try:
                    per_region.append(float(lowered[key]) * float(self._input_gain))
                    direct_hits += 1
                except Exception:
                    per_region.append(drive)
            else:
                per_region.append(drive)

        if direct_hits:
            return per_region
        return [drive for _ in self._regions]

    def step(self, dt_ms: float, inputs: Dict[str, float], neuromodulators: Dict[str, float]) -> MicrocircuitReadout:
        del neuromodulators
        dt_ms = float(dt_ms)
        if not np.isfinite(dt_ms) or dt_ms <= 0.0:
            raise ValueError("dt_ms must be positive")

        drive = self._map_inputs_to_drive(inputs or {})
        self._pending_drive = list(drive)

        # Serialize Brian2 stepping; codegen/runtime state is not designed for concurrent access.
        with self._global_lock:
            for ridx in range(len(self._regions)):
                amp = float(self._pending_drive[ridx]) if ridx < len(self._pending_drive) else 0.0
                if not np.isfinite(amp):
                    amp = 0.0
                try:
                    self._groups_exc[ridx].I_ext = float(amp)
                except Exception:
                    pass
                inh = self._groups_inh[ridx]
                if inh is not None:
                    try:
                        inh.I_ext = float(amp)
                    except Exception:
                        pass

            self._network.run(float(dt_ms) * self.b2.ms, report=None)

        dt_s = float(dt_ms) / 1000.0
        region_counts: Dict[str, int] = {}
        spike_total = 0

        for ridx, name in enumerate(self._regions):
            total = 0
            try:
                total += int(getattr(self._mon_exc[ridx], "num_spikes", 0) or 0)
            except Exception:
                pass
            mon_inh = self._mon_inh[ridx]
            if mon_inh is not None:
                try:
                    total += int(getattr(mon_inh, "num_spikes", 0) or 0)
                except Exception:
                    pass

            last = int(self._last_spike_counts[ridx])
            new_count = max(0, int(total) - last)
            self._last_spike_counts[ridx] = int(total)

            region_counts[str(name)] = int(new_count)
            spike_total += int(new_count)

        total_neurons = int(self._neurons_per_region) * int(len(self._regions))
        rate_hz = float(spike_total) / max(float(total_neurons) * max(float(dt_s), 1e-9), 1e-9)

        alpha = float(np.clip(dt_ms / float(self._smooth_tau_ms), 0.0, 1.0))
        self._rate_smooth = (1.0 - alpha) * float(self._rate_smooth) + alpha * float(rate_hz)

        region_rates: Dict[str, float] = {}
        for ridx, name in enumerate(self._regions):
            region_rates[str(name)] = float(region_counts.get(str(name), 0)) / max(
                float(self._neurons_per_region) * max(float(dt_s), 1e-9), 1e-9
            )

        activation = 1.0 - float(np.exp(-float(self._rate_smooth) / float(self._target_rate_hz)))
        activation = float(np.clip(activation, 0.0, 1.0))

        readout = MicrocircuitReadout(
            activation=activation,
            rate_hz=float(rate_hz),
            rate_hz_smooth=float(self._rate_smooth),
            region_rates_hz=region_rates,
            state={
                "framework": "brian2",
                "spike_count": int(spike_total),
                "region_spike_counts": dict(region_counts),
                "codegen_target": str(getattr(self, "_codegen_target", "unknown") or "unknown"),
            },
        )
        self._last_readout = readout
        return readout


class _LoihiUnavailable(RuntimeError):
    pass


class LoihiMicrocircuit(Microcircuit):
    """Intel Loihi microcircuit wrapper (via nengo_loihi).

    This is intended for mapping a *subset* of the model (e.g., basal ganglia loops,
    sensory microcircuits) onto the Loihi toolchain while keeping the cognitive layer
    unchanged.

    Notes
    - Uses `nengo_loihi` as the programmable interface. Hardware availability is optional:
      set `cfg.loihi.target="sim"` for software execution or `"loihi"` for on-chip runs.
    - The wrapper is intentionally small and focuses on (1) stepping, (2) spike-count readout,
      and (3) clean fallback when Loihi dependencies are missing.
    """

    _global_lock = threading.Lock()

    def __init__(self, *, params: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = dict(cfg or {})
        self.params = dict(params or {})
        self._last_readout: Optional[MicrocircuitReadout] = None
        self._rate_smooth = 0.0

        try:
            import nengo  # type: ignore
            import nengo_loihi  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise _LoihiUnavailable(
                "nengo_loihi is not available. Install nengo + nengo_loihi to use the 'loihi' microcircuit model."
            ) from exc

        self.nengo = nengo
        self.nengo_loihi = nengo_loihi

        loihi_cfg = self.cfg.get("loihi", {})
        if not isinstance(loihi_cfg, dict):
            loihi_cfg = {}

        try:
            self._resolution_ms = float(loihi_cfg.get("resolution_ms", 1.0))
        except Exception:
            self._resolution_ms = 1.0
        if not np.isfinite(self._resolution_ms) or self._resolution_ms <= 0.0:
            self._resolution_ms = 1.0

        try:
            target = str(loihi_cfg.get("target", "sim") or "sim").strip().lower()
        except Exception:
            target = "sim"
        if target not in {"sim", "loihi"}:
            target = "sim"
        self._target = target

        try:
            self._on_chip = bool(loihi_cfg.get("on_chip", target == "loihi"))
        except Exception:
            self._on_chip = target == "loihi"

        try:
            self._dopamine_input_gain = float(loihi_cfg.get("dopamine_input_gain", 0.0))
        except Exception:
            self._dopamine_input_gain = 0.0
        if not np.isfinite(self._dopamine_input_gain) or self._dopamine_input_gain < 0.0:
            self._dopamine_input_gain = 0.0

        try:
            seed = int(self.params.get("seed", loihi_cfg.get("seed", 0)) or 0)
        except Exception:
            seed = 0
        self._seed = int(seed)

        self._input_gain = float(self.cfg.get("input_gain", 25.0))
        self._target_rate_hz = float(self.cfg.get("target_rate_hz", 20.0))
        self._smooth_tau_ms = float(self.cfg.get("smooth_tau_ms", 50.0))
        if not np.isfinite(self._input_gain):
            self._input_gain = 25.0
        if not np.isfinite(self._target_rate_hz) or self._target_rate_hz <= 0.0:
            self._target_rate_hz = 20.0
        if not np.isfinite(self._smooth_tau_ms) or self._smooth_tau_ms <= 0.0:
            self._smooth_tau_ms = 50.0

        self._regions: list[str] = [str(r) for r in (self.params.get("regions") or [])]
        if not self._regions:
            self._regions = ["REGION"]

        try:
            self._neurons_per_region = int(self.params.get("neurons_per_region", 80))
        except Exception:
            self._neurons_per_region = 80
        if self._neurons_per_region <= 0:
            self._neurons_per_region = 80

        try:
            self._excitatory_ratio = float(self.params.get("excitatory_ratio", 0.8))
        except Exception:
            self._excitatory_ratio = 0.8
        self._excitatory_ratio = float(np.clip(self._excitatory_ratio, 0.0, 1.0))

        self._connectome_weights = np.asarray(self.params.get("connectome_weights", []), dtype=np.float32)

        try:
            self._intra_p = float(self.params.get("intra_connection_prob", 0.05))
        except Exception:
            self._intra_p = 0.05
        try:
            self._inter_p = float(self.params.get("inter_connection_prob", 0.02))
        except Exception:
            self._inter_p = 0.02
        if not np.isfinite(self._intra_p) or self._intra_p < 0.0:
            self._intra_p = 0.0
        if not np.isfinite(self._inter_p) or self._inter_p < 0.0:
            self._inter_p = 0.0
        self._intra_p = float(np.clip(self._intra_p, 0.0, 1.0))
        self._inter_p = float(np.clip(self._inter_p, 0.0, 1.0))

        try:
            self._w_exc = float(self.params.get("w_exc", 1.0))
        except Exception:
            self._w_exc = 1.0
        try:
            self._w_inh = float(self.params.get("w_inh", 1.0))
        except Exception:
            self._w_inh = 1.0
        if not np.isfinite(self._w_exc):
            self._w_exc = 1.0
        if not np.isfinite(self._w_inh):
            self._w_inh = 1.0

        # Nengo/Loihi objects
        self._network = None
        self._simulator = None
        self._pop_exc: list[Any] = []
        self._pop_inh: list[Any] = []
        self._probe_exc: list[Any] = []
        self._probe_inh: list[Any] = []
        self._drive_values: list[float] = [0.0 for _ in self._regions]
        self._last_sample_idx = 0

        self._build_network()
        self.reset()

    def reset(self) -> None:
        self._last_readout = None
        self._rate_smooth = 0.0
        self._drive_values = [0.0 for _ in self._regions]
        self._last_sample_idx = 0

        sim = getattr(self, "_simulator", None)
        if sim is not None:
            with self._global_lock:
                try:
                    reset = getattr(sim, "reset", None)
                    if callable(reset):
                        reset()
                except Exception:
                    pass
                self._last_sample_idx = 0

    def apply_control(self, control: Dict[str, Any]) -> Dict[str, Any]:
        return _apply_common_microcircuit_control(self, control or {})

    def _map_inputs_to_drive(self, inputs: Dict[str, float]) -> list[float]:
        total = 0.0
        if isinstance(inputs, dict):
            for _, value in inputs.items():
                try:
                    total += float(value)
                except Exception:
                    continue

        drive = float(total) * float(self._input_gain)
        if not np.isfinite(drive):
            drive = 0.0

        lowered = {str(k).strip().lower(): v for k, v in (inputs or {}).items()}
        per_region: list[float] = []
        direct_hits = 0
        for name in self._regions:
            key = str(name).strip().lower()
            if key in lowered:
                try:
                    per_region.append(float(lowered[key]) * float(self._input_gain))
                    direct_hits += 1
                except Exception:
                    per_region.append(drive)
            else:
                per_region.append(drive)

        if direct_hits:
            return per_region
        return [drive for _ in self._regions]

    def _build_network(self) -> None:
        nengo = self.nengo
        nengo_loihi = self.nengo_loihi

        rng = np.random.default_rng(self._seed if self._seed else None)

        try:
            net = nengo_loihi.Network(seed=self._seed if self._seed else None)
        except Exception:
            net = nengo_loihi.Network()

        with net:
            try:
                nengo_loihi.add_params(net)
            except Exception:
                pass

            if self._on_chip:
                try:
                    net.config[nengo_loihi.Ensemble].on_chip = True
                except Exception:
                    pass

            neuron_type = None
            try:
                neuron_type = nengo_loihi.neurons.LoihiLIF()
            except Exception:
                try:
                    neuron_type = nengo.LIF()
                except Exception:
                    neuron_type = None

            for ridx, name in enumerate(self._regions):
                n_total = int(self._neurons_per_region)
                n_exc = int(round(float(n_total) * float(self._excitatory_ratio)))
                n_exc = int(np.clip(n_exc, 1, max(1, n_total)))
                n_inh = max(0, int(n_total) - int(n_exc))

                def _make_input(idx: int):
                    return lambda t, _idx=idx: float(self._drive_values[_idx])

                inp = nengo.Node(_make_input(ridx), size_out=1, label=f"{name}_drive")

                exc = nengo.Ensemble(
                    n_neurons=n_exc,
                    dimensions=1,
                    neuron_type=neuron_type,
                    label=f"{name}_exc",
                )
                nengo.Connection(inp, exc, synapse=None)
                self._pop_exc.append(exc)

                inh = None
                if n_inh > 0:
                    inh = nengo.Ensemble(
                        n_neurons=n_inh,
                        dimensions=1,
                        neuron_type=neuron_type,
                        label=f"{name}_inh",
                    )
                    nengo.Connection(inp, inh, synapse=None)
                self._pop_inh.append(inh)

                try:
                    self._probe_exc.append(nengo.Probe(exc.neurons))
                except Exception:
                    self._probe_exc.append(None)
                try:
                    self._probe_inh.append(nengo.Probe(inh.neurons) if inh is not None else None)
                except Exception:
                    self._probe_inh.append(None)

            # Intra-(sub)region random sparse connections (neuron-to-neuron).
            for ridx in range(len(self._regions)):
                exc = self._pop_exc[ridx]
                inh = self._pop_inh[ridx]
                if self._intra_p <= 0.0:
                    continue

                def _sparse_matrix(n_post: int, n_pre: int, *, p: float, w: float) -> np.ndarray:
                    mask = rng.random((n_post, n_pre)) < float(p)
                    mat = mask.astype(np.float32) * float(w)
                    return mat

                try:
                    nengo.Connection(
                        exc.neurons,
                        exc.neurons,
                        transform=_sparse_matrix(exc.n_neurons, exc.n_neurons, p=self._intra_p, w=abs(self._w_exc)),
                        synapse=None,
                    )
                except Exception:
                    pass

                if inh is not None:
                    try:
                        nengo.Connection(
                            exc.neurons,
                            inh.neurons,
                            transform=_sparse_matrix(inh.n_neurons, exc.n_neurons, p=self._intra_p, w=abs(self._w_exc)),
                            synapse=None,
                        )
                    except Exception:
                        pass
                    try:
                        nengo.Connection(
                            inh.neurons,
                            exc.neurons,
                            transform=_sparse_matrix(exc.n_neurons, inh.n_neurons, p=self._intra_p, w=-abs(self._w_inh)),
                            synapse=None,
                        )
                    except Exception:
                        pass
                    try:
                        nengo.Connection(
                            inh.neurons,
                            inh.neurons,
                            transform=_sparse_matrix(inh.n_neurons, inh.n_neurons, p=self._intra_p, w=-abs(self._w_inh)),
                            synapse=None,
                        )
                    except Exception:
                        pass

            # Inter-(sub)region projections via connectome weights (optional).
            if self._connectome_weights.ndim == 2 and self._connectome_weights.shape[0] == len(self._regions):
                for src in range(len(self._regions)):
                    for dst in range(len(self._regions)):
                        if src == dst:
                            continue
                        w = float(self._connectome_weights[src, dst])
                        if abs(w) < 1e-9 or self._inter_p <= 0.0:
                            continue

                        if w >= 0.0:
                            source = self._pop_exc[src]
                            weight = float(abs(self._w_exc)) * float(w)
                        else:
                            source = self._pop_inh[src]
                            if source is None:
                                continue
                            weight = -float(abs(self._w_inh)) * float(abs(w))

                        target_exc = self._pop_exc[dst]
                        target_inh = self._pop_inh[dst]

                        try:
                            nengo.Connection(
                                source.neurons,
                                target_exc.neurons,
                                transform=_sparse_matrix(target_exc.n_neurons, source.n_neurons, p=self._inter_p, w=weight),
                                synapse=None,
                            )
                        except Exception:
                            pass
                        if target_inh is not None:
                            try:
                                nengo.Connection(
                                    source.neurons,
                                    target_inh.neurons,
                                    transform=_sparse_matrix(target_inh.n_neurons, source.n_neurons, p=self._inter_p, w=weight),
                                    synapse=None,
                                )
                            except Exception:
                                pass

        self._network = net

        dt_s = float(self._resolution_ms) / 1000.0
        with self._global_lock:
            self._simulator = nengo_loihi.Simulator(net, dt=float(dt_s), target=str(self._target))

    def step(self, dt_ms: float, inputs: Dict[str, float], neuromodulators: Dict[str, float]) -> MicrocircuitReadout:
        dt_ms = float(dt_ms)
        if not np.isfinite(dt_ms) or dt_ms <= 0.0:
            dt_ms = float(self._resolution_ms)

        steps = int(max(1, round(float(dt_ms) / float(self._resolution_ms))))

        drive_values = self._map_inputs_to_drive(inputs or {})
        if self._dopamine_input_gain > 0.0 and isinstance(neuromodulators, dict):
            raw = neuromodulators.get("dopamine")
            if raw is None:
                raw = neuromodulators.get("dopamine_level")
            try:
                dopamine = float(raw) if raw is not None else None
            except Exception:
                dopamine = None
            if dopamine is not None and np.isfinite(dopamine):
                scale = float(np.clip(1.0 + float(self._dopamine_input_gain) * (float(dopamine) - 1.0), 0.0, 5.0))
                drive_values = [float(v) * float(scale) for v in drive_values]

        self._drive_values = drive_values

        sim = getattr(self, "_simulator", None)
        if sim is None:
            return MicrocircuitReadout(
                activation=0.0,
                rate_hz=0.0,
                rate_hz_smooth=float(self._rate_smooth),
                region_rates_hz={name: 0.0 for name in self._regions},
                state={"framework": "loihi", "spike_count": 0, "region_spike_counts": {}},
            )

        with self._global_lock:
            runner = getattr(sim, "run_steps", None)
            if callable(runner):
                sim.run_steps(int(steps))
            else:
                for _ in range(int(steps)):
                    sim.step()

            spike_total = 0
            region_counts: Dict[str, int] = {}

            # Probes should share the same time axis; advance using the minimum available
            # index across probes we can successfully read.
            available_lengths: list[int] = []
            for probe in list(self._probe_exc) + list(self._probe_inh):
                if probe is None:
                    continue
                try:
                    arr = sim.data[probe]
                    available_lengths.append(int(len(arr)))
                except Exception:
                    continue

            start = int(self._last_sample_idx)
            end = int(min(available_lengths)) if available_lengths else int(self._last_sample_idx)
            if end < start:
                start = end

            for ridx, name in enumerate(self._regions):
                count = 0

                probe_exc = self._probe_exc[ridx] if ridx < len(self._probe_exc) else None
                if probe_exc is not None:
                    try:
                        arr = np.asarray(sim.data[probe_exc])
                        count += int(np.sum(arr[start:end]))
                    except Exception:
                        pass

                probe_inh = self._probe_inh[ridx] if ridx < len(self._probe_inh) else None
                if probe_inh is not None:
                    try:
                        arr = np.asarray(sim.data[probe_inh])
                        count += int(np.sum(arr[start:end]))
                    except Exception:
                        pass

                region_counts[str(name)] = int(max(0, count))
                spike_total += int(max(0, count))

            self._last_sample_idx = int(end)

        dt_s = float(dt_ms) / 1000.0
        total_neurons = int(self._neurons_per_region) * int(len(self._regions))
        rate_hz = float(spike_total) / max(float(total_neurons) * max(float(dt_s), 1e-9), 1e-9)

        region_rates: Dict[str, float] = {}
        for name in self._regions:
            region_rates[str(name)] = float(region_counts.get(str(name), 0)) / max(
                float(self._neurons_per_region) * max(float(dt_s), 1e-9), 1e-9
            )

        alpha = float(np.clip(float(dt_ms) / float(self._smooth_tau_ms), 0.0, 1.0))
        self._rate_smooth = (1.0 - alpha) * float(self._rate_smooth) + alpha * float(rate_hz)

        activation = 1.0 - float(np.exp(-float(self._rate_smooth) / float(self._target_rate_hz)))
        activation = float(np.clip(activation, 0.0, 1.0))

        readout = MicrocircuitReadout(
            activation=activation,
            rate_hz=float(rate_hz),
            rate_hz_smooth=float(self._rate_smooth),
            region_rates_hz=region_rates,
            state={
                "framework": "loihi",
                "spike_count": int(spike_total),
                "region_spike_counts": dict(region_counts),
                "target": str(self._target),
                "resolution_ms": float(self._resolution_ms),
                "steps": int(steps),
            },
        )
        self._last_readout = readout
        return readout


class _NeuronUnavailable(RuntimeError):
    pass


class NeuronSimulationEngine:
    """Process-global NEURON simulation engine (single kernel).

    NEURON uses a process-global `h` state (global time, CVode, dt). To avoid
    multi-stepping when multiple regions use NEURON microcircuits, stepping is
    centralized here and `NeuronMicrocircuit` only queues inputs and consumes
    readouts.
    """

    _instance: Optional["NeuronSimulationEngine"] = None
    _instance_lock = threading.Lock()

    def __init__(self, *, dt_ms: float = 0.025, v_init: float = -65.0, celsius: float = 34.0) -> None:
        try:
            from neuron import h  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise _NeuronUnavailable(
                "NEURON is not available. Install the Python package 'neuron' to use the 'neuron' microcircuit model."
            ) from exc

        self.h = h
        self.microcircuits: list["NeuronMicrocircuit"] = []

        try:
            self.h.load_file("stdrun.hoc")
        except Exception:
            pass

        dt_ms = float(dt_ms)
        if not np.isfinite(dt_ms) or dt_ms <= 0.0:
            dt_ms = 0.025
        self._dt_ms = float(dt_ms)

        try:
            self.h.cvode.active(0)
        except Exception:
            pass

        try:
            self.h.dt = float(self._dt_ms)
            self.h.steps_per_ms = float(1.0 / float(self._dt_ms))
        except Exception:
            pass

        try:
            self.h.celsius = float(celsius)
        except Exception:
            pass

        v_init = float(v_init)
        if not np.isfinite(v_init):
            v_init = -65.0
        self._v_init = float(v_init)
        self._initialized = False

    @classmethod
    def get_or_create(cls, *, dt_ms: float = 0.025, v_init: float = -65.0, celsius: float = 34.0) -> "NeuronSimulationEngine":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = NeuronSimulationEngine(dt_ms=float(dt_ms), v_init=float(v_init), celsius=float(celsius))
            return cls._instance

    def register(self, microcircuit: "NeuronMicrocircuit") -> None:
        if microcircuit not in self.microcircuits:
            self.microcircuits.append(microcircuit)

    def step(self, dt_ms: float) -> Dict[str, Any]:
        dt_ms = float(dt_ms)
        if not np.isfinite(dt_ms) or dt_ms <= 0.0:
            raise ValueError("dt_ms must be positive")

        with self._instance_lock:
            try:
                self.h.dt = float(self._dt_ms)
                self.h.steps_per_ms = float(1.0 / float(self._dt_ms))
            except Exception:
                pass

            if not bool(self._initialized):
                try:
                    self.h.finitialize(float(self._v_init))
                except Exception:
                    pass
                self._initialized = True

            for micro in list(self.microcircuits):
                try:
                    micro._apply_pending_drive()
                except Exception:
                    continue

            target_t = float(getattr(self.h, "t", 0.0) or 0.0) + float(dt_ms)
            try:
                continuerun = getattr(self.h, "continuerun", None)
                if callable(continuerun):
                    continuerun(float(target_t))
                else:
                    self.h.tstop = float(target_t)
                    self.h.run()
            except Exception:
                try:
                    self.h.tstop = float(target_t)
                    self.h.run()
                except Exception:
                    pass

            summaries: list[Dict[str, Any]] = []
            total_spikes = 0
            for micro in list(self.microcircuits):
                try:
                    info = micro._collect_after_step(float(dt_ms)) or {}
                except Exception:
                    info = {}
                summaries.append(dict(info) if isinstance(info, dict) else {"state": info})
                try:
                    total_spikes += int(info.get("spike_count", 0) or 0)
                except Exception:
                    pass

            try:
                time_ms = float(getattr(self.h, "t", float("nan")))
            except Exception:
                time_ms = float("nan")

        return {
            "framework": "neuron",
            "time_ms": time_ms,
            "dt_ms": float(dt_ms),
            "microcircuits": summaries,
            "spike_count": int(total_spikes),
        }


class NeuronMicrocircuit(Microcircuit):
    """NEURON-backed microcircuit for multi-compartment HH-style cells (downscaled).

    This backend is intended for **high-detail** microcircuits (active dendrites / compartments),
    not whole-brain scale. Use NEST/SpiNNaker for large SNNs.
    """

    requires_global_step = True
    force_update_each_cycle = True

    def __init__(self, *, params: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = dict(cfg or {})
        self.params = dict(params or {})
        self._last_readout: Optional[MicrocircuitReadout] = None
        self._rate_smooth = 0.0

        neuron_cfg = self.cfg.get("neuron", {})
        if not isinstance(neuron_cfg, dict):
            neuron_cfg = {}

        try:
            dt_ms = float(neuron_cfg.get("dt_ms", neuron_cfg.get("resolution_ms", 0.025)))
        except Exception:
            dt_ms = 0.025
        try:
            v_init = float(neuron_cfg.get("v_init_mV", -65.0))
        except Exception:
            v_init = -65.0
        try:
            celsius = float(neuron_cfg.get("celsius", 34.0))
        except Exception:
            celsius = 34.0
        try:
            self._spike_threshold_mV = float(neuron_cfg.get("spike_threshold_mV", 0.0))
        except Exception:
            self._spike_threshold_mV = 0.0
        if not np.isfinite(self._spike_threshold_mV):
            self._spike_threshold_mV = 0.0

        self._engine = NeuronSimulationEngine.get_or_create(dt_ms=dt_ms, v_init=v_init, celsius=celsius)
        self.h = self._engine.h

        self._input_gain = float(self.cfg.get("input_gain", 25.0))
        self._target_rate_hz = float(self.cfg.get("target_rate_hz", 20.0))
        self._smooth_tau_ms = float(self.cfg.get("smooth_tau_ms", 50.0))
        if not np.isfinite(self._input_gain):
            self._input_gain = 25.0
        if not np.isfinite(self._target_rate_hz) or self._target_rate_hz <= 0.0:
            self._target_rate_hz = 20.0
        if not np.isfinite(self._smooth_tau_ms) or self._smooth_tau_ms <= 0.0:
            self._smooth_tau_ms = 50.0

        self._regions: list[str] = [str(r) for r in (self.params.get("regions") or [])]
        if not self._regions:
            self._regions = ["REGION"]

        try:
            self._neurons_per_region = int(self.params.get("neurons_per_region", 20))
        except Exception:
            self._neurons_per_region = 20
        if self._neurons_per_region <= 0:
            self._neurons_per_region = 20

        try:
            self._excitatory_ratio = float(self.params.get("excitatory_ratio", 0.8))
        except Exception:
            self._excitatory_ratio = 0.8
        self._excitatory_ratio = float(np.clip(self._excitatory_ratio, 0.0, 1.0))

        self._neuron_model = str(self.params.get("neuron_model", "hh") or "hh").strip().lower()

        self._connectome_weights = np.asarray(self.params.get("connectome_weights", []), dtype=np.float32)
        self._connectome_delays_ms = np.asarray(self.params.get("connectome_delays_ms", []), dtype=np.float32)

        try:
            self._intra_p = float(self.params.get("intra_connection_prob", 0.05))
        except Exception:
            self._intra_p = 0.05
        try:
            self._inter_p = float(self.params.get("inter_connection_prob", 0.02))
        except Exception:
            self._inter_p = 0.02
        if not np.isfinite(self._intra_p) or self._intra_p < 0.0:
            self._intra_p = 0.0
        if not np.isfinite(self._inter_p) or self._inter_p < 0.0:
            self._inter_p = 0.0
        self._intra_p = float(np.clip(self._intra_p, 0.0, 1.0))
        self._inter_p = float(np.clip(self._inter_p, 0.0, 1.0))

        try:
            self._base_delay_ms = float(self.params.get("synaptic_delay_ms", 1.0))
        except Exception:
            self._base_delay_ms = 1.0
        if not np.isfinite(self._base_delay_ms) or self._base_delay_ms <= 0.0:
            self._base_delay_ms = 1.0

        try:
            self._w_exc = float(self.params.get("w_exc", 0.01))
        except Exception:
            self._w_exc = 0.01
        try:
            self._w_inh = float(self.params.get("w_inh", 0.02))
        except Exception:
            self._w_inh = 0.02
        if not np.isfinite(self._w_exc):
            self._w_exc = 0.01
        if not np.isfinite(self._w_inh):
            self._w_inh = 0.02

        try:
            seed = int(self.params.get("seed", neuron_cfg.get("seed", 0)) or 0)
        except Exception:
            seed = 0
        self._rng = np.random.default_rng(int(seed) if seed else None)

        self._cells_exc: list[list[Dict[str, Any]]] = []
        self._cells_inh: list[list[Dict[str, Any]]] = []
        self._netcons: list[Any] = []
        self._pending_drive: list[float] = [0.0 for _ in self._regions]
        self._last_spike_totals: list[int] = [0 for _ in self._regions]

        self._build_network()
        self.reset()
        self._engine.register(self)

    @property
    def engine(self) -> NeuronSimulationEngine:
        return self._engine

    def reset(self) -> None:
        self._last_readout = None
        self._rate_smooth = 0.0
        self._pending_drive = [0.0 for _ in self._regions]
        self._last_spike_totals = [0 for _ in self._regions]
        for ridx in range(len(self._regions)):
            try:
                self._last_spike_totals[ridx] = int(self._region_spike_total(ridx))
            except Exception:
                self._last_spike_totals[ridx] = 0

    def apply_control(self, control: Dict[str, Any]) -> Dict[str, Any]:
        return _apply_common_microcircuit_control(self, control or {})

    def prepare_step_inputs(self, dt_ms: float, inputs: Dict[str, float], neuromodulators: Dict[str, float]) -> None:
        del dt_ms
        del neuromodulators
        self._pending_drive = self._map_inputs_to_drive(inputs or {})

    def _apply_pending_drive(self) -> None:
        for ridx in range(len(self._regions)):
            amplitude = float(self._pending_drive[ridx]) if ridx < len(self._pending_drive) else 0.0
            if not np.isfinite(amplitude):
                amplitude = 0.0

            for cell in self._cells_exc[ridx]:
                try:
                    cell["iclamp"].amp = float(amplitude)
                except Exception:
                    continue
            for cell in self._cells_inh[ridx]:
                try:
                    cell["iclamp"].amp = float(amplitude)
                except Exception:
                    continue

    def _region_spike_total(self, ridx: int) -> int:
        total = 0
        for cell in self._cells_exc[ridx]:
            try:
                total += int(len(cell["spike_vec"]))
            except Exception:
                continue
        for cell in self._cells_inh[ridx]:
            try:
                total += int(len(cell["spike_vec"]))
            except Exception:
                continue
        return int(total)

    def _collect_after_step(self, dt_ms: float) -> Dict[str, Any]:
        dt_s = float(dt_ms) / 1000.0
        spike_total = 0
        region_counts: Dict[str, int] = {}

        for ridx, name in enumerate(self._regions):
            try:
                total = int(self._region_spike_total(ridx))
            except Exception:
                total = 0
            last = int(self._last_spike_totals[ridx]) if ridx < len(self._last_spike_totals) else 0
            new_count = max(0, int(total) - int(last))
            if ridx < len(self._last_spike_totals):
                self._last_spike_totals[ridx] = int(total)
            region_counts[str(name)] = int(new_count)
            spike_total += int(new_count)

        region_rates: Dict[str, float] = {}
        for name in self._regions:
            region_rates[str(name)] = float(region_counts.get(str(name), 0)) / max(
                float(self._neurons_per_region) * max(dt_s, 1e-9), 1e-9
            )

        total_neurons = int(self._neurons_per_region) * int(len(self._regions))
        rate_hz = float(spike_total) / max(float(total_neurons) * max(float(dt_s), 1e-9), 1e-9)

        alpha = float(np.clip(float(dt_ms) / float(self._smooth_tau_ms), 0.0, 1.0))
        self._rate_smooth = (1.0 - alpha) * float(self._rate_smooth) + alpha * float(rate_hz)

        activation = 1.0 - float(np.exp(-float(self._rate_smooth) / float(self._target_rate_hz)))
        activation = float(np.clip(activation, 0.0, 1.0))

        readout = MicrocircuitReadout(
            activation=activation,
            rate_hz=float(rate_hz),
            rate_hz_smooth=float(self._rate_smooth),
            region_rates_hz=region_rates,
            state={
                "framework": "neuron",
                "spike_count": int(spike_total),
                "region_spike_counts": dict(region_counts),
            },
        )
        self._last_readout = readout
        return readout.state

    def step(self, dt_ms: float, inputs: Dict[str, float], neuromodulators: Dict[str, float]) -> MicrocircuitReadout:
        self.prepare_step_inputs(dt_ms, inputs or {}, neuromodulators or {})
        if not bool(self.cfg.get("external_step", True)):
            self._engine.step(float(dt_ms))

        if self._last_readout is None:
            return MicrocircuitReadout(
                activation=0.0,
                rate_hz=0.0,
                rate_hz_smooth=float(self._rate_smooth),
                region_rates_hz={name: 0.0 for name in self._regions},
                state={"framework": "neuron", "spike_count": 0, "region_spike_counts": {}},
            )
        return self._last_readout

    def _map_inputs_to_drive(self, inputs: Dict[str, float]) -> list[float]:
        total = 0.0
        if isinstance(inputs, dict):
            for _, value in inputs.items():
                try:
                    total += float(value)
                except Exception:
                    continue

        drive = float(total) * float(self._input_gain)
        if not np.isfinite(drive):
            drive = 0.0

        lowered = {str(k).strip().lower(): v for k, v in (inputs or {}).items()}
        per_region: list[float] = []
        direct_hits = 0
        for name in self._regions:
            key = str(name).strip().lower()
            if key in lowered:
                try:
                    per_region.append(float(lowered[key]) * float(self._input_gain))
                    direct_hits += 1
                except Exception:
                    per_region.append(drive)
            else:
                per_region.append(drive)

        if direct_hits:
            return per_region
        return [drive for _ in self._regions]

    def _make_synapse(self, sec, *, inhibitory: bool) -> Any:
        try:
            syn = self.h.Exp2Syn(sec(0.5))
            if inhibitory:
                syn.e = -75.0
                syn.tau1 = 0.5
                syn.tau2 = 7.0
            else:
                syn.e = 0.0
                syn.tau1 = 0.5
                syn.tau2 = 3.0
            return syn
        except Exception:
            syn = self.h.ExpSyn(sec(0.5))
            if inhibitory:
                try:
                    syn.e = -75.0
                except Exception:
                    pass
                try:
                    syn.tau = 7.0
                except Exception:
                    pass
            else:
                try:
                    syn.e = 0.0
                except Exception:
                    pass
                try:
                    syn.tau = 3.0
                except Exception:
                    pass
            return syn

    def _make_cell(self, *, label: str, multi_compartment: bool) -> Dict[str, Any]:
        soma = self.h.Section(name=f"{label}_soma")
        soma.L = 20.0
        soma.diam = 20.0
        try:
            soma.insert("hh")
        except Exception:
            pass

        dend = None
        if multi_compartment:
            dend = self.h.Section(name=f"{label}_dend")
            dend.L = 200.0
            dend.diam = 2.0
            try:
                dend.connect(soma(1.0))
            except Exception:
                pass
            try:
                dend.insert("hh")
            except Exception:
                try:
                    dend.insert("pas")
                except Exception:
                    pass

        iclamp = self.h.IClamp(soma(0.5))
        iclamp.delay = 0.0
        iclamp.dur = 1e9
        iclamp.amp = 0.0

        spike_vec = self.h.Vector()
        nc = self.h.NetCon(soma(0.5)._ref_v, None, sec=soma)
        nc.threshold = float(self._spike_threshold_mV)
        nc.record(spike_vec)

        syn_exc = self._make_synapse(soma, inhibitory=False)
        syn_inh = self._make_synapse(soma, inhibitory=True)

        return {
            "soma": soma,
            "dend": dend,
            "iclamp": iclamp,
            "spike_vec": spike_vec,
            "spike_detector": nc,
            "syn_exc": syn_exc,
            "syn_inh": syn_inh,
        }

    def _connect_projection(self, pre_cells: list[Dict[str, Any]], post_cells: list[Dict[str, Any]], *, weight: float, delay_ms: float, inhibitory: bool, p: float) -> None:
        if not pre_cells or not post_cells or p <= 0.0:
            return

        weight = float(weight)
        delay_ms = float(delay_ms)
        if not np.isfinite(weight):
            weight = 0.0
        if not np.isfinite(delay_ms) or delay_ms <= 0.0:
            delay_ms = float(self._base_delay_ms)
        if abs(weight) < 1e-12:
            return

        mask = self._rng.random((len(post_cells), len(pre_cells))) < float(p)
        post_idx, pre_idx = np.where(mask)
        for j, i in zip(post_idx.tolist(), pre_idx.tolist()):
            pre = pre_cells[int(i)]
            post = post_cells[int(j)]
            syn = post["syn_inh"] if inhibitory else post["syn_exc"]
            try:
                nc = self.h.NetCon(pre["soma"](0.5)._ref_v, syn, sec=pre["soma"])
                nc.threshold = float(self._spike_threshold_mV)
                nc.delay = float(delay_ms)
                nc.weight[0] = float(abs(weight))
                self._netcons.append(nc)
            except Exception:
                continue

    def _build_network(self) -> None:
        multi_compartment = self._neuron_model in {"mc", "multi_compartment", "multicompartment", "morphology"}

        for ridx, name in enumerate(self._regions):
            n_total = int(self._neurons_per_region)
            n_exc = int(round(float(n_total) * float(self._excitatory_ratio)))
            n_exc = int(np.clip(n_exc, 1, max(1, n_total)))
            n_inh = max(0, int(n_total) - int(n_exc))

            exc_cells = [self._make_cell(label=f"{name}_exc_{i}", multi_compartment=multi_compartment) for i in range(n_exc)]
            inh_cells = [self._make_cell(label=f"{name}_inh_{i}", multi_compartment=multi_compartment) for i in range(n_inh)]
            self._cells_exc.append(exc_cells)
            self._cells_inh.append(inh_cells)

            # Intra-region connectivity.
            self._connect_projection(exc_cells, exc_cells, weight=float(self._w_exc), delay_ms=float(self._base_delay_ms), inhibitory=False, p=float(self._intra_p))
            if inh_cells:
                self._connect_projection(exc_cells, inh_cells, weight=float(self._w_exc), delay_ms=float(self._base_delay_ms), inhibitory=False, p=float(self._intra_p))
                self._connect_projection(inh_cells, exc_cells, weight=float(self._w_inh), delay_ms=float(self._base_delay_ms), inhibitory=True, p=float(self._intra_p))
                self._connect_projection(inh_cells, inh_cells, weight=float(self._w_inh), delay_ms=float(self._base_delay_ms), inhibitory=True, p=float(self._intra_p))

        # Inter-region connectivity via connectome matrices (optional).
        if self._connectome_weights.ndim == 2 and self._connectome_weights.shape[0] == len(self._regions):
            for src in range(len(self._regions)):
                for dst in range(len(self._regions)):
                    if src == dst:
                        continue
                    w = float(self._connectome_weights[src, dst])
                    if abs(w) < 1e-9:
                        continue

                    delay = float(self._base_delay_ms)
                    if (
                        self._connectome_delays_ms.ndim == 2
                        and self._connectome_delays_ms.shape == self._connectome_weights.shape
                    ):
                        try:
                            delay = float(self._connectome_delays_ms[src, dst])
                        except Exception:
                            delay = float(self._base_delay_ms)
                    if not np.isfinite(delay) or delay <= 0.0:
                        delay = float(self._base_delay_ms)

                    if w >= 0.0:
                        pre = self._cells_exc[src]
                        inhibitory = False
                        weight = float(self._w_exc) * float(w)
                    else:
                        pre = self._cells_inh[src]
                        inhibitory = True
                        weight = float(self._w_inh) * float(abs(w))

                    if not pre:
                        continue

                    self._connect_projection(pre, self._cells_exc[dst], weight=weight, delay_ms=delay, inhibitory=inhibitory, p=float(self._inter_p))
                    if self._cells_inh[dst]:
                        self._connect_projection(pre, self._cells_inh[dst], weight=weight, delay_ms=delay, inhibitory=inhibitory, p=float(self._inter_p))

def _hippocampus_dg_ca3_ca1_params() -> Dict[str, Any]:
    weights = np.zeros((3, 3), dtype=np.float32)
    delays = np.full((3, 3), 6.0, dtype=np.float32)
    np.fill_diagonal(weights, 0.0)

    # Canonical trisynaptic loop (downscaled): DG -> CA3 -> CA1 -> (weak) DG feedback.
    weights[0, 1] = 1.0  # DG -> CA3
    weights[1, 2] = 1.0  # CA3 -> CA1
    weights[2, 0] = 0.25  # CA1 -> DG (feedback)
    delays[0, 1] = 5.0
    delays[1, 2] = 5.0
    delays[2, 0] = 8.0

    return {
        "regions": ["DG", "CA3", "CA1"],
        "connectome_weights": weights,
        "connectome_delays_ms": delays,
        "neurons_per_region": 60,
        "neuron_model": "hybrid",
        "synapse_model": "receptor",
        "stp_enabled": True,
        "cell_types_enabled": True,
        "baseline_current_mean": 2.0,
        "baseline_current_std": 1.0,
        "noise_std": 0.5,
        "intra_connection_prob": 0.08,
        "inter_connection_prob": 0.03,
        "max_delay_ms": 25.0,
    }


def _retina_lgn_v1_params() -> Dict[str, Any]:
    """Small feedforward sensory pathway scaffold: Retina -> LGN -> V1 (downscaled)."""

    weights = np.zeros((3, 3), dtype=np.float32)
    delays = np.full((3, 3), 4.0, dtype=np.float32)
    np.fill_diagonal(weights, 0.0)

    # Feedforward pathway with weak corticothalamic feedback.
    weights[0, 1] = 1.0  # Retina -> LGN
    weights[1, 2] = 1.0  # LGN -> V1
    weights[2, 1] = 0.25  # V1 -> LGN (feedback)

    delays[0, 1] = 2.0
    delays[1, 2] = 4.0
    delays[2, 1] = 6.0

    return {
        "regions": ["RETINA", "LGN", "V1"],
        "connectome_weights": weights,
        "connectome_delays_ms": delays,
        "neurons_per_region": 60,
        "neuron_model": "hybrid",
        "synapse_model": "receptor",
        "stp_enabled": True,
        "cell_types_enabled": True,
        "baseline_current_mean": 1.5,
        "baseline_current_std": 0.75,
        "noise_std": 0.5,
        "intra_connection_prob": 0.05,
        "inter_connection_prob": 0.03,
        "max_delay_ms": 25.0,
    }


def create_microcircuit_for_region(region_type: BrainRegion, micro_cfg: Dict[str, Any]) -> Optional[Microcircuit]:
    if not (isinstance(micro_cfg, dict) and bool(micro_cfg.get("enabled", False))):
        return None

    shadow_cfg = micro_cfg.get("shadow")
    shadow_compare = micro_cfg.get("shadow_compare")
    if isinstance(shadow_cfg, dict):
        primary_cfg = dict(micro_cfg)
        primary_cfg.pop("shadow", None)
        primary_cfg.pop("shadow_compare", None)

        primary = create_microcircuit_for_region(region_type, primary_cfg)
        if primary is None:
            return None

        shadow_cfg = dict(shadow_cfg)
        shadow_cfg.pop("shadow", None)
        shadow_cfg.pop("shadow_compare", None)
        shadow_cfg.setdefault("enabled", True)
        shadow_cfg.setdefault("model", primary_cfg.get("model", "biophysical"))
        shadow_cfg.setdefault("preset", primary_cfg.get("preset", "auto"))
        if "params" not in shadow_cfg and isinstance(primary_cfg.get("params"), dict):
            shadow_cfg["params"] = dict(primary_cfg.get("params") or {})
        if "cfg" not in shadow_cfg and isinstance(primary_cfg.get("cfg"), dict):
            shadow_cfg["cfg"] = dict(primary_cfg.get("cfg") or {})

        shadow = None
        shadow_error = None
        try:
            shadow = create_microcircuit_for_region(region_type, shadow_cfg)
        except Exception as exc:
            shadow_error = str(exc)
            shadow = None

        if shadow is None:
            if shadow_error:
                try:
                    info = dict(getattr(primary, "cfg", {}) or {}) if hasattr(primary, "cfg") else {}
                    info.setdefault("shadow_error", shadow_error)
                    setattr(primary, "cfg", info)
                except Exception:
                    pass
            return primary

        compare_cfg = shadow_compare
        if compare_cfg is None:
            compare_cfg = shadow_cfg.get("compare")
        return ShadowMicrocircuit(primary, shadow, compare=compare_cfg if isinstance(compare_cfg, dict) else None)

    model = str(micro_cfg.get("model", "biophysical") or "biophysical").strip().lower()
    preset = str(micro_cfg.get("preset", "auto") or "auto").strip().lower()
    params = dict(micro_cfg.get("params") or {}) if isinstance(micro_cfg.get("params"), dict) else {}
    cfg = dict(micro_cfg.get("cfg") or {}) if isinstance(micro_cfg.get("cfg"), dict) else {}

    if model not in {"biophysical", "nest", "brian2", "spinnaker", "loihi", "neuron", "pynn"}:
        raise ValueError(f"Unsupported microcircuit model: {model}")

    if preset == "auto":
        if region_type == BrainRegion.HIPPOCAMPUS:
            preset = "hippocampus_dg_ca3_ca1"
        else:
            preset = "single_region_spiking"

    if preset == "hippocampus_dg_ca3_ca1":
        base = _hippocampus_dg_ca3_ca1_params()
        base.update(params)
        if model == "nest":
            return NestMicrocircuit(params=base, cfg=cfg)
        if model == "brian2":
            return Brian2Microcircuit(params=base, cfg=cfg)
        if model == "spinnaker":
            return SpiNNakerMicrocircuit(params=base, cfg=cfg)
        if model == "loihi":
            return LoihiMicrocircuit(params=base, cfg=cfg)
        if model == "neuron":
            return NeuronMicrocircuit(params=base, cfg=cfg)
        if model == "pynn":
            return PyNNMicrocircuit(params=base, cfg=cfg)
        return BiophysicalMicrocircuit(params=base, cfg=cfg)

    if preset == "retina_lgn_v1":
        base = _retina_lgn_v1_params()
        base.update(params)
        if model == "nest":
            return NestMicrocircuit(params=base, cfg=cfg)
        if model == "brian2":
            return Brian2Microcircuit(params=base, cfg=cfg)
        if model == "spinnaker":
            return SpiNNakerMicrocircuit(params=base, cfg=cfg)
        if model == "loihi":
            return LoihiMicrocircuit(params=base, cfg=cfg)
        if model == "neuron":
            return NeuronMicrocircuit(params=base, cfg=cfg)
        if model == "pynn":
            return PyNNMicrocircuit(params=base, cfg=cfg)
        return BiophysicalMicrocircuit(params=base, cfg=cfg)

    # Default: one-region microcircuit matching the high-level region name.
    if "regions" not in params:
        params["regions"] = [str(region_type.value).upper()]
    params.setdefault("neurons_per_region", 80)
    params.setdefault("neuron_model", "hybrid")
    params.setdefault("synapse_model", "receptor")
    params.setdefault("stp_enabled", True)
    params.setdefault("cell_types_enabled", True)
    params.setdefault("baseline_current_mean", 2.0)
    params.setdefault("baseline_current_std", 1.0)
    params.setdefault("noise_std", 0.5)
    if model == "nest":
        return NestMicrocircuit(params=params, cfg=cfg)
    if model == "brian2":
        return Brian2Microcircuit(params=params, cfg=cfg)
    if model == "spinnaker":
        return SpiNNakerMicrocircuit(params=params, cfg=cfg)
    if model == "loihi":
        return LoihiMicrocircuit(params=params, cfg=cfg)
    if model == "neuron":
        return NeuronMicrocircuit(params=params, cfg=cfg)
    if model == "pynn":
        return PyNNMicrocircuit(params=params, cfg=cfg)
    return BiophysicalMicrocircuit(params=params, cfg=cfg)


__all__ = [
    "Microcircuit",
    "MicrocircuitReadout",
    "ShadowMicrocircuit",
    "BiophysicalMicrocircuit",
    "NestMicrocircuit",
    "NestSimulationEngine",
    "PyNNMicrocircuit",
    "PyNNSimulationEngine",
    "SpiNNakerMicrocircuit",
    "SpiNNakerSimulationEngine",
    "Brian2Microcircuit",
    "LoihiMicrocircuit",
    "NeuronMicrocircuit",
    "NeuronSimulationEngine",
    "create_microcircuit_for_region",
]
