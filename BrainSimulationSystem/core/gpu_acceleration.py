"""
GPU/Torch acceleration utilities for neuron updates.

Provides a configurable accelerator that batches compatible neuron
updates (currently Izhikevich-style EnhancedNeurons) onto torch tensors
so they can execute on CPU or CUDA devices with vectorised kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import torch

    TORCH_AVAILABLE = bool(
        hasattr(torch, "tensor")
        and hasattr(torch, "device")
        and hasattr(torch, "float32")
        and hasattr(torch, "float16")
        and hasattr(torch, "cuda")
        and hasattr(torch.cuda, "is_available")
    )
    if not TORCH_AVAILABLE:  # pragma: no cover - defensive for stubs
        torch = None  # type: ignore[assignment]
except Exception:  # pragma: no cover - torch is optional
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

_GLOBAL_POLICY: Dict[str, Any] = {"enabled": False}
_GLOBAL_ACCELERATOR: Optional["TorchNeuronAccelerator"] = None


def configure_gpu_acceleration(policy: Optional[Dict[str, Any]]) -> None:
    """Store global acceleration policy and instantiate accelerator."""

    global _GLOBAL_POLICY, _GLOBAL_ACCELERATOR

    _GLOBAL_POLICY = {"enabled": False}
    if policy:
        _GLOBAL_POLICY.update(policy)

    if TORCH_AVAILABLE and bool(_GLOBAL_POLICY.get("enabled", False)):
        _GLOBAL_ACCELERATOR = TorchNeuronAccelerator(_GLOBAL_POLICY)
    else:
        _GLOBAL_ACCELERATOR = None


def get_gpu_accelerator() -> Optional["TorchNeuronAccelerator"]:
    """Return the globally configured accelerator instance (if any)."""

    return _GLOBAL_ACCELERATOR


@dataclass
class _BatchEntry:
    cell_id: int
    neuron: Any
    inputs: Dict[str, Any]


class TorchNeuronAccelerator:
    """Batched neuron updater leveraging torch for vectorised math."""

    def __init__(self, policy: Dict[str, Any]):
        if not TORCH_AVAILABLE:  # pragma: no cover - guarded earlier
            raise RuntimeError("Torch is required for TorchNeuronAccelerator")

        self._policy = dict(policy)
        self._dtype = torch.float16 if self._policy.get("precision") == "fp16" else torch.float32
        preferred_device = self._policy.get("device")
        use_cuda = bool(self._policy.get("use_cuda", True))
        if preferred_device is not None:
            self._device = torch.device(preferred_device)
        elif use_cuda and torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")
        self.available = True

    # --------------------------------------------------------------------- API
    def can_accelerate(self, neuron: Any, inputs: Optional[Dict[str, Any]] = None) -> bool:
        """Return True when the neuron/update pair can be batched."""

        if not self.available:
            return False

        model_type = getattr(neuron, "model_type", "").lower()
        return model_type == "izhikevich" and hasattr(neuron, "izhikevich_params")

    def run(
        self,
        entries: Sequence[Tuple[int, Any, Dict[str, Any]]],
        dt: float,
    ) -> Dict[int, Dict[str, Any]]:
        """Execute batched updates and return per-cell results."""

        if not entries:
            return {}

        izh_entries: List[_BatchEntry] = []
        fallback: List[_BatchEntry] = []
        for cell_id, neuron, inputs in entries:
            entry = _BatchEntry(cell_id, neuron, inputs)
            if self.can_accelerate(neuron, inputs):
                izh_entries.append(entry)
            else:
                fallback.append(entry)

        results: Dict[int, Dict[str, Any]] = {}
        if izh_entries:
            results.update(self._run_izhikevich_batch(izh_entries, dt))

        # Neurons that couldn't be batched fall back to their scalar update
        for entry in fallback:
            results[entry.cell_id] = entry.neuron.update(dt, entry.inputs)

        return results

    # ----------------------------------------------------------------- helpers
    def _run_izhikevich_batch(self, entries: Sequence[_BatchEntry], dt: float) -> Dict[int, Dict[str, Any]]:
        """Vectorised Izhikevich integration."""

        device = self._device
        dtype = self._dtype
        n = len(entries)

        dt_value = float(dt)
        dt_tensor = torch.tensor(dt_value, device=device, dtype=dtype)
        sub_steps = max(1, int((dt_value + 0.4999) // 0.5))  # ceil without math import
        sub_dt = dt_value / sub_steps
        sub_dt_tensor = torch.tensor(sub_dt, device=device, dtype=dtype)

        neurons = [entry.neuron for entry in entries]
        inputs = [entry.inputs for entry in entries]

        def _tensor(seq: Iterable[float], clamp_min: Optional[float] = None) -> torch.Tensor:
            tens = torch.tensor(list(seq), device=device, dtype=dtype)
            if clamp_min is not None:
                tens = torch.clamp(tens, min=clamp_min)
            return tens

        v = _tensor(neuron.membrane_potential for neuron in neurons)
        u = _tensor(neuron.adaptation_current for neuron in neurons)
        dendritic = _tensor(neuron.dendritic_potential for neuron in neurons)
        calcium = _tensor(neuron.calcium_concentration for neuron in neurons)
        refractory = _tensor(getattr(neuron, "refractory_timer", 0.0) for neuron in neurons)
        resting = _tensor(neuron.parameters.resting_potential for neuron in neurons)
        tau_d = _tensor((getattr(neuron, "dendritic_time_constant", 25.0) for neuron in neurons), clamp_min=1e-6)
        cap_d = _tensor((getattr(neuron, "dendritic_capacitance", 150.0) for neuron in neurons), clamp_min=1e-6)
        axial = _tensor((getattr(neuron, "axial_conductance", 5.0) for neuron in neurons))
        soma_share = torch.clamp(
            _tensor((entry.neuron.model_config.get("soma_synaptic_share", 0.5) for entry in entries)),
            0.0,
            1.0,
        )
        dendritic_share = torch.clamp(
            _tensor((entry.neuron.model_config.get("dendritic_synaptic_share", 1.0) for entry in entries)),
            0.0,
            1.0,
        )

        synaptic = _tensor((inp.get("synaptic_current", 0.0) for inp in inputs))
        external = _tensor((inp.get("external_current", 0.0) for inp in inputs))
        noise = _tensor((inp.get("noise", 0.0) for inp in inputs))
        dendritic_drive = _tensor((inp.get("dendritic_current", 0.0) for inp in inputs))
        times = _tensor((inp.get("time", 0.0) for inp in inputs))

        total_input = synaptic + external + noise
        syn_component = synaptic + noise
        external_component = total_input - syn_component

        # Dendritic compartment update
        dendritic_input = dendritic_share * syn_component + dendritic_drive
        relaxation = -(dendritic - resting) / tau_d
        drive = dendritic_input / cap_d
        dendritic = dendritic + (relaxation + drive) * dt_tensor
        coupling = axial * (dendritic - v)

        effective_current = external_component + syn_component * soma_share + coupling

        params = [entry.neuron.izhikevich_params for entry in entries]
        a = _tensor((param.get("a", 0.02) for param in params))
        b = _tensor((param.get("b", 0.2) for param in params))
        c = _tensor((param.get("c", -65.0) for param in params))
        d = _tensor((param.get("d", 6.0) for param in params))
        v_peak = _tensor((param.get("v_peak", 30.0) for param in params))

        spiked = torch.zeros(n, dtype=torch.bool, device=device)
        for _ in range(sub_steps):
            dv = 0.04 * v * v + 5.0 * v + 140.0 - u + effective_current
            du = a * (b * v - u)
            v = v + dv * sub_dt_tensor
            u = u + du * sub_dt_tensor
            emit = v >= v_peak
            if emit.any():
                spiked |= emit
                v = torch.where(emit, c, v)
                u = torch.where(emit, u + d, u)

        # Calcium dynamics (simple spike-triggered influx + decay)
        calcium_tau = _tensor((entry.neuron.model_config.get("calcium_tau", 60.0) for entry in entries), clamp_min=1.0)
        spike_influx = torch.where(spiked, torch.full((n,), 0.8, device=device, dtype=dtype), torch.zeros(n, device=device, dtype=dtype))
        calcium = torch.clamp(calcium + dt_tensor * (spike_influx - calcium / calcium_tau), min=0.0)

        # Refractory handling
        refractory_period = _tensor((entry.neuron.parameters.refractory_period for entry in entries))
        refractory = torch.where(
            spiked,
            refractory_period,
            torch.clamp(refractory - dt_tensor, min=0.0),
        )

        # Move everything back to CPU scalars
        v_list = v.detach().cpu().tolist()
        u_list = u.detach().cpu().tolist()
        dend_list = dendritic.detach().cpu().tolist()
        calcium_list = calcium.detach().cpu().tolist()
        refractory_list = refractory.detach().cpu().tolist()
        effective_list = effective_current.detach().cpu().tolist()
        coupling_list = coupling.detach().cpu().tolist()
        total_input_list = total_input.detach().cpu().tolist()
        spiked_list = spiked.detach().cpu().tolist()
        times_list = times.detach().cpu().tolist()

        results: Dict[int, Dict[str, Any]] = {}
        for idx, entry in enumerate(entries):
            neuron = entry.neuron
            neuron.membrane_potential = float(v_list[idx])
            neuron.adaptation_current = float(u_list[idx])
            neuron.dendritic_potential = float(dend_list[idx])
            neuron.calcium_concentration = float(calcium_list[idx])
            neuron.refractory_timer = max(0.0, float(refractory_list[idx]))
            neuron._prev_voltage = neuron.membrane_potential  # type: ignore[attr-defined]
            neuron._last_effective_current = float(effective_list[idx])  # type: ignore[attr-defined]
            neuron._last_coupling_current = float(coupling_list[idx])  # type: ignore[attr-defined]
            neuron._last_stimulus_current = float(total_input_list[idx])  # type: ignore[attr-defined]

            spike_flag = bool(spiked_list[idx])
            if spike_flag:
                spike_time = float(times_list[idx])
                neuron.spike_times.append(spike_time)
                neuron.last_spike_time = spike_time

            results[entry.cell_id] = {
                "spike": spike_flag,
                "voltage": neuron.membrane_potential,
                "dendrite_voltage": neuron.dendritic_potential,
                "calcium": neuron.calcium_concentration,
                "adaptation": neuron.adaptation_current,
                "model": getattr(neuron, "model_type", "izhikevich"),
                "effective_current": neuron._last_effective_current,  # type: ignore[attr-defined]
            }

        return results


__all__ = [
    "configure_gpu_acceleration",
    "get_gpu_accelerator",
    "TorchNeuronAccelerator",
    "TORCH_AVAILABLE",
]

