"""Parameter/data mapping from cognitive module configs to microcircuit configs.

This module provides a small, opt-in bridge that translates *high-level* module
parameters (e.g., working-memory capacity, decision learning rate) into
downscaled spiking microcircuit configuration knobs (e.g., neurons per region,
STDP enablement, dopamine-gated plasticity gain).

The mapping is intentionally coarse and configurable because:
- The project is downscaled (neurons_per_region instead of true cell counts).
- Real calibration datasets (Allen atlas/connectome, task benchmarks) are not
  bundled and are expected to be provided by the user.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .enums import BrainRegion


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return float(out)


def _safe_int(value: Any) -> Optional[int]:
    try:
        out = int(value)
    except Exception:
        return None
    return int(out)


def _clip_int(value: float, lo: int, hi: int) -> int:
    try:
        return int(np.clip(int(round(float(value))), int(lo), int(hi)))
    except Exception:
        return int(lo)


def _parse_brain_region(value: Any) -> Optional[BrainRegion]:
    if isinstance(value, BrainRegion):
        return value
    raw = str(value).strip().lower().replace(" ", "_")
    if not raw:
        return None
    for region in BrainRegion:
        if raw == str(region.value).strip().lower() or raw == str(region.name).strip().lower():
            return region
    return None


def _parse_region_fraction_spec(spec: Any) -> List[Tuple[str, float]]:
    """Parse a region-spec entry into [(name, fraction)]."""

    out: List[Tuple[str, float]] = []
    if spec is None:
        return out

    if isinstance(spec, str):
        name = str(spec).strip()
        if name:
            out.append((name, 1.0))
        return out

    if isinstance(spec, dict):
        if "name" in spec:
            name = str(spec.get("name", "")).strip()
            frac = _safe_float(spec.get("fraction", 1.0))
            if frac is None:
                frac = 1.0
            if name and frac > 0.0:
                out.append((name, float(frac)))
            return out

        for key, value in spec.items():
            name = str(key).strip()
            frac = _safe_float(value)
            if frac is None:
                continue
            if name and frac > 0.0:
                out.append((name, float(frac)))
        return out

    if isinstance(spec, (list, tuple, set)):
        for entry in spec:
            out.extend(_parse_region_fraction_spec(entry))
        return out

    name = str(spec).strip()
    if name:
        out.append((name, 1.0))
    return out


def _default_atlas_region_map() -> Dict[BrainRegion, List[Tuple[str, float]]]:
    """Conservative mapping from coarse BrainRegion enums to BrainAtlas.default() labels."""

    return {
        BrainRegion.PREFRONTAL_CORTEX: [("Prefrontal Cortex", 1.0)],
        BrainRegion.MOTOR_CORTEX: [("Motor Cortex", 1.0)],
        BrainRegion.SOMATOSENSORY_CORTEX: [("Parietal Lobe", 0.2)],
        BrainRegion.VISUAL_CORTEX: [("Occipital Lobe", 1.0)],
        BrainRegion.AUDITORY_CORTEX: [("Temporal Lobe", 0.2)],
        BrainRegion.PARIETAL_CORTEX: [("Parietal Lobe", 0.8)],
        BrainRegion.TEMPORAL_CORTEX: [("Temporal Lobe", 0.8)],
        BrainRegion.OCCIPITAL_CORTEX: [("Occipital Lobe", 1.0)],
        BrainRegion.HIPPOCAMPUS: [("Hippocampus", 1.0)],
        BrainRegion.AMYGDALA: [("Amygdala", 1.0)],
        BrainRegion.THALAMUS: [("Thalamus", 1.0)],
        BrainRegion.BASAL_GANGLIA: [("Basal Ganglia", 1.0)],
        BrainRegion.CEREBELLUM: [("Cerebellum", 1.0)],
        BrainRegion.BRAINSTEM: [("Brainstem", 1.0)],
    }


def _load_json_file(path: Any) -> Dict[str, Any]:
    raw = str(path or "").strip()
    if not raw:
        raise ValueError("calibration.path is required when source='file'")
    file_path = Path(raw)
    if not file_path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Calibration JSON must be an object")
    return payload


def _parse_neurons_per_region_map(payload: Any) -> Dict[BrainRegion, int]:
    if not isinstance(payload, dict):
        return {}

    out: Dict[BrainRegion, int] = {}
    for key, value in payload.items():
        region = _parse_brain_region(key)
        if region is None:
            continue
        count = _safe_int(value)
        if count is None or count <= 0:
            continue
        out[region] = int(count)
    return out


def _parse_real_neuron_map(payload: Any) -> Dict[BrainRegion, float]:
    if not isinstance(payload, dict):
        return {}

    out: Dict[BrainRegion, float] = {}
    for key, value in payload.items():
        region = _parse_brain_region(key)
        if region is None:
            continue
        count = _safe_float(value)
        if count is None or count <= 0.0:
            continue
        out[region] = float(count)
    return out


def _downscale_real_neurons(
    real: Dict[BrainRegion, float],
    *,
    target_mean_neurons: float,
    scale_divisor: Optional[float],
    min_neurons: int,
    max_neurons: int,
) -> Tuple[Dict[BrainRegion, int], Dict[str, Any]]:
    values = [float(v) for v in real.values() if float(v) > 0.0]
    used_scale = scale_divisor
    if used_scale is None:
        mean_real = float(np.mean(values)) if values else 0.0
        if mean_real <= 0.0:
            used_scale = 1.0
        else:
            target = float(target_mean_neurons) if float(target_mean_neurons) > 0.0 else 120.0
            used_scale = mean_real / target

    used_scale = float(used_scale) if used_scale is not None else 1.0
    if not np.isfinite(used_scale) or used_scale <= 0.0:
        used_scale = 1.0

    out: Dict[BrainRegion, int] = {}
    for region, count in real.items():
        sim = _clip_int(float(count) / float(used_scale), int(min_neurons), int(max_neurons))
        out[region] = int(sim)

    meta = {
        "mode": "downscale",
        "scale_divisor": float(used_scale),
        "target_mean_neurons_per_region": float(target_mean_neurons),
        "min_neurons": int(min_neurons),
        "max_neurons": int(max_neurons),
    }
    return out, meta


def _derive_real_neurons_from_brain_atlas(
    region_map: Dict[BrainRegion, List[Tuple[str, float]]], *, logger: Optional[Any] = None
) -> Dict[BrainRegion, float]:
    try:
        from modules.brain.anatomy import BrainAtlas  # type: ignore
    except Exception as exc:  # pragma: no cover - optional repository dependency
        if logger is not None:
            try:
                logger.warning("BrainAtlas calibration unavailable: %s", exc)
            except Exception:
                pass
        return {}

    atlas = BrainAtlas.default()
    out: Dict[BrainRegion, float] = {}
    for region, specs in region_map.items():
        total = 0.0
        for name, frac in specs:
            try:
                entry = atlas.get(name)
            except Exception:
                entry = None
            if entry is None:
                continue
            try:
                total += float(entry.cell_count()) * float(frac)
            except Exception:
                continue
        if total > 0.0:
            out[region] = float(total)
    return out


def _normalise_microcircuit_cfg(micro: Any) -> Optional[Dict[str, Any]]:
    if micro is None:
        return {"enabled": True, "model": "biophysical", "preset": "auto", "params": {}, "cfg": {}}
    if isinstance(micro, bool):
        if not micro:
            return None
        return {"enabled": True, "model": "biophysical", "preset": "auto", "params": {}, "cfg": {}}
    if isinstance(micro, dict):
        out = dict(micro)
        out.setdefault("enabled", True)
        out.setdefault("model", "biophysical")
        out.setdefault("preset", "auto")
        out.setdefault("params", {})
        out.setdefault("cfg", {})
        if not isinstance(out.get("params"), dict):
            out["params"] = {}
        if not isinstance(out.get("cfg"), dict):
            out["cfg"] = {}
        return out if bool(out.get("enabled", False)) else None
    return {"enabled": True, "model": "biophysical", "preset": "auto", "params": {}, "cfg": {}}


@dataclass(frozen=True)
class ParameterMappingReport:
    enabled: bool
    applied: Dict[str, Dict[str, Any]]


class ModuleParameterMapper:
    """Translate module parameters into region microcircuit configuration."""

    def __init__(self, cfg: Dict[str, Any], *, logger: Optional[Any] = None) -> None:
        self.cfg = dict(cfg or {})
        self.logger = logger

        try:
            self.enabled = bool(self.cfg.get("enabled", False))
        except Exception:
            self.enabled = False

        self._calibration_cfg = self.cfg.get("calibration", {})
        if not isinstance(self._calibration_cfg, dict):
            self._calibration_cfg = {}

        self._calibration_neurons_per_region: Dict[BrainRegion, int] = {}
        self._calibration_meta: Dict[str, Any] = {}
        self._calibration_override_existing = False

        try:
            if bool(self._calibration_cfg.get("enabled", False)):
                self._calibration_override_existing = bool(self._calibration_cfg.get("override_existing", False))
                self._load_calibration()
        except Exception as exc:
            if self.logger is not None:
                try:
                    self.logger.warning("Calibration disabled: %s", exc)
                except Exception:
                    pass

        self._wm_cfg = self.cfg.get("working_memory", {})
        if not isinstance(self._wm_cfg, dict):
            self._wm_cfg = {}

        self._bg_cfg = self.cfg.get("basal_ganglia", {})
        if not isinstance(self._bg_cfg, dict):
            self._bg_cfg = {}

        self._visual_cfg = self.cfg.get("visual", {})
        if not isinstance(self._visual_cfg, dict):
            self._visual_cfg = {}

    def _load_calibration(self) -> None:
        cfg = self._calibration_cfg or {}
        source = str(cfg.get("source", "brain_atlas_default") or "brain_atlas_default").strip().lower()

        try:
            min_neurons = int(cfg.get("min_neurons", 20))
        except Exception:
            min_neurons = 20
        try:
            max_neurons = int(cfg.get("max_neurons", 800))
        except Exception:
            max_neurons = 800
        if min_neurons <= 0:
            min_neurons = 1
        if max_neurons < min_neurons:
            max_neurons = min_neurons

        target_mean = _safe_float(cfg.get("target_mean_neurons_per_region"))
        if target_mean is None or target_mean <= 0.0:
            target_mean = 120.0

        scale_divisor = _safe_float(cfg.get("scale_divisor"))
        if scale_divisor is not None and scale_divisor <= 0.0:
            scale_divisor = None

        if source in {"file", "json"}:
            payload = _load_json_file(cfg.get("path") or cfg.get("file"))
            neurons_direct = _parse_neurons_per_region_map(payload.get("neurons_per_region"))
            if neurons_direct:
                self._calibration_neurons_per_region = neurons_direct
                self._calibration_meta = {"source": "file", "mode": "neurons_per_region"}
                return

            real_neurons = _parse_real_neuron_map(payload.get("real_neurons"))
            if not real_neurons:
                neurons_direct = _parse_neurons_per_region_map(payload)
                if neurons_direct:
                    self._calibration_neurons_per_region = neurons_direct
                    self._calibration_meta = {"source": "file", "mode": "neurons_per_region"}
                    return
                real_neurons = _parse_real_neuron_map(payload)

            neurons, meta = _downscale_real_neurons(
                real_neurons,
                target_mean_neurons=float(target_mean),
                scale_divisor=scale_divisor,
                min_neurons=int(min_neurons),
                max_neurons=int(max_neurons),
            )
            self._calibration_neurons_per_region = neurons
            self._calibration_meta = {"source": "file", **meta}
            return

        if source in {"brain_atlas_default", "atlas"}:
            region_map = _default_atlas_region_map()
            overrides = cfg.get("region_map")
            if isinstance(overrides, dict):
                for key, spec in overrides.items():
                    region = _parse_brain_region(key)
                    if region is None:
                        continue
                    parsed = _parse_region_fraction_spec(spec)
                    if parsed:
                        region_map[region] = parsed

            real_neurons = _derive_real_neurons_from_brain_atlas(region_map, logger=self.logger)
            neurons, meta = _downscale_real_neurons(
                real_neurons,
                target_mean_neurons=float(target_mean),
                scale_divisor=scale_divisor,
                min_neurons=int(min_neurons),
                max_neurons=int(max_neurons),
            )
            self._calibration_neurons_per_region = neurons
            self._calibration_meta = {"source": "brain_atlas_default", **meta}
            return

        raise ValueError(f"Unsupported calibration source: {source}")

    @staticmethod
    def _extract_wm_capacity(global_cfg: Dict[str, Any]) -> Optional[int]:
        candidates = []
        if isinstance(global_cfg, dict):
            candidates.append(global_cfg.get("working_memory_capacity"))
            wm = global_cfg.get("working_memory")
            if isinstance(wm, dict):
                candidates.append(wm.get("capacity"))

            mem = global_cfg.get("memory")
            if isinstance(mem, dict):
                candidates.append(mem.get("working_memory_capacity"))
                mem_wm = mem.get("working_memory")
                if isinstance(mem_wm, dict):
                    candidates.append(mem_wm.get("capacity"))

        for cand in candidates:
            val = _safe_int(cand)
            if val is not None and val > 0:
                return int(val)
        return None

    @staticmethod
    def _extract_decision_learning_rate(global_cfg: Dict[str, Any]) -> Optional[float]:
        candidates = []
        if isinstance(global_cfg, dict):
            dec = global_cfg.get("decision")
            if isinstance(dec, dict):
                candidates.extend(
                    [
                        dec.get("learning_rate"),
                        dec.get("rl_learning_rate"),
                        dec.get("value_learning_rate"),
                    ]
                )
                bg = dec.get("basal_ganglia") or dec.get("bg")
                if isinstance(bg, dict):
                    candidates.extend([bg.get("learning_rate"), bg.get("rl_learning_rate")])

        for cand in candidates:
            val = _safe_float(cand)
            if val is not None and val > 0.0:
                return float(val)
        return None

    def apply_to_region_config(
        self, region_type: BrainRegion, region_config: Dict[str, Any], *, global_config: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if not self.enabled:
            return region_config, {}

        cfg = dict(region_config or {})
        micro = _normalise_microcircuit_cfg(cfg.get("microcircuit"))
        if micro is None:
            return cfg, {}

        params = dict(micro.get("params") or {}) if isinstance(micro.get("params"), dict) else {}
        applied: Dict[str, Any] = {}

        # Optional: baseline calibration (e.g., BrainAtlas or user-provided file).
        if self._calibration_neurons_per_region:
            target = self._calibration_neurons_per_region.get(region_type)
            if target is not None and int(target) > 0:
                should_write = self._calibration_override_existing or "neurons_per_region" not in params
                if should_write:
                    params["neurons_per_region"] = int(target)
                    cal = applied.setdefault("calibration", {})
                    cal["neurons_per_region"] = int(target)
                    if self._calibration_meta:
                        cal["meta"] = dict(self._calibration_meta)

        # Working memory capacity -> PFC sustained-activity substrate size (proxy: neurons_per_region).
        if region_type == BrainRegion.PREFRONTAL_CORTEX:
            capacity = self._extract_wm_capacity(global_config)
            if capacity is not None:
                try:
                    neurons_per_item = int(self._wm_cfg.get("neurons_per_item", 12))
                except Exception:
                    neurons_per_item = 12
                try:
                    min_neurons = int(self._wm_cfg.get("min_neurons", 40))
                except Exception:
                    min_neurons = 40
                try:
                    max_neurons = int(self._wm_cfg.get("max_neurons", 400))
                except Exception:
                    max_neurons = 400
                neurons = _clip_int(float(capacity) * float(neurons_per_item), int(min_neurons), int(max_neurons))
                params["neurons_per_region"] = int(neurons)
                applied["neurons_per_region"] = int(neurons)
                applied["source"] = {"working_memory.capacity": int(capacity)}

        # Decision learning rate -> BG dopamine-gated plasticity knobs (proxy: STDP enabled + dopamine gain).
        if region_type == BrainRegion.BASAL_GANGLIA:
            lr = self._extract_decision_learning_rate(global_config)
            if lr is not None:
                try:
                    enable_stdp = bool(self._bg_cfg.get("enable_stdp", True))
                except Exception:
                    enable_stdp = True
                if enable_stdp:
                    params["stdp_enabled"] = True
                    applied["stdp_enabled"] = True

                gain_scale = _safe_float(self._bg_cfg.get("dopamine_gain_scale", 5.0))
                if gain_scale is None or gain_scale <= 0.0:
                    gain_scale = 5.0
                dopamine_gain = float(np.clip(float(lr) * float(gain_scale), 0.0, 5.0))
                params["dopamine_stdp_gain"] = float(dopamine_gain)
                applied["dopamine_stdp_gain"] = float(dopamine_gain)
                applied["source"] = {"decision.learning_rate": float(lr)}

                # If the BG microcircuit is running on Loihi (nengo_loihi wrapper), optionally
                # make dopamine modulate input excitability as a proxy for D1/D2 gain changes.
                try:
                    model = str(micro.get("model", "") or "").strip().lower()
                except Exception:
                    model = ""
                if model == "loihi":
                    loihi_cfg = micro.get("cfg") if isinstance(micro.get("cfg"), dict) else {}
                    loihi_section = loihi_cfg.get("loihi") if isinstance(loihi_cfg, dict) else {}
                    if not isinstance(loihi_section, dict):
                        loihi_section = {}
                    gain = _safe_float(self._bg_cfg.get("loihi_dopamine_input_gain"))
                    if gain is None:
                        gain = 0.5
                    loihi_section.setdefault("dopamine_input_gain", float(gain))
                    loihi_cfg["loihi"] = loihi_section
                    micro["cfg"] = loihi_cfg
                    applied["loihi_dopamine_input_gain"] = float(loihi_section.get("dopamine_input_gain", 0.0) or 0.0)

        # Visual pipeline: allow swapping a Gabor-like front-end for a retina->LGN->V1 microcircuit preset.
        if region_type == BrainRegion.VISUAL_CORTEX:
            try:
                use_pathway = bool(self._visual_cfg.get("use_retina_lgn_v1", False))
            except Exception:
                use_pathway = False
            if use_pathway:
                micro["preset"] = "retina_lgn_v1"
                applied["preset"] = "retina_lgn_v1"

        if applied:
            micro["params"] = params
            cfg["microcircuit"] = micro
        return cfg, applied


__all__ = [
    "ModuleParameterMapper",
    "ParameterMappingReport",
]
