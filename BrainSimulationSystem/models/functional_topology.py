"""Functional-module to anatomical-region routing (BrainFunctionalTopology bridge).

This module bridges the repo-level `modules/brain/anatomy.py` functional
topology (module -> anatomical regions) into the `BrainSimulationSystem.models`
`BrainRegion` network used by `CognitiveArchitecture`.

The goal is to keep high-level module interfaces unchanged while allowing their
signals to be routed into brain-region microcircuits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .enums import BrainRegion


class _FunctionalTopologyUnavailable(RuntimeError):
    pass


def _norm_key(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "_")


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return float(out)


@dataclass(frozen=True)
class FunctionalTopologySnapshot:
    enabled: bool
    module_activity_in: Dict[str, float]
    module_to_regions: Dict[str, List[str]]
    region_drive_total: Dict[str, float]
    module_activity_out: Dict[str, float]


class FunctionalTopologyRouter:
    """Route module-level activity into the brain-region network."""

    def __init__(self, cfg: Dict[str, Any], *, logger: Optional[Any] = None) -> None:
        self.cfg = dict(cfg or {})
        self.logger = logger

        try:
            enabled = bool(self.cfg.get("enabled", False))
        except Exception:
            enabled = False
        self.enabled = bool(enabled)

        self.key_prefix = str(self.cfg.get("key_prefix", "module_") or "module_")
        self.activity_key = str(self.cfg.get("activity_key", "module_activity") or "module_activity")
        self.input_key_prefix = str(self.cfg.get("input_key_prefix", "functional_") or "functional_")

        try:
            self.gain = float(self.cfg.get("gain", 1.0))
        except Exception:
            self.gain = 1.0
        if not np.isfinite(self.gain):
            self.gain = 1.0

        try:
            self.include_sensory = bool(self.cfg.get("include_sensory", False))
        except Exception:
            self.include_sensory = False

        aliases = self.cfg.get("aliases", {})
        self.aliases: Dict[str, str] = {}
        if isinstance(aliases, dict):
            for key, value in aliases.items():
                k = _norm_key(key)
                v = _norm_key(value)
                if k and v:
                    self.aliases[k] = v

        # Conservative built-in aliases (opt-in mapping for common naming).
        self.aliases.setdefault("vision", "visual")
        self.aliases.setdefault("audition", "auditory")
        self.aliases.setdefault("somatic", "somatosensory")
        self.aliases.setdefault("memory", "cognition")
        self.aliases.setdefault("planning", "cognition")
        self.aliases.setdefault("decision", "cognition")
        self.aliases.setdefault("attention", "consciousness")

        # Map atlas anatomical labels to BrainSimulationSystem BrainRegion enums.
        self._anatomy_to_regions: Dict[str, List[BrainRegion]] = self._default_anatomy_mapping()
        overrides = self.cfg.get("anatomy_region_map", {})
        if isinstance(overrides, dict):
            for name, target in overrides.items():
                mapped = self._parse_region_targets(target)
                if mapped:
                    self._anatomy_to_regions[_norm_key(name)] = mapped

        self._topology = None
        if self.enabled:
            self._topology = self._load_topology(self.cfg)

    @staticmethod
    def _default_anatomy_mapping() -> Dict[str, List[BrainRegion]]:
        return {
            _norm_key("Occipital Lobe"): [BrainRegion.VISUAL_CORTEX],
            _norm_key("Temporal Lobe"): [BrainRegion.TEMPORAL_CORTEX, BrainRegion.AUDITORY_CORTEX],
            _norm_key("Parietal Lobe"): [BrainRegion.PARIETAL_CORTEX, BrainRegion.SOMATOSENSORY_CORTEX],
            _norm_key("Frontal Lobe"): [BrainRegion.PREFRONTAL_CORTEX, BrainRegion.MOTOR_CORTEX],
            _norm_key("Prefrontal Cortex"): [BrainRegion.PREFRONTAL_CORTEX],
            _norm_key("Motor Cortex"): [BrainRegion.MOTOR_CORTEX],
            _norm_key("Insular Cortex"): [BrainRegion.INSULAR_CORTEX],
            _norm_key("Cingulate Cortex"): [BrainRegion.CINGULATE_CORTEX],
            _norm_key("Thalamus"): [BrainRegion.THALAMUS],
            _norm_key("Basal Ganglia"): [BrainRegion.BASAL_GANGLIA],
            _norm_key("Hippocampus"): [BrainRegion.HIPPOCAMPUS],
            _norm_key("Amygdala"): [BrainRegion.AMYGDALA],
            _norm_key("Cerebellum"): [BrainRegion.CEREBELLUM],
            _norm_key("Dentate Nucleus"): [BrainRegion.CEREBELLUM],
            _norm_key("Midbrain"): [BrainRegion.BRAINSTEM],
            _norm_key("Pons"): [BrainRegion.BRAINSTEM],
            _norm_key("Medulla"): [BrainRegion.BRAINSTEM],
        }

    @staticmethod
    def _parse_region_targets(target: Any) -> List[BrainRegion]:
        if target is None:
            return []
        if isinstance(target, (list, tuple, set)):
            out: List[BrainRegion] = []
            for item in target:
                out.extend(FunctionalTopologyRouter._parse_region_targets(item))
            # Dedup while preserving order.
            seen: set[BrainRegion] = set()
            ordered: List[BrainRegion] = []
            for region in out:
                if region not in seen:
                    seen.add(region)
                    ordered.append(region)
            return ordered

        if isinstance(target, BrainRegion):
            return [target]

        raw = _norm_key(target)
        if not raw:
            return []

        # Accept enum name (PREFRONTAL_CORTEX) or enum value (prefrontal_cortex).
        for region in BrainRegion:
            if raw == _norm_key(region.name) or raw == _norm_key(region.value):
                return [region]
        return []

    @staticmethod
    def _load_topology(cfg: Dict[str, Any]) -> Any:
        try:
            from modules.brain.anatomy import BrainAtlas, BrainFunctionalTopology  # type: ignore
        except Exception as exc:  # pragma: no cover - optional repository dependency
            raise _FunctionalTopologyUnavailable(
                "BrainFunctionalTopology is not available (missing 'modules.brain')."
            ) from exc

        atlas = BrainAtlas.default()

        module_to_regions = cfg.get("module_to_regions")
        layers = cfg.get("functional_layers")

        kwargs: Dict[str, Any] = {}
        if isinstance(module_to_regions, dict):
            normalised: Dict[str, List[str]] = {}
            for key, value in module_to_regions.items():
                if isinstance(value, (list, tuple, set)):
                    normalised[_norm_key(key)] = [str(v) for v in value]
            if normalised:
                kwargs["module_to_regions"] = normalised

        if isinstance(layers, dict):
            normalised_layers: Dict[str, List[str]] = {}
            for key, value in layers.items():
                if isinstance(value, (list, tuple, set)):
                    normalised_layers[_norm_key(key)] = [str(v) for v in value]
            if normalised_layers:
                kwargs["functional_layers"] = normalised_layers

        return BrainFunctionalTopology(atlas, **kwargs)

    def _canonical_module(self, module: str) -> str:
        key = _norm_key(module)
        return self.aliases.get(key, key)

    def extract_module_activity(self, sensory_inputs: Dict[str, Any], task_demands: Dict[str, Any]) -> Dict[str, float]:
        module_activity: Dict[str, float] = {}

        def _ingest_mapping(mapping: Any) -> None:
            if not isinstance(mapping, dict):
                return
            for key, value in mapping.items():
                val = _safe_float(value)
                if val is None:
                    continue
                module = self._canonical_module(key)
                if not module:
                    continue
                module_activity[module] = module_activity.get(module, 0.0) + float(val)

        if isinstance(task_demands, dict):
            explicit = task_demands.get(self.activity_key)
            _ingest_mapping(explicit)
            for key, value in task_demands.items():
                if not isinstance(key, str):
                    continue
                if key.startswith(self.key_prefix):
                    module = self._canonical_module(key[len(self.key_prefix) :])
                    val = _safe_float(value)
                    if module and val is not None:
                        module_activity[module] = module_activity.get(module, 0.0) + float(val)

        if self.include_sensory and isinstance(sensory_inputs, dict):
            explicit = sensory_inputs.get(self.activity_key)
            _ingest_mapping(explicit)
            for key, value in sensory_inputs.items():
                if not isinstance(key, str):
                    continue
                if key.startswith(self.key_prefix):
                    module = self._canonical_module(key[len(self.key_prefix) :])
                    val = _safe_float(value)
                    if module and val is not None:
                        module_activity[module] = module_activity.get(module, 0.0) + float(val)

        # Filter non-positive values.
        return {k: float(v) for k, v in module_activity.items() if float(v) > 0.0}

    def _module_regions(self, module: str) -> List[str]:
        topo = self._topology
        if topo is None:
            return []
        try:
            names = topo.module_to_regions.get(module, [])  # type: ignore[attr-defined]
        except Exception:
            names = []
        if not isinstance(names, list):
            return []
        return [str(name) for name in names]

    def resolve_module_regions(
        self, module: str, *, available_regions: Optional[Sequence[BrainRegion]] = None
    ) -> List[BrainRegion]:
        mapped: List[BrainRegion] = []
        for name in self._module_regions(module):
            targets = self._anatomy_to_regions.get(_norm_key(name), [])
            for region in targets:
                mapped.append(region)

        # Dedup while preserving order.
        seen: set[BrainRegion] = set()
        ordered: List[BrainRegion] = []
        for region in mapped:
            if region not in seen:
                seen.add(region)
                ordered.append(region)

        if available_regions is None:
            return ordered

        available = set(available_regions)
        return [region for region in ordered if region in available]

    def apply_to_region_inputs(
        self,
        region_inputs: Dict[BrainRegion, Dict[str, float]],
        *,
        sensory_inputs: Dict[str, Any],
        task_demands: Dict[str, Any],
        available_regions: Sequence[BrainRegion],
    ) -> FunctionalTopologySnapshot:
        if not self.enabled or self._topology is None:
            return FunctionalTopologySnapshot(
                enabled=False, module_activity_in={}, module_to_regions={}, region_drive_total={}, module_activity_out={}
            )

        module_activity = self.extract_module_activity(sensory_inputs, task_demands)
        if not module_activity:
            return FunctionalTopologySnapshot(
                enabled=True, module_activity_in={}, module_to_regions={}, region_drive_total={}, module_activity_out={}
            )

        module_to_regions_out: Dict[str, List[str]] = {}
        region_drive_total: Dict[str, float] = {}

        for module, value in module_activity.items():
            regions = self.resolve_module_regions(module, available_regions=available_regions)
            if not regions:
                continue

            module_to_regions_out[module] = [r.value for r in regions]
            per_region = float(value) / float(len(regions))

            for region in regions:
                bucket = region_inputs.setdefault(region, {})
                key = f"{self.input_key_prefix}{module}"
                bucket[key] = float(bucket.get(key, 0.0)) + float(per_region) * float(self.gain)
                region_drive_total[region.value] = float(region_drive_total.get(region.value, 0.0)) + float(per_region)

        return FunctionalTopologySnapshot(
            enabled=True,
            module_activity_in=dict(module_activity),
            module_to_regions=module_to_regions_out,
            region_drive_total=region_drive_total,
            module_activity_out={},
        )

    @staticmethod
    def compute_module_readout(
        snapshot: FunctionalTopologySnapshot, region_activities: Dict[str, Dict[str, Any]]
    ) -> FunctionalTopologySnapshot:
        if not snapshot.enabled or not snapshot.module_to_regions:
            return snapshot

        module_out: Dict[str, float] = {}
        for module, region_keys in snapshot.module_to_regions.items():
            values: List[float] = []
            for region_key in region_keys:
                payload = region_activities.get(region_key, {})
                if isinstance(payload, dict):
                    val = _safe_float(payload.get("activation"))
                    if val is not None:
                        values.append(float(val))
            if values:
                module_out[module] = float(sum(values) / float(len(values)))

        return FunctionalTopologySnapshot(
            enabled=snapshot.enabled,
            module_activity_in=snapshot.module_activity_in,
            module_to_regions=snapshot.module_to_regions,
            region_drive_total=snapshot.region_drive_total,
            module_activity_out=module_out,
        )


__all__ = [
    "FunctionalTopologyRouter",
    "FunctionalTopologySnapshot",
]

