"""Lightweight anatomical abstractions for the whole-brain simulation.

This module introduces a lightweight anatomical atlas with hierarchical
regions, cell type profiles and connectome utilities.  It is intentionally
numerically stable and dependency-free so it can be used in tests without
heavy datasets while still exposing the concepts needed for a richer model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Mapping, MutableMapping, Optional

import numpy as np


@dataclass
class BrainRegion:
    """Representation of an anatomical brain region.

    Parameters
    ----------
    name:
        Human readable label for the region (e.g. ``"V1"``).
    structure:
        High-level structural grouping (cortex, subcortex, cerebellum, brainstem).
    layer:
        Optional layer or subdivision within the region.  Cortical regions use
        cytoarchitectonic layers (L1-L6) while subcortical structures use nuclei
        names.
    volume_mm3:
        Approximate volumetric extent in cubic millimetres.  The values are
        coarse but allow estimation of cell counts when combined with density
        values.
    cell_densities:
        Mapping of cell type -> density (#cells / mm³).
    subregions:
        Nested anatomical units.
    """

    name: str
    structure: str
    layer: Optional[str]
    volume_mm3: float
    cell_densities: Dict[str, float] = field(default_factory=dict)
    subregions: MutableMapping[str, "BrainRegion"] = field(default_factory=dict)

    def add_subregion(self, region: "BrainRegion") -> None:
        self.subregions[region.name] = region

    def cell_count(self, cell_type: Optional[str] = None) -> float:
        if cell_type:
            density = self.cell_densities.get(cell_type, 0.0)
            return density * self.volume_mm3
        return sum(density * self.volume_mm3 for density in self.cell_densities.values())

    def flatten(self) -> Iterator["BrainRegion"]:
        yield self
        for region in self.subregions.values():
            yield from region.flatten()


class BrainAtlas:
    """Hierarchical atlas of brain regions with cell type metadata."""

    def __init__(self, root: BrainRegion):
        self._root = root
        self._region_index: Dict[str, BrainRegion] = {}
        for region in root.flatten():
            self._region_index[self._normalise_name(region.name)] = region

    @staticmethod
    def _normalise_name(name: str) -> str:
        return name.lower().replace(" ", "_")

    @property
    def root(self) -> BrainRegion:
        return self._root

    @property
    def regions(self) -> Mapping[str, BrainRegion]:
        return self._region_index

    def get(self, name: str) -> Optional[BrainRegion]:
        return self._region_index.get(self._normalise_name(name))

    def iter_regions(self) -> Iterator[BrainRegion]:
        return iter(self._region_index.values())

    def cell_density_map(self) -> Dict[str, Dict[str, float]]:
        return {
            region.name: dict(region.cell_densities) for region in self.iter_regions()
        }

    @classmethod
    def default(cls) -> "BrainAtlas":
        """Construct a conservative atlas covering major anatomical structures."""

        cortex = BrainRegion(
            name="Cerebral Cortex",
            structure="cortex",
            layer=None,
            volume_mm3=623000.0,
            cell_densities={"excitatory": 90000, "inhibitory": 18000},
        )
        for lobe, volume in {
            "Frontal Lobe": 210000.0,
            "Parietal Lobe": 150000.0,
            "Temporal Lobe": 140000.0,
            "Occipital Lobe": 123000.0,
            "Insular Cortex": 40000.0,
            "Prefrontal Cortex": 60000.0,
            "Motor Cortex": 50000.0,
        }.items():
            cortex.add_subregion(
                BrainRegion(
                    name=lobe,
                    structure="cortex",
                    layer=None,
                    volume_mm3=volume,
                    cell_densities={"excitatory": 85000, "inhibitory": 20000},
                )
            )

        subcortex = BrainRegion(
            name="Subcortical Nuclei",
            structure="subcortex",
            layer=None,
            volume_mm3=110000.0,
            cell_densities={"projection": 65000, "interneuron": 12000},
        )
        for nucleus, density in {
            "Thalamus": {"projection": 70000, "relay": 20000},
            "Basal Ganglia": {"medium_spiny": 80000, "fast_spiking": 15000},
            "Hippocampus": {"pyramidal": 95000, "granule": 30000},
            "Amygdala": {"pyramidal": 72000, "interneuron": 18000},
        }.items():
            subcortex.add_subregion(
                BrainRegion(
                    name=nucleus,
                    structure="subcortex",
                    layer=None,
                    volume_mm3=25000.0,
                    cell_densities=density,
                )
            )

        cerebellum = BrainRegion(
            name="Cerebellum",
            structure="cerebellum",
            layer=None,
            volume_mm3=105000.0,
            cell_densities={"granule": 400000, "purkinje": 1200},
        )
        cerebellum.add_subregion(
            BrainRegion(
                name="Dentate Nucleus",
                structure="cerebellum",
                layer=None,
                volume_mm3=9000.0,
                cell_densities={"projection": 65000},
            )
        )

        brainstem = BrainRegion(
            name="Brainstem",
            structure="brainstem",
            layer=None,
            volume_mm3=80000.0,
            cell_densities={"monoaminergic": 25000, "motor": 30000},
        )
        for structure, density in {
            "Midbrain": {"dopaminergic": 28000, "gabaergic": 15000},
            "Pons": {"noradrenergic": 20000, "motor": 25000},
            "Medulla": {"autonomic": 22000, "respiratory": 26000},
        }.items():
            brainstem.add_subregion(
                BrainRegion(
                    name=structure,
                    structure="brainstem",
                    layer=None,
                    volume_mm3=20000.0,
                    cell_densities=density,
                )
            )

        root = BrainRegion(
            name="Whole Brain",
            structure="global",
            layer=None,
            volume_mm3=0.0,
            cell_densities={},
        )
        for region in (cortex, subcortex, cerebellum, brainstem):
            root.add_subregion(region)

        return cls(root)


@dataclass
class ConnectomeMatrix:
    """Directional, weighted structural connectome."""

    labels: List[str]
    weights: np.ndarray
    dataset: str = "unknown"

    def __post_init__(self) -> None:
        if self.weights.shape != (len(self.labels), len(self.labels)):
            raise ValueError("Weight matrix must be square and aligned with labels")
        np.fill_diagonal(self.weights, 0.0)

    @classmethod
    def from_atlas(
        cls,
        atlas: BrainAtlas,
        dataset: str = "hcp",
        weight_scale: float = 1.0,
        sparsity: float = 0.2,
    ) -> "ConnectomeMatrix":
        regions = [region.name for region in atlas.iter_regions()]
        size = len(regions)
        rng = np.random.default_rng(abs(hash(dataset)) % (2**32))
        base = rng.random((size, size))

        volume = np.array([atlas.get(name).volume_mm3 if atlas.get(name) else 1.0 for name in regions])
        volume = volume / (volume.max() or 1.0)
        weight_matrix = (base * (volume[:, None] + volume[None, :]) / 2.0) * 0.1
        weight_matrix *= weight_scale

        if sparsity > 0:
            threshold = np.quantile(weight_matrix, sparsity)
            weight_matrix[weight_matrix < threshold] = 0.0

        return cls(labels=regions, weights=weight_matrix, dataset=dataset)

    def copy(self) -> "ConnectomeMatrix":
        return ConnectomeMatrix(list(self.labels), np.array(self.weights), dataset=self.dataset)

    def scale(self, factor: float) -> "ConnectomeMatrix":
        scaled = self.copy()
        scaled.weights *= factor
        return scaled

    def sparsify(self, keep_fraction: float) -> "ConnectomeMatrix":
        keep_fraction = max(0.0, min(1.0, keep_fraction))
        if keep_fraction == 1.0:
            return self.copy()
        flattened = self.weights.flatten()
        threshold = np.quantile(flattened, 1 - keep_fraction)
        sparsified = self.copy()
        sparsified.weights[sparsified.weights < threshold] = 0.0
        return sparsified

    def coarse_grain(self, factor: int) -> "ConnectomeMatrix":
        factor = max(1, factor)
        if factor == 1:
            return self.copy()
        size = len(self.labels)
        new_size = size // factor
        if new_size == 0:
            raise ValueError("Factor too large for coarse graining")
        reduced = np.zeros((new_size, new_size))
        for i in range(new_size):
            for j in range(new_size):
                block = self.weights[i * factor : (i + 1) * factor, j * factor : (j + 1) * factor]
                reduced[i, j] = block.mean() if block.size else 0.0
        labels = ["|".join(self.labels[i * factor : (i + 1) * factor]) for i in range(new_size)]
        return ConnectomeMatrix(labels=labels, weights=reduced, dataset=self.dataset)

    def propagate(self, activity: Mapping[str, float]) -> Dict[str, float]:
        vector = np.array([activity.get(label, 0.0) for label in self.labels])
        propagated = vector @ self.weights
        return {
            label: float(value)
            for label, value in zip(self.labels, propagated)
            if float(value) > 0.0
        }


@dataclass
class BrainFunctionalTopology:
    """Mapping between anatomical regions and functional subsystems."""

    atlas: BrainAtlas
    module_to_regions: Dict[str, List[str]] = field(default_factory=dict)
    functional_layers: Dict[str, List[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.module_to_regions:
            self.module_to_regions = {
                "visual": ["Occipital Lobe", "Thalamus"],
                "auditory": ["Temporal Lobe", "Midbrain"],
                "somatosensory": ["Parietal Lobe", "Thalamus"],
                "cognition": ["Frontal Lobe", "Hippocampus", "Cerebellum"],
                "emotion": ["Amygdala", "Hippocampus"],
                "curiosity": ["Prefrontal Cortex", "Parietal Lobe"],
                "motor": ["Motor Cortex", "Basal Ganglia", "Cerebellum"],
                "precision_motor": ["Cerebellum", "Dentate Nucleus"],
                "consciousness": ["Insular Cortex", "Thalamus"],
            }
        if not self.functional_layers:
            self.functional_layers = {
                "sensory": ["visual", "auditory", "somatosensory"],
                "cognitive": ["cognition", "consciousness"],
                "affective": ["emotion"],
                "adaptive": ["curiosity"],
                "motor": ["motor", "precision_motor"],
            }

    def resolve_regions(self, module: str) -> List[BrainRegion]:
        names = self.module_to_regions.get(module, [])
        regions: List[BrainRegion] = []
        for name in names:
            region = self.atlas.get(name)
            if region:
                regions.append(region)
        return regions

    def project_activity(self, module_activity: Mapping[str, float]) -> Dict[str, float]:
        anatomical_activity: Dict[str, float] = {}
        for module, activity in module_activity.items():
            regions = self.resolve_regions(module)
            if not regions or activity <= 0:
                continue
            contribution = activity / len(regions)
            for region in regions:
                anatomical_activity[region.name] = anatomical_activity.get(region.name, 0.0) + contribution
        return anatomical_activity

    def layer_activity(self, module_activity: Mapping[str, float]) -> Dict[str, float]:
        return {
            layer: float(sum(module_activity.get(module, 0.0) for module in modules))
            for layer, modules in self.functional_layers.items()
        }

    def build_snapshot(
        self,
        module_activity: Mapping[str, float],
        connectome: ConnectomeMatrix,
    ) -> Dict[str, Dict[str, float]]:
        anatomical = self.project_activity(module_activity)
        propagated = connectome.propagate(anatomical)
        layers = self.layer_activity(module_activity)
        return {
            "functional": dict(module_activity),
            "anatomical": anatomical,
            "connectome": propagated,
            "layers": layers,
        }


__all__ = [
    "BrainRegion",
    "BrainAtlas",
    "ConnectomeMatrix",
    "BrainFunctionalTopology",
]
