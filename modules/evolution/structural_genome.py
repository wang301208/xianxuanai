"""Lightweight structural genome with module/edge genes and toggles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterable
from copy import deepcopy


@dataclass
class ModuleGene:
    """Gene describing a module and its on/off state."""

    id: str
    type: Optional[str] = None
    enabled: bool = True
    params: Dict[str, float] = field(default_factory=dict)


@dataclass
class EdgeGene:
    """Gene describing a directed connection between modules."""

    src: str
    dst: str
    enabled: bool = True
    weight: Optional[float] = None
    innovation: Optional[int] = None


class StructuralGenome:
    """Graph-style genome supporting NEAT-like toggles and mutations."""

    def __init__(
        self,
        modules: Iterable[ModuleGene] | None = None,
        edges: Iterable[EdgeGene] | None = None,
        *,
        next_innovation: int = 0,
    ) -> None:
        self.modules: Dict[str, ModuleGene] = {m.id: deepcopy(m) for m in modules or []}
        self.edges: List[EdgeGene] = [deepcopy(e) for e in edges or []]
        self.next_innovation = next_innovation

    # ------------------------------------------------------------------ #
    def to_topology(self) -> Tuple[Dict[str, List[str]], Dict[str, float]]:
        """Return (connections, gates) for runtime application."""

        gates: Dict[str, float] = {mid: 1.0 if gene.enabled else 0.0 for mid, gene in self.modules.items()}
        connections: Dict[str, List[str]] = {mid: [] for mid in self.modules}
        for edge in self.edges:
            if not edge.enabled:
                continue
            if not gates.get(edge.src, 0.0) or not gates.get(edge.dst, 0.0):
                continue
            connections.setdefault(edge.src, [])
            if edge.dst not in connections[edge.src]:
                connections[edge.src].append(edge.dst)
        return connections, gates

    # ------------------------------------------------------------------ #
    def toggle_module(self, module_id: str, enabled: Optional[bool] = None) -> None:
        gene = self.modules.get(module_id)
        if gene is None:
            return
        gene.enabled = (not gene.enabled) if enabled is None else bool(enabled)

    # ------------------------------------------------------------------ #
    def toggle_edge(self, src: str, dst: str, enabled: Optional[bool] = None) -> None:
        for edge in self.edges:
            if edge.src == src and edge.dst == dst:
                edge.enabled = (not edge.enabled) if enabled is None else bool(enabled)
                return

    # ------------------------------------------------------------------ #
    def add_module(self, module_id: str, *, type: Optional[str] = None, enabled: bool = True) -> ModuleGene:
        gene = ModuleGene(id=module_id, type=type, enabled=enabled)
        self.modules[module_id] = gene
        return gene

    # ------------------------------------------------------------------ #
    def add_edge(self, src: str, dst: str, *, enabled: bool = True, weight: Optional[float] = None) -> EdgeGene:
        innovation = self._allocate_innovation()
        gene = EdgeGene(src=src, dst=dst, enabled=enabled, weight=weight, innovation=innovation)
        self.edges.append(gene)
        return gene

    # ------------------------------------------------------------------ #
    def split_edge(self, src: str, dst: str, new_module_id: str) -> None:
        """NEAT-style split: disable edge, insert new module and two edges."""

        self.toggle_edge(src, dst, enabled=False)
        self.add_module(new_module_id, enabled=True)
        self.add_edge(src, new_module_id, enabled=True)
        self.add_edge(new_module_id, dst, enabled=True)

    # ------------------------------------------------------------------ #
    def clone(self) -> StructuralGenome:
        return StructuralGenome(self.modules.values(), self.edges, next_innovation=self.next_innovation)

    # ------------------------------------------------------------------ #
    def active_modules(self) -> List[str]:
        return [mid for mid, gene in self.modules.items() if gene.enabled]

    # ------------------------------------------------------------------ #
    def active_edges(self) -> List[Tuple[str, str]]:
        return [
            (edge.src, edge.dst)
            for edge in self.edges
            if edge.enabled and self.modules.get(edge.src, ModuleGene("")).enabled and self.modules.get(edge.dst, ModuleGene("")).enabled
        ]

    # ------------------------------------------------------------------ #
    def _allocate_innovation(self) -> int:
        current = self.next_innovation
        self.next_innovation += 1
        return current
