"""Structural evolution manager for topology-level mutations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple
from copy import deepcopy

from .dynamic_architecture import DynamicArchitectureExpander
from .structural_encoding import encode_structure
from .structural_genome import StructuralGenome

if TYPE_CHECKING:
    from .self_evolving_ai_architecture import SelfEvolvingAIArchitecture


StructuralBottleneck = Tuple[str, float]


@dataclass
class StructuralProposal:
    """Description of a structural mutation candidate."""

    topology: Dict[str, List[str]]
    module_gates: Dict[str, float]
    score: float
    reason: str


class StructuralEvolutionManager:
    """Explore topology mutations on top of parameter evolution.

    This manager toggles modules on/off and rewires connections to alleviate
    bottlenecks detected at runtime. It collaborates with
    :class:`SelfEvolvingAIArchitecture` (for parameter/fitness evaluation) and
    optionally a :class:`DynamicArchitectureExpander` (for applying the graph
    changes to a live execution graph).
    """

    def __init__(
        self,
        architecture: "SelfEvolvingAIArchitecture",
        expander: Optional[DynamicArchitectureExpander] = None,
        *,
        exploration_budget: int = 4,
        connection_penalty: float = 0.05,
        initial_topology: Optional[Dict[str, List[str]]] = None,
        initial_module_gates: Optional[Dict[str, float]] = None,
        genome: Optional[StructuralGenome] = None,
    ) -> None:
        self._genome = genome
        self.architecture = architecture
        self.expander = expander
        self.exploration_budget = max(1, exploration_budget)
        self.connection_penalty = max(0.0, connection_penalty)
        topo, gates = self._resolve_initial_structure(
            initial_topology, initial_module_gates, genome
        )
        self._topology = topo
        self.module_gates = gates
        self._version = 0
        self.history: List[StructuralProposal] = []
        self._apply_to_expander()

    # ------------------------------------------------------------------ #
    @property
    def topology(self) -> Dict[str, List[str]]:
        """Current view of the module connection graph."""

        return deepcopy(self._topology)

    # ------------------------------------------------------------------ #
    def evolve_structure(
        self,
        performance: float,
        bottlenecks: Optional[Iterable[StructuralBottleneck]] = None,
        candidate_modules: Optional[Iterable[str]] = None,
        *,
        commit: bool = True,
    ) -> StructuralProposal:
        """Search for a better topology and apply the best candidate."""

        bottleneck_list = list(bottlenecks or [])
        candidates: List[Tuple[Dict[str, List[str]], Dict[str, float], str]] = []
        self._add_candidate(candidates, self._topology, self.module_gates, "status_quo")

        if bottleneck_list:
            target, _ = bottleneck_list[0]
            gates = dict(self.module_gates)
            if gates.get(target, 1.0) >= 0.5:
                gates[target] = 0.0
                pruned = self._prune_disabled(self._topology, gates)
                self._add_candidate(
                    candidates, pruned, gates, f"disable {target} to relieve bottleneck"
                )
            bypass = self._bypass_module(target)
            if bypass != self._topology:
                self._add_candidate(
                    candidates, bypass, dict(self.module_gates), f"bypass {target} connections"
                )

        for module in candidate_modules or []:
            if len(candidates) >= self.exploration_budget + 1:
                break
            gates = dict(self.module_gates)
            if gates.get(module, 0.0) >= 0.5:
                continue
            gates[module] = 1.0
            rewired = self._attach_new_module(module, gates, bottleneck_list)
            self._add_candidate(
                candidates, rewired, gates, f"enable {module} for additional capacity"
            )

        scored = [
            self._score_candidate(topo, gates, reason, performance)
            for topo, gates, reason in candidates[: self.exploration_budget + 1]
        ]
        best = max(scored, key=lambda proposal: proposal.score)
        if commit and self._has_changed(best):
            self._commit(best, performance)
        return best

    # ------------------------------------------------------------------ #
    def _resolve_initial_topology(
        self,
        initial: Optional[Dict[str, List[str]]],
        genome: Optional[StructuralGenome],
    ) -> Dict[str, List[str]]:
        if genome is not None:
            topo, _ = genome.to_topology()
            return topo
        if initial is not None:
            return {name: list(edges) for name, edges in initial.items()}
        if self.expander is not None:
            return {name: list(edges) for name, edges in self.expander.connections.items()}
        return {}

    # ------------------------------------------------------------------ #
    def _resolve_initial_gates(
        self,
        gates: Optional[Dict[str, float]],
        genome: Optional[StructuralGenome],
        topology: Dict[str, List[str]],
    ) -> Dict[str, float]:
        if genome is not None:
            _, resolved = genome.to_topology()
            return resolved
        resolved: Dict[str, float] = {}
        for name in topology:
            resolved[name] = 1.0
        for dsts in topology.values():
            for dst in dsts:
                resolved.setdefault(dst, 1.0)
        if self.expander is not None:
            for name in getattr(self.expander, "modules", {}).keys():
                resolved.setdefault(name, 1.0)
        if gates:
            for name, value in gates.items():
                resolved[name] = 1.0 if float(value) >= 0.5 else 0.0
        return resolved

    # ------------------------------------------------------------------ #
    def _resolve_initial_structure(
        self,
        initial_topology: Optional[Dict[str, List[str]]],
        initial_gates: Optional[Dict[str, float]],
        genome: Optional[StructuralGenome],
    ) -> Tuple[Dict[str, List[str]], Dict[str, float]]:
        topology = self._resolve_initial_topology(initial_topology, genome)
        gates = self._resolve_initial_gates(initial_gates, genome, topology)
        return topology, gates

    # ------------------------------------------------------------------ #
    def _prune_disabled(
        self, topology: Dict[str, List[str]], gates: Dict[str, float]
    ) -> Dict[str, List[str]]:
        cleaned: Dict[str, List[str]] = {name: [] for name in topology}
        for name in gates:
            cleaned.setdefault(name, [])
        for src, dsts in topology.items():
            if gates.get(src, 1.0) < 0.5:
                continue
            active = [
                dst for dst in dsts if gates.get(dst, 1.0) >= 0.5
            ]
            cleaned[src] = active
        return cleaned

    # ------------------------------------------------------------------ #
    def _add_candidate(
        self,
        collection: List[Tuple[Dict[str, List[str]], Dict[str, float], str]],
        topology: Dict[str, List[str]],
        gates: Dict[str, float],
        reason: str,
    ) -> None:
        signature = (tuple(sorted(topology.items())), tuple(sorted(gates.items())))
        for existing, _, _ in collection:
            if tuple(sorted(existing.items())) == signature[0]:
                return
        collection.append((deepcopy(topology), dict(gates), reason))

    # ------------------------------------------------------------------ #
    def _attach_new_module(
        self,
        module: str,
        gates: Dict[str, float],
        bottlenecks: List[StructuralBottleneck],
    ) -> Dict[str, List[str]]:
        topology = deepcopy(self._topology)
        topology.setdefault(module, [])
        attach_from: Optional[str] = bottlenecks[0][0] if bottlenecks else None
        if attach_from is None and topology:
            attach_from = min(topology, key=lambda name: len(topology.get(name, [])))
        if attach_from is not None:
            topology.setdefault(attach_from, [])
            if module not in topology[attach_from]:
                topology[attach_from].append(module)
        return self._prune_disabled(topology, gates)

    # ------------------------------------------------------------------ #
    def _bypass_module(self, module: str) -> Dict[str, List[str]]:
        if module not in self._topology:
            return self._topology
        parents: Dict[str, List[str]] = {}
        for src, dsts in self._topology.items():
            for dst in dsts:
                parents.setdefault(dst, []).append(src)
        children = self._topology.get(module, [])
        bypassed = deepcopy(self._topology)
        for parent in parents.get(module, []):
            for child in children:
                if child == parent:
                    continue
                bypassed.setdefault(parent, [])
                if child not in bypassed[parent]:
                    bypassed[parent].append(child)
            if module in bypassed.get(parent, []):
                bypassed[parent].remove(module)
        bypassed[module] = []
        return bypassed

    # ------------------------------------------------------------------ #
    def _score_candidate(
        self,
        topology: Dict[str, List[str]],
        gates: Dict[str, float],
        reason: str,
        performance: float,
    ) -> StructuralProposal:
        cleaned = self._prune_disabled(topology, gates)
        encoded = encode_structure(cleaned, gates)
        merged = {**self.architecture.architecture, **encoded}
        merged.setdefault("structural_performance_hint", float(performance))
        structural_score = float(self.architecture.evolver.fitness_fn(merged))
        enabled_delta = self._active_count(gates) - self._active_count(self.module_gates)
        if enabled_delta > 0:
            structural_score += 0.05 * enabled_delta
        penalty = self.connection_penalty * self._connection_count(cleaned)
        structural_score -= penalty
        return StructuralProposal(cleaned, dict(gates), structural_score, reason)

    # ------------------------------------------------------------------ #
    def as_architecture(
        self, proposal: StructuralProposal, base: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Merge proposal encoding into a base architecture dict."""

        merged = dict(base or {})
        merged.update(encode_structure(proposal.topology, proposal.module_gates))
        return merged

    # ------------------------------------------------------------------ #
    def commit_proposal(
        self,
        proposal: StructuralProposal,
        performance: float,
        extra_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        if not self._has_changed(proposal):
            return
        self._commit(proposal, performance, extra_metrics)

    def _connection_count(self, topology: Dict[str, List[str]]) -> int:
        return sum(len(dsts) for dsts in topology.values())

    # ------------------------------------------------------------------ #
    def _active_count(self, gates: Optional[Dict[str, float]] = None) -> int:
        gates = gates or self.module_gates
        return sum(1 for value in gates.values() if value >= 0.5)

    # ------------------------------------------------------------------ #
    def _has_changed(self, proposal: StructuralProposal) -> bool:
        return (
            proposal.topology != self._topology
            or proposal.module_gates != self.module_gates
        )

    # ------------------------------------------------------------------ #
    def _commit(
        self,
        proposal: StructuralProposal,
        performance: float,
        extra_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        self._topology = deepcopy(proposal.topology)
        self.module_gates = dict(proposal.module_gates)
        self._version += 1
        self.history.append(proposal)
        self._apply_to_expander()

        update_payload = encode_structure(self._topology, self.module_gates)
        update_payload["structural_version"] = float(self._version)
        metrics = {
            "structure_score": proposal.score,
            "active_modules": float(self._active_count()),
            "connections": float(self._connection_count(self._topology)),
        }
        if extra_metrics:
            metrics.update(extra_metrics)
        self.architecture.update_architecture(
            update_payload,
            performance=performance,
            metrics=metrics,
        )

    # ------------------------------------------------------------------ #
    def apply_neat_genome(
        self,
        genome: StructuralGenome,
        *,
        performance: float,
        reason: str = "neuroevolution winner",
        extra_metrics: Optional[Dict[str, float]] = None,
    ) -> StructuralProposal:
        """Apply a structural genome coming from a NEAT-style backend."""

        topology, gates = genome.to_topology()
        proposal = StructuralProposal(topology, gates, performance, reason)
        self._commit(proposal, performance, extra_metrics)
        self._genome = genome
        return proposal

    # ------------------------------------------------------------------ #
    def _apply_to_expander(self) -> None:
        if self.expander is None:
            return
        cleaned = self._prune_disabled(self._topology, self.module_gates)
        self.expander.connections = deepcopy(cleaned)
