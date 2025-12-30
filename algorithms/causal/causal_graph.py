"""Simple causal graph implementation."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from algorithms.base import Algorithm


class CausalGraph(Algorithm):
    """Directed acyclic graph for causal reasoning.

    Each node may have a structural function specifying how its value is
    derived from its parents. Values can be set via :meth:`intervene`.
    Inference is performed by recursively evaluating the structural equations.
    """

    def __init__(self) -> None:
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: Set[Tuple[str, str]] = set()

    def add_node(self, name: str, func: Optional[Callable[..., Any]] = None) -> None:
        """Add a node to the graph.

        Args:
            name: Identifier of the node.
            func: Optional structural function of the node.
        """

        if name not in self.nodes:
            self.nodes[name] = {"parents": set(), "func": func, "value": None}
        else:
            if func is not None:
                self.nodes[name]["func"] = func

    def add_edge(self, parent: str, child: str) -> None:
        """Create a directed edge from ``parent`` to ``child``."""

        for node in (parent, child):
            if node not in self.nodes:
                self.add_node(node)
        self.nodes[child]["parents"].add(parent)
        self.edges.add((parent, child))

    def intervene(self, name: str, value: Any) -> None:
        """Set the value of a node, clearing previous inferences."""

        if name not in self.nodes:
            self.add_node(name)
        for node in self.nodes:
            if node != name:
                self.nodes[node]["value"] = None
        self.nodes[name]["value"] = value

    def infer(self, name: str) -> Any:
        """Infer the value of ``name`` given current interventions."""

        if name not in self.nodes:
            raise ValueError(f"Node {name} not in graph")
        node = self.nodes[name]
        if node["value"] is not None:
            return node["value"]
        func = node.get("func")
        parents: List[str] = list(node["parents"])
        if func is None:
            return None
        values = [self.infer(p) for p in parents]
        node["value"] = func(*values)
        return node["value"]

    def execute(self, target: str) -> Any:
        """Execute inference for a target node."""

        return self.infer(target)
