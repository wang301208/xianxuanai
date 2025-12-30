"""Runtime architecture expansion utilities."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Set, Tuple

from .self_evolving_ai_architecture import SelfEvolvingAIArchitecture


class DynamicArchitectureExpander:
    """Dynamically add or remove modules and connections.

    The expander maintains a lightweight directed graph of callable modules. It
    can restructure the graph at runtime and optionally synchronise changes with
    :class:`SelfEvolvingAIArchitecture`. When performance metrics or
    environmental feedback indicate that the current setup is sub-optimal the
    expander can invoke the self-evolution machinery to mutate the architecture
    further.
    """

    def __init__(
        self,
        modules: Dict[str, Callable[[float], float]],
        connections: Optional[Dict[str, List[str]]] = None,
        evolver: Optional[SelfEvolvingAIArchitecture] = None,
    ) -> None:
        self.modules = modules
        self.connections = connections or {name: [] for name in modules}
        self.evolver = evolver

    # ------------------------------------------------------------------
    def add_module(self, name: str, module: Callable[[float], float]) -> None:
        """Add a new module to the architecture."""

        self.modules[name] = module
        self.connections.setdefault(name, [])

    # ------------------------------------------------------------------
    def remove_module(self, name: str) -> None:
        """Remove ``name`` and any links pointing to it."""

        self.modules.pop(name, None)
        self.connections.pop(name, None)
        for dsts in self.connections.values():
            if name in dsts:
                dsts.remove(name)

    # ------------------------------------------------------------------
    def connect(self, src: str, dst: str) -> None:
        """Create a connection from ``src`` to ``dst``."""

        self.connections.setdefault(src, [])
        if dst not in self.connections[src]:
            self.connections[src].append(dst)

    # ------------------------------------------------------------------
    def disconnect(self, src: str, dst: str) -> None:
        """Remove the connection from ``src`` to ``dst`` if present."""

        if src in self.connections and dst in self.connections[src]:
            self.connections[src].remove(dst)

    # ------------------------------------------------------------------
    def run(self, start: str, value: float) -> float:
        """Execute the architecture starting from ``start`` on ``value``."""

        visited: Set[str] = set()

        def _run(name: str, val: float) -> float:
            visited.add(name)
            out = self.modules[name](val)
            for nxt in self.connections.get(name, []):
                if nxt not in visited:
                    out = _run(nxt, out)
            return out

        return _run(start, value)

    # ------------------------------------------------------------------
    def auto_expand(
        self,
        performance: float,
        env_feedback: Optional[
            Dict[str, Tuple[str, Callable[[float], float], str] | str]
        ] = None,
        threshold: float = 0.5,
    ) -> None:
        """Restructure the architecture based on feedback.

        Parameters
        ----------
        performance:
            Current performance metric. If below ``threshold`` the underlying
            self-evolution algorithm is triggered.
        env_feedback:
            Optional directives from the environment. Supported keys are::

                "add_module" -> (name, callable, parent)
                    Adds ``name`` and connects it after ``parent``.
                "remove_module" -> name
                    Removes the specified module.
        threshold:
            Performance threshold to trigger self-evolution.
        """

        if env_feedback:
            if "add_module" in env_feedback:
                name, module, parent = env_feedback["add_module"]
                self.add_module(name, module)
                self.connect(parent, name)
            if "remove_module" in env_feedback:
                target = env_feedback["remove_module"]
                self.remove_module(target)

        if performance < threshold and self.evolver is not None:
            candidates = self.evolver.generate_architecture_mutations()
            self.evolver.evolutionary_selection(candidates)

        if self.evolver is not None:
            self._sync_with_evolver()

    # ------------------------------------------------------------------
    def _sync_with_evolver(self) -> None:
        """Update the attached :class:`SelfEvolvingAIArchitecture` with modules."""

        if self.evolver is None:
            return
        arch = {name: 1.0 for name in self.modules}
        self.evolver.update_architecture(arch)
