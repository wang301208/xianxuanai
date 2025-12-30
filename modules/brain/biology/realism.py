"""Subsystems for enhancing biological realism in brain simulations.

This module provides a high level :class:`BiologicalRealismEnhancer` that groups
several biological processes often included in computational neuroscience
models.  The implementations are intentionally lightweight – they only provide
simple numerical updates so that unit tests can exercise the public APIs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Callable


class BiologicalRealismEnhancer:
    """Container object bundling neuromodulation, circadian rhythm,
    synaptic plasticity and developmental processes.

    The class exposes small APIs which update internal state based on simple
    mathematical rules.  The goal is not to be biologically accurate but to
    provide hooks that mirror the kind of interfaces such a component could
    offer.
    """

    # ------------------------------------------------------------------
    # Plasticity rules
    class TripleSTDP:
        """Placeholder triple spike timing dependent plasticity rule."""

        def update(
            self,
            weight: float,
            pre_prev: float,
            pre: float,
            post: float,
            lr: float = 0.01,
        ) -> float:
            # Simplified potentiation based on three factor interaction
            return weight + lr * pre_prev * pre * post

    class BCMRule:
        """Simplified BCM rule adjusting weight based on activity threshold."""

        def update(
            self,
            weight: float,
            pre: float,
            post: float,
            threshold: float = 0.5,
            lr: float = 0.01,
        ) -> float:
            return weight + lr * pre * post * (post - threshold)

    class OjaRule:
        """Oja's rule keeping weights bounded."""

        def update(
            self,
            weight: float,
            pre: float,
            post: float,
            lr: float = 0.01,
        ) -> float:
            return weight + lr * (pre * post - (post ** 2) * weight)

    class SynapticScaling:
        """Homeostatic scaling towards a target weight value."""

        def update(
            self,
            weight: float,
            target: float = 1.0,
            lr: float = 0.01,
        ) -> float:
            return weight + lr * (target - weight)

    # ------------------------------------------------------------------
    @dataclass
    class DevelopmentState:
        neurons: int = 0
        synapses: int = 0
        myelination: float = 0.0

    def __init__(self) -> None:
        # neuromodulator concentrations
        self.neuromodulators: Dict[str, float] = {
            "dopamine": 0.0,
            "serotonin": 0.0,
            "acetylcholine": 0.0,
            "norepinephrine": 0.0,
        }
        # circadian clock in hours
        self.circadian_time: float = 0.0
        # plasticity rules
        self.plasticity_rules: Dict[str, Callable] = {
            "triple_stdp": self.TripleSTDP(),
            "bcm": self.BCMRule(),
            "oja": self.OjaRule(),
            "synaptic_scaling": self.SynapticScaling(),
        }
        self.development = self.DevelopmentState()

    # ------------------------------------------------------------------
    # Neuromodulation
    def update_neuromodulator(self, name: str, delta: float) -> float:
        """Update the concentration of a neuromodulator.

        Parameters
        ----------
        name: str
            Name of the neuromodulator to update.
        delta: float
            Increment (or decrement) to apply.

        Returns
        -------
        float
            The updated concentration level.
        """

        self.neuromodulators[name] = self.neuromodulators.get(name, 0.0) + delta
        return self.neuromodulators[name]

    # ------------------------------------------------------------------
    # Circadian rhythm
    def step_circadian(self, hours: float) -> float:
        """Advance the circadian clock by ``hours``.

        The clock is wrapped into a 24 hour cycle.
        """

        self.circadian_time = (self.circadian_time + hours) % 24
        return self.circadian_time

    # ------------------------------------------------------------------
    # Synaptic plasticity
    def adapt_synaptic_strengths(
        self, weights: List[float], rule: str, **kwargs
    ) -> List[float]:
        """Apply a plasticity rule to a list of weights.

        Parameters
        ----------
        weights: List[float]
            Synaptic strengths to update.
        rule: str
            Key identifying the plasticity rule.  Must be one of
            ``"triple_stdp"``, ``"bcm"``, ``"oja"`` or ``"synaptic_scaling"``.
        **kwargs: Any
            Additional parameters forwarded to the rule's ``update`` method.
        """

        updater = self.plasticity_rules[rule]
        return [updater.update(w, **kwargs) for w in weights]

    # ------------------------------------------------------------------
    # Developmental processes
    def simulate_development(self, event: str, amount: float) -> DevelopmentState:
        """Simulate a developmental process.

        Supported events are ``"neurogenesis"``, ``"pruning"`` and
        ``"myelination"``.  Synapse counts are prevented from dropping below
        zero while myelination is capped in the ``[0, 1]`` range.
        """

        if event == "neurogenesis":
            self.development.neurons += int(amount)
            self.development.synapses += int(amount)
        elif event == "pruning":
            self.development.synapses = max(0, self.development.synapses - int(amount))
        elif event == "myelination":
            self.development.myelination = min(
                1.0, self.development.myelination + amount
            )
        else:
            raise ValueError(f"Unknown developmental event: {event}")
        return self.development
