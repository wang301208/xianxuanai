from typing import Dict, Mapping, Optional


class Neuroplasticity:
    """Simplified neuroplasticity model with basic learning rules.

    This module includes placeholder implementations of a Hebbian learning rule,
    spike timing dependent plasticity (STDP) rule, and a homeostatic plasticity
    rule. These are highly simplified and serve only to demonstrate how such
    interfaces might be structured.
    """

    class HebbianRule:
        """Classic Hebbian learning: neurons that fire together wire together."""

        def update(self, pre_activity, post_activity):
            """Strengthen connections proportional to co-activation."""
            return pre_activity * post_activity

    class STDPRule:
        """Spike Timing Dependent Plasticity (STDP)."""

        def update(self, pre_activity, post_activity):
            """Adjust weights based on relative spike timing.

            A positive value indicates potentiation (pre before post), while a
            negative value indicates depression. Here we simplify this by
            returning the difference between post- and pre-synaptic activity.
            """

            return post_activity - pre_activity

    class HomeostaticRule:
        """Homeostatic plasticity maintaining overall activity levels."""

        def __init__(self, target_activity=0.0):
            self.target_activity = target_activity

        def update(self, activity):
            """Adjust activity towards a target level."""
            return self.target_activity - activity.mean()

    def __init__(self):
        self.hebbian = self.HebbianRule()
        self.spike_timing = self.STDPRule()
        self.homeostatic = self.HomeostaticRule()
        self.modulation_state: Dict[str, float] = {}
        self.learning_gain: float = 1.0

    def adapt_connections(self, pre_activity, post_activity):
        """Adapt synaptic connections using spike timing dependent plasticity."""
        delta = self.spike_timing.update(pre_activity, post_activity)
        gate = self.modulation_state.get("plasticity_gate", 1.0)
        gain = max(0.1, self.learning_gain)
        gated = delta * max(0.0, min(2.0, gate)) * gain
        return gated

    def update_modulation(self, modulation: Optional[Mapping[str, float]]) -> None:
        if not modulation:
            self.modulation_state = {}
            self.learning_gain = 1.0
            self.homeostatic.target_activity = 0.0
            return
        filtered: Dict[str, float] = {
            key: float(value)
            for key, value in modulation.items()
            if isinstance(value, (int, float))
        }
        amplitude = max(0.0, min(1.0, filtered.get("amplitude", filtered.get("amplitude_norm", 0.0))))
        synchrony = max(0.0, min(1.0, filtered.get("synchrony", filtered.get("synchrony_norm", 0.0))))
        rhythmicity = max(0.0, min(1.0, filtered.get("rhythmicity", 0.0)))
        gate = max(0.0, min(2.0, filtered.get("plasticity_gate", (amplitude + synchrony) * 0.5)))
        self.learning_gain = 0.8 + amplitude * 0.6 + rhythmicity * 0.3
        self.modulation_state = {
            **filtered,
            "plasticity_gate": gate,
        }
        self.homeostatic.target_activity = min(0.5, rhythmicity * 0.5 + synchrony * 0.25)
