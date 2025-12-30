"""Dynamic neural models used by oscillatory simulations."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class IonChannelModel:
    """Leaky integrate-and-fire neuron model.

    A minimalist approximation of ion channel dynamics that produces
    spikes when the membrane potential crosses a threshold. The number of
    spikes depends on the input current ``I``.
    """

    threshold: float = 1.0
    reset: float = 0.0
    tau: float = 0.02

    def simulate(self, duration: float = 1.0, dt: float = 0.001, I: float = 1.5) -> np.ndarray:
        """Simulate membrane potential for the given duration.

        Args:
            duration: Simulation length in seconds.
            dt: Time step in seconds.
            I: Input current driving the neuron.

        Returns:
            Array of membrane potential values over time.
        """
        steps = int(duration / dt)
        v = np.zeros(steps)
        for t in range(1, steps):
            dv = (-v[t - 1] + I) / self.tau
            v[t] = v[t - 1] + dv * dt
            if v[t] >= self.threshold:
                v[t - 1] = self.threshold
                v[t] = self.reset
        return v


@dataclass
class CriticalDynamicsModel:
    """Mean-field model capturing network criticality effects.

    The model generates a spatiotemporal activity pattern for a network of
    oscillators. Higher ``criticality`` values lead to more synchronized
    activity across oscillators.
    """

    noise: float = 0.1

    def simulate(
        self,
        num_oscillators: int,
        steps: int,
        criticality: float = 1.0,
    ) -> np.ndarray:
        """Generate activity modulation for each oscillator.

        Args:
            num_oscillators: Number of oscillators in the network.
            steps: Number of simulation steps.
            criticality: Coupling factor determining synchrony. Values
                near 1 produce high synchrony, while values near 0 yield
                independent activity.

        Returns:
            Array of shape ``(steps, num_oscillators)`` representing
            activity modulation for each oscillator over time.
        """
        rng = np.random.default_rng(0)
        global_state = np.zeros(steps)
        global_state[0] = rng.normal(0, self.noise)
        for t in range(1, steps):
            global_state[t] = (
                criticality * global_state[t - 1] + rng.normal(0, self.noise)
            )
        activity = np.tile(global_state[:, None], (1, num_oscillators))
        activity += (1 - criticality) * rng.normal(
            0, self.noise, size=(steps, num_oscillators)
        )
        return activity
