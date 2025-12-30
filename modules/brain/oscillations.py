import numpy as np
from dataclasses import dataclass
from typing import Optional, Sequence

from .dynamics import CriticalDynamicsModel, IonChannelModel


@dataclass
class KuramotoModel:
    """Simple Kuramoto phase synchronization model."""

    def simulate(
        self,
        natural_frequencies: Sequence[float],
        coupling_strength: float,
        initial_phases: Sequence[float],
        duration: float,
        sample_rate: int,
    ) -> np.ndarray:
        dt = 1.0 / sample_rate
        steps = int(duration * sample_rate)
        omega = np.array(natural_frequencies)
        phases = np.zeros((steps, len(omega)))
        phases[0] = np.array(initial_phases)
        for t in range(1, steps):
            theta = phases[t - 1]
            theta_diff = theta[None, :] - theta[:, None]
            coupling = np.sum(np.sin(theta_diff), axis=1)
            dtheta = omega + (coupling_strength / len(omega)) * coupling
            phases[t] = theta + dtheta * dt
        return phases


@dataclass
class WilsonCowanModel:
    """Wilson-Cowan excitatory-inhibitory population model."""

    a_e: float = 1.0
    a_i: float = 1.0
    theta_e: float = 0.0
    theta_i: float = 0.0
    tau_e: float = 0.02
    tau_i: float = 0.02
    w_ee: float = 10.0
    w_ei: float = 10.0
    w_ie: float = 10.0
    w_ii: float = 10.0

    def _sigmoid(self, x: np.ndarray, a: float, theta: float) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-a * (x - theta)))

    def simulate(
        self,
        duration: float = 1.0,
        dt: float = 0.001,
        P_e: float = 0.0,
        P_i: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        steps = int(duration / dt)
        E = np.zeros(steps)
        I = np.zeros(steps)
        for t in range(1, steps):
            dE = (
                -E[t - 1]
                + self._sigmoid(
                    self.w_ee * E[t - 1] - self.w_ei * I[t - 1] + P_e,
                    self.a_e,
                    self.theta_e,
                )
            ) / self.tau_e
            dI = (
                -I[t - 1]
                + self._sigmoid(
                    self.w_ie * E[t - 1] - self.w_ii * I[t - 1] + P_i,
                    self.a_i,
                    self.theta_i,
                )
            ) / self.tau_i
            E[t] = E[t - 1] + dt * dE
            I[t] = I[t - 1] + dt * dI
        return E, I


@dataclass
class Wave:
    """Represents a bound oscillatory wave."""

    region: str
    frequency: float
    phase: float
    data: np.ndarray


class NeuralOscillations:
    """Simulate neural oscillatory generators with basic waveform support."""

    class AlphaGenerator:
        def generate_wave(
            self,
            frequency: float = 10.0,
            duration: float = 1.0,
            phase: float = 0.0,
            sample_rate: int = 1000,
        ) -> np.ndarray:
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            return np.sin(2 * np.pi * frequency * t + phase)

        def bind(
            self,
            region: str,
            frequency: float = 10.0,
            phase: float = 0.0,
            duration: float = 1.0,
            sample_rate: int = 1000,
            lock_to: Optional[Wave] = None,
        ) -> Wave:
            if lock_to is not None:
                phase = lock_to.phase
            data = self.generate_wave(frequency, duration, phase, sample_rate)
            return Wave(region, frequency, phase, data)

    class BetaGenerator:
        def generate_wave(
            self,
            frequency: float = 20.0,
            duration: float = 1.0,
            phase: float = 0.0,
            sample_rate: int = 1000,
        ) -> np.ndarray:
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            return np.sin(2 * np.pi * frequency * t + phase)

        def bind(
            self,
            region: str,
            frequency: float = 20.0,
            phase: float = 0.0,
            duration: float = 1.0,
            sample_rate: int = 1000,
            lock_to: Optional[Wave] = None,
        ) -> Wave:
            if lock_to is not None:
                phase = lock_to.phase
            data = self.generate_wave(frequency, duration, phase, sample_rate)
            return Wave(region, frequency, phase, data)

    class GammaGenerator:
        def generate_wave(
            self,
            frequency: float = 40.0,
            duration: float = 1.0,
            phase: float = 0.0,
            sample_rate: int = 1000,
        ) -> np.ndarray:
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            return np.sin(2 * np.pi * frequency * t + phase)

        def bind(
            self,
            region: str,
            frequency: float = 40.0,
            phase: float = 0.0,
            duration: float = 1.0,
            sample_rate: int = 1000,
            lock_to: Optional[Wave] = None,
        ) -> Wave:
            if lock_to is not None:
                phase = lock_to.phase
            data = self.generate_wave(frequency, duration, phase, sample_rate)
            return Wave(region, frequency, phase, data)

    class ThetaGenerator:
        def generate_wave(
            self,
            frequency: float = 6.0,
            duration: float = 1.0,
            phase: float = 0.0,
            sample_rate: int = 1000,
        ) -> np.ndarray:
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            return np.sin(2 * np.pi * frequency * t + phase)

        def bind(
            self,
            region: str,
            frequency: float = 6.0,
            phase: float = 0.0,
            duration: float = 1.0,
            sample_rate: int = 1000,
            lock_to: Optional[Wave] = None,
        ) -> Wave:
            if lock_to is not None:
                phase = lock_to.phase
            data = self.generate_wave(frequency, duration, phase, sample_rate)
            return Wave(region, frequency, phase, data)

    def __init__(self) -> None:
        self.alpha_waves = self.AlphaGenerator()
        self.beta_waves = self.BetaGenerator()
        self.gamma_waves = self.GammaGenerator()
        self.theta_waves = self.ThetaGenerator()
        self.kuramoto = KuramotoModel()
        self.wilson_cowan = WilsonCowanModel()
        self.ion_channel = IonChannelModel()
        self.critical_dynamics = CriticalDynamicsModel()

    def generate_realistic_oscillations(
        self,
        num_oscillators: int = 2,
        duration: float = 1.0,
        sample_rate: int = 1000,
        coupling_strength: float = 1.0,
        stimulus: float = 1.5,
        criticality: float = 1.0,
    ) -> np.ndarray:
        """Generate synchronized oscillations using multiple neural models.

        The resulting signals combine Wilson-Cowan population dynamics,
        ion-channel bursting and critical network modulation before phase
        coupling via the Kuramoto model.
        """

        dt = 1.0 / sample_rate
        steps = int(duration / dt)
        E, I = self.wilson_cowan.simulate(duration=duration, dt=dt, P_e=1.25, P_i=0.5)
        base_wave = E - I
        ion = self.ion_channel.simulate(duration=duration, dt=dt, I=stimulus)
        ion /= np.max(np.abs(ion)) + 1e-6
        activity = self.critical_dynamics.simulate(num_oscillators, steps, criticality)
        natural_freqs = np.linspace(30.0, 50.0, num_oscillators) * 2 * np.pi
        initial_phases = np.linspace(0, np.pi, num_oscillators)
        phases = self.kuramoto.simulate(
            natural_freqs,
            coupling_strength,
            initial_phases,
            duration,
            sample_rate,
        )
        return np.array([
            base_wave * ion * activity[:, i] * np.sin(phases[:, i])
            for i in range(num_oscillators)
        ])

    def synchronize_regions(
        self,
        regions,
        frequency: float = 40.0,
        phase: float = 0.0,
        duration: float = 1.0,
        sample_rate: int = 1000,
    ):
        """Synchronize a list of regions using gamma oscillations with phase locking."""
        if not regions:
            return []
        reference = self.gamma_waves.bind(
            regions[0], frequency=frequency, phase=phase, duration=duration, sample_rate=sample_rate
        )
        bindings = [reference]
        for region in regions[1:]:
            bindings.append(
                self.gamma_waves.bind(
                    region,
                    frequency=frequency,
                    duration=duration,
                    sample_rate=sample_rate,
                    lock_to=reference,
                )
            )
        return bindings

    @staticmethod
    def oscillatory_modulation(carrier_wave: np.ndarray, modulator_wave: np.ndarray) -> np.ndarray:
        """Amplitude modulation of a carrier wave using a modulating wave."""
        return carrier_wave * (1 + modulator_wave)

    def cross_frequency_coupling(
        self,
        low_freq: float,
        high_freq: float,
        duration: float = 1.0,
        sample_rate: int = 1000,
        coupling_strength: float = 1.0,
    ) -> np.ndarray:
        """Cross-frequency coupling via Wilson-Cowan amplitude and Kuramoto phase dynamics."""
        dt = 1.0 / sample_rate
        E_low, I_low = self.wilson_cowan.simulate(duration=duration, dt=dt, P_e=1.25, P_i=0.5)
        low_amp = E_low - I_low
        natural_freqs = [2 * np.pi * low_freq, 2 * np.pi * high_freq]
        phases = self.kuramoto.simulate(
            natural_freqs,
            coupling_strength,
            [0.0, np.pi / 2],
            duration,
            sample_rate,
        )
        low_signal = low_amp * np.sin(phases[:, 0])
        high_signal = np.sin(phases[:, 1])
        return self.oscillatory_modulation(high_signal, low_signal)
