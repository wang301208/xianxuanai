import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.getcwd()))

from modules.brain.oscillations import KuramotoModel, NeuralOscillations
from modules.brain.dynamics import CriticalDynamicsModel, IonChannelModel


def test_kuramoto_synchronization():
    model = KuramotoModel()
    natural_freqs = [2 * np.pi * 1.0, 2 * np.pi * 1.1, 2 * np.pi * 0.9]
    phases = model.simulate(
        natural_freqs,
        coupling_strength=10.0,
        initial_phases=[0.0, 1.0, 2.0],
        duration=1.0,
        sample_rate=1000,
    )
    final = phases[-1]
    initial_spread = 2.0 - 0.0
    final_spread = np.max(final) - np.min(final)
    assert final_spread < initial_spread


def test_generate_realistic_oscillations():
    osc = NeuralOscillations()
    waves = osc.generate_realistic_oscillations(
        num_oscillators=3, duration=0.5, sample_rate=1000, coupling_strength=5.0
    )
    assert waves.shape[0] == 3
    corr = np.corrcoef(waves)
    assert abs(corr[0, 1]) > 0 and abs(corr[1, 2]) > 0


def test_cross_frequency_coupling():
    osc = NeuralOscillations()
    duration = 0.5
    sample_rate = 1000
    result = osc.cross_frequency_coupling(
        low_freq=5.0,
        high_freq=40.0,
        duration=duration,
        sample_rate=sample_rate,
        coupling_strength=5.0,
    )
    dt = 1.0 / sample_rate
    E_low, I_low = osc.wilson_cowan.simulate(duration=duration, dt=dt, P_e=1.25, P_i=0.5)
    low_amp = E_low - I_low
    phases = osc.kuramoto.simulate(
        [2 * np.pi * 5.0, 2 * np.pi * 40.0],
        5.0,
        [0.0, np.pi / 2],
        duration,
        sample_rate,
    )
    low_signal = low_amp * np.sin(phases[:, 0])
    high_signal = np.sin(phases[:, 1])
    expected = osc.oscillatory_modulation(high_signal, low_signal)
    np.testing.assert_allclose(result, expected)


def test_ion_channel_model_bursting():
    model = IonChannelModel()
    v_high = model.simulate(duration=0.2, dt=0.001, I=1.5)
    v_low = model.simulate(duration=0.2, dt=0.001, I=0.5)
    spikes_high = np.sum((v_high[1:] >= model.threshold) & (v_high[:-1] < model.threshold))
    spikes_low = np.sum((v_low[1:] >= model.threshold) & (v_low[:-1] < model.threshold))
    assert spikes_high > spikes_low


def test_critical_dynamics_synchrony():
    model = CriticalDynamicsModel()
    activity_low = model.simulate(num_oscillators=5, steps=200, criticality=0.1)
    activity_high = model.simulate(num_oscillators=5, steps=200, criticality=0.9)
    corr_low = np.corrcoef(activity_low.T)
    corr_high = np.corrcoef(activity_high.T)
    mean_low = corr_low[np.triu_indices(5, k=1)].mean()
    mean_high = corr_high[np.triu_indices(5, k=1)].mean()
    assert mean_high > mean_low
