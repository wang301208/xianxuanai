import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.brain import NeuralOscillations


def test_synchronize_regions():
    oscillations = NeuralOscillations()
    regions = ["hippocampus", "cortex"]
    result = oscillations.synchronize_regions(regions)
    assert [w.region for w in result] == regions
    assert all(w.frequency == 40.0 for w in result)
    assert len({w.phase for w in result}) == 1  # phase locked


def test_phase_locking():
    oscillations = NeuralOscillations()
    wave1 = oscillations.alpha_waves.bind("r1", frequency=10.0, phase=0.5)
    wave2 = oscillations.alpha_waves.bind("r2", frequency=10.0, lock_to=wave1)
    assert wave2.phase == wave1.phase


def test_cross_frequency_coupling_and_modulation():
    oscillations = NeuralOscillations()
    low = oscillations.theta_waves.generate_wave(frequency=6.0, duration=1.0, phase=0.0)
    high = oscillations.gamma_waves.generate_wave(frequency=40.0, duration=1.0, phase=0.0)
    coupled = oscillations.cross_frequency_coupling(low, high)
    modulated = oscillations.oscillatory_modulation(high, low)
    assert coupled.shape == low.shape == high.shape
    assert modulated.shape == high.shape
