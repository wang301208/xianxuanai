import os
import sys
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.brain.biology.realism import BiologicalRealismEnhancer


def test_neuromodulation_update():
    enhancer = BiologicalRealismEnhancer()
    initial = enhancer.neuromodulators["dopamine"]
    updated = enhancer.update_neuromodulator("dopamine", 0.5)
    assert updated == pytest.approx(initial + 0.5)


def test_circadian_progression():
    enhancer = BiologicalRealismEnhancer()
    assert enhancer.step_circadian(5) == pytest.approx(5)
    # wrap around 24h cycle
    assert enhancer.step_circadian(20) == pytest.approx(1)


def test_plasticity_rule_application():
    enhancer = BiologicalRealismEnhancer()
    weights = [0.5]
    new_weights = enhancer.adapt_synaptic_strengths(
        weights, "oja", pre=0.6, post=0.8, lr=0.1
    )
    assert new_weights[0] == pytest.approx(0.516)


def test_developmental_state_changes():
    enhancer = BiologicalRealismEnhancer()
    state = enhancer.simulate_development("neurogenesis", 5)
    assert state.neurons == 5 and state.synapses == 5
    state = enhancer.simulate_development("pruning", 2)
    assert state.synapses == 3
    state = enhancer.simulate_development("myelination", 0.3)
    assert state.myelination == pytest.approx(0.3)
