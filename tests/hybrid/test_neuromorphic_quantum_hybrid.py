import numpy as np

from modules.brain.hybrid.neuromorphic_quantum_hybrid import NeuromorphicQuantumHybrid


def test_hybrid_outperforms_single_modes():
    hybrid = NeuromorphicQuantumHybrid()
    signal = [0.49, 0.51, 0.52]
    target = np.array([0.0, 1.0, 1.0], dtype=float)
    target /= np.linalg.norm(target)

    hybrid_state = hybrid.hybrid_processing(signal, target_state=target)
    hybrid_err = np.linalg.norm(hybrid_state - target)

    features = hybrid.neuromorphic_preprocess(signal)
    neu_err = np.linalg.norm(np.array(features, dtype=float) - target)

    quantum_state = hybrid.quantum_feature_map(signal)
    quantum_err = np.linalg.norm(quantum_state - target)

    assert hybrid_err < neu_err
    assert hybrid_err < quantum_err
