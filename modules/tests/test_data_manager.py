import time
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from modules.brain.data_manager import (
    DistributedNeuralStorage,
    NeuralDataIndexer,
    SpikeDataCompressor,
    query_neural_patterns,
    store_spike_train,
)


def test_data_manager_storage_and_query() -> None:
    rng = np.random.default_rng(0)
    spike_trains = [np.cumsum(rng.integers(1, 5, size=200)) for _ in range(5)]
    pattern = spike_trains[0][50:55]

    with TemporaryDirectory() as tmpdir:
        storage = DistributedNeuralStorage([Path(tmpdir) / "n0", Path(tmpdir) / "n1"])
        compressor = SpikeDataCompressor()
        indexer = NeuralDataIndexer(window_size=len(pattern))

        ratios = []
        for i, spikes in enumerate(spike_trains):
            ratio = store_spike_train(f"neuron{i}", spikes, compressor, indexer, storage)
            ratios.append(ratio)
        assert all(r < 1.0 for r in ratios)

        start = time.perf_counter()
        matches = query_neural_patterns(pattern, indexer, storage, compressor)
        elapsed = time.perf_counter() - start

        assert "neuron0" in matches
        assert elapsed < 1.0
