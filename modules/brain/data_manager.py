"""Compression, indexing and storage utilities for spike trains.

This module provides simple components that demonstrate how neural spike
trains could be compressed, indexed and stored across a set of storage
nodes.  The focus is on clear, easily testable behaviour rather than
state‑of‑the‑art performance.

Classes
-------
SpikeDataCompressor
    Uses ``zlib`` to compress arrays of spike timestamps.
NeuralDataIndexer
    Builds a tiny sliding‑window pattern index for later lookup.
DistributedNeuralStorage
    Persists compressed spike trains across a number of directories,
    simulating distribution across storage nodes.

Functions
---------
store_spike_train
    Compress, index and store a spike train.
query_neural_patterns
    Query stored spike trains for a pattern using the prebuilt index in
    parallel.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from concurrent.futures import ThreadPoolExecutor
import zlib

import numpy as np


@dataclass
class SpikeDataCompressor:
    """Compress and decompress spike trains using ``zlib``."""

    def compress(self, spikes: Iterable[int]) -> bytes:
        arr = np.asarray(list(spikes), dtype=np.int32)
        return zlib.compress(arr.tobytes())

    def decompress(self, data: bytes) -> np.ndarray:
        if not data:
            return np.array([], dtype=np.int32)
        arr_bytes = zlib.decompress(data)
        return np.frombuffer(arr_bytes, dtype=np.int32)

    def compression_ratio(self, spikes: Iterable[int], compressed: bytes) -> float:
        arr = np.asarray(list(spikes), dtype=np.int32)
        original_size = arr.nbytes or 1
        return len(compressed) / original_size


@dataclass
class NeuralDataIndexer:
    """Index spike trains using a sliding window of fixed size."""

    window_size: int = 5
    index: Dict[Tuple[int, Tuple[int, ...]], List[Tuple[str, int]]] = field(
        default_factory=dict
    )

    def index_train(self, neuron_id: str, spikes: Iterable[int]) -> None:
        arr = np.asarray(list(spikes), dtype=np.int32)
        if arr.size < self.window_size:
            return
        w = self.window_size
        for i in range(arr.size - w + 1):
            window = arr[i : i + w]
            pattern = tuple(window - window[0])
            self.index.setdefault((w, pattern), []).append((neuron_id, i))

    def query(self, pattern: Iterable[int]) -> List[str]:
        arr = np.asarray(list(pattern), dtype=np.int32)
        key = (arr.size, tuple(arr - arr[0]))
        matches = self.index.get(key, [])
        return [neuron_id for neuron_id, _ in matches]


@dataclass
class DistributedNeuralStorage:
    """Very small utility that shards data across directories."""

    node_paths: List[Path]

    def __post_init__(self) -> None:
        self.node_paths = [Path(p) for p in self.node_paths]
        for path in self.node_paths:
            path.mkdir(parents=True, exist_ok=True)

    def _path_for(self, neuron_id: str) -> Path:
        idx = hash(neuron_id) % len(self.node_paths)
        return self.node_paths[idx] / f"{neuron_id}.bin"

    def save(self, neuron_id: str, data: bytes) -> None:
        self._path_for(neuron_id).write_bytes(data)

    def load(self, neuron_id: str) -> bytes:
        return self._path_for(neuron_id).read_bytes()


def store_spike_train(
    neuron_id: str,
    spike_train: Iterable[int],
    compressor: SpikeDataCompressor,
    indexer: NeuralDataIndexer,
    storage: DistributedNeuralStorage,
) -> float:
    """Compress, index and store a spike train.

    Returns the compression ratio (compressed_size/original_size).
    """

    compressed = compressor.compress(spike_train)
    storage.save(neuron_id, compressed)
    indexer.index_train(neuron_id, spike_train)
    return compressor.compression_ratio(spike_train, compressed)


def query_neural_patterns(
    pattern: Iterable[int],
    indexer: NeuralDataIndexer,
    storage: DistributedNeuralStorage,
    compressor: SpikeDataCompressor,
) -> List[str]:
    """Return IDs of neurons whose spike trains contain ``pattern``."""

    candidates = indexer.query(pattern)
    if not candidates:
        return []

    pattern_arr = np.asarray(list(pattern), dtype=np.int32)
    window = pattern_arr.size

    def _match(neuron_id: str) -> str | None:
        data = storage.load(neuron_id)
        spikes = compressor.decompress(data)
        for i in range(spikes.size - window + 1):
            if np.array_equal(spikes[i : i + window], pattern_arr):
                return neuron_id
        return None

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(_match, candidates))
    return [r for r in results if r]


__all__ = [
    "SpikeDataCompressor",
    "NeuralDataIndexer",
    "DistributedNeuralStorage",
    "store_spike_train",
    "query_neural_patterns",
]
