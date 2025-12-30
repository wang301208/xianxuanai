from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .utils import resample_1d, to_1d_float_array


class AuditorySystem:
    """Simplified cochlea → (MGN-like) thalamic vector encoder."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.thalamic_size = int(cfg.get("thalamic_size", 200))
        self.sample_rate = int(cfg.get("sample_rate", 16_000))
        self.min_hz = float(cfg.get("min_hz", 50.0))
        self.max_hz = float(cfg.get("max_hz", 8_000.0))

    def encode(self, waveform: Any, sample_rate: Optional[int] = None) -> np.ndarray:
        sr = int(sample_rate or self.sample_rate)
        signal = to_1d_float_array(waveform)
        if signal.size == 0:
            return np.zeros(self.thalamic_size, dtype=float)

        # Normalise amplitude for stability.
        peak = float(np.max(np.abs(signal))) if signal.size else 0.0
        if peak > 1e-9:
            signal = signal / peak

        # One-shot spectral snapshot (lightweight stand-in for cochlea filterbank).
        window = np.hanning(signal.size)
        spectrum = np.abs(np.fft.rfft(signal * window))
        freqs = np.fft.rfftfreq(signal.size, d=1.0 / float(sr))

        # Restrict to a frequency band.
        band = (freqs >= self.min_hz) & (freqs <= min(self.max_hz, float(sr) / 2.0))
        if not np.any(band):
            usable = spectrum
        else:
            usable = spectrum[band]

        features = np.log1p(usable)
        vec = resample_1d(features, self.thalamic_size)
        # Map to 0..1 range.
        if vec.size:
            vec = vec - float(np.min(vec))
            denom = float(np.max(vec)) + 1e-9
            vec = vec / denom
        return np.clip(vec, 0.0, 1.0)

