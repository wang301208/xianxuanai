from __future__ import annotations

"""Adapters for sensory cortices that expose structured encoder features.

The previous implementation returned placeholder string tokens such as
"edge" or "touch".  This module now bridges the numerically grounded encoders
in :mod:`modules.brain.perception` to the rest of the system, providing rich
feature dictionaries and an optional neuromorphic hand-off when a spiking
backend is attached.
"""

from typing import Any, Dict, Iterable, List, Mapping, Sequence

try:  # pragma: no cover - allow running in lightweight test environments
    import numpy as np
except Exception:  # pragma: no cover - minimal functional stub
    class _Array(list):
        def __init__(self, data: Iterable[float] | None = None) -> None:
            super().__init__(float(x) for x in (data or []))

        @property
        def ndim(self) -> int:
            return 1

        @property
        def size(self) -> int:
            return len(self)

        def reshape(self, *shape: int) -> "_Array" | List["_Array"]:
            if not shape:
                return self
            if len(shape) == 1:
                return self
            rows, cols = shape[0], shape[1] if len(shape) > 1 else len(self)
            matrix: List[_Array] = []
            values = list(self)
            for row in range(max(1, rows)):
                start = row * max(1, cols)
                end = start + max(1, cols)
                matrix.append(_Array(values[start:end]))
            return matrix

        def astype(self, _dtype: Any) -> "_Array":
            return _Array(self)

        def tolist(self) -> List[float]:
            return list(self)

    class _NumpyStub:
        float32 = float

        def asarray(self, data: Any, dtype: Any | None = None) -> _Array:
            if isinstance(data, _Array):
                return data
            if isinstance(data, (list, tuple)):
                return _Array(data)
            return _Array([data])

        def pad(self, arr: _Array, pad_width: tuple[int, int], mode: str = "constant") -> _Array:
            left, right = pad_width
            padded = [0.0] * max(0, left) + list(arr) + [0.0] * max(0, right)
            return _Array(padded)

        def array_split(self, arr: _Array, sections: int) -> List[_Array]:
            data = list(arr)
            if sections <= 0:
                return [_Array(data)]
            chunk = max(1, len(data) // sections)
            splits: List[_Array] = []
            for idx in range(sections):
                start = idx * chunk
                end = start + chunk
                splits.append(_Array(data[start:end]))
            return splits

        def min(self, arr: Iterable[float]) -> float:
            values = list(arr)
            return min(values) if values else 0.0

        def max(self, arr: Iterable[float]) -> float:
            values = list(arr)
            return max(values) if values else 0.0

        def abs(self, arr: Iterable[float]) -> List[float]:
            return [abs(x) for x in arr]

    np = _NumpyStub()  # type: ignore[assignment]

from .perception import AuditoryEncoder, EncodedSignal, TactileEncoder, VisualEncoder


def _as_array(vector: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.reshape(-1)


def _stage_payload(stage: str, signal: EncodedSignal, extra_features: Mapping[str, float]) -> Dict[str, Any]:
    features = {key: float(value) for key, value in signal.features.items()}
    features.update({key: float(value) for key, value in extra_features.items()})
    metadata = {**signal.metadata, "stage": stage}
    return {
        "vector": list(signal.vector),
        "features": features,
        "metadata": metadata,
    }


def _map_vector_to_neurons(vector: Sequence[float], n_neurons: int) -> List[float]:
    if n_neurons <= 0:
        return list(vector)
    arr = _as_array(vector)
    if arr.size == 0:
        return [0.0 for _ in range(n_neurons)]
    if arr.size == n_neurons:
        return arr.astype(float).tolist()
    if arr.size < n_neurons:
        padded = np.pad(arr, (0, n_neurons - arr.size), mode="constant")
        return padded.astype(float).tolist()
    splits = np.array_split(arr, n_neurons)
    return [float(split.mean()) for split in splits]


def _normalize_for_spiking(currents: Iterable[float]) -> List[float]:
    arr = _as_array(list(currents))
    if arr.size == 0:
        return []
    min_val = float(np.min(arr))
    if min_val < 0.0:
        arr = arr - min_val
    max_val = float(np.max(np.abs(arr)))
    if max_val > 0.0 and max_val < 1.0:
        arr = arr / max_val
    return arr.astype(float).tolist()


def _build_neuromorphic_payload(backend: Any, signal: EncodedSignal) -> Dict[str, Any]:
    neurons = getattr(getattr(backend, "neurons", None), "size", 0) or 0

    def _compressed(vec: Sequence[float]) -> List[float]:
        mapped = _map_vector_to_neurons(vec, int(neurons))
        if not mapped and neurons:
            mapped = [0.0 for _ in range(int(neurons))]
        return _normalize_for_spiking(mapped)

    arr = _as_array(signal.vector)
    frames = int(signal.metadata.get("frames", 0) or 0)
    mels = int(signal.metadata.get("mels", 0) or 0)
    input_sequence: List[List[float]]
    if frames and mels and arr.size == frames * mels:
        mel_matrix = arr.reshape(frames, mels)
        input_sequence = [_compressed(row) for row in mel_matrix]
    else:
        input_sequence = [_compressed(signal.vector)]
    input_sequence = [seq for seq in input_sequence if seq]
    events = backend.run(input_sequence) if input_sequence else []
    if hasattr(backend, "synapses") and hasattr(backend.synapses, "adapt"):
        backend.synapses.adapt(backend.spike_times, backend.spike_times)
    n_channels = len(input_sequence[0]) if input_sequence else len(_map_vector_to_neurons(signal.vector, int(neurons)))
    spike_counts = [0 for _ in range(n_channels)]
    formatted_events: List[List[Any]] = []
    if isinstance(events, list):
        for index, event in enumerate(events):
            if isinstance(event, tuple) and len(event) == 2:
                time, spikes = event
            else:
                time, spikes = index, event
            spikes_list = [int(s) for s in spikes]
            for idx, spike in enumerate(spikes_list):
                if idx >= len(spike_counts):
                    spike_counts.extend([0] * (idx + 1 - len(spike_counts)))
                spike_counts[idx] += spike
            formatted_events.append([float(time), spikes_list])
    payload: Dict[str, Any] = {
        "encoded_vector": list(signal.vector),
        "inputs": input_sequence,
        "events": formatted_events,
        "spike_counts": spike_counts,
    }
    if hasattr(backend, "energy_usage"):
        payload["energy_used"] = float(getattr(backend, "energy_usage", 0.0))
    return payload


class EdgeDetector:
    """Adapter producing numeric edge-centric features via ``VisualEncoder``."""

    def __init__(self, encoder: VisualEncoder | None = None) -> None:
        self.encoder = encoder or VisualEncoder()

    def detect(self, image: Any, signal: EncodedSignal | None = None) -> Dict[str, Any]:
        encoded = signal or self.encoder.encode(image)
        arr = _as_array(encoded.vector)
        density = float(np.mean(arr > 0.2)) if arr.size else 0.0
        orientation_hist = encoded.metadata.get("orientation_histogram")
        if not orientation_hist:
            orientation_hist = getattr(self.encoder, "last_orientation_histogram", None) or []
        angles = encoded.metadata.get("orientation_angles") or [0, 45, 90, 135]
        orientation_map = dict(zip(angles, orientation_hist)) if orientation_hist else {}
        orientation_values = np.asarray(list(orientation_map.values()), dtype=np.float32)
        orientation_entropy = 0.0
        dominant_orientation = 0.0
        orientation_strength = 0.0
        if orientation_values.size:
            probs = orientation_values / (orientation_values.sum() or 1.0)
            orientation_entropy = float(-(probs * np.log(probs + 1e-9)).sum())
            dominant_index = int(np.argmax(probs))
            dominant_orientation = float(angles[dominant_index])
            orientation_strength = float(orientation_values.max())
        multi_scale = encoded.metadata.get("multi_scale_contrast")
        if not multi_scale:
            multi_scale = getattr(self.encoder, "_last_contrast", None) or []
        multi_scale_mean = float(np.mean(multi_scale)) if multi_scale else 0.0
        edge_variability = float(arr.std()) if arr.size else 0.0
        return _stage_payload(
            "V1",
            encoded,
            {
                "edge_density": density,
                "edge_variability": edge_variability,
                "orientation_entropy": orientation_entropy,
                "dominant_orientation": dominant_orientation,
                "orientation_strength": orientation_strength,
                "multi_scale_mean": multi_scale_mean,
            },
        )


class V1:
    def __init__(self, encoder: VisualEncoder | None = None) -> None:
        self.edge_detector = EdgeDetector(encoder)

    def process(self, image: Any, signal: EncodedSignal | None = None) -> Dict[str, Any]:
        return self.edge_detector.detect(image, signal)


class V2:
    def __init__(self, encoder: VisualEncoder | None = None) -> None:
        self.encoder = encoder or VisualEncoder()

    def process(self, image: Any, signal: EncodedSignal | None = None) -> Dict[str, Any]:
        encoded = signal or self.encoder.encode(image)
        arr = _as_array(encoded.vector)
        if arr.size:
            side = int(np.sqrt(arr.size))
            if side * side == arr.size:
                grid = arr.reshape(side, side)
            else:
                grid = arr.reshape(1, -1)
            left = float(grid[:, : grid.shape[1] // 2].mean()) if grid.size else 0.0
            right = float(grid[:, grid.shape[1] // 2 :].mean()) if grid.size else left
            top = float(grid[: grid.shape[0] // 2, :].mean()) if grid.size else 0.0
            bottom = (
                float(grid[grid.shape[0] // 2 :, :].mean()) if grid.size else top
            )
            complexity = float(grid.std())
            symmetry = float(abs(left - right) + abs(top - bottom)) / 2.0
        else:
            complexity = 0.0
            symmetry = 0.0
        orientation_balance = float(
            abs(
                encoded.features.get("orientation_0", 0.0)
                - encoded.features.get("orientation_90", 0.0)
            )
        )
        scale_mean = float(encoded.features.get("multi_scale_contrast", 0.0))
        return _stage_payload(
            "V2",
            encoded,
            {
                "form_complexity": complexity,
                "form_symmetry": symmetry,
                "orientation_balance": orientation_balance,
                "scale_contrast": scale_mean,
            },
        )


class V4:
    def __init__(self, encoder: VisualEncoder | None = None) -> None:
        self.encoder = encoder or VisualEncoder()

    def process(self, image: Any, signal: EncodedSignal | None = None) -> Dict[str, Any]:
        encoded = signal or self.encoder.encode(image)
        arr = _as_array(encoded.vector)
        color_salience = float(arr.max() - arr.min()) if arr.size else 0.0
        orientation_strength = encoded.features.get("orientation_0", 0.0) + encoded.features.get(
            "orientation_90", 0.0
        )
        multi_scale = encoded.features.get("multi_scale_contrast", 0.0)
        global_salience = float(multi_scale * (1.0 + orientation_strength))
        return _stage_payload(
            "V4",
            encoded,
            {
                "color_salience": color_salience,
                "global_salience": global_salience,
                "mean_intensity": float(encoded.features.get("mean_intensity", 0.0)),
            },
        )


class MT:
    def __init__(self, encoder: VisualEncoder | None = None) -> None:
        self.encoder = encoder or VisualEncoder()
        self._previous_frame: np.ndarray | None = None

    def process(self, image: Any, signal: EncodedSignal | None = None) -> Dict[str, Any]:
        encoded = signal or self.encoder.encode(image)
        frame = getattr(self.encoder, "last_frame", None)
        motion_energy = 0.0
        motion_bias = 0.0
        motion_velocity = 0.0
        temporal_consistency = 0.0
        if frame is not None and self._previous_frame is not None:
            if frame.shape == self._previous_frame.shape:
                delta = frame - self._previous_frame
                motion_energy = float(np.mean(np.abs(delta)))
                motion_bias = float(delta.mean())
                motion_velocity = float(np.linalg.norm(delta) / (delta.size + 1e-6))
                try:
                    flattened_current = frame.flatten()
                    flattened_previous = self._previous_frame.flatten()
                    if flattened_current.size and flattened_previous.size:
                        temporal_consistency = float(
                            np.corrcoef(flattened_current, flattened_previous)[0, 1]
                        )
                except Exception:
                    temporal_consistency = 0.0
        else:
            arr = _as_array(encoded.vector)
            if arr.size > 1:
                motion_energy = float(np.mean(np.abs(np.diff(arr))))
                motion_bias = float(arr[-1] - arr[0])
                motion_velocity = motion_energy
        self._previous_frame = None if frame is None else np.array(frame, copy=True)
        return _stage_payload(
            "MT",
            encoded,
            {
                "motion_energy": motion_energy,
                "motion_bias": motion_bias,
                "motion_velocity": motion_velocity,
                "temporal_consistency": temporal_consistency,
            },
        )


class VisualCortex:
    """Visual cortex with hierarchical processing areas backed by encoders."""

    def __init__(self, spiking_backend: Any | None = None) -> None:
        self.encoder = VisualEncoder()
        self.v1 = V1(self.encoder)
        self.v2 = V2(self.encoder)
        self.v4 = V4(self.encoder)
        self.mt = MT(self.encoder)
        self.spiking_backend = spiking_backend

    def process(self, image: Any) -> Dict[str, Any]:
        encoded = self.encoder.encode(image)
        result = {
            "v1": self.v1.process(image, encoded),
            "v2": self.v2.process(image, encoded),
            "v4": self.v4.process(image, encoded),
            "mt": self.mt.process(image, encoded),
        }
        if self.spiking_backend:
            result["neuromorphic"] = _build_neuromorphic_payload(
                self.spiking_backend, encoded
            )
        return result


class FrequencyAnalyzer:
    """Adapter extracting spectral summaries with ``AuditoryEncoder``."""

    def __init__(self, encoder: AuditoryEncoder | None = None) -> None:
        self.encoder = encoder or AuditoryEncoder()

    def analyze(self, sound: Any, signal: EncodedSignal | None = None) -> Dict[str, Any]:
        encoded = signal or self.encoder.encode(sound)
        arr = _as_array(encoded.vector)
        frames = int(encoded.metadata.get("frames", 0) or 0)
        mels = int(encoded.metadata.get("mels", 0) or 0)
        if frames and mels and arr.size == frames * mels:
            mel_matrix = arr.reshape(frames, mels)
        elif mels and arr.size >= mels:
            mel_matrix = arr.reshape(-1, mels)
        else:
            mel_matrix = arr.reshape(1, -1)
        band_profile = mel_matrix.mean(axis=0) if mel_matrix.size else np.zeros(0)
        if band_profile.size:
            dominant_band = int(np.argmax(band_profile))
            band_energy = float(band_profile[dominant_band])
        else:
            dominant_band = 0
            band_energy = 0.0
        features = {
            "dominant_band": float(dominant_band),
            "band_energy": band_energy,
            "spectral_flux": float(encoded.features.get("spectral_flux", 0.0)),
            "spectral_rolloff": float(encoded.features.get("spectral_rolloff", 0.0)),
            "temporal_modulation": float(encoded.features.get("temporal_modulation", 0.0)),
        }
        if band_profile.size > 1:
            features["bandwidth"] = float(np.std(band_profile))
        return _stage_payload(
            "A1",
            encoded,
            features,
        )


class A1:
    def __init__(self, encoder: AuditoryEncoder | None = None) -> None:
        self.analyzer = FrequencyAnalyzer(encoder)

    def process(self, sound: Any, signal: EncodedSignal | None = None) -> Dict[str, Any]:
        return self.analyzer.analyze(sound, signal)


class A2:
    def __init__(self, encoder: AuditoryEncoder | None = None) -> None:
        self.encoder = encoder or AuditoryEncoder()

    def process(self, sound: Any, signal: EncodedSignal | None = None) -> Dict[str, Any]:
        encoded = signal or self.encoder.encode(sound)
        arr = _as_array(encoded.vector)
        frames = int(encoded.metadata.get("frames", 0) or 0)
        mels = int(encoded.metadata.get("mels", 0) or 0)
        if frames and mels and arr.size == frames * mels:
            mel_matrix = arr.reshape(frames, mels)
            temporal_profile = mel_matrix.mean(axis=1)
            temporal_variance = float(np.var(temporal_profile)) if temporal_profile.size else 0.0
            spectral_spread = float(np.var(mel_matrix, axis=0).mean()) if mel_matrix.size else 0.0
        else:
            temporal_variance = float(np.var(arr)) if arr.size else 0.0
            spectral_spread = 0.0
        features = {
            "temporal_variance": temporal_variance,
            "spectral_spread": spectral_spread,
            "spectral_flux": float(encoded.features.get("spectral_flux", 0.0)),
            "temporal_modulation": float(encoded.features.get("temporal_modulation", 0.0)),
        }
        return _stage_payload(
            "A2",
            encoded,
            features,
        )


class AuditoryCortex:
    """Auditory cortex with primary and secondary areas."""

    def __init__(self, spiking_backend: Any | None = None) -> None:
        self.encoder = AuditoryEncoder()
        self.a1 = A1(self.encoder)
        self.a2 = A2(self.encoder)
        self.spiking_backend = spiking_backend

    def process(self, sound: Any) -> Dict[str, Any]:
        encoded = self.encoder.encode(sound)
        result = {
            "a1": self.a1.process(sound, encoded),
            "a2": self.a2.process(sound, encoded),
        }
        if self.spiking_backend:
            result["neuromorphic"] = _build_neuromorphic_payload(
                self.spiking_backend, encoded
            )
        return result


class TouchProcessor:
    """Adapter transforming tactile stimuli via ``TactileEncoder``."""

    def __init__(self, encoder: TactileEncoder | None = None) -> None:
        self.encoder = encoder or TactileEncoder()

    def process(self, stimulus: Any, signal: EncodedSignal | None = None) -> Dict[str, Any]:
        encoded = signal or self.encoder.encode(stimulus)
        arr = _as_array(encoded.vector)
        grid_shape = encoded.metadata.get("grid")
        if grid_shape and len(grid_shape) == 2 and arr.size == int(grid_shape[0]) * int(grid_shape[1]):
            height, width = int(grid_shape[0]), int(grid_shape[1])
            grid = arr.reshape(height, width)
            central = grid[height // 2, width // 2]
            edge_values = np.concatenate((grid[0], grid[-1], grid[:, 0], grid[:, -1]))
            edge_pressure = float(edge_values.mean()) if edge_values.size else float(central)
            central_pressure = float(central)
        else:
            central_pressure = float(arr.mean()) if arr.size else 0.0
            edge_pressure = central_pressure
        return _stage_payload(
            "S1",
            encoded,
            {
                "central_pressure": central_pressure,
                "edge_pressure": edge_pressure,
                "gradient_energy": float(encoded.features.get("gradient_energy", 0.0)),
                "shear_ratio": float(encoded.features.get("shear_ratio", 0.0)),
                "pressure_stability": float(abs(central_pressure - edge_pressure)),
            },
        )


class SomatosensoryCortex:
    """Somatosensory cortex for processing tactile information."""

    def __init__(self, spiking_backend: Any | None = None) -> None:
        self.encoder = TactileEncoder()
        self.processor = TouchProcessor(self.encoder)
        self.spiking_backend = spiking_backend

    def process(self, stimulus: Any) -> Dict[str, Any]:
        encoded = self.encoder.encode(stimulus)
        result = {
            "s1": self.processor.process(stimulus, encoded),
        }
        if self.spiking_backend:
            result["neuromorphic"] = _build_neuromorphic_payload(
                self.spiking_backend, encoded
            )
        return result
