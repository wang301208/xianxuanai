"""
Auditory cortex processing module.

Provides a lightweight feature extraction pipeline for audio waveforms with
optional integration of third-party libraries and a numpy-only fallback that
computes spectral summaries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np

try:  # pragma: no cover - optional dependency
    import librosa
except Exception:  # pragma: no cover - librosa may be unavailable
    librosa = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover - torch may be unavailable
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import whisper
except Exception:  # pragma: no cover - whisper may be unavailable
    whisper = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from transformers import Wav2Vec2Model, Wav2Vec2Processor
except Exception:  # pragma: no cover - transformers may be unavailable
    Wav2Vec2Model = None  # type: ignore[assignment]
    Wav2Vec2Processor = None  # type: ignore[assignment]


_LOGGER = logging.getLogger(__name__)

class AuditoryModelUnavailable(RuntimeError):
    """Raised when the requested backend cannot be created."""


def _to_numpy_waveform(
    audio: Union[str, np.ndarray, Iterable[float]],
    sample_rate: Optional[int],
) -> Tuple[np.ndarray, int]:
    """Normalize supported audio inputs to a mono float32 waveform."""

    if isinstance(audio, np.ndarray):
        waveform = audio.astype(np.float32, copy=False)
        sr = sample_rate or 16_000
    elif isinstance(audio, str):
        if librosa is None:  # pragma: no cover
            raise ValueError("librosa is required to load audio from path.")
        waveform, sr = librosa.load(audio, sr=sample_rate)
        waveform = waveform.astype(np.float32, copy=False)
    else:
        waveform = np.asarray(list(audio), dtype=np.float32)
        sr = sample_rate or 16_000

    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=-1)

    return waveform, sr


@dataclass
class AuditoryCortexConfig:
    """Configuration for the auditory cortex model."""

    backend: str = "auto"  # auto | librosa | numpy
    sample_rate: int = 16_000
    frame_length: float = 0.025  # seconds
    frame_step: float = 0.010  # seconds
    n_mels: int = 40
    feature_dim: int = 128


class NumpyAuditoryFilter:
    """Simple spectral feature extractor implemented in numpy."""

    def __init__(self, config: AuditoryCortexConfig) -> None:
        self.config = config

    def process(self, waveform: np.ndarray, sr: int) -> Dict[str, Any]:
        window = int(self.config.frame_length * sr)
        hop = int(self.config.frame_step * sr)
        if hop <= 0:
            hop = max(1, window // 2)
        if window <= 0:
            window = max(1, int(0.025 * sr))

        frames = self._frame(waveform, window=window, hop=hop)
        window_fn = np.hanning(window)
        windowed = frames * window_fn
        spectrum = np.abs(np.fft.rfft(windowed, axis=1))

        # log-scaled energy per band
        spectrogram = np.log1p(spectrum)
        mean_spectrum = spectrogram.mean(axis=0)
        embedding = np.concatenate(
            [
                mean_spectrum,
                spectrogram.var(axis=0),
                [waveform.mean(), waveform.std()],
            ]
        )

        return {
            "embedding": embedding.astype(np.float32),
            "feature_maps": {
                "spectrogram": spectrogram,
                "mean_spectrum": mean_spectrum,
            },
        }

    @staticmethod
    def _frame(signal: np.ndarray, window: int, hop: int) -> np.ndarray:
        if len(signal) < window:
            pad = window - len(signal)
            signal = np.pad(signal, (0, pad), mode="constant")
        num_frames = 1 + (len(signal) - window) // hop
        frames = np.lib.stride_tricks.as_strided(
            signal,
            shape=(num_frames, window),
            strides=(signal.strides[0] * hop, signal.strides[0]),
            writeable=False,
        )
        return frames


class LibrosaAuditoryModel:
    """Auditory feature extractor backed by librosa."""

    def __init__(self, config: AuditoryCortexConfig) -> None:
        if librosa is None:  # pragma: no cover
            raise AuditoryModelUnavailable("librosa backend requested but not installed.")
        self.config = config

    def process(self, waveform: np.ndarray, sr: int) -> Dict[str, Any]:  # pragma: no cover - depends on librosa
        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=sr,
            n_mels=self.config.n_mels,
            hop_length=int(self.config.frame_step * sr),
            n_fft=int(self.config.frame_length * sr),
        )
        log_mel = librosa.power_to_db(mel + 1e-6)
        embedding = log_mel.mean(axis=1)

        mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=min(embedding.shape[0], 20))
        embedding = np.concatenate([embedding, mfcc.mean(axis=1)])

        return {
            "embedding": embedding.astype(np.float32),
            "feature_maps": {
                "log_mel": log_mel,
                "mfcc": mfcc,
            },
        }


class AuditoryCortexModel:
    """High-level auditory cortex abstraction with backend dispatch."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = AuditoryCortexConfig(**(config or {}))

        self.requested_backend = self.config.backend
        self.status: Dict[str, Any] = {"requested_backend": self.requested_backend}
        self._advanced_backend_failed = False
        self.device: Optional["torch.device"] = None
        self._model: Optional[Any] = None
        self._numpy_model: Optional[NumpyAuditoryFilter] = None
        self._librosa_model: Optional[LibrosaAuditoryModel] = None
        self._whisper_model: Any = None
        self._wav2vec_model: Any = None
        self._wav2vec_processor: Any = None

        backend = self.config.backend
        if backend == "auto":
            backend = "librosa" if librosa is not None else "numpy"
        self.backend = backend

        if backend == "librosa":
            self._librosa_model = LibrosaAuditoryModel(self.config)
            self._model = self._librosa_model
        elif backend == "numpy":
            self._init_numpy_backend()
        else:
            try:
                if backend == "whisper":
                    self._setup_whisper_backend()
                elif backend == "wav2vec2":
                    self._setup_wav2vec2_backend()
                else:
                    raise ValueError(f"Unsupported auditory backend '{backend}'.")
            except Exception as exc:  # pragma: no cover - optional dependency errors
                self._handle_advanced_failure(backend, exc)

        self.status["active_backend"] = self.backend

    def process(
        self,
        audio: Union[str, np.ndarray, Iterable[float]],
        *,
        sample_rate: Optional[int] = None,
    ) -> Dict[str, Any]:
        waveform, sr = _to_numpy_waveform(audio, sample_rate or self.config.sample_rate)
        if self.backend == "librosa":
            assert self._librosa_model is not None
            result = self._librosa_model.process(waveform, sr)
        elif self.backend == "numpy":
            assert self._numpy_model is not None
            result = self._numpy_model.process(waveform, sr)
        elif self.backend == "whisper":
            result = self._process_whisper(waveform, sr)
        elif self.backend == "wav2vec2":
            result = self._process_wav2vec2(waveform, sr)
        else:
            raise ValueError(f"Unsupported auditory backend '{self.backend}'.")

        result["confidence"] = result.get("confidence", self._confidence(True))
        if self.status:
            result.setdefault("status", dict(self.status))
        return result

    # ------------------------------------------------------------------ #
    # Backend helpers
    # ------------------------------------------------------------------ #
    def _init_numpy_backend(self) -> None:
        self._numpy_model = NumpyAuditoryFilter(self.config)
        self._model = self._numpy_model

    def _handle_advanced_failure(self, backend: str, exc: Exception) -> None:
        self._advanced_backend_failed = True
        warning = f"Advanced auditory backend '{backend}' unavailable: {exc}"
        self.status["warning"] = warning
        _LOGGER.warning(warning)
        self.backend = "numpy"
        self._init_numpy_backend()
        self.status["active_backend"] = self.backend
        self.device = None

    def _confidence(self, success: bool) -> float:
        if not success:
            return 0.0
        if self.backend in {"whisper", "wav2vec2"}:
            return 0.95
        if self.backend == "librosa":
            return 0.85
        if self.backend == "numpy" and self._advanced_backend_failed:
            return 0.45
        return 0.65

    @property
    def advanced_backend_failed(self) -> bool:
        return self._advanced_backend_failed

    # ------------------------------------------------------------------ #
    # Whisper backend
    # ------------------------------------------------------------------ #
    def _setup_whisper_backend(self) -> None:  # pragma: no cover - depends on whisper
        if torch is None:
            raise AuditoryModelUnavailable("PyTorch is required for the whisper backend.")
        if whisper is None:
            raise AuditoryModelUnavailable("openai-whisper package is required for the whisper backend.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self._whisper_model = whisper.load_model("tiny", device=self.device.type)
        except Exception as exc:
            raise AuditoryModelUnavailable(f"Failed to initialize whisper backend: {exc}") from exc

    def _process_whisper(self, waveform: np.ndarray, sr: int) -> Dict[str, Any]:  # pragma: no cover - depends on whisper
        if torch is None or whisper is None or self._whisper_model is None:
            raise AuditoryModelUnavailable("Whisper backend not initialized.")

        target_sr = 16_000
        if sr != target_sr:
            waveform = self._resample_numpy(waveform, sr, target_sr)
            sr = target_sr

        audio_tensor = torch.from_numpy(waveform).float().to(self.device)
        audio_tensor = whisper.pad_or_trim(audio_tensor)
        mel = whisper.log_mel_spectrogram(audio_tensor).to(self.device)

        with torch.no_grad():
            embedding = self._whisper_model.embed_audio(mel.unsqueeze(0))

        vector = embedding.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
        return {
            "embedding": vector,
            "feature_maps": {"mel_spectrogram": mel.cpu().numpy()},
        }

    # ------------------------------------------------------------------ #
    # Wav2Vec2 backend
    # ------------------------------------------------------------------ #
    def _setup_wav2vec2_backend(self) -> None:  # pragma: no cover - depends on transformers
        if torch is None:
            raise AuditoryModelUnavailable("PyTorch is required for the wav2vec2 backend.")
        if Wav2Vec2Model is None or Wav2Vec2Processor is None:
            raise AuditoryModelUnavailable("transformers package with wav2vec2 is required.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self._wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            self._wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
            self._wav2vec_model.to(self.device).eval()
        except Exception as exc:
            raise AuditoryModelUnavailable(f"Failed to initialize wav2vec2 backend: {exc}") from exc

    def _process_wav2vec2(
        self, waveform: np.ndarray, sr: int
    ) -> Dict[str, Any]:  # pragma: no cover - depends on transformers
        if (
            torch is None
            or self._wav2vec_model is None
            or self._wav2vec_processor is None
        ):
            raise AuditoryModelUnavailable("wav2vec2 backend not initialized.")

        inputs = self._wav2vec_processor(
            waveform,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self._wav2vec_model(**inputs)

        last_hidden = outputs.last_hidden_state
        embedding = last_hidden.mean(dim=1)
        vector = embedding.squeeze(0).cpu().numpy().astype(np.float32, copy=False)

        return {
            "embedding": vector,
            "feature_maps": {"last_hidden_state": last_hidden.squeeze(0).cpu().numpy()},
        }

    @staticmethod
    def _resample_numpy(waveform: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
        if sr == target_sr or len(waveform) == 0:
            return waveform
        duration = len(waveform) / float(sr)
        target_length = max(1, int(round(duration * target_sr)))
        old_indices = np.linspace(0, len(waveform) - 1, num=len(waveform))
        new_indices = np.linspace(0, len(waveform) - 1, num=target_length)
        return np.interp(new_indices, old_indices, waveform).astype(np.float32)
