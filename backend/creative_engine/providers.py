"""Multimodal perception and generation providers for the creative engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

try:  # core numerical dependency for feature extraction
    import numpy as np
except Exception:  # pragma: no cover - numpy may be trimmed in minimal envs
    np = None  # type: ignore[assignment]

try:  # Optional dependencies for vision models
    from PIL import Image
except Exception:  # pragma: no cover - pillow may be absent in minimal envs
    Image = None  # type: ignore[assignment]

try:  # Optional dependency for CLIP embeddings
    from backend.ml.feature_extractor import CLIPFeatureExtractor
except Exception:  # pragma: no cover - clip stack may be unavailable
    CLIPFeatureExtractor = None  # type: ignore[assignment]

try:  # Optional dependency for text-to-image diffusion
    from diffusers import StableDiffusionPipeline
except Exception:  # pragma: no cover - diffusers not installed by default
    StableDiffusionPipeline = None  # type: ignore[assignment]

try:  # Optional dependency for audio feature extraction
    import librosa
except Exception:  # pragma: no cover - librosa is optional
    librosa = None  # type: ignore[assignment]


def _ensure_image(data: Any) -> "Image.Image":
    if Image is None:
        raise RuntimeError("Pillow is required for vision functionality")
    if isinstance(data, Image.Image):
        return data
    if isinstance(data, (bytes, bytearray)):
        from io import BytesIO

        return Image.open(BytesIO(data)).convert("RGB")
    if isinstance(data, (str, Path)):
        path = Path(data)
        if path.exists():
            return Image.open(path).convert("RGB")
    raise ValueError("Unsupported image input; provide a PIL image, path or bytes")


@dataclass
class CLIPEncoderAdapter:
    """Adapter around :class:`CLIPFeatureExtractor` for multimodal inputs."""

    model_name: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"
    device: Optional[str] = None

    def __post_init__(self) -> None:
        if CLIPFeatureExtractor is None:
            raise ImportError("open_clip is required for CLIPEncoderAdapter")
        self._extractor = CLIPFeatureExtractor(
            model_name=self.model_name, pretrained=self.pretrained, device=self.device
        )

    def __call__(self, data: Any) -> Dict[str, Any]:
        if isinstance(data, (str, Path)) and Path(data).exists():
            image = _ensure_image(data)
            image_vec = self._extractor.extract_image_features(image)
            return {"embedding": image_vec.cpu().numpy(), "type": "image"}
        if Image is not None and isinstance(data, Image.Image):
            image_vec = self._extractor.extract_image_features(data)
            return {"embedding": image_vec.cpu().numpy(), "type": "image"}
        text = str(data)
        text_vec = self._extractor.extract_text_features(text)
        return {"embedding": text_vec.cpu().numpy(), "type": "text", "text": text}


@dataclass
class DiffusersImageGenerator:
    """Text-to-image generation using the Hugging Face diffusers API."""

    model_id: str = "runwayml/stable-diffusion-v1-5"
    device: Optional[str] = None
    torch_dtype: Any | None = None
    safety_checker: bool = False

    def __post_init__(self) -> None:
        if StableDiffusionPipeline is None:
            raise ImportError("diffusers is required for DiffusersImageGenerator")
        kwargs: Dict[str, Any] = {}
        if self.torch_dtype is not None:
            kwargs["torch_dtype"] = self.torch_dtype
        if not self.safety_checker:
            kwargs["safety_checker"] = None
        self._pipeline = StableDiffusionPipeline.from_pretrained(self.model_id, **kwargs)
        if self.device:
            self._pipeline.to(self.device)

    def __call__(self, prompt: str, _concepts: Sequence[Any], **_: Any) -> Any:
        result = self._pipeline(prompt)
        return result.images[0] if hasattr(result, "images") else result


class MelSpectrogramEncoder:
    """Lightweight audio encoder using mel-spectrogram features."""

    def __init__(self, sample_rate: int = 16000, n_mels: int = 64) -> None:
        self.sample_rate = sample_rate
        self.n_mels = n_mels

    def __call__(self, data: Any) -> Dict[str, Any]:
        waveform = self._to_array(data)
        if waveform.size == 0:
            raise ValueError("Audio input must contain samples")
        if np is None:
            raise RuntimeError("numpy is required for MelSpectrogramEncoder")
        if librosa is not None:
            mel = librosa.feature.melspectrogram(
                y=waveform, sr=self.sample_rate, n_mels=self.n_mels
            )
            features = librosa.power_to_db(mel).astype(np.float32)
        else:
            window = np.hanning(min(len(waveform), self.sample_rate))
            spectrum = np.abs(np.fft.rfft(waveform[: window.size] * window))
            features = spectrum[: self.n_mels].astype(np.float32)
        return {"embedding": features.flatten(), "type": "audio"}

    def _to_array(self, data: Any) -> np.ndarray:
        if np is None:
            raise RuntimeError("numpy is required for MelSpectrogramEncoder")
        if isinstance(data, np.ndarray):
            return data.astype(np.float32)
        if isinstance(data, Sequence):
            return np.asarray(list(data), dtype=np.float32)
        if isinstance(data, (bytes, bytearray)) and librosa is not None:
            import io

            audio, _ = librosa.load(io.BytesIO(data), sr=self.sample_rate)
            return audio.astype(np.float32)
        if isinstance(data, (str, Path)) and librosa is not None:
            audio, _ = librosa.load(Path(data), sr=self.sample_rate)
            return audio.astype(np.float32)
        raise ValueError("Unsupported audio input for MelSpectrogramEncoder")


__all__ = [
    "CLIPEncoderAdapter",
    "DiffusersImageGenerator",
    "MelSpectrogramEncoder",
]

