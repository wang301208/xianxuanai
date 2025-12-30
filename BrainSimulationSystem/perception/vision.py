"""Vision perception module bridging raw images to the visual cortex."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from BrainSimulationSystem.core.visual_cortex import VisualCortex
from BrainSimulationSystem.environment.base import PerceptionPacket
from .base import PerceptionModule
from .self_supervised import ContrastiveLearner

try:  # Optional high-performance backend
    import torch
    from torch import nn
except Exception:  # pragma: no cover - torch missing
    torch = None  # type: ignore
    nn = None  # type: ignore

try:  # torchvision provides ready-made CNNs, but it is optional
    from torchvision import models, transforms
except Exception:  # pragma: no cover - torchvision missing
    models = None  # type: ignore
    transforms = None  # type: ignore

try:
    from PIL import Image
except Exception:  # pragma: no cover - PIL missing
    Image = None  # type: ignore


@dataclass
class VisionPerceptionConfig:
    input_size: Tuple[int, int] = (128, 128)
    backbone: str = "resnet18"
    pretrained: bool = False
    device: str | None = None
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


class VisionPerceptionModule(PerceptionModule):
    """Process raw camera frames and update the simulated visual cortex."""

    def __init__(
        self,
        cortex: VisualCortex | None = None,
        *,
        config: VisionPerceptionConfig | None = None,
    ) -> None:
        self.config = config or VisionPerceptionConfig()
        self.cortex = cortex or VisualCortex(config={"resolution": self.config.input_size})
        self.self_supervised: ContrastiveLearner | None = None
        self._torch_available = torch is not None and nn is not None and models is not None
        self._device = None
        self._backbone: nn.Module | None = None
        self._transform = None
        if self._torch_available:
            self._setup_backbone()

    # ------------------------------------------------------------------ #
    def _setup_backbone(self) -> None:
        assert torch is not None and nn is not None
        self._device = torch.device(
            self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        constructor = getattr(models, self.config.backbone, None)
        if constructor is None:
            self._torch_available = False
            return
        kwargs: Dict[str, Any] = {}
        if self.config.pretrained and "weights" in constructor.__code__.co_varnames:
            kwargs["weights"] = "DEFAULT"  # type: ignore[assignment]
        else:
            kwargs["weights"] = None
        self._backbone = constructor(**kwargs)  # type: ignore[call-arg]
        # Remove classification head to use penultimate features.
        if hasattr(self._backbone, "fc"):
            self._backbone.fc = nn.Identity()
        self._backbone.eval().to(self._device)
        if transforms is not None:
            self._transform = transforms.Compose(
                [
                    transforms.Resize(self.config.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=self.config.normalize_mean,
                        std=self.config.normalize_std,
                    ),
                ]
            )

    # ------------------------------------------------------------------ #
    def attach_self_supervised(self, learner: ContrastiveLearner) -> None:
        self.self_supervised = learner

    # ------------------------------------------------------------------ #
    def process(
        self,
        packet: PerceptionPacket,
        *,
        info: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        if packet.vision is None:
            raise ValueError("VisionPerceptionModule requires packet.vision data")
        image = self._to_numpy(packet.vision)
        resized = self._resize(image, self.config.input_size)
        features = self._extract_features(resized)
        cortical_response = self._run_cortex(resized)
        observation = {
            "features": features,
            "cortical_response": cortical_response,
            "attention_map": cortical_response.get("attention_map"),
            "saliency_map": cortical_response.get("saliency_map"),
        }
        if self.self_supervised is not None:
            self.self_supervised.observe(features)
        return observation

    # ------------------------------------------------------------------ #
    def _run_cortex(self, image: np.ndarray) -> Dict[str, Any]:
        visual_input = SimpleNamespace(data=image)
        return self.cortex.process_visual_input(visual_input)

    # ------------------------------------------------------------------ #
    def _extract_features(self, image: np.ndarray) -> np.ndarray:
        if self._torch_available and self._backbone is not None and self._transform is not None:
            assert torch is not None
            tensor = self._transform(Image.fromarray(image)) if Image else self._simple_to_tensor(image)
            tensor = tensor.unsqueeze(0).to(self._device)
            with torch.no_grad():
                feats = self._backbone(tensor).squeeze(0).cpu().numpy()
            return feats
        return self._edge_histogram(image)

    # ------------------------------------------------------------------ #
    def _edge_histogram(self, image: np.ndarray) -> np.ndarray:
        gray = image if image.ndim == 2 else np.mean(image, axis=2)
        grad_x = np.gradient(gray, axis=1)
        grad_y = np.gradient(gray, axis=0)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        hist, _ = np.histogram(magnitude, bins=64, range=(0.0, magnitude.max() + 1e-6))
        hist = hist.astype(np.float32)
        return hist / (np.linalg.norm(hist) + 1e-6)

    # ------------------------------------------------------------------ #
    def _resize(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        target_h, target_w = size
        if Image is not None:
            pil = Image.fromarray(image)
            resized = pil.resize((target_w, target_h))
            return np.array(resized)
        # Fallback: simple nearest neighbour via numpy indexing
        h, w = image.shape[:2]
        row_idx = (np.linspace(0, h - 1, target_h)).astype(int)
        col_idx = (np.linspace(0, w - 1, target_w)).astype(int)
        return image[np.ix_(row_idx, col_idx)]

    # ------------------------------------------------------------------ #
    def _to_numpy(self, image: Any) -> np.ndarray:
        if isinstance(image, np.ndarray):
            return image.astype(np.uint8)
        if Image is not None and isinstance(image, Image.Image):
            return np.array(image)
        if hasattr(image, "tolist"):
            return np.array(image.tolist(), dtype=np.uint8)
        raise TypeError(f"Unsupported image type: {type(image)!r}")

    def _simple_to_tensor(self, image: np.ndarray):
        assert torch is not None
        arr = image.astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=-1)
        arr = arr.transpose(2, 0, 1)
        tensor = torch.from_numpy(arr)
        return tensor
