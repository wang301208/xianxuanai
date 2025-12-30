"""
视觉皮层模拟模块

提供基于可选深度学习后端（PyTorch）的层级特征提取网络，以及在缺少
深度学习框架时的 Numpy 备选实现，用于处理图像视觉信号。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - torch may be unavailable
    torch = None
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from torchvision import models as tv_models
except Exception:  # pragma: no cover - torchvision may be unavailable
    tv_models = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import clip as openai_clip
except Exception:  # pragma: no cover - clip may be unavailable
    openai_clip = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]


_LOGGER = logging.getLogger(__name__)

class VisionModelUnavailable(RuntimeError):
    """Raised when the requested backend is not available."""


def _ensure_three_channel(image: np.ndarray) -> np.ndarray:
    """Ensure the image has 3 channels."""

    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    return image


def _to_numpy(image: Union[str, np.ndarray, "Image.Image", Iterable[float]]) -> np.ndarray:
    """Convert supported inputs to a float32 numpy array."""

    if isinstance(image, np.ndarray):
        arr = image
    elif Image is not None and isinstance(image, Image.Image):  # pragma: no cover - requires pillow
        arr = np.array(image)
    elif isinstance(image, str):
        if Image is None:  # pragma: no cover
            raise ValueError("Pillow is required to load images from path.")
        arr = np.array(Image.open(image).convert("RGB"))  # type: ignore[arg-type]
    else:
        arr = np.asarray(list(image))

    if arr.ndim == 1:
        # assume flattened image; try to infer square shape
        side = int(np.sqrt(arr.size // 3))
        if side * side * 3 != arr.size:
            raise ValueError("Unable to infer image shape from 1D input.")
        arr = arr.reshape(side, side, 3)
    return arr.astype(np.float32)


if torch is not None and nn is not None:

    class _SimpleVisionCNN(nn.Module):  # type: ignore[misc]
        """A lightweight CNN approximating early visual cortex."""

        def __init__(self, in_channels: int = 3, feature_dim: int = 128) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
            self.bn3 = nn.BatchNorm2d(128)
            self.avg_pool = nn.AdaptiveAvgPool2d((4, 4))
            self.fc = nn.Linear(128 * 4 * 4, feature_dim)

        def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:  # pragma: no cover - torch path
            features: Dict[str, torch.Tensor] = {}
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.relu(x)
            features["conv1"] = x

            x = self.conv2(x)
            x = self.bn2(x)
            x = F.relu(x)
            features["conv2"] = x

            x = self.conv3(x)
            x = self.bn3(x)
            x = F.relu(x)
            features["conv3"] = x

            pooled = self.avg_pool(x)
            features["pooled"] = pooled

            flattened = pooled.flatten(1)
            embedding = self.fc(flattened)
            features["embedding"] = embedding
            return embedding, features

else:  # pragma: no cover - torch backend disabled
    _SimpleVisionCNN = None  # type: ignore[assignment]


@dataclass
class VisualCortexConfig:
    """Configuration for the visual cortex model."""

    backend: str = "auto"  # "auto" | "torch" | "numpy"
    feature_dim: int = 128
    input_size: Tuple[int, int] = (224, 224)
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


class NumpyVisionFilter:
    """Fallback vision encoder implemented with numpy operations."""

    def __init__(self, config: VisualCortexConfig) -> None:
        self.config = config
        # Sobel-like edge detectors
        self._kernels = np.array(
            [
                [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
            ],
            dtype=np.float32,
        )

    def _resize(self, image: np.ndarray) -> np.ndarray:
        """Nearest-neighbor resize to target size."""
        target_h, target_w = self.config.input_size
        h, w = image.shape[:2]
        if (h, w) == (target_h, target_w):
            return image

        y_indices = (np.linspace(0, h - 1, target_h)).astype(np.int32)
        x_indices = (np.linspace(0, w - 1, target_w)).astype(np.int32)
        resized = image[y_indices][:, x_indices]
        return resized

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        mean = np.array(self.config.normalize_mean, dtype=np.float32)
        std = np.array(self.config.normalize_std, dtype=np.float32)
        return (image / 255.0 - mean) / std

    def _apply_filters(self, image: np.ndarray) -> np.ndarray:
        """Apply simple edge filters to simulate V1 responses."""

        gray = image.mean(axis=-1, keepdims=True)
        responses = []
        for kernel in self._kernels:
            response = self._convolve2d(gray[..., 0], kernel)
            responses.append(response)
        stacked = np.stack(responses, axis=0)  # (2, H, W)
        return stacked

    @staticmethod
    def _convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
        output = np.zeros_like(image)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded[i : i + kh, j : j + kw]
                output[i, j] = np.sum(region * kernel)
        return output

    def process(self, image: Union[str, np.ndarray, Iterable[float], "Image.Image"]):
        arr = _ensure_three_channel(_to_numpy(image))
        arr = self._resize(arr)
        normalized = self._normalize(arr)
        v1 = self._apply_filters(normalized)
        pooled = v1.reshape(v1.shape[0], -1).mean(axis=1, keepdims=True)
        embedding = np.concatenate([v1.flatten(), pooled.flatten()])
        return {
            "embedding": embedding,
            "feature_maps": {
                "edge_horizontal": v1[0],
                "edge_vertical": v1[1],
            },
        }


class VisualCortexModel:
    """High-level wrapper that dispatches to torch or numpy implementations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = VisualCortexConfig(**(config or {}))

        self.requested_backend = self.config.backend
        self.status: Dict[str, Any] = {"requested_backend": self.requested_backend}
        self._advanced_backend_failed = False
        self.device: Optional["torch.device"] = None
        self._clip_model: Any = None
        self._clip_preprocess: Any = None
        self._torchvision_model: Any = None
        self._torchvision_mean: Optional[Any] = None
        self._torchvision_std: Optional[Any] = None

        selected_backend = self.config.backend
        if selected_backend == "auto":
            selected_backend = "torch" if torch is not None else "numpy"

        self.backend = selected_backend
        self._torch_model: Optional[_SimpleVisionCNN] = None
        self._numpy_model: Optional[NumpyVisionFilter] = None

        if self.backend == "torch":
            if torch is None or nn is None or _SimpleVisionCNN is None:  # pragma: no cover
                raise VisionModelUnavailable("PyTorch is required for the torch backend.")
            self._torch_model = _SimpleVisionCNN(feature_dim=self.config.feature_dim)
            self._torch_model.eval()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._torch_model.to(self.device)
        elif self.backend == "numpy":
            self._init_numpy_backend()
        elif self.backend == "torchvision":
            try:
                self._setup_torchvision_backend()
            except Exception as exc:  # pragma: no cover - optional dependency errors
                self._handle_advanced_failure("torchvision", exc)
        elif self.backend == "clip":
            try:
                self._setup_clip_backend()
            except Exception as exc:  # pragma: no cover - optional dependency errors
                self._handle_advanced_failure("clip", exc)
        else:
            raise ValueError(f"Unsupported backend '{self.backend}' for visual cortex model.")

        self.status["active_backend"] = self.backend

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def process(
        self,
        image: Union[str, np.ndarray, Iterable[float], "Image.Image"],
        *,
        return_feature_maps: bool = True,
    ) -> Dict[str, Any]:
        """Process image input and return hierarchical features."""

        if self.backend == "torch":
            assert self._torch_model is not None  # for type checker
            return self._process_torch(image, return_feature_maps=return_feature_maps)
        if self.backend == "torchvision":
            return self._process_torchvision(image)
        if self.backend == "clip":
            return self._process_clip(image)

        assert self._numpy_model is not None
        result = self._numpy_model.process(image)
        result["confidence"] = result.get("confidence", self._confidence(True))
        if self.status:
            result.setdefault("status", dict(self.status))
        return result

    # ------------------------------------------------------------------ #
    # Torch implementation
    # ------------------------------------------------------------------ #
    def _process_torch(
        self,
        image: Union[str, np.ndarray, Iterable[float], "Image.Image"],
        *,
        return_feature_maps: bool,
    ) -> Dict[str, Any]:  # pragma: no cover - depends on torch
        arr = _ensure_three_channel(_to_numpy(image))
        arr = self._resize_with_torch(arr)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        if tensor.dtype != torch.float32:
            tensor = tensor.float()

        tensor = tensor.to(self.device)

        mean = torch.tensor(self.config.normalize_mean, device=self.device).view(1, -1, 1, 1)
        std = torch.tensor(self.config.normalize_std, device=self.device).view(1, -1, 1, 1)
        tensor = (tensor / 255.0 - mean) / std

        with torch.no_grad():
            embedding, feature_dict = self._torch_model(tensor)  # type: ignore[operator]

        result: Dict[str, Any] = {
            "embedding": embedding.squeeze(0).cpu().numpy(),
        }

        if return_feature_maps:
            result["feature_maps"] = {
                name: value.squeeze(0).cpu().numpy() for name, value in feature_dict.items()
            }

        result["confidence"] = self._confidence(True)
        if self.status:
            result.setdefault("status", dict(self.status))

        return result

    def _resize_with_torch(self, image: np.ndarray) -> np.ndarray:  # pragma: no cover - depends on torch
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        resized = torch.nn.functional.interpolate(
            tensor,
            size=self.config.input_size,
            mode="bilinear",
            align_corners=False,
        )
        resized = resized.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return resized

    # ------------------------------------------------------------------ #
    # Backend helpers
    # ------------------------------------------------------------------ #
    def _init_numpy_backend(self) -> None:
        self._numpy_model = NumpyVisionFilter(self.config)

    def _handle_advanced_failure(self, backend: str, exc: Exception) -> None:
        self._advanced_backend_failed = True
        warning = f"Advanced backend '{backend}' unavailable: {exc}"
        self.status["warning"] = warning
        _LOGGER.warning(warning)
        self.backend = "numpy"
        self._init_numpy_backend()
        self.status["active_backend"] = self.backend

    @property
    def advanced_backend_failed(self) -> bool:
        return self._advanced_backend_failed

    def _confidence(self, success: bool) -> float:
        if not success:
            return 0.0
        if self.backend in {"clip", "torchvision"}:
            return 0.95
        if self.backend == "torch":
            return 0.9
        if self.backend == "numpy" and self._advanced_backend_failed:
            return 0.4
        return 0.6

    # ------------------------------------------------------------------ #
    # Torchvision backend
    # ------------------------------------------------------------------ #
    def _setup_torchvision_backend(self) -> None:  # pragma: no cover - depends on torchvision
        if torch is None or nn is None:
            raise VisionModelUnavailable("PyTorch is required for the torchvision backend.")
        if tv_models is None:
            raise VisionModelUnavailable("torchvision is required for the torchvision backend.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            if hasattr(tv_models, "ResNet18_Weights"):
                weights_enum = tv_models.ResNet18_Weights.DEFAULT  # type: ignore[attr-defined]
                model = tv_models.resnet18(weights=weights_enum)
                meta = getattr(weights_enum, "meta", {})
                mean = tuple(meta.get("mean", self.config.normalize_mean))  # type: ignore[assignment]
                std = tuple(meta.get("std", self.config.normalize_std))  # type: ignore[assignment]
            else:  # pragma: no cover - fallback for older torchvision
                model = tv_models.resnet18(pretrained=True)
                mean = self.config.normalize_mean
                std = self.config.normalize_std
            feature_dim = getattr(model.fc, "in_features", self.config.feature_dim)
            model.fc = nn.Identity()
        except Exception as exc:
            raise VisionModelUnavailable(f"Failed to initialize torchvision backend: {exc}") from exc

        model.to(device)
        model.eval()

        self.device = device
        self._torchvision_model = model
        self._torchvision_mean = torch.tensor(mean, device=device).view(1, -1, 1, 1)
        self._torchvision_std = torch.tensor(std, device=device).view(1, -1, 1, 1)
        self.config.feature_dim = int(feature_dim)

    def _process_torchvision(
        self, image: Union[str, np.ndarray, Iterable[float], "Image.Image"]
    ) -> Dict[str, Any]:  # pragma: no cover - depends on torch
        if torch is None:
            raise VisionModelUnavailable("PyTorch is required for torchvision processing.")
        if self._torchvision_model is None or self._torchvision_mean is None:
            raise VisionModelUnavailable("Torchvision backend not initialized.")

        arr = _ensure_three_channel(_to_numpy(image))
        arr = self._resize_with_torch(arr)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        tensor = tensor / 255.0
        tensor = (tensor - self._torchvision_mean) / self._torchvision_std

        with torch.no_grad():
            embedding = self._torchvision_model(tensor)

        vector = embedding.squeeze(0).cpu().numpy()
        result = {
            "embedding": vector.astype(np.float32, copy=False),
            "feature_maps": {"resnet_embedding": vector.astype(np.float32, copy=False)},
            "confidence": self._confidence(True),
        }
        if self.status:
            result.setdefault("status", dict(self.status))
        return result

    # ------------------------------------------------------------------ #
    # CLIP backend
    # ------------------------------------------------------------------ #
    def _setup_clip_backend(self) -> None:  # pragma: no cover - depends on clip
        if torch is None:
            raise VisionModelUnavailable("PyTorch is required for the CLIP backend.")
        if openai_clip is None:
            raise VisionModelUnavailable("openai-clip package is required for the CLIP backend.")
        if Image is None:
            raise VisionModelUnavailable("Pillow is required for the CLIP backend.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model, preprocess = openai_clip.load("ViT-B/32", device=device)
        except Exception as exc:
            raise VisionModelUnavailable(f"Failed to initialize CLIP backend: {exc}") from exc

        self.device = device
        self._clip_model = model.eval()
        self._clip_preprocess = preprocess
        width = None
        visual = getattr(model, "visual", None)
        transformer = getattr(model, "transformer", None)
        if hasattr(visual, "output_dim"):
            width = getattr(visual, "output_dim")
        elif hasattr(transformer, "width"):
            width = getattr(transformer, "width")
        elif hasattr(model, "embed_dim"):
            width = getattr(model, "embed_dim")
        if width is not None:
            self.config.feature_dim = int(width)

    def _process_clip(
        self, image: Union[str, np.ndarray, Iterable[float], "Image.Image"]
    ) -> Dict[str, Any]:  # pragma: no cover - depends on clip
        if self._clip_model is None or self._clip_preprocess is None:
            raise VisionModelUnavailable("CLIP backend not initialized.")
        if Image is None:
            raise VisionModelUnavailable("Pillow is required for the CLIP backend.")

        arr = _ensure_three_channel(_to_numpy(image)).astype(np.uint8)
        pil_image = Image.fromarray(arr)
        tensor = self._clip_preprocess(pil_image).unsqueeze(0).to(self.device)
        if tensor.dtype != torch.float32:
            tensor = tensor.float()

        with torch.no_grad():
            embedding = self._clip_model.encode_image(tensor)

        norm = embedding.norm(dim=-1, keepdim=True)
        if torch.all(norm == 0):  # pragma: no cover - defensive
            vector = embedding.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
        else:
            embedding = embedding / norm
            vector = embedding.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
        result = {
            "embedding": vector,
            "feature_maps": {"clip_embedding": vector},
            "confidence": self._confidence(True),
        }
        if self.status:
            result.setdefault("status", dict(self.status))
        return result

