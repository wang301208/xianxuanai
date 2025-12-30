"""Model architectures and factory helpers for AutoGPT training."""
from __future__ import annotations

from typing import Dict, Sequence, Type


try:  # pragma: no cover - optional dependency
    import torch
    from torch import nn
except Exception:  # pragma: no cover - torch may be missing at runtime
    torch = None  # type: ignore

    class _StubModule:  # minimal placeholder to allow subclassing
        pass

    class nn:  # type: ignore
        Module = _StubModule

try:  # pragma: no cover - optional dependency
    from transformers import AutoModel
except Exception:  # pragma: no cover - transformers may be missing
    AutoModel = None  # type: ignore


class TransformerTextModel(nn.Module):
    """Wrapper around ``transformers.AutoModel`` for text tasks."""

    def __init__(self, model_name: str = "distilbert-base-uncased") -> None:
        if AutoModel is None:  # pragma: no cover - runtime dependency check
            raise ImportError(
                "transformers package is required for TransformerTextModel"
            )
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, *args, **kwargs):  # type: ignore[override]
        return self.model(*args, **kwargs)


class VisionCNN(nn.Module):
    """Simple convolutional network for vision tasks.

    The model now supports arbitrary input spatial dimensions. An
    ``AdaptiveAvgPool2d`` layer ensures the convolutional feature map is
    pooled to ``1x1`` so the classifier always receives a fixed-size input.
    The constructor signature remains unchanged for backward compatibility,
    and images of any size can be passed during the forward pass.
    """

    def __init__(self, num_classes: int = 10) -> None:
        if torch is None:  # pragma: no cover - runtime dependency check
            raise ImportError("torch is required for VisionCNN")
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[override]
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class SequenceRNN(nn.Module):
    """Recurrent network (LSTM/GRU) for sequential data."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        rnn_type: str = "lstm",
    ) -> None:
        if torch is None:  # pragma: no cover - runtime dependency check
            raise ImportError("torch is required for SequenceRNN")
        super().__init__()
        rnn_cls = nn.LSTM if rnn_type.lower() == "lstm" else nn.GRU
        self.rnn = rnn_cls(input_size, hidden_size, num_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[override]
        out, _ = self.rnn(x)
        return self.output(out[:, -1, :])


class MLP(nn.Module):
    """Simple configurable multi-layer perceptron."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (64, 32),
        output_dim: int = 1,
    ) -> None:
        if torch is None:  # pragma: no cover - runtime dependency check
            raise ImportError("torch is required for MLP")
        super().__init__()
        layers: list[nn.Module] = []
        last = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, output_dim))
        self.network = nn.Sequential(*layers)
        # attribute used by trainers to check input dimensionality
        self.input_dim = input_dim

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[override]
        return self.network(x)


_MODELS: Dict[str, Type[nn.Module]] = {
    "mlp": MLP,
    "cnn": VisionCNN,
    "rnn": SequenceRNN,
    "transformer": TransformerTextModel,
    # backwards compatibility with previous keys
    "vision_cnn": VisionCNN,
    "sequence_rnn": SequenceRNN,
}


def get_model(model_type: str, **kwargs) -> nn.Module:
    """Instantiate a model by type name.

    Parameters
    ----------
    model_type:
        Key identifying the model class. Supported values are
        ``"mlp"``, ``"cnn"``, ``"rnn"`` and ``"transformer"``.
    **kwargs:
        Additional arguments passed to the model constructor.
    """

    model_cls = _MODELS.get(model_type.lower())
    if model_cls is None:
        raise ValueError(f"Unknown model type: {model_type}")
    return model_cls(**kwargs)


__all__ = [
    "MLP",
    "TransformerTextModel",
    "VisionCNN",
    "SequenceRNN",
    "get_model",
]
