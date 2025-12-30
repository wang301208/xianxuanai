"""Optional PyTorch-based self-supervised predictor for online world-model updates.

The runtime treats this module as optional: when PyTorch is unavailable, the
exported predictor symbols are set to ``None`` so callers can safely fall back
to the NumPy implementation in :mod:`BrainSimulationSystem.learning.self_supervised`.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Mapping, Optional, Sequence

try:  # pragma: no cover - optional dependency
    import torch
    from torch import Tensor, nn
except Exception:  # pragma: no cover - allow importing without torch
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[assignment]


@dataclass
class TorchSelfSupervisedConfig:
    observation_dim: int = 192
    latent_dim: int = 64
    hidden_dim: int = 128
    action_embedding_dim: int = 24
    learning_rate: float = 1e-3
    prediction_learning_rate: float = 1e-3
    reconstruction_weight: float = 0.7
    prediction_weight: float = 0.3
    device: str = "cpu"
    lr_scheduler_enabled: bool = False
    lr_target_loss: float = 0.05
    lr_decay: float = 0.5
    lr_growth: float = 1.02
    lr_min: float = 1e-5
    lr_max: float = 1e-2
    lr_ema_beta: float = 0.9


if nn is not None:

    class _AutoEncoder(nn.Module):
        def __init__(self, obs_dim: int, latent_dim: int, hidden_dim: int) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, obs_dim),
            )

        def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
            latent = self.encoder(x)
            recon = self.decoder(latent)
            return latent, recon


    class _Transition(nn.Module):
        def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(latent_dim + action_dim + 1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
            )

        def forward(self, latent: Tensor, action: Tensor) -> Tensor:
            bias = torch.ones((latent.shape[0], 1), dtype=latent.dtype, device=latent.device)
            x = torch.cat([latent, action, bias], dim=-1)
            return self.net(x)


class TorchSelfSupervisedPredictor:
    """Self-supervised predictor with an autoencoder + latent transition model."""

    def __init__(self, config: Optional[TorchSelfSupervisedConfig] = None) -> None:
        if torch is None or nn is None:
            raise RuntimeError("PyTorch is required for TorchSelfSupervisedPredictor")
        self.config = config or TorchSelfSupervisedConfig()

        obs_dim = int(max(1, self.config.observation_dim))
        latent_dim = int(max(1, self.config.latent_dim))
        hidden_dim = int(max(8, self.config.hidden_dim))
        action_dim = int(max(0, self.config.action_embedding_dim))

        self.device = torch.device(self.config.device)
        self.model = _AutoEncoder(obs_dim, latent_dim, hidden_dim).to(self.device)
        self.transition = _Transition(latent_dim, action_dim, hidden_dim).to(self.device)

        self._optim_recon = torch.optim.Adam(self.model.parameters(), lr=float(self.config.learning_rate))
        self._optim_pred = torch.optim.Adam(self.transition.parameters(), lr=float(self.config.prediction_learning_rate))
        self._loss_ema: Optional[float] = None

        self._last_latent: Optional[Tensor] = None
        self._pending_prediction: Optional[Tensor] = None
        self._last_action_vec = torch.zeros(action_dim, dtype=torch.float32, device=self.device)

        self._latest_summary: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    def record_action(self, action: Any, *, metadata: Optional[Dict[str, Any]] = None) -> None:
        del metadata
        action_dim = int(max(0, self.config.action_embedding_dim))
        if action_dim <= 0:
            self._last_action_vec = torch.zeros(0, dtype=torch.float32, device=self.device)
            return
        seed = abs(hash(str(action))) % (2**32)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        vec = torch.randn(action_dim, generator=gen, dtype=torch.float32)
        vec = vec / max(1e-6, float(torch.linalg.norm(vec)))
        self._last_action_vec = vec.to(self.device)

    def observe(self, data: Mapping[str, Any], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        del metadata
        x = self._vectorize(data).to(self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        latent, recon = self.model(x)
        recon_loss = torch.mean((recon - x) ** 2)

        pred_loss = torch.tensor(0.0, device=self.device)
        pred_err = None
        predicted_latent = None
        if self._last_latent is not None and self._last_action_vec.numel() > 0:
            action = self._last_action_vec.view(1, -1).expand(latent.shape[0], -1)
            predicted_latent = self.transition(self._last_latent, action)
            pred_loss = torch.mean((predicted_latent - latent.detach()) ** 2)
            pred_obs = self.model.decoder(predicted_latent)
            self._pending_prediction = pred_obs.detach()
        if self._pending_prediction is not None:
            pred_err = float(torch.mean((self._pending_prediction - x) ** 2).detach().cpu().item())

        loss = float(self.config.reconstruction_weight) * recon_loss + float(self.config.prediction_weight) * pred_loss

        self._optim_recon.zero_grad(set_to_none=True)
        self._optim_pred.zero_grad(set_to_none=True)
        loss.backward()
        self._optim_recon.step()
        self._optim_pred.step()

        self._last_latent = latent.detach()

        summary = {
            "reconstruction_loss": float(recon_loss.detach().cpu().item()),
            "prediction_loss": float(pred_loss.detach().cpu().item()),
            "prediction_error": pred_err,
            "lr": float(self._optim_recon.param_groups[0]["lr"]),
        }
        self._latest_summary = summary
        self._maybe_adjust_learning_rates(summary)
        return summary

    def observe_batch(
        self,
        batch: Sequence[Mapping[str, Any]],
        *,
        metadata: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        del metadata
        summaries: List[Dict[str, Any]] = []
        for item in batch:
            try:
                summaries.append(self.observe(item))
            except Exception:
                continue
        if not summaries:
            return {}
        recon = [s["reconstruction_loss"] for s in summaries if s.get("reconstruction_loss") is not None]
        pred = [s["prediction_loss"] for s in summaries if s.get("prediction_loss") is not None]
        err = [s["prediction_error"] for s in summaries if s.get("prediction_error") is not None]
        return {
            "batch": float(len(summaries)),
            "reconstruction_loss": float(sum(recon) / max(1, len(recon))) if recon else None,
            "prediction_loss": float(sum(pred) / max(1, len(pred))) if pred else None,
            "prediction_error": float(sum(err) / max(1, len(err))) if err else None,
        }

    def set_learning_rates(
        self,
        *,
        reconstruction_lr: float | None = None,
        prediction_lr: float | None = None,
    ) -> None:
        if reconstruction_lr is not None:
            lr = float(max(self.config.lr_min, min(self.config.lr_max, float(reconstruction_lr))))
            self._optim_recon.param_groups[0]["lr"] = lr
        if prediction_lr is not None:
            lr = float(max(self.config.lr_min, min(self.config.lr_max, float(prediction_lr))))
            self._optim_pred.param_groups[0]["lr"] = lr

    def learning_rates(self) -> Dict[str, float]:
        return {
            "learning_rate": float(self._optim_recon.param_groups[0]["lr"]),
            "prediction_learning_rate": float(self._optim_pred.param_groups[0]["lr"]),
        }

    def get_state(self) -> Dict[str, Any]:
        if torch is None:
            return {}
        model_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        transition_state = {k: v.detach().cpu().clone() for k, v in self.transition.state_dict().items()}
        return {
            "config": asdict(self.config),
            "model": model_state,
            "transition": transition_state,
            "loss_ema": self._loss_ema,
        }

    def set_state(self, state: Mapping[str, Any]) -> None:
        if torch is None or not isinstance(state, Mapping):
            return
        cfg = state.get("config")
        if isinstance(cfg, Mapping):
            for key, value in cfg.items():
                if not hasattr(self.config, key):
                    continue
                try:
                    setattr(self.config, key, value)
                except Exception:
                    continue
        model_state = state.get("model")
        if isinstance(model_state, Mapping):
            try:
                self.model.load_state_dict(model_state, strict=False)
            except Exception:
                pass
        transition_state = state.get("transition")
        if isinstance(transition_state, Mapping):
            try:
                self.transition.load_state_dict(transition_state, strict=False)
            except Exception:
                pass
        loss_ema = state.get("loss_ema")
        try:
            self._loss_ema = None if loss_ema is None else float(loss_ema)
        except Exception:
            pass

    def latest_summary(self) -> Dict[str, Any]:
        return dict(self._latest_summary)

    # ------------------------------------------------------------------
    def _vectorize(self, data: Mapping[str, Any]) -> Tensor:
        values: List[float] = []

        def _collect(item: Any) -> None:
            if item is None:
                return
            if isinstance(item, (int, float)):
                values.append(float(item))
            elif isinstance(item, (list, tuple)):
                for elem in item:
                    _collect(elem)
            elif isinstance(item, dict):
                for key in sorted(item.keys()):
                    _collect(item[key])

        _collect(data)
        if not values:
            values.append(0.0)

        dim = int(max(1, self.config.observation_dim))
        vec = torch.zeros(dim, dtype=torch.float32)
        limited = values[:dim]
        vec[: len(limited)] = torch.tensor(limited, dtype=torch.float32)
        return vec

    def _maybe_adjust_learning_rates(self, summary: Mapping[str, Any]) -> None:
        if not self.config.lr_scheduler_enabled:
            return
        try:
            loss = float(summary.get("reconstruction_loss") or 0.0) + float(summary.get("prediction_loss") or 0.0)
        except Exception:
            return
        beta = max(0.0, min(0.999, float(self.config.lr_ema_beta)))
        self._loss_ema = loss if self._loss_ema is None else beta * self._loss_ema + (1.0 - beta) * loss
        target = float(self.config.lr_target_loss)
        scale = float(self.config.lr_decay) if self._loss_ema > target else float(self.config.lr_growth)
        current = self.learning_rates()
        self.set_learning_rates(
            reconstruction_lr=current["learning_rate"] * scale,
            prediction_lr=current["prediction_learning_rate"] * scale,
        )


__all__ = [
    "TorchSelfSupervisedConfig",
    "TorchSelfSupervisedPredictor",
]
