"""Self-supervised predictive coding components linking perception and memory."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass
class SelfSupervisedConfig:
    """Configuration for the streaming self-supervised predictor."""

    max_observation_dim: int = 192
    latent_dim: int = 48
    action_embedding_dim: int = 24
    learning_rate: float = 0.005
    prediction_learning_rate: float = 0.003
    reconstruction_weight: float = 0.7
    prediction_weight: float = 0.3
    contrastive_margin: float = 0.2
    contrastive_weight: float = 0.15
    history_size: int = 256
    normalize_inputs: bool = True
    min_std: float = 1e-3
    preview_length: int = 24
    auto_store_enabled: bool = False
    auto_store_error_threshold: float = 0.05
    lr_scheduler_enabled: bool = False
    lr_target_loss: float = 0.05
    lr_decay: float = 0.5
    lr_growth: float = 1.02
    lr_min: float = 1e-4
    lr_max: float = 0.05
    lr_ema_beta: float = 0.9


class _RunningNormalizer:
    """Maintain streaming statistics for zero-mean unit-variance scaling."""

    def __init__(self, dim: int, eps: float) -> None:
        self.dim = dim
        self.eps = eps
        self.count = 0
        self.mean = np.zeros(dim, dtype=np.float32)
        self.m2 = np.zeros(dim, dtype=np.float64)

    def update(self, vector: np.ndarray) -> None:
        self.count += 1
        delta = vector - self.mean
        self.mean += delta / float(self.count)
        delta2 = vector - self.mean
        self.m2 += (delta * delta2).astype(np.float64)

    def std(self) -> np.ndarray:
        if self.count < 2:
            return np.ones(self.dim, dtype=np.float32)
        variance = (self.m2 / float(max(1, self.count - 1))).astype(np.float32)
        return np.sqrt(np.maximum(variance, self.eps))

    def normalize(self, vector: np.ndarray) -> np.ndarray:
        self.update(vector)
        std = self.std()
        return (vector - self.mean) / std

    def denormalize(self, vector: np.ndarray) -> np.ndarray:
        std = self.std()
        return vector * std + self.mean


class _LatentBuffer:
    """FIFO buffer for negative sampling."""

    def __init__(self, capacity: int) -> None:
        self.capacity = max(1, capacity)
        self._entries: List[np.ndarray] = []

    def push(self, latent: np.ndarray) -> None:
        if latent.ndim != 1:
            latent = latent.reshape(-1)
        if len(self._entries) >= self.capacity:
            self._entries.pop(0)
        self._entries.append(latent.astype(np.float32))

    def sample(self, rng: np.random.Generator) -> Optional[np.ndarray]:
        if not self._entries:
            return None
        idx = rng.integers(0, len(self._entries))
        return self._entries[idx]

    def __len__(self) -> int:
        return len(self._entries)


class SelfSupervisedPredictor:
    """Streaming autoencoder + latent transition model for predictive coding."""

    def __init__(self, config: Optional[SelfSupervisedConfig] = None) -> None:
        self.config = config or SelfSupervisedConfig()
        self._rng = np.random.default_rng()
        dim = self.config.max_observation_dim
        latent = self.config.latent_dim
        action_dim = self.config.action_embedding_dim

        scale = 1.0 / np.sqrt(dim)
        self.encoder_w = self._rng.normal(0.0, scale, size=(latent, dim)).astype(np.float32)
        self.encoder_b = np.zeros(latent, dtype=np.float32)
        self.decoder_w = self._rng.normal(0.0, scale, size=(dim, latent)).astype(np.float32)
        self.decoder_b = np.zeros(dim, dtype=np.float32)

        trans_input = latent + action_dim + 1
        trans_scale = 1.0 / np.sqrt(trans_input)
        self.transition_w = self._rng.normal(0.0, trans_scale, size=(latent, trans_input)).astype(np.float32)

        self._normalizer = (
            _RunningNormalizer(dim, self.config.min_std) if self.config.normalize_inputs else None
        )
        self._history = _LatentBuffer(self.config.history_size)

        self._last_latent: Optional[np.ndarray] = None
        self._last_action_vec = np.zeros(action_dim, dtype=np.float32)
        self._action_buffer = np.zeros(action_dim, dtype=np.float32)
        self._action_buffer_meta: Dict[str, Any] = {}
        self._action_buffer_value: Any = None
        self._pending_prediction: Optional[np.ndarray] = None
        self._latest_summary: Dict[str, Any] = {}
        self._last_metadata: Dict[str, Any] = {}
        self._loss_ema: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def observe(
        self,
        perception: Dict[str, Any],
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Ingest a new sensory snapshot and update predictive models."""

        metadata = metadata or {}
        observation = self._vectorize_perception(perception)

        normalized = observation
        if self._normalizer is not None:
            normalized = self._normalizer.normalize(observation)

        latent, reconstruction, recon_loss = self._autoencoder_step(normalized)
        reconstruction = self._denormalize_if_needed(reconstruction)
        prediction_error = self._compute_prediction_error(observation)

        transition_summary = self._update_transition(latent)
        predicted_next = transition_summary.get("predicted_latent")
        if predicted_next is not None:
            decoded_prediction = self._decode(predicted_next)
            decoded_prediction = self._denormalize_if_needed(decoded_prediction)
            self._pending_prediction = decoded_prediction
            transition_summary["predicted_observation"] = decoded_prediction

        self._history.push(latent)
        self._last_latent = latent
        self._last_metadata = metadata

        summary = {
            "reconstruction_loss": float(recon_loss),
            "prediction_loss": float(transition_summary.get("prediction_loss", 0.0)),
            "contrastive_loss": float(transition_summary.get("contrastive_loss", 0.0)),
            "prediction_error": float(prediction_error) if prediction_error is not None else None,
            "latent_norm": float(np.linalg.norm(latent)),
            "latent_preview": self._vector_preview(latent),
            "reconstruction_preview": self._vector_preview(reconstruction),
            "predicted_preview": self._vector_preview(
                transition_summary.get("predicted_observation")
            ),
            "timestamp": metadata.get("timestamp"),
            "reward": metadata.get("reward"),
            "previous_action": transition_summary.get("action_value"),
            "action_confidence": transition_summary.get("action_meta", {}).get("confidence"),
        }

        self._latest_summary = summary
        self._maybe_adjust_learning_rates(summary)
        return summary

    def observe_batch(
        self,
        batch: Sequence[Dict[str, Any]],
        *,
        metadata: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Observe a batch of perceptions and return averaged loss metrics.

        This is compatible with :func:`backend.execution.online_updates.apply_online_model_updates`.
        """

        summaries: List[Dict[str, Any]] = []
        meta_list = list(metadata or [])
        for idx, item in enumerate(list(batch)):
            meta = meta_list[idx] if idx < len(meta_list) and isinstance(meta_list[idx], dict) else {}
            try:
                summary = self.observe(item, metadata=meta)
            except Exception:
                continue
            if isinstance(summary, dict):
                summaries.append(summary)

        if not summaries:
            return {}

        def _avg(key: str) -> Optional[float]:
            values: List[float] = []
            for s in summaries:
                value = s.get(key)
                if value is None:
                    continue
                try:
                    values.append(float(value))
                except Exception:
                    continue
            if not values:
                return None
            return float(np.mean(values))

        return {
            "batch": float(len(summaries)),
            "reconstruction_loss": _avg("reconstruction_loss"),
            "prediction_loss": _avg("prediction_loss"),
            "prediction_error": _avg("prediction_error"),
        }

    def set_learning_rates(
        self,
        *,
        reconstruction_lr: float | None = None,
        prediction_lr: float | None = None,
    ) -> None:
        """Update learning rates (clamped) for the streaming predictor."""

        if reconstruction_lr is not None:
            try:
                lr = float(reconstruction_lr)
            except Exception:
                lr = self.config.learning_rate
            self.config.learning_rate = float(np.clip(lr, self.config.lr_min, self.config.lr_max))
        if prediction_lr is not None:
            try:
                lr = float(prediction_lr)
            except Exception:
                lr = self.config.prediction_learning_rate
            self.config.prediction_learning_rate = float(
                np.clip(lr, self.config.lr_min, self.config.lr_max)
            )

    def learning_rates(self) -> Dict[str, float]:
        return {
            "learning_rate": float(self.config.learning_rate),
            "prediction_learning_rate": float(self.config.prediction_learning_rate),
        }

    def get_state(self) -> Dict[str, Any]:
        """Return a snapshot of model parameters for rollback/early-stop."""

        normalizer_state = None
        if self._normalizer is not None:
            normalizer_state = {
                "dim": int(self._normalizer.dim),
                "eps": float(self._normalizer.eps),
                "count": int(self._normalizer.count),
                "mean": self._normalizer.mean.copy(),
                "m2": self._normalizer.m2.copy(),
            }

        history_entries = []
        try:
            history_entries = [e.copy() for e in getattr(self._history, "_entries", [])]
        except Exception:
            history_entries = []

        rng_state = None
        try:
            rng_state = self._rng.bit_generator.state
        except Exception:
            rng_state = None

        return {
            "config": asdict(self.config),
            "encoder_w": self.encoder_w.copy(),
            "encoder_b": self.encoder_b.copy(),
            "decoder_w": self.decoder_w.copy(),
            "decoder_b": self.decoder_b.copy(),
            "transition_w": self.transition_w.copy(),
            "normalizer": normalizer_state,
            "history": history_entries,
            "history_capacity": int(getattr(self._history, "capacity", len(history_entries) or 1)),
            "last_latent": None if self._last_latent is None else self._last_latent.copy(),
            "last_action_vec": self._last_action_vec.copy(),
            "action_buffer": self._action_buffer.copy(),
            "action_buffer_meta": dict(self._action_buffer_meta),
            "action_buffer_value": self._action_buffer_value,
            "pending_prediction": None if self._pending_prediction is None else self._pending_prediction.copy(),
            "latest_summary": dict(self._latest_summary),
            "last_metadata": dict(self._last_metadata),
            "loss_ema": self._loss_ema,
            "rng_state": rng_state,
        }

    def set_state(self, state: Mapping[str, Any]) -> None:
        """Restore model parameters from a snapshot produced by :meth:`get_state`."""

        if not isinstance(state, Mapping):
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

        for name in ("encoder_w", "encoder_b", "decoder_w", "decoder_b", "transition_w"):
            if name not in state:
                continue
            try:
                value = state[name]
                if value is None:
                    continue
                setattr(self, name, np.asarray(value, dtype=np.float32).copy())
            except Exception:
                continue

        normalizer = state.get("normalizer")
        if isinstance(normalizer, Mapping):
            try:
                dim = int(normalizer.get("dim", self.config.max_observation_dim))
                eps = float(normalizer.get("eps", self.config.min_std))
                restored = _RunningNormalizer(dim, eps)
                restored.count = int(normalizer.get("count", 0))
                restored.mean = np.asarray(normalizer.get("mean", restored.mean), dtype=np.float32).copy()
                restored.m2 = np.asarray(normalizer.get("m2", restored.m2), dtype=np.float64).copy()
                self._normalizer = restored
            except Exception:
                self._normalizer = None
        else:
            self._normalizer = None

        history_capacity = state.get("history_capacity")
        try:
            capacity = int(history_capacity) if history_capacity is not None else int(self.config.history_size)
        except Exception:
            capacity = int(self.config.history_size)
        self._history = _LatentBuffer(max(1, capacity))
        history = state.get("history")
        if isinstance(history, list):
            for entry in history[-self._history.capacity :]:
                try:
                    self._history.push(np.asarray(entry, dtype=np.float32))
                except Exception:
                    continue

        last_latent = state.get("last_latent")
        self._last_latent = None if last_latent is None else np.asarray(last_latent, dtype=np.float32).copy()
        try:
            self._last_action_vec = np.asarray(
                state.get("last_action_vec", self._last_action_vec), dtype=np.float32
            ).copy()
        except Exception:
            pass
        try:
            self._action_buffer = np.asarray(
                state.get("action_buffer", self._action_buffer), dtype=np.float32
            ).copy()
        except Exception:
            pass
        meta = state.get("action_buffer_meta")
        self._action_buffer_meta = dict(meta) if isinstance(meta, Mapping) else {}
        self._action_buffer_value = state.get("action_buffer_value")

        pending = state.get("pending_prediction")
        self._pending_prediction = (
            None if pending is None else np.asarray(pending, dtype=np.float32).copy()
        )
        latest = state.get("latest_summary")
        self._latest_summary = dict(latest) if isinstance(latest, Mapping) else {}
        last_meta = state.get("last_metadata")
        self._last_metadata = dict(last_meta) if isinstance(last_meta, Mapping) else {}
        self._loss_ema = state.get("loss_ema") if state.get("loss_ema") is None else float(state.get("loss_ema"))

        rng_state = state.get("rng_state")
        if rng_state is not None:
            try:
                self._rng.bit_generator.state = rng_state
            except Exception:
                pass

    def record_action(self, action: Any, *, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Store the action that will lead to the next observation."""

        metadata = metadata or {}
        self._action_buffer = self._encode_action(action)
        self._action_buffer_meta = dict(metadata)
        self._action_buffer_value = action

    def latest_summary(self) -> Dict[str, Any]:
        return dict(self._latest_summary)

    def predicted_observation(self) -> Optional[np.ndarray]:
        if self._pending_prediction is None:
            return None
        return self._pending_prediction.copy()

    def _maybe_adjust_learning_rates(self, summary: Mapping[str, Any]) -> None:
        if not self.config.lr_scheduler_enabled:
            return
        try:
            recon = summary.get("reconstruction_loss")
            pred = summary.get("prediction_loss")
            recon_v = float(recon) if recon is not None else 0.0
            pred_v = float(pred) if pred is not None else 0.0
            loss = recon_v + pred_v
        except Exception:
            return

        beta = float(np.clip(float(self.config.lr_ema_beta), 0.0, 0.999))
        if self._loss_ema is None:
            self._loss_ema = float(loss)
        else:
            self._loss_ema = beta * float(self._loss_ema) + (1.0 - beta) * float(loss)

        try:
            target = float(self.config.lr_target_loss)
        except Exception:
            target = 0.05

        scale = float(self.config.lr_decay) if float(self._loss_ema) > target else float(self.config.lr_growth)
        self.set_learning_rates(
            reconstruction_lr=float(self.config.learning_rate) * scale,
            prediction_lr=float(self.config.prediction_learning_rate) * scale,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _vectorize_perception(self, data: Dict[str, Any]) -> np.ndarray:
        values: List[float] = []

        def _collect(item: Any) -> None:
            if item is None:
                return
            if isinstance(item, (int, float)):
                values.append(float(item))
            elif isinstance(item, (list, tuple)):
                for elem in item:
                    _collect(elem)
            elif isinstance(item, np.ndarray):
                for elem in item.flatten():
                    _collect(float(elem))
            elif isinstance(item, dict):
                for key in sorted(item.keys()):
                    _collect(item[key])

        _collect(data)
        if not values:
            values.append(0.0)

        dim = self.config.max_observation_dim
        array = np.zeros(dim, dtype=np.float32)
        limited = values[:dim]
        array[: len(limited)] = np.asarray(limited, dtype=np.float32)
        return array

    def _encode_action(self, action: Any) -> np.ndarray:
        if self.config.action_embedding_dim <= 0:
            return np.zeros(0, dtype=np.float32)

        if action is None:
            return np.zeros(self.config.action_embedding_dim, dtype=np.float32)

        seed = np.uint32(np.abs(hash(str(action))) % (2**32))
        rng = np.random.default_rng(int(seed))
        vec = rng.normal(0.0, 1.0, size=self.config.action_embedding_dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def _autoencoder_step(self, vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        latent_linear = self.encoder_w @ vector + self.encoder_b
        latent = np.tanh(latent_linear)
        reconstruction = self.decoder_w @ latent + self.decoder_b
        error = reconstruction - vector
        recon_loss = 0.5 * float(np.mean(error ** 2))

        lr = self.config.learning_rate
        grad_decoder = np.outer(error, latent)
        grad_decoder_b = error
        grad_latent = self.decoder_w.T @ error
        grad_latent *= (1.0 - latent ** 2)
        grad_encoder = np.outer(grad_latent, vector)
        grad_encoder_b = grad_latent

        self.decoder_w -= lr * grad_decoder.astype(np.float32)
        self.decoder_b -= lr * grad_decoder_b.astype(np.float32)
        self.encoder_w -= lr * grad_encoder.astype(np.float32)
        self.encoder_b -= lr * grad_encoder_b.astype(np.float32)

        return latent, reconstruction, recon_loss

    def _update_transition(self, target_latent: np.ndarray) -> Dict[str, Any]:
        if self._last_latent is None:
            self._pending_prediction = None
            return {"prediction_loss": 0.0, "contrastive_loss": 0.0}

        action_vec = self._action_buffer
        action_meta = dict(self._action_buffer_meta)
        action_value = self._action_buffer_value

        input_vec = np.concatenate(
            [self._last_latent, action_vec, np.array([1.0], dtype=np.float32)]
        )
        predicted_latent = self.transition_w @ input_vec

        latent_error = predicted_latent - target_latent
        prediction_loss = 0.5 * float(np.mean(latent_error ** 2))

        contrastive_loss = 0.0
        if len(self._history) > 1:
            negative = self._history.sample(self._rng)
            if negative is not None and not np.allclose(negative, target_latent):
                margin = self.config.contrastive_margin
                pos_score = float(np.dot(predicted_latent, target_latent))
                neg_score = float(np.dot(predicted_latent, negative))
                violation = margin + neg_score - pos_score
                if violation > 0:
                    contrastive_loss = violation
                    latent_error = (
                        latent_error
                        + self.config.contrastive_weight * violation * (negative - target_latent)
                    )

        lr = self.config.prediction_learning_rate
        grad = np.outer(latent_error, input_vec)
        self.transition_w -= lr * grad.astype(np.float32)

        self._last_action_vec = action_vec
        self._action_buffer = np.zeros_like(action_vec)
        self._action_buffer_meta = {}
        self._action_buffer_value = None

        return {
            "prediction_loss": prediction_loss,
            "contrastive_loss": contrastive_loss,
            "predicted_latent": predicted_latent,
            "action_value": action_value,
            "action_meta": action_meta,
            "action_vector_norm": float(np.linalg.norm(action_vec)),
        }

    def _decode(self, latent: np.ndarray) -> np.ndarray:
        return self.decoder_w @ np.tanh(latent) + self.decoder_b

    def _denormalize_if_needed(self, vector: np.ndarray) -> np.ndarray:
        if self._normalizer is None:
            return vector
        return self._normalizer.denormalize(vector)

    def _compute_prediction_error(self, observation: np.ndarray) -> Optional[float]:
        if self._pending_prediction is None:
            return None
        diff = observation - self._pending_prediction
        return float(np.mean(diff ** 2))

    def reset_state(self) -> None:
        """Reset streaming buffers while preserving learned parameters."""
        self._last_latent = None
        self._last_action_vec = np.zeros_like(self._last_action_vec)
        self._action_buffer = np.zeros_like(self._action_buffer)
        self._action_buffer_meta = {}
        self._action_buffer_value = None
        self._pending_prediction = None
        self._latest_summary = {}
        self._last_metadata = {}
        self._history = _LatentBuffer(self.config.history_size)
        if self.config.normalize_inputs:
            self._normalizer = _RunningNormalizer(self.config.max_observation_dim, self.config.min_std)

    def _vector_preview(self, vector: Optional[np.ndarray]) -> Optional[List[float]]:
        if vector is None:
            return None
        preview = vector[: self.config.preview_length]
        return [float(x) for x in preview]
