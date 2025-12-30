from __future__ import annotations

"""Behavior cloning utilities for learning action policies from demonstrations."""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


def _softmax(logits: np.ndarray, *, temperature: float = 1.0) -> np.ndarray:
    temp = max(1e-6, float(temperature))
    values = logits.astype(np.float64, copy=False) / temp
    values = values - float(np.max(values))
    exp = np.exp(values)
    denom = float(np.sum(exp))
    if denom <= 0:
        return np.ones_like(values, dtype=np.float64) / float(values.size or 1)
    return exp / denom


def _as_float_vector(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros(0, dtype=np.float32)
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False).reshape(-1)
    if isinstance(value, (list, tuple)):
        return np.asarray(list(value), dtype=np.float32).reshape(-1)
    if isinstance(value, (int, float)):
        return np.asarray([float(value)], dtype=np.float32)
    return np.asarray([float(abs(hash(str(value))) % 997) / 997.0], dtype=np.float32)


@dataclass
class BehaviorCloningConfig:
    state_dim: int = 64
    lr: float = 0.05
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    entropy_bonus: float = 0.01
    action_vocab_limit: int = 128
    inference_temperature: float = 1.0
    inference_uniform_mix: float = 0.05
    seed: int | None = None


class BehaviorCloningPolicy:
    """Online behavior cloning policy (linear softmax classifier)."""

    OTHER_ACTION = "<OTHER>"

    def __init__(self, config: Optional[BehaviorCloningConfig] = None) -> None:
        self.config = config or BehaviorCloningConfig()
        self._state_dim = int(max(1, self.config.state_dim))
        self._rng = np.random.default_rng(self.config.seed)
        self._action_to_idx: Dict[str, int] = {}
        self._idx_to_action: List[str] = []
        self._weights: Optional[np.ndarray] = None
        self._bias: Optional[np.ndarray] = None
        self.steps: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def observe(self, state: Mapping[str, Any], action: str, *, weight: float = 1.0) -> Dict[str, float]:
        """Update the policy from one (state, action) supervised sample."""

        vector = self._vectorize_state(state)
        action_idx = self._ensure_action(action)
        if self._weights is None or self._bias is None:
            return {"loss": 0.0, "entropy": 0.0, "accuracy": 0.0}

        logits = self._weights @ vector + self._bias
        probs = _softmax(logits)

        num_actions = int(len(self._idx_to_action))
        smoothing = float(np.clip(self.config.label_smoothing, 0.0, 0.5))
        target = np.full(num_actions, smoothing / float(max(1, num_actions)), dtype=np.float64)
        target[int(action_idx)] += 1.0 - smoothing

        logp = np.log(probs + 1e-12)
        ce_loss = float(-np.sum(target * logp))
        entropy = float(-np.sum(probs * logp))
        total_loss = ce_loss - float(self.config.entropy_bonus) * entropy

        grad = probs - target
        if float(self.config.entropy_bonus) != 0.0:
            g = logp + 1.0
            expected = float(np.sum(probs * g))
            grad_entropy = probs * (g - expected)
            grad = grad + float(self.config.entropy_bonus) * grad_entropy

        grad = grad * float(weight)
        lr = float(self.config.lr)
        weight_decay = float(self.config.weight_decay)
        if weight_decay > 0:
            self._weights *= (1.0 - lr * weight_decay)

        grad_f32 = grad.astype(np.float32, copy=False)
        self._weights -= lr * grad_f32[:, None] * vector[None, :]
        self._bias -= lr * grad_f32

        self.steps += 1
        predicted = int(np.argmax(probs))
        accuracy = 1.0 if predicted == int(action_idx) else 0.0
        return {
            "loss": float(total_loss) * float(weight),
            "entropy": float(entropy),
            "accuracy": float(accuracy),
            "actions": float(num_actions),
        }

    def observe_batch(self, batch: Sequence[MutableMapping[str, Any]]) -> Dict[str, float]:
        """Observe a batch of demonstration mappings and return averaged metrics."""

        losses: List[float] = []
        entropies: List[float] = []
        accuracies: List[float] = []
        updated = 0

        for sample in batch:
            if not isinstance(sample, Mapping):
                continue
            action = sample.get("action") or sample.get("command")
            state = sample.get("state")
            if not action or not isinstance(state, Mapping):
                continue
            summary = self.observe(state, str(action))
            if isinstance(summary, Mapping):
                if summary.get("loss") is not None:
                    losses.append(float(summary["loss"]))
                if summary.get("entropy") is not None:
                    entropies.append(float(summary["entropy"]))
                if summary.get("accuracy") is not None:
                    accuracies.append(float(summary["accuracy"]))
            updated += 1

        if updated <= 0:
            return {}

        return {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
        }

    def predict_proba(self, state: Mapping[str, Any]) -> Dict[str, float]:
        if self._weights is None or self._bias is None or not self._idx_to_action:
            return {}
        vector = self._vectorize_state(state)
        logits = self._weights @ vector + self._bias
        probs = _softmax(logits, temperature=float(self.config.inference_temperature))

        mix = float(np.clip(self.config.inference_uniform_mix, 0.0, 1.0))
        if mix > 0:
            probs = (1.0 - mix) * probs + mix * (1.0 / float(len(probs)))

        return {action: float(probs[idx]) for idx, action in enumerate(self._idx_to_action)}

    def suggest_actions(
        self,
        state: Mapping[str, Any],
        *,
        top_k: int = 3,
        exclude: Sequence[str] = (),
    ) -> List[str]:
        exclude_set = {str(x) for x in exclude if str(x)}
        scored = self.predict_proba(state)
        if not scored:
            return []
        ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
        suggestions: List[str] = []
        for action, _ in ranked:
            if action in exclude_set:
                continue
            suggestions.append(action)
            if len(suggestions) >= int(max(1, top_k)):
                break
        return suggestions

    def get_state(self) -> Dict[str, Any]:
        """Return a serialisable snapshot of the current policy state."""

        rng_state = None
        try:
            rng_state = self._rng.bit_generator.state
        except Exception:
            rng_state = None

        return {
            "config": asdict(self.config),
            "state_dim": int(self._state_dim),
            "action_to_idx": dict(self._action_to_idx),
            "idx_to_action": list(self._idx_to_action),
            "weights": None if self._weights is None else self._weights.copy(),
            "bias": None if self._bias is None else self._bias.copy(),
            "steps": int(self.steps),
            "rng_state": rng_state,
        }

    def set_state(self, state: Mapping[str, Any]) -> None:
        """Restore the policy to a state produced by :meth:`get_state`."""

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

        try:
            self._state_dim = int(max(1, state.get("state_dim", self._state_dim)))
        except Exception:
            pass
        self._action_to_idx = dict(state.get("action_to_idx") or {})
        self._idx_to_action = list(state.get("idx_to_action") or [])

        weights = state.get("weights")
        bias = state.get("bias")
        self._weights = None if weights is None else np.asarray(weights, dtype=np.float32).copy()
        self._bias = None if bias is None else np.asarray(bias, dtype=np.float32).copy()

        try:
            self.steps = int(state.get("steps", 0))
        except Exception:
            self.steps = 0

        rng_state = state.get("rng_state")
        if rng_state is not None:
            try:
                self._rng.bit_generator.state = rng_state
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _ensure_action(self, action: str) -> int:
        token = str(action).strip()
        if not token:
            token = self.OTHER_ACTION
        if token not in self._action_to_idx and len(self._idx_to_action) >= int(self.config.action_vocab_limit):
            token = self.OTHER_ACTION

        if token in self._action_to_idx:
            return int(self._action_to_idx[token])

        idx = len(self._idx_to_action)
        self._action_to_idx[token] = idx
        self._idx_to_action.append(token)

        if self._weights is None:
            self._weights = np.zeros((1, self._state_dim), dtype=np.float32)
            self._bias = np.zeros((1,), dtype=np.float32)
        else:
            self._weights = np.vstack([self._weights, np.zeros((1, self._state_dim), dtype=np.float32)])
            self._bias = np.concatenate([self._bias, np.zeros((1,), dtype=np.float32)])

        return int(idx)

    def _vectorize_state(self, state: Mapping[str, Any]) -> np.ndarray:
        values: List[float] = []

        fused = state.get("fused_embedding")
        if isinstance(fused, (list, tuple, np.ndarray)):
            values.extend(float(x) for x in _as_float_vector(fused))
        else:
            modality = state.get("modality_embeddings")
            if isinstance(modality, Mapping):
                for key in sorted(modality.keys()):
                    vec = modality.get(key)
                    if isinstance(vec, (list, tuple, np.ndarray)):
                        values.extend(float(x) for x in _as_float_vector(vec))

        detail = state.get("detail")
        if isinstance(detail, Mapping):
            for key in ("distance", "steps"):
                if detail.get(key) is not None:
                    try:
                        values.append(float(detail[key]))
                    except Exception:
                        continue

        metadata = state.get("metadata")
        if isinstance(metadata, Mapping):
            reward = metadata.get("reward")
            if reward is not None:
                try:
                    values.append(float(reward))
                except Exception:
                    pass

        if not values:
            values.extend(float(x) for x in _as_float_vector(state.get("state_vector") or []))
        if not values:
            values.append(0.0)

        vector = np.zeros(self._state_dim, dtype=np.float32)
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        length = min(self._state_dim, int(arr.size))
        if length:
            vector[:length] = arr[:length]
        return vector


__all__ = ["BehaviorCloningConfig", "BehaviorCloningPolicy"]
