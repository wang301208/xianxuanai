"""Utility classes for decision-centric reinforcement learning policies."""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:  # pragma: no cover - optional dependency
    import gym
except Exception:  # pragma: no cover
    gym = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.base_class import BaseAlgorithm
    from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
except Exception:  # pragma: no cover
    DQN = PPO = BaseAlgorithm = DummyVecEnv = VecEnv = None  # type: ignore


_STABLE_BASELINES_AVAILABLE = PPO is not None and DQN is not None and BaseAlgorithm is not None

logger = logging.getLogger(__name__)


class _OutcomeReplayEnv:
    """Minimal env that replays (observation, reward) pairs for fine-tuning."""

    def __init__(self, outcomes: Sequence[Tuple[np.ndarray, float]]) -> None:
        if gym is None:  # pragma: no cover - runtime guard
            raise RuntimeError("gym is required for outcome replay")
        self._outcomes: List[Tuple[np.ndarray, float]] = [
            (np.asarray(obs, dtype=np.float32), float(rew)) for obs, rew in outcomes
        ]
        self._index = 0
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=self._outcomes[0][0].shape, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._index = 0
        obs = self._outcomes[self._index][0]
        if getattr(gym, "__name__", "") == "gymnasium":  # pragma: no cover - compatibility
            return obs, {}
        return obs

    def step(self, action):
        # Reward encourages actions that align with recorded reward sign/magnitude.
        target_obs, recorded_reward = self._outcomes[self._index]
        predicted = float(np.asarray(action).mean())
        target = float(np.tanh(recorded_reward))  # squash to [-1, 1]
        reward = target * predicted

        self._index += 1
        terminated = self._index >= len(self._outcomes)
        if terminated:
            next_obs = np.zeros_like(target_obs)
        else:
            next_obs = self._outcomes[self._index][0]

        if getattr(gym, "__name__", "") == "gymnasium":  # pragma: no cover - compatibility
            return next_obs, reward, terminated, False, {}
        return next_obs, reward, terminated, {}


@dataclass
class RLAgentConfig:
    """Configuration describing how to load or create an RL model."""

    algorithm: str = "ppo"
    policy: str = "MlpPolicy"
    model_path: Optional[str] = None
    device: Union[str, int] = "auto"
    verbose: int = 0


def build_option_observation(
    context: Optional[Dict[str, float]],
    option: Dict[str, Union[int, float, Dict[str, float]]],
    index: int,
    total_options: int,
) -> np.ndarray:
    """Return an 8-D feature vector describing an option in its context."""

    vector = np.zeros(8, dtype=np.float32)
    context = context or {}

    expected_value = float(option.get("expected_value", 0.0))
    risk = float(option.get("risk", 0.0))
    cost = float(option.get("cost", 0.0))
    stress = float(option.get("stress", context.get("stress", 0.0)))
    arousal = float(option.get("arousal", context.get("arousal", 0.5)))
    social_norms = option.get("social_norms", {})
    social_support = 0.0
    if isinstance(social_norms, dict) and social_norms:
        social_support = float(np.mean(list(map(float, social_norms.values()))))

    urgency = float(context.get("urgency", 0.0))
    resource_level = float(context.get("resource_level", context.get("resources", 0.0)))

    vector[0] = expected_value
    vector[1] = 1.0 - risk
    vector[2] = max(0.0, 1.0 - cost)
    vector[3] = 1.0 - stress
    vector[4] = arousal
    vector[5] = social_support
    vector[6] = urgency
    vector[7] = (index + 1) / max(1, total_options)

    vector = np.clip(vector, -1.0, 1.0)
    return vector


class DecisionRLAgent:
    """Light-weight wrapper around Stable-Baselines3 policies."""

    def __init__(
        self,
        config: Optional[RLAgentConfig] = None,
        env_builder: Optional[Callable[[], "gym.Env"]] = None,
    ) -> None:
        self.config = config or RLAgentConfig()
        self._env_builder = env_builder
        self._vec_env: Optional[VecEnv] = None
        self.model: Optional[BaseAlgorithm] = None
        self._pending_observations: List[Tuple[np.ndarray, float]] = []

        if not _STABLE_BASELINES_AVAILABLE:  # pragma: no cover - handled at runtime
            return

        if self.config.model_path and os.path.exists(self.config.model_path):
            try:
                self.model = self._load_model(self.config.model_path)
            except Exception:
                self.model = None

        if self.model is None and self._env_builder is not None:
            self._ensure_vec_env()

    @property
    def is_available(self) -> bool:
        return self.model is not None

    def _algorithm_cls(self):
        algo = (self.config.algorithm or "ppo").lower()
        if algo == "dqn":
            return DQN
        return PPO

    def _ensure_vec_env(self) -> Optional[VecEnv]:
        if not _STABLE_BASELINES_AVAILABLE:
            return None

        if self._vec_env is None and self._env_builder is not None:
            self._vec_env = DummyVecEnv([self._env_builder])
        return self._vec_env

    def _load_model(self, path: str) -> Optional[BaseAlgorithm]:
        if not _STABLE_BASELINES_AVAILABLE:
            return None

        algo_cls = self._algorithm_cls()
        if algo_cls is None:
            return None

        try:
            return algo_cls.load(path, device=self.config.device)
        except Exception:
            return None

    def _build_model(self) -> Optional[BaseAlgorithm]:
        if not _STABLE_BASELINES_AVAILABLE:
            return None

        algo_cls = self._algorithm_cls()
        env = self._ensure_vec_env()
        if algo_cls is None or env is None:
            return None

        try:
            model = algo_cls(self.config.policy, env, verbose=self.config.verbose, device=self.config.device)
        except TypeError:
            model = algo_cls(self.config.policy, env, verbose=self.config.verbose)
        return model

    def set_environment(self, env_builder: Callable[[], "gym.Env"]) -> None:
        if not _STABLE_BASELINES_AVAILABLE:
            return

        self._env_builder = env_builder
        self._vec_env = None
        self._ensure_vec_env()
        if self.model is not None:
            self.model.set_env(self._vec_env)

    def predict_action(
        self,
        observations: Sequence[Sequence[float]],
        deterministic: bool = True,
    ) -> Tuple[Optional[int], Dict[str, Union[float, List[float]]]]:
        if not _STABLE_BASELINES_AVAILABLE:
            return None, {"scores": [], "confidence": 0.0}

        if self.model is None and self.config.model_path:
            self.model = self._load_model(self.config.model_path)

        if self.model is None:
            return None, {"scores": [], "confidence": 0.0}

        obs_array = np.asarray(observations, dtype=np.float32)
        if obs_array.ndim == 1:
            obs_array = obs_array.reshape(1, -1)

        scores: List[float] = []
        for obs in obs_array:
            action, _ = self.model.predict(obs, deterministic=deterministic)
            if isinstance(action, (list, tuple, np.ndarray)):
                score = float(np.mean(action))
            else:
                score = float(action)
            scores.append(score)

        best_index = int(np.argmax(scores)) if scores else None
        if scores:
            exp_scores = np.exp(scores - np.max(scores))
            confidence = float(exp_scores[best_index] / np.sum(exp_scores))
        else:
            confidence = 0.0

        return best_index, {"scores": scores, "confidence": confidence}

    def record_outcome(self, observation: np.ndarray, reward: float) -> None:
        self._pending_observations.append((observation.astype(np.float32), float(reward)))

    def update(
        self,
        total_timesteps: int = 1024,
        save_path: Optional[str] = None,
        on_policy_observations: Optional[Sequence[Tuple[np.ndarray, float]]] = None,
    ) -> Dict[str, Union[str, int, float]]:
        if not _STABLE_BASELINES_AVAILABLE:
            return {"status": "unavailable"}

        if on_policy_observations:
            for obs, rew in on_policy_observations:
                self._pending_observations.append((np.asarray(obs, dtype=np.float32), float(rew)))

        replay_env: Optional[VecEnv] = None
        if self._pending_observations and DummyVecEnv is not None and gym is not None:
            pending_copy = list(self._pending_observations)
            try:
                replay_env = DummyVecEnv([lambda: _OutcomeReplayEnv(pending_copy)])
                # Clear only after successful env construction to avoid data loss.
                self._pending_observations.clear()
            except Exception:  # pragma: no cover - best-effort
                logger.debug("Failed to build outcome replay environment", exc_info=True)
                replay_env = None

        if self.model is None:
            # Prefer a replay env if no external environment is available.
            if replay_env is not None:
                self._vec_env = replay_env
            self.model = self._build_model()
            if self.model is None:
                return {"status": "no_model"}

        env = self._ensure_vec_env()
        if env is None:
            env = replay_env
        if env is None:
            return {"status": "no_environment"}

        if replay_env is not None and env is replay_env:
            try:
                self.model.set_env(env)
            except Exception:  # pragma: no cover - defensive
                pass

        self.model.learn(total_timesteps=total_timesteps)

        if save_path or self.config.model_path:
            target = save_path or self.config.model_path
            if target:
                os.makedirs(os.path.dirname(target), exist_ok=True)
                self.model.save(target)

        return {"status": "updated", "timesteps": total_timesteps}

    def save(self, path: Optional[str] = None) -> None:
        if not _STABLE_BASELINES_AVAILABLE or self.model is None:
            return

        target = path or self.config.model_path
        if target:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            self.model.save(target)
