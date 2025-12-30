from __future__ import annotations

"""Online reinforcement-learning agents for event-driven action-perception loops.

This module is intentionally lightweight (NumPy-only) so that it can run in the
core runtime without requiring external RL frameworks.  It is designed to sit on
top of :class:`modules.environment.loop.ActionPerceptionLoop` by consuming
``(state, action, reward, next_state, done)`` transitions and updating a policy.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def _as_float_vector(value: Any, *, dim: int | None) -> np.ndarray:
    if value is None:
        data = np.zeros(0, dtype=np.float32)
    elif isinstance(value, np.ndarray):
        data = value.astype(np.float32, copy=False).reshape(-1)
    elif isinstance(value, (list, tuple)):
        data = np.asarray(list(value), dtype=np.float32).reshape(-1)
    else:
        data = np.asarray([float(value)], dtype=np.float32).reshape(-1)

    if dim is None:
        return data

    dim = int(max(1, dim))
    if data.size == dim:
        return data
    padded = np.zeros(dim, dtype=np.float32)
    length = min(dim, int(data.size))
    if length:
        padded[:length] = data[:length]
    return padded


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64, copy=False)
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    denom = float(np.sum(exp))
    if denom <= 0:
        return np.ones_like(logits, dtype=np.float64) / float(logits.size or 1)
    return exp / denom


@dataclass(frozen=True)
class LearningAgentConfig:
    """Tunable hyper-parameters for :class:`LearningAgent`.

    Attributes:
        algorithm: ``"q_learning"`` or ``"actor_critic"``.
        actions: Discrete action tokens (typically environment command strings).
        state_dim: Optional fixed feature dimension; when ``None`` it is inferred
            from the first observed state vector.
        gamma: Discount factor.
        epsilon: Exploration probability for epsilon-greedy policies.
        epsilon_decay: Multiplicative decay applied after each episode.
        min_epsilon: Lower bound for epsilon.
        lr: Learning rate for Q-learning updates.
        policy_lr: Learning rate for actor updates (actor-critic).
        value_lr: Learning rate for critic/value updates (actor-critic).
        replay_capacity: Transition buffer capacity (used for Q-learning).
        batch_size: Mini-batch size for Q-learning updates.
        min_replay_size: Minimum transitions required before training starts.
        train_every_steps: Perform training every N observed transitions.
        updates_per_train: Gradient steps per training trigger.
        seed: Optional RNG seed.
    """

    algorithm: str = "q_learning"
    actions: Sequence[str] = ()
    state_dim: int | None = None
    gamma: float = 0.95
    epsilon: float = 0.2
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.02
    lr: float = 0.05
    policy_lr: float = 0.01
    value_lr: float = 0.02
    replay_capacity: int = 4096
    batch_size: int = 32
    min_replay_size: int = 64
    train_every_steps: int = 1
    updates_per_train: int = 1
    seed: int | None = None


class _ReplayBuffer:
    def __init__(self, capacity: int, rng: np.random.Generator) -> None:
        self._capacity = int(max(1, capacity))
        self._rng = rng
        self._storage: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self._pos = 0

    def add(self, transition: Tuple[np.ndarray, int, float, np.ndarray, bool]) -> None:
        if len(self._storage) < self._capacity:
            self._storage.append(transition)
        else:
            self._storage[self._pos] = transition
            self._pos = (self._pos + 1) % self._capacity

    def sample(self, batch_size: int) -> List[Tuple[np.ndarray, int, float, np.ndarray, bool]]:
        if not self._storage:
            return []
        batch_size = int(max(1, batch_size))
        indices = self._rng.integers(0, len(self._storage), size=min(batch_size, len(self._storage)))
        return [self._storage[int(i)] for i in indices]

    def __len__(self) -> int:
        return len(self._storage)


class LearningAgent:
    """Discrete-action learning agent with Q-learning or linear actor-critic backends."""

    def __init__(self, config: LearningAgentConfig) -> None:
        self.config = config
        self.actions = list(config.actions)
        if not self.actions:
            raise ValueError("LearningAgentConfig.actions must be non-empty")

        self.algorithm = str(config.algorithm or "q_learning").strip().lower()
        if self.algorithm not in {"q_learning", "actor_critic"}:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

        self.rng = np.random.default_rng(config.seed)

        self.state_dim: int | None = None if config.state_dim is None else int(max(1, config.state_dim))
        self.steps: int = 0
        self.episodes: int = 0
        self._epsilon_value: float = float(config.epsilon)

        self._q_weights: np.ndarray | None = None
        self._actor_weights: np.ndarray | None = None
        self._value_weights: np.ndarray | None = None

        self._replay = _ReplayBuffer(config.replay_capacity, self.rng)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def policy(self, state: Any, *, deterministic: bool = False) -> str:
        action_idx, _ = self.select_action(state, deterministic=deterministic)
        return self.actions[int(action_idx)]

    def select_action(self, state: Any, *, deterministic: bool = False) -> Tuple[int, Dict[str, float]]:
        vector = self._ensure_state_dim(state)
        epsilon = self._epsilon()

        if self.algorithm == "actor_critic":
            logits = self._actor_logits(vector)
            probs = _softmax(logits)
            if deterministic:
                action_idx = int(np.argmax(probs))
            else:
                action_idx = int(self.rng.choice(len(self.actions), p=probs))
            return action_idx, {
                "epsilon": float(epsilon),
                "confidence": float(np.max(probs)),
            }

        q_values = self._q_values(vector)
        if deterministic:
            action_idx = int(np.argmax(q_values))
        elif float(self.rng.random()) < epsilon:
            action_idx = int(self.rng.integers(0, len(self.actions)))
        else:
            action_idx = int(np.argmax(q_values))
        return action_idx, {
            "epsilon": float(epsilon),
            "q_max": float(np.max(q_values)) if q_values.size else 0.0,
        }

    def observe(
        self,
        state: Any,
        action: int | str,
        reward: float,
        next_state: Any,
        done: bool,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Record a transition and optionally update the policy."""

        del metadata  # reserved for future logging/importance weighting

        state_vec = self._ensure_state_dim(state)
        next_vec = self._ensure_state_dim(next_state)
        action_idx = self._coerce_action(action)
        transition = (state_vec, int(action_idx), float(reward), next_vec, bool(done))
        self._replay.add(transition)

        self.steps += 1
        report = {"status": "recorded", "steps": int(self.steps), "replay_size": int(len(self._replay))}

        every = int(max(1, self.config.train_every_steps))
        if len(self._replay) < int(max(1, self.config.min_replay_size)) or self.steps % every != 0:
            return report

        metrics = None
        for _ in range(int(max(1, self.config.updates_per_train))):
            metrics = self.update()
        if metrics:
            report["trained"] = True
            report["metrics"] = metrics
        return report

    def update(self) -> Dict[str, Any]:
        if len(self._replay) == 0:
            return {"status": "empty_buffer"}
        if self.algorithm == "actor_critic":
            return self._actor_critic_update()
        return self._q_learning_update()

    def end_episode(self) -> None:
        self.episodes += 1
        self._decay_epsilon()

    @property
    def epsilon(self) -> float:
        return float(self._epsilon())

    # ------------------------------------------------------------------
    # Convenience helpers for integration layers
    # ------------------------------------------------------------------
    def extract_state(self, perception_event: Mapping[str, Any] | None) -> np.ndarray:
        """Extract a numeric state vector from a perception event payload."""

        if not perception_event:
            return self._ensure_state_dim([])

        fused = perception_event.get("fused_embedding")
        if isinstance(fused, (list, tuple, np.ndarray)):
            return self._ensure_state_dim(fused)

        modality = perception_event.get("modality_embeddings")
        if isinstance(modality, dict) and modality:
            merged: List[float] = []
            for key in sorted(modality.keys()):
                value = modality.get(key)
                if isinstance(value, (list, tuple, np.ndarray)):
                    merged.extend(float(v) for v in value)
            if merged:
                return self._ensure_state_dim(merged)

        detail = perception_event.get("detail")
        if isinstance(detail, dict) and detail:
            fallback: List[float] = []
            for key in ("distance", "steps"):
                if detail.get(key) is not None:
                    try:
                        fallback.append(float(detail[key]))
                    except Exception:
                        continue
            if fallback:
                return self._ensure_state_dim(fallback)

        return self._ensure_state_dim([])

    # ------------------------------------------------------------------
    # Internal: shared state handling
    # ------------------------------------------------------------------
    def _ensure_state_dim(self, state: Any) -> np.ndarray:
        vector = _as_float_vector(state, dim=self.state_dim)
        if self.state_dim is None:
            inferred = int(max(1, vector.size))
            self.state_dim = inferred
            vector = _as_float_vector(state, dim=self.state_dim)
        self._ensure_weights()
        return vector

    def _ensure_weights(self) -> None:
        if self.state_dim is None:
            return
        action_dim = len(self.actions)
        dim = int(self.state_dim)

        if self._q_weights is None:
            self._q_weights = np.zeros((action_dim, dim), dtype=np.float32)
        if self._actor_weights is None:
            self._actor_weights = np.zeros((action_dim, dim), dtype=np.float32)
        if self._value_weights is None:
            self._value_weights = np.zeros(dim, dtype=np.float32)

    def _coerce_action(self, action: int | str) -> int:
        if isinstance(action, int):
            return int(max(0, min(len(self.actions) - 1, action)))
        token = str(action)
        try:
            return int(self.actions.index(token))
        except ValueError:
            # Unknown actions are treated as the first action.
            return 0

    def _epsilon(self) -> float:
        eps = float(self._epsilon_value)
        return float(max(float(self.config.min_epsilon), min(1.0, eps)))

    def _decay_epsilon(self) -> None:
        eps = float(self._epsilon_value)
        eps *= float(self.config.epsilon_decay)
        eps = max(float(self.config.min_epsilon), eps)
        self._epsilon_value = float(eps)

    # ------------------------------------------------------------------
    # Q-learning backend (linear function approximation)
    # ------------------------------------------------------------------
    def _q_values(self, state: np.ndarray) -> np.ndarray:
        self._ensure_weights()
        if self._q_weights is None:
            return np.zeros(len(self.actions), dtype=np.float32)
        return (self._q_weights @ state).astype(np.float32, copy=False)

    def _q_learning_update(self) -> Dict[str, Any]:
        if self._q_weights is None or self.state_dim is None:
            return {"status": "uninitialised"}

        batch = self._replay.sample(self.config.batch_size)
        if not batch:
            return {"status": "empty_batch"}

        states = np.stack([t[0] for t in batch], axis=0)
        actions = np.asarray([t[1] for t in batch], dtype=np.int64)
        rewards = np.asarray([t[2] for t in batch], dtype=np.float32)
        next_states = np.stack([t[3] for t in batch], axis=0)
        dones = np.asarray([t[4] for t in batch], dtype=np.float32)

        q_next = next_states @ self._q_weights.T
        q_next_max = np.max(q_next, axis=1).astype(np.float32)
        targets = rewards + float(self.config.gamma) * (1.0 - dones) * q_next_max

        q_pred = np.sum(states * self._q_weights[actions], axis=1)
        td_error = targets - q_pred

        alpha = float(self.config.lr)
        for action_idx in range(self._q_weights.shape[0]):
            mask = actions == action_idx
            if not np.any(mask):
                continue
            grad = (td_error[mask][:, None] * states[mask]).mean(axis=0)
            self._q_weights[action_idx] += alpha * grad.astype(np.float32)

        return {"status": "trained", "td_error": float(np.mean(np.abs(td_error)))}

    # ------------------------------------------------------------------
    # Actor-critic backend (linear-softmax actor + linear critic)
    # ------------------------------------------------------------------
    def _actor_logits(self, state: np.ndarray) -> np.ndarray:
        self._ensure_weights()
        if self._actor_weights is None:
            return np.zeros(len(self.actions), dtype=np.float32)
        return (self._actor_weights @ state).astype(np.float32, copy=False)

    def _value(self, state: np.ndarray) -> float:
        self._ensure_weights()
        if self._value_weights is None:
            return 0.0
        return float(np.dot(self._value_weights, state))

    def _actor_critic_update(self) -> Dict[str, Any]:
        if self._actor_weights is None or self._value_weights is None or self.state_dim is None:
            return {"status": "uninitialised"}

        batch = self._replay.sample(1)
        if not batch:
            return {"status": "empty_batch"}
        state, action, reward, next_state, done = batch[0]
        gamma = float(self.config.gamma)

        v = self._value(state)
        v_next = 0.0 if done else self._value(next_state)
        delta = float(reward) + gamma * float(v_next) - float(v)

        # Critic update
        self._value_weights += float(self.config.value_lr) * float(delta) * state

        # Actor update (policy gradient w/ baseline)
        logits = self._actor_logits(state).astype(np.float64, copy=False)
        probs = _softmax(logits)
        one_hot = np.zeros(len(self.actions), dtype=np.float64)
        one_hot[int(action)] = 1.0
        grad_logits = (one_hot - probs)[:, None] * state[None, :]
        self._actor_weights += float(self.config.policy_lr) * float(delta) * grad_logits.astype(np.float32)

        return {"status": "trained", "advantage": float(delta), "value": float(v)}


def run_rl_training(
    *,
    loop: Any,
    agent: LearningAgent,
    episodes: int = 10,
    max_steps: int = 64,
    agent_id: str = "learning_agent",
    task_id: str = "learning:rl",
    publish_events: bool = False,
) -> List[Dict[str, Any]]:
    """Run a synchronous RL training loop using an ActionPerceptionLoop-like object.

    The loop object is expected to implement:
    - ``reset_environment() -> Dict[str, Any]``
    - ``step_and_process(...) -> Dict[str, Any]`` (added in this repo)
    - ``process_reset(...) -> Dict[str, Any]`` (added in this repo)
    """

    reports: List[Dict[str, Any]] = []
    episodes = int(max(1, episodes))
    max_steps = int(max(1, max_steps))

    for ep in range(episodes):
        reset_event = None
        if hasattr(loop, "process_reset"):
            reset_event = loop.process_reset(
                agent_id=agent_id,
                task_id=f"{task_id}:{ep}",
                cycle=0,
                ingest=False,
                publish=bool(publish_events),
                broadcast_workspace=bool(publish_events),
            )
        else:
            loop.reset_environment()
        state = agent.extract_state(reset_event or {})

        total_reward = 0.0
        steps = 0
        done = False

        for step in range(max_steps):
            action_idx, _ = agent.select_action(state)
            command = agent.actions[action_idx]
            perception = loop.step_and_process(
                agent_id=agent_id,
                task_id=f"{task_id}:{ep}",
                cycle=step,
                command=command,
                arguments={},
                ingest=False,
                publish=bool(publish_events),
                broadcast_workspace=bool(publish_events),
            )
            reward = float((perception.get("metadata") or {}).get("reward") or 0.0)
            done = bool((perception.get("metadata") or {}).get("done") or False)
            next_state = agent.extract_state(perception)
            agent.observe(state, action_idx, reward, next_state, done)
            total_reward += reward
            steps += 1
            state = next_state
            if done:
                break

        agent.end_episode()
        reports.append(
            {
                "episode": ep,
                "steps": steps,
                "total_reward": float(total_reward),
                "done": bool(done),
                "epsilon": float(agent.epsilon),
            }
        )

    return reports


__all__ = [
    "LearningAgent",
    "LearningAgentConfig",
    "run_rl_training",
]
