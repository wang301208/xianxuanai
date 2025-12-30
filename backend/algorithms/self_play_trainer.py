"""Coordinator for running self-play policy optimisation loops.

This module exposes :class:`SelfPlayTrainer`, a lightweight orchestration
utility that can spin up reinforcement-learning agents implemented in
``modules.evolution`` (PPO, A3C and SAC) inside task specific environments.
It is deliberately focused on configurability and observability so that new
teams can bolt the self-play loop onto bespoke optimisation problems without
having to write boilerplate data collection code.

The implementation purposefully keeps the default environment contract close
to Gym/Gymnasium while avoiding a hard dependency.  Environments are expected
to provide ``reset`` and ``step`` methods together with ``observation_space``
and ``action_space`` attributes exposing ``shape``/``n`` metadata.  The trainer
falls back to inspecting sampled actions/observations when these hints are not
available, enabling simple toy environments used in tests or documentation
snippets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence

import math
import random

try:  # pragma: no cover - optional dependency for tensor operations
    import numpy as np
except Exception:  # pragma: no cover - fallback when numpy unavailable
    np = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency for RL policies
    import torch
    from torch import Tensor, nn
    from torch.distributions import Categorical, Normal
except Exception:  # pragma: no cover - allow lazy failure when torch missing
    torch = None  # type: ignore[assignment]

    class _DummyModule:  # pragma: no cover - stub when torch missing
        pass

    class _NNStub:  # pragma: no cover - stub when torch missing
        Module = _DummyModule

        def __getattr__(self, name: str):
            raise RuntimeError("PyTorch is required for self-play training")

    nn = _NNStub()  # type: ignore[assignment]
    Tensor = Any  # type: ignore[assignment]

    class _DistributionStub:  # pragma: no cover - stub when torch missing
        pass

    Categorical = Normal = _DistributionStub  # type: ignore[assignment]

try:  # pragma: no cover - Gymnasium preferred
    from gymnasium import spaces as gym_spaces
except Exception:  # pragma: no cover - fallback to classic gym
    try:
        from gym import spaces as gym_spaces  # type: ignore
    except Exception:  # pragma: no cover - environments may be custom
        gym_spaces = None  # type: ignore[assignment]

from modules.evolution import A3C, A3CConfig, PPO, PPOConfig, SAC, SACConfig, SpecialistModule


class EnvironmentFactory(Protocol):
    """Callable returning a new environment instance."""

    def __call__(self, task: "TaskContext | None" = None) -> Any:  # pragma: no cover - protocol
        ...


@dataclass(slots=True)
class SelfPlayTrainerConfig:
    """Configuration controlling the self-play training loop."""

    algorithm: str = "ppo"
    episodes: int = 25
    max_steps_per_episode: int = 200
    gamma: float = 0.99
    seed: Optional[int] = None
    capability_tag: str = "self_play"
    policy_name: Optional[str] = None
    evaluation_interval: int = 5
    opponent_factory: Optional[Callable[["PolicyRunner"], Callable[[Any], Any]]] = None
    algorithm_kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EpisodeMetrics:
    """Recorded metrics for a single training episode."""

    episode: int
    reward: float
    length: int


class PolicyRunner:
    """Callable wrapper around a trained policy network."""

    def __init__(
        self,
        policy: nn.Module,
        action_type: str,
        device: torch.device,
        action_dim: int,
    ) -> None:
        self.policy = policy
        self.action_type = action_type
        self.device = device
        self.action_dim = action_dim

    def __call__(self, observation: Any, deterministic: bool = True) -> Any:
        tensor = _to_tensor(observation, device=self.device)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        with torch.no_grad():
            dist = self.policy(tensor)
            if self.action_type == "discrete":
                if deterministic:
                    logits = getattr(dist, "logits", None)
                    if logits is None:
                        probs = dist.probs
                        action_tensor = torch.argmax(probs, dim=-1)
                    else:
                        action_tensor = torch.argmax(logits, dim=-1)
                else:
                    action_tensor = dist.sample()
                action = int(action_tensor.squeeze(0).item())
                return action
            else:
                if deterministic:
                    mean = getattr(dist, "mean", None)
                    if mean is None:
                        action_tensor = dist.loc
                    else:
                        action_tensor = mean
                else:
                    action_tensor = dist.rsample()
                action = action_tensor.squeeze(0).cpu().numpy()
                if action.shape == ():
                    return float(action)
                return action


@dataclass(slots=True)
class SelfPlayResult:
    """Container describing the outcome of a self-play training session."""

    algorithm: str
    policy: PolicyRunner
    metrics: List[EpisodeMetrics]
    capability_tag: str
    raw_policy: nn.Module

    def build_specialist(
        self,
        name: str,
        capabilities: Iterable[str],
        *,
        priority: float = 1.0,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> SpecialistModule:
        """Create a :class:`SpecialistModule` exposing the trained policy."""

        meta = dict(metadata or {})
        runner = self.policy

        def solver(architecture: Dict[str, float], task: "TaskContext") -> Dict[str, float]:
            adapter = None
            if task.metadata and isinstance(task.metadata, Mapping):
                adapter = task.metadata.get("self_play_adapter")  # type: ignore[assignment]
                if adapter is None:
                    adapter = task.metadata.get("action_adapter")  # type: ignore[assignment]
            if callable(adapter):
                return adapter(runner, architecture, task, self)

            if not architecture:
                return {}

            values = list(architecture.values())
            observation = np.asarray(values, dtype=float) if np is not None else values
            action = runner(observation, deterministic=True)

            updated = dict(architecture)
            key = next(iter(updated))
            try:
                if isinstance(action, (list, tuple)):
                    delta = float(action[0])
                elif isinstance(action, np.ndarray):  # type: ignore[attr-defined]
                    delta = float(action.flat[0])
                else:
                    delta = float(action)
            except Exception:
                delta = 0.0
            updated[key] = float(updated[key]) + 0.01 * delta
            updated[key] = float(updated[key])
            return updated

        return SpecialistModule(
            name=name,
            capabilities=set(capabilities) | {self.capability_tag},
            solver=solver,
            priority=priority,
        )


class SelfPlayTrainer:
    """Coordinate self-play reinforcement learning for specialist policies."""

    def __init__(self, env_factory: EnvironmentFactory, config: SelfPlayTrainerConfig | None = None):
        if torch is None or PPO is None or A3C is None or SAC is None:
            raise RuntimeError(
                "SelfPlayTrainer requires PyTorch and the PPO/A3C/SAC "
                "implementations from modules.evolution."
            )
        self.env_factory = env_factory
        self.config = config or SelfPlayTrainerConfig()
        algo = self.config.algorithm.lower()
        if algo not in {"ppo", "a3c", "sac"}:
            raise ValueError(f"Unsupported algorithm '{self.config.algorithm}'")
        self.algorithm_name = algo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._last_result: Optional[SelfPlayResult] = None

    @property
    def last_result(self) -> Optional[SelfPlayResult]:
        """Return metrics from the most recent training run."""

        return self._last_result

    def train(self, task: "TaskContext | None" = None) -> SelfPlayResult:
        """Execute the self-play loop and return the resulting policy."""

        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            random.seed(self.config.seed)
            if np is not None:
                np.random.seed(self.config.seed)  # type: ignore[attr-defined]

        env = self.env_factory(task)
        observation_space = getattr(env, "observation_space", None)
        action_space = getattr(env, "action_space", None)

        obs_dim = _infer_observation_dim(env, observation_space)
        action_info = _infer_action_space(env, action_space)
        action_type = action_info["type"]
        action_dim = action_info["dim"]

        policy, value, aux_modules = _build_networks(
            obs_dim,
            action_dim,
            action_type,
            device=self.device,
        )

        trainer, policy_runner = self._initialise_algorithm(policy, value, aux_modules)

        metrics: List[EpisodeMetrics] = []
        opponent_policy = None
        if callable(self.config.opponent_factory):
            opponent_policy = self.config.opponent_factory(policy_runner)

        buffer: List[Dict[str, Tensor]] = []

        for episode in range(self.config.episodes):
            observation, _ = _reset_env(env)
            done = False
            episode_reward = 0.0
            transitions: List[Dict[str, Any]] = []

            for step in range(self.config.max_steps_per_episode):
                state_tensor = _to_tensor(observation, device=self.device)
                if state_tensor.dim() == 1:
                    state_tensor = state_tensor.unsqueeze(0)

                policy_dist = policy(state_tensor)
                action_tensor = policy_dist.sample()
                logp = policy_dist.log_prob(action_tensor)
                if action_type == "discrete":
                    env_action = int(action_tensor.squeeze(0).item())
                    tensor_action = action_tensor.squeeze(0).to(torch.long)
                else:
                    env_action = action_tensor.squeeze(0).cpu().numpy()
                    tensor_action = action_tensor.squeeze(0)

                opponent_action = None
                if opponent_policy is not None:
                    opponent_action = opponent_policy(observation)

                next_observation, reward, done = _step_env(env, env_action, opponent_action)
                reward_tensor = torch.tensor(float(reward), device=self.device).unsqueeze(0)
                value_estimate = value(state_tensor).detach()
                episode_reward += float(reward)

                transitions.append(
                    {
                        "state": state_tensor,
                        "action": tensor_action,
                        "logp": logp.detach(),
                        "reward": reward_tensor,
                        "value": value_estimate,
                        "next_state": _to_tensor(next_observation, device=self.device).unsqueeze(0),
                        "done": torch.tensor(float(done), device=self.device).unsqueeze(0),
                    }
                )

                observation = next_observation
                if done:
                    break

            metrics.append(EpisodeMetrics(episode=episode, reward=episode_reward, length=len(transitions)))

            if self.algorithm_name == "ppo":
                trajectories = _build_ppo_trajectories(transitions, self.config.gamma)
                trainer.update([trajectories])
            elif self.algorithm_name == "a3c":
                a3c_batch = [
                    {
                        "state": t["state"],
                        "action": t["action"].unsqueeze(0),
                        "reward": t["reward"],
                        "next_state": t["next_state"],
                        "done": t["done"],
                    }
                    for t in transitions
                ]
                trainer.update(a3c_batch)
            else:  # SAC
                buffer.extend(transitions)
                batch = _merge_sac_batch(buffer)
                if batch is not None:
                    trainer.update(batch)

        result = SelfPlayResult(
            algorithm=self.algorithm_name,
            policy=policy_runner,
            metrics=metrics,
            capability_tag=self.config.capability_tag,
            raw_policy=policy,
        )
        self._last_result = result
        return result

    # ------------------------------------------------------------------
    def _initialise_algorithm(
        self,
        policy: nn.Module,
        value: nn.Module,
        aux_modules: Dict[str, nn.Module],
    ) -> tuple[Any, PolicyRunner]:
        algo = self.algorithm_name
        kwargs = dict(self.config.algorithm_kwargs)
        if algo == "ppo":
            trainer = PPO(policy, value, PPOConfig(**kwargs) if kwargs else None)
        elif algo == "a3c":
            trainer = A3C(policy, value, A3CConfig(**kwargs) if kwargs else None)
        else:
            trainer = SAC(
                policy,
                aux_modules["q1"],
                aux_modules["q2"],
                aux_modules["target_q1"],
                aux_modules["target_q2"],
                SACConfig(**kwargs) if kwargs else None,
            )

        runner = PolicyRunner(
            policy=policy,
            action_type=aux_modules["action_type"],
            device=self.device,
            action_dim=aux_modules["action_dim"],
        )
        return trainer, runner


# ---------------------------------------------------------------------------
def _reset_env(env: Any) -> tuple[Any, Mapping[str, Any]]:
    result = env.reset()
    if isinstance(result, tuple) and len(result) == 2:
        return result
    return result, {}


def _step_env(env: Any, action: Any, opponent_action: Any | None) -> tuple[Any, float, bool]:
    if opponent_action is not None:
        result = env.step((action, opponent_action))
    else:
        result = env.step(action)

    if isinstance(result, tuple) and len(result) == 5:
        obs, reward, terminated, truncated, _ = result
        return obs, float(reward), bool(terminated or truncated)
    if isinstance(result, tuple) and len(result) == 4:
        obs, reward, done, _ = result
        return obs, float(reward), bool(done)
    obs, reward, done = result
    return obs, float(reward), bool(done)


def _infer_observation_dim(env: Any, observation_space: Any) -> int:
    if observation_space is not None:
        shape = getattr(observation_space, "shape", None)
        if shape:
            return int(np.prod(shape)) if np is not None else int(math.prod(shape))
    sample = getattr(env, "observation", None)
    if callable(sample):
        obs = sample()
    else:
        obs, _ = _reset_env(env)
    arr = np.asarray(obs).flatten() if np is not None else list(_flatten(obs))
    return int(arr.size if np is not None else len(arr))


def _infer_action_space(env: Any, action_space: Any) -> Dict[str, Any]:
    if gym_spaces is not None and action_space is not None:
        if isinstance(action_space, gym_spaces.Discrete):
            return {"type": "discrete", "dim": int(action_space.n)}
        if isinstance(action_space, gym_spaces.Box):
            shape = action_space.shape
            dim = int(np.prod(shape)) if np is not None else int(math.prod(shape))
            return {"type": "continuous", "dim": dim}

    if action_space is not None:
        n = getattr(action_space, "n", None)
        if isinstance(n, int):
            return {"type": "discrete", "dim": n}
        shape = getattr(action_space, "shape", None)
        if shape:
            dim = int(np.prod(shape)) if np is not None else int(math.prod(shape))
            return {"type": "continuous", "dim": dim}

    sample = getattr(env, "sample_action", None)
    if callable(sample):
        action = sample()
    else:
        action = env.action_space.sample() if action_space is not None else 0
    if isinstance(action, (int, float)):
        return {"type": "discrete", "dim": 1}
    arr = np.asarray(action).flatten() if np is not None else list(_flatten(action))
    dim = int(arr.size if np is not None else len(arr))
    return {"type": "continuous", "dim": dim}


def _build_networks(
    obs_dim: int,
    action_dim: int,
    action_type: str,
    device: torch.device,
) -> tuple[nn.Module, nn.Module, Dict[str, Any]]:
    hidden = 64

    policy: nn.Module
    if action_type == "discrete":
        policy = _CategoricalPolicy(obs_dim, action_dim).to(device)
    else:
        policy = _GaussianPolicy(obs_dim, action_dim).to(device)

    value = _ValueNetwork(obs_dim).to(device)

    aux: Dict[str, Any] = {"action_type": action_type, "action_dim": action_dim}
    if action_type == "continuous":
        aux.update(
            {
                "q1": _QNetwork(obs_dim, action_dim).to(device),
                "q2": _QNetwork(obs_dim, action_dim).to(device),
                "target_q1": _QNetwork(obs_dim, action_dim).to(device),
                "target_q2": _QNetwork(obs_dim, action_dim).to(device),
            }
        )
    else:
        aux.update(
            {
                "q1": _QNetwork(obs_dim, action_dim).to(device),
                "q2": _QNetwork(obs_dim, action_dim).to(device),
                "target_q1": _QNetwork(obs_dim, action_dim).to(device),
                "target_q2": _QNetwork(obs_dim, action_dim).to(device),
            }
        )
    return policy, value, aux


def _to_tensor(value: Any, device: torch.device) -> Tensor:
    if isinstance(value, np.ndarray):  # type: ignore[attr-defined]
        array = torch.from_numpy(value.astype(np.float32))
    elif isinstance(value, (list, tuple)):
        array = torch.tensor(value, dtype=torch.float32)
    elif isinstance(value, (float, int)):
        array = torch.tensor([value], dtype=torch.float32)
    else:
        array = torch.as_tensor(value, dtype=torch.float32)
    return array.to(device)


def _build_ppo_trajectories(transitions: List[Dict[str, Any]], gamma: float) -> Dict[str, Tensor]:
    returns: List[Tensor] = []
    advantages: List[Tensor] = []
    next_return = torch.zeros(1, dtype=torch.float32, device=transitions[0]["state"].device)

    for transition in reversed(transitions):
        reward = transition["reward"]
        done = transition["done"]
        next_return = reward + gamma * next_return * (1 - done)
        returns.insert(0, next_return.clone())
        advantages.insert(0, next_return - transition["value"])

    states = torch.cat([t["state"] for t in transitions], dim=0)
    actions = torch.stack([t["action"] for t in transitions])
    logp = torch.stack([t["logp"] for t in transitions])
    returns_tensor = torch.cat(returns, dim=0)
    adv_tensor = torch.cat(advantages, dim=0)

    return {
        "state": states.detach(),
        "action": actions.detach(),
        "logp": logp.detach(),
        "return": returns_tensor.detach(),
        "adv": adv_tensor.detach(),
    }


def _merge_sac_batch(buffer: List[Dict[str, Any]]) -> Optional[Dict[str, Tensor]]:
    if not buffer:
        return None
    states = torch.cat([t["state"] for t in buffer], dim=0)
    actions = torch.stack([t["action"] for t in buffer])
    rewards = torch.cat([t["reward"] for t in buffer], dim=0)
    next_states = torch.cat([t["next_state"] for t in buffer], dim=0)
    dones = torch.cat([t["done"] for t in buffer], dim=0)
    return {
        "state": states.detach(),
        "action": actions.detach(),
        "reward": rewards.detach(),
        "next_state": next_states.detach(),
        "done": dones.detach(),
    }


def _flatten(value: Any) -> Iterable[float]:
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten(item)
    else:
        yield float(value)


class _CategoricalPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x: Tensor) -> Categorical:
        logits = self.net(x)
        return Categorical(logits=logits)


class _GaussianPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x: Tensor) -> Normal:
        mean = self.net(x)
        std = self.log_std.exp().clamp(min=1e-4, max=10.0)
        return Normal(mean, std)


class _ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class _QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# Avoid circular imports by annotating TaskContext only
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import for type checkers only
    from modules.evolution.evolution_engine import TaskContext


__all__ = [
    "SelfPlayTrainer",
    "SelfPlayTrainerConfig",
    "SelfPlayResult",
    "EpisodeMetrics",
    "EnvironmentFactory",
]

