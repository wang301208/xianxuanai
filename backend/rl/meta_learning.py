"""Meta-learning utilities for reinforcement-learning agents."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .agents import ActorCriticAgent
from .simulated_env import SimulatedTaskEnv, TaskConfig, TaskSuite, vectorize_observation


@dataclass
class MetaLearningConfig:
    """Hyper-parameters controlling meta-learning."""

    meta_iterations: int = 50
    meta_batch_size: int = 4
    inner_steps: int = 5
    inner_learning_rate: float = 1e-3
    meta_learning_rate: float = 1e-2
    eval_episodes: int = 5
    skill_success_threshold: float = 0.6
    max_skill_library_size: int = 100
    seed: int = 7
    algorithm: str = "reptile"
    meta_grad_clip: Optional[float] = None


@dataclass
class MetaLearningStats:
    """Metrics collected during meta-training."""

    iteration: int
    avg_inner_return: float
    avg_success_rate: float
    update_magnitude: float
    skill_count: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "iteration": self.iteration,
            "avg_inner_return": self.avg_inner_return,
            "avg_success_rate": self.avg_success_rate,
            "update_magnitude": self.update_magnitude,
            "skill_count": self.skill_count,
        }


@dataclass(order=True)
class SkillRecord:
    """Representation of a learned skill that can be replayed later."""

    reward: float
    task_name: str = field(compare=False)
    actions: List[str] = field(compare=False, default_factory=list)
    success_rate: float = field(compare=False, default=0.0)
    metadata: Dict[str, float] = field(compare=False, default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        data: Dict[str, object] = {
            "task_name": self.task_name,
            "reward": self.reward,
            "success_rate": self.success_rate,
            "actions": list(self.actions),
        }
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data


class SkillLibrary:
    """Simple container for solutions discovered during learning."""

    def __init__(self, *, max_size: int = 100) -> None:
        self.max_size = max_size
        self._skills: List[SkillRecord] = []

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._skills)

    @property
    def skills(self) -> Tuple[SkillRecord, ...]:
        return tuple(self._skills)

    def add_skill(self, record: SkillRecord) -> None:
        """Insert a new skill, keeping only the highest-reward entries."""

        self._skills.append(record)
        self._skills.sort(reverse=True)
        if len(self._skills) > self.max_size:
            self._skills = self._skills[: self.max_size]

    def to_dict(self) -> Dict[str, List[Dict[str, object]]]:
        return {"skills": [skill.to_dict() for skill in self._skills]}

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)


class MetaLearner:
    """First-order meta-learner based on the Reptile algorithm."""

    def __init__(
        self,
        agent: ActorCriticAgent,
        suite: TaskSuite,
        config: MetaLearningConfig,
        *,
        action_space: Optional[Sequence[str]] = None,
        skill_library: Optional[SkillLibrary] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        if not suite.tasks:
            raise ValueError("Task suite is empty; meta-learning requires at least one task")

        self.agent = agent
        self.suite = suite
        self.config = config
        self.action_space = list(action_space or suite.action_space)
        self.skill_library = skill_library or SkillLibrary(
            max_size=config.max_skill_library_size
        )
        self.rng = rng or random.Random(config.seed)

    def meta_train(self, *, log_interval: int = 10) -> List[MetaLearningStats]:
        """Run the meta-training loop and return logged statistics."""

        stats_history: List[MetaLearningStats] = []
        base_state = {name: param.detach().clone() for name, param in self.agent.named_parameters()}

        for iteration in range(1, self.config.meta_iterations + 1):
            task_batch = self._sample_task_batch()
            if self.config.algorithm.lower() == "maml":
                accumulator = {
                    name: torch.zeros_like(param, device=param.device)
                    for name, param in base_state.items()
                }
            else:
                accumulator = {
                    name: torch.zeros_like(param, device=param.device)
                    for name, param in base_state.items()
                }
            batch_returns: List[float] = []
            batch_success_rates: List[float] = []

            for task in task_batch:
                if self.config.algorithm.lower() == "maml":
                    (
                        gradients,
                        actions,
                        reward,
                        success_rate,
                    ) = self._maml_task_update(task)
                    for name, grad in gradients.items():
                        accumulator[name] += grad
                else:
                    adapted_agent, actions, reward, success_rate = self._adapt_to_task(task)
                    adapted_state = {
                        name: param.detach()
                        for name, param in adapted_agent.named_parameters()
                    }
                    for name, base_param in base_state.items():
                        accumulator[name] += adapted_state[name] - base_param

                batch_returns.append(reward)
                batch_success_rates.append(success_rate)

                if actions and success_rate >= self.config.skill_success_threshold:
                    self.skill_library.add_skill(
                        SkillRecord(
                            task_name=task.name,
                            actions=actions,
                            reward=reward,
                            success_rate=success_rate,
                            metadata={"iteration": float(iteration)},
                        )
                    )

            batch_size = float(max(1, len(task_batch)))
            if self.config.algorithm.lower() == "maml":
                update_magnitude = self._apply_maml_update(accumulator, batch_size)
            else:
                update_magnitude = self._apply_reptile_update(accumulator, batch_size)

            avg_return = float(np.mean(batch_returns)) if batch_returns else 0.0
            avg_success = float(np.mean(batch_success_rates)) if batch_success_rates else 0.0

            stats = MetaLearningStats(
                iteration=iteration,
                avg_inner_return=avg_return,
                avg_success_rate=avg_success,
                update_magnitude=update_magnitude,
                skill_count=len(self.skill_library),
            )
            stats_history.append(stats)

            if log_interval and iteration % log_interval == 0:
                print(
                    f"[meta_iter={iteration}] return={avg_return:.3f} "
                    f"success={avg_success:.3f} update_norm={update_magnitude:.4f} "
                    f"skills={len(self.skill_library)}"
                )

            base_state = {
                name: param.detach().clone()
                for name, param in self.agent.named_parameters()
            }

        return stats_history

    def evaluate(self, episodes: Optional[int] = None) -> Tuple[float, float]:
        """Evaluate the meta-trained agent across random tasks."""

        episodes = episodes or self.config.eval_episodes
        returns: List[float] = []
        successes = 0

        for _ in range(episodes):
            task = self.suite.sample(rng=self.rng)
            env = SimulatedTaskEnv(task, rng=self.rng)
            observation = env.reset()
            episode_return = 0.0

            for _ in range(env.max_steps):
                features, mask = vectorize_observation(env, observation, self.action_space)
                obs_tensor = torch.from_numpy(features).unsqueeze(0)
                mask_tensor = torch.from_numpy(mask).unsqueeze(0)
                with torch.no_grad():
                    action_idx, _, _, _ = self.agent.act(obs_tensor, mask_tensor, deterministic=True)
                observation, reward, done, info = env.step(self.action_space[action_idx])
                episode_return += reward
                if done:
                    successes += int(info.get("success", 0.0))
                    break

            returns.append(episode_return)

        avg_return = float(np.mean(returns)) if returns else 0.0
        success_rate = float(successes) / float(max(1, episodes))
        return avg_return, success_rate

    def _apply_reptile_update(
        self, delta_accumulator: Dict[str, torch.Tensor], batch_size: float
    ) -> float:
        """Apply the accumulated Reptile-style update to the base agent."""

        squared_sum = 0.0
        scale = self.config.meta_learning_rate / batch_size
        with torch.no_grad():
            for name, param in self.agent.named_parameters():
                update = delta_accumulator[name] * scale
                param.add_(update)
                squared_sum += float(torch.sum(update.pow(2)))
        return float(np.sqrt(squared_sum))

    def _apply_maml_update(
        self, grad_accumulator: Dict[str, torch.Tensor], batch_size: float
    ) -> float:
        """Apply a (first-order) MAML update using accumulated gradients."""

        squared_sum = 0.0
        scale = self.config.meta_learning_rate / batch_size
        clip = self.config.meta_grad_clip
        with torch.no_grad():
            for name, param in self.agent.named_parameters():
                grad = grad_accumulator.get(name)
                if grad is None:
                    continue
                update = grad
                if clip is not None and clip > 0:
                    update = torch.clamp(update, -clip, clip)
                update = update * scale
                param.add_(-update)
                squared_sum += float(torch.sum(update.pow(2)))
        return float(np.sqrt(squared_sum))

    def _adapt_to_task(
        self, task: TaskConfig
    ) -> Tuple[ActorCriticAgent, List[str], float, float]:
        """Perform inner-loop learning on a single task."""

        env = SimulatedTaskEnv(task, rng=self.rng)
        adapted_agent = self._clone_agent()
        best_actions: List[str] = []
        best_reward = -float("inf")
        success_counter = 0
        reward_total = 0.0

        for _ in range(self.config.inner_steps):
            (
                reward,
                _length,
                success,
                _losses,
                actions,
            ) = self._train_episode(adapted_agent, env)
            reward_total += reward
            success_counter += int(success)
            if actions and reward > best_reward:
                best_reward = reward
                best_actions = actions

        success_rate = success_counter / float(max(1, self.config.inner_steps))
        if best_reward == -float("inf"):
            best_reward = reward_total / float(max(1, self.config.inner_steps))

        return adapted_agent, best_actions, best_reward, success_rate

    def _maml_task_update(
        self, task: TaskConfig
    ) -> Tuple[Dict[str, torch.Tensor], List[str], float, float]:
        """Perform inner-loop adaptation and compute first-order MAML gradients."""

        env = SimulatedTaskEnv(task, rng=self.rng)
        adapted_agent = self._clone_agent()
        best_actions: List[str] = []
        best_reward = -float("inf")
        success_counter = 0

        for _ in range(self.config.inner_steps):
            reward, _steps, success, _losses, actions = self._train_episode(adapted_agent, env)
            success_counter += int(success)
            if actions and reward > best_reward:
                best_reward = reward
                best_actions = actions

        success_rate = success_counter / float(max(1, self.config.inner_steps))

        query_env = SimulatedTaskEnv(task, rng=self.rng)
        (
            query_reward,
            _query_steps,
            _query_success,
            transitions,
            query_actions,
        ) = self._rollout_episode(adapted_agent, query_env)

        query_loss, _, _, _ = adapted_agent.compute_loss(transitions)
        named_params = list(adapted_agent.named_parameters())
        grads = torch.autograd.grad(
            query_loss,
            [param for _, param in named_params],
            allow_unused=True,
        )
        grad_dict: Dict[str, torch.Tensor] = {}
        for (name, _param), grad in zip(named_params, grads):
            if grad is None:
                continue
            grad_dict[name] = grad.detach()

        if query_actions and query_reward > best_reward:
            best_reward = query_reward
            best_actions = query_actions

        return grad_dict, best_actions, best_reward, success_rate

    def _train_episode(
        self, agent: ActorCriticAgent, env: SimulatedTaskEnv
    ) -> Tuple[float, int, bool, Tuple[float, float, float], List[str]]:
        reward, steps, success, transitions, actions_taken = self._rollout_episode(agent, env)
        loss_terms = agent.update(transitions)
        return reward, steps, success, loss_terms, actions_taken

    def _rollout_episode(
        self, agent: ActorCriticAgent, env: SimulatedTaskEnv
    ) -> Tuple[float, int, bool, List[Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]], List[str]]:
        observation = env.reset()
        transitions: List[Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]] = []
        total_reward = 0.0
        steps = 0
        success = False
        actions_taken: List[str] = []

        for _ in range(env.max_steps):
            features, mask = vectorize_observation(env, observation, self.action_space)
            obs_tensor = torch.from_numpy(features).unsqueeze(0)
            mask_tensor = torch.from_numpy(mask).unsqueeze(0)
            action_idx, log_prob, value, entropy = agent.act(obs_tensor, mask_tensor)
            action = self.action_space[action_idx]
            actions_taken.append(action)
            next_obs, reward, done, info = env.step(action)
            total_reward += reward
            transitions.append((log_prob, value, reward, entropy))
            observation = next_obs
            steps += 1
            if done:
                success = bool(info.get("success", 0.0))
                break

        return total_reward, steps, success, transitions, actions_taken

    def _clone_agent(self) -> ActorCriticAgent:
        clone = copy.deepcopy(self.agent)
        clone.optimizer = torch.optim.Adam(
            clone.parameters(), lr=self.config.inner_learning_rate
        )
        return clone

    def _sample_task_batch(self) -> List[TaskConfig]:
        tasks = list(self.suite.tasks.values())
        if len(tasks) >= self.config.meta_batch_size:
            return self.rng.sample(tasks, self.config.meta_batch_size)
        return [self.rng.choice(tasks) for _ in range(self.config.meta_batch_size)]

