from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


def make_bandit_intention_policy_trainer(
    cognitive_module: Any,
    *,
    policy_path: str | Path,
) -> Callable[[Iterable[Dict[str, Any]]], Dict[str, Any]]:
    """Build a ContinualLearningCoordinator-compatible trainer for intention selection.

    The returned callable:
    - loads/keeps a :class:`modules.brain.whole_brain_policy.BanditCognitivePolicy`
    - updates it from trajectories produced by WholeBrainSimulation
    - persists the updated policy to ``policy_path``
    - hot-swaps the brain's active policy via ``cognitive_module.set_policy(...)``
    """

    from modules.brain.whole_brain_policy import BanditCognitivePolicy

    target = Path(policy_path)
    policy = BanditCognitivePolicy.load(target)

    def _trainer(dataset: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        samples = [dict(item) for item in dataset if isinstance(item, dict)]
        updated = policy.update_from_trajectories(samples)
        if updated:
            policy.save(target)
            if hasattr(cognitive_module, "set_policy"):
                cognitive_module.set_policy(policy)
        return {
            "status": "updated" if updated else "skipped",
            "updated": int(updated),
            "policy_path": str(target),
        }

    return _trainer


def make_self_play_intention_policy_trainer(
    cognitive_module: Any,
    *,
    policy_path: str | Path | None = None,
    algorithm: str = "ppo",
    episodes: int = 25,
    seed: int | None = None,
) -> Callable[[Iterable[Dict[str, Any]]], Dict[str, Any]]:
    """Optional trainer that uses backend.algorithms.SelfPlayTrainer (PPO/A3C/SAC).

    This trainer is only available when PyTorch is installed. When unavailable,
    it raises at construction time so callers can fall back to the bandit trainer.
    """

    import importlib

    if importlib.util.find_spec("torch") is None:
        raise RuntimeError("PyTorch is required for self-play intention training.")

    from backend.algorithms.self_play_trainer import SelfPlayTrainer, SelfPlayTrainerConfig

    class _Space:
        def __init__(self, *, n: int | None = None, shape: Tuple[int, ...] | None = None) -> None:
            self.n = n
            self.shape = shape

    class _ContextBanditEnv:
        def __init__(self, samples: Sequence[Dict[str, Any]], *, intentions: Sequence[str]) -> None:
            self._samples = list(samples)
            self._intentions = list(intentions)
            self._index = 0
            self._current: Dict[str, Any] | None = None
            self.observation_space = _Space(shape=(8,))
            self.action_space = _Space(n=len(self._intentions))

        def reset(self, **_kwargs: Any) -> List[float]:
            import random

            self._current = random.choice(self._samples) if self._samples else None
            return self._observation()

        def step(self, action: int):
            if self._current is None:
                obs = self.reset()
                return obs, 0.0, True, {}
            intended = self._current.get("intention")
            chosen = self._intentions[int(action)] if 0 <= int(action) < len(self._intentions) else None
            reward = float(self._current.get("reward", 0.0) or 0.0)
            # Proxy reward: reinforce actions that historically led to reward.
            if chosen != intended:
                reward = min(0.0, reward) - 0.05
            return self._observation(), reward, True, {}

        def _observation(self) -> List[float]:
            if self._current is None:
                return [0.0] * 8
            features = self._current.get("features", {}) if isinstance(self._current.get("features"), dict) else {}
            focus = features.get("focus") or ""
            focus_hash = float(abs(hash(str(focus))) % 997) / 997.0
            return [
                float(features.get("valence", 0.0) or 0.0),
                float(features.get("arousal", 0.0) or 0.0),
                float(features.get("novelty", 0.0) or 0.0),
                float(features.get("threat", 0.0) or 0.0),
                float(features.get("safety", 0.0) or 0.0),
                float(features.get("cpu", 0.0) or 0.0),
                float(features.get("memory", 0.0) or 0.0),
                focus_hash,
            ]

    class _SelfPlayRunnerPolicy:
        def __init__(self, runner: Any, *, intentions: Sequence[str]) -> None:
            self._runner = runner
            self._intentions = list(intentions)

        def select_intention(
            self,
            perception: Any,
            summary: Dict[str, float],
            emotion: Any,
            personality: Any,
            curiosity: Any,
            context: Dict[str, Any],
            learning_prediction: Optional[Dict[str, float]] = None,
            history: Optional[Sequence[Dict[str, Any]]] = None,
        ):
            from modules.brain.whole_brain_policy import CognitiveDecision, StructuredPlanner

            focus = max(summary, key=summary.get) if summary else None
            obs = [
                float(emotion.dimensions.get("valence", 0.0)),
                float(emotion.dimensions.get("arousal", 0.5)),
                float(getattr(curiosity, "last_novelty", 0.0)),
                float(context.get("threat", 0.0) or 0.0),
                float(context.get("safety", 0.0) or 0.0),
                float((learning_prediction or {}).get("cpu", 0.0)),
                float((learning_prediction or {}).get("memory", 0.0)),
                float(abs(hash(str(focus))) % 997) / 997.0 if focus else 0.0,
            ]
            action_idx = int(self._runner(obs, deterministic=True))
            if action_idx < 0 or action_idx >= len(self._intentions):
                action_idx = 0
            intention = self._intentions[action_idx]
            planner = StructuredPlanner(min_steps=4)
            plan = planner.build_plan(intention, summary, context=context)
            weights = {k: 0.0 for k in self._intentions}
            weights[intention] = 1.0
            return CognitiveDecision(
                intention=intention,
                confidence=0.55,
                plan=plan,
                weights=weights,
                tags=[intention, "self-play-policy"],
                focus=focus,
                summary=", ".join(f"{k}:{v:.2f}" for k, v in summary.items()) or "no-salient-modalities",
                thought_trace=["policy=self_play"],
                perception_summary=dict(summary),
                metadata={"policy": "self_play", "algorithm": algorithm},
            )

    intentions = ("observe", "approach", "withdraw", "explore")
    target_path = Path(policy_path) if policy_path else None

    def _trainer(dataset: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        samples = [dict(item) for item in dataset if isinstance(item, dict)]
        if len(samples) < 8:
            return {"status": "skipped", "reason": "insufficient_samples", "samples": len(samples)}

        env_factory = lambda _task=None: _ContextBanditEnv(samples, intentions=intentions)
        config = SelfPlayTrainerConfig(
            algorithm=str(algorithm),
            episodes=int(episodes),
            seed=seed,
            capability_tag="self_play_intentions",
        )
        trainer = SelfPlayTrainer(env_factory, config)
        result = trainer.train(task=None)  # type: ignore[arg-type]
        policy = _SelfPlayRunnerPolicy(result.policy, intentions=intentions)

        if hasattr(cognitive_module, "set_policy"):
            cognitive_module.set_policy(policy)

        if target_path is not None:
            try:
                import torch

                target_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"algorithm": algorithm, "state_dict": result.raw_policy.state_dict()}, target_path)
            except Exception:
                pass

        return {"status": "updated", "algorithm": algorithm, "episodes": int(episodes), "samples": len(samples)}

    return _trainer


__all__ = ["make_bandit_intention_policy_trainer", "make_self_play_intention_policy_trainer"]
