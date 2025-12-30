"""Environment interface layer bridging simulators and BrainSimulationSystem.

This module defines a light abstraction for simulation backends (Unity, MuJoCo,
custom gym environments, etc.) and the adaptation utilities that convert raw
observations into multimodal tensors consumable by ``BrainSimulationSystem`` as
well as the global :mod:`backend.world_model`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Dict, Iterable, List, MutableMapping, Optional, Protocol, Tuple

from backend.world_model import WorldModel


class SimulationEnvironment(Protocol):
    """Minimum API every simulator backend must expose."""

    def initialize(self) -> None: ...

    def reset(self, **kwargs: Any) -> Dict[str, Any]: ...

    def step(
        self,
        action: Dict[str, Any] | List[float] | Any,
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]: ...

    def close(self) -> None: ...


@dataclass
class PerceptionPacket:
    """Multimodal observation bundle produced by an environment step/reset."""

    vision: Any | None = None
    audio: Any | None = None
    text: str | None = None
    state_vector: List[float] | None = None
    depth_map: Any | None = None
    proprioception: Dict[str, float] | None = None
    rewards: Dict[str, float] = field(default_factory=dict)
    terminated: bool = False
    info: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_world_model(self, world_model: WorldModel, agent_id: str) -> None:
        """Write the packet into the shared world model."""

        if self.vision is not None or self.metadata.get("vision_features") is not None:
            world_model.add_visual_observation(
                agent_id,
                image=self.vision,
                features=self.metadata.get("vision_features"),
                vit_features=self.metadata.get("vision_vit"),
                text=self.text,
            )

        if self.audio is not None:
            world_model.add_multimodal_observation(
                agent_id,
                "audio",
                data=self.audio,
                features=self.metadata.get("audio_features"),
            )

        if self.state_vector is not None:
            world_model.add_multimodal_observation(
                agent_id,
                "state",
                data=self.state_vector,
                features=self.metadata.get("state_features"),
            )

        if self.proprioception is not None:
            world_model.add_multimodal_observation(
                agent_id,
                "proprioception",
                data=self.proprioception,
            )


class ObservationTransformer:
    """Convert backend-specific observation dicts into :class:`PerceptionPacket`."""

    def __init__(
        self,
        *,
        vision_keys: Iterable[str] | None = None,
        audio_keys: Iterable[str] | None = None,
        text_key: str | None = None,
        state_key: str | None = None,
        proprio_key: str | None = None,
    ) -> None:
        self._vision_keys = list(vision_keys or ("rgb", "image", "vision"))
        self._audio_keys = list(audio_keys or ("audio", "waveform"))
        self._text_key = text_key
        self._state_key = state_key
        self._proprio_key = proprio_key

    def transform(self, observation: MutableMapping[str, Any]) -> PerceptionPacket:
        """Return a :class:`PerceptionPacket` extracted from ``observation``."""

        packet = PerceptionPacket(info=dict(observation))

        vision_stack: List[Dict[str, Any]] = []
        for key in self._vision_keys:
            if key in observation:
                vision_stack.append({"key": key, "value": observation.get(key)})
                if packet.vision is None:
                    packet.vision = observation.get(key)
        if len(vision_stack) > 1:
            packet.metadata["vision_stack"] = vision_stack

        audio_stack: List[Dict[str, Any]] = []
        for key in self._audio_keys:
            if key in observation:
                audio_stack.append({"key": key, "value": observation.get(key)})
                if packet.audio is None:
                    packet.audio = observation.get(key)
        if len(audio_stack) > 1:
            packet.metadata["audio_stack"] = audio_stack

        if self._text_key and self._text_key in observation:
            packet.text = observation[self._text_key]
        elif "text" in observation and isinstance(observation.get("text"), str):
            packet.text = observation.get("text")

        state_key = self._state_key
        if state_key is None and "state" in observation:
            state_key = "state"
        if state_key and state_key in observation:
            state = observation[state_key]
            if isinstance(state, list):
                packet.state_vector = state
            elif hasattr(state, "tolist"):
                packet.state_vector = list(state.tolist())
            else:
                packet.state_vector = list(state)

        proprio_key = self._proprio_key
        if proprio_key is None and "proprioception" in observation:
            proprio_key = "proprioception"
        if proprio_key and proprio_key in observation:
            proprio = observation[proprio_key]
            if isinstance(proprio, dict):
                packet.proprioception = proprio

        if "depth" in observation:
            packet.depth_map = observation["depth"]

        return packet


class EnvironmentAdapter:
    """Bridge SimulationEnvironment outputs to :class:`PerceptionPacket`s."""

    def __init__(
        self,
        backend: SimulationEnvironment,
        *,
        transformer: ObservationTransformer | None = None,
        reward_key: str | None = "reward",
    ) -> None:
        self._backend = backend
        self._transformer = transformer or ObservationTransformer()
        self._reward_key = reward_key
        self._initialized = False

    @property
    def backend(self) -> SimulationEnvironment:
        return self._backend

    def initialize(self) -> None:
        if not self._initialized:
            self._backend.initialize()
            self._initialized = True

    def reset(self, **kwargs: Any) -> PerceptionPacket:
        self.initialize()
        raw = self._backend.reset(**kwargs)
        return self._decorate(raw, reward=0.0, terminated=False, info={})

    def step(
        self,
        action: Dict[str, Any] | List[float] | Any,
    ) -> Tuple[PerceptionPacket, float, bool, Dict[str, Any]]:
        raw, reward, terminated, info = self._backend.step(action)
        packet = self._decorate(raw, reward=reward, terminated=terminated, info=info)
        return packet, reward, terminated, info

    def close(self) -> None:
        self._backend.close()
        self._initialized = False

    def _decorate(
        self,
        raw: Dict[str, Any],
        *,
        reward: float,
        terminated: bool,
        info: Dict[str, Any],
    ) -> PerceptionPacket:
        packet = self._transformer.transform(raw)
        packet.rewards = {self._reward_key or "reward": reward}
        packet.terminated = terminated
        packet.info = info
        return packet


class EnvironmentController:
    """High-level helper that pumps data into the world model and observers."""

    def __init__(
        self,
        adapter: EnvironmentAdapter,
        *,
        world_model: WorldModel | None = None,
        agent_id: str = "agent",
    ) -> None:
        self._adapter = adapter
        self._world_model = world_model
        self._agent_id = agent_id
        self._observers: List[Callable[[PerceptionPacket], None]] = []

    def register_observer(self, callback: Callable[[PerceptionPacket], None]) -> None:
        """Register a callback that receives every :class:`PerceptionPacket`."""

        self._observers.append(callback)

    def reset(self, **kwargs: Any) -> PerceptionPacket:
        packet = self._adapter.reset(**kwargs)
        self._dispatch(packet)
        return packet

    def step(
        self,
        action: Dict[str, Any] | List[float] | Any,
    ) -> Tuple[PerceptionPacket, float, bool, Dict[str, Any]]:
        packet, reward, terminated, info = self._adapter.step(action)
        self._dispatch(packet)
        return packet, reward, terminated, info

    def close(self) -> None:
        self._adapter.close()

    def run_loop(
        self,
        agent: Callable[[PerceptionPacket], Any],
        *,
        max_steps: int = 256,
        reset_kwargs: Dict[str, Any] | None = None,
        sleep_s: float = 0.0,
        stop_predicate: Callable[[PerceptionPacket, int], bool] | None = None,
        action_validator: Callable[[Any, PerceptionPacket], Tuple[bool, str | None, Any | None]] | None = None,
        on_error: Callable[[Exception, str], None] | None = None,
        max_errors: int = 1,
        retry_on_error: bool = True,
        store_packets: bool = False,
    ) -> Dict[str, Any]:
        """Run a long-lived interaction loop.

        Args:
            agent: Callable mapping the current :class:`PerceptionPacket` to an action.
            max_steps: Maximum number of environment steps to execute.
            reset_kwargs: Optional kwargs forwarded to :meth:`reset`.
            sleep_s: Optional delay between steps.
            stop_predicate: Optional early stop predicate receiving (packet, step_idx).
            action_validator: Optional safety hook returning (allowed, reason, replacement_action).
            on_error: Optional callback invoked with (exc, phase) on reset/step errors.
            max_errors: Maximum number of exceptions tolerated before aborting.
            retry_on_error: If True, attempts to reset and continue after exceptions.
            store_packets: If True, include packets in returned result (can be large).
        """

        reset_kwargs = dict(reset_kwargs or {})
        errors: List[Dict[str, Any]] = []
        blocked_actions = 0
        total_reward = 0.0
        packets: List[PerceptionPacket] = []
        steps_executed = 0

        def _record_error(exc: Exception, phase: str) -> None:
            errors.append({"phase": phase, "error": repr(exc)})
            if on_error is not None:
                on_error(exc, phase)

        try:
            packet = self.reset(**reset_kwargs)
        except Exception as exc:
            _record_error(exc, "reset")
            return {
                "steps": 0,
                "total_reward": 0.0,
                "terminated": True,
                "errors": errors,
                "blocked_actions": 0,
            }

        if store_packets:
            packets.append(packet)

        terminated = False
        last_info: Dict[str, Any] = {}
        for step_idx in range(int(max_steps)):
            if packet.terminated:
                terminated = True
                break
            if stop_predicate is not None and stop_predicate(packet, step_idx):
                break

            action = agent(packet)
            if action_validator is not None:
                allowed, reason, replacement = action_validator(action, packet)
                if not allowed:
                    blocked_actions += 1
                    if replacement is None:
                        last_info = {"blocked": True, "reason": reason, "action": action}
                        terminated = True
                        break
                    action = replacement

            try:
                packet, reward, terminated, info = self.step(action)
            except Exception as exc:
                _record_error(exc, "step")
                if len(errors) > int(max_errors):
                    terminated = True
                    break
                if retry_on_error:
                    try:
                        packet = self.reset(**reset_kwargs)
                        continue
                    except Exception as reset_exc:
                        _record_error(reset_exc, "reset_after_error")
                        terminated = True
                        break
                terminated = True
                break

            steps_executed += 1
            total_reward += float(reward)
            last_info = dict(info or {})
            if store_packets:
                packets.append(packet)

            if terminated:
                break
            if sleep_s:
                time.sleep(float(sleep_s))

        result: Dict[str, Any] = {
            "steps": steps_executed,
            "total_reward": total_reward,
            "terminated": bool(terminated),
            "blocked_actions": blocked_actions,
            "errors": errors,
            "last_info": last_info,
        }
        if store_packets:
            result["packets"] = packets
        return result

    def _dispatch(self, packet: PerceptionPacket) -> None:
        if self._world_model is not None:
            packet.update_world_model(self._world_model, self._agent_id)
        for observer in self._observers:
            observer(packet)


class UnityEnvironmentBridge(SimulationEnvironment):
    """Lazy wrapper around Unity ML-Agents environments."""

    def __init__(
        self,
        file_name: str,
        *,
        worker_id: int = 0,
        no_graphics: bool = True,
        time_scale: float = 1.0,
    ) -> None:
        self._file_name = file_name
        self._worker_id = worker_id
        self._no_graphics = no_graphics
        self._time_scale = time_scale
        self._env = None
        self._current_behavior: str | None = None

    def initialize(self) -> None:
        if self._env is not None:
            return
        try:
            from mlagents_envs.environment import UnityEnvironment
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Unity ML-Agents is not installed. Run `pip install mlagents-envs`."
            ) from exc

        self._env = UnityEnvironment(
            file_name=self._file_name,
            worker_id=self._worker_id,
            no_graphics=self._no_graphics,
        )
        self._env.reset()
        self._current_behavior = list(self._env.behavior_specs)[0]
        if hasattr(self._env, "set_time_scale"):
            self._env.set_time_scale(self._time_scale)

    def reset(self, **kwargs: Any) -> Dict[str, Any]:
        assert self._env is not None, "Call initialize() first"
        self._env.reset()
        decision_steps, _ = self._env.get_steps(self._current_behavior)
        return self._format_steps(decision_steps, agent_id=kwargs.get("agent_id", 0))

    def step(
        self,
        action: Dict[str, Any] | List[float] | Any,
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        assert self._env is not None, "Call initialize() first"
        action_array = action
        if isinstance(action, dict):
            action_array = action.get("continuous") or action.get("discrete")
        self._env.set_actions(self._current_behavior, action_array)
        self._env.step()
        decision_steps, terminal_steps = self._env.get_steps(self._current_behavior)
        source_steps = decision_steps if len(decision_steps) else terminal_steps
        agent_ids = list(source_steps.agent_id) if hasattr(source_steps, "agent_id") else [0]
        agent_id = agent_ids[0] if agent_ids else 0
        obs = self._format_steps(source_steps, agent_id=agent_id)
        reward_array = getattr(source_steps, "reward", [])
        reward = float(reward_array[0] if len(reward_array) else 0.0)
        terminated = len(terminal_steps) > 0
        return obs, reward, terminated, {}

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def _format_steps(self, steps: Any, agent_id: int) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        if steps is None:
            return data
        if hasattr(steps, "obs"):
            obs = steps.obs
            if obs:
                data["rgb"] = obs[0]
            if len(obs) > 1:
                data["state"] = obs[1].tolist()
        data["agent_id"] = agent_id
        return data


class MuJoCoEnvironmentBridge(SimulationEnvironment):
    """Gymnasium-compatible wrapper exposing the :class:`SimulationEnvironment` API."""

    def __init__(
        self,
        env_id: str,
        *,
        render_mode: str | None = None,
    ) -> None:
        self._env_id = env_id
        self._render_mode = render_mode
        self._env = None

    def initialize(self) -> None:
        if self._env is not None:
            return
        try:
            import gymnasium as gym
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Gymnasium is required for MuJoCo environments. Install via `pip install gymnasium[mujoco]`."
            ) from exc

        self._env = gym.make(self._env_id, render_mode=self._render_mode)

    def reset(self, **kwargs: Any) -> Dict[str, Any]:
        assert self._env is not None, "Call initialize() first"
        obs, _ = self._env.reset(**kwargs)
        payload: Dict[str, Any] = {"state": obs.tolist() if hasattr(obs, "tolist") else obs}
        if self._render_mode == "rgb_array":
            payload["rgb"] = self._env.render()
        return payload

    def step(
        self,
        action: Dict[str, Any] | List[float] | Any,
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        assert self._env is not None, "Call initialize() first"
        obs, reward, terminated, truncated, info = self._env.step(action)
        payload: Dict[str, Any] = {"state": obs.tolist() if hasattr(obs, "tolist") else obs}
        if self._render_mode == "rgb_array":
            payload["rgb"] = self._env.render()
        done = bool(terminated or truncated)
        return payload, float(reward), done, info

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None


class GymnasiumEnvironmentBridge(SimulationEnvironment):
    """Generic Gymnasium/Gym wrapper exposing the :class:`SimulationEnvironment` API."""

    def __init__(
        self,
        env_id: str,
        *,
        render_mode: str | None = None,
        kwargs: Dict[str, Any] | None = None,
    ) -> None:
        self._env_id = env_id
        self._render_mode = render_mode
        self._kwargs = dict(kwargs or {})
        self._env = None
        self._is_gymnasium = False

    def initialize(self) -> None:
        if self._env is not None:
            return
        gym = None
        try:
            import gymnasium as gym  # type: ignore

            self._is_gymnasium = True
        except Exception:
            try:
                import gym  # type: ignore

                self._is_gymnasium = False
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "Gymnasium or gym is required. Install via `pip install gymnasium` (recommended) or `pip install gym`."
                ) from exc

        kwargs = dict(self._kwargs)
        if self._render_mode is not None and self._is_gymnasium:
            kwargs.setdefault("render_mode", self._render_mode)
        self._env = gym.make(self._env_id, **kwargs)

    def reset(self, **kwargs: Any) -> Dict[str, Any]:
        assert self._env is not None, "Call initialize() first"
        if self._is_gymnasium:
            obs, info = self._env.reset(**kwargs)
        else:
            obs = self._env.reset(**kwargs)
            info = {}
        payload: Dict[str, Any] = {
            "state": obs.tolist() if hasattr(obs, "tolist") else obs,
            "info": info,
        }
        if self._render_mode == "rgb_array":
            try:
                payload["rgb"] = self._env.render()
            except Exception:
                pass
        return payload

    def step(
        self,
        action: Dict[str, Any] | List[float] | Any,
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        assert self._env is not None, "Call initialize() first"
        if self._is_gymnasium:
            obs, reward, terminated, truncated, info = self._env.step(action)
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = self._env.step(action)
        payload: Dict[str, Any] = {"state": obs.tolist() if hasattr(obs, "tolist") else obs}
        if self._render_mode == "rgb_array":
            try:
                payload["rgb"] = self._env.render()
            except Exception:
                pass
        return payload, float(reward), bool(done), dict(info or {})

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
