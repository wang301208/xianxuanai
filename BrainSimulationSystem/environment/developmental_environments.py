"""Stage-aligned training environments built on the SimulationEnvironment interface.

This module provides a small curriculum of progressively richer environments
that all emit a consistent observation dict (rgb/audio/text/state), which is
converted into :class:`~BrainSimulationSystem.environment.base.PerceptionPacket`
via :class:`~BrainSimulationSystem.environment.base.EnvironmentAdapter`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from BrainSimulationSystem.config.stage_profiles import build_stage_config

from .base import EnvironmentAdapter, EnvironmentController, ObservationTransformer, SimulationEnvironment, UnityEnvironmentBridge


_RGB_BY_COLOR = {
    "red": (255, 60, 60),
    "green": (60, 220, 80),
    "blue": (80, 140, 255),
    "yellow": (245, 220, 60),
    "purple": (180, 90, 220),
    "orange": (255, 150, 60),
}


def _nested(mapping: Mapping[str, Any], key: str) -> Dict[str, Any]:
    value = mapping.get(key, {})
    return dict(value) if isinstance(value, dict) else {}


def _parse_size(value: Any, *, default: Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            height = int(value[0])
            width = int(value[1])
        except Exception:
            return default
        return max(1, height), max(1, width)
    return default


def _solid_image(color: str, *, size: Tuple[int, int]) -> np.ndarray:
    height, width = size
    rgb = _RGB_BY_COLOR.get(color, (200, 200, 200))
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = rgb[0]
    image[..., 1] = rgb[1]
    image[..., 2] = rgb[2]
    return image


@dataclass(frozen=True)
class StageEnvironmentConfig:
    stage: str
    kind: str = "toy_room"
    episode_length: int = 32
    vision_size: Tuple[int, int] = (64, 64)
    include_audio: bool = True
    object_count: int = 3
    object_colors: Tuple[str, ...] = ("red", "green", "blue")
    grid_size: int = 9
    unity_file_name: Optional[str] = None
    unity_kwargs: Dict[str, Any] = field(default_factory=dict)
    action_space: Tuple[Any, ...] = ()


@dataclass(frozen=True)
class StageEnvironmentBundle:
    stage: str
    config: StageEnvironmentConfig
    environment: SimulationEnvironment
    adapter: EnvironmentAdapter
    controller: EnvironmentController
    action_space: Tuple[Any, ...]


def extract_stage_environment_config(stage_config: Mapping[str, Any]) -> StageEnvironmentConfig:
    meta = _nested(stage_config, "metadata")
    stage = str(meta.get("stage") or stage_config.get("stage") or "unknown")

    env = _nested(stage_config, "environment")
    kind = str(env.get("kind", "toy_room") or "toy_room").strip().lower()

    learning = _nested(stage_config, "learning")
    interactive = _nested(learning, "interactive_language_loop")

    try:
        episode_length = int(env.get("episode_length") or interactive.get("max_steps") or 32)
    except Exception:
        episode_length = 32
    episode_length = max(1, episode_length)

    perception = _nested(stage_config, "perception")
    vision = _nested(perception, "vision")
    model = _nested(vision, "model")
    vision_size = _parse_size(env.get("vision_size") or model.get("input_size"), default=(64, 64))

    include_audio = env.get("include_audio")
    include_audio = bool(include_audio) if include_audio is not None else True

    try:
        object_count = int(env.get("object_count", 3))
    except Exception:
        object_count = 3
    object_count = max(1, object_count)

    colors = env.get("object_colors")
    if isinstance(colors, (list, tuple)) and colors:
        object_colors = tuple(str(c) for c in colors if str(c))
    else:
        palette = ("red", "green", "blue", "yellow", "purple", "orange")
        object_colors = tuple(palette[i % len(palette)] for i in range(object_count))

    grid = _nested(env, "grid_world")
    try:
        grid_size = int(grid.get("size", env.get("grid_size", 9)))
    except Exception:
        grid_size = 9
    grid_size = max(3, grid_size)

    unity = _nested(env, "unity")
    unity_file = unity.get("file_name")
    unity_file_name = str(unity_file).strip() if isinstance(unity_file, str) and unity_file.strip() else None
    unity_kwargs = {k: v for k, v in unity.items() if k != "file_name"}

    action_space = env.get("action_space")
    if isinstance(action_space, (list, tuple)) and action_space:
        action_space_tuple: Tuple[Any, ...] = tuple(action_space)
    else:
        action_space_tuple = ()

    return StageEnvironmentConfig(
        stage=stage,
        kind=kind,
        episode_length=episode_length,
        vision_size=vision_size,
        include_audio=include_audio,
        object_count=object_count,
        object_colors=object_colors,
        grid_size=grid_size,
        unity_file_name=unity_file_name,
        unity_kwargs=unity_kwargs,
        action_space=action_space_tuple,
    )


class ToyRoomEnvironment(SimulationEnvironment):
    """Simple closed-room environment with a handful of colourful objects."""

    ACTION_SPACE: Tuple[str, ...] = ("look_left", "look_right", "touch")

    def __init__(
        self,
        *,
        object_colors: Sequence[str],
        vision_size: Tuple[int, int] = (64, 64),
        episode_length: int = 32,
        include_audio: bool = True,
        seed: int | None = None,
    ) -> None:
        self._colors = [str(c) for c in object_colors] or ["red", "green", "blue"]
        self._vision_size = tuple(int(x) for x in vision_size)
        self._episode_length = int(max(1, episode_length))
        self._include_audio = bool(include_audio)
        self._rng = np.random.default_rng(seed)
        self._step = 0
        self._focus_idx = 0
        self._touched: set[int] = set()
        self._initialized = False

    def initialize(self) -> None:
        self._initialized = True

    def reset(self, **kwargs: Any) -> Dict[str, Any]:
        if not self._initialized:
            self.initialize()
        self._step = 0
        self._touched.clear()
        self._focus_idx = int(self._rng.integers(0, len(self._colors)))
        return self._observation()

    def step(self, action: Dict[str, Any] | Sequence[float] | Any):
        if not self._initialized:
            self.initialize()
        self._step += 1

        token = self._coerce_action(action)
        reward = 0.0
        touched_new = False

        if token == "look_left":
            self._focus_idx = (self._focus_idx - 1) % len(self._colors)
            reward = 0.01
        elif token == "look_right":
            self._focus_idx = (self._focus_idx + 1) % len(self._colors)
            reward = 0.01
        elif token == "touch":
            if self._focus_idx not in self._touched:
                self._touched.add(self._focus_idx)
                touched_new = True
                reward = 1.0
            else:
                reward = 0.1
        else:
            reward = 0.0

        terminated = self._step >= self._episode_length
        info = {
            "focus_idx": int(self._focus_idx),
            "focus_color": self._colors[self._focus_idx],
            "touched_new": bool(touched_new),
            "terminated": bool(terminated),
        }
        return self._observation(), float(reward), bool(terminated), info

    def close(self) -> None:
        self._initialized = False

    def _coerce_action(self, action: Any) -> str:
        if isinstance(action, str):
            return action
        if isinstance(action, dict):
            for key in ("token", "action", "discrete"):
                value = action.get(key)
                if isinstance(value, str):
                    return value
                if isinstance(value, (int, np.integer)):
                    idx = int(value)
                    if 0 <= idx < len(self.ACTION_SPACE):
                        return self.ACTION_SPACE[idx]
        if isinstance(action, (int, np.integer)):
            idx = int(action)
            if 0 <= idx < len(self.ACTION_SPACE):
                return self.ACTION_SPACE[idx]
        return str(action)

    def _observation(self, *, text: str | None = None) -> Dict[str, Any]:
        color = self._colors[self._focus_idx]
        obs: Dict[str, Any] = {
            "rgb": _solid_image(color, size=self._vision_size),
            "state": [
                float(self._focus_idx) / float(max(len(self._colors) - 1, 1)),
                float(self._step) / float(max(self._episode_length, 1)),
                float(len(self._touched)) / float(max(len(self._colors), 1)),
            ],
        }
        if self._include_audio:
            obs["audio"] = [float(self._focus_idx) / float(max(len(self._colors), 1))]
        if text is not None:
            obs["text"] = str(text)
        return obs


class TeacherInstructionEnvironment(ToyRoomEnvironment):
    """ToyRoom variant that provides a teacher instruction as text each step."""

    def __init__(
        self,
        *,
        object_colors: Sequence[str],
        vision_size: Tuple[int, int] = (128, 128),
        episode_length: int = 32,
        include_audio: bool = True,
        seed: int | None = None,
        instruction_templates: Sequence[str] | None = None,
    ) -> None:
        super().__init__(
            object_colors=object_colors,
            vision_size=vision_size,
            episode_length=episode_length,
            include_audio=include_audio,
            seed=seed,
        )
        self._templates = list(instruction_templates or ("Please touch the {color} object.",))
        self._target_color = None

    def reset(self, **kwargs: Any) -> Dict[str, Any]:
        obs = super().reset(**kwargs)
        self._target_color = self._colors[int(self._rng.integers(0, len(self._colors)))]
        return self._observation(text=self._instruction())

    def step(self, action: Dict[str, Any] | Sequence[float] | Any):
        token = self._coerce_action(action)
        if token == "touch":
            success = self._colors[self._focus_idx] == self._target_color
            reward = 1.0 if success else 0.0
            terminated = bool(success or self._step + 1 >= self._episode_length)
            self._step += 1
            info = {
                "focus_idx": int(self._focus_idx),
                "focus_color": self._colors[self._focus_idx],
                "target_color": self._target_color,
                "instruction": self._instruction(),
                "teacher_action": self._teacher_action_hint(),
                "success": bool(success),
                "task_success": bool(success),
            }
            return self._observation(text=self._instruction()), float(reward), bool(terminated), info

        obs, reward, terminated, info = super().step(action)
        info = dict(info or {})
        info.update(
            {
                "target_color": self._target_color,
                "instruction": self._instruction(),
                "teacher_action": self._teacher_action_hint(),
                "success": False,
                "task_success": False,
            }
        )
        obs["text"] = self._instruction()
        return obs, float(reward), bool(terminated), info

    def _instruction(self) -> str:
        template = self._templates[0] if self._templates else "Touch the {color} object."
        return str(template).format(color=self._target_color or "red")

    def _teacher_action_hint(self) -> str:
        if self._target_color is None:
            return "touch"
        if self._colors[self._focus_idx] == self._target_color:
            return "touch"
        target_idx = self._colors.index(self._target_color)
        dist_right = (target_idx - self._focus_idx) % len(self._colors)
        dist_left = (self._focus_idx - target_idx) % len(self._colors)
        return "look_right" if dist_right <= dist_left else "look_left"


class GridWorldEnvironment(SimulationEnvironment):
    """Small open-world-like grid with a goal location."""

    ACTION_SPACE: Tuple[str, ...] = ("north", "south", "west", "east", "interact")

    def __init__(
        self,
        *,
        grid_size: int = 9,
        vision_size: Tuple[int, int] = (192, 192),
        episode_length: int = 64,
        include_audio: bool = False,
        seed: int | None = None,
    ) -> None:
        self._size = int(max(3, grid_size))
        self._vision_size = tuple(int(x) for x in vision_size)
        self._episode_length = int(max(1, episode_length))
        self._include_audio = bool(include_audio)
        self._rng = np.random.default_rng(seed)
        self._step = 0
        self._pos = (0, 0)
        self._goal = (self._size - 1, self._size - 1)
        self._initialized = False

    def initialize(self) -> None:
        self._initialized = True

    def reset(self, **kwargs: Any) -> Dict[str, Any]:
        if not self._initialized:
            self.initialize()
        self._step = 0
        self._pos = (0, 0)
        self._goal = (self._size - 1, self._size - 1)
        return self._observation()

    def step(self, action: Dict[str, Any] | Sequence[float] | Any):
        if not self._initialized:
            self.initialize()
        self._step += 1

        token = self._coerce_action(action)
        x, y = self._pos
        if token == "north":
            y = max(0, y - 1)
        elif token == "south":
            y = min(self._size - 1, y + 1)
        elif token == "west":
            x = max(0, x - 1)
        elif token == "east":
            x = min(self._size - 1, x + 1)
        elif token == "interact":
            pass
        self._pos = (x, y)

        success = token == "interact" and self._pos == self._goal
        reward = 1.0 if success else 0.0
        terminated = bool(success or self._step >= self._episode_length)
        info = {
            "position": self._pos,
            "goal": self._goal,
            "success": bool(success),
            "task_success": bool(success),
        }
        return self._observation(), float(reward), bool(terminated), info

    def close(self) -> None:
        self._initialized = False

    def _coerce_action(self, action: Any) -> str:
        if isinstance(action, str):
            return action
        if isinstance(action, dict):
            for key in ("token", "action", "discrete"):
                value = action.get(key)
                if isinstance(value, str):
                    return value
                if isinstance(value, (int, np.integer)):
                    idx = int(value)
                    if 0 <= idx < len(self.ACTION_SPACE):
                        return self.ACTION_SPACE[idx]
        if isinstance(action, (int, np.integer)):
            idx = int(action)
            if 0 <= idx < len(self.ACTION_SPACE):
                return self.ACTION_SPACE[idx]
        return str(action)

    def _observation(self) -> Dict[str, Any]:
        height, width = self._vision_size
        image = np.zeros((height, width, 3), dtype=np.uint8)

        cell_h = max(1, height // self._size)
        cell_w = max(1, width // self._size)

        gx, gy = self._goal
        x, y = self._pos

        def _paint(cell_x: int, cell_y: int, rgb: Tuple[int, int, int]):
            top = cell_y * cell_h
            left = cell_x * cell_w
            image[top : top + cell_h, left : left + cell_w, 0] = rgb[0]
            image[top : top + cell_h, left : left + cell_w, 1] = rgb[1]
            image[top : top + cell_h, left : left + cell_w, 2] = rgb[2]

        _paint(gx, gy, _RGB_BY_COLOR["red"])
        _paint(x, y, _RGB_BY_COLOR["green"])

        size_minus = float(max(self._size - 1, 1))
        obs: Dict[str, Any] = {
            "rgb": image,
            "state": [
                float(x) / size_minus,
                float(y) / size_minus,
                float(gx) / size_minus,
                float(gy) / size_minus,
                float(self._step) / float(max(self._episode_length, 1)),
            ],
        }
        if self._include_audio:
            manhattan = abs(gx - x) + abs(gy - y)
            obs["audio"] = [float(manhattan) / float(max(1, 2 * (self._size - 1)))]
        return obs


def build_stage_environment(
    stage_config: Mapping[str, Any],
    *,
    world_model: Any | None = None,
    agent_id: str = "agent",
) -> StageEnvironmentBundle:
    """Build a stage-appropriate environment + controller bundle."""

    cfg = extract_stage_environment_config(stage_config)

    kind = cfg.kind
    env: SimulationEnvironment
    action_space: Tuple[Any, ...]

    if kind == "toy_room":
        env = ToyRoomEnvironment(
            object_colors=cfg.object_colors[: cfg.object_count],
            vision_size=cfg.vision_size,
            episode_length=cfg.episode_length,
            include_audio=cfg.include_audio,
        )
        action_space = ToyRoomEnvironment.ACTION_SPACE
    elif kind == "toy_teacher":
        env = TeacherInstructionEnvironment(
            object_colors=cfg.object_colors[: cfg.object_count],
            vision_size=cfg.vision_size,
            episode_length=cfg.episode_length,
            include_audio=cfg.include_audio,
        )
        action_space = ToyRoomEnvironment.ACTION_SPACE
    elif kind in {"grid_world", "open_world"}:
        if kind == "open_world" and cfg.unity_file_name:
            if not cfg.action_space:
                raise ValueError("open_world with Unity requires environment.action_space to be configured")
            env = UnityEnvironmentBridge(cfg.unity_file_name, **cfg.unity_kwargs)
            action_space = cfg.action_space
        else:
            env = GridWorldEnvironment(
                grid_size=cfg.grid_size,
                vision_size=cfg.vision_size,
                episode_length=cfg.episode_length,
                include_audio=cfg.include_audio,
            )
            action_space = GridWorldEnvironment.ACTION_SPACE
    else:
        raise ValueError(f"Unsupported environment kind: {cfg.kind!r}")

    adapter = EnvironmentAdapter(env, transformer=ObservationTransformer())
    controller = EnvironmentController(adapter, world_model=world_model, agent_id=agent_id)
    return StageEnvironmentBundle(
        stage=cfg.stage,
        config=cfg,
        environment=env,
        adapter=adapter,
        controller=controller,
        action_space=tuple(action_space),
    )


def build_stage_environment_from_stage(
    stage: str,
    *,
    overrides: Mapping[str, Any] | None = None,
    base_profile: str | None = None,
    world_model: Any | None = None,
    agent_id: str = "agent",
) -> StageEnvironmentBundle:
    """Convenience helper that loads a stage config then builds its environment."""

    stage_config = build_stage_config(stage, overrides=dict(overrides or {}), base_profile=base_profile)
    return build_stage_environment(stage_config, world_model=world_model, agent_id=agent_id)


__all__ = [
    "GridWorldEnvironment",
    "StageEnvironmentBundle",
    "StageEnvironmentConfig",
    "TeacherInstructionEnvironment",
    "ToyRoomEnvironment",
    "build_stage_environment",
    "build_stage_environment_from_stage",
    "extract_stage_environment_config",
]

