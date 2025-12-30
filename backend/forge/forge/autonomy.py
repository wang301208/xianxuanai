from __future__ import annotations

import logging
import os
import random
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from modules.brain.state import BrainRuntimeConfig
from modules.brain.whole_brain import WholeBrainSimulation
from modules.environment.simulator import GridWorldEnvironment
from modules.learning.experience_hub import EpisodeRecord, ExperienceHub

logger = logging.getLogger(__name__)


def _parse_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class HumanFeedbackRequest(BaseModel):
    task_id: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1)
    agent_response: str = Field(..., min_length=1)
    correct_response: Optional[str] = None
    rating: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AutonomyStatus(BaseModel):
    enabled: bool
    running: bool
    agent_id: str
    interval_seconds: float
    episode: int
    step: int
    last_reward: float
    last_done: bool
    env_position: tuple[int, int] | None = None
    env_goal: tuple[int, int] | None = None
    last_intention: str | None = None
    last_confidence: float | None = None


class _AutonomySettings(BaseModel):
    enabled: bool = True
    interval_seconds: float = Field(default=1.0, ge=0.05, le=60.0)
    max_episode_steps: int = Field(default=64, ge=4, le=4096)
    agent_id: str = Field(default="forge-autonomy", min_length=1)

    @classmethod
    def from_env(cls) -> "_AutonomySettings":
        enabled = _parse_bool(os.getenv("AUTONOMY_ENABLED"), default=True)
        interval_raw = os.getenv("AUTONOMY_LOOP_INTERVAL_SECONDS", os.getenv("AUTONOMY_INTERVAL_SECONDS"))
        max_steps_raw = os.getenv("AUTONOMY_MAX_EPISODE_STEPS")
        agent_id = os.getenv("AUTONOMY_AGENT_ID") or "forge-autonomy"
        try:
            interval = float(interval_raw) if interval_raw else 1.0
        except ValueError:
            interval = 1.0
        try:
            max_steps = int(max_steps_raw) if max_steps_raw else 64
        except ValueError:
            max_steps = 64
        return cls(
            enabled=enabled,
            interval_seconds=interval,
            max_episode_steps=max_steps,
            agent_id=agent_id,
        )


class AutonomousWholeBrainLoop:
    def __init__(self, settings: _AutonomySettings) -> None:
        self._settings = settings
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._base_interval_seconds = float(settings.interval_seconds)
        self._env_adapter: Any | None = None

        runtime = BrainRuntimeConfig(
            enable_continual_learning=True,
            enable_self_evolution=True,
            enable_meta_cognition=True,
        )
        self.brain = WholeBrainSimulation(config=runtime, neuromorphic=runtime.use_neuromorphic)
        self.env = GridWorldEnvironment(max_steps=settings.max_episode_steps)
        self._experience_hub = ExperienceHub(Path(runtime.continual_learning_experience_root))

        self._episode = 0
        self._step = 0
        self._episode_reward = 0.0
        self._last_feedback: Dict[str, Any] | None = None
        self._last_reward = 0.0
        self._last_done = False
        self._last_result: Any | None = None

    def start(self) -> None:
        if not self._settings.enabled:
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="whole-brain-autonomy", daemon=True)
        self._thread.start()
        self._maybe_start_env_adapter()

    def stop(self, *, timeout: float | None = 5.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None
        adapter = self._env_adapter
        if adapter is not None and hasattr(adapter, "stop"):
            try:
                adapter.stop(timeout=2.0)
            except Exception:
                pass
        try:
            self.brain.shutdown(wait=False)
        except Exception:
            logger.debug("WholeBrain shutdown failed.", exc_info=True)

    def status(self) -> AutonomyStatus:
        with self._lock:
            result = self._last_result
            return AutonomyStatus(
                enabled=bool(self._settings.enabled),
                running=bool(self._thread and self._thread.is_alive()),
                agent_id=self._settings.agent_id,
                interval_seconds=float(self._settings.interval_seconds),
                episode=int(self._episode),
                step=int(self._step),
                last_reward=float(self._last_reward),
                last_done=bool(self._last_done),
                env_position=tuple(self.env.position) if hasattr(self.env, "position") else None,
                env_goal=tuple(self.env.goal) if hasattr(self.env, "goal") else None,
                last_intention=getattr(getattr(result, "intent", None), "intention", None) if result else None,
                last_confidence=float(getattr(getattr(result, "intent", None), "confidence", 0.0))
                if result
                else None,
            )

    def submit_human_feedback(self, payload: HumanFeedbackRequest) -> Any:
        with self._lock:
            return self.brain.submit_human_feedback(
                task_id=payload.task_id,
                prompt=payload.prompt,
                agent_response=payload.agent_response,
                correct_response=payload.correct_response,
                rating=payload.rating,
                metadata=dict(payload.metadata or {}),
            )

    def pause(self) -> None:
        self.stop(timeout=2.0)

    def resume(self) -> None:
        self.start()

    # ------------------------------------------------------------------ #
    def _maybe_start_env_adapter(self) -> None:
        if not _parse_bool(os.getenv("AUTONOMY_ENV_ADAPTER_ENABLED"), default=False):
            return
        if self._env_adapter is not None:
            return
        try:
            from modules.environment.environment_adapter import EnvironmentAdapter
        except Exception:
            return

        interval_raw = os.getenv("AUTONOMY_ENV_ADAPTER_INTERVAL_SECONDS", "5")
        try:
            adapter_interval = float(interval_raw)
        except ValueError:
            adapter_interval = 5.0

        def _apply(update: Dict[str, Any]) -> None:
            conc = update.get("concurrency")
            meta = update.get("metadata") if isinstance(update.get("metadata"), dict) else {}
            base_max = meta.get("base_max_concurrency") if isinstance(meta, dict) else None
            try:
                conc_val = int(conc) if conc is not None else None
            except Exception:
                conc_val = None
            try:
                base_val = int(base_max) if base_max is not None else None
            except Exception:
                base_val = None
            if conc_val is None or conc_val <= 0:
                return
            if base_val is None or base_val <= 0:
                base_val = conc_val
            factor = float(base_val) / float(max(1, conc_val))
            recommended = self._base_interval_seconds * max(1.0, factor)
            recommended = max(0.05, min(60.0, float(recommended)))
            with self._lock:
                self._settings.interval_seconds = recommended

        try:
            adapter = EnvironmentAdapter(
                worker_id=f"forge-autonomy-env:{self._settings.agent_id}",
                event_bus=None,
                apply_callback=_apply,
                interval_seconds=adapter_interval,
            )
            adapter.start()
            self._env_adapter = adapter
        except Exception:
            logger.debug("Failed to start EnvironmentAdapter for autonomy loop.", exc_info=True)

    def _run(self) -> None:  # pragma: no cover - background thread
        obs = self.env.reset()
        with self._lock:
            self._episode = 1
            self._step = 0
            self._episode_reward = 0.0
            self._last_feedback = None
            self._last_reward = 0.0
            self._last_done = False

        while not self._stop.is_set():
            started = time.time()
            try:
                next_obs = self._tick(obs)
                obs = next_obs
            except Exception:
                logger.exception("Autonomy tick failed.")
            elapsed = time.time() - started
            delay = max(0.0, float(self._settings.interval_seconds) - elapsed)
            self._stop.wait(delay)

    def _tick(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        observation = obs.get("observation") if isinstance(obs, dict) else None
        if not isinstance(observation, dict):
            observation = {}

        with self._lock:
            self._step += 1
            pos = getattr(self.env, "position", (0, 0))
            goal = getattr(self.env, "goal", (0, 0))
            text = f"GridWorld episode={self._episode} step={self._step} pos={pos} goal={goal}"
            input_data: Dict[str, Any] = {
                "agent_id": self._settings.agent_id,
                "image": observation.get("vision"),
                "sound": observation.get("audio"),
                "text": text,
                "context": {
                    "task": "reach_goal",
                    "episode": float(self._episode),
                    "step": float(self._step),
                },
            }
            if self._last_feedback is not None:
                input_data["execution_feedback"] = dict(self._last_feedback)

            result = self.brain.process_cycle(input_data)
            self._last_result = result

            action = self._choose_action(result)
            step_result = self.env.step(action, {})
            reward = float(step_result.get("reward", 0.0) or 0.0)
            done = bool(step_result.get("done", False))
            info = step_result.get("info", {}) if isinstance(step_result.get("info"), dict) else {}
            success = 1.0 if done and reward > 0 else 0.0
            self._last_reward = reward
            self._last_done = done
            self._episode_reward += reward
            self._last_feedback = {
                "reward": reward,
                "success": success,
                "done": 1.0 if done else 0.0,
                "distance": float(info.get("distance", 0.0) or 0.0),
                "steps": float(info.get("steps", self._step) or self._step),
            }

            if done:
                self._persist_episode(info)
                self._episode += 1
                self._step = 0
                self._episode_reward = 0.0
                self._last_feedback = None
                return self.env.reset()

            return step_result

    def _choose_action(self, result: Any) -> str:
        intention = getattr(getattr(result, "intent", None), "intention", "") or "observe"
        pos = getattr(self.env, "position", (0, 0))
        goal = getattr(self.env, "goal", (0, 0))
        dx = int(goal[0] - pos[0])
        dy = int(goal[1] - pos[1])

        def _towards() -> str:
            if abs(dx) >= abs(dy):
                return "move_east" if dx > 0 else "move_west" if dx < 0 else ("move_south" if dy > 0 else "move_north")
            return "move_south" if dy > 0 else "move_north" if dy < 0 else ("move_east" if dx > 0 else "move_west")

        def _away() -> str:
            towards = _towards()
            inverse = {
                "move_east": "move_west",
                "move_west": "move_east",
                "move_south": "move_north",
                "move_north": "move_south",
            }
            return inverse.get(towards, "move_north")

        if intention == "approach":
            return _towards()
        if intention == "withdraw":
            return _away()
        if intention == "explore":
            return random.choice(["move_north", "move_south", "move_east", "move_west"])
        return "noop"

    def _persist_episode(self, info: Dict[str, Any]) -> None:
        try:
            episode_record = EpisodeRecord(
                task_id="gridworld",
                policy_version="whole_brain",
                total_reward=float(self._episode_reward),
                steps=int(info.get("steps", self._settings.max_episode_steps) or self._settings.max_episode_steps),
                success=bool(float(info.get("distance", 1.0) or 1.0) <= 0.0),
                metadata={
                    "episode": int(self._episode),
                    "final_position": list(getattr(self.env, "position", (0, 0))),
                    "goal": list(getattr(self.env, "goal", (0, 0))),
                    "info": dict(info),
                },
            )
            self._experience_hub.append(episode_record)
        except Exception:
            logger.debug("Failed to persist gridworld episode.", exc_info=True)


def build_autonomy_router(loop: AutonomousWholeBrainLoop) -> APIRouter:
    router = APIRouter()

    @router.get("/autonomy/status", response_model=AutonomyStatus)
    def autonomy_status() -> AutonomyStatus:
        return loop.status()

    @router.post("/autonomy/pause")
    def autonomy_pause() -> Dict[str, Any]:
        loop.pause()
        return {"status": "paused"}

    @router.post("/autonomy/resume")
    def autonomy_resume() -> Dict[str, Any]:
        loop.resume()
        return {"status": "running" if loop.status().running else "stopped"}

    @router.post("/autonomy/human_feedback")
    def autonomy_human_feedback(payload: HumanFeedbackRequest) -> Dict[str, Any]:
        try:
            result = loop.submit_human_feedback(payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"status": "accepted", "result": result}

    @router.get("/autonomy/brain_snapshot")
    def autonomy_brain_snapshot() -> Dict[str, Any]:
        with loop._lock:
            brain = loop.brain
            return {
                "cycle_index": int(getattr(brain, "cycle_index", 0)),
                "last_decision": getattr(brain, "last_decision", {}),
                "last_learning_prediction": getattr(brain, "last_learning_prediction", {}),
                "meta_learning": {
                    "learned_skills": {
                        k: {
                            "success_rate": float(v.get("success_rate", 0.0)),
                            "uses": int(v.get("uses", 0)),
                            "successes": int(v.get("successes", 0)),
                            "failures": int(v.get("failures", 0)),
                            "skill_id": v.get("skill_id"),
                        }
                        for k, v in getattr(brain.meta_learning, "learned_skills", {}).items()
                    }
                },
            }

    return router


def attach_autonomy(app: Any) -> None:
    settings = _AutonomySettings.from_env()
    loop = AutonomousWholeBrainLoop(settings)
    app.state.autonomy_loop = loop
    app.include_router(build_autonomy_router(loop), prefix="/internal", tags=["autonomy"])

    @app.on_event("startup")
    async def _startup() -> None:
        if settings.enabled:
            loop.start()
            logger.info(
                "Autonomy enabled (interval=%.2fs, max_steps=%s, agent_id=%s)",
                settings.interval_seconds,
                settings.max_episode_steps,
                settings.agent_id,
            )

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        loop.stop(timeout=3.0)
