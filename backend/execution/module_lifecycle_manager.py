from __future__ import annotations

"""Continuous module lifecycle management (track -> suggest prune -> optional actions).

This component complements module acquisition by handling the *other half* of
the lifecycle: unloading/retiring modules that are unused or redundant.

It is intentionally conservative:
- default is suggest-only (no disables/uninstalls)
- optional auto-unload can be enabled to reclaim memory from idle modules

It listens to existing runtime events:
- `module.loaded` / `module.unloaded` / `module.used` / `module.requirements`
- `resource.adaptation.architecture` (to learn which module flags trend off)
- `learning.cycle_completed` (evaluation tick)

It publishes suggestions/actions as events:
- `module.lifecycle.suggest_unload`
- `module.lifecycle.suggest_disable`
- `module.lifecycle.suggest_uninstall`
- `module.lifecycle.unloaded` (when auto-unload is enabled)
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

try:  # optional in some deployments
    from events import EventBus  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    EventBus = None  # type: ignore

import importlib
import re

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return int(default)
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except Exception:
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _ema(prev: float | None, value: float, *, alpha: float = 0.2) -> float:
    if prev is None:
        return float(value)
    return float(prev) * (1.0 - float(alpha)) + float(value) * float(alpha)


_MODULE_FLAG_RE = re.compile(r"^module_(?P<name>[a-zA-Z0-9_\-]+)_flag$")


@dataclass
class ModuleStats:
    name: str
    loads: int = 0
    unloads: int = 0
    required: int = 0
    last_used_ts: float = 0.0
    last_load_ts: float = 0.0
    last_unload_ts: float = 0.0
    total_loaded_secs: float = 0.0
    currently_loaded: bool = False
    loaded_since_ts: float | None = None
    load_seconds_ema: float | None = None
    last_load_seconds: float | None = None
    arch_flag_seen: int = 0
    arch_flag_off: int = 0
    last_arch_ts: float = 0.0
    enabled: bool | None = None
    notes: Dict[str, Any] = field(default_factory=dict)

    def arch_off_ratio(self) -> float:
        if self.arch_flag_seen <= 0:
            return 0.0
        return float(self.arch_flag_off) / float(self.arch_flag_seen)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["arch_off_ratio"] = float(self.arch_off_ratio())
        return data


class ModuleLifecycleManager:
    def __init__(
        self,
        *,
        event_bus: EventBus,
        module_manager: Any | None = None,
        enabled: bool | None = None,
        eval_interval_secs: float | None = None,
        unload_idle_secs: float | None = None,
        disable_idle_secs: float | None = None,
        uninstall_idle_secs: float | None = None,
        min_uses_to_keep: int | None = None,
        high_cost_load_secs: float | None = None,
        arch_off_threshold: float | None = None,
        arch_min_samples: int | None = None,
        auto_unload: bool | None = None,
        cost_overrides: Mapping[str, float] | None = None,
        disabled_state_path: str | os.PathLike[str] | None = None,
        logger_: Optional[logging.Logger] = None,
    ) -> None:
        if event_bus is None:
            raise ValueError("event_bus is required")
        self._bus = event_bus
        self._module_manager = module_manager
        self._logger = logger_ or logger

        self.enabled = _env_bool("MODULE_LIFECYCLE_ENABLED", False) if enabled is None else bool(enabled)
        self._eval_interval = (
            _env_float("MODULE_LIFECYCLE_EVAL_INTERVAL_SECS", 600.0)
            if eval_interval_secs is None
            else float(eval_interval_secs)
        )
        self._unload_idle_secs = (
            _env_float("MODULE_LIFECYCLE_UNLOAD_IDLE_SECS", 900.0)
            if unload_idle_secs is None
            else float(unload_idle_secs)
        )
        self._disable_idle_secs = (
            _env_float("MODULE_LIFECYCLE_DISABLE_IDLE_SECS", 86400.0)
            if disable_idle_secs is None
            else float(disable_idle_secs)
        )
        self._uninstall_idle_secs = (
            _env_float("MODULE_LIFECYCLE_UNINSTALL_IDLE_SECS", 604800.0)
            if uninstall_idle_secs is None
            else float(uninstall_idle_secs)
        )
        self._min_uses_to_keep = (
            _env_int("MODULE_LIFECYCLE_MIN_USES", 3) if min_uses_to_keep is None else int(min_uses_to_keep)
        )
        self._high_cost_load_secs = (
            _env_float("MODULE_LIFECYCLE_HIGH_COST_LOAD_SECS", 1.5)
            if high_cost_load_secs is None
            else float(high_cost_load_secs)
        )
        self._arch_off_threshold = (
            _env_float("MODULE_LIFECYCLE_ARCH_OFF_THRESHOLD", 0.8)
            if arch_off_threshold is None
            else float(arch_off_threshold)
        )
        self._arch_min_samples = (
            _env_int("MODULE_LIFECYCLE_ARCH_MIN_SAMPLES", 10)
            if arch_min_samples is None
            else int(arch_min_samples)
        )
        self._auto_unload = _env_bool("MODULE_LIFECYCLE_AUTO_UNLOAD", False) if auto_unload is None else bool(auto_unload)
        self._suggest_cooldown = _env_float("MODULE_LIFECYCLE_SUGGEST_COOLDOWN_SECS", 3600.0)

        self._cost_overrides = {str(k): float(v) for k, v in dict(cost_overrides or {}).items() if str(k)}

        path = disabled_state_path or os.getenv("MODULE_LIFECYCLE_DISABLED_PATH") or "memory/disabled_modules.json"
        self._disabled_state_path = Path(path)

        self._stats: Dict[str, ModuleStats] = {}
        self._last_suggested: Dict[tuple[str, str], float] = {}
        self._last_eval_ts: float = 0.0
        self._started_ts: float = time.time()

        # Apply persisted disabled modules (best-effort).
        if self.enabled:
            self._apply_persisted_disabled()

        self._subscriptions: list[Callable[[], None]] = [
            self._bus.subscribe("module.loaded", self._on_module_loaded),
            self._bus.subscribe("module.unloaded", self._on_module_unloaded),
            self._bus.subscribe("module.used", self._on_module_used),
            self._bus.subscribe("module.requirements", self._on_module_requirements),
            self._bus.subscribe("resource.adaptation.architecture", self._on_architecture_event),
            self._bus.subscribe("learning.cycle_completed", self._on_learning_cycle),
            self._bus.subscribe("module.lifecycle.unload", self._on_unload_request),
            self._bus.subscribe("module.lifecycle.disable", self._on_disable_request),
            self._bus.subscribe("module.lifecycle.enable", self._on_enable_request),
        ]

    def close(self) -> None:
        subs = list(self._subscriptions)
        self._subscriptions.clear()
        for cancel in subs:
            try:
                cancel()
            except Exception:
                continue

    # ------------------------------------------------------------------ capability imports
    def _capability_import_prefixes(self) -> list[str]:
        prefixes: list[str] = []
        manager_module = ""
        if self._module_manager is not None:
            try:
                manager_module = str(getattr(type(self._module_manager), "__module__", "") or "")
            except Exception:
                manager_module = ""
        if manager_module.startswith("capability."):
            prefixes.append("capability")
        elif manager_module.startswith("backend.capability."):
            prefixes.append("backend.capability")
        for candidate in ("capability", "backend.capability"):
            if candidate not in prefixes:
                prefixes.append(candidate)
        return prefixes

    def _import_capability_module(self, suffix: str) -> Any | None:
        suffix = str(suffix or "").strip().lstrip(".")
        if not suffix:
            return None
        for prefix in self._capability_import_prefixes():
            try:
                return importlib.import_module(f"{prefix}.{suffix}")
            except Exception:
                continue
        return None

    # ------------------------------------------------------------------ persistence
    def _apply_persisted_disabled(self) -> None:
        registry = self._import_capability_module("module_registry")
        disable = getattr(registry, "disable_module", None) if registry is not None else None
        if not callable(disable):
            return
        if not self._disabled_state_path.exists():
            return
        try:
            data = json.loads(self._disabled_state_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(data, list):
            return
        for item in data:
            token = str(item or "").strip()
            if not token:
                continue
            try:
                disable(token)
            except Exception:
                continue

    def _persist_disabled(self) -> None:
        registry = self._import_capability_module("module_registry")
        disabled_fn = getattr(registry, "disabled_modules", None) if registry is not None else None
        if not callable(disabled_fn):
            return
        try:
            disabled = list(disabled_fn())
        except Exception:
            return
        try:
            self._disabled_state_path.parent.mkdir(parents=True, exist_ok=True)
            self._disabled_state_path.write_text(json.dumps(disabled, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            return

    # ------------------------------------------------------------------ stats helpers
    def _ensure(self, name: str) -> ModuleStats:
        token = str(name or "").strip()
        if not token:
            token = "<unknown>"
        stats = self._stats.get(token)
        if stats is None:
            stats = ModuleStats(name=token)
            self._stats[token] = stats
        return stats

    def snapshot(self, *, limit: int = 50) -> list[Dict[str, Any]]:
        items = list(self._stats.values())
        items.sort(key=lambda s: (s.last_used_ts, s.loads, s.required), reverse=True)
        limit = max(0, int(limit))
        if limit:
            items = items[:limit]
        return [s.to_dict() for s in items]

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, *, now: float | None = None) -> None:
        if not self.enabled:
            return
        now_ts = time.time() if now is None else float(now)
        if self._eval_interval > 0 and (now_ts - float(self._last_eval_ts)) < float(self._eval_interval):
            return
        self._last_eval_ts = now_ts

        registry = self._import_capability_module("module_registry")
        available = getattr(registry, "available_modules", None) if registry is not None else None
        is_enabled = getattr(registry, "is_module_enabled", None) if registry is not None else None
        all_modules: Sequence[str] = ()
        if callable(available):
            try:
                all_modules = list(available(include_disabled=True))
            except TypeError:
                all_modules = list(available())
            except Exception:
                all_modules = ()

        for name in list(all_modules):
            stats = self._ensure(name)
            if callable(is_enabled):
                try:
                    stats.enabled = bool(is_enabled(name))
                except Exception:
                    stats.enabled = None

        loaded_modules: Sequence[str] = ()
        if self._module_manager is not None and hasattr(self._module_manager, "loaded_modules"):
            try:
                loaded_modules = list(self._module_manager.loaded_modules())
            except Exception:
                loaded_modules = ()
        loaded_set = {str(n) for n in loaded_modules if str(n)}

        def _should_suggest(module: str, action: str) -> bool:
            if float(self._suggest_cooldown) <= 0:
                return True
            key = (str(module), str(action))
            last = float(self._last_suggested.get(key, 0.0) or 0.0)
            if (now_ts - last) < float(self._suggest_cooldown):
                return False
            self._last_suggested[key] = float(now_ts)
            return True

        # Suggest unloading idle loaded modules.
        for name in sorted(loaded_set):
            stats = self._ensure(name)
            stats.currently_loaded = True
            last_used = float(stats.last_used_ts or stats.last_load_ts or 0.0)
            idle = now_ts - last_used if last_used > 0 else float("inf")
            if self._unload_idle_secs >= 0 and idle >= float(self._unload_idle_secs):
                if not _should_suggest(name, "unload"):
                    continue
                suggestion = {
                    "time": now_ts,
                    "module": name,
                    "action": "unload",
                    "reason": f"idle_for_secs={idle:.1f}",
                    "stats": stats.to_dict(),
                }
                try:
                    self._bus.publish("module.lifecycle.suggest_unload", suggestion)
                except Exception:
                    pass
                if self._auto_unload and self._module_manager is not None and hasattr(self._module_manager, "unload"):
                    try:
                        self._module_manager.unload(name)
                        self._bus.publish("module.lifecycle.unloaded", {"time": time.time(), "module": name, "reason": "auto_unload"})
                    except Exception:
                        pass

        # Suggest disabling rarely used + costly/redundant modules.
        for name, stats in list(self._stats.items()):
            if name in loaded_set:
                continue
            enabled_state = True if stats.enabled is None else bool(stats.enabled)
            if not enabled_state:
                continue
            uses = max(int(stats.loads), int(stats.required))
            last_seen = max(
                float(stats.last_used_ts or 0.0),
                float(stats.last_arch_ts or 0.0),
                float(stats.last_unload_ts or 0.0),
                float(stats.last_load_ts or 0.0),
            )
            if last_seen <= 0:
                last_seen = float(self._started_ts)
            idle = now_ts - float(last_seen)

            cost = self._cost_overrides.get(name)
            if cost is None:
                cost = stats.load_seconds_ema or 0.0
            off_ratio = float(stats.arch_off_ratio())
            if uses >= int(self._min_uses_to_keep):
                continue
            arch_redundant = (
                int(stats.arch_flag_seen) >= int(max(1, self._arch_min_samples))
                and float(off_ratio) >= float(self._arch_off_threshold)
            )
            costly_and_idle = float(idle) >= float(self._disable_idle_secs) and float(cost) >= float(self._high_cost_load_secs)
            if not arch_redundant and not costly_and_idle:
                continue
            if not _should_suggest(name, "disable"):
                continue

            suggestion = {
                "time": now_ts,
                "module": name,
                "action": "disable",
                "reason": f"rarely_used={uses};idle_secs={idle:.0f};cost={float(cost):.3f};arch_off_ratio={off_ratio:.2f}",
                "stats": stats.to_dict(),
            }
            try:
                self._bus.publish("module.lifecycle.suggest_disable", suggestion)
            except Exception:
                pass

        # Suggest uninstalling modules that are disabled and long-unused.
        for name, stats in list(self._stats.items()):
            enabled_state = True if stats.enabled is None else bool(stats.enabled)
            if enabled_state:
                continue
            uses = max(int(stats.loads), int(stats.required))
            last_seen = max(
                float(stats.last_used_ts or 0.0),
                float(stats.last_arch_ts or 0.0),
                float(stats.last_unload_ts or 0.0),
                float(stats.last_load_ts or 0.0),
            )
            if last_seen <= 0:
                last_seen = float(self._started_ts)
            idle = now_ts - float(last_seen)
            off_ratio = float(stats.arch_off_ratio())
            arch_redundant = (
                int(stats.arch_flag_seen) >= int(max(1, self._arch_min_samples))
                and float(off_ratio) >= float(self._arch_off_threshold)
            )
            if float(idle) < float(self._uninstall_idle_secs) and not arch_redundant:
                continue
            if uses > 0 and not arch_redundant:
                continue
            if not _should_suggest(name, "uninstall"):
                continue
            suggestion = {
                "time": now_ts,
                "module": name,
                "action": "uninstall",
                "reason": f"disabled_and_idle_secs={idle:.0f};arch_off_ratio={off_ratio:.2f}",
                "stats": stats.to_dict(),
            }
            try:
                self._bus.publish("module.lifecycle.suggest_uninstall", suggestion)
            except Exception:
                pass

        try:
            self._bus.publish("module.lifecycle.snapshot", {"time": now_ts, "modules": self.snapshot(limit=200)})
        except Exception:
            pass

    # ------------------------------------------------------------------ event handlers
    async def _on_module_loaded(self, event: Dict[str, Any]) -> None:
        if not self.enabled or not isinstance(event, Mapping):
            return
        name = event.get("module")
        if not isinstance(name, str):
            return
        now = float(event.get("time", time.time()) or time.time())
        stats = self._ensure(name)
        stats.loads += 1
        stats.currently_loaded = True
        stats.loaded_since_ts = now
        stats.last_load_ts = now
        stats.last_used_ts = max(float(stats.last_used_ts), now)

        load_seconds = _safe_float(event.get("load_seconds"))
        if load_seconds is not None:
            stats.last_load_seconds = float(load_seconds)
            stats.load_seconds_ema = _ema(stats.load_seconds_ema, float(load_seconds))

    async def _on_module_unloaded(self, event: Dict[str, Any]) -> None:
        if not self.enabled or not isinstance(event, Mapping):
            return
        name = event.get("module")
        if not isinstance(name, str):
            return
        now = float(event.get("time", time.time()) or time.time())
        stats = self._ensure(name)
        stats.unloads += 1
        stats.currently_loaded = False
        stats.last_unload_ts = now
        loaded_secs = _safe_float(event.get("loaded_seconds"))
        if loaded_secs is None:
            if stats.loaded_since_ts:
                loaded_secs = max(0.0, now - float(stats.loaded_since_ts))
        if loaded_secs is not None:
            stats.total_loaded_secs += float(max(0.0, loaded_secs))
        stats.loaded_since_ts = None

    async def _on_module_used(self, event: Dict[str, Any]) -> None:
        if not self.enabled or not isinstance(event, Mapping):
            return
        name = event.get("module")
        if not isinstance(name, str):
            return
        now = float(event.get("time", time.time()) or time.time())
        stats = self._ensure(name)
        stats.last_used_ts = max(float(stats.last_used_ts), now)

    async def _on_module_requirements(self, event: Dict[str, Any]) -> None:
        if not self.enabled or not isinstance(event, Mapping):
            return
        now = float(event.get("time", time.time()) or time.time())
        needed = event.get("needed") or event.get("modules") or []
        if not isinstance(needed, (list, tuple)):
            return
        for item in needed:
            name = str(item or "").strip()
            if not name:
                continue
            stats = self._ensure(name)
            stats.required += 1
            stats.last_used_ts = max(float(stats.last_used_ts), now)

    async def _on_architecture_event(self, event: Dict[str, Any]) -> None:
        if not self.enabled or not isinstance(event, Mapping):
            return
        arch = event.get("architecture")
        if not isinstance(arch, Mapping):
            return
        now = time.time()
        for key, value in arch.items():
            match = _MODULE_FLAG_RE.match(str(key))
            if not match:
                continue
            name = match.group("name")
            score = _safe_float(value)
            if score is None:
                continue
            stats = self._ensure(name)
            stats.last_arch_ts = float(now)
            stats.arch_flag_seen += 1
            if float(score) < 0.5:
                stats.arch_flag_off += 1

    async def _on_learning_cycle(self, event: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        now = _safe_float(event.get("time")) if isinstance(event, Mapping) else None
        self.evaluate(now=now)

    async def _on_unload_request(self, event: Dict[str, Any]) -> None:
        if not self.enabled or self._module_manager is None:
            return
        name = event.get("module") if isinstance(event, Mapping) else None
        if not isinstance(name, str) or not name.strip():
            return
        try:
            self._module_manager.unload(name)
        except Exception:
            return

    async def _on_disable_request(self, event: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        name = event.get("module") if isinstance(event, Mapping) else None
        if not isinstance(name, str) or not name.strip():
            return
        registry = self._import_capability_module("module_registry")
        disable = getattr(registry, "disable_module", None) if registry is not None else None
        if not callable(disable):
            return
        try:
            disable(name)
            self._persist_disabled()
        except Exception:
            return

    async def _on_enable_request(self, event: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        name = event.get("module") if isinstance(event, Mapping) else None
        if not isinstance(name, str) or not name.strip():
            return
        registry = self._import_capability_module("module_registry")
        enable = getattr(registry, "enable_module", None) if registry is not None else None
        if not callable(enable):
            return
        try:
            enable(name)
            self._persist_disabled()
        except Exception:
            return


__all__ = ["ModuleLifecycleManager", "ModuleStats"]
