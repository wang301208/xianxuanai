from __future__ import annotations

import json
import logging
import sqlite3
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import aiofiles
from cachetools import LRUCache, TTLCache


logger = logging.getLogger(__name__)


PYTHON_SKILL_TYPE = "python"
CALLABLE_SKILL_TYPE = "callable"
SPECIALIST_SKILL_TYPE = "specialist"


def _normalise_metadata(value: Any) -> Any:
    """Return a JSON-serialisable representation of ``value``."""

    if isinstance(value, dict):
        return {key: _normalise_metadata(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise_metadata(item) for item in value]
    if isinstance(value, set):
        return sorted(_normalise_metadata(item) for item in value)
    return value


class SkillLibrary:
    """Store and retrieve skill source code and metadata in a Git repository."""

    def __init__(
        self,
        repo_path: str | Path,
        storage_dir: str = "skills",
        cache_size: int = 128,
        cache_ttl: int | None = None,
        persist_path: str | Path | None = None,
    ) -> None:
        self.repo_path = Path(repo_path)
        self.storage_dir = self.repo_path / storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self._cache_ttl = cache_ttl
        if cache_ttl is not None:
            self._cache: "TTLCache[str, Tuple[str, Dict]]" = TTLCache(
                maxsize=cache_size, ttl=cache_ttl
            )
        else:
            self._cache: "LRUCache[str, Tuple[str, Dict]]" = LRUCache(maxsize=cache_size)

        self.hits = 0
        self.misses = 0

        self.persist_path = Path(persist_path or (self.storage_dir / "cache.sqlite"))
        self._dynamic_skills: Dict[str, Dict[str, Any]] = {}
        self._callable_registry: Dict[str, Callable[..., Any]] = {}
        self._callable_lookup: Dict[str, str] = {}
        self._init_persist()
        self._load_dynamic_registry()
        self._warm_cache()

    # ------------------------------------------------------------------
    # Persistence helpers
    def _init_persist(self) -> None:
        self._db = sqlite3.connect(self.persist_path)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS cache (name TEXT PRIMARY KEY, code TEXT, metadata TEXT, timestamp REAL)"
        )
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS dynamic_skills (name TEXT PRIMARY KEY, metadata TEXT)"
        )
        self._db.commit()

    def _load_dynamic_registry(self) -> None:
        cur = self._db.execute("SELECT name, metadata FROM dynamic_skills")
        for name, metadata_json in cur.fetchall():
            try:
                metadata = json.loads(metadata_json)
            except json.JSONDecodeError:
                logger.warning("Failed to decode metadata for dynamic skill %s", name)
                continue
            self._dynamic_skills[name] = metadata
            self._cache.setdefault(name, (metadata.get("code", ""), metadata))
        self._db.commit()

    def _save_to_persist(self, name: str, code: str, metadata: Dict) -> None:
        ts = time.time()
        self._db.execute(
            "INSERT OR REPLACE INTO cache (name, code, metadata, timestamp) VALUES (?, ?, ?, ?)",
            (name, code, json.dumps(metadata), ts),
        )
        self._db.commit()

    def _save_dynamic_metadata(self, name: str, metadata: Dict[str, Any]) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO dynamic_skills (name, metadata) VALUES (?, ?)",
            (name, json.dumps(metadata)),
        )
        self._db.commit()

    def _load_from_persist(self, name: str) -> Tuple[str, Dict] | None:
        cur = self._db.execute(
            "SELECT code, metadata, timestamp FROM cache WHERE name = ?", (name,)
        )
        row = cur.fetchone()
        if not row:
            return None
        code, metadata_json, ts = row
        if self._cache_ttl is not None and time.time() - ts > self._cache_ttl:
            self._db.execute("DELETE FROM cache WHERE name = ?", (name,))
            self._db.commit()
            return None
        return code, json.loads(metadata_json)

    def _warm_cache(self) -> None:
        cur = self._db.execute(
            "SELECT name, code, metadata, timestamp FROM cache ORDER BY timestamp"
        )
        rows = cur.fetchall()
        now = time.time()
        for name, code, metadata_json, ts in rows:
            if self._cache_ttl is not None and now - ts > self._cache_ttl:
                self._db.execute("DELETE FROM cache WHERE name = ?", (name,))
                continue
            self._cache[name] = (code, json.loads(metadata_json))
        self._db.commit()

    def close(self) -> None:
        try:
            self._db.close()
        except Exception:
            pass

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        self.close()

    # ------------------------------------------------------------------
    # Public API
    def add_skill(self, name: str, code: str, metadata: Dict) -> None:
        """Add a skill to the library and commit the change to Git."""
        skill_file = self.storage_dir / f"{name}.py"
        meta_file = self.storage_dir / f"{name}.json"
        skill_file.write_text(code, encoding="utf-8")
        if name.startswith("MetaSkill_") and "active" not in metadata:
            metadata["active"] = False
        meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        subprocess.run(
            ["git", "add", str(skill_file), str(meta_file)],
            cwd=self.repo_path,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", f"Add skill {name}"],
            cwd=self.repo_path,
            check=True,
        )
        # Remove any stale cached entry for this skill.
        self.invalidate(name)

    def register_callable_skill(
        self,
        name: str,
        func: Callable[..., Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a callable-backed skill without creating a Python snippet."""

        meta: Dict[str, Any] = dict(metadata or {})
        callable_id = meta.get("callable_id") or f"{name}:{uuid.uuid4().hex}"
        meta["callable_id"] = callable_id
        meta.setdefault("type", CALLABLE_SKILL_TYPE)
        meta.setdefault("signature", f"{meta['type']}::{name}")
        normalised_meta = _normalise_metadata(meta)

        self._callable_registry[callable_id] = func
        self._callable_lookup[name] = callable_id
        self._dynamic_skills[name] = normalised_meta
        self._cache[name] = ("", normalised_meta)
        self._save_dynamic_metadata(name, normalised_meta)
        self._save_to_persist(name, "", normalised_meta)

    def resolve_callable(self, name: str, metadata: Dict[str, Any]) -> Callable[..., Any] | None:
        """Return the callable registered for ``name`` when present."""

        callable_id = metadata.get("callable_id") or self._callable_lookup.get(name)
        if not callable_id:
            return None
        return self._callable_registry.get(callable_id)

    async def _read_file(self, path: Path) -> str:
        """Read text from ``path`` asynchronously."""
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            return await f.read()

    async def _load_skill(self, name: str) -> Tuple[str, Dict]:
        """Load a skill's source and metadata from disk with caching."""
        if name in self._cache:
            self.hits += 1
            return self._cache[name]

        self.misses += 1
        if name in self._dynamic_skills:
            metadata = self._dynamic_skills[name]
            result = (metadata.get("code", ""), metadata)
            self._cache[name] = result
            return result

        persisted = self._load_from_persist(name)
        if persisted:
            self._cache[name] = persisted
            return persisted

        skill_file = self.storage_dir / f"{name}.py"
        meta_file = self.storage_dir / f"{name}.json"
        code = await self._read_file(skill_file)
        metadata = json.loads(await self._read_file(meta_file))
        if name.startswith("MetaSkill_") and not metadata.get("active"):
            logger.warning(
                "Meta-skill %s requested while inactive; activating automatically.",
                name,
            )
            try:
                await self.activate_meta_skill(name)
            except Exception as err:  # pragma: no cover - best effort logging
                logger.error(
                    "Failed to auto-activate meta-skill %s: %s", name, err
                )
                raise PermissionError(
                    "Meta-skill version could not be activated"
                ) from err
            metadata["active"] = True
        self._cache[name] = (code, metadata)
        self._save_to_persist(name, code, metadata)
        return code, metadata

    async def get_skill(self, name: str) -> Tuple[str, Dict]:
        """Retrieve a skill's source code and metadata using an in-memory cache."""
        return await self._load_skill(name)

    async def activate_meta_skill(self, name: str) -> None:
        """Mark a meta-skill as active and commit the change to Git."""
        meta_file = self.storage_dir / f"{name}.json"
        metadata = json.loads(await self._read_file(meta_file))
        metadata["active"] = True
        meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        subprocess.run(["git", "add", str(meta_file)], cwd=self.repo_path, check=True)
        subprocess.run(
            ["git", "commit", "-m", f"Activate meta-skill {name}"],
            cwd=self.repo_path,
            check=True,
        )
        # Ensure cache is invalidated so future reads get the updated metadata.
        self.invalidate(name)

    def list_skills(self) -> List[str]:
        """List all available skills."""
        disk_skills = {p.stem for p in self.storage_dir.glob("*.py")}
        return sorted(disk_skills | set(self._dynamic_skills))

    def history(self, name: str) -> str:
        """Return the Git commit history for a skill file."""
        skill_file = self.storage_dir / f"{name}.py"
        result = subprocess.run(
            ["git", "log", "--", str(skill_file)],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout

    # ------------------------------------------------------------------
    # Cache utilities
    def cache_stats(self) -> Dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "currsize": self._cache.currsize,
            "maxsize": self._cache.maxsize,
        }

    def invalidate(self, name: str | None = None) -> None:
        if name is None:
            self._cache.clear()
            self._db.execute("DELETE FROM cache")
            self._dynamic_skills.clear()
            self._callable_registry.clear()
            self._callable_lookup.clear()
            self._db.execute("DELETE FROM dynamic_skills")
        else:
            self._cache.pop(name, None)
            self._db.execute("DELETE FROM cache WHERE name = ?", (name,))
            self._dynamic_skills.pop(name, None)
            callable_id = self._callable_lookup.pop(name, None)
            if callable_id is not None:
                self._callable_registry.pop(callable_id, None)
            self._db.execute("DELETE FROM dynamic_skills WHERE name = ?", (name,))
        self._db.commit()


def register_specialist_skill(
    library: SkillLibrary,
    specialist: "SpecialistModule",
    *,
    skill_name: Optional[str] = None,
    default_architecture: Optional[Dict[str, float]] = None,
    task_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Bridge a :class:`SpecialistModule` into the ``SkillLibrary``.

    Parameters
    ----------
    library
        Target :class:`SkillLibrary` instance.
    specialist
        Specialist module created by ``SelfPlayResult.build_specialist``.
    skill_name
        Optional override for the registered skill name. Defaults to the
        specialist name prefixed with ``"specialist_"``.
    default_architecture
        Architecture passed to the specialist solver when invoked. Defaults to
        an empty mapping.
    task_metadata
        Optional metadata provided to the :class:`TaskContext` wrapper.
    """

    from modules.evolution.evolution_engine import TaskContext

    resolved_name = skill_name or f"specialist_{specialist.name}"
    architecture: Dict[str, float] = dict(default_architecture or {})
    capabilities = tuple(sorted(specialist.capabilities))
    metadata: Dict[str, Any] = {
        "type": SPECIALIST_SKILL_TYPE,
        "signature": f"{SPECIALIST_SKILL_TYPE}::{resolved_name}",
        "specialist": {
            "name": specialist.name,
            "capabilities": list(capabilities),
            "priority": specialist.priority,
        },
        "task_metadata": _normalise_metadata(task_metadata) if task_metadata else None,
    }

    def _invoke_specialist() -> Any:
        task = TaskContext(
            name=specialist.name,
            required_capabilities=capabilities,
            metadata=task_metadata,
        )
        return specialist.solver(dict(architecture), task)

    library.register_callable_skill(resolved_name, _invoke_specialist, metadata)
    return resolved_name


try:  # pragma: no cover - optional type checking import
    from modules.evolution.evolution_engine import SpecialistModule
except Exception:  # pragma: no cover - optional dependency
    SpecialistModule = Any  # type: ignore

