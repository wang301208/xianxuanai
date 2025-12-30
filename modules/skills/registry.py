from __future__ import annotations

"""Dynamic skill registry with knowledge-graph integration."""

import importlib
import inspect
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Tuple

from backend.autogpt.autogpt.core.knowledge_graph.graph_store import GraphStore
from backend.autogpt.autogpt.core.knowledge_graph.ontology import EntityType, RelationType
from backend.knowledge.registry import get_graph_store_instance

logger = logging.getLogger(__name__)


class SkillHandler(Protocol):
    """Protocol for callable skill handlers."""

    def __call__(self, payload: Dict[str, Any], **kwargs: Any) -> Any:
        ...


@dataclass
class SkillSpec:
    """Description of a skill available to the agent."""

    name: str
    description: str
    execution_mode: str = "local"
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    cost: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    provider: str = "builtin"
    version: str = "0.1.0"
    enabled: bool = True
    entrypoint: Optional[str] = None
    source: Optional[str] = None

    def __post_init__(self) -> None:
        mode = (self.execution_mode or "local").strip().lower()
        self.execution_mode = mode if mode in {"local", "rpc"} else "local"

    def to_properties(self) -> Dict[str, Any]:
        data = asdict(self)
        # Knowledge graph nodes should avoid nested structures when possible.
        data["tags"] = ",".join(self.tags)
        data["input_schema_json"] = json.dumps(self.input_schema, ensure_ascii=False)
        data["output_schema_json"] = json.dumps(self.output_schema, ensure_ascii=False)
        data["cost_json"] = json.dumps(self.cost, ensure_ascii=False)
        data["execution_mode"] = self.execution_mode
        return data


@dataclass
class SkillEntry:
    """Registered skill containing its spec and an optional handler."""

    spec: SkillSpec
    handler: Optional[SkillHandler] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SkillRegistrationError(RuntimeError):
    """Raised when registering a skill fails."""


class SkillRegistry:
    """Manage hot-pluggable skills with optional knowledge graph syncing."""

    def __init__(
        self,
        *,
        graph_store: Optional[GraphStore] = None,
        owner_node: str = "SELF",
        auto_graph_updates: bool = True,
    ) -> None:
        self._skills: Dict[str, SkillEntry] = {}
        self._graph = graph_store if graph_store is not None else get_graph_store_instance()
        self._owner_node = owner_node
        self._auto_graph_updates = auto_graph_updates

    # ------------------------------------------------------------------
    def register(
        self,
        spec: SkillSpec,
        handler: Optional[SkillHandler] = None,
        *,
        replace: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register ``spec`` optionally binding ``handler`` for invocation."""

        previous_entry: SkillEntry | None = None
        if spec.name in self._skills and not replace:
            raise SkillRegistrationError(f"Skill '{spec.name}' already registered.")
        if replace and spec.name in self._skills:
            previous_entry = self._skills[spec.name]

        entry = SkillEntry(spec=spec, handler=handler, metadata=dict(metadata or {}))
        entry.metadata.setdefault("name", spec.name)
        entry.metadata.setdefault("execution_mode", spec.execution_mode)
        self._skills[spec.name] = entry
        logger.info("Registered skill '%s' (provider=%s, version=%s)", spec.name, spec.provider, spec.version)

        if self._auto_graph_updates:
            self._upsert_graph(spec)
        if previous_entry is not None:
            self._unregister_from_environment(previous_entry)
        self._register_with_environment(entry)

    # ------------------------------------------------------------------
    def unregister(self, name: str) -> None:
        """Remove ``name`` from the registry and detach from the knowledge graph."""

        entry = self._skills.pop(name, None)
        if entry is None:
            return
        logger.info("Unregistered skill '%s'", name)
        if self._auto_graph_updates:
            self._delete_graph(entry.spec)
        self._unregister_from_environment(entry)

    # ------------------------------------------------------------------
    def get(self, name: str) -> SkillEntry:
        if name not in self._skills:
            raise KeyError(f"Skill '{name}' not found.")
        return self._skills[name]

    # ------------------------------------------------------------------
    def list_specs(self) -> List[SkillSpec]:
        return [entry.spec for entry in self._skills.values()]

    # ------------------------------------------------------------------
    async def invoke(self, name: str, payload: Dict[str, Any], **kwargs: Any) -> Any:
        entry = self.get(name)
        if entry.handler is None:
            raise SkillRegistrationError(f"Skill '{name}' is not invokable (no handler).")

        result = entry.handler(payload, **kwargs)
        if inspect.isawaitable(result):
            return await result  # type: ignore[return-value]
        return result

    # ------------------------------------------------------------------
    def refresh_from_directory(
        self,
        root: Path | str,
        *,
        pattern: str = "*.skill.json",
        prune_missing: bool = False,
        auto_entrypoint: bool = True,
    ) -> None:
        """Scan ``root`` for manifest files and register/update skills."""

        root_path = Path(root)
        manifests = list(root_path.rglob(pattern))
        discovered: set[str] = set()
        for manifest in manifests:
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
            except Exception as err:
                logger.warning("Failed to parse skill manifest %s: %s", manifest, err)
                continue
            try:
                spec = self._spec_from_dict(data, source=str(manifest))
            except Exception as err:
                logger.warning("Invalid skill manifest %s: %s", manifest, err)
                continue

            handler = None
            if auto_entrypoint and spec.entrypoint:
                handler = self._resolve_entrypoint(spec.entrypoint)
            manifest_metadata: Dict[str, Any] = {}
            extra_metadata = data.get("metadata")
            if isinstance(extra_metadata, Mapping):
                manifest_metadata.update(dict(extra_metadata))
            rpc_config = data.get("rpc_config")
            if isinstance(rpc_config, Mapping):
                manifest_metadata.setdefault("rpc_config", dict(rpc_config))
            manifest_metadata.setdefault("manifest", str(manifest))
            manifest_metadata.setdefault("execution_mode", spec.execution_mode)
            manifest_metadata.setdefault("name", spec.name)
            self.register(spec, handler=handler, replace=True, metadata=manifest_metadata)
            discovered.add(spec.name)

        if prune_missing:
            for name in list(self._skills.keys()):
                entry = self._skills[name]
                if entry.spec.source and entry.spec.source.startswith(str(root_path)):
                    if name not in discovered:
                        self.unregister(name)

    # ------------------------------------------------------------------
    async def sync_from_skill_library(
        self,
        library,
        names: Optional[Iterable[str]] = None,
        *,
        provider: str = "skill_library",
    ) -> None:
        """Register skills based on metadata retrieved from ``library``."""

        try:
            available = names or library.list_skills()
        except Exception as err:
            logger.error("Failed to query skill library: %s", err)
            return

        for name in available:
            try:
                code, metadata = await library.get_skill(name)
            except Exception as err:
                logger.warning("Unable to load skill '%s' from library: %s", name, err)
                continue
            spec = SkillSpec(
                name=name,
                description=metadata.get("description", f"Skill {name}"),
                execution_mode=str(metadata.get("execution_mode", metadata.get("mode", "local"))).lower(),
                input_schema=metadata.get("input_schema", {}),
                output_schema=metadata.get("output_schema", {}),
                cost=metadata.get("cost", {}),
                tags=metadata.get("tags", []),
                provider=metadata.get("provider", provider),
                version=str(metadata.get("version", metadata.get("commit", "0.1.0"))),
                enabled=metadata.get("active", True),
                entrypoint=metadata.get("entrypoint"),
                source="skill_library",
            )
            handler = None
            if spec.entrypoint:
                handler = self._resolve_entrypoint(spec.entrypoint)
            entry_metadata: Dict[str, Any] = {"code": code}
            rpc_config = metadata.get("rpc_config")
            if isinstance(rpc_config, Mapping):
                entry_metadata["rpc_config"] = dict(rpc_config)
            entry_metadata.setdefault("execution_mode", spec.execution_mode)
            self.register(spec, handler=handler, replace=True, metadata=entry_metadata)

    # ------------------------------------------------------------------
    def _spec_from_dict(self, data: Mapping[str, Any], *, source: Optional[str]) -> SkillSpec:
        required = {"name", "description"}
        missing = required - set(data.keys())
        if missing:
            raise SkillRegistrationError(f"Missing fields: {missing}")
        tags = data.get("tags") or []
        if isinstance(tags, str):
            tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
        execution_mode = str(data.get("execution_mode", "local") or "local").strip().lower()
        spec = SkillSpec(
            name=str(data["name"]),
            description=str(data["description"]),
            execution_mode=execution_mode,
            input_schema=dict(data.get("input_schema") or {}),
            output_schema=dict(data.get("output_schema") or {}),
            cost=dict(data.get("cost") or {}),
            tags=list(tags),
            provider=str(data.get("provider", "plugin")),
            version=str(data.get("version", "0.1.0")),
            enabled=bool(data.get("enabled", True)),
            entrypoint=data.get("entrypoint"),
            source=source,
        )
        return spec

    # ------------------------------------------------------------------
    def _resolve_entrypoint(self, entrypoint: str) -> Optional[SkillHandler]:
        try:
            module_name, func_name = entrypoint.split(":", 1)
        except ValueError as err:
            raise SkillRegistrationError(f"Invalid entrypoint '{entrypoint}'") from err
        try:
            module = importlib.import_module(module_name)
            handler = getattr(module, func_name)
        except Exception as err:
            raise SkillRegistrationError(f"Failed to resolve entrypoint '{entrypoint}': {err}") from err

        if not callable(handler):
            raise SkillRegistrationError(f"Entrypoint '{entrypoint}' is not callable")
        return handler  # type: ignore[return-value]

    # ------------------------------------------------------------------
    def _upsert_graph(self, spec: SkillSpec) -> None:
        node_id = self._skill_node(spec.name)
        try:
            properties = spec.to_properties()
            properties["enabled"] = spec.enabled
            self._graph.add_node(node_id, EntityType.SKILL, **properties)
            self._graph.add_edge(self._owner_node, node_id, RelationType.PERFORMS)
        except Exception:  # pragma: no cover - best effort logging
            logger.debug("Failed to upsert skill node '%s' into knowledge graph.", node_id, exc_info=True)

    # ------------------------------------------------------------------
    def _delete_graph(self, spec: SkillSpec) -> None:
        node_id = self._skill_node(spec.name)
        try:
            self._graph.remove_node(node_id)
        except Exception:
            logger.debug("Failed to remove skill node '%s' from knowledge graph.", node_id, exc_info=True)

    # ------------------------------------------------------------------
    def _skill_node(self, name: str) -> str:
        safe = name.lower().replace(" ", "_")
        return f"skill:{safe}"

    def _register_with_environment(self, entry: SkillEntry) -> None:
        metadata = entry.metadata or {}
        execution_mode = str(metadata.get("execution_mode", entry.spec.execution_mode)).lower()
        if execution_mode != "rpc":
            return
        try:
            from modules.environment import register_skill_service

            register_skill_service(entry.spec, metadata)
        except Exception:
            logger.debug(
                "Skill environment registration failed for %s", entry.spec.name, exc_info=True
            )

    def _unregister_from_environment(self, entry: SkillEntry) -> None:
        metadata = entry.metadata or {}
        execution_mode = str(metadata.get("execution_mode", entry.spec.execution_mode)).lower()
        if execution_mode != "rpc":
            return
        service_id = f"skill:{entry.spec.name}"
        try:
            from modules.environment import unregister_service

            unregister_service(service_id)
        except Exception:
            logger.debug(
                "Skill environment unregister failed for %s", entry.spec.name, exc_info=True
            )


__all__ = ["SkillSpec", "SkillEntry", "SkillRegistry", "SkillRegistrationError"]
