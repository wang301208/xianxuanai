from __future__ import annotations

"""Self-knowledge bootstrap + retrieval helpers.

This module turns internal metadata (skills, module docstrings, configs) into
structured "self knowledge" that can be retrieved by an LLM-driven agent.

It supports three storage surfaces (all optional):
 - Knowledge base (semantic search): :class:`backend.knowledge.UnifiedKnowledgeBase`
 - Knowledge graph (structure): :class:`backend.autogpt.autogpt.core.knowledge_graph.GraphStore`
 - Long-term memory (persistence): :class:`backend.memory.long_term.LongTermMemory`
"""

from dataclasses import dataclass, field
import ast
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


logger = logging.getLogger(__name__)


_TOKEN_RE = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]+", re.IGNORECASE)


def _tokens(text: str) -> List[str]:
    if not text:
        return []
    return [tok.lower() for tok in _TOKEN_RE.findall(text)]


def _clip(text: str, *, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_module_name(path: Path, *, repo_root: Path) -> str:
    try:
        rel = path.resolve().relative_to(repo_root.resolve())
    except Exception:
        rel = path
    parts = list(rel.parts)
    if parts and parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    return ".".join(p for p in parts if p and p != "__init__")


def _extract_python_docstring(text: str) -> str:
    if not text:
        return ""
    try:
        module = ast.parse(text)
        doc = ast.get_docstring(module)
        return doc.strip() if doc else ""
    except Exception:
        return ""


def _read_text(path: Path, *, max_chars: int = 60_000) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logger.debug("Failed to read %s: %s", path, exc)
        return ""
    return content if max_chars <= 0 else content[:max_chars]


def _summarize_config(path: Path, *, max_keys: int = 40, max_chars: int = 2400) -> str:
    suffix = path.suffix.lower()
    raw = _read_text(path, max_chars=120_000)
    if not raw:
        return ""

    data: Any = None
    if suffix in {".json"}:
        try:
            data = json.loads(raw)
        except Exception:
            data = None
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception:
            yaml = None  # type: ignore
        if yaml is not None:
            try:
                data = yaml.safe_load(raw)
            except Exception:
                data = None
    elif suffix in {".toml"}:
        try:
            import toml  # type: ignore
        except Exception:
            toml = None  # type: ignore
        if toml is not None:
            try:
                data = toml.loads(raw)
            except Exception:
                data = None

    if isinstance(data, Mapping):
        keys = list(data.keys())
        preview = ", ".join(str(k) for k in keys[:max_keys])
        more = f" (+{len(keys) - max_keys} more)" if len(keys) > max_keys else ""
        return _clip(f"Top-level keys: {preview}{more}", max_chars=max_chars)

    if suffix == ".py":
        doc = _extract_python_docstring(raw)
        if doc:
            return _clip(doc, max_chars=max_chars)

    # Fallback: first non-empty lines.
    lines = [ln.rstrip() for ln in raw.splitlines() if ln.strip()]
    return _clip("\n".join(lines[:12]), max_chars=max_chars)


@dataclass(frozen=True)
class SelfKnowledgeFact:
    subject: str
    predicate: str
    obj: str
    context: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float | None = None
    source: str | None = None
    timestamp: float | None = None


@dataclass(frozen=True)
class ComponentDoc:
    module: str
    path: str
    summary: str
    label: str
    tags: Tuple[str, ...] = ()

    @property
    def node_id(self) -> str:
        return f"component:{self.module}"


def default_component_paths(repo_root: Path) -> List[Path]:
    candidates = [
        repo_root / "BrainSimulationSystem" / "environment" / "tool_bridge.py",
        repo_root / "BrainSimulationSystem" / "environment" / "security_manager.py",
        repo_root / "BrainSimulationSystem" / "environment" / "autonomous_task_loop.py",
        repo_root / "backend" / "introspection" / "__init__.py",
        repo_root / "backend" / "knowledge" / "unified.py",
        repo_root / "backend" / "execution" / "manager.py",
    ]
    return [path for path in candidates if path.exists()]


def default_config_paths(repo_root: Path) -> List[Path]:
    candidates = [
        repo_root / "BrainSimulationSystem" / "README.md",
        repo_root / "BrainSimulationSystem" / "config" / "enhanced_brain_config.yaml",
        repo_root / "BrainSimulationSystem" / "config" / "default_config.py",
        repo_root / "pyproject.toml",
    ]
    return [path for path in candidates if path.exists()]


def collect_component_docs(
    repo_root: Path,
    *,
    paths: Sequence[Path] | None = None,
    max_summary_chars: int = 1200,
) -> List[ComponentDoc]:
    repo_root = Path(repo_root)
    docs: List[ComponentDoc] = []
    for path in (list(paths) if paths is not None else default_component_paths(repo_root)):
        path = Path(path)
        raw = _read_text(path, max_chars=120_000)
        if not raw:
            continue
        module_name = _as_module_name(path, repo_root=repo_root)
        label = path.stem
        summary = _extract_python_docstring(raw) if path.suffix.lower() == ".py" else ""
        if not summary:
            summary = _summarize_config(path, max_chars=max_summary_chars)
        summary = _clip(summary.strip(), max_chars=max_summary_chars)
        if not summary:
            continue
        tags: List[str] = []
        for token in _tokens(module_name):
            if token and token not in tags:
                tags.append(token)
        docs.append(
            ComponentDoc(
                module=module_name,
                path=str(path),
                summary=summary,
                label=label,
                tags=tuple(tags[:8]),
            )
        )
    return docs


def collect_skill_facts(
    registry: Any,
    *,
    max_skills: int = 200,
    source: str = "skill_registry",
) -> List[SelfKnowledgeFact]:
    if registry is None:
        return []
    try:
        specs = list(registry.list_specs())
    except Exception:
        return []
    facts: List[SelfKnowledgeFact] = []
    for spec in specs[: max(0, int(max_skills))]:
        name = str(getattr(spec, "name", "") or "").strip()
        if not name:
            continue
        desc = str(getattr(spec, "description", "") or "").strip()
        tags = list(getattr(spec, "tags", []) or [])
        facts.append(
            SelfKnowledgeFact(
                subject=f"skill:{name}",
                predicate="described_as",
                obj=desc or f"Skill {name}",
                context="",
                metadata={
                    "kind": "skill",
                    "name": name,
                    "provider": getattr(spec, "provider", None),
                    "execution_mode": getattr(spec, "execution_mode", None),
                    "tags": tags,
                    "enabled": bool(getattr(spec, "enabled", True)),
                },
                source=source,
            )
        )
    return facts


def collect_component_facts(
    components: Sequence[ComponentDoc],
    *,
    source: str = "code_docstrings",
) -> List[SelfKnowledgeFact]:
    facts: List[SelfKnowledgeFact] = []
    for doc in components:
        facts.append(
            SelfKnowledgeFact(
                subject=f"module:{doc.module}",
                predicate="described_as",
                obj=doc.label,
                context=doc.summary,
                metadata={
                    "kind": "component",
                    "module": doc.module,
                    "path": doc.path,
                    "tags": list(doc.tags),
                },
                source=source,
            )
        )
    return facts


def collect_config_facts(
    repo_root: Path,
    *,
    paths: Sequence[Path] | None = None,
    source: str = "config_docs",
) -> List[SelfKnowledgeFact]:
    repo_root = Path(repo_root)
    facts: List[SelfKnowledgeFact] = []
    for path in (list(paths) if paths is not None else default_config_paths(repo_root)):
        path = Path(path)
        summary = _summarize_config(path)
        if not summary:
            continue
        module_name = _as_module_name(path, repo_root=repo_root)
        facts.append(
            SelfKnowledgeFact(
                subject=f"doc:{module_name}",
                predicate="summarized_as",
                obj=path.name,
                context=summary,
                metadata={
                    "kind": "doc",
                    "path": str(path),
                    "format": path.suffix.lower().lstrip("."),
                },
                source=source,
            )
        )
    return facts


def build_capability_manifest(
    *,
    skill_facts: Sequence[SelfKnowledgeFact] = (),
    component_docs: Sequence[ComponentDoc] = (),
    max_skills: int = 30,
    max_components: int = 10,
) -> str:
    skills = [f for f in skill_facts if isinstance(f, SelfKnowledgeFact) and f.metadata.get("kind") == "skill"]
    components = list(component_docs)

    lines: List[str] = ["Agent capability manifest:"]
    if skills:
        lines.append("Skills:")
        for fact in skills[: max(0, int(max_skills))]:
            name = str(fact.metadata.get("name") or fact.subject.replace("skill:", "")).strip()
            desc = str(fact.obj or "").strip()
            tags = fact.metadata.get("tags") or []
            tags_text = ""
            if isinstance(tags, list) and tags:
                tags_text = ", tags=" + ", ".join(str(t) for t in tags[:6] if str(t).strip())
            lines.append(f"- {name}: {desc}{tags_text}")
    if components:
        lines.append("Core modules/components:")
        for doc in components[: max(0, int(max_components))]:
            summary = doc.summary.replace("\n", " ")
            lines.append(f"- {doc.module}: {_clip(summary, max_chars=180)}")
    return "\n".join(lines).strip()


def _try_get_global_skill_registry() -> Any:
    try:
        from backend.capability.skill_registry import get_skill_registry  # type: ignore

        return get_skill_registry()
    except Exception:
        return None


def _try_get_global_graph_store() -> Any:
    try:
        from backend.knowledge.registry import get_graph_store_instance

        return get_graph_store_instance()
    except Exception:
        return None


def _upsert_component_graph(
    graph_store: Any,
    components: Sequence[ComponentDoc],
    *,
    owner_node: str = "SELF",
) -> None:
    try:
        from backend.autogpt.autogpt.core.knowledge_graph.ontology import EntityType, RelationType
    except Exception:
        return
    try:
        graph_store.add_node(owner_node, EntityType.AGENT, label=owner_node, description="Agent self node")
    except Exception:
        pass

    for doc in components:
        try:
            graph_store.add_node(
                doc.node_id,
                EntityType.CONCEPT,
                label=doc.label,
                description=doc.summary,
                category="component",
                module=doc.module,
                path=doc.path,
                tags=",".join(doc.tags),
            )
            graph_store.add_edge(
                owner_node,
                doc.node_id,
                RelationType.RELATED_TO,
                relation="has_component",
            )
        except Exception:
            logger.debug("Failed to upsert component node %s", doc.node_id, exc_info=True)


def get_self_structure_graph(
    *,
    graph_store: Any | None = None,
    owner_node: str = "SELF",
) -> Dict[str, Any]:
    """Return a serialisable SELF->component subgraph snapshot."""

    graph_store = graph_store or _try_get_global_graph_store()
    if graph_store is None:
        return {"nodes": [], "edges": [], "note": "graph_store unavailable"}
    try:
        snapshot = graph_store.query(node_id=str(owner_node))
    except Exception:
        try:
            snapshot = graph_store.get_snapshot()
        except Exception:
            return {"nodes": [], "edges": [], "note": "graph_store query failed"}

    nodes = snapshot.get("nodes", []) if isinstance(snapshot, dict) else []
    edges = snapshot.get("edges", []) if isinstance(snapshot, dict) else []
    node_payload: List[Dict[str, Any]] = []
    for node in nodes or []:
        node_payload.append(
            {
                "id": getattr(node, "id", None),
                "type": getattr(getattr(node, "type", None), "value", getattr(node, "type", None)),
                "properties": dict(getattr(node, "properties", {}) or {}),
            }
        )
    edge_payload: List[Dict[str, Any]] = []
    for edge in edges or []:
        edge_payload.append(
            {
                "source": getattr(edge, "source", None),
                "target": getattr(edge, "target", None),
                "type": getattr(getattr(edge, "type", None), "value", getattr(edge, "type", None)),
                "properties": dict(getattr(edge, "properties", {}) or {}),
            }
        )
    return {"nodes": node_payload, "edges": edge_payload, "owner": str(owner_node)}


def bootstrap_self_knowledge(
    *,
    repo_root: str | Path | None = None,
    knowledge_base: Any | None = None,
    registry: Any | None = None,
    graph_store: Any | None = None,
    memory: Any | None = None,
    component_paths: Sequence[str | Path] | None = None,
    config_paths: Sequence[str | Path] | None = None,
    max_skill_facts: int = 200,
    owner_node: str = "SELF",
) -> Dict[str, Any]:
    """Populate knowledge surfaces with prompt-friendly self knowledge.

    This function is safe to call multiple times; callers may still want to
    gate it externally for performance reasons.
    """

    if repo_root is None:
        repo_root = Path.cwd()
    repo_root = Path(repo_root)

    registry = registry or _try_get_global_skill_registry()
    graph_store = graph_store or _try_get_global_graph_store()

    components = collect_component_docs(
        repo_root,
        paths=[Path(p) for p in component_paths] if component_paths is not None else None,
    )
    skill_facts = collect_skill_facts(registry, max_skills=max_skill_facts)
    component_facts = collect_component_facts(components)
    config_facts = collect_config_facts(
        repo_root,
        paths=[Path(p) for p in config_paths] if config_paths is not None else None,
    )

    manifest = build_capability_manifest(skill_facts=skill_facts, component_docs=components)
    manifest_fact = SelfKnowledgeFact(
        subject=owner_node,
        predicate="capability_manifest",
        obj="internal",
        context=manifest,
        metadata={"kind": "manifest", "format": "text"},
        source="self_knowledge",
    )

    facts: List[SelfKnowledgeFact] = [*skill_facts, *component_facts, *config_facts, manifest_fact]

    ingested = 0
    if knowledge_base is not None:
        if getattr(knowledge_base, "_self_knowledge_bootstrapped", False):
            ingested = 0
        else:
            try:
                result = knowledge_base.ingest_facts(facts, embed=True)
                ingested = int(result.get("imported", len(facts))) if isinstance(result, dict) else len(facts)
                setattr(knowledge_base, "_self_knowledge_bootstrapped", True)
            except Exception:
                logger.debug("Self-knowledge ingestion into knowledge_base failed.", exc_info=True)
                ingested = 0

    if graph_store is not None:
        _upsert_component_graph(graph_store, components, owner_node=owner_node)

    stored_memory_id = None
    if memory is not None:
        store = getattr(memory, "add", None) or getattr(memory, "store", None)
        if callable(store):
            try:
                stored_memory_id = store(  # type: ignore[misc]
                    "self_knowledge",
                    manifest,
                    tags=("capability_manifest",),
                    metadata={"source": "self_knowledge"},
                )
            except TypeError:
                try:
                    stored_memory_id = store(manifest, metadata={"category": "self_knowledge", "tags": ["capability_manifest"]})  # type: ignore[misc]
                except Exception:
                    stored_memory_id = None
            except Exception:
                stored_memory_id = None

    return {
        "facts_built": len(facts),
        "facts_ingested": ingested,
        "components_indexed": len(components),
        "skills_indexed": len(skill_facts),
        "configs_indexed": len(config_facts),
        "memory_id": stored_memory_id,
    }


def query_self_structure(
    module_name: str,
    *,
    knowledge_base: Any | None = None,
    graph_store: Any | None = None,
    top_k: int = 5,
    as_text: bool = False,
) -> Dict[str, Any] | str:
    """Query self-structure knowledge for a module/component name."""

    query = (module_name or "").strip()
    if not query:
        payload = {"query": "", "results": [], "note": "empty query"}
        return _format_query(payload) if as_text else payload

    graph_store = graph_store or _try_get_global_graph_store()

    results: List[Dict[str, Any]] = []

    if knowledge_base is not None:
        try:
            kb = knowledge_base.query(query, semantic=True, top_k=max(1, int(top_k)))
        except Exception:
            kb = {}
        facts = kb.get("facts") if isinstance(kb, dict) else None
        if isinstance(facts, list):
            for hit in facts:
                if not isinstance(hit, dict):
                    continue
                meta = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else hit
                if meta.get("kind") in {"component", "doc", "manifest"}:
                    results.append(hit)

    if not results and graph_store is not None:
        try:
            snapshot = graph_store.get_snapshot()
        except Exception:
            snapshot = {}
        nodes = snapshot.get("nodes") if isinstance(snapshot, dict) else None
        if isinstance(nodes, list):
            terms = set(_tokens(query))
            scored: List[Tuple[int, Dict[str, Any]]] = []
            for node in nodes:
                node_id = str(getattr(node, "id", "") or "")
                props = getattr(node, "properties", {}) or {}
                category = props.get("category")
                if category != "component":
                    continue
                text = " ".join(
                    [
                        node_id,
                        str(props.get("label") or ""),
                        str(props.get("module") or ""),
                        str(props.get("description") or ""),
                    ]
                ).lower()
                if not text:
                    continue
                score = len(terms & set(_tokens(text)))
                if score:
                    scored.append(
                        (
                            score,
                            {
                                "id": node_id,
                                "text": str(props.get("description") or ""),
                                "metadata": {
                                    "module": props.get("module"),
                                    "path": props.get("path"),
                                    "kind": "component",
                                },
                            },
                        )
                    )
            scored.sort(key=lambda item: item[0], reverse=True)
            results = [item[1] for item in scored[: max(1, int(top_k))]]

    payload = {"query": query, "results": results, "returned": len(results)}
    return _format_query(payload) if as_text else payload


def _format_query(payload: Mapping[str, Any]) -> str:
    query = str(payload.get("query") or "")
    results = payload.get("results") or []
    lines = [f"Self structure query: {query}"]
    if not isinstance(results, list) or not results:
        return "\n".join(lines + ["(no matches)"]).strip()
    for idx, hit in enumerate(results[:10], start=1):
        if not isinstance(hit, Mapping):
            continue
        meta = hit.get("metadata") if isinstance(hit.get("metadata"), Mapping) else hit
        kind = str(meta.get("kind") or "")
        subj = str(hit.get("id") or hit.get("subject") or "").strip()
        text = str(hit.get("text") or "").strip()
        if not text and isinstance(meta, Mapping):
            text = str(meta.get("context") or "")
        summary = _clip(text.replace("\n", " "), max_chars=220)
        label = f"{kind} " if kind else ""
        lines.append(f"{idx}. {label}{subj}: {summary}")
    return "\n".join(lines).strip()


__all__ = [
    "ComponentDoc",
    "SelfKnowledgeFact",
    "bootstrap_self_knowledge",
    "collect_component_docs",
    "collect_component_facts",
    "collect_config_facts",
    "collect_skill_facts",
    "default_component_paths",
    "default_config_paths",
    "get_self_structure_graph",
    "query_self_structure",
]
