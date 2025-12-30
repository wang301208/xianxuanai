from __future__ import annotations

"""Opt-in module acquisition workflow (discover -> suggest -> (optional) install).

This module implements a conservative "module search / generation / import" loop:
1) Search internal capability/skill registries for a match and auto-load it.
2) If missing, optionally run lightweight research (docs + web) and publish a
   suggestion event for human/agent approval.
3) If explicitly requested and allowlisted, optionally install a PyPI package
   into an isolated target directory and register an import-based capability.

The default behavior is *suggest-only*: no network installs or code changes
unless enabled via environment variables and (for pip installs) an allowlist.
"""

import difflib
import importlib
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

try:  # optional dependency
    from events import EventBus  # type: ignore
except Exception:  # pragma: no cover
    EventBus = None  # type: ignore

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


_TOKEN_RE = re.compile(r"[a-zA-Z0-9_][a-zA-Z0-9_\-]+")
_PIP_INSTALL_RE = re.compile(r"\bpip\s+install\s+([a-zA-Z0-9_.\-]+)")
_PYPI_RE = re.compile(r"pypi\.org/project/([a-zA-Z0-9_.\-]+)")
_GITHUB_RE = re.compile(r"github\.com/([a-zA-Z0-9_.\-]+/[a-zA-Z0-9_.\-]+)")


def _tokenize(text: Any) -> List[str]:
    blob = str(text or "").lower()
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(blob)]


def _unique(seq: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in seq:
        key = str(item or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _similarity(a: str, b: str) -> float:
    return float(difflib.SequenceMatcher(a=str(a or "").lower(), b=str(b or "").lower()).ratio())


def _extract_pip_candidates(text: str) -> List[str]:
    hits: List[str] = []
    for match in _PIP_INSTALL_RE.finditer(text or ""):
        hits.append(match.group(1))
    for match in _PYPI_RE.finditer(text or ""):
        hits.append(match.group(1))
    return _unique(hits)


def _extract_github_candidates(text: str) -> List[str]:
    return _unique(match.group(1) for match in _GITHUB_RE.finditer(text or ""))


def _default_package_hints(query: str) -> List[str]:
    tokens = set(_tokenize(query))
    hints: List[str] = []
    if {"vector", "embedding", "vectordb", "vectorstore"} & tokens:
        hints.extend(["chromadb", "qdrant-client", "weaviate-client", "faiss-cpu"])
    if {"translate", "translation"} & tokens:
        hints.extend(["transformers", "sentencepiece"])
    if {"speech", "asr", "whisper"} & tokens:
        hints.extend(["openai-whisper"])
    if {"ocr"} & tokens:
        hints.extend(["pytesseract"])
    if {"pdf"} & tokens:
        hints.extend(["pypdf", "pdfplumber"])
    if {"schedule", "scheduler", "cron"} & tokens:
        hints.extend(["apscheduler"])
    if {"optuna", "tune"} & tokens:
        hints.extend(["optuna", "ray[tune]"])
    return _unique(hints)


@dataclass(frozen=True)
class ModuleCandidate:
    kind: str  # "capability" | "skill"
    name: str
    score: float
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "kind": self.kind,
            "name": self.name,
            "score": float(self.score),
        }
        if self.description:
            payload["description"] = self.description
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class AcquisitionSuggestion:
    query: str
    created_at: float
    internal_matches: List[ModuleCandidate] = field(default_factory=list)
    doc_hits: List[Dict[str, Any]] = field(default_factory=list)
    web_hits: List[Dict[str, Any]] = field(default_factory=list)
    proposed_pip_packages: List[str] = field(default_factory=list)
    proposed_github_repos: List[str] = field(default_factory=list)
    notes: str = ""
    require_human_approval: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "created_at": float(self.created_at),
            "internal_matches": [m.to_dict() for m in self.internal_matches],
            "doc_hits": list(self.doc_hits),
            "web_hits": list(self.web_hits),
            "proposed_pip_packages": list(self.proposed_pip_packages),
            "proposed_github_repos": list(self.proposed_github_repos),
            "notes": self.notes,
            "require_human_approval": bool(self.require_human_approval),
        }


class ModuleAcquisitionManager:
    """Coordinate best-effort discovery and optional installation of new modules."""

    def __init__(
        self,
        *,
        module_manager: Any | None = None,
        task_manager: Any | None = None,
        event_bus: EventBus | None = None,
        enabled: bool | None = None,
        internal_match_threshold: float | None = None,
        cooldown_secs: float | None = None,
        max_internal_matches: int | None = None,
        search_docs: bool | None = None,
        search_web: bool | None = None,
        pip_target_dir: str | Path | None = None,
        allow_pip_install: bool | None = None,
        pip_allowlist: Sequence[str] | None = None,
        require_human_approval: bool | None = None,
        docker_validate: bool | None = None,
        docker_image: str | None = None,
        docker_timeout_secs: float | None = None,
    ) -> None:
        self.enabled = _env_bool("MODULE_ACQUISITION_ENABLED", False) if enabled is None else bool(enabled)
        self._internal_threshold = _env_float("MODULE_ACQUISITION_INTERNAL_THRESHOLD", 0.92) if internal_match_threshold is None else float(internal_match_threshold)
        self._cooldown_secs = _env_float("MODULE_ACQUISITION_COOLDOWN_SECS", 600.0) if cooldown_secs is None else float(cooldown_secs)
        self._max_internal_matches = _env_int("MODULE_ACQUISITION_MAX_INTERNAL", 5) if max_internal_matches is None else int(max_internal_matches)
        self._search_docs = _env_bool("MODULE_ACQUISITION_DOC_SEARCH", True) if search_docs is None else bool(search_docs)
        self._search_web = _env_bool("MODULE_ACQUISITION_WEB_SEARCH", False) if search_web is None else bool(search_web)
        self._allow_pip_install = _env_bool("MODULE_ACQUISITION_ALLOW_PIP_INSTALL", False) if allow_pip_install is None else bool(allow_pip_install)
        self._require_human_approval = _env_bool("MODULE_ACQUISITION_REQUIRE_APPROVAL", True) if require_human_approval is None else bool(require_human_approval)
        self._docker_validate = _env_bool("MODULE_ACQUISITION_DOCKER_VALIDATE", False) if docker_validate is None else bool(docker_validate)
        self._docker_image = (
            str(os.getenv("MODULE_ACQUISITION_DOCKER_IMAGE", "") if docker_image is None else docker_image).strip()
            or "python:3.11-slim"
        )
        self._docker_timeout_secs = (
            _env_float("MODULE_ACQUISITION_DOCKER_TIMEOUT_SECS", 180.0)
            if docker_timeout_secs is None
            else float(docker_timeout_secs)
        )

        allowlist_env = os.getenv("MODULE_ACQUISITION_PIP_ALLOWLIST", "")
        if pip_allowlist is None:
            allowlist = [p.strip() for p in allowlist_env.split(",") if p.strip()]
            self._pip_allowlist = tuple(allowlist)
        else:
            self._pip_allowlist = tuple(str(p).strip() for p in pip_allowlist if str(p).strip())

        target = pip_target_dir or os.getenv("MODULE_ACQUISITION_PIP_TARGET_DIR") or "data/external_packages"
        self._pip_target_dir = Path(target)

        self._module_manager = module_manager
        self._task_manager = task_manager
        self._bus = event_bus

        self._recent_requests: Dict[str, float] = {}
        self._suggestions: deque[AcquisitionSuggestion] = deque(maxlen=200)

        if self._bus is not None:
            try:
                self._bus.subscribe("module.acquisition.request", self._on_request)  # type: ignore[arg-type]
                self._bus.subscribe("module.acquisition.install", self._on_install)  # type: ignore[arg-type]
            except Exception:  # pragma: no cover - optional wiring
                logger.debug("Module acquisition event subscriptions failed.", exc_info=True)

    def _capability_import_prefixes(self) -> List[str]:
        """Return preferred import prefixes for capability packages.

        The codebase sometimes runs with `backend/` injected into `sys.path`,
        exposing packages as `capability.*` / `execution.*`. In other contexts
        it imports via `backend.capability.*`. To avoid split registries we try
        to align with the active RuntimeModuleManager namespace when available.
        """

        prefixes: List[str] = []
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

    # ------------------------------------------------------------------ public API
    def suggestions(self, *, limit: int = 20) -> List[AcquisitionSuggestion]:
        limit = max(0, int(limit))
        items = list(self._suggestions)
        return items[-limit:] if limit else items

    def missing_capability_modules(self, tasks: Sequence[Any]) -> List[str]:
        candidates = [str(t).strip() for t in (tasks or []) if isinstance(t, str) and str(t).strip()]
        registry = self._import_capability_module("module_registry")
        if registry is None:
            return candidates
        available = getattr(registry, "available_modules", None)
        if not callable(available):
            return candidates
        known = set(available())
        return [name for name in candidates if name not in known]

    def request_for_tasks(
        self,
        tasks: Sequence[Any],
        *,
        goal: str | None = None,
        reason: str = "capability_gap",
        context: Optional[Mapping[str, Any]] = None,
    ) -> List[str]:
        if not self.enabled:
            return []
        missing = self.missing_capability_modules(tasks)
        for name in missing:
            self.request(
                name,
                context={
                    **dict(context or {}),
                    "goal": goal,
                    "tasks": list(tasks or []),
                    "reason": reason,
                },
            )
        return missing

    def request(self, query: str, *, context: Optional[Mapping[str, Any]] = None) -> None:
        if not self.enabled:
            return
        query = str(query or "").strip()
        if not query:
            return

        now = time.time()
        last = self._recent_requests.get(query)
        if last is not None and (now - last) < self._cooldown_secs:
            return
        self._recent_requests[query] = now

        if self._task_manager is not None:
            try:
                self._task_manager.submit(
                    self._handle_query,
                    query,
                    dict(context or {}),
                    priority=10,
                    category="module_acquisition",
                    name=f"module_acquisition:{query}",
                    metadata={"query": query},
                )
                return
            except Exception:
                logger.debug("Failed to schedule module acquisition task.", exc_info=True)

        # Fallback: run inline.
        self._handle_query(query, dict(context or {}))

    # ------------------------------------------------------------------ install API (explicit)
    def install_pip_and_register(
        self,
        *,
        package: str,
        module_name: str,
        entrypoint: str,
        replace: bool = False,
    ) -> Dict[str, Any]:
        """Install a pip package (allowlisted) into the target dir and register a capability."""

        package = str(package or "").strip()
        module_name = str(module_name or "").strip()
        entrypoint = str(entrypoint or "").strip()
        if not package or not module_name or ":" not in entrypoint:
            return {"status": "invalid_request"}
        if not self._allow_pip_install:
            return {"status": "disabled", "reason": "pip_install_disabled"}
        if self._pip_allowlist and package not in self._pip_allowlist:
            return {"status": "blocked", "reason": "package_not_allowlisted", "package": package}
        if self._require_human_approval:
            return {"status": "blocked", "reason": "requires_human_approval"}
        if self._docker_validate:
            docker_validation = self._docker_validate_entrypoint(package, entrypoint)
            if not docker_validation.get("ok"):
                return {"status": "blocked", "reason": "docker_validation_failed", **docker_validation}

        install = self._pip_install(package)
        if install.get("status") != "installed":
            return install

        validation = self._validate_entrypoint(entrypoint)
        if not validation.get("ok"):
            return {"status": "failed", "reason": "entrypoint_import_failed", **validation}

        registered = self._register_entrypoint_module(module_name, entrypoint, replace=replace)
        if not registered:
            return {"status": "failed", "reason": "register_failed"}

        loaded = None
        if self._module_manager is not None and hasattr(self._module_manager, "load"):
            try:
                loaded = self._module_manager.load(module_name)
            except Exception:
                loaded = None
        return {"status": "ok", "package": package, "module": module_name, "entrypoint": entrypoint, "loaded": bool(loaded)}

    # ------------------------------------------------------------------ event handlers
    async def _on_request(self, event: Dict[str, Any]) -> None:
        query = event.get("query") or event.get("module") or event.get("capability")
        if not isinstance(query, str):
            return
        self.request(query, context=event)

    async def _on_install(self, event: Dict[str, Any]) -> None:
        if not isinstance(event, dict):
            return
        package = event.get("package")
        module_name = event.get("module") or event.get("name")
        entrypoint = event.get("entrypoint")
        if not (isinstance(package, str) and isinstance(module_name, str) and isinstance(entrypoint, str)):
            return
        result = self.install_pip_and_register(
            package=package,
            module_name=module_name,
            entrypoint=entrypoint,
            replace=bool(event.get("replace", False)),
        )
        if self._bus is not None:
            try:
                self._bus.publish("module.acquisition.install_result", {"result": result})
            except Exception:
                pass

    # ------------------------------------------------------------------ core implementation
    def _handle_query(self, query: str, context: Dict[str, Any]) -> None:
        query = str(query or "").strip()
        if not query:
            return
        loaded = self._maybe_load_internal(query)
        if loaded is not None:
            return
        suggestion = self._build_suggestion(query, context=context)
        self._suggestions.append(suggestion)
        if self._bus is not None:
            try:
                self._bus.publish("module.acquisition.suggested", {"suggestion": suggestion.to_dict()})
            except Exception:
                pass

    def _maybe_load_internal(self, query: str) -> Any | None:
        if self._module_manager is None or not hasattr(self._module_manager, "load"):
            return None
        query = str(query or "").strip()
        if not query:
            return None

        registry = self._import_capability_module("module_registry")
        if registry is None:
            return None
        available_fn = getattr(registry, "available_modules", None)
        if not callable(available_fn):
            return None
        available = list(available_fn())
        if query in available:
            try:
                module = self._module_manager.load(query)
            except Exception:
                module = None
            if module is not None and self._bus is not None:
                try:
                    self._bus.publish("module.acquisition.loaded", {"module": query, "source": "exact_match"})
                except Exception:
                    pass
            return module

        matches = self.find_internal(query, limit=self._max_internal_matches)
        best = next((m for m in matches if m.kind == "capability"), None)
        if best is None:
            return None
        if float(best.score) < self._internal_threshold:
            return None
        try:
            module = self._module_manager.load(best.name)
        except Exception:
            return None
        if module is not None and self._bus is not None:
            try:
                self._bus.publish(
                    "module.acquisition.loaded",
                    {"module": best.name, "source": "fuzzy_match", "query": query, "score": float(best.score)},
                )
            except Exception:
                pass
        return module

    def find_internal(self, query: str, *, limit: int | None = None) -> List[ModuleCandidate]:
        query = str(query or "").strip()
        if not query:
            return []
        limit = self._max_internal_matches if limit is None else max(1, int(limit))

        candidates: List[ModuleCandidate] = []

        # Capability modules
        try:
            registry = self._import_capability_module("module_registry")
            available_modules = getattr(registry, "available_modules", None) if registry is not None else None
            is_enabled = getattr(registry, "is_module_enabled", None) if registry is not None else None
            if not callable(available_modules):
                raise RuntimeError("capability.module_registry unavailable")

            try:
                names = list(available_modules(include_disabled=True))
            except TypeError:
                names = list(available_modules())
            for name in names:
                score = 0.0
                if name == query:
                    score = 1.0
                elif query in name or name in query:
                    score = max(score, 0.85)
                score = max(score, _similarity(query, name))
                meta: Dict[str, Any] = {}
                if callable(is_enabled):
                    try:
                        meta["enabled"] = bool(is_enabled(name))
                    except Exception:
                        pass
                candidates.append(ModuleCandidate(kind="capability", name=name, score=score, metadata=meta))
        except Exception:
            pass

        # Skills (from global registry, if present)
        try:
            skill_registry = self._import_capability_module("skill_registry")
            get_registry = getattr(skill_registry, "get_skill_registry", None) if skill_registry is not None else None
            if not callable(get_registry):
                raise RuntimeError("capability.skill_registry unavailable")

            registry_obj = get_registry()
            for spec in registry_obj.list_specs():
                name = getattr(spec, "name", None)
                desc = getattr(spec, "description", "") or ""
                if not isinstance(name, str) or not name.strip():
                    continue
                score = max(_similarity(query, name), _similarity(query, desc))
                candidates.append(
                    ModuleCandidate(
                        kind="skill",
                        name=name,
                        score=score,
                        description=str(desc)[:240],
                        metadata={"enabled": bool(getattr(spec, "enabled", True))},
                    )
                )
        except Exception:
            pass

        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[:limit]

    def _build_suggestion(self, query: str, *, context: Mapping[str, Any]) -> AcquisitionSuggestion:
        internal = self.find_internal(query, limit=self._max_internal_matches)
        doc_hits: List[Dict[str, Any]] = []
        web_hits: List[Dict[str, Any]] = []
        pip_candidates: List[str] = []
        github_candidates: List[str] = []
        notes: List[str] = []

        if self._search_docs or self._search_web:
            try:
                from modules.knowledge.research_tool import ResearchTool

                tool = ResearchTool()
                if self._search_docs:
                    docs = tool.query_docs(query, max_results=3)
                    doc_hits = [hit.to_dict() for hit in docs]
                if self._search_web:
                    hits = tool.search_web(query, max_results=3)
                    web_hits = [hit.to_dict() for hit in hits]
            except Exception:
                logger.debug("Research tool failed during module acquisition.", exc_info=True)

        # Heuristics from evidence
        for hit in web_hits:
            snippet = str(hit.get("snippet") or "")
            pip_candidates.extend(_extract_pip_candidates(snippet))
            github_candidates.extend(_extract_github_candidates(snippet))
            url = str(hit.get("url") or "")
            pip_candidates.extend(_extract_pip_candidates(url))
            github_candidates.extend(_extract_github_candidates(url))

        pip_candidates.extend(_default_package_hints(query))
        pip_candidates = _unique(pip_candidates)
        github_candidates = _unique(github_candidates)

        if self._allow_pip_install and not self._pip_allowlist:
            notes.append("pip_install_enabled_but_no_allowlist")
        if self._require_human_approval:
            notes.append("requires_human_approval")

        # Optional: ask LLM-backed ProblemAnalyzer to refine candidates (best-effort).
        try:
            from modules.knowledge.problem_analyzer import ProblemAnalyzer

            analyzer = ProblemAnalyzer.from_env()
            if analyzer is not None:
                subqs = analyzer.analyze_problem(f"Find a Python library to enable: {query}", context=dict(context), max_subquestions=3)
                if subqs:
                    notes.append(f"sub_questions={len(subqs)}")
        except Exception:
            pass

        return AcquisitionSuggestion(
            query=query,
            created_at=time.time(),
            internal_matches=internal,
            doc_hits=doc_hits,
            web_hits=web_hits,
            proposed_pip_packages=pip_candidates,
            proposed_github_repos=github_candidates,
            notes=";".join(notes),
            require_human_approval=bool(self._require_human_approval),
        )

    # ------------------------------------------------------------------ pip install helpers
    def _docker_validate_entrypoint(self, package: str, entrypoint: str) -> Dict[str, Any]:
        """Best-effort sandbox validation using Docker (opt-in).

        Runs a disposable Python container, installs the requested package, and
        imports the configured entrypoint to ensure it is importable.
        """

        docker_exe = shutil.which("docker")
        if not docker_exe:
            return {"ok": False, "detail": {"reason": "docker_unavailable"}}

        image = str(self._docker_image or "").strip()
        if not image:
            image = "python:3.11-slim"

        script = (
            "import importlib, os, subprocess, sys\n"
            "pkg=(os.environ.get('MODULE_PACKAGE') or '').strip()\n"
            "entry=(os.environ.get('MODULE_ENTRYPOINT') or '').strip()\n"
            "if not pkg or ':' not in entry:\n"
            "    raise SystemExit(2)\n"
            "subprocess.check_call([\n"
            "    sys.executable, '-m', 'pip', 'install', pkg,\n"
            "    '--no-input', '--no-cache-dir', '--disable-pip-version-check'\n"
            "])\n"
            "mod, attr = entry.split(':', 1)\n"
            "m = importlib.import_module(mod)\n"
            "getattr(m, attr)\n"
            "print('validated')\n"
        )
        cmd = [
            docker_exe,
            "run",
            "--rm",
            "--env",
            f"MODULE_PACKAGE={package}",
            "--env",
            f"MODULE_ENTRYPOINT={entrypoint}",
            image,
            "python",
            "-c",
            script,
        ]
        timeout = max(5.0, float(self._docker_timeout_secs))
        try:
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            return {"ok": False, "detail": {"reason": "docker_timeout", "error": str(exc)}}
        except Exception as exc:
            return {"ok": False, "detail": {"reason": "docker_failed", "error": str(exc)}}

        stdout = (proc.stdout or "")[-4000:]
        stderr = (proc.stderr or "")[-4000:]
        if proc.returncode != 0:
            return {
                "ok": False,
                "detail": {
                    "reason": "docker_nonzero_exit",
                    "returncode": proc.returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                },
            }
        return {"ok": True, "detail": {"stdout": stdout, "stderr": stderr}}

    def _pip_install(self, package: str) -> Dict[str, Any]:
        self._pip_target_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            str(package),
            "--upgrade",
            "--no-input",
            "--target",
            str(self._pip_target_dir),
        ]
        try:
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        except Exception as exc:
            return {"status": "failed", "reason": "pip_failed", "error": str(exc)}
        if proc.returncode != 0:
            return {
                "status": "failed",
                "reason": "pip_failed",
                "returncode": proc.returncode,
                "stdout": (proc.stdout or "")[-4000:],
                "stderr": (proc.stderr or "")[-4000:],
            }
        # Make the target importable for the current process.
        target_str = str(self._pip_target_dir.resolve())
        if target_str not in sys.path:
            sys.path.insert(0, target_str)
        return {"status": "installed", "package": package, "target": target_str}

    def _validate_entrypoint(self, entrypoint: str) -> Dict[str, Any]:
        entrypoint = str(entrypoint or "").strip()
        if ":" not in entrypoint:
            return {"ok": False, "reason": "invalid_entrypoint"}
        module_name, attr = entrypoint.split(":", 1)
        module_name = module_name.strip()
        attr = attr.strip()
        if not module_name or not attr:
            return {"ok": False, "reason": "invalid_entrypoint"}
        target_str = str(self._pip_target_dir.resolve())
        env = dict(os.environ)
        env["PYTHONPATH"] = target_str + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        cmd = [sys.executable, "-c", f"import importlib; m=importlib.import_module('{module_name}'); getattr(m, '{attr}')"]
        try:
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env, timeout=15.0)
        except Exception as exc:
            return {"ok": False, "reason": "import_failed", "error": str(exc)}
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": (proc.stdout or "")[-2000:],
            "stderr": (proc.stderr or "")[-2000:],
        }

    def _register_entrypoint_module(self, name: str, entrypoint: str, *, replace: bool) -> bool:
        name = str(name or "").strip()
        entrypoint = str(entrypoint or "").strip()
        if not name or ":" not in entrypoint:
            return False
        module_name, attr = entrypoint.split(":", 1)
        module_name = module_name.strip()
        attr = attr.strip()
        if not module_name or not attr:
            return False

        def _factory() -> Any:
            module = importlib.import_module(module_name)
            obj = getattr(module, attr)
            if isinstance(obj, type):
                try:
                    return obj()
                except Exception:
                    return obj
            return obj

        reg = self._import_capability_module("module_registry")
        if reg is None:
            return False
        if not replace and name in getattr(reg, "_REGISTRY", {}):
            return False
        try:
            register_module = getattr(reg, "register_module", None)
            if not callable(register_module):
                return False
            register_module(name, _factory)
        except Exception:
            return False
        return True


__all__ = [
    "ModuleCandidate",
    "AcquisitionSuggestion",
    "ModuleAcquisitionManager",
]
