"""Knowledge-driven action guard that validates agent commands."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

from backend.autogpt.autogpt.core.knowledge_graph.ontology import (
    EntityType,
    RelationType,
)
from backend.knowledge.registry import get_graph_store_instance

LOGGER = logging.getLogger(__name__)


@dataclass
class ActionGuardResult:
    """Outcome of an action constraint evaluation."""

    allowed: bool
    reason: str = ""
    violations: List[str] = field(default_factory=list)


class ActionGuard:
    """Evaluate pending actions against knowledge graph constraints."""

    def __init__(
        self,
        *,
        graph_store: Any | None = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._graph = graph_store or get_graph_store_instance()
        self._logger = logger or LOGGER

    def evaluate(
        self,
        command_name: str,
        command_args: Dict[str, Any],
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> ActionGuardResult:
        """Return whether ``command_name`` is permitted given the knowledge rules."""

        if not self._graph:
            return ActionGuardResult(True)

        snapshot = self._graph.get_snapshot()
        nodes: Iterable[Any] = snapshot.get("nodes", [])
        edges: Iterable[Any] = snapshot.get("edges", [])
        node_map = {node.id: node for node in nodes if hasattr(node, "id")}

        # ------------------------------------------------------------------ knowledge-source deny list (ops safety)
        # Allow operators to mark "bad knowledge sources" in the knowledge graph.
        # This check applies even when there is no explicit action-node rule.
        denied_reason = _blocked_source_reason(command_name, command_args, nodes)
        if denied_reason:
            self._logger.info("Action guard blocked '%s' due to blocked source: %s", command_name, denied_reason)
            return ActionGuardResult(False, reason=denied_reason, violations=[denied_reason])

        matches = [
            node
            for node in nodes
            if _node_matches_action(node, command_name)
        ]
        if not matches:
            return ActionGuardResult(True)

        violations: List[str] = []
        for action_node in matches:
            props = getattr(action_node, "properties", {}) or {}
            if props.get("allow") is False:
                reason = props.get("reason") or _default_forbidden_reason(command_name)
                violations.append(reason)
                continue

            action_edges = [
                edge
                for edge in edges
                if edge.source == action_node.id
                and _relation_type(edge) == RelationType.REQUIRES.value
            ]
            for edge in action_edges:
                requirement = node_map.get(edge.target)
                ok, reason = _evaluate_requirement(
                    requirement,
                    command_args,
                    context or {},
                    getattr(edge, "properties", {}) or {},
                )
                if not ok:
                    violations.append(reason or _default_requirement_reason(requirement, command_name))

        if violations:
            combined = "; ".join(violations)
            self._logger.info("Action guard blocked '%s': %s", command_name, combined)
            return ActionGuardResult(False, reason=combined, violations=violations)
        return ActionGuardResult(True)


def _blocked_source_reason(command_name: str, command_args: Dict[str, Any], nodes: Iterable[Any]) -> str:
    """Return a deny reason if a knowledge source used by this action is blocked.

    Supported source nodes (concept nodes) can use properties like:
      - {"source_kind": "web_domain", "domain": "example.com", "blocked": true, "reason": "..."}
      - {"source_kind": "github_repo", "repo": "owner/repo", "blocked": true}

    For web actions (`web_get`, `web_scrape`, `github_repo_ingest`) the guard
    extracts a hostname/repo from `command_args` and suffix-matches against
    blocked entries.
    """

    action = str(command_name or "").strip().lower()
    host = ""
    repo = ""
    if action in {"web_get", "web_scrape"}:
        url = command_args.get("url") or command_args.get("href") or ""
        host = _hostname_from_url(str(url))
    elif action in {"github_repo_ingest"}:
        raw_repo = command_args.get("repo") or command_args.get("repository") or command_args.get("url") or ""
        repo = _normalize_github_repo(str(raw_repo))

    if not host and not repo:
        return ""

    for node in nodes or []:
        props = getattr(node, "properties", None) or {}
        if not isinstance(props, dict) or not props:
            continue

        if not _is_blocked_source_node(props):
            continue

        source_kind = str(props.get("source_kind") or props.get("kind") or "").strip().lower()
        if not source_kind:
            # Heuristic: treat "domain"/"repo" properties as indicative.
            if "repo" in props:
                source_kind = "github_repo"
            elif "domain" in props or "host" in props:
                source_kind = "web_domain"

        reason = str(props.get("reason") or props.get("note") or "").strip()
        if source_kind in {"web_domain", "domain", "hostname", "web_host"} and host:
            domains = _as_string_list(props.get("domain") or props.get("host") or props.get("value"))
            for dom in domains:
                d = str(dom or "").strip().lower().strip(".")
                if not d:
                    continue
                if host == d or host.endswith("." + d):
                    return reason or f"Blocked web domain: {d}"

        if source_kind in {"github_repo", "repo", "github"} and repo:
            repos = _as_string_list(props.get("repo") or props.get("value"))
            for entry in repos:
                r = _normalize_github_repo(str(entry))
                if r and r == repo:
                    return reason or f"Blocked GitHub repo: {r}"

    return ""


def _is_blocked_source_node(props: Dict[str, Any]) -> bool:
    if props.get("blocked") is True:
        return True
    trust = str(props.get("trust") or "").strip().lower()
    if trust in {"blocked", "deny", "denied", "untrusted", "blacklist", "blacklisted"}:
        return True
    if props.get("allow") is False and str(props.get("source_kind") or props.get("kind") or "").strip():
        return True
    return False


def _as_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if str(v).strip()]
    raw = str(value).strip()
    if not raw:
        return []
    return [raw]


def _hostname_from_url(url: str) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    try:
        host = str(urlparse(raw).hostname or "")
    except Exception:
        host = ""
    return host.strip().lower().strip(".")


def _normalize_github_repo(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    # Accept either "owner/repo" or a full URL.
    if raw.startswith("http://") or raw.startswith("https://"):
        try:
            parts = urlparse(raw).path.strip("/").split("/")
        except Exception:
            parts = []
        if len(parts) >= 2:
            raw = "/".join(parts[:2])
    raw = raw.strip().strip("/")
    if not raw or "/" not in raw:
        return ""
    owner, repo_name = raw.split("/", 1)
    return f"{owner.strip().lower()}/{repo_name.strip().lower()}"


def _node_matches_action(node: Any, command_name: str) -> bool:
    if getattr(node, "type", None) not in (EntityType.CONCEPT, EntityType.SKILL, str(EntityType.CONCEPT)):
        return False
    props = getattr(node, "properties", {}) or {}

    command = props.get("command")
    if isinstance(command, str) and command.lower() == command_name.lower():
        return True
    aliases = props.get("aliases")
    if isinstance(aliases, (list, tuple)):
        if any(str(alias).lower() == command_name.lower() for alias in aliases):
            return True
    actions = props.get("actions")
    if isinstance(actions, (list, tuple)):
        if any(str(action).lower() == command_name.lower() for action in actions):
            return True
    return False


def _relation_type(edge: Any) -> str:
    edge_type = getattr(edge, "type", "")
    if hasattr(edge_type, "value"):
        return edge_type.value
    return str(edge_type)


def _evaluate_requirement(
    node: Any,
    command_args: Dict[str, Any],
    context: Dict[str, Any],
    edge_props: Dict[str, Any],
) -> tuple[bool, str]:
    props = getattr(node, "properties", {}) or {}
    props = {**edge_props, **props}

    if props.get("allow") is False:
        return False, props.get("reason") or "Action forbidden by knowledge rule."

    rule_type = str(props.get("rule_type") or props.get("type") or "").lower()
    if not rule_type:
        return True, ""

    if rule_type in {"deny", "forbidden"}:
        return False, props.get("reason") or "Action blocked by policy."

    if rule_type == "flag":
        key = props.get("key")
        expected = props.get("expected", True)
        if key is None:
            return True, ""
        actual = _lookup_value(command_args, context, key)
        if bool(actual) == bool(expected):
            return True, ""
        reason = props.get("reason") or f"Flag '{key}' must be set to {expected!r}."
        return False, reason

    if rule_type == "equals":
        key = props.get("key")
        expected = props.get("value")
        if key is None:
            return True, ""
        actual = _lookup_value(command_args, context, key)
        if actual == expected:
            return True, ""
        reason = props.get("reason") or f"Argument '{key}' must equal {expected!r}."
        return False, reason

    if rule_type == "regex":
        key = props.get("key")
        pattern = props.get("pattern")
        if not key or not pattern:
            return True, ""
        value = str(_lookup_value(command_args, context, key) or "")
        if re.fullmatch(pattern, value):
            return True, ""
        reason = props.get("reason") or f"Argument '{key}' must match pattern '{pattern}'."
        return False, reason

    if rule_type == "mode":
        required_mode = str(props.get("value") or props.get("mode") or "").lower()
        if not required_mode:
            return True, ""
        current_mode = str(context.get("mode") or "").lower()
        if current_mode == required_mode:
            return True, ""
        reason = props.get("reason") or f"Action allowed only in mode '{required_mode}'."
        return False, reason

    return True, ""


def _lookup_value(command_args: Dict[str, Any], context: Dict[str, Any], key: str) -> Any:
    if key in command_args:
        return command_args[key]
    if key in context:
        return context[key]
    return None


def _default_forbidden_reason(command_name: str) -> str:
    return f"Action '{command_name}' is disallowed by knowledge constraints."


def _default_requirement_reason(node: Any, command_name: str) -> str:
    node_id = getattr(node, "id", None)
    return (
        f"Knowledge requirement '{node_id}' is not satisfied for action '{command_name}'."
        if node_id
        else f"Knowledge requirement unmet for action '{command_name}'."
    )
