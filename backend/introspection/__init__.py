from __future__ import annotations

"""Agent-facing introspection helpers.

This module provides lightweight, prompt-friendly self-query utilities that can
be called by orchestration code before (or during) an LLM-driven task.

Primary interfaces:
 - :func:`get_loaded_skills`
 - :func:`summarize_my_abilities`
 - :func:`explain_my_plan`
"""

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .self_knowledge import bootstrap_self_knowledge, get_self_structure_graph, query_self_structure


DEFAULT_SAFETY_BOUNDARIES: Tuple[str, ...] = (
    "Avoid destructive filesystem changes unless explicitly requested.",
    "Do not reveal secrets/credentials; treat logs as potentially shared.",
    "Prefer least-privilege actions; keep scope limited to the task.",
    "Be transparent about uncertainty; surface assumptions and risks.",
    "Validate changes with focused tests when possible.",
)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp_unit(value: float | None) -> float | None:
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))


def _confidence_level(confidence: float | None) -> str:
    if confidence is None:
        return "unknown"
    conf = _clamp_unit(confidence)
    if conf is None:
        return "unknown"
    if conf >= 0.75:
        return "high"
    if conf >= 0.45:
        return "medium"
    return "low"


def _clip_string(value: str, *, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "…"


def _redact_payload(value: Any, *, max_str_chars: int, redact_keys: Sequence[str] = ("text", "content")) -> Any:
    if isinstance(value, str):
        return _clip_string(value, max_chars=max_str_chars)
    if isinstance(value, Mapping):
        output: Dict[str, Any] = {}
        for key, val in value.items():
            key_str = str(key)
            if key_str in redact_keys and isinstance(val, str):
                output[key_str] = _clip_string(val, max_chars=max_str_chars)
            else:
                output[key_str] = _redact_payload(val, max_str_chars=max_str_chars, redact_keys=redact_keys)
        return output
    if isinstance(value, list):
        return [_redact_payload(item, max_str_chars=max_str_chars, redact_keys=redact_keys) for item in value]
    return value


_TOKEN_RE = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]+", re.IGNORECASE)


def _tokens(text: str) -> List[str]:
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _try_get_global_skill_registry() -> Any:
    try:
        from backend.capability.skill_registry import get_skill_registry  # type: ignore

        return get_skill_registry()
    except Exception:
        return None


def _skill_spec_to_dict(spec: Any, *, include_schemas: bool) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "name": getattr(spec, "name", ""),
        "description": getattr(spec, "description", ""),
        "execution_mode": getattr(spec, "execution_mode", "local"),
        "provider": getattr(spec, "provider", "unknown"),
        "version": getattr(spec, "version", "0.0.0"),
        "enabled": bool(getattr(spec, "enabled", True)),
        "tags": list(getattr(spec, "tags", []) or []),
        "entrypoint": getattr(spec, "entrypoint", None),
        "source": getattr(spec, "source", None),
    }
    if include_schemas:
        payload["input_schema"] = dict(getattr(spec, "input_schema", {}) or {})
        payload["output_schema"] = dict(getattr(spec, "output_schema", {}) or {})
        payload["cost"] = dict(getattr(spec, "cost", {}) or {})
    return payload


def get_loaded_skills(
    registry: Any | None = None,
    *,
    limit: int = 50,
    include_disabled: bool = False,
    include_schemas: bool = False,
    as_text: bool = False,
) -> Dict[str, Any] | str:
    """Return the currently loaded skills from the runtime registry.

    Parameters
    ----------
    registry
        Optional registry object. When omitted, a best-effort global registry is
        used (via :func:`backend.capability.skill_registry.get_skill_registry`).
    limit
        Maximum number of skills to return (sorted by name).
    include_disabled
        Include disabled skills when True.
    include_schemas
        Include input/output schema and cost payloads when True.
    as_text
        When True, return a prompt-friendly text block; otherwise return a dict.
    """

    registry = registry or _try_get_global_skill_registry()
    if registry is None:
        payload = {
            "skills": [],
            "returned": 0,
            "total": 0,
            "truncated": False,
            "note": "skill registry unavailable",
        }
        return _format_loaded_skills(payload) if as_text else payload

    try:
        specs = list(registry.list_specs())
    except Exception as err:
        payload = {
            "skills": [],
            "returned": 0,
            "total": 0,
            "truncated": False,
            "note": f"failed to list skills: {err}",
        }
        return _format_loaded_skills(payload) if as_text else payload

    filtered = [
        spec for spec in specs if include_disabled or bool(getattr(spec, "enabled", True))
    ]
    filtered.sort(key=lambda s: str(getattr(s, "name", "")).lower())

    safe_limit = max(0, int(limit))
    selected = filtered if safe_limit == 0 else filtered[:safe_limit]
    skills = [_skill_spec_to_dict(spec, include_schemas=include_schemas) for spec in selected]
    payload = {
        "skills": skills,
        "returned": len(skills),
        "total": len(filtered),
        "truncated": len(filtered) > len(skills),
        "filters": {
            "include_disabled": bool(include_disabled),
            "include_schemas": bool(include_schemas),
            "limit": safe_limit,
        },
    }
    return _format_loaded_skills(payload) if as_text else payload


def _format_loaded_skills(payload: Mapping[str, Any]) -> str:
    total = int(payload.get("total") or 0)
    returned = int(payload.get("returned") or 0)
    truncated = bool(payload.get("truncated"))
    note = payload.get("note")
    lines = [f"Loaded skills: {total} (showing {returned}{'+' if truncated else ''})"]
    if note:
        lines.append(f"Note: {note}")
    for entry in payload.get("skills", []) or []:
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("name") or "").strip() or "unnamed"
        desc = str(entry.get("description") or "").strip()
        provider = str(entry.get("provider") or "unknown").strip()
        mode = str(entry.get("execution_mode") or "local").strip()
        enabled = "enabled" if entry.get("enabled", True) else "disabled"
        tag_list = entry.get("tags") or []
        tags = ""
        if isinstance(tag_list, Sequence) and not isinstance(tag_list, (str, bytes)):
            tags = ", ".join(str(t) for t in tag_list if str(t).strip())
        extra = f"provider={provider}, mode={mode}, {enabled}"
        if tags:
            extra += f", tags={tags}"
        if desc:
            lines.append(f"- {name}: {desc} ({extra})")
        else:
            lines.append(f"- {name} ({extra})")
    return "\n".join(lines).strip()


@dataclass(frozen=True)
class AbilityEntry:
    name: str
    confidence: float
    level: str
    details: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"name": self.name, "confidence": self.confidence, "level": self.level}
        if self.details:
            payload["details"] = dict(self.details)
        return payload


def _extract_abilities(self_model: Any) -> Tuple[List[AbilityEntry], str]:
    if self_model is None:
        return [], "self_model unavailable"

    # Case 1: backend.self_model.SelfModel (capabilities: name -> float)
    try:
        capabilities = getattr(self_model, "capabilities")
    except Exception:
        capabilities = None

    if isinstance(capabilities, Mapping):
        entries: list[AbilityEntry] = []
        for name, raw in capabilities.items():
            if name is None:
                continue
            confidence = _clamp_unit(_safe_float(raw))
            if confidence is None:
                continue
            entries.append(
                AbilityEntry(
                    name=str(name),
                    confidence=float(confidence),
                    level=_confidence_level(confidence),
                    details={},
                )
            )
        return entries, "self_model.capabilities"

    # Case 2: modules.brain.self_model.SelfModel (capability_summary/state.capabilities)
    summary_table = None
    method = getattr(self_model, "capability_summary", None)
    if callable(method):
        try:
            summary_table = method()
        except Exception:
            summary_table = None
    if summary_table is None:
        state = getattr(self_model, "state", None)
        summary_table = getattr(state, "capabilities", None) if state is not None else None

    if isinstance(summary_table, Mapping):
        entries = []
        for name, raw in summary_table.items():
            if name is None:
                continue
            details: Dict[str, Any] = {}
            confidence_val: float | None = None
            if isinstance(raw, Mapping):
                confidence_val = _clamp_unit(_safe_float(raw.get("weight")))
                for key in ("success_rate", "attempts", "successes", "failures", "last_outcome", "last_used"):
                    if key in raw:
                        details[key] = raw.get(key)
            else:
                confidence_val = _clamp_unit(_safe_float(raw))
            if confidence_val is None:
                continue
            entries.append(
                AbilityEntry(
                    name=str(name),
                    confidence=float(confidence_val),
                    level=_confidence_level(confidence_val),
                    details=details,
                )
            )
        return entries, "self_model.capability_summary"

    return [], "no capabilities exposed by self_model"


def summarize_my_abilities(
    self_model: Any | None = None,
    *,
    max_items: int = 12,
    min_confidence: float = 0.0,
    as_text: bool = False,
) -> Dict[str, Any] | str:
    """Summarize the agent's abilities and confidence.

    The implementation prefers :attr:`SelfModel.capabilities` when present, but
    also supports the whole-brain self model's ``capability_summary()`` output.
    """

    abilities, source = _extract_abilities(self_model)
    min_conf = _clamp_unit(_safe_float(min_confidence))
    if min_conf is None:
        min_conf = 0.0

    filtered = [entry for entry in abilities if entry.confidence >= float(min_conf)]
    filtered.sort(key=lambda e: (-e.confidence, e.name.lower()))
    limit = max(0, int(max_items))
    selected = filtered if limit == 0 else filtered[:limit]

    if selected:
        avg_conf = sum(entry.confidence for entry in selected) / len(selected)
    else:
        avg_conf = 0.0

    level_counts: Dict[str, int] = {"high": 0, "medium": 0, "low": 0, "unknown": 0}
    for entry in selected:
        level_counts[entry.level] = int(level_counts.get(entry.level, 0)) + 1

    summary = (
        f"Abilities: {len(selected)} shown"
        f" (avg_conf={avg_conf:.2f}; high={level_counts['high']},"
        f" medium={level_counts['medium']}, low={level_counts['low']})"
    )
    payload = {
        "summary": summary,
        "abilities": [entry.as_dict() for entry in selected],
        "returned": len(selected),
        "total": len(filtered),
        "truncated": len(filtered) > len(selected),
        "stats": {
            "avg_confidence": round(avg_conf, 3),
            "levels": dict(level_counts),
            "min_confidence": float(min_conf),
        },
        "source": source,
    }

    # Optional: include weakness table when exposed by whole-brain self model.
    weaknesses = None
    state = getattr(self_model, "state", None) if self_model is not None else None
    if state is not None:
        weaknesses = getattr(state, "weaknesses", None)
    if isinstance(weaknesses, Mapping) and weaknesses:
        payload["weaknesses"] = dict(weaknesses)

    return _format_abilities(payload) if as_text else payload


def _format_abilities(payload: Mapping[str, Any]) -> str:
    lines = [str(payload.get("summary") or "Abilities summary")]
    note = payload.get("note")
    if note:
        lines.append(f"Note: {note}")
    for entry in payload.get("abilities", []) or []:
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("name") or "").strip() or "unknown"
        conf = entry.get("confidence")
        level = str(entry.get("level") or "unknown")
        try:
            conf_str = f"{float(conf):.2f}"
        except Exception:
            conf_str = "?"
        lines.append(f"- {name}: {conf_str} ({level})")
    weaknesses = payload.get("weaknesses")
    if isinstance(weaknesses, Mapping) and weaknesses:
        weak_keys = ", ".join(str(k) for k in list(weaknesses.keys())[:6])
        lines.append(f"Weakness signals: {weak_keys}")
    return "\n".join(lines).strip()


def _score_skill_for_task(task_text: str, task_tokens: set[str], skill: Mapping[str, Any]) -> float:
    name = str(skill.get("name") or "")
    desc = str(skill.get("description") or "")
    tags_raw = skill.get("tags") or []
    tags: list[str] = []
    if isinstance(tags_raw, Sequence) and not isinstance(tags_raw, (str, bytes)):
        tags = [str(t) for t in tags_raw if str(t).strip()]

    score = 0.0
    task_lower = task_text.lower()
    name_lower = name.lower().strip()
    if name_lower and name_lower in task_lower:
        score += 6.0

    for token in _tokens(name_lower):
        if len(token) >= 3 and token in task_tokens:
            score += 2.0

    for tag in tags:
        tag_lower = tag.lower().strip()
        if not tag_lower:
            continue
        if tag_lower in task_tokens:
            score += 1.5
        elif len(tag_lower) >= 4 and tag_lower in task_lower:
            score += 0.8

    # Lightly weight description overlap (cap at first 40 tokens).
    for token in _tokens(desc)[:40]:
        if len(token) >= 4 and token in task_tokens:
            score += 0.2

    return score


def _default_plan(task_text: str) -> List[Dict[str, str]]:
    base = [
        ("Clarify goal and constraints", "Ensure the plan matches the task and constraints."),
        ("Inspect relevant code/data", "Ground decisions in the current architecture and interfaces."),
        ("Design minimal change", "Prefer the smallest correct change to reduce risk."),
        ("Implement and validate", "Apply changes with focused checks/tests."),
        ("Summarize results and risks", "Communicate what changed and what remains uncertain."),
    ]
    if task_text and any(word in task_text.lower() for word in ("bug", "error", "fail", "exception", "crash")):
        base.insert(
            2,
            ("Reproduce/locate failure", "Confirm root cause before implementing a fix."),
        )
    return [{"step": title, "reason": reason} for title, reason in base]


def _action_expectation(action: Mapping[str, Any], *, goal: str = "", step_title: str = "") -> Dict[str, str]:
    action_type = str(action.get("type") or "").strip().lower()
    if not action_type:
        return {
            "expected": "Action has no 'type'; execution is likely to fail validation.",
            "verification": "Ensure the action dict includes a non-empty 'type'.",
            "rationale": "Fix the action specification before attempting execution.",
        }

    def _effect(prefix: str) -> str:
        if goal and step_title:
            return f"{prefix} to advance goal '{goal}' via step '{step_title}'."
        if goal:
            return f"{prefix} to advance goal '{goal}'."
        if step_title:
            return f"{prefix} for step '{step_title}'."
        return prefix + "."

    if action_type == "read_file":
        path = str(action.get("path") or "").strip() or "<path>"
        max_chars = action.get("max_chars")
        expected = f"Read file '{path}'"
        if isinstance(max_chars, (int, float)):
            expected += f" (up to {int(max_chars)} chars)."
        else:
            expected += "."
        return {
            "expected": expected,
            "verification": "Confirm the returned observation contains the expected text and no read error.",
            "rationale": _effect("Gather information from local files"),
        }

    if action_type == "list_dir":
        path = str(action.get("path") or ".").strip() or "."
        max_entries = action.get("max_entries")
        expected = f"List directory '{path}'"
        if isinstance(max_entries, (int, float)):
            expected += f" (up to {int(max_entries)} entries)."
        else:
            expected += "."
        return {
            "expected": expected,
            "verification": "Check the observation lists the expected files/directories.",
            "rationale": _effect("Inspect the workspace structure"),
        }

    if action_type in {"create_dir"}:
        path = str(action.get("path") or "").strip() or "<path>"
        return {
            "expected": f"Ensure directory '{path}' exists (create if missing).",
            "verification": "List the directory or confirm subsequent steps can write into it.",
            "rationale": _effect("Prepare the filesystem workspace"),
        }

    if action_type in {"write_file", "create_file", "modify_file"}:
        path = str(action.get("path") or "").strip() or "<path>"
        text = action.get("text")
        chars = len(text) if isinstance(text, str) else None
        op = {
            "write_file": "Write",
            "create_file": "Create",
            "modify_file": "Modify",
        }.get(action_type, "Write")
        suffix = f" ({chars} chars)" if chars is not None else ""
        return {
            "expected": f"{op} file '{path}'{suffix}.",
            "verification": "Read the file back or run focused checks/tests to confirm behavior.",
            "rationale": _effect("Apply a concrete change to the workspace"),
        }

    if action_type == "delete_file":
        path = str(action.get("path") or "").strip() or "<path>"
        return {
            "expected": f"Delete file '{path}' (may be blocked by safety policy).",
            "verification": "Confirm the file is removed only if explicitly intended and approved/confirmed.",
            "rationale": _effect("Remove an unwanted artifact"),
        }

    if action_type in {"shell", "exec_system_cmd"}:
        cmd = str(action.get("cmd") or action.get("command") or "").strip() or "<command>"
        return {
            "expected": f"Execute command: {cmd}",
            "verification": "Check exit status/output; avoid running destructive commands without explicit confirmation.",
            "rationale": _effect("Use system tooling to inspect/build/verify"),
        }

    if action_type in {"launch_program"}:
        program = str(action.get("program") or action.get("cmd") or "").strip() or "<program>"
        return {
            "expected": f"Launch program: {program}",
            "verification": "Confirm the process starts successfully and clean up if it is no longer needed.",
            "rationale": _effect("Start a required local process"),
        }

    if action_type in {"kill_process"}:
        pid = action.get("pid")
        return {
            "expected": f"Terminate process pid={pid} (may be blocked by safety policy).",
            "verification": "Confirm it targets the intended process and does not disrupt unrelated services.",
            "rationale": _effect("Stop a runaway or unneeded process"),
        }

    if action_type in {"docker", "docker_compose"}:
        return {
            "expected": f"Invoke Docker control action '{action_type}' (may change container state).",
            "verification": "Confirm the intended containers/services are affected and nothing else.",
            "rationale": _effect("Manage containerized services for the task"),
        }

    if action_type == "run_script":
        path = str(action.get("path") or "").strip() or "<script>"
        return {
            "expected": f"Run approved script '{path}'.",
            "verification": "Review output and confirm side effects are expected and within scope.",
            "rationale": _effect("Execute a predefined automation script"),
        }

    if action_type == "remote_tool":
        endpoint = str(action.get("endpoint") or "").strip() or "<endpoint>"
        inner = action.get("action")
        inner_type = ""
        if isinstance(inner, Mapping):
            inner_type = str(inner.get("type") or "").strip()
        return {
            "expected": f"Proxy tool action to remote endpoint '{endpoint}'{(' (' + inner_type + ')') if inner_type else ''}.",
            "verification": "Validate remote results and treat remote side effects as high risk.",
            "rationale": _effect("Delegate execution to a remote tool service"),
        }

    if action_type.startswith("sandbox_"):
        return {
            "expected": f"Perform filesystem sandbox operation '{action_type}'.",
            "verification": "Review sandbox status before committing staged changes.",
            "rationale": _effect("Use a transaction-style sandbox for safer file operations"),
        }

    if action_type in {"ui", "motor"}:
        return {
            "expected": f"Execute {action_type} automation action(s).",
            "verification": "Confirm UI automation targets the correct window/context and respects safety policy.",
            "rationale": _effect("Perform an embodied UI/actuation step"),
        }

    return {
        "expected": f"Execute action type '{action_type}'.",
        "verification": "Check the returned info for errors/blocked reasons and validate the outcome.",
        "rationale": _effect("Progress the task with an available tool action"),
    }


def explain_my_plan(
    task: str | Mapping[str, Any],
    *,
    registry: Any | None = None,
    self_model: Any | None = None,
    max_plan_steps: int = 8,
    max_skill_suggestions: int = 6,
    safety_boundaries: Sequence[str] | None = None,
    as_text: bool = False,
) -> Dict[str, Any] | str:
    """Explain how the agent intends to approach ``task``.

    The output includes:
      - a short plan with per-step rationale,
      - candidate skills (best-effort matching against the loaded registry),
      - explicit safety boundaries suitable to embed into an LLM prompt.
    """

    raw_action: Mapping[str, Any] | None = None
    step_title = ""
    goal_hint = ""
    if isinstance(task, Mapping):
        task_text = (
            task.get("task")
            or task.get("goal")
            or task.get("name")
            or task.get("description")
            or ""
        )
        task_text = str(task_text)
        goal_hint = str(task.get("goal") or task.get("task") or "").strip()
        step_title = str(task.get("step") or task.get("step_title") or task.get("title") or "").strip()
        candidate_action = task.get("action") or task.get("step_action")
        if isinstance(candidate_action, Mapping):
            raw_action = candidate_action
            action_type = str(candidate_action.get("type") or "").strip()
            if action_type:
                task_text = f"{task_text} [{action_type}]".strip() if task_text else f"[{action_type}]"
            if step_title and step_title.lower() not in task_text.lower():
                task_text = f"{task_text} ({step_title})".strip()
    else:
        task_text = str(task)
    task_text = task_text.strip()

    plan_steps = _default_plan(task_text)
    plan_limit = max(0, int(max_plan_steps))
    if plan_limit:
        plan_steps = plan_steps[:plan_limit]

    skill_payload = get_loaded_skills(
        registry,
        limit=0,
        include_disabled=False,
        include_schemas=False,
        as_text=False,
    )
    skills: list[dict[str, Any]] = []
    if isinstance(skill_payload, Mapping):
        raw_skills = skill_payload.get("skills") or []
        if isinstance(raw_skills, Sequence):
            for entry in raw_skills:
                if isinstance(entry, Mapping):
                    skills.append(dict(entry))

    task_tokens = set(_tokens(task_text))
    scored: list[tuple[float, dict[str, Any]]] = []
    for skill in skills:
        score = _score_skill_for_task(task_text, task_tokens, skill)
        if score <= 0:
            continue
        scored.append((score, skill))
    scored.sort(key=lambda item: (-item[0], str(item[1].get("name", "")).lower()))

    skill_limit = max(0, int(max_skill_suggestions))
    selected_skills: list[dict[str, Any]] = []
    for score, skill in (scored if skill_limit == 0 else scored[:skill_limit]):
        selected_skills.append(
            {
                "name": skill.get("name"),
                "description": skill.get("description"),
                "provider": skill.get("provider"),
                "tags": skill.get("tags"),
                "score": round(float(score), 3),
            }
        )

    abilities = summarize_my_abilities(self_model, max_items=8, as_text=False)

    boundaries = [str(b).strip() for b in (safety_boundaries or DEFAULT_SAFETY_BOUNDARIES) if str(b).strip()]
    payload = {
        "task": task_text,
        "plan": plan_steps,
        "skills": {
            "suggested": selected_skills,
            "available_count": int(skill_payload.get("total", 0)) if isinstance(skill_payload, Mapping) else 0,
        },
        "abilities": abilities,
        "safety_boundaries": boundaries,
    }
    if raw_action is not None:
        action_dict = dict(raw_action)
        action_expectation = _action_expectation(action_dict, goal=goal_hint, step_title=step_title)
        payload["step_title"] = step_title
        payload["action"] = _redact_payload(action_dict, max_str_chars=256)
        payload.update(action_expectation)
    return _format_plan_explanation(payload) if as_text else payload


def _format_plan_explanation(payload: Mapping[str, Any]) -> str:
    task = str(payload.get("task") or "").strip() or "unspecified task"
    lines = [f"Task: {task}", "Plan:"]
    for idx, step in enumerate(payload.get("plan", []) or [], start=1):
        if not isinstance(step, Mapping):
            continue
        title = str(step.get("step") or "").strip()
        reason = str(step.get("reason") or "").strip()
        if title and reason:
            lines.append(f"{idx}. {title} — {reason}")
        elif title:
            lines.append(f"{idx}. {title}")

    skills = payload.get("skills") or {}
    if isinstance(skills, Mapping):
        suggested = skills.get("suggested") or []
        if isinstance(suggested, Sequence) and suggested:
            lines.append("Candidate skills:")
            for entry in suggested:
                if not isinstance(entry, Mapping):
                    continue
                name = str(entry.get("name") or "").strip()
                desc = str(entry.get("description") or "").strip()
                score = entry.get("score")
                score_str = f"{float(score):.1f}" if isinstance(score, (int, float)) else "?"
                if name and desc:
                    lines.append(f"- {name}: {desc} (score={score_str})")
                elif name:
                    lines.append(f"- {name} (score={score_str})")

    boundaries = payload.get("safety_boundaries") or []
    if isinstance(boundaries, Sequence) and boundaries:
        lines.append("Safety boundaries:")
        for boundary in boundaries:
            text = str(boundary).strip()
            if text:
                lines.append(f"- {text}")

    expected = payload.get("expected")
    if isinstance(expected, str) and expected.strip():
        lines.append(f"Expected: {expected.strip()}")
    verification = payload.get("verification")
    if isinstance(verification, str) and verification.strip():
        lines.append(f"Verify: {verification.strip()}")
    rationale = payload.get("rationale")
    if isinstance(rationale, str) and rationale.strip():
        lines.append(f"Rationale: {rationale.strip()}")

    abilities = payload.get("abilities")
    if isinstance(abilities, Mapping):
        abilities_text = _format_abilities(abilities)
        if abilities_text:
            lines.append("Abilities snapshot:")
            lines.append(abilities_text)

    return "\n".join(lines).strip()


class IntrospectionInterface:
    """Small façade bundling introspection state for repeated calls."""

    def __init__(
        self,
        *,
        registry: Any | None = None,
        self_model: Any | None = None,
        knowledge_base: Any | None = None,
        graph_store: Any | None = None,
    ) -> None:
        self._registry = registry
        self._self_model = self_model
        self._knowledge_base = knowledge_base
        self._graph_store = graph_store

    @property
    def registry(self) -> Any | None:
        return self._registry

    @registry.setter
    def registry(self, value: Any | None) -> None:
        self._registry = value

    @property
    def self_model(self) -> Any | None:
        return self._self_model

    @self_model.setter
    def self_model(self, value: Any | None) -> None:
        self._self_model = value

    @property
    def knowledge_base(self) -> Any | None:
        return self._knowledge_base

    @knowledge_base.setter
    def knowledge_base(self, value: Any | None) -> None:
        self._knowledge_base = value

    @property
    def graph_store(self) -> Any | None:
        return self._graph_store

    @graph_store.setter
    def graph_store(self, value: Any | None) -> None:
        self._graph_store = value

    def get_loaded_skills(self, **kwargs: Any) -> Dict[str, Any] | str:
        return get_loaded_skills(self._registry, **kwargs)

    def summarize_my_abilities(self, **kwargs: Any) -> Dict[str, Any] | str:
        return summarize_my_abilities(self._self_model, **kwargs)

    def explain_my_plan(self, task: str | Mapping[str, Any], **kwargs: Any) -> Dict[str, Any] | str:
        return explain_my_plan(task, registry=self._registry, self_model=self._self_model, **kwargs)

    def bootstrap_self_knowledge(self, **kwargs: Any) -> Dict[str, Any]:
        return bootstrap_self_knowledge(
            knowledge_base=self._knowledge_base,
            registry=self._registry,
            graph_store=self._graph_store,
            **kwargs,
        )

    def query_self_structure(self, module_name: str, **kwargs: Any) -> Dict[str, Any] | str:
        return query_self_structure(
            module_name,
            knowledge_base=self._knowledge_base,
            graph_store=self._graph_store,
            **kwargs,
        )

    def get_self_structure_graph(self, **kwargs: Any) -> Dict[str, Any]:
        return get_self_structure_graph(graph_store=self._graph_store, **kwargs)


__all__ = [
    "AbilityEntry",
    "DEFAULT_SAFETY_BOUNDARIES",
    "IntrospectionInterface",
    "bootstrap_self_knowledge",
    "explain_my_plan",
    "get_self_structure_graph",
    "get_loaded_skills",
    "query_self_structure",
    "summarize_my_abilities",
]
