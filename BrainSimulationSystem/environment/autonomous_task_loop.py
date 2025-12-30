"""Natural-language task planning + closed-loop execution helpers.

This module provides a small "plan -> execute -> observe -> (optional) replan"
loop for grounding high-level user goals into concrete environment actions.

Design goals:
  - Safety-first: leave dangerous capabilities disabled by default and provide
    an explicit confirmation gate (Governor) before executing high-risk steps.
  - Dependency-light: works without external LLMs; supports optional LLMService.
  - Testable: deterministic parsing of explicit JSON plans enables unit tests.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from BrainSimulationSystem.environment.tool_bridge import ToolEnvironmentBridge

try:  # pragma: no cover - optional cross-package introspection utilities
    from backend.introspection import IntrospectionInterface
except Exception:  # pragma: no cover - keep module usable without backend package
    IntrospectionInterface = None  # type: ignore[assignment]

try:  # pragma: no cover - optional integration
    from BrainSimulationSystem.integration.llm_service import LLMService
except Exception:  # pragma: no cover - keep module usable without LLM layer
    LLMService = None  # type: ignore[assignment]

try:  # pragma: no cover - optional security layer (approvals/permissions/audit)
    from BrainSimulationSystem.environment.security_manager import (
        SecurityManager,
        action_fingerprint,
        redact_action,
    )
except Exception:  # pragma: no cover
    SecurityManager = None  # type: ignore[assignment]
    action_fingerprint = None  # type: ignore[assignment]
    redact_action = None  # type: ignore[assignment]

try:  # pragma: no cover - optional evolution loop (strategy self-improvement)
    from modules.evolution.agent_self_improvement import AgentSelfImprovementController
except Exception:  # pragma: no cover - keep loop usable without modules/
    AgentSelfImprovementController = None  # type: ignore[assignment]

try:  # pragma: no cover - optional meta-learning retrieval policy
    from modules.learning.meta_retrieval_policy import MetaRetrievalPolicy, MetaRetrievalPolicyConfig
except Exception:  # pragma: no cover - keep loop usable without modules/
    MetaRetrievalPolicy = None  # type: ignore[assignment]
    MetaRetrievalPolicyConfig = None  # type: ignore[assignment]

try:  # pragma: no cover - optional knowledge consolidation into long-term memory
    from modules.knowledge.knowledge_consolidation import (
        ExternalKnowledgeConsolidator,
        KnowledgeConsolidationConfig,
    )
except Exception:  # pragma: no cover - keep loop usable without modules/
    ExternalKnowledgeConsolidator = None  # type: ignore[assignment]
    KnowledgeConsolidationConfig = None  # type: ignore[assignment]

try:  # pragma: no cover - optional telemetry event type
    from modules.monitoring.collector import MetricEvent
except Exception:  # pragma: no cover
    MetricEvent = None  # type: ignore[assignment]


class TaskPlanningError(ValueError):
    """Raised when a goal cannot be converted into an executable plan."""


def _clip_text(value: str, *, max_chars: int = 2048) -> str:
    if max_chars <= 0:
        return ""
    if not value:
        return ""
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "..."


def _env_flag(name: str) -> bool:
    value = str(os.environ.get(name) or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _safe_unit(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, num))


def _extract_uncertainty(info: Dict[str, Any]) -> Optional[float]:
    if not isinstance(info, dict):
        return None
    for key in ("uncertainty", "risk"):
        if key in info:
            return _safe_unit(info.get(key))
    if "confidence" in info:
        conf = _safe_unit(info.get("confidence"))
        if conf is None:
            return None
        return max(0.0, min(1.0, 1.0 - conf))
    return None


def _hostname_from_url(url: str) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    try:
        host = str(urlparse(raw).hostname or "")
    except Exception:
        host = ""
    return host.strip().lower().strip(".")


def _tool_info_for_metrics(action_type: str, info: Dict[str, Any]) -> Dict[str, Any]:
    """Return a small, JSON-friendly subset of tool `info` for telemetry."""

    if not isinstance(info, dict):
        return {}
    action = str(action_type or "").strip().lower()
    out: Dict[str, Any] = {}

    if action in {"write_file", "create_file"}:
        for key in ("path", "chars", "append", "overwritten", "content_sha1", "sandboxed"):
            if key in info and isinstance(info.get(key), (str, int, float, bool)):
                out[key] = info.get(key)
        return out

    if action in {"modify_file"}:
        for key in ("path", "changes", "operation", "sandboxed"):
            if key in info and isinstance(info.get(key), (str, int, float, bool)):
                out[key] = info.get(key)
        return out

    if action in {"delete_file"}:
        for key in ("path", "deleted", "missing", "sandboxed"):
            if key in info and isinstance(info.get(key), (str, int, float, bool)):
                out[key] = info.get(key)
        return out

    if action in {"web_search"}:
        for key in ("query", "returned", "unique_hosts", "avg_trust"):
            if key in info and isinstance(info.get(key), (str, int, float, bool)):
                out[key] = info.get(key)
        trust_counts = info.get("trust_counts")
        if isinstance(trust_counts, dict):
            out["trust_counts"] = {
                k: int(trust_counts.get(k, 0)) for k in ("high", "medium", "low") if k in trust_counts
            }
        low_urls = info.get("low_trust_urls")
        if isinstance(low_urls, list):
            out["low_trust_urls"] = [str(u) for u in low_urls[:5] if u]
        return out

    if action in {"documentation_tool"}:
        for key in ("query", "returned"):
            if key in info and isinstance(info.get(key), (str, int, float, bool)):
                out[key] = info.get(key)
        consensus = info.get("consensus")
        if isinstance(consensus, dict):
            for key in ("level", "similarity_avg", "avg_trust", "unique_hosts", "needs_verification"):
                if key in consensus and isinstance(consensus.get(key), (str, int, float, bool)):
                    out[f"consensus_{key}"] = consensus.get(key)
            warnings = consensus.get("warnings")
            if isinstance(warnings, list):
                out["consensus_warnings"] = [str(w) for w in warnings[:6] if w]
        source_hosts = info.get("source_hosts")
        if isinstance(source_hosts, list):
            out["source_hosts"] = [str(h) for h in source_hosts[:6] if h]
        blocked_urls = info.get("blocked_urls")
        if isinstance(blocked_urls, list):
            out["blocked_urls"] = [str(u) for u in blocked_urls[:6] if u]
        return out

    if action in {"github_code_search"}:
        for key in ("query", "returned", "status_code"):
            if key in info and isinstance(info.get(key), (str, int, float, bool)):
                out[key] = info.get(key)
        urls = info.get("urls")
        if isinstance(urls, list):
            out["urls"] = [str(u) for u in urls[:5] if u]
        return out

    if action in {"github_repo_ingest"}:
        for key in ("repo", "ref", "repo_root", "blocked", "reason"):
            if key in info and isinstance(info.get(key), (str, int, float, bool)):
                out[key] = info.get(key)
        license_info = info.get("license")
        if isinstance(license_info, dict):
            for key in ("spdx", "copyleft", "confidence"):
                if key in license_info and isinstance(license_info.get(key), (str, int, float, bool)):
                    out[f"license_{key}"] = license_info.get(key)
        return out

    if action in {"run_script", "shell"}:
        for key in ("path", "returncode", "stdout_chars", "stderr_chars", "truncated", "sandboxed"):
            if key in info and isinstance(info.get(key), (str, int, float, bool)):
                out[key] = info.get(key)
        return out

    if action in {"web_scrape", "web_get"}:
        for key in ("host", "status_code", "content_type", "returned_chars", "truncated"):
            if key in info and isinstance(info.get(key), (str, int, float, bool)):
                out[key] = info.get(key)
        return out

    return out


def _summarize_external_references(references: List[Dict[str, Any]]) -> Dict[str, Any]:
    hosts: Dict[str, int] = defaultdict(int)
    low_hosts: Dict[str, int] = defaultdict(int)
    low_urls: List[str] = []
    total = 0
    low = 0

    for ref in references or []:
        if not isinstance(ref, dict):
            continue
        url = str(ref.get("url") or "").strip()
        if not url:
            continue
        if not url.startswith(("http://", "https://")):
            continue
        total += 1
        host = str(ref.get("host") or "").strip().lower().strip(".")
        if not host:
            host = _hostname_from_url(url)
        if host:
            hosts[host] += 1
        trust = str(ref.get("trust") or "").strip().lower()
        trust_score = ref.get("trust_score")
        blocked_domain = bool(ref.get("blocked_domain")) if "blocked_domain" in ref else False
        is_low = blocked_domain or trust == "low"
        if trust_score is not None:
            try:
                is_low = is_low or float(trust_score) < 0.4
            except Exception:
                pass
        if is_low:
            low += 1
            if host:
                low_hosts[host] += 1
            if len(low_urls) < 6:
                low_urls.append(url)

    top_low_hosts = sorted(low_hosts.items(), key=lambda item: (-item[1], item[0]))[:6]
    return {
        "total": int(total),
        "low": int(low),
        "unique_hosts": int(len(hosts)),
        "top_hosts": [h for h, _ in sorted(hosts.items(), key=lambda item: (-item[1], item[0]))[:6]],
        "top_low_hosts": [h for h, _ in top_low_hosts],
        "low_urls": low_urls,
    }


class _ToolBridgeCapabilityModel:
    """Provide a minimal `capabilities` mapping for backend introspection."""

    def __init__(self, tool_bridge: ToolEnvironmentBridge) -> None:
        snapshot: Dict[str, Any] = {}
        if hasattr(tool_bridge, "capability_snapshot") and callable(getattr(tool_bridge, "capability_snapshot")):
            try:
                snapshot = dict(tool_bridge.capability_snapshot())  # type: ignore[call-arg]
            except Exception:
                snapshot = {}
        enabled = snapshot.get("enabled_actions")
        if isinstance(enabled, dict):
            self._capabilities = {f"tool:{k}": (1.0 if bool(v) else 0.0) for k, v in enabled.items()}
        else:
            self._capabilities = {}

    @property
    def capabilities(self) -> Dict[str, float]:
        return dict(self._capabilities)


@dataclass(frozen=True)
class TaskStep:
    """One executable plan step."""

    index: int
    title: str
    action: Dict[str, Any]
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskPlan:
    """A concrete plan produced by a TaskPlanner."""

    goal: str
    steps: Tuple[TaskStep, ...]
    planner: str = "heuristic"
    raw: Any = None


@dataclass(frozen=True)
class ExecutionEvent:
    """Captured outcome for one executed step."""

    step_index: int
    step_title: str
    action: Dict[str, Any]
    observation: Dict[str, Any]
    reward: float
    terminated: bool
    info: Dict[str, Any]
    status: str
    latency_s: float
    timestamp: float


@dataclass(frozen=True)
class ExecutionReport:
    """Final report returned by the task executor."""

    goal: str
    plan: TaskPlan
    events: Tuple[ExecutionEvent, ...]
    success: bool
    blocked: bool
    error: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GovernorConfig:
    """Safety configuration for the execution governor."""

    enabled: bool = True
    confirm_token: Optional[str] = None
    require_confirmation_for: Tuple[str, ...] = (
        "delete_file",
        "modify_file",
        "exec_system_cmd",
        "change_system_setting",
        "launch_program",
        "kill_process",
        "docker",
        "docker_compose",
        "run_script",
        "remote_tool",
        "ui",
        "motor",
    )
    env_kill_switch: str = "BSS_AUTONOMOUS_TASK_DISABLE"


class ActionGovernor:
    """Pre-flight safety checks before executing an action."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        env_confirm = os.environ.get("BSS_AUTONOMOUS_TASK_CONFIRM_TOKEN")
        self.config = GovernorConfig(
            enabled=bool(cfg.get("enabled", True)),
            confirm_token=(cfg.get("confirm_token") or env_confirm or None),
            require_confirmation_for=tuple(cfg.get("require_confirmation_for") or GovernorConfig.require_confirmation_for),
            env_kill_switch=str(cfg.get("env_kill_switch", GovernorConfig.env_kill_switch) or GovernorConfig.env_kill_switch),
        )

    def check(self, action: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Return (allowed, info). When blocked, info contains reason fields."""

        if not self.config.enabled:
            return True, {"governor": "disabled"}

        if self.config.env_kill_switch and os.environ.get(self.config.env_kill_switch):
            return False, {"blocked": True, "reason": "autonomous_task_disabled", "kill_switch": self.config.env_kill_switch}

        action_type = str(action.get("type") or "").strip()
        if not action_type:
            return False, {"blocked": True, "reason": "missing_action_type"}

        requires_confirm = action_type in set(self.config.require_confirmation_for)
        # Power operations are always high-risk.
        if action_type == "change_system_setting":
            name = str(action.get("name") or "")
            if name in {"power.shutdown", "power.restart"}:
                requires_confirm = True

        if not requires_confirm:
            return True, {"governor": "ok"}

        expected = self.config.confirm_token
        # Confirmation token is distinct from action-specific auth tokens (e.g. remote_tool).
        provided = action.get("confirm_token")
        if expected and provided == expected:
            return True, {"governor": "confirmed", "requires_confirmation": True}

        return (
            False,
            {
                "blocked": True,
                "reason": "confirmation_required",
                "action_type": action_type,
                "requires_confirmation": True,
                "confirm_token_configured": bool(expected),
            },
        )


class TaskPlanner:
    """Planner interface."""

    def plan(self, goal: str, *, context: Optional[Dict[str, Any]] = None) -> TaskPlan:
        raise NotImplementedError

    def replan(
        self,
        goal: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        events: Sequence[ExecutionEvent] = (),
    ) -> TaskPlan:
        return self.plan(goal, context=context)


class HeuristicTaskPlanner(TaskPlanner):
    """Deterministic planner: parses explicit JSON plans or simple text patterns."""

    def plan(self, goal: str, *, context: Optional[Dict[str, Any]] = None) -> TaskPlan:
        goal_text = str(goal or "")
        parsed = _extract_plan_json(goal_text)
        if parsed is not None:
            return _normalize_plan(parsed, goal_text, planner="explicit_json")

        steps = _heuristic_steps_from_text(goal_text)
        if steps:
            return _normalize_plan({"goal": goal_text, "steps": [{"action": s} for s in steps]}, goal_text, planner="heuristic_rules")

        return TaskPlan(
            goal=goal_text,
            steps=tuple(),
            planner="heuristic_empty",
            raw={"needs_clarification": True},
        )


class LLMTaskPlanner(TaskPlanner):
    """Optional planner using LLMService. Falls back to HeuristicTaskPlanner."""

    def __init__(self, llm: Any, *, fallback: Optional[TaskPlanner] = None) -> None:
        self.llm = llm
        self.fallback = fallback or HeuristicTaskPlanner()

    def _build_system_prompt(self, *, context: Optional[Dict[str, Any]]) -> str:
        strategy = context.get("strategy") if isinstance(context, dict) else None
        prompt_cfg = strategy.get("prompt") if isinstance(strategy, dict) else {}
        planner_cfg = strategy.get("planner") if isinstance(strategy, dict) else {}

        variant_raw = prompt_cfg.get("variant") if isinstance(prompt_cfg, dict) else 0
        try:
            variant = int(variant_raw)
        except Exception:
            try:
                variant = int(round(float(variant_raw)))
            except Exception:
                variant = 0
        variant = max(0, variant)
        json_strictness = _safe_unit(prompt_cfg.get("json_strictness"))
        safety_bias = _safe_unit(prompt_cfg.get("safety_bias"))
        structured = bool(planner_cfg.get("structured", True)) if isinstance(planner_cfg, dict) else True

        strict_line = ""
        if json_strictness is not None and json_strictness >= 0.7:
            strict_line = (
                "Output ONLY a valid JSON object (no markdown/code fences, no extra keys).\n"
            )
        elif json_strictness is not None and json_strictness >= 0.4:
            strict_line = "Prefer strict JSON output; avoid extra text.\n"

        safety_line = ""
        if safety_bias is not None and safety_bias >= 0.7:
            safety_line = (
                "Safety: Prefer read-only actions (read_file/list_dir). Avoid destructive/system actions unless explicitly confirmed.\n"
            )
        elif safety_bias is not None and safety_bias >= 0.4:
            safety_line = "Safety: Avoid destructive/system actions unless necessary and confirmed.\n"

        structure_line = ""
        if structured:
            structure_line = "Each step should include a short 'title' and an 'action' dict.\n"

        base = (
            "You are a task planning module. Convert the user goal into a JSON object with the key 'steps'.\n"
            "Each step must be an object with an 'action' field, where action is a ToolEnvironmentBridge action dict.\n"
            "Only include actions that are necessary and safe. If unsure, return an empty steps list.\n"
            "If you need implementation references, prefer local search via code_index_build/code_index_search.\n"
            "If context contains knowledge_acquisition (retrieval_context/references), use it to refine the plan.\n"
        )

        if variant <= 0:
            return strict_line + safety_line + structure_line + base
        if variant == 1:
            return (
                strict_line
                + safety_line
                + structure_line
                + base
                + "Before outputting JSON, reason internally about the minimal safe plan.\n"
                + "Do NOT include your reasoning in the output.\n"
                + "Schema example:\n"
                + "{\"steps\": [{\"title\": \"...\", \"action\": {\"type\": \"read_file\", \"path\": \"...\"}}]}\n"
            )
        # variant >= 2
        return (
            strict_line
            + safety_line
            + structure_line
            + base
            + "Hard constraints:\n"
            + "- Never request dangerous actions unless the goal explicitly demands it.\n"
            + "- Prefer plans that can be completed in <= 5 steps.\n"
            + "- If an action needs confirmation, omit it unless a confirm_token is provided.\n"
        )

    def plan(self, goal: str, *, context: Optional[Dict[str, Any]] = None) -> TaskPlan:
        # Always allow an explicit plan embedded by the user.
        fallback_plan = self.fallback.plan(goal, context=context)
        if fallback_plan.steps:
            return fallback_plan

        if self.llm is None:
            return fallback_plan

        prompt_context = dict(context or {})
        prompt_context.setdefault("safety", "Prefer read-only actions. Avoid destructive/system-changing actions unless explicitly confirmed.")

        instructions = {"system": self._build_system_prompt(context=prompt_context)}
        inputs = {"user": str(goal or ""), "context": prompt_context}

        try:
            response = self.llm.structured_chat(instructions, inputs)
        except Exception:
            return fallback_plan

        if response is None:
            return fallback_plan

        parsed = response
        if isinstance(response, dict) and "steps" not in response:
            # Some providers wrap the JSON as text.
            text = response.get("text")
            if isinstance(text, str):
                parsed = _try_parse_json(text) or response

        plan = _normalize_plan(parsed, str(goal or ""), planner="llm") if parsed is not None else fallback_plan
        return plan if plan.steps else fallback_plan


class AutonomousTaskExecutor:
    """Execute a planned task against ToolEnvironmentBridge with a feedback loop."""

    def __init__(
        self,
        tool_bridge: ToolEnvironmentBridge,
        *,
        planner: Optional[TaskPlanner] = None,
        governor: Optional[ActionGovernor] = None,
        security_manager: Optional["SecurityManager"] = None,
        motor_control: Any = None,
        self_improvement: Any = None,
        knowledge_consolidator: Any = None,
        metrics_collector: Any = None,
        evolution_target: Any = None,
        introspection: Any = None,
        introspection_config: Optional[Dict[str, Any]] = None,
        max_steps: int = 32,
        max_replans: int = 1,
    ) -> None:
        self.tool_bridge = tool_bridge
        self.planner = planner or HeuristicTaskPlanner()
        self.governor = governor or ActionGovernor()
        self.security_manager = security_manager
        self.motor_control = motor_control
        self.self_improvement = self_improvement
        self.knowledge_consolidator = knowledge_consolidator
        self.metrics_collector = metrics_collector
        self.evolution_target = evolution_target

        cfg = dict(introspection_config or {})
        self._introspection_enabled = bool(cfg.get("enabled", True))
        self._introspection_failure_streak_threshold = max(1, int(cfg.get("failure_streak_threshold", 2)))
        self._introspection_uncertainty_threshold = float(cfg.get("uncertainty_threshold", 0.75))
        self._introspection_skill_review_limit = max(0, int(cfg.get("skill_review_limit", 24)))
        self._introspection_ability_limit = max(0, int(cfg.get("ability_limit", 12)))
        self._introspection_attach_event_info = bool(cfg.get("attach_event_info", True))
        self._introspection_explain_max_plan_steps = max(0, int(cfg.get("explain_max_plan_steps", 5)))
        self._introspection_explain_max_skill_suggestions = max(0, int(cfg.get("explain_max_skill_suggestions", 6)))

        self.introspection = introspection
        if self.introspection is None and self._introspection_enabled and IntrospectionInterface is not None:
            try:
                self.introspection = IntrospectionInterface(self_model=_ToolBridgeCapabilityModel(tool_bridge))
            except Exception:
                self.introspection = None
        self.max_steps = max(0, int(max_steps))
        self.max_replans = max(0, int(max_replans))
        self._approval_cache: Dict[str, str] = {}
        self._meta_retrieval_policy: Any = None

    def run(self, goal: str, *, context: Optional[Dict[str, Any]] = None) -> ExecutionReport:
        ctx: Dict[str, Any] = dict(context or {})
        events: List[ExecutionEvent] = []
        report_meta: Dict[str, Any] = {}
        metric_events: List[Any] = []

        improver = self.self_improvement
        if improver is None and AgentSelfImprovementController is not None:
            # Default to an in-memory improver when available; keep it opt-in via env.
            if os.environ.get("BSS_SELF_IMPROVEMENT_ENABLED") in {"1", "true", "yes", "on"}:
                try:
                    improver = AgentSelfImprovementController()
                except Exception:
                    improver = None
                self.self_improvement = improver
        if improver is not None and hasattr(improver, "strategy_context"):
            try:
                ctx.setdefault("strategy", improver.strategy_context())
                report_meta.setdefault("self_improvement", {})["strategy"] = ctx.get("strategy")
            except Exception:
                pass

        intro = self.introspection if self._introspection_enabled else None
        if intro is not None:
            try:
                abilities = intro.summarize_my_abilities(max_items=self._introspection_ability_limit, as_text=False)
            except Exception:
                abilities = {"summary": "abilities unavailable", "abilities": [], "returned": 0, "total": 0}
            tool_snapshot: Dict[str, Any] = {}
            if hasattr(self.tool_bridge, "capability_snapshot") and callable(getattr(self.tool_bridge, "capability_snapshot")):
                try:
                    tool_snapshot = dict(self.tool_bridge.capability_snapshot())  # type: ignore[call-arg]
                except Exception:
                    tool_snapshot = {}
            intro_payload = {"abilities": abilities, "tool_bridge": tool_snapshot}
            ctx.setdefault("introspection", {}).update(intro_payload)
            report_meta["introspection"] = intro_payload
            self._audit_event("introspection.plan_start", {"goal": str(goal or ""), "introspection": intro_payload})

        plan = self.planner.plan(goal, context=ctx)
        replans_used = 0
        failure_streak = 0

        if not plan.steps:
            self._finalize_learning(
                report_meta,
                ctx=ctx,
                goal=plan.goal,
                success=False,
                events=events,
                blocked=False,
                error="no_plan_steps",
            )
            return ExecutionReport(
                goal=plan.goal,
                plan=plan,
                events=tuple(),
                success=False,
                blocked=False,
                error="no_plan_steps",
                meta={**report_meta, "replans": replans_used, "needs_clarification": True},
            )

        step_limit = len(plan.steps) if self.max_steps <= 0 else min(len(plan.steps), self.max_steps)

        idx = 0
        while idx < step_limit:
            step = plan.steps[idx]
            action = dict(step.action)
            step_introspection: Dict[str, Any] | None = None
            if intro is not None:
                try:
                    step_introspection = intro.explain_my_plan(
                        {"goal": plan.goal, "step": step.title, "action": action},
                        max_plan_steps=self._introspection_explain_max_plan_steps,
                        max_skill_suggestions=self._introspection_explain_max_skill_suggestions,
                        as_text=False,
                    )
                except Exception:
                    step_introspection = None
                if isinstance(step_introspection, dict):
                    expected = str(step_introspection.get("expected") or "").strip()
                    verification = str(step_introspection.get("verification") or "").strip()
                    rationale = str(step_introspection.get("rationale") or "").strip()
                    compact = "\n".join(
                        chunk
                        for chunk in (
                            f"Expected: {expected}" if expected else "",
                            f"Verify: {verification}" if verification else "",
                            f"Rationale: {rationale}" if rationale else "",
                        )
                        if chunk
                    )
                    ctx.setdefault("introspection", {})["last_step_explanation"] = step_introspection
                    if compact:
                        ctx.setdefault("introspection", {})["last_step_explanation_text"] = _clip_text(
                            compact, max_chars=1536
                        )

            fp = action_fingerprint(action) if callable(action_fingerprint) else None
            if fp and "approval_id" not in action and fp in self._approval_cache:
                action["approval_id"] = self._approval_cache[fp]

            if self.security_manager is not None:
                decision_context = {"goal": str(goal or ""), "step": step.index, "title": step.title}
                if isinstance(step_introspection, dict):
                    decision_context["introspection"] = {
                        "expected": step_introspection.get("expected"),
                        "verification": step_introspection.get("verification"),
                        "rationale": step_introspection.get("rationale"),
                        "skills": (step_introspection.get("skills") or {}).get("suggested")
                        if isinstance(step_introspection.get("skills"), dict)
                        else None,
                    }
                decision = self.security_manager.decide(action, context=decision_context)
                if fp and decision.approval_id:
                    self._approval_cache[fp] = decision.approval_id
                if decision.blocked:
                    info_out = decision.as_info()
                    if self._introspection_attach_event_info and isinstance(step_introspection, dict):
                        info_out = {**info_out, "introspection": step_introspection}
                    info_out.setdefault("latency_s", 0.0)
                    if self.metrics_collector is not None and callable(getattr(self.metrics_collector, "emit_event", None)):
                        try:
                            self.metrics_collector.emit_event(
                                str(action.get("type") or "unknown"),
                                latency=0.0,
                                energy=0.0,
                                throughput=0.0,
                                status="blocked",
                                confidence=None,
                                stage=str(plan.goal or "autonomous_task"),
                                metadata={"blocked": True, "reason": str(decision.reason or "blocked")},
                            )
                        except Exception:
                            pass
                    blocked_event = ExecutionEvent(
                        step_index=step.index,
                        step_title=step.title,
                        action=action,
                        observation={"text": ""},
                        reward=-1.0,
                        terminated=False,
                        info=info_out,
                        status="blocked",
                        latency_s=0.0,
                        timestamp=time.time(),
                    )
                    events.append(blocked_event)
                    self._audit_step(
                        plan.goal,
                        action,
                        step=step,
                        status="blocked",
                        reward=-1.0,
                        terminated=False,
                        info=info_out,
                        observation={"text": ""},
                        introspection=step_introspection,
                    )
                    self._finalize_learning(
                        report_meta,
                        ctx=ctx,
                        goal=plan.goal,
                        success=False,
                        events=events,
                        blocked=True,
                        error=str(decision.reason or "blocked"),
                    )
                    return ExecutionReport(
                        goal=plan.goal,
                        plan=plan,
                        events=tuple(events),
                        success=False,
                        blocked=True,
                        error=str(decision.reason or "blocked"),
                        meta={**report_meta, "replans": replans_used},
                    )
            else:
                allowed, gov_info = self.governor.check(action)
                if not allowed:
                    info_out = dict(gov_info)
                    if self._introspection_attach_event_info and isinstance(step_introspection, dict):
                        info_out["introspection"] = step_introspection
                    info_out.setdefault("latency_s", 0.0)
                    if self.metrics_collector is not None and callable(getattr(self.metrics_collector, "emit_event", None)):
                        try:
                            self.metrics_collector.emit_event(
                                str(action.get("type") or "unknown"),
                                latency=0.0,
                                energy=0.0,
                                throughput=0.0,
                                status="blocked",
                                confidence=None,
                                stage=str(plan.goal or "autonomous_task"),
                                metadata={"blocked": True, "reason": str(gov_info.get("reason") or "blocked")},
                            )
                        except Exception:
                            pass
                    blocked_event = ExecutionEvent(
                        step_index=step.index,
                        step_title=step.title,
                        action=action,
                        observation={"text": ""},
                        reward=-1.0,
                        terminated=False,
                        info=info_out,
                        status="blocked",
                        latency_s=0.0,
                        timestamp=time.time(),
                    )
                    events.append(blocked_event)
                    self._audit_step(
                        plan.goal,
                        action,
                        step=step,
                        status="blocked",
                        reward=-1.0,
                        terminated=False,
                        info=info_out,
                        observation={"text": ""},
                        introspection=step_introspection,
                    )
                    self._finalize_learning(
                        report_meta,
                        ctx=ctx,
                        goal=plan.goal,
                        success=False,
                        events=events,
                        blocked=True,
                        error=str(gov_info.get("reason") or "blocked"),
                    )
                    return ExecutionReport(
                        goal=plan.goal,
                        plan=plan,
                        events=tuple(events),
                        success=False,
                        blocked=True,
                        error=str(gov_info.get("reason") or "blocked"),
                        meta={**report_meta, "replans": replans_used},
                    )

            step_start = time.perf_counter()
            observation, reward, terminated, info = self._execute_action(action)
            latency_s = max(0.0, time.perf_counter() - step_start)
            status = _status_from_result(reward, info)
            info_out = dict(info or {})
            if "latency_s" not in info_out:
                info_out["latency_s"] = float(latency_s)
            if self._introspection_attach_event_info and isinstance(step_introspection, dict):
                info_out["introspection"] = step_introspection
            event = ExecutionEvent(
                step_index=step.index,
                step_title=step.title,
                action=action,
                observation=observation,
                reward=float(reward),
                terminated=bool(terminated),
                info=info_out,
                status=status,
                latency_s=float(latency_s),
                timestamp=time.time(),
            )
            events.append(event)
            self._audit_step(
                plan.goal,
                action,
                step=step,
                status=status,
                reward=reward,
                terminated=terminated,
                info=info_out,
                observation=observation,
                introspection=step_introspection,
            )

            ctx.setdefault("execution_events", []).append(
                {
                    "step": step.index,
                    "title": step.title,
                    "action": action,
                    "status": status,
                    "reward": float(reward),
                    "latency_s": float(latency_s),
                    "info": dict(info_out or {}),
                }
            )

            if MetricEvent is not None:
                try:
                    action_type = str(action.get("type") or "unknown")
                    throughput = 0.0 if latency_s <= 0 else 1.0 / float(latency_s)
                    uncertainty = _extract_uncertainty(info_out)
                    confidence = None
                    if uncertainty is not None:
                        confidence = max(0.0, min(1.0, 1.0 - float(uncertainty)))
                    tool_meta = _tool_info_for_metrics(action_type, info if isinstance(info, dict) else {})
                    metric_events.append(
                        MetricEvent(
                            module=action_type,
                            latency=float(latency_s),
                            energy=0.0,
                            throughput=float(throughput),
                            timestamp=time.time(),
                            status=str(status),
                            confidence=confidence,
                            stage=str(plan.goal or "autonomous_task"),
                            metadata={
                                "goal": str(plan.goal or ""),
                                "step_index": int(step.index),
                                "reward": float(reward),
                                "tool": tool_meta,
                            },
                        )
                    )
                except Exception:
                    pass
            if self.metrics_collector is not None and callable(getattr(self.metrics_collector, "emit_event", None)):
                try:
                    action_type = str(action.get("type") or "unknown")
                    throughput = 0.0 if latency_s <= 0 else 1.0 / float(latency_s)
                    uncertainty = _extract_uncertainty(info_out)
                    confidence = None
                    if uncertainty is not None:
                        confidence = max(0.0, min(1.0, 1.0 - float(uncertainty)))
                    tool_meta = _tool_info_for_metrics(action_type, info if isinstance(info, dict) else {})
                    self.metrics_collector.emit_event(
                        action_type,
                        latency=float(latency_s),
                        energy=0.0,
                        throughput=float(throughput),
                        status=str(status),
                        confidence=confidence,
                        stage=str(plan.goal or "autonomous_task"),
                        metadata={
                            "goal": str(plan.goal or ""),
                            "step_index": int(step.index),
                            "reward": float(reward),
                            "terminated": 1.0 if bool(terminated) else 0.0,
                            "tool": tool_meta,
                        },
                    )
                except Exception:
                    pass

            if improver is not None and metric_events and status != "success" and hasattr(improver, "observe"):
                try:
                    update = improver.observe(
                        [metric_events[-1]],
                        extra={
                            "goal": str(plan.goal or ""),
                            "step": int(step.index),
                            "action_type": str(action.get("type") or ""),
                            "status": str(status),
                            "reward": float(reward),
                        },
                    )
                    if update is not None:
                        report_meta.setdefault("self_improvement", {})["last_update"] = {
                            "version": getattr(update, "version", None),
                            "source": getattr(update, "source", None),
                            "genes": getattr(update, "genes", None),
                        }
                        if hasattr(improver, "strategy_context"):
                            ctx["strategy"] = improver.strategy_context()
                            report_meta.setdefault("self_improvement", {})["strategy"] = ctx.get("strategy")
                        target = self.evolution_target
                        if target is not None and hasattr(improver, "apply_to_architecture"):
                            try:
                                improver.apply_to_architecture(
                                    target,
                                    performance=None,
                                    metrics={
                                        "resource_score": float(reward),
                                        "avg_latency": float(latency_s),
                                        "avg_throughput": 0.0 if float(latency_s) <= 0 else 1.0 / float(latency_s),
                                    },
                                )
                                report_meta.setdefault("self_improvement", {})["applied_to_architecture"] = True
                            except Exception:
                                report_meta.setdefault("self_improvement", {})["applied_to_architecture"] = False
                except Exception:
                    pass

            if status != "success":
                failure_streak += 1
                uncertainty = _extract_uncertainty(info_out)
                review_skills = failure_streak >= self._introspection_failure_streak_threshold or (
                    uncertainty is not None and uncertainty >= self._introspection_uncertainty_threshold
                )
                if review_skills and intro is not None:
                    try:
                        skills = intro.get_loaded_skills(limit=self._introspection_skill_review_limit, as_text=False)
                    except Exception:
                        skills = {
                            "skills": [],
                            "returned": 0,
                            "total": 0,
                            "truncated": False,
                            "note": "skill review failed",
                        }
                    ctx.setdefault("introspection", {})["skills_review"] = {
                        "trigger": "failure_streak"
                        if failure_streak >= self._introspection_failure_streak_threshold
                        else "uncertainty",
                        "failure_streak": int(failure_streak),
                        "uncertainty": uncertainty,
                        "skills": skills,
                    }
                    report_meta.setdefault("introspection", {}).update(
                        {"skills_review": ctx["introspection"]["skills_review"]}
                    )
                    self._audit_event(
                        "introspection.skill_review",
                        {
                            "goal": str(goal or ""),
                            "step": step.index,
                            "trigger": ctx["introspection"]["skills_review"]["trigger"],
                            "failure_streak": int(failure_streak),
                            "uncertainty": uncertainty,
                            "skills_total": int(skills.get("total", 0)) if isinstance(skills, dict) else 0,
                        },
                    )
                if replans_used < self.max_replans:
                    self._maybe_acquire_knowledge(
                        goal=str(goal or ""),
                        ctx=ctx,
                        report_meta=report_meta,
                        step=step,
                        action=action,
                        info=info_out,
                    )
                    replans_used += 1
                    ctx["last_failure"] = {
                        "step": step.index,
                        "title": step.title,
                        "action": action,
                        "info": dict(info_out or {}),
                    }
                    if intro is not None:
                        try:
                            abilities = intro.summarize_my_abilities(
                                max_items=self._introspection_ability_limit, as_text=False
                            )
                            ctx.setdefault("introspection", {})["abilities"] = abilities
                            report_meta.setdefault("introspection", {}).update({"abilities": abilities})
                        except Exception:
                            pass
                    plan = self.planner.replan(goal, context=ctx, events=events)
                    if not plan.steps:
                        self._finalize_learning(
                            report_meta,
                            ctx=ctx,
                            goal=plan.goal,
                            success=False,
                            events=events,
                            blocked=False,
                            error="replan_failed",
                        )
                        return ExecutionReport(
                            goal=str(goal or ""),
                            plan=plan,
                            events=tuple(events),
                            success=False,
                            blocked=False,
                            error="replan_failed",
                            meta={**report_meta, "replans": replans_used},
                        )
                    step_limit = len(plan.steps) if self.max_steps <= 0 else min(len(plan.steps), self.max_steps)
                    idx = 0
                    continue

                self._finalize_learning(
                    report_meta,
                    ctx=ctx,
                    goal=plan.goal,
                    success=False,
                    events=events,
                    blocked=bool(info_out.get("blocked")),
                    error=str(info_out.get("error") or info_out.get("reason") or "step_failed"),
                )
                return ExecutionReport(
                    goal=plan.goal,
                    plan=plan,
                    events=tuple(events),
                    success=False,
                    blocked=bool(info_out.get("blocked")),
                    error=str(info_out.get("error") or info_out.get("reason") or "step_failed"),
                    meta={**report_meta, "replans": replans_used},
                )

            failure_streak = 0
            if terminated:
                break

            idx += 1

        success = all(event.status == "success" for event in events)
        self._finalize_learning(
            report_meta,
            ctx=ctx,
            goal=plan.goal,
            success=success,
            events=events,
            blocked=False,
            error=None if success else "incomplete",
        )
        return ExecutionReport(
            goal=plan.goal,
            plan=plan,
            events=tuple(events),
            success=success,
            blocked=False,
            error=None if success else "incomplete",
            meta={**report_meta, "replans": replans_used},
        )

    def _finalize_learning(
        self,
        report_meta: Dict[str, Any],
        *,
        ctx: Dict[str, Any],
        goal: str,
        success: bool,
        events: Sequence[ExecutionEvent],
        blocked: bool,
        error: str | None,
    ) -> None:
        """Apply post-run learning hooks (meta-policy + knowledge consolidation)."""

        self._maybe_observe_meta_retrieval_policy(report_meta, success=success, goal=goal)
        self._maybe_consolidate_external_knowledge(
            report_meta,
            ctx=ctx,
            goal=goal,
            success=success,
            events=events,
            blocked=blocked,
            error=error,
        )

    def _maybe_consolidate_external_knowledge(
        self,
        report_meta: Dict[str, Any],
        *,
        ctx: Dict[str, Any],
        goal: str,
        success: bool,
        events: Sequence[ExecutionEvent],
        blocked: bool,
        error: str | None,
    ) -> None:
        cfg: Dict[str, Any] = {}
        raw_cfg = ctx.get("knowledge_consolidation")
        if isinstance(raw_cfg, dict):
            cfg.update(raw_cfg)

        enabled: bool | None = cfg.get("enabled")
        if enabled is None:
            enabled = _env_flag("BSS_KNOWLEDGE_CONSOLIDATION_ENABLED")
        if not enabled:
            return

        ka = report_meta.get("knowledge_acquisition")
        if not isinstance(ka, list) or not ka:
            return

        consolidator = getattr(self, "knowledge_consolidator", None)
        if consolidator is None and ExternalKnowledgeConsolidator is not None:
            config_obj = None
            if KnowledgeConsolidationConfig is not None:
                try:
                    max_summary = int(cfg.get("max_summary_chars", KnowledgeConsolidationConfig.max_summary_chars))
                except Exception:
                    max_summary = KnowledgeConsolidationConfig.max_summary_chars
                try:
                    max_context = int(cfg.get("max_context_chars", KnowledgeConsolidationConfig.max_context_chars))
                except Exception:
                    max_context = KnowledgeConsolidationConfig.max_context_chars
                try:
                    max_refs = int(cfg.get("max_references", KnowledgeConsolidationConfig.max_references))
                except Exception:
                    max_refs = KnowledgeConsolidationConfig.max_references
                try:
                    max_channels = int(cfg.get("max_channels", KnowledgeConsolidationConfig.max_channels))
                except Exception:
                    max_channels = KnowledgeConsolidationConfig.max_channels

                config_obj = KnowledgeConsolidationConfig(
                    enabled=True,
                    max_summary_chars=max(500, min(max_summary, 50_000)),
                    max_context_chars=max(200, min(max_context, 50_000)),
                    max_references=max(1, min(max_refs, 200)),
                    max_channels=max(1, min(max_channels, 50)),
                    ingest_graph=bool(cfg.get("ingest_graph", True)),
                    store_vector_summary=bool(cfg.get("store_vector_summary", True)),
                )
            try:
                consolidator = ExternalKnowledgeConsolidator(config=config_obj)
            except Exception:
                consolidator = None
            self.knowledge_consolidator = consolidator

        if consolidator is None or not hasattr(consolidator, "consolidate"):
            return

        human_reward: float | None = None
        hf = report_meta.get("human_feedback")
        if isinstance(hf, dict) and hf.get("reward") is not None:
            try:
                human_reward = float(hf.get("reward"))
            except Exception:
                human_reward = None

        total_reward = 0.0
        for ev in events:
            try:
                total_reward += float(getattr(ev, "reward", 0.0))
            except Exception:
                pass

        task_meta = {
            "blocked": bool(blocked),
            "error": str(error) if error else None,
            "steps": int(len(events)),
            "total_reward": float(total_reward),
        }

        try:
            result = consolidator.consolidate(
                goal=str(goal or ""),
                knowledge_acquisition=ka,
                success=bool(success),
                human_reward=human_reward,
                task_metadata=task_meta,
                source="autonomous_task",
            )
        except Exception:
            return

        if isinstance(result, dict) and result:
            report_meta["knowledge_consolidation"] = result

    def _maybe_observe_meta_retrieval_policy(
        self,
        report_meta: Dict[str, Any],
        *,
        success: bool,
        goal: str | None = None,
    ) -> None:
        policy = getattr(self, "_meta_retrieval_policy", None)
        if policy is None:
            return
        ka = report_meta.get("knowledge_acquisition")
        if not isinstance(ka, list) or not ka:
            return

        feedback_reward: float | None = None
        goal_key = str(goal or "").strip()
        if goal_key and self.metrics_collector is not None and callable(getattr(self.metrics_collector, "events", None)):
            try:
                all_events = list(self.metrics_collector.events())
            except Exception:
                all_events = []
            for event in reversed(all_events):
                try:
                    module = str(getattr(event, "module", "") or "")
                    stage = str(getattr(event, "stage", "") or "")
                except Exception:
                    continue
                if module != "human_feedback":
                    continue
                if stage and stage != goal_key:
                    continue
                conf = getattr(event, "confidence", None)
                if conf is not None:
                    try:
                        feedback_reward = max(0.0, min(1.0, float(conf)))
                    except Exception:
                        feedback_reward = None
                meta = getattr(event, "metadata", None)
                if isinstance(meta, dict) and meta:
                    report_meta.setdefault("human_feedback", {})["metadata"] = dict(meta)
                if feedback_reward is not None:
                    report_meta.setdefault("human_feedback", {})["reward"] = float(feedback_reward)
                break

        try:
            for payload in ka:
                if not isinstance(payload, dict):
                    continue
                meta = payload.get("meta_policy")
                if not isinstance(meta, dict):
                    continue
                domain = str(meta.get("domain") or "").strip() or "general"
                channels = payload.get("channels_used")
                if not isinstance(channels, list):
                    channels = meta.get("channels") if isinstance(meta.get("channels"), list) else []
                channels_list = [str(c or "").strip() for c in channels if c]
                if not channels_list:
                    continue
                observe = getattr(policy, "observe", None)
                if callable(observe):
                    observe(domain=domain, channels=channels_list, success=bool(success), reward=feedback_reward)
        except Exception:
            return

    def _execute_action(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        action_type = str(action.get("type") or "").strip().lower()
        if action_type in {"ui", "motor"}:
            if self.motor_control is None:
                return {"text": ""}, -1.0, False, {"error": "motor_control_not_configured"}

            intention: Any
            if action_type == "ui":
                actions = action.get("actions") or action.get("ui_actions") or []
                if isinstance(actions, dict):
                    actions = [actions]
                if not isinstance(actions, list):
                    actions = []
                intention = {"ui_actions": actions}
            else:
                intention = action.get("intention") or action.get("payload") or {}
            try:
                output = self.motor_control.compute(intention)
            except Exception as exc:  # pragma: no cover - defensive
                return {"text": ""}, -1.0, False, {"error": "motor_control_failed", "exception": repr(exc)}

            ui_out = output.get("ui") if isinstance(output, dict) else None
            if isinstance(ui_out, dict) and ui_out.get("error"):
                return {"text": "", "motor_output": output}, -1.0, False, {"error": "ui_action_failed", "ui": ui_out}
            return {"text": "", "motor_output": output}, 0.2, False, {"motor": True}

        return self.tool_bridge.step(action)

    def _maybe_acquire_knowledge(
        self,
        *,
        goal: str,
        ctx: Dict[str, Any],
        report_meta: Dict[str, Any],
        step: TaskStep,
        action: Dict[str, Any],
        info: Dict[str, Any],
    ) -> None:
        cfg: Dict[str, Any] = {}
        if isinstance(ctx.get("knowledge_acquisition"), dict):
            cfg.update(ctx["knowledge_acquisition"])
        strategy = ctx.get("strategy") if isinstance(ctx.get("strategy"), dict) else None
        if isinstance(strategy, dict) and isinstance(strategy.get("knowledge_acquisition"), dict):
            cfg = {**cfg, **strategy["knowledge_acquisition"]}

        enabled: bool | None = cfg.get("enabled")
        if enabled is None:
            enabled = _env_flag("BSS_KNOWLEDGE_ACQ_ENABLED")
        if not enabled:
            return

        allowed_roots: List[str] = []
        tool_snapshot: Dict[str, Any] | None = None
        if isinstance(ctx.get("introspection"), dict) and isinstance(ctx["introspection"].get("tool_bridge"), dict):
            tool_snapshot = ctx["introspection"]["tool_bridge"]
        if tool_snapshot is None:
            if hasattr(self.tool_bridge, "capability_snapshot") and callable(getattr(self.tool_bridge, "capability_snapshot")):
                try:
                    tool_snapshot = dict(self.tool_bridge.capability_snapshot())  # type: ignore[call-arg]
                except Exception:
                    tool_snapshot = None
        if isinstance(tool_snapshot, dict):
            roots = tool_snapshot.get("constraints", {}).get("allowed_roots")
            if isinstance(roots, list):
                allowed_roots = [str(r) for r in roots if r]
        enabled_actions: Dict[str, Any] = {}
        if isinstance(tool_snapshot, dict) and isinstance(tool_snapshot.get("enabled_actions"), dict):
            enabled_actions = dict(tool_snapshot.get("enabled_actions") or {})

        meta_enabled: bool | None = cfg.get("meta_retrieval_enabled")
        if meta_enabled is None:
            meta_enabled = _env_flag("BSS_META_RETRIEVAL_ENABLED")

        meta_policy_info: Dict[str, Any] | None = None
        if meta_enabled and MetaRetrievalPolicy is not None:
            if getattr(self, "_meta_retrieval_policy", None) is None:
                try:
                    if MetaRetrievalPolicyConfig is None:
                        self._meta_retrieval_policy = MetaRetrievalPolicy()  # type: ignore[call-arg]
                    else:
                        from pathlib import Path

                        raw_state_path = (
                            cfg.get("meta_state_path")
                            or cfg.get("meta_state")
                            or os.environ.get("BSS_META_RETRIEVAL_STATE_PATH")
                            or os.environ.get("BSS_META_RETRIEVAL_STATE")
                        )
                        if raw_state_path:
                            state_path = Path(str(raw_state_path)).expanduser()
                        else:
                            state_path = Path(__file__).resolve().parents[2] / "data" / "meta_retrieval_policy.json"

                        top_channels = max(1, min(int(cfg.get("meta_top_channels", 2)), 4))
                        allow_github = bool(
                            cfg.get(
                                "meta_allow_github_code_search",
                                _env_flag("BSS_META_RETRIEVAL_ALLOW_GITHUB_CODE_SEARCH"),
                            )
                        )
                        cfg_obj = MetaRetrievalPolicyConfig(  # type: ignore[misc]
                            state_path=state_path,
                            save_on_update=True,
                            top_channels=top_channels,
                            allow_github_code_search=allow_github,
                        )
                        self._meta_retrieval_policy = MetaRetrievalPolicy(cfg_obj)  # type: ignore[call-arg]
                except Exception:
                    self._meta_retrieval_policy = None

            policy = getattr(self, "_meta_retrieval_policy", None)
            if policy is not None and hasattr(policy, "suggest"):
                try:
                    error = str(info.get("error") or info.get("reason") or "").strip()
                    action_type = str(action.get("type") or "").strip()
                    task_text = "\n".join(part for part in (goal, f"Failure: {error}" if error else "", f"Action: {action_type}" if action_type else "", f"Step: {step.title}" if step.title else "") if part)
                    has_roots = bool(
                        cfg.get("code_roots")
                        or cfg.get("roots")
                        or cfg.get("code_root")
                        or cfg.get("root")
                        or allowed_roots
                    )
                    meta_policy_info = policy.suggest(  # type: ignore[call-arg]
                        task_text=task_text,
                        enabled_actions=enabled_actions,
                        has_local_roots=has_roots,
                    )
                    patch = meta_policy_info.get("config_patch") if isinstance(meta_policy_info, dict) else None
                    if isinstance(patch, dict):
                        for key, value in patch.items():
                            cfg.setdefault(key, value)
                except Exception:
                    meta_policy_info = None

        payload: Dict[str, Any] = {}
        try:
            payload = self._acquire_knowledge_payload(
                goal=goal,
                cfg=cfg,
                step=step,
                action=action,
                info=info,
                allowed_roots=allowed_roots,
            )
        except Exception:
            payload = {}

        if not payload:
            return

        if isinstance(meta_policy_info, dict) and meta_policy_info:
            payload.setdefault("meta_policy", meta_policy_info)

        ctx.setdefault("knowledge_acquisition", {})["last"] = payload
        ctx.setdefault("knowledge_acquisition", {})["references"] = payload.get("references") or []
        ctx.setdefault("knowledge_acquisition", {})["retrieval_context"] = payload.get("retrieval_context") or ""
        report_meta.setdefault("knowledge_acquisition", []).append(payload)

        # ------------------------------------------------------------------ telemetry + monitoring
        # Emit a compact, structured event for ops/debugging and optionally
        # quarantine web access if the agent keeps pulling low-trust sources.
        references = payload.get("references") if isinstance(payload, dict) else None
        refs_list = list(references) if isinstance(references, list) else []
        ref_summary = _summarize_external_references([dict(r) for r in refs_list if isinstance(r, dict)])

        web_info = payload.get("web_search", {}).get("info") if isinstance(payload.get("web_search"), dict) else None
        web_avg_trust = None
        web_low_count = None
        if isinstance(web_info, dict):
            try:
                web_avg_trust = float(web_info.get("avg_trust")) if web_info.get("avg_trust") is not None else None
            except Exception:
                web_avg_trust = None
            trust_counts = web_info.get("trust_counts")
            if isinstance(trust_counts, dict):
                try:
                    web_low_count = int(trust_counts.get("low", 0))
                except Exception:
                    web_low_count = None

        doc_info = payload.get("web", {}).get("info") if isinstance(payload.get("web"), dict) else None
        doc_consensus = doc_info.get("consensus") if isinstance(doc_info, dict) else None
        doc_level = str(doc_consensus.get("level") or "") if isinstance(doc_consensus, dict) else ""
        doc_needs_verification = bool(doc_consensus.get("needs_verification")) if isinstance(doc_consensus, dict) else False

        total_refs = int(ref_summary.get("total", 0) or 0)
        low_refs = int(ref_summary.get("low", 0) or 0)
        low_ratio = (low_refs / total_refs) if total_refs > 0 else 0.0
        needs_verification = bool(doc_needs_verification or (web_low_count or 0) > 0 or low_refs > 0)
        quality_confidence = max(0.0, min(1.0, 1.0 - low_ratio))
        if doc_level.lower() == "low":
            quality_confidence = min(quality_confidence, 0.45)

        if self.metrics_collector is not None and callable(getattr(self.metrics_collector, "emit_event", None)):
            try:
                self.metrics_collector.emit_event(
                    "knowledge_acquisition",
                    latency=0.0,
                    energy=0.0,
                    throughput=0.0,
                    status="needs_verification" if needs_verification else "ok",
                    confidence=quality_confidence,
                    stage=str(goal or "autonomous_task"),
                    metadata={
                        "query": payload.get("query"),
                        "returned": payload.get("returned"),
                        "channels_used": list(payload.get("channels_used") or []),
                        "web_skipped": bool(payload.get("web_skipped")),
                        "web_avg_trust": web_avg_trust,
                        "web_low_count": web_low_count,
                        "doc_consensus_level": doc_level or None,
                        "doc_needs_verification": bool(doc_needs_verification),
                        "reference_summary": ref_summary,
                    },
                )
            except Exception:
                pass

        if _env_flag("BSS_SOURCE_MONITOR_ENABLED"):
            state = ctx.setdefault(
                "source_monitor",
                {"total_refs": 0, "low_refs": 0, "quarantines": 0, "last_quarantine_ts": 0.0},
            )
            try:
                state["total_refs"] = int(state.get("total_refs", 0)) + total_refs
                state["low_refs"] = int(state.get("low_refs", 0)) + low_refs
            except Exception:
                pass

            try:
                ratio = float(state.get("low_refs", 0)) / max(1, int(state.get("total_refs", 0)))
            except Exception:
                ratio = low_ratio

            try:
                min_refs = int(os.environ.get("BSS_SOURCE_MONITOR_MIN_REFS", "8") or 8)
            except Exception:
                min_refs = 8
            try:
                threshold = float(os.environ.get("BSS_SOURCE_MONITOR_LOW_TRUST_RATIO", "0.6") or 0.6)
            except Exception:
                threshold = 0.6
            try:
                cooldown_s = float(os.environ.get("BSS_SOURCE_MONITOR_COOLDOWN_S", "300") or 300)
            except Exception:
                cooldown_s = 300.0

            now_ts = time.time()
            recently_quarantined = bool(now_ts - float(state.get("last_quarantine_ts", 0.0) or 0.0) < cooldown_s)
            should_quarantine = (
                not recently_quarantined
                and int(state.get("total_refs", 0) or 0) >= min_refs
                and ratio >= threshold
            )
            if should_quarantine and hasattr(self.tool_bridge, "disable_web_access"):
                reason = f"source_monitor_low_trust_ratio:{ratio:.3f}"
                top_low_host = None
                top = ref_summary.get("top_low_hosts")
                if isinstance(top, list) and top:
                    top_low_host = str(top[0] or "").strip()
                    if top_low_host:
                        reason += f":{top_low_host}"
                try:
                    self.tool_bridge.disable_web_access(reason=reason, cooldown_s=cooldown_s)  # type: ignore[attr-defined]
                except Exception:
                    pass
                if top_low_host and hasattr(self.tool_bridge, "block_web_domain") and _env_flag("BSS_SOURCE_MONITOR_BLOCK_DOMAINS"):
                    try:
                        self.tool_bridge.block_web_domain(top_low_host, reason="source_monitor_low_trust")  # type: ignore[attr-defined]
                    except Exception:
                        pass

                state["quarantines"] = int(state.get("quarantines", 0) or 0) + 1
                state["last_quarantine_ts"] = float(now_ts)
                alert_entry = {
                    "type": "source_monitor_quarantine",
                    "ratio": ratio,
                    "threshold": threshold,
                    "total_refs": int(state.get("total_refs", 0) or 0),
                    "low_refs": int(state.get("low_refs", 0) or 0),
                    "top_low_host": top_low_host,
                }
                ctx.setdefault("alerts", []).append(alert_entry)
                report_meta.setdefault("alerts", []).append(dict(alert_entry))
                self._audit_event(
                    "source_monitor",
                    {
                        "goal": goal,
                        "action": "disable_web_access",
                        "reason": reason,
                        "cooldown_s": float(cooldown_s),
                        "low_trust_ratio": float(ratio),
                        "threshold": float(threshold),
                        "summary": ref_summary,
                    },
                )
                if self.metrics_collector is not None and callable(getattr(self.metrics_collector, "emit_event", None)):
                    try:
                        self.metrics_collector.emit_event(
                            "source_monitor",
                            latency=0.0,
                            energy=0.0,
                            throughput=0.0,
                            status="alert",
                            confidence=max(0.0, min(1.0, 1.0 - ratio)),
                            stage=str(goal or "autonomous_task"),
                            metadata={
                                "action": "disable_web_access",
                                "reason": reason,
                                "cooldown_s": float(cooldown_s),
                                "low_trust_ratio": float(ratio),
                                "threshold": float(threshold),
                                "summary": ref_summary,
                            },
                        )
                    except Exception:
                        pass

        self._audit_event(
            "knowledge_acquisition",
            {
                "goal": goal,
                "step": int(step.index),
                "action_type": str(action.get("type") or ""),
                "query": payload.get("query"),
                "roots": payload.get("roots"),
                "returned": payload.get("returned"),
                "needs_verification": bool(needs_verification),
                "reference_summary": ref_summary,
                "web_avg_trust": web_avg_trust,
                "doc_consensus": doc_level or None,
            },
        )

    def _acquire_knowledge_payload(
        self,
        *,
        goal: str,
        cfg: Dict[str, Any],
        step: TaskStep,
        action: Dict[str, Any],
        info: Dict[str, Any],
        allowed_roots: Sequence[str],
    ) -> Dict[str, Any]:
        use_code_index = bool(cfg.get("use_code_index", cfg.get("code_index", True)))
        use_web_search = bool(cfg.get("web_search") or cfg.get("use_web_search"))
        use_github_code_search = bool(cfg.get("github_code_search") or cfg.get("use_github_code_search"))
        use_knowledge_graph = bool(cfg.get("knowledge_graph") or cfg.get("use_knowledge_graph", True))
        use_long_term_memory = bool(cfg.get("long_term_memory") or cfg.get("use_long_term_memory", True))
        use_documentation_tool = bool(
            cfg.get("documentation")
            or cfg.get("use_documentation_tool")
            or cfg.get("documentation_tool")
            or cfg.get("use_documentation")
            or cfg.get("web")
            or cfg.get("use_web")
        )

        roots: List[str] = []
        if use_code_index:
            root_cfg = cfg.get("code_roots") or cfg.get("roots") or cfg.get("code_root") or cfg.get("root")
            if isinstance(root_cfg, (list, tuple)):
                roots = [str(r) for r in root_cfg if r]
            elif root_cfg:
                roots = [str(root_cfg)]
            if not roots and allowed_roots:
                roots = [str(allowed_roots[0])]
            roots = [r for r in roots if r]

        max_roots = max(1, min(int(cfg.get("max_roots", 1)), 3))
        roots = roots[:max_roots]

        query = str(cfg.get("query") or "").strip()
        if not query:
            error = str(info.get("error") or info.get("reason") or "").strip()
            action_type = str(action.get("type") or "").strip()
            query = "\n".join(
                part
                for part in (
                    str(goal or "").strip(),
                    f"Failure: {error}" if error else "",
                    f"Action: {action_type}" if action_type else "",
                    f"Step: {step.title}" if step.title else "",
                )
                if part
            )
        query = _clip_text(query, max_chars=700)

        top_k = max(1, min(int(cfg.get("top_k", 5)), 25))
        max_files = max(1, min(int(cfg.get("max_files", 800)), 5000))
        embedding_dimensions = max(8, min(int(cfg.get("embedding_dimensions", 128)), 2048))
        max_chars_per_hit = max(0, int(cfg.get("max_chars_per_hit", 2000)))
        max_reference_chars = max(0, int(cfg.get("max_reference_chars", 2000)))
        include_suffixes = cfg.get("include_suffixes") or cfg.get("suffixes") or (".py", ".md", ".txt", ".rst")

        if (
            not (use_code_index and roots)
            and not (use_web_search or use_documentation_tool or use_github_code_search)
            and not (use_long_term_memory or use_knowledge_graph)
        ):
            return {}

        results: List[Dict[str, Any]] = []
        references: List[Dict[str, Any]] = []
        channels_used: List[str] = []
        returned_memory_hits = 0
        returned_knowledge_hits = 0
        returned_code_hits = 0
        returned_web_results = 0
        returned_github_results = 0
        returned_doc_sources = 0

        long_term_memory: Dict[str, Any] | None = None
        knowledge_graph: Dict[str, Any] | None = None
        web_skipped = False
        web_skip_reason: str | None = None

        # ------------------------------------------------------------------ internal-first retrieval
        # Prefer local long-term memory + knowledge graph before any web calls to reduce
        # repeated searches. These are best-effort and do not require network access.
        memory_hits: List[Dict[str, Any]] = []
        if use_long_term_memory:
            try:
                from pathlib import Path

                raw_root = (
                    cfg.get("long_term_memory_root")
                    or cfg.get("ltm_root")
                    or os.environ.get("BSS_LONG_TERM_MEMORY_ROOT")
                    or os.environ.get("BSS_LTM_ROOT")
                )
                if raw_root:
                    ltm_root = Path(str(raw_root)).expanduser()
                else:
                    repo_root = Path(__file__).resolve().parents[2]
                    ltm_root = repo_root / "data" / "long_term_memory"
                    if allowed_roots:
                        try:
                            cand = Path(str(allowed_roots[0])).resolve() / "data" / "long_term_memory"
                            if cand.exists():
                                ltm_root = cand
                        except Exception:
                            pass

                vector_root = ltm_root / "vector_store"
                if (ltm_root / "metadata.json").exists():
                    vector_root = ltm_root
                meta_path = vector_root / "metadata.json"

                if meta_path.exists():
                    raw = json.loads(meta_path.read_text(encoding="utf-8"))
                    records = raw.get("records") if isinstance(raw, dict) else None
                    if isinstance(records, list) and records:
                        token_re = re.compile(r"\w+")
                        tokens = [t for t in token_re.findall(str(query or "").lower()) if len(t) >= 2]
                        tokens = tokens[:24] if tokens else []

                        def _lex_score(text: str) -> float:
                            if not tokens:
                                return 0.0
                            hay = str(text or "").lower()
                            hits = sum(1 for t in tokens if t in hay)
                            return hits / max(1, len(tokens))

                        scored: List[Dict[str, Any]] = []
                        for rec in records:
                            if not isinstance(rec, dict):
                                continue
                            rec_id = str(rec.get("id") or "").strip()
                            rec_text = str(rec.get("text") or "").strip()
                            rec_meta = rec.get("metadata")
                            rec_meta_dict = dict(rec_meta) if isinstance(rec_meta, dict) else {}
                            if rec_meta_dict.get("archived") is True:
                                continue
                            if not rec_text:
                                continue
                            needs_verification = bool(
                                rec_meta_dict.get("needs_verification")
                                or rec_meta_dict.get("verification_needed")
                                or rec_meta_dict.get("needs_review")
                            )
                            verification_status = str(rec_meta_dict.get("verification_status") or "").strip().lower()
                            needs_verification = bool(
                                needs_verification and verification_status not in {"verified", "confirmed", "trusted"}
                            )
                            score = _lex_score(rec_text)
                            if score <= 0.0:
                                continue
                            scored.append(
                                {
                                    "id": rec_id or None,
                                    "score": round(float(score), 4),
                                    "text": _clip_text(rec_text, max_chars=1600),
                                    "metadata": rec_meta_dict,
                                    "needs_verification": bool(needs_verification),
                                }
                            )
                        scored.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
                        memory_hits = scored[:top_k]
                        if memory_hits:
                            returned_memory_hits = len(memory_hits)
                            channels_used.append("long_term_memory")
                            lines = []
                            for hit in memory_hits:
                                title = str(hit.get("metadata", {}).get("goal") or hit.get("metadata", {}).get("query") or hit.get("id") or "memory")
                                flag = " needs_verification" if hit.get("needs_verification") else ""
                                lines.append(f"- {title} (score={float(hit.get('score', 0.0)):.3f}{flag})")
                                ref_id = str(hit.get("id") or "")
                                if ref_id:
                                    references.append({"url": f"memory:{ref_id}", "title": title, "source": "long_term_memory"})
                            mem_text = "long_term_memory hits:\n" + "\n".join(lines) if lines else ""
                            long_term_memory = {
                                "root": str(ltm_root),
                                "query": query,
                                "meta_path": str(meta_path),
                                "returned": int(returned_memory_hits),
                                "hits": memory_hits,
                                "text": _clip_text(mem_text, max_chars=2000),
                            }
                            results.append(
                                {
                                    "channel": "long_term_memory",
                                    "info": {"returned": int(returned_memory_hits), "root": str(ltm_root)},
                                    "text": long_term_memory.get("text") if isinstance(long_term_memory, dict) else "",
                                    "hits": memory_hits,
                                    "references": [],
                                }
                            )
            except Exception:
                memory_hits = []

        knowledge_hits: List[Dict[str, Any]] = []
        knowledge_max_sim = 0.0
        if use_knowledge_graph:
            try:
                from backend.knowledge.registry import get_default_aligner

                aligner = get_default_aligner()
                entities = getattr(aligner, "entities", None) if aligner is not None else None
                has_entities = isinstance(entities, dict) and bool(entities)
            except Exception:
                has_entities = False

            if has_entities:
                try:
                    channels_used.append("knowledge_graph")
                    obs, _, _, kg_info = self.tool_bridge.step(
                        {
                            "type": "knowledge_query",
                            "query": query,
                            "top_k": top_k,
                            "include_metadata": True,
                            "include_relations": False,
                        }
                    )
                    kg_results = obs.get("results") if isinstance(obs, dict) else None
                    kg_refs = obs.get("references") if isinstance(obs, dict) else None
                    knowledge_hits = list(kg_results) if isinstance(kg_results, list) else []
                    refs_list = list(kg_refs) if isinstance(kg_refs, list) else []
                    returned_knowledge_hits = len(knowledge_hits)

                    for entry in knowledge_hits:
                        if not isinstance(entry, dict):
                            continue
                        sim = entry.get("similarity")
                        try:
                            knowledge_max_sim = max(knowledge_max_sim, float(sim))
                        except Exception:
                            pass

                    for item in refs_list:
                        if isinstance(item, dict):
                            references.append(dict(item))

                    knowledge_graph = {
                        "query": query,
                        "info": dict(kg_info or {}),
                        "text": obs.get("text") if isinstance(obs, dict) else "",
                        "results": knowledge_hits,
                        "references": refs_list,
                    }
                    results.append(
                        {
                            "channel": "knowledge_graph",
                            "info": dict(kg_info or {}),
                            "text": knowledge_graph.get("text") if isinstance(knowledge_graph, dict) else "",
                            "hits": knowledge_hits,
                            "references": refs_list,
                        }
                    )
                except Exception:
                    knowledge_hits = []

        # Decide whether to call any web channels. We only skip web calls when the
        # internal knowledge stores already have relevant hits (memory/graph).
        if (use_web_search or use_documentation_tool or use_github_code_search) and bool(
            cfg.get("web_if_internal_insufficient", True)
        ):
            try:
                mem_min = float(cfg.get("memory_min_score", cfg.get("internal_min_score", 0.2)))
            except Exception:
                mem_min = 0.2
            try:
                kg_min = float(cfg.get("knowledge_min_similarity", cfg.get("internal_min_score", 0.25)))
            except Exception:
                kg_min = 0.25

            mem_best_verified = 0.0
            for hit in memory_hits[: max(1, top_k)]:
                if isinstance(hit, dict) and hit.get("needs_verification"):
                    continue
                try:
                    mem_best_verified = max(mem_best_verified, float(hit.get("score", 0.0)))
                except Exception:
                    pass

            internal_sufficient = (returned_memory_hits > 0 and mem_best_verified >= mem_min) or (
                returned_knowledge_hits > 0 and knowledge_max_sim >= kg_min
            )
            if internal_sufficient:
                use_web_search = False
                use_documentation_tool = False
                use_github_code_search = False
                web_skipped = True
                web_skip_reason = "internal_memory_or_knowledge_graph_hit"

        if use_code_index and roots:
            channels_used.append("code_index")
            for root in roots:
                search_action = {
                    "type": "code_index_search",
                    "root": root,
                    "query": query,
                    "top_k": top_k,
                    "max_chars_per_hit": max_chars_per_hit,
                    "max_reference_chars": max_reference_chars,
                }
                obs, _, _, search_info = self.tool_bridge.step(search_action)
                if isinstance(search_info, dict) and search_info.get("error") == "code_index_not_built":
                    build_action = {
                        "type": "code_index_build",
                        "root": root,
                        "max_files": max_files,
                        "embedding_dimensions": embedding_dimensions,
                        "include_suffixes": include_suffixes,
                    }
                    self.tool_bridge.step(build_action)
                    obs, _, _, search_info = self.tool_bridge.step(search_action)

                hits = obs.get("hits") if isinstance(obs, dict) else None
                refs = obs.get("references") if isinstance(obs, dict) else None
                hits_list = list(hits) if isinstance(hits, list) else []
                refs_list = list(refs) if isinstance(refs, list) else []
                returned_code_hits += len(hits_list)

                results.append(
                    {
                        "root": str(root),
                        "info": dict(search_info or {}),
                        "text": obs.get("text") if isinstance(obs, dict) else "",
                        "hits": hits_list,
                        "references": refs_list,
                    }
                )
                for item in refs_list:
                    if isinstance(item, dict):
                        references.append(dict(item))

        web_search: Dict[str, Any] | None = None
        if use_web_search:
            channels_used.append("web_search")
            web_query = str(cfg.get("web_search_query") or cfg.get("web_query") or query or goal or "").strip()
            if web_query:
                max_results = max(1, min(int(cfg.get("web_search_max_results", 5)), 10))
                obs, _, _, web_info = self.tool_bridge.step({"type": "web_search", "query": web_query, "max_results": max_results})
                returned_web_results = int(web_info.get("returned", 0)) if isinstance(web_info, dict) else 0
                web_text = obs.get("text") if isinstance(obs, dict) else ""
                web_search = {
                    "query": web_query,
                    "info": dict(web_info or {}),
                    "text": _clip_text(str(web_text or ""), max_chars=4000),
                    "results": obs.get("results") if isinstance(obs, dict) else None,
                }
                results_list = obs.get("results") if isinstance(obs, dict) else None
                if isinstance(results_list, list):
                    for item in results_list[:max_results]:
                        if not isinstance(item, dict):
                            continue
                        url = str(item.get("url") or "")
                        title = str(item.get("title") or "") or url
                        snippet = str(item.get("snippet") or "")
                        if not url:
                            continue
                        ref: Dict[str, Any] = {"url": url, "title": title, "snippet": snippet, "source": "web_search"}
                        trust = item.get("trust")
                        trust_score = item.get("trust_score")
                        host = item.get("host")
                        blocked_domain = item.get("blocked_domain")
                        if trust is not None:
                            ref["trust"] = trust
                        if trust_score is not None:
                            ref["trust_score"] = trust_score
                        if host:
                            ref["host"] = host
                        if blocked_domain is not None:
                            ref["blocked_domain"] = bool(blocked_domain)
                        references.append(ref)

        github_code_search: Dict[str, Any] | None = None
        if use_github_code_search:
            channels_used.append("github_code_search")
            gh_query = str(cfg.get("github_query") or cfg.get("github_code_search_query") or query or goal or "").strip()
            if gh_query:
                max_results = max(1, min(int(cfg.get("github_max_results", 5)), 20))
                gh_action: Dict[str, Any] = {"type": "github_code_search", "query": gh_query, "max_results": max_results}
                token = cfg.get("github_token") or cfg.get("token") or os.environ.get("GITHUB_TOKEN")
                if token:
                    gh_action["token"] = token
                obs, _, _, gh_info = self.tool_bridge.step(gh_action)
                returned_github_results = int(gh_info.get("returned", 0)) if isinstance(gh_info, dict) else 0
                gh_text = obs.get("text") if isinstance(obs, dict) else ""
                github_code_search = {
                    "query": gh_query,
                    "info": dict(gh_info or {}),
                    "text": _clip_text(str(gh_text or ""), max_chars=4000),
                    "results": obs.get("results") if isinstance(obs, dict) else None,
                }
                gh_results = obs.get("results") if isinstance(obs, dict) else None
                if isinstance(gh_results, list):
                    for item in gh_results[:max_results]:
                        if not isinstance(item, dict):
                            continue
                        url = str(item.get("html_url") or item.get("url") or "")
                        repo = str(item.get("repository") or "")
                        path = str(item.get("path") or "")
                        title = f"{repo}/{path}".strip("/") if (repo or path) else url
                        if not url:
                            continue
                        references.append({"url": url, "title": title or url, "source": "github_code_search"})

        web: Dict[str, Any] | None = None
        if use_documentation_tool:
            channels_used.append("documentation_tool")
            web_query = str(cfg.get("web_query") or cfg.get("documentation_query") or goal or "").strip()
            if web_query:
                max_sources = max(1, min(int(cfg.get("web_max_sources", 2)), 5))
                timeout_s = float(cfg.get("web_timeout_s", cfg.get("timeout_s", 10.0)))
                obs, _, _, web_info = self.tool_bridge.step(
                    {
                        "type": "documentation_tool",
                        "query": web_query,
                        "max_sources": max_sources,
                        "timeout_s": timeout_s,
                    }
                )
                web_text = obs.get("text") if isinstance(obs, dict) else ""
                if isinstance(obs, dict) and isinstance(obs.get("sources"), list):
                    returned_doc_sources = len(obs.get("sources") or [])
                    for src in obs.get("sources")[:max_sources]:
                        if not isinstance(src, dict):
                            continue
                        search = src.get("search") if isinstance(src.get("search"), dict) else {}
                        page = src.get("page") if isinstance(src.get("page"), dict) else {}
                        url = str(page.get("final_url") or page.get("url") or search.get("url") or "").strip()
                        title = str(page.get("title") or search.get("title") or "").strip() or url
                        if not url:
                            continue
                        ref: Dict[str, Any] = {"url": url, "title": title, "source": "documentation_tool"}
                        trust = search.get("trust")
                        trust_score = search.get("trust_score")
                        host = search.get("host")
                        blocked_domain = search.get("blocked_domain")
                        if trust is not None:
                            ref["trust"] = trust
                        if trust_score is not None:
                            ref["trust_score"] = trust_score
                        if host:
                            ref["host"] = host
                        if blocked_domain is not None:
                            ref["blocked_domain"] = bool(blocked_domain)
                        references.append(ref)
                few_shot: Dict[str, Any] | None = None
                if isinstance(obs, dict) and isinstance(obs.get("sources"), list) and obs.get("sources"):
                    try:
                        from modules.knowledge.few_shot_extractor import extract_few_shot_material

                        few_shot = extract_few_shot_material(
                            obs.get("sources"),
                            max_signatures=6,
                            max_snippets=3,
                            max_chars_per_snippet=800,
                            max_total_chars=2200,
                        )
                    except Exception:
                        few_shot = None
                web = {
                    "query": web_query,
                    "info": dict(web_info or {}),
                    "text": _clip_text(str(web_text or ""), max_chars=4000),
                }
                if isinstance(few_shot, dict) and few_shot:
                    web["few_shot"] = few_shot

        retrieval_parts: List[str] = []
        for entry in results:
            text = str(entry.get("text") or "").strip()
            if text:
                retrieval_parts.append(text)
        if isinstance(web_search, dict) and web_search.get("text"):
            retrieval_parts.append(str(web_search.get("text") or ""))
        if isinstance(github_code_search, dict) and github_code_search.get("text"):
            retrieval_parts.append(str(github_code_search.get("text") or ""))
        if isinstance(web, dict) and web.get("text"):
            retrieval_parts.append(str(web.get("text") or ""))
        retrieval_context = _clip_text("\n\n".join(retrieval_parts), max_chars=4000)

        seen: set[tuple[str, str]] = set()
        unique_refs: List[Dict[str, Any]] = []
        for ref in references:
            url = str(ref.get("url") or "")
            title = str(ref.get("title") or "")
            key = (url, title)
            if not url or key in seen:
                continue
            seen.add(key)
            unique_refs.append(ref)

        total_returned = int(
            returned_memory_hits
            + returned_knowledge_hits
            + returned_code_hits
            + returned_web_results
            + returned_github_results
            + returned_doc_sources
        )
        return {
            "trigger": "on_failure",
            "query": query,
            "roots": [str(r) for r in roots],
            "returned": total_returned,
            "returned_memory_hits": int(returned_memory_hits),
            "returned_knowledge_hits": int(returned_knowledge_hits),
            "returned_code_hits": int(returned_code_hits),
            "returned_web_results": int(returned_web_results),
            "returned_doc_sources": int(returned_doc_sources),
            "returned_github_results": int(returned_github_results),
            "channels_used": list(channels_used),
            "retrieval_context": retrieval_context,
            "references": unique_refs[: max(1, top_k * max_roots)],
            "results": results,
            "long_term_memory": long_term_memory,
            "knowledge_graph": knowledge_graph,
            "web_skipped": bool(web_skipped),
            "web_skip_reason": web_skip_reason,
            "web": web,
            "web_search": web_search,
            "github_code_search": github_code_search,
            "timestamp": time.time(),
        }

    def _audit_step(
        self,
        goal: str,
        action: Dict[str, Any],
        *,
        step: TaskStep,
        status: str,
        reward: float,
        terminated: bool,
        info: Dict[str, Any],
        observation: Dict[str, Any],
        introspection: Optional[Dict[str, Any]] = None,
    ) -> None:
        security = self.security_manager
        audit = getattr(security, "audit", None) if security is not None else None
        if audit is None or not getattr(audit, "enabled", True):
            return
        try:
            payload: Dict[str, Any] = {
                "event": "task_step",
                "goal": str(goal or ""),
                "step_index": int(step.index),
                "step_title": str(step.title),
                "action_type": str(action.get("type") or ""),
                "action": redact_action(action, max_chars=512) if callable(redact_action) else dict(action),
                "status": str(status),
                "reward": float(reward),
                "terminated": bool(terminated),
                "info": dict(info or {}),
            }
            if isinstance(introspection, dict) and introspection:
                payload["introspection"] = introspection
            text = observation.get("text")
            if isinstance(text, str):
                payload["observation_chars"] = len(text)
            audit.log(payload)
        except Exception:
            return

    def _audit_event(self, name: str, payload: Dict[str, Any]) -> None:
        security = self.security_manager
        audit = getattr(security, "audit", None) if security is not None else None
        if audit is None or not getattr(audit, "enabled", True):
            return
        try:
            audit.log({"event": str(name), **(payload or {})})
        except Exception:
            return


def _status_from_result(reward: float, info: Dict[str, Any]) -> str:
    if info.get("blocked"):
        return "blocked"
    if info.get("error"):
        return "error"
    if isinstance(reward, (int, float)) and float(reward) < 0:
        return "error"
    return "success"


def _try_parse_json(payload: str) -> Optional[Any]:
    payload = (payload or "").strip()
    if not payload:
        return None

    parsed = _loads_json_loose(payload)
    if parsed is not None:
        return parsed

    start = payload.find("{")
    end = payload.rfind("}")
    if start != -1 and end != -1 and end > start:
        parsed = _loads_json_loose(payload[start : end + 1])
        if parsed is not None:
            return parsed

    start = payload.find("[")
    end = payload.rfind("]")
    if start != -1 and end != -1 and end > start:
        parsed = _loads_json_loose(payload[start : end + 1])
        if parsed is not None:
            return parsed

    return None


def _loads_json_loose(payload: str) -> Optional[Any]:
    """Parse JSON, with a best-effort repair for unescaped Windows backslashes."""

    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        repaired = _escape_backslashes_in_json_strings(payload)
        if repaired != payload:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                return None
    return None


def _escape_backslashes_in_json_strings(payload: str) -> str:
    r"""Escape backslashes inside JSON string literals.

    This is a forgiving helper for common user-provided plans on Windows where
    paths are pasted as ``C:\path\file`` instead of ``C:\\path\\file``.
    We only keep escapes needed for JSON syntax (``\\`` and ``\"``); all other
    backslashes inside strings are treated as literal path separators.
    """

    if "\\" not in payload:
        return payload

    out: List[str] = []
    in_string = False
    i = 0
    while i < len(payload):
        ch = payload[i]
        if ch == '"':
            in_string = not in_string
            out.append(ch)
            i += 1
            continue

        if in_string and ch == "\\":
            nxt = payload[i + 1] if i + 1 < len(payload) else ""
            if nxt in {'"', "\\"}:
                out.append("\\")
                out.append(nxt)
                i += 2
                continue
            out.append("\\\\")
            i += 1
            continue

        out.append(ch)
        i += 1

    return "".join(out)


_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def _extract_plan_json(text: str) -> Optional[Any]:
    """Extract JSON from code-fences or inline JSON fragments."""

    for match in _FENCED_JSON_RE.finditer(text or ""):
        parsed = _try_parse_json(match.group(1))
        if parsed is not None:
            return parsed

    return _try_parse_json(text)


def _normalize_plan(parsed: Any, goal: str, *, planner: str) -> TaskPlan:
    raw = parsed
    plan_goal = str(goal or "")
    steps_raw: Any = None

    if isinstance(parsed, dict):
        if "goal" in parsed and parsed.get("goal") is not None:
            plan_goal = str(parsed.get("goal"))
        steps_raw = parsed.get("steps")
    elif isinstance(parsed, list):
        steps_raw = parsed
        raw = {"steps": parsed}
    else:
        return TaskPlan(goal=plan_goal, steps=tuple(), planner=planner, raw={"error": "invalid_plan_format", "raw": raw})

    if not isinstance(steps_raw, list):
        return TaskPlan(goal=plan_goal, steps=tuple(), planner=planner, raw={"error": "missing_steps_list", "raw": raw})

    steps: List[TaskStep] = []
    next_index = 1
    for entry in steps_raw:
        title = f"step {next_index}"
        action: Optional[Dict[str, Any]] = None
        meta: Dict[str, Any] = {}

        if isinstance(entry, dict):
            if isinstance(entry.get("action"), dict):
                action = dict(entry["action"])
                title = str(entry.get("title") or entry.get("name") or entry.get("step") or title)
                meta = {k: v for k, v in entry.items() if k not in {"action"}}
            elif isinstance(entry.get("tool_action"), dict):
                action = dict(entry["tool_action"])
                title = str(entry.get("title") or entry.get("name") or entry.get("step") or title)
                meta = {k: v for k, v in entry.items() if k not in {"tool_action"}}
            elif isinstance(entry.get("type"), str):
                action = dict(entry)
                title = str(entry.get("title") or entry.get("name") or entry.get("step") or title)
            else:
                continue
        else:
            continue

        action_type = str(action.get("type") or "").strip() if action else ""
        if not action_type:
            continue

        steps.append(TaskStep(index=next_index, title=title, action=action, meta=meta))
        next_index += 1

    return TaskPlan(goal=plan_goal, steps=tuple(steps), planner=planner, raw=raw)


def _heuristic_steps_from_text(text: str) -> List[Dict[str, Any]]:
    """Very small heuristic parser for common local tasks.

    This is intentionally conservative: it only emits steps for explicit patterns
    that map directly to ToolEnvironmentBridge actions.
    """

    text = str(text or "").strip()
    if not text:
        return []

    steps: List[Dict[str, Any]] = []

    # create_dir: "create dir <path>" / "创建目录 <path>"
    create_dir_patterns = [
        r"(?i)\bcreate\s+dir(?:ectory)?\s+(?P<path>[^\s]+)",
        r"创建目录\s+(?P<path>[^\s]+)",
        r"新建文件夹\s+(?P<path>[^\s]+)",
    ]
    for pat in create_dir_patterns:
        match = re.search(pat, text)
        if match:
            steps.append({"type": "create_dir", "path": match.group("path"), "parents": True, "exist_ok": True})
            break

    # write_file: "write file <path> with <text>" / "写入文件 <path> 内容 <text>"
    write_match = re.search(r"(?is)\bwrite\s+file\s+(?P<path>[^\s]+)\s+with\s+(?P<text>.+)", text)
    if write_match:
        steps.append({"type": "write_file", "path": write_match.group("path"), "text": write_match.group("text")})
        return steps

    write_cn = re.search(r"(?is)(写入文件|写文件)\s+(?P<path>[^\s]+)\s+(内容|写入)\s+(?P<text>.+)", text)
    if write_cn:
        steps.append({"type": "write_file", "path": write_cn.group("path"), "text": write_cn.group("text")})
        return steps

    return steps


__all__ = [
    "ActionGovernor",
    "AutonomousTaskExecutor",
    "ExecutionEvent",
    "ExecutionReport",
    "GovernorConfig",
    "HeuristicTaskPlanner",
    "LLMTaskPlanner",
    "TaskPlan",
    "TaskPlanner",
    "TaskPlanningError",
    "TaskStep",
]
