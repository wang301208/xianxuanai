from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .guardian import GuardianDecision, RemediationGuardian


RestartFn = Callable[[Dict[str, Any]], bool]
FallbackFn = Callable[[Dict[str, Any]], bool]
PatchFn = Callable[[Dict[str, Any]], "RemediationPatch"]


@dataclass
class RemediationPatch:
    """Represents a candidate code patch generated during remediation."""

    description: str
    applied: bool
    tests_run: List[str] = field(default_factory=list)
    notes: str = ""
    artifacts: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RemediationResult:
    """Result of attempting to recover a failing module."""

    module: str
    success: bool
    method: Optional[str] = None
    steps: List[str] = field(default_factory=list)
    error: Optional[str] = None
    patch: Optional[RemediationPatch] = None


@dataclass
class ModuleRemediationPlan:
    """Configured remediation actions for a specific module."""

    restart: Optional[RestartFn] = None
    fallback: Optional[FallbackFn] = None
    patch: Optional[PatchFn] = None
    protected: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class RemediationManager:
    """Coordinate automated module remediation with guard rails."""

    def __init__(self, guardian: Optional[RemediationGuardian] = None) -> None:
        self._plans: Dict[str, ModuleRemediationPlan] = {}
        self._guardian = guardian

    # ------------------------------------------------------------------
    def register_module(
        self,
        module: str,
        *,
        restart: Optional[RestartFn] = None,
        fallback: Optional[FallbackFn] = None,
        patch: Optional[PatchFn] = None,
        protected: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._plans[module] = ModuleRemediationPlan(
            restart=restart,
            fallback=fallback,
            patch=patch,
            protected=protected,
            metadata=dict(metadata or {}),
        )

    def unregister_module(self, module: str) -> None:
        self._plans.pop(module, None)

    def plan_for(self, module: str) -> Optional[ModuleRemediationPlan]:
        return self._plans.get(module)

    # ------------------------------------------------------------------
    def attempt(
        self,
        module: str,
        context: Optional[Dict[str, Any]] = None,
        *,
        allow_code_patch: bool = False,
    ) -> RemediationResult:
        plan = self._plans.get(module)
        result = RemediationResult(module=module, success=False)
        ctx = dict(context or {})
        ctx.setdefault("module", module)

        if plan is None:
            result.error = "no_plan_registered"
            return result
        ctx.setdefault("plan_metadata", plan.metadata)

        def _approved(action: str) -> GuardianDecision:
            if self._guardian is None:
                return GuardianDecision(True)
            return self._guardian.approve(module, action, ctx)

        # Restart attempt ------------------------------------------------
        if plan.restart is not None:
            decision = _approved("restart")
            if not decision.allowed:
                result.steps.append(f"restart_denied:{decision.rule or 'guardian'}")
            else:
                try:
                    restarted = bool(plan.restart(ctx))
                    result.steps.append("restart")
                    if restarted:
                        result.success = True
                        result.method = "restart"
                        return result
                except Exception as exc:  # pragma: no cover - defensive
                    result.steps.append(f"restart_error:{exc}")
        else:
            result.steps.append("restart_unavailable")

        # Fallback attempt -----------------------------------------------
        if plan.fallback is not None:
            decision = _approved("fallback")
            if not decision.allowed:
                result.steps.append(f"fallback_denied:{decision.rule or 'guardian'}")
            else:
                try:
                    switched = bool(plan.fallback(ctx))
                    result.steps.append("fallback")
                    if switched:
                        result.success = True
                        result.method = "fallback"
                        return result
                except Exception as exc:  # pragma: no cover - defensive
                    result.steps.append(f"fallback_error:{exc}")
        else:
            result.steps.append("fallback_unavailable")

        # Code patch attempt ---------------------------------------------
        if allow_code_patch and plan.patch is not None and not plan.protected:
            decision = _approved("patch")
            if not decision.allowed:
                result.steps.append(f"patch_denied:{decision.rule or 'guardian'}")
                result.error = decision.reason or "patch_denied"
                return result
            try:
                patch = plan.patch(ctx)
                result.steps.append("patch")
                result.patch = patch
                result.success = bool(patch.applied)
                result.method = "patch" if patch.applied else None
                if not patch.applied:
                    result.error = "patch_not_applied"
                return result
            except Exception as exc:  # pragma: no cover - defensive
                result.steps.append(f"patch_error:{exc}")
                result.error = str(exc)
                return result

        if plan.protected and allow_code_patch:
            result.steps.append("patch_denied_protected")
        result.error = result.error or "remediation_exhausted"
        return result
