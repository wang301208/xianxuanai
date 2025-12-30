from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple


ApprovalRule = Callable[[str, str, Dict[str, object]], Tuple[bool, Optional[str]]]


@dataclass
class GuardianDecision:
    """Outcome of evaluating remediation action."""

    allowed: bool
    reason: Optional[str] = None
    rule: Optional[str] = None


class RemediationGuardian:
    """Independent approval layer supervising remediation actions."""

    def __init__(self) -> None:
        self._rules: List[Tuple[str, ApprovalRule]] = []

    def add_rule(self, name: str, rule: ApprovalRule) -> None:
        """Register a named ``rule`` used during approval."""

        self._rules.append((name, rule))

    def clear_rules(self) -> None:
        self._rules.clear()

    def approve(self, module: str, action: str, context: Dict[str, object]) -> GuardianDecision:
        """Return approval decision for ``module`` attempting ``action``."""

        for name, rule in self._rules:
            allowed, reason = rule(module, action, context)
            if not allowed:
                return GuardianDecision(False, reason=reason, rule=name)
        return GuardianDecision(True)
