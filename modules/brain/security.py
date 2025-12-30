"""Basic security utilities for neural modules.

This module provides minimalistic detectors and guards to simulate
security mechanisms within neural components. The implementations are
intentionally lightweight so that they remain easy to unit test while
still demonstrating core concepts like adversarial input detection,
backdoor scanning and memory integrity checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
import copy


# ---------------------------------------------------------------------------
# Adversarial input detection
# ---------------------------------------------------------------------------


@dataclass
class AdversarialInputDetector:
    """Detects and sanitises adversarial numeric patterns.

    Any numeric value whose magnitude exceeds ``threshold`` is treated as
    adversarial.  Sanitisation simply clips offending values to the
    allowed range.
    """

    threshold: float = 10.0

    def detect(self, data: Any) -> bool:
        """Return ``True`` if adversarial content is found."""
        found = False

        def _check(value: Any) -> None:
            nonlocal found
            if isinstance(value, (int, float)):
                if abs(value) > self.threshold:
                    found = True
            elif isinstance(value, (list, tuple)):
                for v in value:
                    _check(v)
            elif isinstance(value, dict):
                for v in value.values():
                    _check(v)

        _check(data)
        return found

    def sanitize(self, data: Any) -> Any:
        """Clip numeric values to the accepted range."""

        def _sanitize(value: Any) -> Any:
            if isinstance(value, (int, float)):
                if value > self.threshold:
                    return self.threshold
                if value < -self.threshold:
                    return -self.threshold
                return value
            if isinstance(value, list):
                return [_sanitize(v) for v in value]
            if isinstance(value, tuple):
                return tuple(_sanitize(v) for v in value)
            if isinstance(value, dict):
                return {k: _sanitize(v) for k, v in value.items()}
            return value

        return _sanitize(data)


# ---------------------------------------------------------------------------
# Backdoor trigger scanning
# ---------------------------------------------------------------------------


@dataclass
class BackdoorScanner:
    """Detects simple backdoor trigger strings.

    The scanner looks for a configured ``trigger_token`` within any string
    value and removes it during cleaning.
    """

    trigger_token: str = "trigger"

    def scan(self, data: Any) -> bool:
        found = False

        def _check(value: Any) -> None:
            nonlocal found
            if isinstance(value, str) and self.trigger_token in value:
                found = True
            elif isinstance(value, (list, tuple)):
                for v in value:
                    _check(v)
            elif isinstance(value, dict):
                for v in value.values():
                    _check(v)

        _check(data)
        return found

    def clean(self, data: Any) -> Any:
        def _clean(value: Any) -> Any:
            if isinstance(value, str):
                return value.replace(self.trigger_token, "")
            if isinstance(value, list):
                return [_clean(v) for v in value]
            if isinstance(value, tuple):
                return tuple(_clean(v) for v in value)
            if isinstance(value, dict):
                return {k: _clean(v) for k, v in value.items()}
            return value

        return _clean(data)


# ---------------------------------------------------------------------------
# Memory integrity checking
# ---------------------------------------------------------------------------


@dataclass
class MemoryIntegrityChecker:
    """Maintains a baseline snapshot and repairs deviations."""

    baseline: Dict[str, Any] | None = field(default=None, init=False)

    def scan(self, memory: Dict[str, Any]) -> List[str]:
        """Return keys whose values differ from the baseline.

        The first call establishes the baseline and returns an empty list.
        Subsequent calls compare against this snapshot.
        """

        if self.baseline is None:
            self.baseline = copy.deepcopy(memory)
            return []
        return [k for k, v in memory.items() if self.baseline.get(k) != v]

    def repair(self, memory: Dict[str, Any]) -> None:
        """Restore memory to the baseline snapshot."""
        if self.baseline is None:
            self.baseline = copy.deepcopy(memory)
            return
        for k, v in self.baseline.items():
            memory[k] = copy.deepcopy(v)


# ---------------------------------------------------------------------------
# Security guard orchestrator
# ---------------------------------------------------------------------------


@dataclass
class NeuralSecurityGuard:
    """Coordinates security checks for neural processing."""

    adversarial_detector: AdversarialInputDetector = field(
        default_factory=AdversarialInputDetector
    )
    backdoor_scanner: BackdoorScanner = field(default_factory=BackdoorScanner)
    memory_checker: MemoryIntegrityChecker = field(
        default_factory=MemoryIntegrityChecker
    )
    # The guard enters ``isolated`` mode whenever a threat is detected.  The
    # flag is cleared once recovery logic runs.
    isolated: bool = field(default=False, init=False)

    # -- Input validation ----------------------------------------------------
    def validate_neural_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect and clean adversarial or backdoor patterns in input data."""
        if self.adversarial_detector.detect(input_data):
            input_data = self.adversarial_detector.sanitize(input_data)
        if self.backdoor_scanner.scan(input_data):
            input_data = self.backdoor_scanner.clean(input_data)
        return input_data

    # -- Memory protection ---------------------------------------------------
    def protect_neural_memory(self, memory: Dict[str, Any]) -> List[str]:
        """Scan memory for corruption and repair deviations."""
        issues = self.memory_checker.scan(memory)
        if issues:
            self.memory_checker.repair(memory)
        return issues

    # -- Combined monitoring -------------------------------------------------
    def monitor_and_harden(
        self, input_data: Dict[str, Any], memory: Dict[str, Any]
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Run all checks and isolate/recover on detected threats.

        Returns a tuple of the possibly sanitised ``input_data`` and a dictionary
        summarising any issues found.  If a threat is detected the guard enters
        an isolated state, repairs memory and then exits isolation to simulate a
        restart or downgrade of the affected module.
        """

        issues: Dict[str, Any] = {"adversarial": False, "backdoor": False, "memory": []}

        if self.adversarial_detector.detect(input_data):
            self.isolated = True
            issues["adversarial"] = True
            input_data = self.adversarial_detector.sanitize(input_data)

        if self.backdoor_scanner.scan(input_data):
            self.isolated = True
            issues["backdoor"] = True
            input_data = self.backdoor_scanner.clean(input_data)

        mem_issues = self.memory_checker.scan(memory)
        if mem_issues:
            self.isolated = True
            issues["memory"] = mem_issues

        if self.isolated:
            # Recovery phase: repair memory and exit isolation to mimic a
            # restart/downgrade of the component.
            self.memory_checker.repair(memory)
            self.isolated = False

        return input_data, issues


__all__ = [
    "AdversarialInputDetector",
    "BackdoorScanner",
    "MemoryIntegrityChecker",
    "NeuralSecurityGuard",
]
