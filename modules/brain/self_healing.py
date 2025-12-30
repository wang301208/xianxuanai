from __future__ import annotations

"""Self-healing brain utilities.

This module combines performance monitoring and security checks to
produce diagnostic reports and enact automatic repairs for simple
simulations.  The implementation intentionally favours clarity over
complexity so it can serve as a lightweight example for unit tests."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
import copy

from .performance import profile_brain_performance, auto_optimize_performance
from .security import NeuralSecurityGuard


@dataclass
class SelfHealingBrain:
    """Orchestrates diagnosis and repair using monitoring modules."""

    security_guard: NeuralSecurityGuard = field(default_factory=NeuralSecurityGuard)

    def comprehensive_diagnosis(
        self,
        spikes: List[float],
        currents: List[float],
        latencies: List[float],
        input_data: Dict[str, Any],
        memory: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run performance profiling and security scans.

        Returns a nested dictionary containing performance metrics,
        suggestions, security findings and sanitized input.
        """

        # Performance analysis
        metrics, suggestions = profile_brain_performance(
            spikes, currents, latencies
        )

        # Security analysis on a copy so the caller's data remains unchanged
        original_input = copy.deepcopy(input_data)
        sanitized_input = self.security_guard.validate_neural_input(
            copy.deepcopy(input_data)
        )
        input_issues = sanitized_input != original_input

        memory_copy = copy.deepcopy(memory)
        memory_issues = self.security_guard.memory_checker.scan(memory_copy)

        report = {
            "performance": {"metrics": metrics, "suggestions": suggestions},
            "security": {
                "input_issues": input_issues,
                "sanitized_input": sanitized_input,
                "memory_issues": memory_issues,
            },
        }
        return report

    def generate_repair_plans(self, diagnosis: Dict[str, Any]) -> List[str]:
        """Create human readable repair actions from *diagnosis*."""

        plans: List[str] = []
        plans.extend(diagnosis["performance"]["suggestions"])
        if diagnosis["security"]["input_issues"]:
            plans.append(
                "Sanitize input data to remove adversarial patterns or triggers"
            )
        if diagnosis["security"]["memory_issues"]:
            plans.append(
                "Restore memory keys: "
                + ", ".join(diagnosis["security"]["memory_issues"])
            )
        return plans

    def execute_repairs(
        self,
        spikes: List[float],
        currents: List[float],
        latencies: List[float],
        input_data: Dict[str, Any],
        memory: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply optimisations and security repairs in-place.

        The returned dictionary summarises adjustments taken and resulting
        states of input data and memory.
        """

        performance_result = auto_optimize_performance(spikes, currents, latencies)
        sanitized_input = self.security_guard.validate_neural_input(input_data)
        memory_issues = self.security_guard.protect_neural_memory(memory)

        return {
            "performance": performance_result,
            "sanitized_input": sanitized_input,
            "memory_issues": memory_issues,
        }


__all__ = ["SelfHealingBrain"]
