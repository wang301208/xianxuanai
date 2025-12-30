from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from ..performance import profile_brain_performance


@dataclass
class SelfImprovementManager:
    """Generate learning objectives from performance metrics and feedback.

    The manager observes simple performance measurements and optional
    external feedback to set new learning goals.  Goals are expressed as a
    target loss the system should aim for in the next learning cycle and a
    difficulty parameter controlling task sampling.
    """

    history: List[float] = field(default_factory=list)

    def generate_goal(
        self,
        spikes: Sequence[float],
        currents: Sequence[float],
        latencies: Sequence[float],
        feedback: float | None = None,
    ) -> Dict[str, float]:
        """Return a new learning goal based on current performance.

        Parameters
        ----------
        spikes, currents, latencies:
            Raw performance measurements forwarded to
            :func:`profile_brain_performance`.
        feedback:
            Optional numeric guidance from external sources.  Positive values
            make the manager more ambitious by lowering the target loss.
        """

        metrics, _ = profile_brain_performance(list(spikes), list(currents), list(latencies))
        loss = metrics["latency"]["average_latency"]
        self.history.append(loss)

        reduction = 0.1 * loss
        if feedback is not None:
            reduction += float(feedback)
        target_loss = max(loss - reduction, 0.0)

        # A simple curriculum: keep difficulty constant for stability.
        # Future versions could adjust this based on domain knowledge.
        difficulty = 1

        return {"target_loss": target_loss, "difficulty": difficulty}
