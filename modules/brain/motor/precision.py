class PrimaryMotorCortex:
    """Primary motor cortex responsible for executing motor commands."""

    def execute(self, command: str) -> str:
        return f"executed {command}"


from numbers import Real


class DorsalPremotor:
    """Dorsal premotor area planning reaching trajectories."""

    def plan(self, intention: str) -> str:
        return f"dorsal plan for {intention}"


class VentralPremotor:
    """Ventral premotor area planning grasping movements."""

    def plan(self, intention: str) -> str:
        return f"ventral plan for {intention}"


class SupplementaryMotorArea:
    """Supplementary motor area coordinating complex sequences."""

    def coordinate(self, plan: str) -> str:
        return f"SMA coordination of {plan}"


class PreSupplementaryMotorArea:
    """Pre-supplementary motor area organizing abstract plans."""

    def organize(self, intention: str) -> str:
        return f"preSMA organization of {intention}"


class BasalGangliaCircuit:
    """Basal ganglia circuit gating and modulating motor plans."""

    def __init__(self):
        self.gating_history = []

    def gate(self, plan: str) -> str:
        self.gating_history.append(plan)
        return f"{plan} [BG]"


class Cerebellum:
    """Cerebellum fine-tuning motor commands and learning from feedback."""

    def __init__(self):
        self.learned = []
        self.metric_history: list[dict[str, float]] = []

    def fine_tune(self, command: str) -> str:
        if self.learned:
            return f"fine-tuned {command} with {self.learned[-1]}"
        return f"fine-tuned {command}"

    def learn(self, feedback: str | dict[str, float]) -> str:
        if isinstance(feedback, dict):
            summary = self.update_feedback(feedback)
            return f"adapted metrics {summary}" if summary else "adapted metrics"
        self.learned.append(feedback)
        return f"adapted to {feedback}"

    def update_feedback(self, metrics: dict[str, float]) -> str | None:
        numeric: dict[str, float] = {}
        for key, value in metrics.items():
            if isinstance(value, Real):
                numeric[key] = float(value)
        if not numeric:
            return None
        summary = ", ".join(f"{key}={value:.3f}" for key, value in sorted(numeric.items()))
        self.metric_history.append(dict(sorted(numeric.items())))
        self.learned.append(summary)
        return summary


class MotorPlanningSystem:
    """System orchestrating motor planning and trajectory optimization."""

    def __init__(self, dorsal: DorsalPremotor, ventral: VentralPremotor, sma: SupplementaryMotorArea, psma: PreSupplementaryMotorArea):
        self.dorsal = dorsal
        self.ventral = ventral
        self.sma = sma
        self.psma = psma

    def create_plan(self, intention: str) -> str:
        plan = self.psma.organize(intention)
        plan = self.sma.coordinate(plan)
        plan = self.dorsal.plan(plan)
        plan = self.ventral.plan(plan)
        return plan

    def optimize_trajectory(self, plan: str) -> str:
        return f"{plan} optimized for obstacles and forces"


class PrecisionMotorSystem:
    """High precision motor system integrating cortical, basal ganglia, and cerebellar circuits."""

    def __init__(self):
        self.primary_motor = PrimaryMotorCortex()
        self.dorsal_premotor = DorsalPremotor()
        self.ventral_premotor = VentralPremotor()
        self.sma = SupplementaryMotorArea()
        self.psma = PreSupplementaryMotorArea()
        self.basal_ganglia = BasalGangliaCircuit()
        self.cerebellum = Cerebellum()
        self.planner = MotorPlanningSystem(
            self.dorsal_premotor, self.ventral_premotor, self.sma, self.psma
        )

    def plan_movement(self, intention: str) -> str:
        """Plan movement using motor hierarchy, basal ganglia gating and trajectory optimization."""
        plan = self.planner.create_plan(intention)
        plan = self.basal_ganglia.gate(plan)
        plan = self.planner.optimize_trajectory(plan)
        return plan

    def execute_action(self, plan: str) -> str:
        """Execute an action with basal ganglia gating and cerebellar fine-tuning."""
        gated = self.basal_ganglia.gate(plan)
        tuned = self.cerebellum.fine_tune(gated)
        return self.primary_motor.execute(tuned)

    def learn(self, feedback: str) -> str:
        """Update cerebellar learning based on feedback."""
        return self.cerebellum.learn(feedback)

    def update_feedback(self, metrics: dict[str, float]) -> None:
        """Propagate structured metric feedback to the cerebellum."""

        self.cerebellum.update_feedback(metrics)
