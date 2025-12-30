"""Developmental growth orchestration utilities.

This module bridges the curriculum-stage runtime policy (metrics -> promotion)
with the :class:`~BrainSimulationSystem.brain_simulation.BrainSimulation` stage
presets from :mod:`BrainSimulationSystem.config.stage_profiles`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from BrainSimulationSystem.brain_simulation import BrainSimulation
from BrainSimulationSystem.core.stage_manager import CurriculumStageManager, StageTransition


@dataclass
class DevelopmentalGrowthController:
    """Apply :class:`~BrainSimulationSystem.core.stage_manager.CurriculumStageManager` transitions to a simulation."""

    simulation: BrainSimulation
    stage_manager: CurriculumStageManager
    preserve_state_on_promotion: bool = True

    def ingest_events(self, events: Sequence[Any]) -> Optional[StageTransition]:
        """Feed metric events and upgrade the simulation if a promotion occurs."""

        transition = self.stage_manager.ingest_events(events)
        if transition is None:
            return None
        self.simulation.upgrade_stage(
            transition.current.key,
            base_profile=transition.current.base_profile,
            preserve_state=self.preserve_state_on_promotion,
        )
        return transition

    def step(self) -> Optional[StageTransition]:
        """Poll the attached collector and upgrade the simulation if a promotion occurs."""

        transition = self.stage_manager.step()
        if transition is None:
            return None
        self.simulation.upgrade_stage(
            transition.current.key,
            base_profile=transition.current.base_profile,
            preserve_state=self.preserve_state_on_promotion,
        )
        return transition

