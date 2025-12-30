import asyncio
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("inflection")


from third_party.autogpt.autogpt.core.ability.schema import AbilityResult
from third_party.autogpt.autogpt.core.agent.cognition import SimpleBrainAdapter
from third_party.autogpt.autogpt.core.agent.simple import SimpleAgent
from third_party.autogpt.autogpt.core.resource.model_providers.schema import CompletionModelFunction
from modules.brain.state import (
    BrainCycleResult,
    CognitiveIntent,
    CuriosityState,
    EmotionSnapshot,
    PerceptionSnapshot,
    PersonalityProfile,
)
from schemas.emotion import EmotionType


class _DummyActionLogger:
    def __init__(self, *args, **kwargs):
        self.records = []

    def log(self, payload):  # pragma: no cover - trivial
        self.records.append(payload)


sys.modules.setdefault(
    "monitoring", SimpleNamespace(ActionLogger=_DummyActionLogger)
)


class StubBrain:
    def __init__(self, **kwargs):
        self.cycles: list[dict] = []

    def process_cycle(self, input_data: dict) -> BrainCycleResult:
        self.cycles.append(input_data)
        intent = CognitiveIntent(
            intention="observe",
            salience=False,
            plan=["monitor_environment", "record_observations"],
            confidence=0.82,
            weights={"observe": 0.7},
            tags=["observe"],
        )
        perception = PerceptionSnapshot(modalities={})
        emotion = EmotionSnapshot(
            primary=EmotionType.CALM,
            intensity=0.2,
            mood=0.1,
            dimensions={},
            context={},
            decay=0.05,
        )
        curiosity = CuriosityState()
        personality = PersonalityProfile()
        return BrainCycleResult(
            perception=perception,
            emotion=emotion,
            intent=intent,
            personality=personality,
            curiosity=curiosity,
            energy_used=1,
            idle_skipped=0,
            thoughts=None,
            feeling=None,
            metrics={"intent_confidence": float(intent.confidence)},
            metadata={"cognitive_plan": ", ".join(intent.plan)},
        )


class StubMemory:
    def __init__(self) -> None:
        self.entries: list[str] = []

    def get_relevant(self, query, k, config):  # pragma: no cover - simple stub
        return []

    def add(self, item: str) -> None:  # pragma: no cover - simple stub
        self.entries.append(item)

    def get(self, limit: int | None = None):  # pragma: no cover - simple stub
        return list(self.entries)

    def get_scores_for_task(self, task_desc: str, ability_name: str):
        return []


class StubWorkspace:
    def __init__(self, root: Path) -> None:
        self.root = root

    def get_path(self, name: str) -> Path:  # pragma: no cover - simple stub
        return self.root / name


class StubAbility:
    def __init__(self) -> None:
        self.spec = CompletionModelFunction(
            name="self_assess",
            description="Review recent context",
            parameters={},
        )

    @staticmethod
    def name() -> str:  # pragma: no cover - trivial accessor
        return "self_assess"

    description = "Review recent context"

    async def __call__(self, **kwargs) -> AbilityResult:
        return AbilityResult(
            ability_name="self_assess",
            ability_args=kwargs,
            success=True,
            message="ok",
        )


class StubAbilityRegistry:
    def __init__(self) -> None:
        self._ability = StubAbility()

    def list_abilities(self):
        return ["self_assess: Review recent context"]

    def dump_abilities(self):
        return [self._ability.spec]

    def get_ability(self, name: str):  # pragma: no cover - simple lookup
        return self._ability


@pytest.mark.asyncio
async def test_simple_agent_uses_neuromorphic_backend_by_default(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "autogpt.core.agent.cognition.WholeBrainSimulation",
        lambda **kwargs: StubBrain(),
    )

    logger = logging.getLogger("simple-agent-neuromorphic")
    memory = StubMemory()
    workspace = StubWorkspace(tmp_path)
    ability_registry = StubAbilityRegistry()

    adapter_settings = SimpleBrainAdapter.default_settings.copy(deep=True)
    cognition = SimpleBrainAdapter(adapter_settings, logger.getChild("cognition"))

    agent_settings = SimpleAgent.default_settings.copy(deep=True)
    agent = SimpleAgent(
        settings=agent_settings,
        logger=logger,
        ability_registry=ability_registry,
        memory=memory,
        model_providers={},
        planning=None,
        workspace=workspace,
        cognition=cognition,
        creative_planning=None,
    )

    assert agent._use_neuromorphic_backend()  # type: ignore[attr-defined]

    plan = await agent.build_initial_plan()
    assert plan["backend"] == "whole_brain"
    assert cognition._brain.cycles  # type: ignore[attr-defined]

    task, ability_info = await agent.determine_next_ability()
    assert ability_info["backend"] == "whole_brain"
    assert ability_info["next_ability"] == "self_assess"
    assert cognition._brain.cycles  # type: ignore[attr-defined]
    assert len(cognition._brain.cycles) >= 2  # type: ignore[attr-defined]

    assert task.context.status == task.context.status  # sanity check access