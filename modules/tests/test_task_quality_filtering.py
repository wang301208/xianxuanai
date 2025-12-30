import logging
import os
import sys
from pathlib import Path

import pytest

# Allow importing the autogpt package
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "autogpts", "autogpt")))

# Stub out optional dependencies that are not required for this test
import types
import logging

google_cloud = types.ModuleType("google.cloud")
logging_v2 = types.ModuleType("google.cloud.logging_v2")
handlers = types.ModuleType("google.cloud.logging_v2.handlers")

class _CloudLoggingFilter(logging.Filter):
    pass

class _StructuredLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - stub
        pass

handlers.CloudLoggingFilter = _CloudLoggingFilter
handlers.StructuredLogHandler = _StructuredLogHandler
logging_v2.handlers = handlers
sys.modules["google"] = types.ModuleType("google")
sys.modules["google.cloud"] = google_cloud
sys.modules["google.cloud.logging_v2"] = logging_v2
sys.modules["google.cloud.logging_v2.handlers"] = handlers

# Stub optional speech dependencies
playsound_mod = types.ModuleType("playsound")
playsound_mod.playsound = lambda *args, **kwargs: None
sys.modules["playsound"] = playsound_mod

gtts_mod = types.ModuleType("gtts")
class _gTTS:
    def __init__(self, *args, **kwargs):
        pass

    def save(self, *args, **kwargs) -> None:  # pragma: no cover - stub
        pass

gtts_mod.gTTS = _gTTS
sys.modules["gtts"] = gtts_mod

monitoring_mod = types.ModuleType("monitoring")
class _ActionLogger:
    def __init__(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs) -> None:  # pragma: no cover - stub
        pass

monitoring_mod.ActionLogger = _ActionLogger
sys.modules["monitoring"] = monitoring_mod

from third_party.autogpt.autogpt.core.agent.simple import SimpleAgent
from third_party.autogpt.autogpt.core.planning.schema import TaskStatus, TaskType


class DummyPlan:
    def __init__(self, parsed_result):
        self.parsed_result = parsed_result


class DummyPlanner:
    def __init__(self, plan):
        self._plan = plan

    async def make_initial_plan(self, **kwargs):
        return DummyPlan(self._plan)


class DummyAbilityRegistry:
    def list_abilities(self):
        return []


class DummyMemory:
    pass


class DummyOpenAIProvider:
    pass


class DummyWorkspace:
    root = Path(".")


@pytest.mark.asyncio
async def test_low_quality_tasks_filtered_and_prioritized():
    plan = {
        "task_list": [
            {
                "objective": "Good task",
                "type": TaskType.TEST,
                "priority": 1,
                "ready_criteria": ["do it"],
                "acceptance_criteria": ["done"],
            },
            {
                "objective": "Bad task",
                "type": TaskType.TEST,
                "priority": 1,
                "ready_criteria": [],
                "acceptance_criteria": [],
            },
        ]
    }

    agent = SimpleAgent(
        settings=SimpleAgent.default_settings,
        logger=logging.getLogger("test"),
        ability_registry=DummyAbilityRegistry(),
        memory=DummyMemory(),
        model_providers={},
        planning=DummyPlanner(plan),
        workspace=DummyWorkspace(),
    )

    await agent.build_initial_plan()

    assert len(agent._task_queue) == 1
    priority, task = agent._task_queue[0]
    assert task.objective == "Good task"
    assert priority == 0  # original priority 1 minus score 1
    assert task.context.status == TaskStatus.READY
