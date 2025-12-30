import logging
import os
import sys
from pathlib import Path

import pytest

# Allow importing the autogpt package
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "autogpts", "autogpt")))

# Stub out optional dependencies that are not required for this test
import types

# Stub modules that may not be installed in the test environment
inflection_mod = types.ModuleType("inflection")
inflection_mod.underscore = lambda s: s
inflection_mod.camelize = lambda s: s
sys.modules["inflection"] = inflection_mod

sentry_sdk_mod = types.ModuleType("sentry_sdk")
sentry_sdk_mod.init = lambda *args, **kwargs: None
sys.modules["sentry_sdk"] = sentry_sdk_mod

jsonschema_mod = types.ModuleType("jsonschema")
class _DummyValidator:
    def __init__(self, *args, **kwargs):
        pass
    @staticmethod
    def check_schema(*args, **kwargs):
        return True

class _ValidationError(Exception):
    pass

jsonschema_mod.Draft7Validator = _DummyValidator
jsonschema_mod.ValidationError = _ValidationError
sys.modules["jsonschema"] = jsonschema_mod

demjson3_mod = types.ModuleType("demjson3")
demjson3_mod.decode = lambda *args, **kwargs: {}
sys.modules["demjson3"] = demjson3_mod

sys.modules.setdefault("auto_gpt_plugin_template", types.SimpleNamespace(AutoGPTPluginTemplate=object))

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

import heapq
from third_party.autogpt.autogpt.core.agent.simple import SimpleAgent
from third_party.autogpt.autogpt.core.planning.schema import Task, TaskStatus, TaskType
from third_party.autogpt.autogpt.core.ability import AbilityResult


class DummyAbilityRegistry:
    def list_abilities(self):
        return []

    def dump_abilities(self):  # pragma: no cover - stub
        return []


class DummyPlanner:
    pass


class DummyMemory:
    def __init__(self):
        self._items = []

    def add(self, *items):
        self._items.extend(items)

    def get(self):
        return self._items


class DummyWorkspace:
    def __init__(self, root: Path):
        self.root = root

    def get_path(self, name: str) -> Path:
        return self.root / name


@pytest.mark.asyncio
async def test_tasks_marked_complete_when_acceptance_criteria_met(tmp_path):
    agent = SimpleAgent(
        settings=SimpleAgent.default_settings,
        logger=logging.getLogger("test"),
        ability_registry=DummyAbilityRegistry(),
        memory=DummyMemory(),
        model_providers={},
        planning=DummyPlanner(),
        workspace=DummyWorkspace(tmp_path),
    )

    current_task = Task(
        objective="Task 1",
        type=TaskType.TEST,
        priority=1,
        ready_criteria=["start"],
        acceptance_criteria=["done"],
    )

    other_task = Task(
        objective="Task 2",
        type=TaskType.TEST,
        priority=2,
        ready_criteria=["start"],
        acceptance_criteria=["finish"],
    )
    other_task.context.status = TaskStatus.DONE
    heapq.heappush(agent._task_queue, (current_task.priority, current_task))
    heapq.heappush(agent._task_queue, (other_task.priority, other_task))
    agent._task_queue[0][1].context.status = TaskStatus.READY
    _, task = heapq.heappop(agent._task_queue)
    agent._current_task = task

    result = AbilityResult(
        ability_name="test",
        ability_args={},
        success=True,
        message="This task is done",
    )

    await agent._update_tasks_and_memory(result)

    assert agent._current_task.context.status == TaskStatus.DONE
    assert agent._task_queue == []
    assert agent._completed_tasks == [other_task]

    # Simulate post-execution queue handling
    if agent._current_task.context.status == TaskStatus.DONE:
        agent._completed_tasks.append(agent._current_task)
    else:  # pragma: no cover - not expected
        heapq.heappush(agent._task_queue, (agent._current_task.priority, agent._current_task))
        agent._task_queue[0][1].context.status = TaskStatus.READY

    assert current_task in agent._completed_tasks
