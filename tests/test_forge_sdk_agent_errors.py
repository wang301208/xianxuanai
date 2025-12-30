import pathlib
import sys
import types
from importlib import import_module
from unittest.mock import AsyncMock

import pytest

# Create a lightweight 'sdk' package pointing to the SDK source directory to avoid
# executing the heavy package __init__ that requires additional dependencies.
SDK_PATH = pathlib.Path(__file__).resolve().parent.parent / "backend/forge/forge/sdk"
sdk_pkg = types.ModuleType("sdk")
sdk_pkg.__path__ = [str(SDK_PATH)]
sys.modules.setdefault("sdk", sdk_pkg)

# Stub out optional dependencies used by the SDK modules during import
google_mod = types.ModuleType("google")
cloud_mod = types.ModuleType("cloud")
storage_mod = types.ModuleType("storage")
cloud_mod.storage = storage_mod
google_mod.cloud = cloud_mod
sys.modules.setdefault("google", google_mod)
sys.modules.setdefault("google.cloud", cloud_mod)
sys.modules.setdefault("google.cloud.storage", storage_mod)

agent_module = import_module("sdk.agent")
errors_module = import_module("sdk.errors")
model_module = import_module("sdk.model")

Agent = agent_module.Agent
DatabaseError = errors_module.DatabaseError
TaskRequestBody = model_module.TaskRequestBody


class _Workspace:
    """Minimal workspace stub for Agent tests."""

    def write(self, *args, **kwargs):
        pass


@pytest.mark.asyncio
async def test_create_task_database_error(monkeypatch):
    """Agent.create_task should surface DatabaseError from the DB layer."""

    db = AsyncMock()
    db.create_task.side_effect = DatabaseError("db failure")
    agent = Agent(db, _Workspace())
    task_request = TaskRequestBody(input="test")

    with pytest.raises(DatabaseError) as exc_info:
        await agent.create_task(task_request)

    assert "Failed to create task" in str(exc_info.value)
