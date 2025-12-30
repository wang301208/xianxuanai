import numpy as np
from datetime import datetime
import sys
import types
from pathlib import Path
from pydantic import BaseModel, Field


class Task(BaseModel):
    input: str
    additional_input: dict | None = None
    created_at: datetime
    modified_at: datetime
    task_id: str
    artifacts: list = Field(default_factory=list)


forge_module = types.ModuleType("forge")
sdk_module = types.ModuleType("forge.sdk")
model_module = types.ModuleType("forge.sdk.model")
model_module.Task = Task
sdk_module.model = model_module
forge_module.sdk = sdk_module
sys.modules.setdefault("forge", forge_module)
sys.modules.setdefault("forge.sdk", sdk_module)
sys.modules.setdefault("forge.sdk.model", model_module)
sys.modules.setdefault("spacy", types.ModuleType("spacy"))
torch_module = types.ModuleType("torch")
torch_module.nn = types.ModuleType("torch.nn")
torch_module.nn.Module = type("Module", (), {})
sys.modules.setdefault("torch", torch_module)
sys.modules.setdefault("torch.nn", torch_module.nn)


class EventBus: ...


def create_event_bus(*args, **kwargs):
    return EventBus()


class EventClient:
    def __init__(self, bus):
        self.bus = bus

    def publish(self, *args, **kwargs):
        pass


class TaskStatus: ...


class TaskStatusEvent:
    def __init__(self, *args, **kwargs):
        pass

    def to_dict(self):
        return {}


events_module = types.ModuleType("events")
events_client_module = types.ModuleType("events.client")
events_coord_module = types.ModuleType("events.coordination")
events_module.EventBus = EventBus
events_module.create_event_bus = create_event_bus
events_client_module.EventClient = EventClient
events_coord_module.TaskStatus = TaskStatus
events_coord_module.TaskStatusEvent = TaskStatusEvent

sys.modules.setdefault("events", events_module)
sys.modules.setdefault("events.client", events_client_module)
sys.modules.setdefault("events.coordination", events_coord_module)

features_pkg = types.ModuleType("autogpt.agents.features")
context_module = types.ModuleType("autogpt.agents.features.context")
agent_file_manager_module = types.ModuleType(
    "autogpt.agents.features.agent_file_manager"
)
watchdog_module = types.ModuleType("autogpt.agents.features.watchdog")


class ContextMixin: ...


class AgentFileManagerMixin: ...


class WatchdogMixin: ...


context_module.ContextMixin = ContextMixin
agent_file_manager_module.AgentFileManagerMixin = AgentFileManagerMixin
watchdog_module.WatchdogMixin = WatchdogMixin

sys.modules.setdefault("autogpt.agents.features", features_pkg)
sys.modules.setdefault("autogpt.agents.features.context", context_module)
sys.modules.setdefault(
    "autogpt.agents.features.agent_file_manager", agent_file_manager_module
)
sys.modules.setdefault("autogpt.agents.features.watchdog", watchdog_module)

from autogpt.agent_factory.configurators import create_agent
from autogpt.config import AIProfile, Config
from autogpt.memory.vector.memory_item import MemoryItem
from autogpt.models.command_registry import CommandRegistry


def test_agent_memory_injection(config: Config, storage):
    config.memory_backend = "json_file"
    object.__setattr__(config, "workspace_path", storage.root)
    config.plugins_dir = str(Path(storage.root) / "plugins")
    Path(config.plugins_dir).mkdir(parents=True, exist_ok=True)
    CommandRegistry.with_command_modules = classmethod(
        lambda cls, modules, config: CommandRegistry()
    )

    class DummyProvider:
        pass

    llm_provider = DummyProvider()

    task = Task(
        input="test task",
        additional_input=None,
        created_at=datetime.now(),
        modified_at=datetime.now(),
        task_id="task-id",
        artifacts=[],
    )

    ai_profile = AIProfile(
        ai_name="Test Agent",
        ai_role="Tester",
        ai_goals=["Test"],
    )

    agent = create_agent(
        agent_id="agent-id",
        task=task,
        ai_profile=ai_profile,
        app_config=config,
        file_storage=storage,
        llm_provider=llm_provider,
    )

    assert hasattr(agent, "memory")

    embedding = np.array([0.1, 0.2], dtype=np.float32)
    item = MemoryItem(
        raw_content="hello",
        summary="hello",
        chunks=["hello"],
        chunk_summaries=["hello"],
        e_summary=embedding,
        e_weighted=embedding,
        e_chunks=[embedding],
        metadata={"source": "test"},
    )

    agent.memory.add(item)
    assert item in agent.memory
