from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import pytest

if "auto_gpt_plugin_template" not in sys.modules:
    plugin_module = types.ModuleType("auto_gpt_plugin_template")

    class AutoGPTPluginTemplate:  # pragma: no cover - test stub
        pass

    plugin_module.AutoGPTPluginTemplate = AutoGPTPluginTemplate
    sys.modules["auto_gpt_plugin_template"] = plugin_module

if "events" not in sys.modules:
    events_module = types.ModuleType("events")

    class _StubEventBus:  # pragma: no cover - test stub
        pass

    def _create_event_bus(*args, **kwargs):  # pragma: no cover - test stub
        return _StubEventBus()

    events_module.EventBus = _StubEventBus
    events_module.create_event_bus = _create_event_bus
    sys.modules["events"] = events_module

    client_module = types.ModuleType("events.client")

    class _StubEventClient:  # pragma: no cover - test stub
        def __init__(self, bus):
            self.bus = bus

    client_module.EventClient = _StubEventClient
    sys.modules["events.client"] = client_module

    coordination_module = types.ModuleType("events.coordination")

    class _StubTaskStatus:  # pragma: no cover - test stub
        pass

    class _StubTaskStatusEvent:  # pragma: no cover - test stub
        pass

    coordination_module.TaskStatus = _StubTaskStatus
    coordination_module.TaskStatusEvent = _StubTaskStatusEvent
    sys.modules["events.coordination"] = coordination_module

if "monitoring" not in sys.modules:
    monitoring_module = types.ModuleType("monitoring")

    class WorkspaceMessage:  # pragma: no cover - test stub
        pass

    monitoring_module.WorkspaceMessage = WorkspaceMessage
    monitoring_module.global_workspace = object()
    sys.modules["monitoring"] = monitoring_module

if "openai" not in sys.modules:
    openai_module = types.ModuleType("openai")
    exceptions_module = types.ModuleType("openai._exceptions")
    base_client_module = types.ModuleType("openai._base_client")
    types_module = types.ModuleType("openai.types")
    chat_types_module = types.ModuleType("openai.types.chat")

    class APIStatusError(Exception):  # pragma: no cover - test stub
        pass

    class RateLimitError(Exception):  # pragma: no cover - test stub
        pass

    class CreateEmbeddingResponse:  # pragma: no cover - test stub
        pass

    class ChatCompletion:  # pragma: no cover - test stub
        pass

    class ChatCompletionMessage:  # pragma: no cover - test stub
        pass

    class ChatCompletionMessageParam:  # pragma: no cover - test stub
        pass

    exceptions_module.APIStatusError = APIStatusError
    exceptions_module.RateLimitError = RateLimitError
    types_module.CreateEmbeddingResponse = CreateEmbeddingResponse
    chat_types_module.ChatCompletion = ChatCompletion
    chat_types_module.ChatCompletionMessage = ChatCompletionMessage
    chat_types_module.ChatCompletionMessageParam = ChatCompletionMessageParam
    types_module.chat = chat_types_module
    base_client_module.log = logging.getLogger("openai-stub")
    openai_module._exceptions = exceptions_module
    openai_module._base_client = base_client_module
    openai_module.types = types_module
    sys.modules["openai"] = openai_module
    sys.modules["openai._exceptions"] = exceptions_module
    sys.modules["openai._base_client"] = base_client_module
    sys.modules["openai.types"] = types_module
    sys.modules["openai.types.chat"] = chat_types_module

if "playsound" not in sys.modules:
    playsound_module = types.ModuleType("playsound")

    def playsound(*args, **kwargs):  # pragma: no cover - test stub
        return None

    playsound_module.playsound = playsound
    sys.modules["playsound"] = playsound_module

if "docx" not in sys.modules:
    docx_module = types.ModuleType("docx")

    class Document:  # pragma: no cover - test stub
        pass

    docx_module.Document = Document
    sys.modules["docx"] = docx_module

if "pypdf" not in sys.modules:
    pypdf_module = types.ModuleType("pypdf")

    class PdfReader:  # pragma: no cover - test stub
        def __init__(self, *args, **kwargs):
            pass

    pypdf_module.PdfReader = PdfReader
    sys.modules["pypdf"] = pypdf_module

if "bs4" not in sys.modules:
    bs4_module = types.ModuleType("bs4")

    class BeautifulSoup:  # pragma: no cover - test stub
        def __init__(self, *args, **kwargs):
            self.text = ""

    bs4_module.BeautifulSoup = BeautifulSoup
    sys.modules["bs4"] = bs4_module

if "pylatexenc" not in sys.modules:
    pylatexenc_module = types.ModuleType("pylatexenc")
    latex2text_module = types.ModuleType("pylatexenc.latex2text")

    class LatexNodes2Text:  # pragma: no cover - test stub
        def latex_to_text(self, text: str) -> str:
            return text

    latex2text_module.LatexNodes2Text = LatexNodes2Text
    pylatexenc_module.latex2text = latex2text_module
    sys.modules["pylatexenc"] = pylatexenc_module
    sys.modules["pylatexenc.latex2text"] = latex2text_module

if "torch" not in sys.modules:
    torch_module = types.ModuleType("torch")

    class _Tensor:  # pragma: no cover - test stub
        def __init__(self, data, dtype=None):
            self._data = data

        def tolist(self):
            if isinstance(self._data, (list, tuple)):
                return list(self._data)
            return [self._data]

        def numel(self):
            return len(self._data) if hasattr(self._data, "__len__") else 1

        def argmax(self):
            return self

        def item(self):
            return 0

    def as_tensor(data, dtype=None):  # pragma: no cover - test stub
        return _Tensor(data, dtype)

    torch_module.Tensor = _Tensor
    torch_module.as_tensor = as_tensor
    nn_module = types.ModuleType("torch.nn")
    optim_module = types.ModuleType("torch.optim")
    functional_module = types.ModuleType("torch.nn.functional")

    def _noop(*args, **kwargs):  # pragma: no cover - test stub
        return args[0] if args else None

    functional_module.softmax = _noop
    functional_module.relu = _noop

    class _Module:  # pragma: no cover - test stub
        def __init__(self, *args, **kwargs):
            pass

    class _Optimizer:  # pragma: no cover - test stub
        def __init__(self, *args, **kwargs):
            pass

    torch_module.nn = nn_module
    torch_module.optim = optim_module
    nn_module.Module = _Module
    nn_module.functional = functional_module
    optim_module.Optimizer = _Optimizer
    sys.modules["torch"] = torch_module
    sys.modules["torch.nn"] = nn_module
    sys.modules["torch.nn.functional"] = functional_module
    sys.modules["torch.optim"] = optim_module

if "forge" not in sys.modules:
    forge_module = types.ModuleType("forge")
    sdk_module = types.ModuleType("forge.sdk")
    model_module = types.ModuleType("forge.sdk.model")

    class _Task:  # pragma: no cover - test stub
        def __init__(
            self,
            input: str,
            additional_input=None,
            created_at=None,
            modified_at=None,
            task_id: str = "",
            artifacts=None,
        ) -> None:
            self.input = input
            self.additional_input = additional_input
            self.created_at = created_at
            self.modified_at = modified_at
            self.task_id = task_id
            self.artifacts = list(artifacts or [])

        @classmethod
        def __get_validators__(cls):
            yield cls.validate

        @classmethod
        def validate(cls, value):
            return value

    model_module.Task = _Task
    sdk_module.model = model_module
    forge_module.sdk = sdk_module
    sys.modules["forge"] = forge_module
    sys.modules["forge.sdk"] = sdk_module
    sys.modules["forge.sdk.model"] = model_module

if "reasoning" not in sys.modules:
    reasoning_module = types.ModuleType("reasoning")
    decision_module = types.ModuleType("reasoning.decision_engine")

    class ActionDirective:  # pragma: no cover - test stub
        pass

    decision_module.ActionDirective = ActionDirective
    reasoning_module.decision_engine = decision_module
    sys.modules["reasoning"] = reasoning_module
    sys.modules["reasoning.decision_engine"] = decision_module

from autogpt.agents.base import BaseAgent
from autogpt.core.brain.config import BrainBackend, BrainSimulationConfig
from modules.brain.backends import BrainSimulationSystemAdapter


class DummyPromptStrategy:
    pass


class DummyLLMProvider:
    name = "dummy"


class DummyCommandRegistry:
    commands: dict[str, object] = {}


class DummyFileStorage:
    def __init__(self, root: Path) -> None:
        self.root = root


class DummyLegacyConfig:
    event_bus_backend = "memory"
    event_bus_redis_host = "localhost"
    event_bus_redis_port = 6379
    event_bus_redis_password = None


class DummyEventBus:
    pass


class DummyEventClient:
    def __init__(self, bus: DummyEventBus) -> None:
        self.bus = bus


class MinimalAgent(BaseAgent):
    async def execute(
        self,
        command_name: str,
        command_args: dict[str, str] = {},
        user_input: str = "",
    ) -> None:
        raise NotImplementedError

    def parse_and_process_response(self, llm_response, prompt, scratchpad):
        return "", {}, {}


def test_brain_simulation_reuse_preserves_runtime_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "autogpt.agents.base.create_event_bus", lambda *_, **__: DummyEventBus()
    )
    monkeypatch.setattr(
        "autogpt.agents.base.EventClient", lambda bus: DummyEventClient(bus)
    )

    brain_module = types.ModuleType("BrainSimulationSystem")
    sim_module = types.ModuleType("BrainSimulationSystem.brain_simulation")

    class StubBrainSimulation:
        def __init__(self, overrides=None, profile=None, stage=None):
            self.profile = profile
            self.stage = stage
            self.config = {"runtime": {"baseline": True}}
            self.updates: list[dict] = []
            if overrides:
                self.update_parameters(overrides)

        def update_parameters(self, overrides):
            self.updates.append(dict(overrides))
            runtime = overrides.get("runtime")
            if isinstance(runtime, dict):
                self.config.setdefault("runtime", {}).update(runtime)
            elif runtime is not None:
                self.config["runtime"] = runtime
            for key, value in overrides.items():
                if key != "runtime":
                    self.config[key] = value

    sim_module.BrainSimulation = StubBrainSimulation
    brain_module.brain_simulation = sim_module
    monkeypatch.setitem(sys.modules, "BrainSimulationSystem", brain_module)
    monkeypatch.setitem(sys.modules, "BrainSimulationSystem.brain_simulation", sim_module)

    adapter = BrainSimulationSystemAdapter(
        BrainSimulationConfig(overrides={"runtime": {"baseline": True}})
    )

    settings = BaseAgent.default_settings.copy(deep=True)
    settings.config.brain_backend = BrainBackend.BRAIN_SIMULATION
    settings.config.brain_simulation = BrainSimulationConfig(
        overrides={"runtime": {"injected": {"flag": True}}}
    )

    MinimalAgent(
        settings=settings,
        llm_provider=DummyLLMProvider(),
        prompt_strategy=DummyPromptStrategy(),
        command_registry=DummyCommandRegistry(),
        file_storage=DummyFileStorage(tmp_path),
        legacy_config=DummyLegacyConfig(),
        whole_brain=adapter,
    )

    runtime = adapter._brain.config.get("runtime")
    assert isinstance(runtime, dict)
    assert runtime.get("baseline") is True
    assert runtime.get("injected", {}).get("flag") is True
