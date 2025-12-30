import sys
import types
from pathlib import Path
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend" / "autogpt"))

# Stub configuration modules
autogpt_config = types.ModuleType("autogpt.config")
class Config: ...
class ConfigBuilder:
    default_settings = types.SimpleNamespace(prompt_settings_file="")
autogpt_config.Config = Config
autogpt_config.ConfigBuilder = ConfigBuilder

class AIProfile(BaseModel):
    ai_name: str = "AutoGPT"

class AIDirectives(BaseModel):
    @classmethod
    def from_file(cls, *args, **kwargs):
        return cls()

autogpt_config.AIProfile = AIProfile
autogpt_config.AIDirectives = AIDirectives
autogpt_config.__path__ = []
sys.modules["autogpt.config"] = autogpt_config
sys.modules["autogpt.config.ai_profile"] = types.ModuleType("autogpt.config.ai_profile")
sys.modules["autogpt.config.ai_profile"].AIProfile = AIProfile
sys.modules["autogpt.config.ai_directives"] = types.ModuleType("autogpt.config.ai_directives")
sys.modules["autogpt.config.ai_directives"].AIDirectives = AIDirectives

# Stub events
events_module = types.ModuleType("events")
class EventBus: ...

def create_event_bus(*args, **kwargs):
    return EventBus()

events_module.EventBus = EventBus
events_module.create_event_bus = create_event_bus
client_module = types.ModuleType("events.client")
class EventClient:
    def __init__(self, bus):
        self.bus = bus
    def publish(self, *args, **kwargs):
        pass
client_module.EventClient = EventClient
coord_module = types.ModuleType("events.coordination")
class TaskStatus: ...
class TaskStatusEvent:
    def __init__(self, *args, **kwargs):
        pass
    def to_dict(self):
        return {}
coord_module.TaskStatus = TaskStatus
coord_module.TaskStatusEvent = TaskStatusEvent
sys.modules["events"] = events_module
sys.modules["events.client"] = client_module
sys.modules["events.coordination"] = coord_module

# Stub forge Task
forge_module = types.ModuleType("forge")
sdk_module = types.ModuleType("forge.sdk")
model_module = types.ModuleType("forge.sdk.model")
class Task(BaseModel):
    input: str = ""
    additional_input: str | None = None
    created_at: object | None = None
    modified_at: object | None = None
    task_id: str = ""
    artifacts: list = []
model_module.Task = Task
sdk_module.model = model_module
forge_module.sdk = sdk_module
sys.modules["forge"] = forge_module
sys.modules["forge.sdk"] = sdk_module
sys.modules["forge.sdk.model"] = model_module

# Stub context features
agents_features_pkg = types.ModuleType("autogpt.agents.features")
context_module = types.ModuleType("autogpt.agents.features.context")
class AgentContext:
    def __init__(self, items=None):
        self.items = items or []
    def add(self, item):
        self.items.append(item)
class ContextMixin:
    def __init__(self, **kwargs):
        self.context = AgentContext()
        super().__init__(**kwargs)

def get_agent_context(agent):
    return getattr(agent, "context", None)
context_module.AgentContext = AgentContext
context_module.ContextMixin = ContextMixin
context_module.get_agent_context = get_agent_context
sys.modules["autogpt.agents.features"] = agents_features_pkg
sys.modules["autogpt.agents.features.context"] = context_module

# Stub context item model
context_item_module = types.ModuleType("autogpt.models.context_item")
class StaticContextItem:
    def __init__(self, description, source, content):
        self.description = description
        self.source = source
        self.content = content
    def fmt(self):
        return f"{self.description} (source: {self.source})\n```\n{self.content}\n```"
context_item_module.StaticContextItem = StaticContextItem
class ContextItem: ...
context_item_module.ContextItem = ContextItem
sys.modules["autogpt.models.context_item"] = context_item_module

# Other stubs
sys.modules["sentry_sdk"] = types.ModuleType("sentry_sdk")
sys.modules["tenacity"] = types.ModuleType("tenacity")
sys.modules["tiktoken"] = types.ModuleType("tiktoken")
sys.modules["demjson3"] = types.ModuleType("demjson3")
sys.modules["docx"] = types.ModuleType("docx")
sys.modules["pypdf"] = types.ModuleType("pypdf")
bs4_module = types.ModuleType("bs4")
class BeautifulSoup:
    pass
bs4_module.BeautifulSoup = BeautifulSoup
sys.modules["bs4"] = bs4_module
latex_module = types.ModuleType("pylatexenc")
latex_submodule = types.ModuleType("pylatexenc.latex2text")
class LatexNodes2Text:
    def __init__(self, *args, **kwargs):
        pass
latex_submodule.LatexNodes2Text = LatexNodes2Text
sys.modules["pylatexenc"] = latex_module
sys.modules["pylatexenc.latex2text"] = latex_submodule
spacy_module = types.ModuleType("spacy")
spacy_module.load = lambda *args, **kwargs: None
sys.modules["spacy"] = spacy_module
torch_module = types.ModuleType("torch")
torch_module.nn = types.ModuleType("torch.nn")
torch_module.nn.Module = type("Module", (), {})
sys.modules["torch"] = torch_module
sys.modules["torch.nn"] = torch_module.nn

from third_party.autogpt.autogpt.agents.base import BaseAgent, PromptScratchpad
from third_party.autogpt.autogpt.agents.features.context import ContextMixin, AgentContext
from third_party.autogpt.autogpt.models.action_history import EpisodicActionHistory
from third_party.autogpt.autogpt.models.action_history import ActionResult, ActionSuccessResult
from third_party.autogpt.autogpt.core.prompting.schema import ChatPrompt, ChatMessage
from third_party.autogpt.autogpt.core.resource.model_providers.schema import (
    AssistantChatMessage,
    ChatModelResponse,
    ChatModelInfo,
    ModelProviderService,
    ModelProviderName,
)


class DummyAgent(ContextMixin, BaseAgent):
    async def execute(self, command_name: str, command_args: dict[str, str] | None = None, user_input: str = "") -> ActionResult:
        return ActionSuccessResult(outputs="done")

    def parse_and_process_response(self, llm_response, prompt, scratchpad):
        return llm_response.parsed_result


def make_agent() -> DummyAgent:
    agent = DummyAgent.__new__(DummyAgent)
    agent.event_history = EpisodicActionHistory()
    agent.context = AgentContext()
    return agent


def test_on_response_updates_memory_and_context():
    agent = make_agent()

    model_info = ChatModelInfo(
        name="test-model",
        service=ModelProviderService.CHAT,
        provider_name=ModelProviderName.OPENAI,
        prompt_token_cost=0.0,
        completion_token_cost=0.0,
        max_tokens=1000,
        has_function_call_api=False,
    )

    parsed_result = (
        "test_cmd",
        {"arg": "val"},
        {"thoughts": {"reasoning": "Because"}, "context": "ctx info"},
    )

    llm_response = ChatModelResponse[
        BaseAgent.ThoughtProcessOutput
    ](
        response=AssistantChatMessage(content="test"),
        parsed_result=parsed_result,
        prompt_tokens_used=0,
        completion_tokens_used=0,
        model_info=model_info,
    )

    prompt = ChatPrompt(messages=[ChatMessage.system("sys")])
    scratchpad = PromptScratchpad()

    result = agent.on_response(llm_response, prompt, scratchpad)

    assert result == parsed_result
    assert len(agent.event_history.episodes) == 1
    ep = agent.event_history.episodes[0]
    assert ep.action.name == "test_cmd"
    assert ep.action.args == {"arg": "val"}
    assert ep.action.reasoning == "Because"

    assert len(agent.context.items) == 1
    assert agent.context.items[0].content == "ctx info"
