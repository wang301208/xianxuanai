import logging
import sys
from pathlib import Path
import types
import enum

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party/autogpt"))

# Stub out heavy dependencies from third_party.autogpt.autogpt.config
config_stub = types.ModuleType("autogpt.config")
class AIProfile:
    ai_name = "GPT"
    ai_role = "assistant"
    api_budget = 0
class AIDirectives:
    constraints = []
    resources = []
    best_practices = []
config_stub.AIProfile = AIProfile
config_stub.AIDirectives = AIDirectives
sys.modules["autogpt.config"] = config_stub

# Stub forge Task
forge_module = types.ModuleType("forge")
forge_sdk_module = types.ModuleType("forge.sdk")
forge_sdk_model_module = types.ModuleType("forge.sdk.model")
class Task:
    def __init__(self, input=""):
        self.input = input
forge_sdk_model_module.Task = Task
forge_sdk_module.model = forge_sdk_model_module
forge_module.sdk = forge_sdk_module
sys.modules["forge"] = forge_module
sys.modules["forge.sdk"] = forge_sdk_module
sys.modules["forge.sdk.model"] = forge_sdk_model_module

# Stub core prompting and resource modules
prompting_stub = types.ModuleType("autogpt.core.prompting")
class PromptStrategy:
    pass
prompting_stub.PromptStrategy = PromptStrategy
prompt_schema_stub = types.ModuleType("autogpt.core.prompting.schema")
class LanguageModelClassification(str, enum.Enum):
    FAST_MODEL = "fast_model"
    SMART_MODEL = "smart_model"
class ChatPrompt:
    def __init__(self, messages=None, functions=None, tokens_used=0):
        self.messages = messages or []
        self.functions = functions or []
        self.tokens_used = tokens_used
prompt_schema_stub.LanguageModelClassification = LanguageModelClassification
prompt_schema_stub.ChatPrompt = ChatPrompt
sys.modules["autogpt.core.prompting.schema"] = prompt_schema_stub

prompting_stub.ChatPrompt = ChatPrompt
prompting_stub.LanguageModelClassification = LanguageModelClassification
sys.modules["autogpt.core.prompting"] = prompting_stub

resource_schema_stub = types.ModuleType(
    "autogpt.core.resource.model_providers.schema"
)
class AssistantChatMessage:
    pass
class ChatMessage:
    def __init__(self, content=""):
        self.content = content
    @staticmethod
    def system(content):
        return ChatMessage(content)
    @staticmethod
    def user(content):
        return ChatMessage(content)
class CompletionModelFunction:
    pass
resource_schema_stub.AssistantChatMessage = AssistantChatMessage
resource_schema_stub.ChatMessage = ChatMessage
resource_schema_stub.CompletionModelFunction = CompletionModelFunction
sys.modules["autogpt.core.resource.model_providers.schema"] = resource_schema_stub

json_schema_stub = types.ModuleType("autogpt.core.utils.json_schema")
class JSONSchema(dict):
    class Type:
        OBJECT = "object"
        STRING = "string"
        ARRAY = "array"
    @classmethod
    def from_dict(cls, d):
        return cls(d)
    def to_dict(self):
        return dict(self)
    def copy(self, deep=False):
        return JSONSchema(dict(self))
    def to_typescript_object_interface(self, name: str) -> str:
        return "{}"
json_schema_stub.JSONSchema = JSONSchema
sys.modules["autogpt.core.utils.json_schema"] = json_schema_stub

json_utils_stub = types.ModuleType("autogpt.core.utils.json_utils")
json_utils_stub.extract_dict_from_json = lambda x: {}
sys.modules["autogpt.core.utils.json_utils"] = json_utils_stub

prompts_utils_stub = types.ModuleType("autogpt.prompts.utils")
prompts_utils_stub.format_numbered_list = lambda x: ""
prompts_utils_stub.indent = lambda text, indent_level=0: text
sys.modules["autogpt.prompts.utils"] = prompts_utils_stub

from third_party.autogpt.autogpt.agents.prompt_strategies.one_shot import (
    OneShotAgentPromptConfiguration,
    OneShotAgentPromptStrategy,
)

def fake_count_message_tokens(messages):
    if isinstance(messages, list):
        return sum(len(m.content) for m in messages)
    return len(messages.content)


def test_prompt_tokens_are_tracked():
    config = OneShotAgentPromptConfiguration(
        model_classification=LanguageModelClassification.FAST_MODEL
    )
    strategy = OneShotAgentPromptStrategy(
        configuration=config, logger=logging.getLogger(__name__)
    )
    task = Task(input="do something")
    prompt = strategy.build_prompt(
        task=task,
        ai_profile=AIProfile(),
        ai_directives=AIDirectives(),
        commands=[],
        event_history=[],
        include_os_info=False,
        max_prompt_tokens=1000,
        count_tokens=len,
        count_message_tokens=fake_count_message_tokens,
    )
    expected = fake_count_message_tokens(prompt.messages)
    assert prompt.tokens_used == expected
