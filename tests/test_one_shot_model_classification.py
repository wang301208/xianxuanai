import logging
import sys
from pathlib import Path
import types
import enum

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party/autogpt"))

# Stub out heavy dependencies from third_party.autogpt.autogpt.config to avoid importing modules
config_stub = types.ModuleType("autogpt.config")

class AIProfile:  # minimal placeholder
    pass


class AIDirectives:  # minimal placeholder
    pass


config_stub.AIProfile = AIProfile
config_stub.AIDirectives = AIDirectives
sys.modules["autogpt.config"] = config_stub

# Stub out minimal core prompting and resource modules
prompting_stub = types.ModuleType("autogpt.core.prompting")

class LanguageModelClassification(str, enum.Enum):
    FAST_MODEL = "fast_model"
    SMART_MODEL = "smart_model"


class PromptStrategy:
    pass


class ChatPrompt:
    def __init__(self, messages=None, functions=None):
        self.messages = messages or []
        self.functions = functions or []


prompting_stub.LanguageModelClassification = LanguageModelClassification
prompting_stub.PromptStrategy = PromptStrategy
prompting_stub.ChatPrompt = ChatPrompt
sys.modules["autogpt.core.prompting"] = prompting_stub

resource_schema_stub = types.ModuleType(
    "autogpt.core.resource.model_providers.schema"
)


class AssistantChatMessage:
    pass


class ChatMessage:
    @staticmethod
    def system(content):
        return content

    @staticmethod
    def user(content):
        return content


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
from third_party.autogpt.autogpt.core.prompting import LanguageModelClassification


def test_model_classification_fast():
    config = OneShotAgentPromptConfiguration(
        model_classification=LanguageModelClassification.FAST_MODEL
    )
    strategy = OneShotAgentPromptStrategy(
        configuration=config, logger=logging.getLogger(__name__)
    )
    assert strategy.model_classification == LanguageModelClassification.FAST_MODEL


def test_model_classification_smart():
    config = OneShotAgentPromptConfiguration(
        model_classification=LanguageModelClassification.SMART_MODEL
    )
    strategy = OneShotAgentPromptStrategy(
        configuration=config, logger=logging.getLogger(__name__)
    )
    assert strategy.model_classification == LanguageModelClassification.SMART_MODEL
