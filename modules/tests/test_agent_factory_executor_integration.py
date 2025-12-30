import os
import sys
import subprocess
import logging
from pathlib import Path
import yaml
import pytest
import types
import hashlib

# Ensure root path and autogpt package are importable
sys.path.insert(0, os.path.abspath(os.getcwd()))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "autogpts", "autogpt")))
sys.modules.setdefault("auto_gpt_plugin_template", types.SimpleNamespace(AutoGPTPluginTemplate=object))

# Stub Google Cloud logging to avoid optional dependency
class _StubHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        pass


class _StubFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return True


cloud_logging_handlers = types.SimpleNamespace(
    CloudLoggingFilter=_StubFilter, StructuredLogHandler=_StubHandler
)
logging_v2 = types.SimpleNamespace(handlers=cloud_logging_handlers)
sys.modules.setdefault("google", types.SimpleNamespace(cloud=types.SimpleNamespace(logging_v2=logging_v2)))
sys.modules.setdefault("google.cloud", types.SimpleNamespace(logging_v2=logging_v2))
sys.modules.setdefault("google.cloud.logging_v2", logging_v2)
sys.modules.setdefault("google.cloud.logging_v2.handlers", cloud_logging_handlers)
sys.modules.setdefault("playsound", types.SimpleNamespace(playsound=lambda *a, **k: None))
sys.modules.setdefault(
    "autogpt.speech", types.SimpleNamespace(TextToSpeechProvider=object, TTSConfig=object)
)
sys.modules.setdefault("spacy", types.SimpleNamespace(load=lambda *a, **k: None))
sys.modules.setdefault(
    "autogpt.commands.file_operations_utils",
    types.SimpleNamespace(decode_textual_file=lambda *a, **k: ""),
)
plugins_module = types.ModuleType("autogpt.plugins")
plugins_module.scan_plugins = lambda config: []
sys.modules.setdefault("autogpt.plugins", plugins_module)
plugins_config_module = types.ModuleType("autogpt.plugins.plugins_config")
class _PluginsConfig:
    def __init__(self, plugins=None):
        self.plugins = plugins or {}

plugins_config_module.PluginsConfig = _PluginsConfig
sys.modules.setdefault("autogpt.plugins.plugins_config", plugins_config_module)
commands_module = types.ModuleType("autogpt.commands")
commands_module.COMMAND_CATEGORIES = []
sys.modules.setdefault("autogpt.commands", commands_module)

from agent_factory import create_agent_from_blueprint
from capability.skill_library import SkillLibrary
from execution import Executor
from third_party.autogpt.autogpt.core.errors import SkillExecutionError
from third_party.autogpt.autogpt.config import Config
from third_party.autogpt.autogpt.file_storage.local import LocalFileStorage, FileStorageConfiguration


class DummyProvider:
    def count_tokens(self, text: str, model_name: str) -> int:
        return len(text)

    def get_token_limit(self, model_name: str) -> int:
        return 1000

    def get_tokenizer(self, model_name: str):
        class _Tok:
            def encode(self, text):
                return list(text)

            def decode(self, tokens):
                return "".join(tokens)

        return _Tok()

    def count_message_tokens(self, messages, model_name: str) -> int:
        return 0

    async def get_available_models(self):
        return []

    async def create_chat_completion(self, *args, **kwargs):
        raise NotImplementedError


def init_repo(path: Path) -> None:
    subprocess.run(["git", "init"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)


def test_agent_factory_and_executor(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    init_repo(repo)
    lib = SkillLibrary(repo)

    hello_code = (
        "from third_party.autogpt.autogpt.command_decorator import command\n"
        "@command('hello', 'say hi', {})\n"
        "def hello():\n    return 'hi'\n"
    )
    lib.add_skill(
        "hello",
        hello_code,
        {
            "lang": "python",
            "signature": hashlib.sha256(hello_code.encode()).hexdigest(),
        },
    )
    fail_code = "def fail():\n    raise RuntimeError('boom')\n"
    lib.add_skill(
        "fail",
        fail_code,
        {"lang": "python", "signature": hashlib.sha256(fail_code.encode()).hexdigest()},
    )

    blueprint = {
        "role_name": "Tester",
        "core_prompt": "Do things",
        "authorized_tools": ["hello"],
        "subscribed_topics": [],
    }
    blueprint_path = repo / "blueprint.yaml"
    blueprint_path.write_text(yaml.safe_dump(blueprint), encoding="utf-8")

    cfg = Config()
    cfg.fast_llm = "gpt-3.5-turbo"
    cfg.smart_llm = "gpt-3.5-turbo"
    cfg.prompt_settings_file = Path(__file__).resolve().parents[1] / "prompt_settings.yaml"
    storage = LocalFileStorage(
        FileStorageConfiguration(root=repo / "storage", restrict_to_root=False)
    )
    storage.initialize()
    provider = DummyProvider()

    cwd = os.getcwd()
    os.chdir(repo)
    try:
        agent = create_agent_from_blueprint(blueprint_path, cfg, provider, storage)
    finally:
        os.chdir(cwd)

    assert "hello" in agent.command_registry.commands

    executor = Executor(lib)
    result = executor.execute("hello")
    assert result["hello"] == "hi"

    with caplog.at_level(logging.ERROR):
        with pytest.raises(SkillExecutionError) as exc_info:
            executor.execute("fail")
    assert "boom" in str(exc_info.value)
    assert any("fail" in r.message and "boom" in r.message for r in caplog.records)
    lib.close()
