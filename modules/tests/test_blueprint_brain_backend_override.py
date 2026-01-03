from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.modules.pop("reasoning", None)
sys.modules.pop("reasoning.decision_engine", None)

from backend.agent_factory import create_agent_from_blueprint
from third_party.autogpt.autogpt.config import Config
from third_party.autogpt.autogpt.core.brain.config import BrainBackend
from third_party.autogpt.autogpt.file_storage.base import FileStorageConfiguration
from third_party.autogpt.autogpt.file_storage.local import LocalFileStorage


class _DummyProvider:
    def count_tokens(self, text: str, model_name: str) -> int:  # pragma: no cover - minimal stub
        return len(text)

    def get_token_limit(self, model_name: str) -> int:  # pragma: no cover - minimal stub
        return 4096

    def get_tokenizer(self, model_name: str):  # pragma: no cover - minimal stub
        class _Tok:
            def encode(self, text):  # pragma: no cover - minimal stub
                return list(text)

            def decode(self, tokens):  # pragma: no cover - minimal stub
                return "".join(tokens)

        return _Tok()

    def count_message_tokens(self, messages, model_name: str) -> int:  # pragma: no cover
        return 0

    async def get_available_models(self):  # pragma: no cover - not needed
        return []

    async def create_chat_completion(self, *args, **kwargs):  # pragma: no cover - not needed
        raise NotImplementedError


def _write_blueprint(path: Path, *, backend: str) -> Path:
    blueprint = {
        "role_name": "Tester",
        "core_prompt": "Do things",
        "authorized_tools": [],
        "subscribed_topics": [],
        "brain_backend": backend,
    }
    blueprint_path = path / "blueprint.yaml"
    blueprint_path.write_text(yaml.safe_dump(blueprint), encoding="utf-8")
    return blueprint_path


def _build_config() -> Config:
    cfg = Config()
    repo_root = Path(__file__).resolve().parents[2]
    cfg.prompt_settings_file = repo_root / "config" / "prompt_settings.yaml"
    return cfg


def test_blueprint_override_llm_backend(tmp_path):
    blueprint_path = _write_blueprint(tmp_path, backend="llm")
    storage = LocalFileStorage(
        FileStorageConfiguration(root=tmp_path / "storage", restrict_to_root=False)
    )
    storage.initialize()

    agent = create_agent_from_blueprint(
        blueprint_path,
        config=_build_config(),
        llm_provider=_DummyProvider(),
        file_storage=storage,
    )

    try:
        assert agent.config.brain_backend == BrainBackend.LLM
        assert agent.whole_brain is None
        assert getattr(agent, "brain", None) is None
    finally:
        agent.long_term_memory.close()


def test_blueprint_override_transformer_backend_auto_enables(monkeypatch, tmp_path):
    blueprint_path = _write_blueprint(tmp_path, backend="transformer")
    storage = LocalFileStorage(
        FileStorageConfiguration(root=tmp_path / "storage", restrict_to_root=False)
    )
    storage.initialize()

    created: dict[str, bool] = {"brain": False}

    class _StubTransformerBrain:
        def __init__(self, *_args, **_kwargs) -> None:  # pragma: no cover - simple marker
            created["brain"] = True

    monkeypatch.setattr(
        "backend.agent_factory.TransformerBrain",
        _StubTransformerBrain,
    )

    agent = create_agent_from_blueprint(
        blueprint_path,
        config=_build_config(),
        llm_provider=_DummyProvider(),
        file_storage=storage,
    )

    try:
        assert created["brain"] is True
        assert agent.config.brain_backend == BrainBackend.TRANSFORMER
        assert agent.whole_brain is None
        assert isinstance(getattr(agent, "brain", None), _StubTransformerBrain)
    finally:
        agent.long_term_memory.close()


def test_blueprint_override_whole_brain_backend(monkeypatch, tmp_path):
    blueprint_path = _write_blueprint(tmp_path, backend="whole_brain")
    storage = LocalFileStorage(
        FileStorageConfiguration(root=tmp_path / "storage", restrict_to_root=False)
    )
    storage.initialize()

    captured: dict[str, object] = {}

    class _StubStructuredBackend:
        def process_cycle(self, _payload):  # pragma: no cover - not exercised in this test
            raise NotImplementedError

    def fake_create_brain_backend(backend: BrainBackend, *args, **kwargs):
        captured["backend"] = backend
        return _StubStructuredBackend()

    monkeypatch.setattr(
        "backend.agent_factory.create_brain_backend",
        fake_create_brain_backend,
    )

    agent = create_agent_from_blueprint(
        blueprint_path,
        config=_build_config(),
        llm_provider=_DummyProvider(),
        file_storage=storage,
    )

    try:
        assert captured["backend"] == BrainBackend.WHOLE_BRAIN
        assert agent.config.brain_backend == BrainBackend.WHOLE_BRAIN
        assert agent.whole_brain is not None
    finally:
        agent.long_term_memory.close()
