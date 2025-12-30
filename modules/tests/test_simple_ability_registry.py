import importlib
import logging
import subprocess

import pytest
import sys
from pathlib import Path
import types

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "autogpts" / "autogpt"))

sys.modules.setdefault(
    "autogpt.core.resource.model_providers.openai",
    types.SimpleNamespace(
        OPEN_AI_CHAT_MODELS={},
        OPEN_AI_EMBEDDING_MODELS={},
        OPEN_AI_MODELS={},
        OpenAIModelName=str,
        OpenAIProvider=object,
        OpenAISettings=object,
    ),
)
sys.modules.setdefault(
    "autogpt.core.ability.builtins", types.SimpleNamespace(BUILTIN_ABILITIES={})
)
sys.modules.setdefault(
    "autogpt.core.planning.simple", types.SimpleNamespace(LanguageModelConfiguration=object)
)

from third_party.autogpt.autogpt.core.ability.base import Ability, AbilityConfiguration
from third_party.autogpt.autogpt.core.ability.schema import AbilityResult
from third_party.autogpt.autogpt.core.ability.simple import (
    AbilityRegistryConfiguration,
    AbilityRegistrySettings,
    SimpleAbilityRegistry,
)
from third_party.autogpt.autogpt.core.plugin.base import PluginLocation, PluginStorageFormat
from third_party.autogpt.autogpt.core.plugin.simple import SimplePluginService


class DummyAbility(Ability):
    """Ability that relies on a missing package."""

    description = "dummy ability"
    parameters = {}

    def __init__(self, logger, configuration):
        self.logger = logger
        self.configuration = configuration

    async def __call__(self) -> AbilityResult:  # pragma: no cover - raising error
        importlib.import_module("missing_pkg")
        return AbilityResult(
            ability_name=self.name(),
            ability_args={},
            success=True,
            message="ok",
        )


class ErrorAbility(Ability):
    description = "always fails"
    parameters = {}

    def __init__(self, logger, configuration):
        self.logger = logger
        self.configuration = configuration

    async def __call__(self) -> AbilityResult:  # pragma: no cover - raising error
        raise ValueError("boom")


@pytest.mark.asyncio
async def test_missing_package_install_attempt(monkeypatch, caplog):
    ability_config = AbilityConfiguration(
        location=PluginLocation(
            storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
            storage_route="dummy",
        ),
        packages_required=["missing_pkg"],
    )
    settings = AbilityRegistrySettings(
        name="test_registry",
        description="test",
        configuration=AbilityRegistryConfiguration(
            abilities={"dummy_ability": ability_config}
        ),
    )

    monkeypatch.setattr(SimplePluginService, "get_plugin", lambda loc: DummyAbility)

    real_import_module = importlib.import_module

    def fake_import_module(name, *a, **k):
        if name == "missing_pkg":
            raise ImportError("No module named 'missing_pkg'")
        return real_import_module(name, *a, **k)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    run_calls = []

    def fake_run(cmd, *a, **k):
        run_calls.append(cmd)
        raise subprocess.CalledProcessError(returncode=1, cmd=cmd)

    monkeypatch.setattr(subprocess, "run", fake_run)
    caplog.set_level(logging.WARNING)

    registry = SimpleAbilityRegistry(
        settings,
        logging.getLogger("test"),
        memory=object(),
        workspace=object(),
        model_providers={},
    )

    assert run_calls, "pip install should have been attempted"

    result = await registry.perform("dummy_ability")
    assert not result.success
    assert "No module named 'missing_pkg'" in result.message


@pytest.mark.asyncio
async def test_perform_surfaces_errors(monkeypatch):
    ability_config = AbilityConfiguration(
        location=PluginLocation(
            storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
            storage_route="error",
        )
    )
    settings = AbilityRegistrySettings(
        name="test_registry",
        description="test",
        configuration=AbilityRegistryConfiguration(
            abilities={"error_ability": ability_config}
        ),
    )

    monkeypatch.setattr(SimplePluginService, "get_plugin", lambda loc: ErrorAbility)

    registry = SimpleAbilityRegistry(
        settings,
        logging.getLogger("test"),
        memory=object(),
        workspace=object(),
        model_providers={},
    )

    result = await registry.perform("error_ability")
    assert not result.success
    assert "boom" in result.message
