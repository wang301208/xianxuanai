import asyncio
import importlib.util
import logging
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "backend", "autogpt"))

import types

# Minimal stubs for autogpt dependencies
inflection = types.ModuleType("inflection")
inflection.underscore = lambda value: value.lower().replace(" ", "_")
sys.modules.setdefault("inflection", inflection)

ability_base = types.ModuleType("autogpt.core.ability.base")


class AbilityConfiguration:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class Ability:
    default_configuration = None

    @classmethod
    def name(cls) -> str:
        return cls.__name__


ability_base.AbilityConfiguration = AbilityConfiguration
ability_base.Ability = Ability
class AbilityRegistry:
    ...

ability_base.AbilityRegistry = AbilityRegistry
sys.modules.setdefault("autogpt.core.ability.base", ability_base)

schema_module = types.ModuleType("autogpt.core.ability.schema")


class ContentType(str):
    TEXT = "text"


class Knowledge:
    def __init__(self, content, content_type, content_metadata):
        self.content = content
        self.content_type = content_type
        self.content_metadata = content_metadata


class AbilityResult:
    def __init__(self, ability_name, ability_args, success, message, new_knowledge=None):
        self.ability_name = ability_name
        self.ability_args = ability_args
        self.success = success
        self.message = message
        self.new_knowledge = new_knowledge


schema_module.ContentType = ContentType
schema_module.Knowledge = Knowledge
schema_module.AbilityResult = AbilityResult
sys.modules.setdefault("autogpt.core.ability.schema", schema_module)

plugin_module = types.ModuleType("autogpt.core.plugin.simple")


class PluginStorageFormat(str):
    INSTALLED_PACKAGE = "installed_package"


class PluginLocation:
    def __init__(self, storage_format, storage_route):
        self.storage_format = storage_format
        self.storage_route = storage_route


plugin_module.PluginStorageFormat = PluginStorageFormat
plugin_module.PluginLocation = PluginLocation
sys.modules.setdefault("autogpt.core.plugin.simple", plugin_module)

json_schema_module = types.ModuleType("autogpt.core.utils.json_schema")


class JSONSchema:
    class Type(str):
        STRING = "string"
        INTEGER = "integer"

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


json_schema_module.JSONSchema = JSONSchema
sys.modules.setdefault("autogpt.core.utils.json_schema", json_schema_module)

MODULE_PATH = os.path.join(
    ROOT,
    "backend",
    "autogpt",
    "autogpt",
    "core",
    "ability",
    "builtins",
    "web_search.py",
)
spec = importlib.util.spec_from_file_location("web_search_test_module", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)
WebSearch = module.WebSearch


def test_web_search_with_stub_client():
    def fake_search(query: str, max_results: int):
        return [
            {"title": "First Result", "url": "https://example.com/1", "snippet": "Snippet 1"},
            {"title": "Second Result", "url": "https://example.com/2", "snippet": "Snippet 2"},
        ][:max_results]

    ability = WebSearch(
        logger=logging.getLogger("test"),
        configuration=WebSearch.default_configuration,
        search_client=fake_search,
    )

    result = asyncio.run(ability("autonomous agents", max_results=2))

    assert result.success
    assert "First Result" in result.message
    assert result.new_knowledge is not None
    assert result.new_knowledge.content_metadata["source"] == "web_search"
