import importlib.util
import logging
import os
import sys
from pathlib import Path

import asyncio
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_PATH = os.path.join(ROOT, "backend", "autogpt")
sys.path.append(BACKEND_PATH)

# Ensure pydantic v1 API compatibility if running with v2
import pydantic
import pydantic.fields as pydantic_fields

if not hasattr(pydantic_fields, "ModelField"):
    class ModelField:  # type: ignore
        pass

    class UndefinedType:  # type: ignore
        pass

    Undefined = UndefinedType()
    pydantic_fields.ModelField = ModelField
    pydantic_fields.Undefined = Undefined
    pydantic_fields.UndefinedType = UndefinedType

# Stub heavy dependencies required for importing the ability module
import types

# Create lightweight ability package and dependencies
ability_pkg = types.ModuleType("autogpt.core.ability")
ability_pkg.__path__ = []  # type: ignore
sys.modules["autogpt.core.ability"] = ability_pkg

ability_base = types.ModuleType("autogpt.core.ability.base")
class Ability:  # type: ignore
    @classmethod
    def name(cls):
        return cls.__name__
class AbilityConfiguration:  # type: ignore
    def __init__(self, **kwargs):
        pass
ability_base.Ability = Ability
ability_base.AbilityConfiguration = AbilityConfiguration
sys.modules["autogpt.core.ability.base"] = ability_base

plugin_simple = types.ModuleType("autogpt.core.plugin.simple")
class PluginLocation:  # type: ignore
    def __init__(self, **kwargs):
        pass
class PluginStorageFormat:  # type: ignore
    INSTALLED_PACKAGE = "installed_package"
plugin_simple.PluginLocation = PluginLocation
plugin_simple.PluginStorageFormat = PluginStorageFormat
sys.modules["autogpt.core.plugin.simple"] = plugin_simple

json_schema_mod = types.ModuleType("autogpt.core.utils.json_schema")
class JSONSchema:  # type: ignore
    class Type:
        STRING = "string"
    def __init__(self, **kwargs):
        pass
json_schema_mod.JSONSchema = JSONSchema
sys.modules["autogpt.core.utils.json_schema"] = json_schema_mod

workspace_mod = types.ModuleType("autogpt.core.workspace")
class Workspace:  # type: ignore
    def get_path(self, relative_path):
        return Path(relative_path)
workspace_mod.Workspace = Workspace
sys.modules["autogpt.core.workspace"] = workspace_mod

schema_spec = importlib.util.spec_from_file_location(
    "autogpt.core.ability.schema",
    os.path.join(BACKEND_PATH, "autogpt", "core", "ability", "schema.py"),
)
schema_mod = importlib.util.module_from_spec(schema_spec)
schema_spec.loader.exec_module(schema_mod)  # type: ignore
sys.modules["autogpt.core.ability.schema"] = schema_mod
AbilityResult = schema_mod.AbilityResult
ContentType = schema_mod.ContentType
Knowledge = schema_mod.Knowledge

spec = importlib.util.spec_from_file_location(
    "file_operations",
    os.path.join(BACKEND_PATH, "autogpt", "core", "ability", "builtins", "file_operations.py"),
)
file_ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(file_ops)  # type: ignore
ReadFile = file_ops.ReadFile


class DummyWorkspace(Workspace):
    def __init__(self, root: Path):
        self._root = root

    @property
    def root(self) -> Path:  # type: ignore[override]
        return self._root

    @property
    def restrict_to_workspace(self) -> bool:  # type: ignore[override]
        return True

    @staticmethod
    def setup_workspace(configuration, logger):
        return Path(configuration.root)

    def get_path(self, relative_path: str | Path) -> Path:  # type: ignore[override]
        return self._root / relative_path


def test_readfile_extracts_structured_metadata(tmp_path):
    sample = (
        "# Heading 1\n\n"
        "Some text here.\n\n"
        "| a | b |\n|---|---|\n| 1 | 2 |\n\n"
        "## Heading 2\n"
    )
    file_path = tmp_path / "sample.md"
    file_path.write_text(sample, encoding="utf-8")

    workspace = DummyWorkspace(tmp_path)
    ability = ReadFile(logger=logging.getLogger("test"), workspace=workspace)
    result = asyncio.run(ability("sample.md"))

    assert result.success
    meta = result.new_knowledge.content_metadata
    assert meta["filename"] == "sample.md"
    assert meta.get("headings") == ["Heading 1", "Heading 2"]
    assert len(meta.get("tables", [])) == 1
    assert "a" in meta["tables"][0]


def test_readfile_handles_images(tmp_path):
    from PIL import Image

    img_path = tmp_path / "image.png"
    Image.new("RGB", (10, 20), color="red").save(img_path)

    workspace = DummyWorkspace(tmp_path)
    ability = ReadFile(logger=logging.getLogger("test"), workspace=workspace)
    result = asyncio.run(ability("image.png"))

    assert result.success
    meta = result.new_knowledge.content_metadata
    assert meta["mime_type"].startswith("image")
    assert meta["width"] == 10
    assert meta["height"] == 20

