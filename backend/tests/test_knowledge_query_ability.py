import asyncio
import json
import logging
import os
import sys
import types
from enum import Enum

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "backend", "autogpt"))

if "inflection" not in sys.modules:
    inflection = types.ModuleType("inflection")

    def _underscore(value: str) -> str:
        return value.lower().replace(" ", "_")

    def _camelize(value: str) -> str:
        parts = value.replace("-", "_").split("_")
        return "".join(word.capitalize() for word in parts if word)

    inflection.underscore = _underscore
    inflection.camelize = _camelize
    sys.modules["inflection"] = inflection

if "autogpt.core.ability.base" not in sys.modules:
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

    ability_base.Ability = Ability
    ability_base.AbilityConfiguration = AbilityConfiguration
    class AbilityRegistry:
        ...

    ability_base.AbilityRegistry = AbilityRegistry
    sys.modules["autogpt.core.ability.base"] = ability_base

if "autogpt.core.ability.schema" not in sys.modules:
    schema_module = types.ModuleType("autogpt.core.ability.schema")

    class ContentType(str, Enum):
        TEXT = "text"

    class Knowledge:
        def __init__(self, content: str, content_type: ContentType, content_metadata: dict):
            self.content = content
            self.content_type = content_type
            self.content_metadata = content_metadata

    class AbilityResult:
        def __init__(
            self,
            *,
            ability_name: str,
            ability_args: dict,
            success: bool,
            message: str,
            new_knowledge: Knowledge | None = None,
        ):
            self.ability_name = ability_name
            self.ability_args = ability_args
            self.success = success
            self.message = message
            self.new_knowledge = new_knowledge

    schema_module.ContentType = ContentType
    schema_module.Knowledge = Knowledge
    schema_module.AbilityResult = AbilityResult
    sys.modules["autogpt.core.ability.schema"] = schema_module

if "autogpt.core.plugin.simple" not in sys.modules:
    plugin_module = types.ModuleType("autogpt.core.plugin.simple")

    class PluginStorageFormat(str, Enum):
        INSTALLED_PACKAGE = "installed_package"

    class PluginLocation:
        def __init__(self, storage_format: PluginStorageFormat, storage_route: str):
            self.storage_format = storage_format
            self.storage_route = storage_route

    plugin_module.PluginLocation = PluginLocation
    plugin_module.PluginStorageFormat = PluginStorageFormat
    sys.modules["autogpt.core.plugin.simple"] = plugin_module

if "autogpt.core.utils.json_schema" not in sys.modules:
    json_schema_module = types.ModuleType("autogpt.core.utils.json_schema")

    class JSONSchema:
        class Type(str, Enum):
            STRING = "string"
            INTEGER = "integer"
            BOOLEAN = "boolean"

        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    json_schema_module.JSONSchema = JSONSchema
    sys.modules["autogpt.core.utils.json_schema"] = json_schema_module

import importlib.util

from backend.concept_alignment import ConceptAligner
from backend.knowledge.registry import set_default_aligner, set_graph_store
from backend.autogpt.autogpt.core.knowledge_graph.graph_store import GraphStore
from backend.autogpt.autogpt.core.knowledge_graph.ontology import EntityType, RelationType
from modules.common.concepts import ConceptNode

MODULE_PATH = os.path.join(
    ROOT,
    "backend",
    "autogpt",
    "autogpt",
    "core",
    "ability",
    "builtins",
    "knowledge_query.py",
)
spec = importlib.util.spec_from_file_location("knowledge_query_test_module", MODULE_PATH)
knowledge_module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(knowledge_module)
KnowledgeQuery = knowledge_module.KnowledgeQuery


class DummyLibrarian:
    def search(self, *args, **kwargs):
        return []


def test_knowledge_query_returns_relevant_concepts(tmp_path):
    aligner = ConceptAligner(DummyLibrarian(), {})
    store = GraphStore()
    set_default_aligner(aligner)
    set_graph_store(store)

    embedding = KnowledgeQuery._hash_embedding("renewable energy storage")
    concept = ConceptNode(
        id="concept.energy",
        label="Renewable Energy Storage",
        modalities={"text": embedding},
        metadata={"description": "Energy stored from renewable sources."},
    )
    aligner.entities[concept.id] = concept
    store.add_node(
        concept.id,
        EntityType.CONCEPT,
        description=concept.metadata["description"],
    )

    ability = KnowledgeQuery(logging.getLogger("test"), KnowledgeQuery.default_configuration)

    result = asyncio.run(ability(query="renewable energy storage", top_k=1))

    assert result.success
    assert "Renewable Energy Storage" in result.message
    assert result.new_knowledge is not None
    payload = json.loads(result.new_knowledge.content)
    assert payload["query"] == "renewable energy storage"
    assert payload["results"][0]["id"] == concept.id


def test_knowledge_query_includes_relations():
    aligner = ConceptAligner(DummyLibrarian(), {})
    store = GraphStore()
    set_default_aligner(aligner)
    set_graph_store(store)

    source = ConceptNode(
        id="concept.source",
        label="Solar Farm",
        modalities={"text": KnowledgeQuery._hash_embedding("solar farm")},
        metadata={},
    )
    target = ConceptNode(
        id="concept.target",
        label="Microgrid",
        modalities={"text": KnowledgeQuery._hash_embedding("microgrid")},
        metadata={},
    )
    aligner.entities[source.id] = source
    aligner.entities[target.id] = target

    store.add_node(source.id, EntityType.CONCEPT, label=source.label)
    store.add_node(target.id, EntityType.CONCEPT, label=target.label)
    store.add_edge(
        source.id,
        target.id,
        RelationType.RELATED_TO,
        note="Supplies power",
        weight=0.9,
    )

    ability = KnowledgeQuery(logging.getLogger("test"), KnowledgeQuery.default_configuration)

    result = asyncio.run(
        ability(
            query="solar farm supplying microgrid",
            top_k=2,
            include_relations=True,
        )
    )

    assert result.success
    payload = json.loads(result.new_knowledge.content)
    relations = payload["results"][0].get("relations", [])
    assert relations, "relations should be included when requested"
    relation = relations[0]
    assert relation["source"] == source.id
    assert relation["target"] == target.id
