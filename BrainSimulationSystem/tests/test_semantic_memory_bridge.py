"""Tests for episodic->semantic bridging and semantic forgetting."""

from __future__ import annotations

import asyncio

from ..models.language_processing import SemanticNetwork
from ..models.memory import ComprehensiveMemorySystem, MemoryType


def _run(coro):
    return asyncio.run(coro)


def _make_memory_system(*, forgetting: bool = False, similarity_threshold: float = -1.0, wm_indexed: bool = False):
    config = {
        "semantic_bridge": {"enabled": True, "max_associations": 8},
        "neocortical": {
            "stable_concepts": True,
            "embedding_size": 32,
            "similarity_threshold": float(similarity_threshold),
            "max_concepts": 1024,
        },
    }

    if forgetting:
        config["neocortical"]["forgetting"] = {
            "enabled": True,
            "strength_decay_rate": 1.0,
            "edge_decay_rate": 1.0,
            "min_concept_strength": 0.05,
            "min_edge_weight": 0.05,
            "prune_batch": 128,
        }

    if wm_indexed:
        config["working_memory"] = {
            "strategy": "indexed",
            "capacity": 4,
            "decay_rate": 1.0,
            "min_weight": 0.9,
        }

    return ComprehensiveMemorySystem(config)


def test_episodic_encoding_bridges_to_symbolic_semantic_network():
    memory_system = _make_memory_system(similarity_threshold=-1.0)
    semantic_network = SemanticNetwork()
    memory_system.attach_semantic_network(semantic_network)

    episode = {
        "concept": "Apple",
        "location": "kitchen",
        "tags": ["fruit"],
        "entities": ["banana"],
    }
    context = {"attention_directives": {"semantic_focus": ["food"]}}

    _run(memory_system.encode_memory(episode, MemoryType.EPISODIC, context))

    assert "concept::apple" in memory_system.neocortical_system.concept_embeddings
    assert "apple" in semantic_network.nodes
    assert "fruit" in semantic_network.nodes
    assert "banana" in semantic_network.nodes
    assert "food" in semantic_network.nodes

    relation = semantic_network.relations.get("apple", {}).get("fruit")
    assert relation is not None
    assert "episodic_association" in relation.get("types", set())


def test_update_memory_system_prunes_semantic_concepts_when_forgetting_enabled():
    memory_system = _make_memory_system(forgetting=True, similarity_threshold=-1.0)
    memory_system.attach_semantic_network(SemanticNetwork())

    episode = {"concept": "Apple", "tags": ["fruit"]}
    _run(memory_system.encode_memory(episode, MemoryType.EPISODIC, {}))

    assert memory_system.neocortical_system.concept_embeddings

    _run(memory_system.update_memory_system(dt=10.0))

    assert not memory_system.neocortical_system.concept_embeddings


def test_update_memory_system_updates_indexed_working_memory_decay():
    memory_system = _make_memory_system(wm_indexed=True)

    _run(
        memory_system.encode_memory(
            {"id": "item1", "text": "hello"},
            MemoryType.WORKING,
            {"attention_weight": 1.0},
        )
    )
    assert memory_system.working_memory.items

    _run(memory_system.update_memory_system(dt=1.0))

    assert not memory_system.working_memory.items
