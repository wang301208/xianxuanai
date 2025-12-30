"""Tests for syntax dependency extraction and semantic grounding."""

from __future__ import annotations

from BrainSimulationSystem.models.language_processing import SemanticNetwork, SyntaxProcessor
from BrainSimulationSystem.models.language_comprehension import LanguageComprehension
from BrainSimulationSystem.models.semantic_analyser import SemanticAnalyser


def test_syntax_processor_emits_dependency_metadata():
    syntax = SyntaxProcessor({})
    parsed = syntax.parse_sentence(["You", "run", "system", "quickly"])

    meta = parsed.get("meta") or {}
    assert meta.get("pos_tags")

    dependency = meta.get("dependency") or {}
    arcs = dependency.get("arcs") or []
    assert arcs, "dependency arcs should be present for semantic grounding"


def test_semantic_analyser_grounds_subject_object_relations():
    network = SemanticNetwork({})
    analyser = SemanticAnalyser(network, {})
    syntax = SyntaxProcessor({})

    tokens = ["You", "run", "system", "quickly"]
    normalized = [t.lower() for t in tokens]
    parsed = syntax.parse_sentence(tokens)

    result = analyser.analyse("You run system quickly", tokens, normalized, parsed)
    relations = result.relations

    assert {"head": "run", "dependent": "you", "relation": "subject"} in relations
    assert {"head": "run", "dependent": "system", "relation": "object"} in relations
    assert network.relation_strength("run", "system") > 0.0


def test_language_comprehension_ingests_visual_concepts_into_semantic_network():
    network = SemanticNetwork({})
    comprehension = LanguageComprehension(
        None,
        None,
        None,
        network,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        {},
    )

    multimodal = comprehension._ingest_multimodal_context(  # noqa: SLF001 - intentional test hook
        inputs={"visual_concepts": ["Dog", "Tree"]},
        context={},
        semantic_info=None,
    )

    assert multimodal.get("visual_concepts") == ["dog", "tree"]
    assert "visual_scene" in network.nodes
    assert "dog" in network.nodes
    assert network.relation_strength("visual_scene", "dog") > 0.0

