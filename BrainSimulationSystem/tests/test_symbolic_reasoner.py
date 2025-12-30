"""Tests for knowledge graph and symbolic reasoner."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from BrainSimulationSystem.models.knowledge_graph import KnowledgeGraph, KnowledgeConstraint
from BrainSimulationSystem.models.symbolic_reasoner import SymbolicReasoner, Rule


def test_knowledge_graph_add_query_constraints():
    graph = KnowledgeGraph()
    graph.add('agent', 'type', 'robot')
    graph.add('agent', 'supports', 'exploration')

    assert graph.exists('agent', 'type', 'robot')
    assert ('agent', 'supports', 'exploration') in graph.query('agent')

    constraints = [
        KnowledgeConstraint(
            description='requires_exploration_support',
            required=[('agent', 'supports', 'exploration')],
        ),
        KnowledgeConstraint(
            description='forbid_shutdown',
            forbidden=[('agent', 'state', 'shutdown')],
        ),
    ]
    results = graph.check_constraints(constraints)
    assert all(results.values())


def test_symbolic_reasoner_forward_chaining():
    graph = KnowledgeGraph()
    graph.add('goal', 'type', 'exploration')
    reasoner = SymbolicReasoner(graph)
    reasoner.add_rule(
        Rule(
            name='exploration_support',
            antecedents=[('goal', 'type', 'exploration')],
            consequent=('agent', 'supports', 'exploration'),
        )
    )

    inferred = reasoner.infer()
    assert ('agent', 'supports', 'exploration') in inferred
    explanation = reasoner.explain(('agent', 'supports', 'exploration'))
    assert explanation == ['exploration_support']


def test_knowledge_graph_find_paths_returns_multi_hop_sequences():
    graph = KnowledgeGraph()
    graph.add('deploy_model', 'requires', 'train_model')
    graph.add('train_model', 'requires', 'prepare_dataset')
    graph.add('prepare_dataset', 'requires', 'collect_data')

    paths = graph.find_paths('deploy_model', predicates=['requires'], max_depth=3, max_paths=2)

    assert paths, "Expected at least one path"
    first = paths[0]
    assert first["nodes"][0] == 'deploy_model'
    assert first["triples"], "Path should include hop triples"


def test_knowledge_graph_upsert_and_metadata_roundtrip():
    graph = KnowledgeGraph()
    inserted = graph.upsert_triples(
        [
            ('agent', 'type', 'robot'),
            ('agent', 'status', 'active'),
        ],
        default_metadata={'provenance': 'perception'},
    )
    assert inserted == 2
    assert graph.get_metadata('agent', 'type', 'robot')['provenance'] == 'perception'

    inserted = graph.upsert_triples(
        [('agent', 'type', 'robot')],
        default_metadata={'provenance': 'override'},
    )
    assert inserted == 0
    assert graph.get_metadata('agent', 'type', 'robot')['provenance'] == 'override'

    snapshot = graph.to_dict()
    clone = KnowledgeGraph.from_dict(snapshot)
    assert clone.exists('agent', 'status', 'active')
    assert clone.get_metadata('agent', 'type', 'robot')['provenance'] == 'override'
