"""Utility functions for working with causal relations between concept nodes."""
from __future__ import annotations

from typing import Any, Dict, List

from .concepts import CausalRelation, ConceptNode

# Simple registry to allow lookups by node id
_NODE_REGISTRY: Dict[str, ConceptNode] = {}


def _register(node: ConceptNode) -> None:
    """Register a node in the local registry."""
    _NODE_REGISTRY[node.id] = node


def add_causal_relation(
    node_a: ConceptNode,
    node_b: ConceptNode,
    weight: float = 1.0,
    metadata: Dict[str, Any] | None = None,
) -> CausalRelation:
    """Create and store a causal relation between two nodes.

    Parameters
    ----------
    node_a:
        The cause node.
    node_b:
        The effect node.
    weight:
        Optional strength of the causal influence.
    metadata:
        Optional additional information for the relation.
    """

    relation = CausalRelation(
        cause=node_a.id,
        effect=node_b.id,
        weight=weight,
        metadata=metadata or {},
    )
    node_a.causal_links.append(relation)
    node_b.causal_links.append(relation)
    _register(node_a)
    _register(node_b)
    return relation


def get_effects(cause_id: str) -> List[str]:
    """Return ids of effects directly caused by the given node id."""
    node = _NODE_REGISTRY.get(cause_id)
    if not node:
        return []
    return [rel.effect for rel in node.causal_links if rel.cause == cause_id]
