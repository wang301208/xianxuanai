"""Ontology definitions for the AutoGPT knowledge graph."""

from __future__ import annotations

from enum import Enum


class EntityType(str, Enum):
    """Types of entities tracked in the knowledge graph."""

    SKILL = "skill"
    TASK = "task"
    AGENT = "agent"
    CONCEPT = "concept"


class RelationType(str, Enum):
    """Types of relations between entities in the knowledge graph."""

    REQUIRES = "requires"
    PERFORMS = "performs"
    RELATED_TO = "related_to"
