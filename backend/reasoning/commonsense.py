"""Commonsense knowledge utilities for validating agent outputs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass
class CommonsenseJudgement:
    """Result of a commonsense validation pass."""

    subject: str
    relation: str
    obj: str
    status: str
    message: str
    evidence: List[Tuple[str, str, str]]
    suggestions: List[str]


class CommonsenseKnowledge:
    """Simple knowledge store capturing positive and negative triples."""

    def __init__(self) -> None:
        self._positives: Dict[Tuple[str, str], List[str]] = {}
        self._negatives: Dict[Tuple[str, str], List[str]] = {}

    def add_fact(
        self,
        subject: str,
        relation: str,
        obj: str,
        *,
        truth: bool = True,
    ) -> None:
        key = (subject.lower(), relation.lower())
        target = self._positives if truth else self._negatives
        target.setdefault(key, []).append(obj.lower())

    def extend(self, facts: Iterable[Tuple[str, str, str, bool]]) -> None:
        for subject, relation, obj, truth in facts:
            self.add_fact(subject, relation, obj, truth=truth)

    def supports(self, subject: str, relation: str, obj: str) -> bool:
        key = (subject.lower(), relation.lower())
        return obj.lower() in self._positives.get(key, [])

    def contradicts(self, subject: str, relation: str, obj: str) -> bool:
        key = (subject.lower(), relation.lower())
        return obj.lower() in self._negatives.get(key, [])

    def evidence(self, subject: str, relation: str) -> List[Tuple[str, str, str]]:
        key = (subject.lower(), relation.lower())
        positives = [(subject, relation, obj) for obj in self._positives.get(key, [])]
        negatives = [(subject, f"not {relation}", obj) for obj in self._negatives.get(key, [])]
        return positives + negatives


class CommonsenseValidator:
    """Validate structured statements against commonsense knowledge."""

    def __init__(
        self,
        knowledge: CommonsenseKnowledge,
        *,
        auto_suggest: bool = True,
    ) -> None:
        self.knowledge = knowledge
        self.auto_suggest = auto_suggest

    def validate(
        self,
        subject: str,
        relation: str,
        obj: str,
        *,
        context: Sequence[str] | None = None,
    ) -> CommonsenseJudgement:
        subject = subject.strip()
        relation = relation.strip()
        obj = obj.strip()
        context = list(context or [])

        evidence = self.knowledge.evidence(subject, relation)
        if self.knowledge.contradicts(subject, relation, obj):
            suggestions = []
            if self.auto_suggest and evidence:
                positives = [trip for trip in evidence if not trip[1].startswith("not ")]
                suggestions = [f"Consider: {s} {r} {o}" for s, r, o in positives[:3]]
            return CommonsenseJudgement(
                subject,
                relation,
                obj,
                "contradiction",
                f"Statement conflicts with known commonsense: {subject} {relation} {obj}.",
                evidence,
                suggestions,
            )

        if self.knowledge.supports(subject, relation, obj):
            return CommonsenseJudgement(
                subject,
                relation,
                obj,
                "consistent",
                "Statement aligns with commonsense knowledge.",
                evidence,
                [],
            )

        hints: List[str] = []
        if context:
            hints.append("Context inspected: " + "; ".join(context[:3]))
        hints.extend([f"Known fact: {s} {r} {o}" for s, r, o in evidence[:3]])
        return CommonsenseJudgement(
            subject,
            relation,
            obj,
            "unknown",
            "No matching commonsense facts were found; treat statement cautiously.",
            evidence,
            hints,
        )
