"""Conflict detection and resolution for Genesis team."""

from __future__ import annotations

import json
import difflib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Iterable, List


class ConflictDetectionStrategy(ABC):
    """Interface for conflict detection strategies."""

    @abstractmethod
    def detect(self, agent_name: str, logs: Dict[str, str]) -> bool:
        """Return True if a conflict is present for the given agent."""


class KeywordConflictStrategy(ConflictDetectionStrategy):
    """Detect conflicts based on presence of configured keywords."""

    def __init__(self, keywords: Iterable[str] | None = None) -> None:
        self.keywords = [
            keyword.lower()
            for keyword in (keywords or ["conflict", "version mismatch", "overlap", "error"])
        ]

    def detect(self, agent_name: str, logs: Dict[str, str]) -> bool:
        output = logs.get(agent_name, "").lower()
        return any(keyword in output for keyword in self.keywords)


class StructuredDataConflictStrategy(ConflictDetectionStrategy):
    """Parse JSON outputs and detect overlapping edits or version mismatches."""

    def detect(self, agent_name: str, logs: Dict[str, str]) -> bool:  # noqa: D401 - see base class
        try:
            current = json.loads(logs[agent_name])
        except (KeyError, json.JSONDecodeError):
            return False

        file_name = current.get("file")
        version = current.get("version")
        content = current.get("content", "")

        for name, raw in logs.items():
            if name == agent_name:
                continue
            try:
                other = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if other.get("file") != file_name:
                continue
            if other.get("version") != version:
                return True  # version mismatch
            other_content = other.get("content", "")
            if other_content != content:
                diff = list(
                    difflib.unified_diff(
                        other_content.splitlines(),
                        content.splitlines(),
                        lineterm="",
                    )
                )
                if diff:
                    return True  # overlapping edits
        return False


@dataclass
class ConflictResolver:
    """Detects conflicts in agent outputs and coordinates resolution."""

    history: List[str] = field(default_factory=list)
    strategy: ConflictDetectionStrategy = field(default_factory=KeywordConflictStrategy)

    def detect(self, agent_name: str, logs: Dict[str, str]) -> bool:
        """Return True if the agent's output conflicts with existing logs."""

        return self.strategy.detect(agent_name, logs)

    def resolve(self, agent_name: str, logs: Dict[str, str]) -> str:
        """Resolve detected conflicts by rolling back or merging.

        Parameters
        ----------
        agent_name
            Name of the agent whose output was most recently produced.
        logs
            Mapping of agent names to their output logs.

        Returns
        -------
        str
            Decision summary, either a merge or rollback description.
        """

        output = logs[agent_name]
        if self.detect(agent_name, logs):
            logs[agent_name] = f"ROLLED BACK: {output}"
            decision = f"{agent_name}: rollback"
        else:
            decision = f"{agent_name}: merge"
        self.history.append(decision)
        return decision
