import json
import logging
import math
from collections import Counter

from autogpt.core.configuration import Configurable, SystemConfiguration, SystemSettings
from autogpt.core.knowledge_graph import (
    EntityType,
    RelationType,
    get_graph_store,
)
from autogpt.core.memory.base import Memory
from autogpt.core.workspace import Workspace


class MemoryConfiguration(SystemConfiguration):
    pass


class MemorySettings(SystemSettings):
    configuration: MemoryConfiguration


class MessageHistory:
    def __init__(self, previous_message_history: list[str]):
        self._message_history = previous_message_history

    def append(self, message: str) -> None:
        self._message_history.append(message)

    def as_list(self) -> list[str]:
        return self._message_history


class SimpleMemory(Memory, Configurable):
    default_settings = MemorySettings(
        name="simple_memory",
        description="A simple memory.",
        configuration=MemoryConfiguration(),
    )

    def __init__(
        self,
        settings: MemorySettings,
        logger: logging.Logger,
        workspace: Workspace,
    ):
        self._configuration = settings.configuration
        self._logger = logger
        self._workspace = workspace
        self._graph = get_graph_store()
        self._message_history = self._load_message_history(workspace)
        self._summary_archive = self._load_summary_archive(workspace)
        self._ability_scores = self._load_scores(workspace)

    @staticmethod
    def _load_message_history(workspace: Workspace):
        message_history_path = workspace.get_path("message_history.json")
        if message_history_path.exists():
            with message_history_path.open("r") as f:
                message_history = json.load(f)
        else:
            message_history = []
        return MessageHistory(message_history)

    @staticmethod
    def _load_summary_archive(workspace: Workspace) -> list[str]:
        archive_path = workspace.get_path("long_term_memory.json")
        if archive_path.exists():
            with archive_path.open("r") as f:
                return json.load(f)
        return []

    @staticmethod
    def _load_scores(workspace: Workspace) -> list[dict]:
        path = workspace.get_path("ability_scores.json")
        if path.exists():
            with path.open("r") as f:
                return json.load(f)
        return []

    def add(self, message: str) -> None:
        """Store a message in persistent memory."""
        self._message_history.append(message)
        path = self._workspace.get_path("message_history.json")
        with path.open("w") as f:
            json.dump(self._message_history.as_list(), f)
        # After adding a message, attempt to archive if history grows large
        self.summarize_and_archive()

    def summarize_and_archive(self, max_history_length: int = 100) -> None:
        """Summarize old messages and archive them as long-term memory."""
        messages = self._message_history.as_list()
        if len(messages) <= max_history_length:
            return

        old_messages = messages[:-max_history_length]
        summary = self._summarize(old_messages)
        if summary:
            self._summary_archive.append(summary)
            archive_path = self._workspace.get_path("long_term_memory.json")
            with archive_path.open("w") as f:
                json.dump(self._summary_archive, f)

        # keep only recent messages
        self._message_history = MessageHistory(messages[-max_history_length:])
        path = self._workspace.get_path("message_history.json")
        with path.open("w") as f:
            json.dump(self._message_history.as_list(), f)

    def _summarize(self, messages: list[str]) -> str:
        """Very simple summarization by concatenation."""
        if not messages:
            return ""
        summary = " ".join(messages)
        # limit summary size
        return summary[:1000]

    # --- Similarity search helpers ---
    def _vectorize(self, text: str) -> Counter:
        return Counter(text.lower().split())

    def _cosine_similarity(self, vec1: Counter, vec2: Counter) -> float:
        intersection = set(vec1.keys()) & set(vec2.keys())
        dot_product = sum(vec1[x] * vec2[x] for x in intersection)
        mag1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        mag2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
        if mag1 == 0 or mag2 == 0:
            return 0.0
        return dot_product / (mag1 * mag2)

    def _semantic_search(
        self, query: str, texts: list[str], limit: int | None = None
    ) -> list[str]:
        query_vec = self._vectorize(query)
        scored = []
        for text in texts:
            sim = self._cosine_similarity(query_vec, self._vectorize(text))
            scored.append((sim, text))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [t for s, t in scored if s > 0]
        return results[:limit] if limit else results

    def get(
        self, limit: int | None = None, query: str | None = None
    ) -> list[str]:
        """Return messages from memory or query for relevant ones.

        Args:
            limit: If provided, return only the most relevant `limit` messages.
            query: If provided, perform similarity search across archived and active
                memories and return those most relevant to the query.

        Returns:
            List of stored messages.
        """
        if query:
            texts = self._message_history.as_list() + self._summary_archive
            return self._semantic_search(query, texts, limit)

        messages = self._message_history.as_list()
        return messages[-limit:] if limit else messages

    # --- Ability score helpers ---
    def log_score(
        self, task_id: str, task_description: str, ability: str, score: float
    ) -> None:
        entry = {
            "task_id": task_id,
            "task_description": task_description,
            "ability": ability,
            "score": score,
        }
        self._ability_scores.append(entry)
        path = self._workspace.get_path("ability_scores.json")
        with path.open("w") as f:
            json.dump(self._ability_scores, f)

        # Store task and ability relation in the knowledge graph
        self.store_task(task_id, task_description, ability, score)

    def get_scores_for_task(self, task_description: str, ability: str) -> list[float]:
        entries = [e for e in self._ability_scores if e.get("ability") == ability]
        if not entries:
            return []
        query_vec = self._vectorize(task_description)
        scored = []
        for e in entries:
            sim = self._cosine_similarity(
                query_vec, self._vectorize(e.get("task_description", ""))
            )
            if sim > 0:
                scored.append((sim, e["score"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [score for _, score in scored]

    # -- Knowledge graph helpers ---------------------------------------------
    def store_task(
        self, task_id: str, description: str, ability: str, score: float | None = None
    ) -> None:
        """Persist task/skill info into the knowledge graph."""

        self._graph.add_node(task_id, EntityType.TASK, description=description)
        self._graph.add_node(ability, EntityType.SKILL)
        properties = {"score": score} if score is not None else {}
        self._graph.add_edge(task_id, ability, RelationType.REQUIRES, **properties)

    def query_graph(self, **kwargs):
        """Proxy to the underlying graph store's query method."""

        return self._graph.query(**kwargs)
