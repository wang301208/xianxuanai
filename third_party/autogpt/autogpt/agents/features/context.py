from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from autogpt.core.resource.model_providers import ChatMessage

if TYPE_CHECKING:
    from autogpt.core.prompting import ChatPrompt
    from autogpt.models.context_item import ContextItem

    from ..base import BaseAgent


class AgentContext:
    """Manages contextual information for agent interactions."""

    items: list[ContextItem]

    def __init__(self, items: Optional[list[ContextItem]] = None):
        """Initialize the agent context."""

        self.items = items or []

    def __bool__(self) -> bool:
        """Return whether the context contains any items."""

        return len(self.items) > 0

    def __contains__(self, item: ContextItem) -> bool:
        """Return whether an item with the same source exists in the context."""

        return any(existing.source == item.source for existing in self.items)

    def add(self, item: ContextItem) -> None:
        """Add a new context item to the collection."""

        self.items.append(item)

    def close(self, index: int) -> None:
        """Remove a context item by its 1-based index."""

        self.items.pop(index - 1)

    def clear(self) -> None:
        """Remove all context items from the collection."""

        self.items.clear()

    def format_numbered(self) -> str:
        """Format the context items as a numbered list suitable for prompts."""

        return "\n\n".join(f"{i}. {item.fmt()}" for i, item in enumerate(self.items, 1))


class ContextMixin:
    """Mixin that adds context support to a ``BaseAgent`` subclass."""

    context: AgentContext

    def __init__(self, **kwargs: Any):
        self.context = AgentContext()
        super().__init__(**kwargs)

    async def build_prompt(
        self,
        *args: Any,
        extra_messages: Optional[list[ChatMessage]] = None,
        **kwargs: Any,
    ) -> ChatPrompt:
        extra_messages = list(extra_messages or [])

        if self.context:
            extra_messages.insert(
                0,
                ChatMessage.system(
                    "## Context\n",
                    f"{self.context.format_numbered()}\n\n",
                    "When a context item is no longer needed and you are not done yet, "
                    "you can hide the item by specifying its number in the list above "
                    "to `hide_context_item`.",
                ),
            )

        return await super().build_prompt(
            *args,
            extra_messages=extra_messages,
            **kwargs,
        )  # type: ignore


def get_agent_context(agent: BaseAgent) -> AgentContext | None:
    if isinstance(agent, ContextMixin):
        return agent.context

    return None
