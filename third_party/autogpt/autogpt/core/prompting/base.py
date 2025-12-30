"""Abstract definitions for prompt strategies."""

from __future__ import annotations

import abc

from autogpt.core.configuration import SystemConfiguration
from autogpt.core.resource.model_providers.schema import AssistantChatMessage

from .schema import ChatPrompt, LanguageModelClassification


class PromptStrategy(abc.ABC):
    """Interface for constructing prompts and parsing model responses."""

    default_configuration: SystemConfiguration

    @property
    @abc.abstractmethod
    def model_classification(self) -> LanguageModelClassification:
        """Return the classification of model this strategy targets."""

    @abc.abstractmethod
    def build_prompt(self, *_, **kwargs) -> ChatPrompt:
        """Construct the prompt for the language model."""

    @abc.abstractmethod
    def parse_response_content(self, response_content: AssistantChatMessage):
        """Parse the content of the model's response."""
