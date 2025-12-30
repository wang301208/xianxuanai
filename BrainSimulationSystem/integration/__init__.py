"""Integration layer exposing high-level system orchestration utilities."""

from .thalamocortical_integration import ThalamocorticalIntegration, ThalamocorticalConfig
from .llm_service import LLMService

__all__ = ['ThalamocorticalIntegration', 'ThalamocorticalConfig', 'LLMService']
