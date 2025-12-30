"""Public API for spawning agents from blueprint files."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from third_party.autogpt.autogpt.config import Config
from third_party.autogpt.autogpt.core.resource.model_providers import ChatModelProvider
from third_party.autogpt.autogpt.file_storage.base import FileStorage

from agent_factory import create_agent_from_blueprint
from concept_alignment import ConceptAligner
from creative_engine import CrossModalCreativeEngine
from modules.common.concepts import ConceptNode


def spawn_agent(
    blueprint_path: str | Path,
    *,
    config: Config,
    llm_provider: ChatModelProvider,
    file_storage: FileStorage,
    world_model: "WorldModel | None" = None,
):
    """Create a new AutoGPT agent from a blueprint.

    Parameters
    ----------
    blueprint_path: str | Path
        Path to the blueprint YAML file describing the agent.
    config: Config
        Application configuration to apply to the agent.
    llm_provider: ChatModelProvider
        LLM provider used for the agent's thinking.
    file_storage: FileStorage
        Storage backend for the agent's file operations.
    """
    return create_agent_from_blueprint(
        blueprint_path,
        config=config,
        llm_provider=llm_provider,
        file_storage=file_storage,
        world_model=world_model,
    )


def create_creative_engine(
    aligner: ConceptAligner,
    encoders: Dict[str, Callable[[str], List[float]]],
    generators: Optional[Dict[str, Callable[[str, List[ConceptNode]], Any]]] = None,
) -> CrossModalCreativeEngine:
    """Instantiate a :class:`CrossModalCreativeEngine` for creative synthesis."""
    return CrossModalCreativeEngine(
        aligner=aligner, encoders=encoders, generators=generators
    )


__all__ = ["spawn_agent", "create_creative_engine"]
