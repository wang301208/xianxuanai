from __future__ import annotations

import logging
import os

from pydantic import BaseSettings, Field, SecretStr

from autogpt.core.brain.config import BrainBackend

logger = logging.getLogger(__name__)


class EnvConfig(BaseSettings):
    """Required environment variables for AutoGPT."""

    openai_api_key: SecretStr = Field(..., env="OPENAI_API_KEY")

def _brain_backend_from_env() -> BrainBackend:
    value = os.getenv("BRAIN_BACKEND")
    if not value:
        return BrainBackend.BRAIN_SIMULATION
    try:
        return BrainBackend(value.lower())
    except ValueError:
        logger.warning(
            "Invalid BRAIN_BACKEND value '%s'; falling back to '%s'.",
            value,
            BrainBackend.BRAIN_SIMULATION.value,
        )
        return BrainBackend.BRAIN_SIMULATION


def validate_env() -> EnvConfig | None:
    """Validate required environment variables.

    Returns the parsed ``EnvConfig``. Raises ``ValidationError`` if validation
    fails.
    """

    if _brain_backend_from_env() != BrainBackend.LLM:
        return None

    EnvConfig.update_forward_refs(SecretStr=SecretStr)
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key:
        return EnvConfig(openai_api_key=SecretStr(api_key))
    return EnvConfig()
