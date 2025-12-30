from __future__ import annotations

import os

from pydantic import BaseSettings, Field, SecretStr

from autogpt.core.brain.config import BrainBackend


class EnvConfig(BaseSettings):
    """Required environment variables for AutoGPT."""

    openai_api_key: SecretStr = Field(..., env="OPENAI_API_KEY")


def validate_env() -> EnvConfig | None:
    """Validate required environment variables.

    Returns the parsed ``EnvConfig``. Raises ``ValidationError`` if validation
    fails.
    """

    if os.getenv("BRAIN_BACKEND", "").lower() == BrainBackend.WHOLE_BRAIN.value:
        return None

    EnvConfig.update_forward_refs(SecretStr=SecretStr)
    return EnvConfig()
