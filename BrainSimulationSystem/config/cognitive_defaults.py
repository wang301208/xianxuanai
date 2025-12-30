"""
Shared defaults for cognitive controller components.

The values here document the expected operating ranges for the attention
system and the working memory module used by the API and visualization
fallbacks.  They can be overridden by providing a JSON file via the
``BRAIN_COGNITIVE_CONFIG_PATH`` environment variable or by passing explicit
overrides to the helper functions exposed in this module.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any, Dict

CONFIG_ENV_VAR = "BRAIN_COGNITIVE_CONFIG_PATH"

# Default parameters for the attention subsystem.  These values keep the stub
# behaviour aligned with the production controller defaults while remaining
# easily discoverable and configurable.
DEFAULT_ATTENTION_PARAMS: Dict[str, float] = {
    "ach_sensitivity": 1.2,
    "bottom_up_weight": 0.6,
    "top_down_weight": 0.4,
    "attention_span": 3,
}

# Default parameters for the working memory subsystem.  They approximate the
# behaviour of the acetylcholine modulated implementation and summarise the
# meaning of the original literals (for example, a capacity of seven items is
# the classic Miller limit for short term memory).
DEFAULT_WORKING_MEMORY_PARAMS: Dict[str, float] = {
    "capacity": 7,
    "decay_rate": 0.05,
    "ach_sensitivity": 1.5,
    "attention_boost": 0.4,
    "encoding_factor": 1.3,
    "retrieval_threshold": 0.35,
    "interference_factor": 0.75,
}


def _deep_merge(
    base: Dict[str, Any],
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overrides.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_env_config() -> Dict[str, Any]:
    """Return overrides from the JSON file referenced by the env var, if any."""

    path = os.getenv(CONFIG_ENV_VAR)
    if not path:
        return {}

    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        # Invalid paths or malformed files are ignored to keep the fallback
        # controller usable even without optional configuration.
        return {}

    return data if isinstance(data, dict) else {}


def load_cognitive_defaults(
    overrides: Dict[str, Dict[str, Any]] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Return the merged cognitive controller defaults.

    Order of precedence (ascending):
        1. Module defaults defined above.
        2. JSON overrides supplied via ``BRAIN_COGNITIVE_CONFIG_PATH``.
        3. Explicit overrides provided to this function.
    """

    defaults: Dict[str, Dict[str, Any]] = {
        "attention": deepcopy(DEFAULT_ATTENTION_PARAMS),
        "working_memory": deepcopy(DEFAULT_WORKING_MEMORY_PARAMS),
    }

    env_config = _load_env_config()
    if env_config:
        defaults = _deep_merge(defaults, env_config)

    if overrides:
        defaults = _deep_merge(defaults, overrides)

    return defaults


def load_component_defaults(
    component: str,
    overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Convenience helper to obtain defaults for a specific component.

    ``component`` should be either ``\"attention\"`` or ``\"working_memory\"``.
    Unrecognised names return an empty dictionary so callers remain robust.
    """

    all_defaults = load_cognitive_defaults()
    values = deepcopy(all_defaults.get(component, {}))
    if overrides:
        values.update(overrides)
    return values

