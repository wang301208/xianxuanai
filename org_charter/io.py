"""Blueprint schema loader for agent blueprint YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCHEMA_PATH = _REPO_ROOT / "modules" / "schemas" / "agent_blueprint.yaml"


def _load_schema(path: Path) -> Dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        payload = None
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid blueprint schema at {path}")
    return payload


BLUEPRINT_SCHEMA: Dict[str, Any] = _load_schema(_SCHEMA_PATH)


__all__ = ["BLUEPRINT_SCHEMA"]

