"""I/O utilities for organization charters and agent blueprints."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import subprocess

import yaml
from jsonschema import validate

SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent / "schemas" / "agent_blueprint.yaml"
)
BLUEPRINT_DIR = Path(__file__).resolve().parent / "blueprints"

with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    BLUEPRINT_SCHEMA = yaml.safe_load(f)


def _next_version(name: str) -> int:
    """Return the next available version number for a blueprint name."""
    versions = []
    for file in BLUEPRINT_DIR.glob(f"{name}_v*.yaml"):
        try:
            versions.append(int(file.stem.split("_v")[-1]))
        except ValueError:
            continue
    return max(versions, default=0) + 1


def save_blueprint(
    data: Dict[str, Any], name: Optional[str] = None, commit: bool = True
) -> Path:
    """Persist a blueprint to disk and optionally commit it to git.

    Args:
        data: Blueprint content conforming to ``schemas/agent_blueprint.yaml``.
        name: Optional override for the blueprint name. Defaults to
            ``data['role_name']``.
        commit: Whether to stage and commit the saved file for audit purposes.

    Returns:
        Path to the saved blueprint file.
    """
    if name is None:
        name = data.get("role_name")
    if not name:
        raise ValueError(
            "Blueprint name must be provided or included in data['role_name']"
        )

    validate(instance=data, schema=BLUEPRINT_SCHEMA)

    version = _next_version(name)
    BLUEPRINT_DIR.mkdir(parents=True, exist_ok=True)
    path = BLUEPRINT_DIR / f"{name}_v{version}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    if commit:
        subprocess.run(["git", "add", str(path)], check=True)
        subprocess.run(
            ["git", "commit", "-m", f"Add blueprint {name} v{version}"],
            check=True,
        )

    return path


def load_blueprint(name: str, version: Optional[int] = None) -> Dict[str, Any]:
    """Load a blueprint by name and optional version.

    Args:
        name: Blueprint name (typically the ``role_name``).
        version: Specific version number to load. If omitted, the latest is used.

    Returns:
        Parsed blueprint dictionary.
    """
    if version is None:
        candidates = sorted(BLUEPRINT_DIR.glob(f"{name}_v*.yaml"))
        if not candidates:
            raise FileNotFoundError(f"No blueprint found for {name}")
        path = candidates[-1]
    else:
        path = BLUEPRINT_DIR / f"{name}_v{version}.yaml"

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    validate(instance=data, schema=BLUEPRINT_SCHEMA)
    return data
