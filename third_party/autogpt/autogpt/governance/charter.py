"""Governance charter models and loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, ValidationError, validator


class CharterError(Exception):
    """Base exception for charter related errors."""


class CharterValidationError(CharterError):
    """Raised when a charter definition fails validation."""


class Permission(BaseModel):
    """A specific action an agent is permitted to perform."""

    name: str
    description: Optional[str] = None


class Role(BaseModel):
    """An agent role comprised of permissions and allowed tasks."""

    name: str
    description: Optional[str] = None
    permissions: List[Permission] = []
    allowed_tasks: List[str] = []

    @validator("permissions")
    def unique_permissions(cls, perms: List[Permission]) -> List[Permission]:  # noqa: N805
        names = [p.name for p in perms]
        if len(names) != len(set(names)):
            raise ValueError("permission names must be unique")
        return perms


class Charter(BaseModel):
    """Top-level governance charter definition."""

    name: str
    roles: List[Role]
    core_directives: List[str] = []

    @validator("roles")
    def unique_roles(cls, roles: List[Role]) -> List[Role]:  # noqa: N805
        names = [r.name for r in roles]
        if len(names) != len(set(names)):
            raise ValueError("role names must be unique")
        return roles


def load_charter(name: str, directory: Optional[Path] = None) -> Charter:
    """Load a charter definition by name.

    Parameters
    ----------
    name
        Base name of the charter file (without extension).
    directory
        Directory to search for charter files. Defaults to the built-in
        ``data/charter`` directory.
    """

    if directory is None:
        package_dir = Path(__file__).resolve().parent.parent
        candidate_dirs = (
            package_dir / "data" / "charter",
            package_dir.parent / "data" / "charter",
        )
        directory = next((path for path in candidate_dirs if path.exists()), candidate_dirs[0])

    for ext in (".json", ".yaml", ".yml"):
        file_path = directory / f"{name}{ext}"
        if file_path.exists():
            break
    else:
        raise FileNotFoundError(
            f"Charter file '{name}' not found in {directory}."
        )

    try:
        if file_path.suffix == ".json":
            data = json.loads(file_path.read_text())
        else:
            data = yaml.safe_load(file_path.read_text())
    except Exception as exc:  # pragma: no cover - unexpected parse errors
        raise CharterValidationError(
            f"Failed to parse charter file '{file_path}': {exc}"
        ) from exc

    try:
        return Charter.parse_obj(data)
    except ValidationError as exc:
        raise CharterValidationError(
            f"Invalid charter data in '{file_path}': {exc}"
        ) from exc
