"""Tests for the governance charter loader."""

from pathlib import Path

import pytest
import yaml
import json

from autogpt.governance.charter import (
    Charter,
    CharterValidationError,
    load_charter,
)


def test_load_charter_from_yaml(tmp_path: Path) -> None:
    data = {
        "name": "Test Charter",
        "core_directives": ["Be kind"],
        "roles": [
            {
                "name": "assistant",
                "permissions": [{"name": "write"}],
                "allowed_tasks": ["respond"],
            }
        ],
    }
    path = tmp_path / "test.yaml"
    path.write_text(yaml.safe_dump(data))

    charter = load_charter("test", directory=tmp_path)
    assert isinstance(charter, Charter)
    assert charter.roles[0].permissions[0].name == "write"


def test_invalid_charter_raises(tmp_path: Path) -> None:
    """Duplicate role names should trigger validation errors."""
    data = {
        "name": "Bad Charter",
        "roles": [{"name": "dup"}, {"name": "dup"}],
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))

    with pytest.raises(CharterValidationError):
        load_charter("bad", directory=tmp_path)
