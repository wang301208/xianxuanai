#!/usr/bin/env python3
"""Validate that version numbers are consistent."""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
CHANGELOG = ROOT / "CHANGELOG.md"


def get_version() -> str | None:
    content = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"(\d+\.\d+\.\d+)"', content, re.MULTILINE)
    return match.group(1) if match else None


def get_changelog_version() -> str | None:
    if not CHANGELOG.exists():
        return None
    content = CHANGELOG.read_text(encoding="utf-8")
    match = re.search(r'^##\s+(\d+\.\d+\.\d+)', content, re.MULTILINE)
    return match.group(1) if match else None


def main() -> None:
    version = get_version()
    changelog_version = get_changelog_version()
    if not version:
        print("Version is missing or not in MAJOR.MINOR.PATCH format in pyproject.toml")
        sys.exit(1)
    if changelog_version is None:
        print("No CHANGELOG.md found; skipping changelog version check")
    elif version != changelog_version:
        print("Version in CHANGELOG.md does not match pyproject.toml")
        sys.exit(1)
    print(f"Version {version} verified")


if __name__ == "__main__":
    main()
