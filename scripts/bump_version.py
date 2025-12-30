#!/usr/bin/env python3
"""Bump project version stored in pyproject.toml and create a git tag."""
from __future__ import annotations

import argparse
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
CHANGELOG = ROOT / "CHANGELOG.md"


def bump(part: str) -> str:
    content = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"(\d+)\.(\d+)\.(\d+)"', content, re.MULTILINE)
    if not match:
        raise RuntimeError("Could not find version in pyproject.toml")
    major, minor, patch = map(int, match.groups())
    if part == "major":
        major, minor, patch = major + 1, 0, 0
    elif part == "minor":
        minor, patch = minor + 1, 0
    else:
        patch += 1
    new_version = f"{major}.{minor}.{patch}"
    new_content = re.sub(r'^version\s*=\s*"\d+\.\d+\.\d+"', f'version = "{new_version}"', content, flags=re.MULTILINE)
    PYPROJECT.write_text(new_content, encoding="utf-8")
    if CHANGELOG.exists():
        changelog = CHANGELOG.read_text(encoding="utf-8")
        if f"## {new_version}" not in changelog:
            changelog = re.sub(
                r"# Changelog\n",
                f"# Changelog\n\n## {new_version}\n- Describe changes here.\n",
                changelog,
                count=1,
            )
            CHANGELOG.write_text(changelog, encoding="utf-8")
    subprocess.run(["git", "tag", f"v{new_version}"], check=True)
    return new_version


def main() -> None:
    parser = argparse.ArgumentParser(description="Bump project version.")
    parser.add_argument("part", choices=["major", "minor", "patch"], help="Part of the version to bump")
    args = parser.parse_args()
    version = bump(args.part)
    print(version)


if __name__ == "__main__":
    main()
