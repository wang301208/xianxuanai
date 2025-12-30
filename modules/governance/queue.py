"""Simple proposal queue for founder-agent blueprints."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List
import subprocess

import yaml
from jsonschema import validate

from org_charter.io import BLUEPRINT_SCHEMA
from third_party.autogpt.autogpt.core.errors import AutoGPTError
from third_party.autogpt.autogpt.core.logging import handle_exception

# Default directories relative to repository root
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROPOSAL_DIR = REPO_ROOT / "org_charter" / "proposals"
DEFAULT_BLUEPRINT_DIR = REPO_ROOT / "org_charter" / "blueprints"


class ProposalQueue:
    """Queue storing blueprint proposals awaiting human approval."""

    def __init__(
        self,
        proposal_dir: Path | None = None,
        blueprint_dir: Path | None = None,
        reload_callback: Callable[[], None] | None = None,
    ) -> None:
        self.proposal_dir = Path(proposal_dir or DEFAULT_PROPOSAL_DIR)
        self.blueprint_dir = Path(blueprint_dir or DEFAULT_BLUEPRINT_DIR)
        self.reload_callback = reload_callback

    # ------------------------------------------------------------------
    # proposal submission and listing
    # ------------------------------------------------------------------
    def enqueue(self, data: Dict[str, Any]) -> Path:
        """Persist *data* as a proposal and return its path."""
        self.proposal_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        path = self.proposal_dir / f"proposal_{timestamp}.yaml"
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        return path

    def list_pending(self) -> List[Path]:
        """Return paths to all pending proposal files."""
        if not self.proposal_dir.exists():
            return []
        return sorted(self.proposal_dir.glob("*.yaml"))

    # ------------------------------------------------------------------
    # proposal approval workflow
    # ------------------------------------------------------------------
    def approve(self, proposal_path: Path, commit: bool = True) -> Path:
        """Approve *proposal_path* and merge into blueprints."""
        with open(proposal_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        validate(instance=data, schema=BLUEPRINT_SCHEMA)

        name = data.get("role_name")
        if not name:
            raise ValueError("Proposal missing 'role_name'")

        version = self._next_version(name)
        self.blueprint_dir.mkdir(parents=True, exist_ok=True)
        blueprint_path = self.blueprint_dir / f"{name}_v{version}.yaml"
        with open(blueprint_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)

        if commit:
            subprocess.run(["git", "add", str(blueprint_path)], check=True)
            subprocess.run(
                ["git", "commit", "-m", f"Add blueprint {name} v{version}"],
                check=True,
            )

        proposal_path.unlink()
        self._trigger_reload()
        return blueprint_path

    def reject(self, proposal_path: Path) -> None:
        """Remove *proposal_path* without merging."""
        proposal_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _next_version(self, name: str) -> int:
        versions = []
        for file in self.blueprint_dir.glob(f"{name}_v*.yaml"):
            try:
                versions.append(int(file.stem.split("_v")[-1]))
            except ValueError:
                continue
        return max(versions, default=0) + 1

    def _trigger_reload(self) -> None:
        """Reload agents from updated blueprints."""
        if self.reload_callback:
            self.reload_callback()
            return

        try:
            from agent_factory import create_agent_from_blueprint

            for blueprint in self.blueprint_dir.glob("*.yaml"):
                create_agent_from_blueprint(blueprint)
        except AutoGPTError as err:
            handle_exception(err)
