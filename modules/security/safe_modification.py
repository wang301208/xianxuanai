"""Utilities for safely testing self-modifying behaviour.

This module provides a sandboxed environment that allows code to modify files in
isolation before committing them back to the original location. Every change is
tracked in an audit log and can be rolled back if a failure occurs.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional


class SafeModificationSandbox:
    """Create an isolated copy of a directory to test self-modification.

    The sandbox works by copying the target directory to a temporary location
    where modifications can be applied. On :meth:`commit` the changes are copied
    back to the original directory. If an error happens during commit or if the
    changes are deemed unsafe, :meth:`rollback` restores the previous state.
    """

    def __init__(self, target_path: Path | str):
        self.target_path = Path(target_path)
        self._tmp_root: Optional[Path] = None
        self._sandbox_path: Optional[Path] = None
        self.audit_log: List[str] = []
        self._backups: Dict[Path, Path] = {}

    # ------------------------------------------------------------------
    # Context manager interface
    def __enter__(self) -> "SafeModificationSandbox":
        self._tmp_root = Path(tempfile.mkdtemp(prefix="autogpt_sandbox_"))
        self._sandbox_path = self._tmp_root / "snapshot"
        shutil.copytree(self.target_path, self._sandbox_path)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401 - part of CM protocol
        # If an exception occurred and commit wasn't successful, rollback.
        if exc_type:
            self.rollback()
        else:
            self.cleanup()

    # ------------------------------------------------------------------
    @property
    def sandbox_path(self) -> Path:
        if not self._sandbox_path:
            raise RuntimeError("Sandbox has not been initialized")
        return self._sandbox_path

    # ------------------------------------------------------------------
    def apply_change(self, relative_path: str, content: str) -> None:
        """Apply a modification inside the sandbox.

        Args:
            relative_path: Path within the sandbox to modify.
            content: New file contents.

        Raises:
            ValueError: If the path attempts to escape the sandbox.
        """

        sandbox = self.sandbox_path
        dest = (sandbox / relative_path).resolve()
        # Prevent path traversal attacks
        if not str(dest).startswith(str(sandbox.resolve())):
            raise ValueError("Modification escapes sandbox")

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)
        self.audit_log.append(relative_path)

    # ------------------------------------------------------------------
    def commit(self) -> None:
        """Commit changes back to the target directory.

        If an error occurs while writing the files, the previous state is
        restored automatically.
        """

        sandbox = self.sandbox_path
        try:
            for rel_path in self.audit_log:
                src = sandbox / rel_path
                dest = self.target_path / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                # Save a backup before overwriting so we can roll back later
                if dest.exists() and dest not in self._backups:
                    backup = self._tmp_root / "backup" / rel_path
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(dest, backup)
                    self._backups[dest] = backup
                shutil.copy2(src, dest)
        except Exception:
            # On any failure, revert to backed up state
            self.rollback()
            raise
        else:
            self.cleanup()

    # ------------------------------------------------------------------
    def rollback(self) -> None:
        """Restore original files and discard sandbox changes."""

        # Restore backups
        for dest, backup in self._backups.items():
            shutil.copy2(backup, dest)

        # Remove newly created files
        for rel_path in self.audit_log:
            dest = self.target_path / rel_path
            if dest not in self._backups and dest.exists():
                os.remove(dest)

        self.cleanup()

    # ------------------------------------------------------------------
    def cleanup(self) -> None:
        """Delete temporary files and reset internal state."""

        if self._tmp_root and self._tmp_root.exists():
            shutil.rmtree(self._tmp_root)
        self._tmp_root = None
        self._sandbox_path = None
        self.audit_log.clear()
        self._backups.clear()
