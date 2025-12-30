"""Utilities for recording and querying commit approvals.

This module uses git notes to track which commits have been approved by a
`human_architect`.  The notes are stored in the local git repository and can be
shared like any other ref.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
from typing import List, Optional


class ApprovalService:
    """Encapsulates commit approval logic for a git repository."""

    def __init__(self, repo_path: Optional[Path | str] = None) -> None:
        self.repo_path = Path(repo_path or ".").resolve()

    def _run_git(self, *args: str) -> str:
        """Run a git command and return its stdout."""
        result = subprocess.run(
            ["git", *args],
            cwd=self.repo_path,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    # ------------------------------------------------------------------
    # query helpers
    # ------------------------------------------------------------------
    def approvals_for(self, commit: str) -> List[str]:
        """Return a list of approvers for the given commit hash."""
        try:
            notes = self._run_git("notes", "show", commit)
        except subprocess.CalledProcessError:
            return []
        approvers = []
        for line in notes.splitlines():
            if line.lower().startswith("approved-by:"):
                approvers.append(line.split(":", 1)[1].strip())
        return approvers

    def is_approved(self, commit: str) -> bool:
        """Return True if the commit has any recorded approvals."""
        return bool(self.approvals_for(commit))

    # ------------------------------------------------------------------
    # approval workflow
    # ------------------------------------------------------------------
    def add_approval(self, commit: str, approver: str) -> None:
        """Record an approval for ``commit`` by ``approver``.

        Approvals are stored using ``git notes`` so the underlying commit
        remains unchanged.
        """
        note = f"Approved-by: {approver}"
        try:
            # append if a note already exists
            self._run_git("notes", "append", "-m", note, commit)
        except subprocess.CalledProcessError:
            # otherwise create a new note
            self._run_git("notes", "add", "-m", note, commit)

    def pending_commits(self, base: str = "origin/main", head: str = "HEAD") -> List[str]:
        """Return commits between ``base`` and ``head`` lacking approval."""
        revs = self._run_git("rev-list", f"{base}..{head}")
        commits = revs.splitlines()
        return [c for c in commits if not self.is_approved(c)]

