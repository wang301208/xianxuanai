"""Agent responsible for recording commit approvals.

The :class:`HumanArchitect` coordinates with humans to approve or review
commits. It delegates actual git operations to :class:`ApprovalService`.
"""

from __future__ import annotations

from abc import ABC

from .approvals import ApprovalService


class HumanArchitect(ABC):
    """Agent that records commit approvals using :class:`ApprovalService`."""

    def __init__(self, name: str, approval_service: ApprovalService | None = None) -> None:
        """Create a new ``HumanArchitect``.

        Parameters
        ----------
        name:
            Name of the human architect recording approvals.
        approval_service:
            Optional :class:`ApprovalService` instance. If not provided a
            default instance operating on the current repository is used.
        """

        self.name = name
        self.approval_service = approval_service or ApprovalService()

    # ------------------------------------------------------------------
    # approval workflow
    # ------------------------------------------------------------------
    def approve_commit(self, commit_hash: str) -> None:
        """Record this architect's approval of ``commit_hash``."""
        self.approval_service.add_approval(commit_hash, self.name)

    def pending_commits(self, base: str = "origin/main", head: str = "HEAD") -> list[str]:
        """Return commits between ``base`` and ``head`` lacking approval."""
        return self.approval_service.pending_commits(base=base, head=head)

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------
    def perform(self, base: str = "origin/main", head: str = "HEAD") -> str:
        """Report commits pending approval."""
        commits = self.pending_commits(base=base, head=head)
        if commits:
            return "\n".join(commits)
        return "No pending commits."
