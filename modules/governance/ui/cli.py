"""CLI for reviewing and approving blueprint proposals."""
from __future__ import annotations

import click

from governance.queue import ProposalQueue

queue = ProposalQueue()


@click.group()
def cli() -> None:
    """Interact with the proposal queue."""
    pass


@cli.command("list")
def list_cmd() -> None:
    """List pending proposals."""
    proposals = queue.list_pending()
    if not proposals:
        click.echo("No pending proposals")
        return
    for path in proposals:
        click.echo(path.name)


@cli.command()
@click.argument("proposal")
def approve(proposal: str) -> None:
    """Approve a proposal file."""
    path = queue.proposal_dir / proposal
    result = queue.approve(path)
    click.echo(f"Approved {proposal} -> {result.name}")


@cli.command()
@click.argument("proposal")
def reject(proposal: str) -> None:
    """Reject and remove a proposal file."""
    path = queue.proposal_dir / proposal
    queue.reject(path)
    click.echo(f"Rejected {proposal}")
