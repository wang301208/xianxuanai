import subprocess

from governance import HumanArchitect, ApprovalService


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _git_output(cwd, *args) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def test_approve_and_query_commits(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    # initialize repository
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")

    # first commit
    file_path = repo / "file.txt"
    file_path.write_text("one")
    _git(repo, "add", "file.txt")
    _git(repo, "commit", "-m", "initial")

    # second commit
    file_path.write_text("two")
    _git(repo, "add", "file.txt")
    _git(repo, "commit", "-m", "update")
    commit_hash = _git_output(repo, "rev-parse", "HEAD")

    agent = HumanArchitect("Alice", ApprovalService(repo))

    pending = agent.pending_commits(base="HEAD~1")
    assert commit_hash in pending

    agent.approve_commit(commit_hash)

    assert commit_hash not in agent.pending_commits(base="HEAD~1")
    assert "Alice" in agent.approval_service.approvals_for(commit_hash)
