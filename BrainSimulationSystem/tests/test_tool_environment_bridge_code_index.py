"""Local code index actions for ToolEnvironmentBridge."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from BrainSimulationSystem.environment.tool_bridge import ToolEnvironmentBridge  # noqa: E402


def test_code_index_build_and_search(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "algo.py").write_text(
        "def foo(x: int) -> int:\n"
        "    return x + 1\n",
        encoding="utf-8",
    )

    bridge = ToolEnvironmentBridge(allowed_roots=[tmp_path])

    obs, reward, terminated, info = bridge.step({"type": "code_index_build", "root": str(repo_root), "max_files": 10})
    assert terminated is False
    assert reward > 0
    assert info.get("root") == str(repo_root.resolve())

    obs, reward, terminated, info = bridge.step(
        {"type": "code_index_search", "root": str(repo_root), "query": "def foo return x", "top_k": 3}
    )
    assert terminated is False
    assert reward > 0
    hits = obs.get("hits")
    assert isinstance(hits, list)
    assert hits and hits[0].get("path") == "algo.py"
    assert hits[0].get("symbol") == "foo"

    refs = obs.get("references")
    assert isinstance(refs, list)
    assert refs and refs[0].get("url") == "algo.py"


def test_code_index_reads_filesystem_sandbox_overlay(tmp_path: Path) -> None:
    sandbox_root = tmp_path / ".sandbox"
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    target = repo_root / "algo.py"
    target.write_text(
        "def foo() -> int:\n"
        "    return 1\n",
        encoding="utf-8",
    )

    bridge = ToolEnvironmentBridge(
        allowed_roots=[tmp_path],
        allow_file_write=True,
        filesystem_sandbox={"enabled": True, "root": str(sandbox_root), "keep_history": True},
    )

    token = "SANDBOX_UNIQUE_123"
    obs, reward, terminated, info = bridge.step(
        {
            "type": "write_file",
            "path": str(target),
            "text": "def foo() -> int:\n" f"    \"\"\"{token}\"\"\"\n" "    return 1\n",
        }
    )
    assert info.get("sandboxed") is True

    bridge.step({"type": "code_index_build", "root": str(repo_root), "max_files": 10})
    obs, reward, terminated, info = bridge.step(
        {"type": "code_index_search", "root": str(repo_root), "query": token, "top_k": 3, "max_chars_per_hit": 5000}
    )

    hits = obs.get("hits")
    assert isinstance(hits, list)
    assert hits
    assert token in (hits[0].get("text") or "")

