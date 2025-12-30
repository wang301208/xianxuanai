"""GitHub repo ingest action for ToolEnvironmentBridge."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys
import zipfile

import pytest


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from BrainSimulationSystem.environment.tool_bridge import ToolEnvironmentBridge  # noqa: E402


def _zip_bytes(files: dict[str, str]) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


def test_repo_ingest_blocked_without_file_write(tmp_path: Path) -> None:
    bridge = ToolEnvironmentBridge(allowed_roots=[tmp_path], allow_web_access=True, allow_file_write=False)
    obs, reward, terminated, info = bridge.step({"type": "github_repo_ingest", "repo": "org/repo"})
    assert terminated is False
    assert info.get("blocked") is True
    assert info.get("reason") == "file_write_disabled"


def test_repo_ingest_extracts_and_builds_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import modules.knowledge.github_repo_ingest as ingest_mod

    payload = _zip_bytes(
        {
            "repo-main/adder.py": "def add(a: int, b: int) -> int:\n    return a + b\n",
            "repo-main/README.md": "# Demo\n",
        }
    )

    def fake_download(url: str, max_bytes: int, timeout_s: float, headers: dict[str, str]) -> bytes:  # noqa: ARG001
        assert "codeload.github.com" in url
        return payload

    monkeypatch.setattr(ingest_mod, "_default_download", fake_download)
    monkeypatch.setattr(ToolEnvironmentBridge, "_is_public_hostname", lambda _self, _host: True)

    bridge = ToolEnvironmentBridge(
        allowed_roots=[tmp_path],
        allow_web_access=True,
        allow_file_write=True,
        prefer_real_web_search=False,
    )

    dest_root = tmp_path / "external"
    obs, reward, terminated, info = bridge.step(
        {
            "type": "github_repo_ingest",
            "repo": "org/repo",
            "ref": "main",
            "dest_root": str(dest_root),
            "build_index": True,
            "index_max_files": 25,
            "index_include_suffixes": [".py"],
        }
    )

    assert terminated is False
    assert reward > 0
    assert info.get("repo") == "org/repo"
    repo_root = obs.get("repo_root")
    assert isinstance(repo_root, str) and repo_root

    obs2, reward2, terminated2, info2 = bridge.step(
        {"type": "code_index_search", "root": repo_root, "query": "def add return a + b", "top_k": 3}
    )
    assert terminated2 is False
    assert reward2 > 0
    hits = obs2.get("hits")
    assert isinstance(hits, list)
    assert hits
    assert hits[0].get("path") == "adder.py"
    assert hits[0].get("symbol") == "add"


def test_repo_ingest_detects_license_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import modules.knowledge.github_repo_ingest as ingest_mod

    payload = _zip_bytes(
        {
            "repo-main/LICENSE": "GNU GENERAL PUBLIC LICENSE\nVersion 3, 29 June 2007\n",
            "repo-main/adder.py": "def add(a: int, b: int) -> int:\n    return a + b\n",
        }
    )

    def fake_download(url: str, max_bytes: int, timeout_s: float, headers: dict[str, str]) -> bytes:  # noqa: ARG001
        assert "codeload.github.com" in url
        return payload

    monkeypatch.setattr(ingest_mod, "_default_download", fake_download)
    monkeypatch.setattr(ToolEnvironmentBridge, "_is_public_hostname", lambda _self, _host: True)

    bridge = ToolEnvironmentBridge(
        allowed_roots=[tmp_path],
        allow_web_access=True,
        allow_file_write=True,
        prefer_real_web_search=False,
    )

    dest_root = tmp_path / "external"
    obs, reward, terminated, info = bridge.step(
        {
            "type": "github_repo_ingest",
            "repo": "org/repo",
            "ref": "main",
            "dest_root": str(dest_root),
            "build_index": False,
        }
    )

    assert terminated is False
    assert reward > 0
    lic = info.get("license")
    assert isinstance(lic, dict)
    assert lic.get("spdx") == "GPL-3.0"
    assert lic.get("copyleft") is True
    assert info.get("license_allowed") is True

    lic2 = obs.get("license")
    assert isinstance(lic2, dict)
    assert lic2.get("spdx") == "GPL-3.0"


def test_repo_ingest_blocks_disallowed_license_when_enforced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import modules.knowledge.github_repo_ingest as ingest_mod

    payload = _zip_bytes(
        {
            "repo-main/LICENSE": "GNU GENERAL PUBLIC LICENSE\nVersion 3, 29 June 2007\n",
            "repo-main/adder.py": "def add(a: int, b: int) -> int:\n    return a + b\n",
        }
    )

    def fake_download(url: str, max_bytes: int, timeout_s: float, headers: dict[str, str]) -> bytes:  # noqa: ARG001
        assert "codeload.github.com" in url
        return payload

    monkeypatch.setattr(ingest_mod, "_default_download", fake_download)
    monkeypatch.setattr(ToolEnvironmentBridge, "_is_public_hostname", lambda _self, _host: True)

    bridge = ToolEnvironmentBridge(
        allowed_roots=[tmp_path],
        allow_web_access=True,
        allow_file_write=True,
        prefer_real_web_search=False,
    )

    dest_root = tmp_path / "external"
    obs, reward, terminated, info = bridge.step(
        {
            "type": "github_repo_ingest",
            "repo": "org/repo",
            "ref": "main",
            "dest_root": str(dest_root),
            "build_index": False,
            "enforce_license": True,
            "license_denylist": ["GPL-3.0"],
        }
    )

    assert terminated is False
    assert info.get("blocked") is True
    assert info.get("reason") == "repo_license_not_allowed"
    lic = info.get("license")
    assert isinstance(lic, dict)
    assert lic.get("spdx") == "GPL-3.0"
    cleanup = info.get("cleanup")
    assert isinstance(cleanup, dict)
    assert cleanup.get("attempted") is True
