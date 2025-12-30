"""Web tool actions for ToolEnvironmentBridge."""

from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from BrainSimulationSystem.environment.tool_bridge import ToolEnvironmentBridge  # noqa: E402


def test_web_tools_blocked_by_default(tmp_path: Path) -> None:
    bridge = ToolEnvironmentBridge(allowed_roots=[tmp_path])
    obs, reward, terminated, info = bridge.step({"type": "web_search", "query": "hello"})
    assert terminated is False
    assert info.get("blocked") is True
    assert info.get("reason") == "web_access_disabled"

    obs, reward, terminated, info = bridge.step({"type": "web_scrape", "url": "https://93.184.216.34/"})
    assert terminated is False
    assert info.get("blocked") is True
    assert info.get("reason") == "web_access_disabled"

    obs, reward, terminated, info = bridge.step({"type": "web_get", "url": "https://93.184.216.34/file.txt"})
    assert terminated is False
    assert info.get("blocked") is True
    assert info.get("reason") == "web_access_disabled"


def test_web_search_uses_stub_backend_when_real_disabled(tmp_path: Path) -> None:
    bridge = ToolEnvironmentBridge(
        allowed_roots=[tmp_path],
        allow_web_access=True,
        prefer_real_web_search=False,
    )
    obs, reward, terminated, info = bridge.step(
        {"type": "web_search", "query": "autonomous agents", "max_results": 2}
    )

    assert terminated is False
    assert info.get("backend") == "stub"
    assert isinstance(obs.get("results"), list)
    assert len(obs["results"]) == 2


def test_web_scrape_uses_scrape_backend_and_clips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_scrape_url(_self, url: str, **_kwargs):
        return {
            "url": url,
            "final_url": url,
            "status_code": 200,
            "content_type": "text/html",
            "title": "Example",
            "text": "X" * 200,
            "code_blocks": [],
        }

    monkeypatch.setattr(ToolEnvironmentBridge, "_scrape_url", fake_scrape_url)

    bridge = ToolEnvironmentBridge(
        allowed_roots=[tmp_path],
        allow_web_access=True,
        prefer_real_web_search=False,
        max_web_output_chars=32,
    )
    obs, reward, terminated, info = bridge.step(
        {"type": "web_scrape", "url": "https://93.184.216.34/", "max_chars": 500}
    )

    assert terminated is False
    assert reward > 0
    assert len(obs.get("text") or "") == 32
    assert info.get("truncated") is True
    assert info.get("host") == "93.184.216.34"


def test_web_get_uses_stubbed_requests_and_clips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import BrainSimulationSystem.environment.tool_bridge as tool_bridge_mod

    class StubResponse:
        status_code = 200
        url = "https://93.184.216.34/file.txt"
        headers = {"Content-Type": "text/plain; charset=utf-8"}
        encoding = "utf-8"
        content = b"Y" * 200

    stub_requests = types.SimpleNamespace(get=lambda *_a, **_k: StubResponse())
    monkeypatch.setattr(tool_bridge_mod, "requests", stub_requests)

    bridge = ToolEnvironmentBridge(
        allowed_roots=[tmp_path],
        allow_web_access=True,
        prefer_real_web_search=False,
        max_web_output_chars=32,
    )
    obs, reward, terminated, info = bridge.step({"type": "web_get", "url": "https://93.184.216.34/file.txt"})

    assert terminated is False
    assert reward > 0
    assert len(obs.get("text") or "") == 32
    assert info.get("truncated") is True
    assert info.get("host") == "93.184.216.34"


def test_github_code_search_uses_stubbed_requests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import BrainSimulationSystem.environment.tool_bridge as tool_bridge_mod

    class StubResponse:
        status_code = 200

        def json(self):
            return {
                "total_count": 1,
                "items": [
                    {
                        "path": "src/example.py",
                        "html_url": "https://github.com/org/repo/blob/main/src/example.py",
                        "repository": {"full_name": "org/repo"},
                        "text_matches": [{"fragment": "def hello():\\n  return 1"}],
                    }
                ],
            }

    stub_requests = types.SimpleNamespace(get=lambda *_a, **_k: StubResponse())
    monkeypatch.setattr(tool_bridge_mod, "requests", stub_requests)

    bridge = ToolEnvironmentBridge(
        allowed_roots=[tmp_path],
        allow_web_access=True,
        prefer_real_web_search=False,
    )
    obs, reward, terminated, info = bridge.step(
        {"type": "github_code_search", "query": "language:python hello", "max_results": 1}
    )

    assert terminated is False
    assert reward > 0
    assert "org/repo" in (obs.get("text") or "")
    assert info.get("returned") == 1


def test_documentation_tool_flags_low_consensus_for_conflicting_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_handle_web_search(_self, _action):
        return (
            {
                "text": "",
                "results": [
                    {
                        "title": "Doc A",
                        "url": "https://93.184.216.34/doc_a",
                        "trust": "high",
                        "trust_score": 1.0,
                    },
                    {
                        "title": "Doc B",
                        "url": "https://93.184.216.35/doc_b",
                        "trust": "high",
                        "trust_score": 1.0,
                    },
                ],
                "tool_state": _self._tool_state(),
            },
            0.12,
            False,
            {"backend": "stub", "returned": 2},
        )

    def fake_scrape_url(_self, url: str, **_kwargs):
        text = "alpha beta gamma" if "93.184.216.34" in url else "delta epsilon zeta"
        return {
            "url": url,
            "final_url": url,
            "status_code": 200,
            "content_type": "text/html",
            "title": "Example",
            "text": text,
            "code_blocks": [],
        }

    monkeypatch.setattr(ToolEnvironmentBridge, "_handle_web_search", fake_handle_web_search)
    monkeypatch.setattr(ToolEnvironmentBridge, "_scrape_url", fake_scrape_url)

    bridge = ToolEnvironmentBridge(allowed_roots=[tmp_path], allow_web_access=True, prefer_real_web_search=False)
    obs, reward, terminated, info = bridge.step(
        {"type": "documentation_tool", "query": "Example docs", "max_sources": 2}
    )

    assert terminated is False
    assert reward > 0

    consensus = info.get("consensus")
    assert isinstance(consensus, dict)
    assert consensus.get("unique_hosts") == 2
    assert consensus.get("level") == "low"
    assert consensus.get("needs_verification") is True
    assert float(consensus.get("similarity_avg") or 0.0) <= 0.01

    obs_consensus = obs.get("consensus")
    assert isinstance(obs_consensus, dict)
    assert obs_consensus.get("needs_verification") is True
    assert "consensus=low" in (obs.get("text") or "")
    assert "needs_verification" in (obs.get("text") or "")
