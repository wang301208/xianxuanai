"""Parsing tool actions for ToolEnvironmentBridge (code + documents)."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from BrainSimulationSystem.environment.security_manager import SecurityManager  # noqa: E402
from BrainSimulationSystem.environment.tool_bridge import ToolEnvironmentBridge  # noqa: E402


def test_parse_code_python_file(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    target = repo_root / "sample.py"
    target.write_text(
        "import math\n"
        "\n"
        "def add(a: int, b: int = 1) -> int:\n"
        "    \"\"\"Add two integers.\"\"\"\n"
        "    return a + b\n"
        "\n"
        "class Greeter:\n"
        "    \"\"\"Simple class.\"\"\"\n"
        "    def hello(self, name: str) -> str:\n"
        "        return f\"hi {name}\"\n",
        encoding="utf-8",
    )

    bridge = ToolEnvironmentBridge(allowed_roots=[tmp_path])
    obs, reward, terminated, info = bridge.step({"type": "parse_code", "path": str(target)})

    assert terminated is False
    assert reward > 0
    assert info.get("language") == "python"
    assert info.get("functions") == 1
    assert info.get("classes") == 1

    parsed = obs.get("parsed")
    assert isinstance(parsed, dict)
    module = parsed.get("module")
    assert isinstance(module, dict)
    functions = module.get("functions")
    classes = module.get("classes")
    assert isinstance(functions, list)
    assert isinstance(classes, list)
    assert any(isinstance(fn, dict) and fn.get("name") == "add" for fn in functions)
    assert any(isinstance(cl, dict) and cl.get("name") == "Greeter" for cl in classes)


def test_summarize_doc_markdown_extracts_headings_and_formulas(tmp_path: Path) -> None:
    doc = tmp_path / "note.md"
    doc.write_text(
        "# Dijkstra\n\n"
        "Shortest path algorithm.\n\n"
        "## Complexity\n\n"
        "Time: $O(E \\log V)$\n"
        "a=b\n",
        encoding="utf-8",
    )

    bridge = ToolEnvironmentBridge(allowed_roots=[tmp_path])
    obs, reward, terminated, info = bridge.step({"type": "summarize_doc", "path": str(doc)})

    assert terminated is False
    assert reward > 0
    summary = obs.get("summary")
    assert isinstance(summary, dict)
    headings = summary.get("headings")
    assert isinstance(headings, list)
    assert any(str(h).endswith("Dijkstra") for h in headings)

    formulas = summary.get("formulas")
    assert isinstance(formulas, list)
    assert any("O(E" in str(f) or "a=b" == str(f) for f in formulas)


def test_security_manager_allows_parse_actions_in_read_only_mode(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    target.write_text("def foo():\n    return 1\n", encoding="utf-8")

    security = SecurityManager({"enabled": True, "permission_level": "read"})
    bridge = ToolEnvironmentBridge(allowed_roots=[tmp_path], security_manager=security)

    obs, reward, terminated, info = bridge.step({"type": "parse_code", "path": str(target)})
    assert terminated is False
    assert reward > 0
    assert info.get("blocked") is not True

    obs, reward, terminated, info = bridge.step({"type": "summarize_doc", "path": str(target)})
    assert terminated is False
    assert reward > 0
    assert info.get("blocked") is not True

