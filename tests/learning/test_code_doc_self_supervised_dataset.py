from __future__ import annotations

from pathlib import Path

from modules.learning.code_doc_self_supervised import CodeDocDatasetConfig, build_self_supervised_examples


def test_build_self_supervised_examples_extracts_code_and_markdown(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    (repo / "algo.py").write_text(
        "def dijkstra(graph, start):\n"
        "    \"\"\"Compute shortest paths from start using Dijkstra.\"\"\"\n"
        "    return {}\n"
        "\n"
        "class Solver:\n"
        "    def bfs(self, graph, start):\n"
        "        \"\"\"Breadth-first search traversal.\"\"\"\n"
        "        return []\n",
        encoding="utf-8",
    )

    (repo / "README.md").write_text(
        "# Algorithms\n\n"
        "## Dijkstra\n\n"
        "```python\n"
        "dist = dijkstra(g, 0)\n"
        "```\n",
        encoding="utf-8",
    )

    cfg = CodeDocDatasetConfig(
        max_files=20,
        max_examples=200,
        max_chars_per_file=50_000,
        tasks=("python_function_name", "python_docstring", "markdown_code_to_heading", "code_doc_pair"),
    )
    result = build_self_supervised_examples([repo], config=cfg)

    stats = result.get("stats") or {}
    assert stats.get("files_scanned") >= 2

    examples = result.get("examples")
    assert isinstance(examples, list)
    tasks = {ex.get("task") for ex in examples if isinstance(ex, dict)}
    assert "python_function_name" in tasks
    assert "python_docstring" in tasks
    assert "markdown_code_to_heading" in tasks
    assert "code_doc_pair" in tasks

    # Sanity check: expected symbols appear in at least one example.
    combined = "\n".join(str(ex.get("input") or "") + "\n" + str(ex.get("output") or "") for ex in examples if isinstance(ex, dict))
    assert "dijkstra" in combined
    assert "Dijkstra" in combined

