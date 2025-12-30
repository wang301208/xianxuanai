from __future__ import annotations

import json

from modules.knowledge.problem_analyzer import ProblemAnalyzer
from modules.knowledge.research_tool import ResearchTool


def test_research_tool_search_web_uses_stub() -> None:
    tool = ResearchTool()
    hits = tool.search_web("autonomous agents", max_results=2)
    assert len(hits) == 2
    assert hits[0].url.startswith("https://")
    assert "Stub result" in hits[0].title


def test_research_tool_query_docs_finds_matches(tmp_path) -> None:
    (tmp_path / "README.md").write_text("Hello world\nSome details.\n", encoding="utf-8")
    tool = ResearchTool(workspace_root=tmp_path, docs_roots=[tmp_path], max_scan_files=20)
    hits = tool.query_docs("hello", max_results=3)
    assert hits
    assert "README.md" in hits[0].path


def test_problem_analyzer_parses_json_list() -> None:
    def llm_stub(_: str) -> str:
        return "```json\n[\"q1\", \"q2\", \"q3\"]\n```"

    analyzer = ProblemAnalyzer(llm=llm_stub)
    questions = analyzer.analyze_problem("goal", max_subquestions=2)
    assert questions == ["q1", "q2"]


def test_problem_analyzer_falls_back_without_llm() -> None:
    analyzer = ProblemAnalyzer()
    questions = analyzer.analyze_problem("build a foo", max_subquestions=2)
    assert len(questions) == 2
    assert "build a foo" in questions[0].lower()


def test_problem_analyzer_analyze_and_solve(tmp_path) -> None:
    (tmp_path / "notes.md").write_text("q1: local hint here\n", encoding="utf-8")

    def llm_stub(prompt: str) -> str:
        if "Return a JSON list of short" in prompt:
            return "```json\n[\"q1\", \"q2\"]\n```"
        return (
            "```json\n"
            "{\n"
            '  \"summary\": \"ok\",\n'
            '  \"steps\": [\"step1\", \"step2\"],\n'
            '  \"risks\": [],\n'
            '  \"open_questions\": []\n'
            "}\n"
            "```"
        )

    analyzer = ProblemAnalyzer(llm=llm_stub)
    tool = ResearchTool(workspace_root=tmp_path, docs_roots=[tmp_path], max_scan_files=50)
    result = analyzer.analyze_and_solve(
        "some goal",
        research_tool=tool,
        max_subquestions=2,
        max_results_per_query=1,
    )

    assert result["sub_questions"] == ["q1", "q2"]
    assert len(result["evidence"]) == 2
    plan = json.loads(result["plan"])
    assert plan["steps"] == ["step1", "step2"]

