from __future__ import annotations

from backend.introspection import explain_my_plan, get_loaded_skills, summarize_my_abilities
from backend.introspection import bootstrap_self_knowledge, query_self_structure
from backend.knowledge.unified import UnifiedKnowledgeBase
from backend.autogpt.autogpt.core.knowledge_graph.graph_store import GraphStore
from backend.self_model import SelfModel
from modules.skills import SkillRegistry, SkillSpec


def test_get_loaded_skills_reports_registered_specs() -> None:
    registry = SkillRegistry(graph_store=object(), auto_graph_updates=False)
    registry.register(SkillSpec(name="search_docs", description="Search documentation", tags=["docs"]))
    registry.register(
        SkillSpec(name="disabled_skill", description="Should not show", enabled=False),
        replace=True,
    )

    payload = get_loaded_skills(registry, include_disabled=False, as_text=False)
    assert payload["total"] == 1
    assert payload["skills"][0]["name"] == "search_docs"

    payload_all = get_loaded_skills(registry, include_disabled=True, as_text=False)
    names = {s["name"] for s in payload_all["skills"]}
    assert {"search_docs", "disabled_skill"} <= names

    text = get_loaded_skills(registry, include_disabled=False, as_text=True)
    assert "Loaded skills:" in text
    assert "search_docs" in text


def test_summarize_my_abilities_uses_self_model_capabilities() -> None:
    model = SelfModel()
    model.set_capability("planning", 0.9)
    model.set_capability("debugging", 0.4)

    payload = summarize_my_abilities(model, as_text=False)
    assert payload["returned"] == 2
    assert payload["abilities"][0]["name"] == "planning"
    assert payload["abilities"][0]["level"] == "high"
    assert payload["abilities"][1]["name"] == "debugging"
    assert payload["abilities"][1]["level"] == "low"

    text = summarize_my_abilities(model, as_text=True)
    assert "Abilities:" in text
    assert "planning" in text


def test_explain_my_plan_includes_plan_skills_and_safety() -> None:
    registry = SkillRegistry(graph_store=object(), auto_graph_updates=False)
    registry.register(SkillSpec(name="search_docs", description="Search documentation", tags=["docs"]))
    registry.register(SkillSpec(name="run_tests", description="Run pytest suite", tags=["testing"]))

    model = SelfModel()
    model.set_capability("planning", 0.8)

    payload = explain_my_plan(
        "Please search docs for the API and propose a safe plan.",
        registry=registry,
        self_model=model,
        as_text=False,
    )
    assert payload["plan"]
    assert payload["skills"]["suggested"]
    assert payload["skills"]["suggested"][0]["name"] == "search_docs"
    assert payload["safety_boundaries"]
    assert payload["abilities"]["returned"] == 1

    text = explain_my_plan(
        "Please search docs for the API and propose a safe plan.",
        registry=registry,
        self_model=model,
        as_text=True,
    )
    assert "Plan:" in text
    assert "Safety boundaries:" in text


def test_explain_my_plan_action_payload_includes_expectation_fields() -> None:
    registry = SkillRegistry(graph_store=object(), auto_graph_updates=False)
    registry.register(SkillSpec(name="read_file", description="Read file content", tags=["fs", "read"]))

    payload = explain_my_plan(
        {
            "goal": "inspect configuration",
            "step_title": "read config",
            "action": {"type": "read_file", "path": "/tmp/config.yaml", "max_chars": 50},
        },
        registry=registry,
        as_text=False,
    )
    assert payload["expected"]
    assert payload["verification"]
    assert payload["rationale"]


def test_bootstrap_self_knowledge_and_query_self_structure(tmp_path) -> None:
    repo_root = tmp_path
    demo_module = repo_root / "BrainSimulationSystem" / "environment" / "demo_component.py"
    demo_module.parent.mkdir(parents=True, exist_ok=True)
    demo_module.write_text('"""Demo component.\n\nProvides a tiny capability.\n"""\n', encoding="utf-8")

    graph = GraphStore()
    registry = SkillRegistry(graph_store=graph, auto_graph_updates=False)
    registry.register(SkillSpec(name="search_docs", description="Search documentation", tags=["docs"]))

    kb = UnifiedKnowledgeBase()
    result = bootstrap_self_knowledge(
        repo_root=repo_root,
        knowledge_base=kb,
        registry=registry,
        graph_store=graph,
        component_paths=[demo_module],
        config_paths=[],
    )
    assert result["facts_built"] > 0
    assert result["facts_ingested"] > 0

    query = query_self_structure(
        "demo_component",
        knowledge_base=kb,
        graph_store=graph,
        top_k=5,
        as_text=False,
    )
    assert query["returned"] > 0
    hit = query["results"][0]
    assert "demo" in (hit.get("text") or hit.get("context") or "").lower() or "demo" in str(hit)
