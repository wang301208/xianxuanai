"""Autonomous natural-language task planning + execution loop."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from BrainSimulationSystem.environment.autonomous_task_loop import (  # noqa: E402
    ActionGovernor,
    AutonomousTaskExecutor,
    HeuristicTaskPlanner,
    LLMTaskPlanner,
)
from BrainSimulationSystem.environment.tool_bridge import ToolEnvironmentBridge  # noqa: E402
from modules.evolution.agent_self_improvement import AgentSelfImprovementController  # noqa: E402
from modules.knowledge.knowledge_consolidation import ExternalKnowledgeConsolidator  # noqa: E402
from modules.knowledge.long_term_memory import LongTermMemoryCoordinator  # noqa: E402
from modules.monitoring.collector import RealTimeMetricsCollector  # noqa: E402


def test_autonomous_task_executor_runs_explicit_json_plan(tmp_path):
    bridge = ToolEnvironmentBridge(allowed_roots=[tmp_path], allow_file_write=True)
    executor = AutonomousTaskExecutor(bridge, planner=HeuristicTaskPlanner())

    target_dir = tmp_path / "demo"
    target_file = target_dir / "note.txt"
    goal = f"""
    Please execute this plan:
    ```json
    {{
      "goal": "write a note",
      "steps": [
        {{"title": "mkdir", "action": {{"type": "create_dir", "path": "{target_dir}", "parents": true, "exist_ok": true}}}},
        {{"title": "write", "action": {{"type": "write_file", "path": "{target_file}", "text": "hello"}}}},
        {{"title": "read", "action": {{"type": "read_file", "path": "{target_file}", "max_chars": 100}}}}
      ]
    }}
    ```
    """

    report = executor.run(goal)
    assert report.success is True
    assert report.blocked is False
    assert len(report.events) == 3
    assert target_file.read_text(encoding="utf-8") == "hello"
    assert "hello" in (report.events[-1].observation.get("text") or "")


def test_governor_blocks_delete_without_confirmation(tmp_path):
    target = tmp_path / "deleteme.txt"
    target.write_text("x", encoding="utf-8")

    bridge = ToolEnvironmentBridge(allowed_roots=[tmp_path], allow_file_delete=True)
    governor = ActionGovernor({"confirm_token": "OK"})
    executor = AutonomousTaskExecutor(bridge, governor=governor)

    goal = f"""
    ```json
    {{
      "goal": "cleanup",
      "steps": [
        {{"title": "delete", "action": {{"type": "delete_file", "path": "{target}"}}}}
      ]
    }}
    ```
    """

    report = executor.run(goal)
    assert report.success is False
    assert report.blocked is True
    assert report.events[-1].status == "blocked"
    assert target.exists()


def test_governor_allows_delete_with_confirmation(tmp_path):
    target = tmp_path / "deleteme.txt"
    target.write_text("x", encoding="utf-8")

    bridge = ToolEnvironmentBridge(allowed_roots=[tmp_path], allow_file_delete=True)
    governor = ActionGovernor({"confirm_token": "OK"})
    executor = AutonomousTaskExecutor(bridge, governor=governor)

    goal = f"""
    ```json
    {{
      "goal": "cleanup",
      "steps": [
        {{"title": "delete", "action": {{"type": "delete_file", "path": "{target}", "confirm_token": "OK"}}}}
      ]
    }}
    ```
    """

    report = executor.run(goal)
    assert report.success is True
    assert target.exists() is False


def test_heuristic_planner_write_file_pattern(tmp_path):
    bridge = ToolEnvironmentBridge(allowed_roots=[tmp_path], allow_file_write=True)
    executor = AutonomousTaskExecutor(bridge, planner=HeuristicTaskPlanner())

    target = tmp_path / "hello.txt"
    report = executor.run(f"write file {target} with hello world")
    assert report.success is True
    assert target.read_text(encoding="utf-8") == "hello world"


def test_executor_injects_introspection_metadata(tmp_path):
    bridge = ToolEnvironmentBridge(allowed_roots=[tmp_path], allow_file_write=True)
    executor = AutonomousTaskExecutor(bridge, planner=HeuristicTaskPlanner())

    target = tmp_path / "hello.txt"
    report = executor.run(f"write file {target} with hello world")

    introspection = report.meta.get("introspection")
    assert isinstance(introspection, dict)
    assert "abilities" in introspection
    assert introspection["abilities"]["returned"] > 0
    assert "tool_bridge" in introspection
    assert introspection["tool_bridge"]["enabled_actions"]["write_file"] is True

    assert report.events
    assert "introspection" in report.events[0].info
    assert report.events[0].info["introspection"]["expected"]


def test_executor_self_improvement_updates_prompt_genes_on_blocked_step(tmp_path):
    # Shell execution is disabled by default -> tool bridge returns a blocked step.
    bridge = ToolEnvironmentBridge(allowed_roots=[tmp_path])
    improver = AgentSelfImprovementController(enabled=True, ga_seed=0)
    executor = AutonomousTaskExecutor(
        bridge,
        planner=HeuristicTaskPlanner(),
        self_improvement=improver,
        max_replans=0,
    )

    goal = """
    ```json
    {
      "goal": "attempt a shell command",
      "steps": [
        {"title": "shell", "action": {"type": "shell", "command": ["echo", "hi"], "timeout_s": 1}}
      ]
    }
    ```
    """
    report = executor.run(goal)
    assert report.success is False
    assert report.blocked is True
    assert report.events
    assert report.events[0].status == "blocked"
    assert report.events[0].latency_s >= 0.0
    assert report.events[0].info.get("latency_s") is not None

    meta = report.meta.get("self_improvement")
    assert isinstance(meta, dict)
    assert meta.get("last_update") is not None
    assert meta.get("strategy") is not None
    assert meta["strategy"]["prompt"]["variant"] == 2
    assert meta["strategy"]["prompt"]["safety_bias"] >= 0.8


def test_llm_task_planner_prompt_variant_changes_system_prompt():
    planner = LLMTaskPlanner(llm=None)
    prompt = planner._build_system_prompt(
        context={
            "strategy": {
                "prompt": {"variant": 2, "json_strictness": 0.9, "safety_bias": 0.9},
                "planner": {"structured": True},
            }
        }
    )
    assert "Hard constraints" in prompt
    assert "Output ONLY a valid JSON object" in prompt


def test_executor_emits_metric_events_when_collector_attached(tmp_path):
    bridge = ToolEnvironmentBridge(allowed_roots=[tmp_path], allow_file_write=True)
    collector = RealTimeMetricsCollector(monitor=None)
    executor = AutonomousTaskExecutor(
        bridge,
        planner=HeuristicTaskPlanner(),
        metrics_collector=collector,
    )

    target = tmp_path / "hello.txt"
    report = executor.run(f"write file {target} with hello world")
    assert report.success is True

    events = collector.events()
    assert events
    assert any(event.module == "write_file" for event in events)
    assert any(isinstance(event.metadata, dict) and "reward" in event.metadata for event in events)


def test_executor_can_acquire_local_code_references_on_failure(tmp_path):
    token = "ALGO_TOKEN_123"
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "algo.py").write_text(
        "def foo() -> int:\n"
        f"    \"\"\"{token}\"\"\"\n"
        "    return 1\n",
        encoding="utf-8",
    )

    bridge = ToolEnvironmentBridge(allowed_roots=[tmp_path])
    executor = AutonomousTaskExecutor(bridge, planner=HeuristicTaskPlanner(), max_replans=1)

    goal = f"""
    ```json
    {{
      "goal": "use {token} algorithm",
      "steps": [
        {{"title": "fail", "action": {{"type": "unknown_action_type"}}}}
      ]
    }}
    ```
    """

    report = executor.run(
        goal,
        context={
            "knowledge_acquisition": {
                "enabled": True,
                "code_root": str(repo_root),
                "max_files": 50,
                "top_k": 5,
            }
        },
    )

    assert report.success is False
    ka = report.meta.get("knowledge_acquisition")
    assert isinstance(ka, list) and ka
    payload = ka[0]
    assert payload.get("returned", 0) >= 1

    refs = payload.get("references")
    assert isinstance(refs, list)
    assert any(isinstance(ref, dict) and ref.get("url") == "algo.py" for ref in refs)


def test_executor_prefers_long_term_memory_before_web_search(tmp_path):
    import json

    token = "ALGO_TOKEN_LTM_SKIP_001"
    ltm_root = tmp_path / "ltm"
    vector_store = ltm_root / "vector_store"
    vector_store.mkdir(parents=True)
    (vector_store / "metadata.json").write_text(
        json.dumps(
            {
                "backend": "brute",
                "dimension": 8,
                "records": [
                    {"id": "rec-1", "text": f"{token} notes", "metadata": {"goal": f"Learn {token}"}},
                ],
            }
        ),
        encoding="utf-8",
    )

    bridge = ToolEnvironmentBridge(allowed_roots=[tmp_path], allow_web_access=True, prefer_real_web_search=False)
    executor = AutonomousTaskExecutor(bridge, planner=HeuristicTaskPlanner(), max_replans=1)

    goal = f"""
    ```json
    {{
      "goal": "Explain {token} algorithm",
      "steps": [
        {{"title": "fail", "action": {{"type": "unknown_action_type"}}}}
      ]
    }}
    ```
    """

    report = executor.run(
        goal,
        context={
            "knowledge_acquisition": {
                "enabled": True,
                "query": token,
                "web_search": True,
                "use_code_index": False,
                "use_long_term_memory": True,
                "long_term_memory_root": str(ltm_root),
                "web_if_internal_insufficient": True,
            }
        },
    )

    assert report.success is False
    ka = report.meta.get("knowledge_acquisition")
    assert isinstance(ka, list) and ka
    payload = ka[0]

    assert payload.get("web_skipped") is True
    assert payload.get("web_skip_reason") == "internal_memory_or_knowledge_graph_hit"
    assert "long_term_memory" in (payload.get("channels_used") or [])
    assert "web_search" not in (payload.get("channels_used") or [])
    assert payload.get("web_search") is None
    assert payload.get("returned_memory_hits") == 1

    ltm = payload.get("long_term_memory")
    assert isinstance(ltm, dict)
    assert ltm.get("returned") == 1

    refs = payload.get("references")
    assert isinstance(refs, list)
    assert any(isinstance(ref, dict) and ref.get("url") == "memory:rec-1" for ref in refs)


def test_executor_does_not_skip_web_when_only_unverified_memory_hits(tmp_path):
    import json

    token = "ALGO_TOKEN_LTM_NEEDS_VERIFY_002"
    ltm_root = tmp_path / "ltm"
    vector_store = ltm_root / "vector_store"
    vector_store.mkdir(parents=True)
    (vector_store / "metadata.json").write_text(
        json.dumps(
            {
                "backend": "brute",
                "dimension": 8,
                "records": [
                    {
                        "id": "rec-1",
                        "text": f"{token} notes",
                        "metadata": {"goal": f"Learn {token}", "needs_verification": True},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    bridge = ToolEnvironmentBridge(allowed_roots=[tmp_path], allow_web_access=True, prefer_real_web_search=False)
    executor = AutonomousTaskExecutor(bridge, planner=HeuristicTaskPlanner(), max_replans=1)

    goal = f"""
    ```json
    {{
      "goal": "Explain {token} algorithm",
      "steps": [
        {{"title": "fail", "action": {{"type": "unknown_action_type"}}}}
      ]
    }}
    ```
    """

    report = executor.run(
        goal,
        context={
            "knowledge_acquisition": {
                "enabled": True,
                "query": token,
                "web_search": True,
                "use_code_index": False,
                "use_knowledge_graph": False,
                "use_long_term_memory": True,
                "long_term_memory_root": str(ltm_root),
                "web_if_internal_insufficient": True,
            }
        },
    )

    assert report.success is False
    ka = report.meta.get("knowledge_acquisition")
    assert isinstance(ka, list) and ka
    payload = ka[0]

    assert payload.get("returned_memory_hits") == 1
    assert payload.get("web_skipped") is False
    assert "long_term_memory" in (payload.get("channels_used") or [])
    assert "web_search" in (payload.get("channels_used") or [])
    assert isinstance(payload.get("web_search"), dict)
    assert int(payload.get("returned_web_results") or 0) > 0


def test_source_monitor_quarantines_web_access_on_low_trust_sources(tmp_path, monkeypatch):
    monkeypatch.setenv("BSS_SOURCE_MONITOR_ENABLED", "1")
    monkeypatch.setenv("BSS_SOURCE_MONITOR_MIN_REFS", "1")
    monkeypatch.setenv("BSS_SOURCE_MONITOR_LOW_TRUST_RATIO", "0.5")
    monkeypatch.setenv("BSS_SOURCE_MONITOR_COOLDOWN_S", "60")
    monkeypatch.setenv("BSS_BLOCKED_WEB_DOMAINS", "example.com")

    bridge = ToolEnvironmentBridge(allowed_roots=[tmp_path], allow_web_access=True, prefer_real_web_search=False)
    executor = AutonomousTaskExecutor(bridge, planner=HeuristicTaskPlanner(), max_replans=1)

    goal = """
    ```json
    {
      "goal": "Trigger knowledge acquisition",
      "steps": [
        {"title": "fail", "action": {"type": "unknown_action_type"}}
      ]
    }
    ```
    """

    report = executor.run(
        goal,
        context={
            "knowledge_acquisition": {
                "enabled": True,
                "query": "example",
                "web_search": True,
                "use_code_index": False,
                "use_long_term_memory": False,
                "use_knowledge_graph": False,
            }
        },
    )

    assert report.success is False
    alerts = report.meta.get("alerts")
    assert isinstance(alerts, list)
    assert any(isinstance(item, dict) and item.get("type") == "source_monitor_quarantine" for item in alerts)

    obs, reward, terminated, info = bridge.step({"type": "web_search", "query": "hello", "max_results": 1})
    assert terminated is False
    assert reward < 0
    assert info.get("blocked") is True
    assert info.get("reason") == "web_access_disabled"


def test_executor_meta_retrieval_policy_adds_web_search_payload(tmp_path, monkeypatch):
    state_path = tmp_path / "meta_retrieval_policy.json"
    monkeypatch.setenv("BSS_META_RETRIEVAL_ENABLED", "1")
    monkeypatch.setenv("BSS_META_RETRIEVAL_STATE_PATH", str(state_path))

    token = "ALGO_TOKEN_META_456"
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "algo.py").write_text(
        "def foo() -> int:\n"
        f"    \"\"\"{token}\"\"\"\n"
        "    return 1\n",
        encoding="utf-8",
    )

    bridge = ToolEnvironmentBridge(allowed_roots=[tmp_path], allow_web_access=True, prefer_real_web_search=False)
    executor = AutonomousTaskExecutor(bridge, planner=HeuristicTaskPlanner(), max_replans=1)

    goal = f"""
    ```json
    {{
      "goal": "Explain {token} algorithm",
      "steps": [
        {{"title": "fail", "action": {{"type": "unknown_action_type"}}}}
      ]
    }}
    ```
    """

    report = executor.run(
        goal,
        context={
            "knowledge_acquisition": {
                "enabled": True,
                "code_root": str(repo_root),
                "max_files": 50,
                "top_k": 5,
                "meta_retrieval_enabled": True,
            }
        },
    )

    assert report.success is False
    ka = report.meta.get("knowledge_acquisition")
    assert isinstance(ka, list) and ka
    payload = ka[0]

    assert isinstance(payload.get("meta_policy"), dict)
    assert "web_search" in (payload.get("channels_used") or [])
    assert isinstance(payload.get("web_search"), dict)
    assert payload["web_search"].get("text")
    assert state_path.exists()


def test_executor_meta_retrieval_policy_uses_human_feedback_reward(tmp_path, monkeypatch):
    import json

    state_path = tmp_path / "meta_retrieval_policy_reward.json"
    monkeypatch.setenv("BSS_META_RETRIEVAL_ENABLED", "1")
    monkeypatch.setenv("BSS_META_RETRIEVAL_STATE_PATH", str(state_path))

    token = "ALGO_TOKEN_RLHF_789"
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "algo.py").write_text(
        "def foo() -> int:\n"
        f"    \"\"\"{token}\"\"\"\n"
        "    return 1\n",
        encoding="utf-8",
    )

    bridge = ToolEnvironmentBridge(allowed_roots=[tmp_path], allow_web_access=True, prefer_real_web_search=False)
    collector = RealTimeMetricsCollector(monitor=None)
    executor = AutonomousTaskExecutor(bridge, planner=HeuristicTaskPlanner(), max_replans=1, metrics_collector=collector)

    goal_text = f"Explain {token} algorithm"
    # Pre-seed a human feedback event for this goal so the RLHF reward is available
    # when the meta-retrieval policy is updated after the run.
    collector.emit_event(
        "human_feedback",
        latency=0.0,
        energy=0.0,
        throughput=0.0,
        status=None,
        confidence=0.9,
        stage=goal_text,
        metadata={"rating": 4.5},
    )

    goal = f"""
    ```json
    {{
      "goal": "{goal_text}",
      "steps": [
        {{"title": "fail", "action": {{"type": "unknown_action_type"}}}}
      ]
    }}
    ```
    """

    report = executor.run(
        goal,
        context={
            "knowledge_acquisition": {
                "enabled": True,
                "code_root": str(repo_root),
                "max_files": 50,
                "top_k": 5,
                "meta_retrieval_enabled": True,
            }
        },
    )

    assert report.success is False
    assert state_path.exists()

    raw = json.loads(state_path.read_text(encoding="utf-8"))
    domains = raw.get("domains") if isinstance(raw, dict) else None
    assert isinstance(domains, dict)
    algo = domains.get("algorithm")
    assert isinstance(algo, dict)
    web = algo.get("web_search")
    assert isinstance(web, dict)
    assert float(web.get("successes", 0.0)) > 0.0


def test_executor_can_consolidate_external_knowledge_into_long_term_memory(tmp_path, monkeypatch):
    monkeypatch.setenv("BSS_KNOWLEDGE_CONSOLIDATION_ENABLED", "1")

    token = "ALGO_TOKEN_CONSOLIDATE_999"
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "algo.py").write_text(
        "def foo() -> int:\n"
        f"    \"\"\"{token}\"\"\"\n"
        "    return 1\n",
        encoding="utf-8",
    )

    bridge = ToolEnvironmentBridge(allowed_roots=[tmp_path])
    memory = LongTermMemoryCoordinator(storage_root=tmp_path / "ltm")
    consolidator = ExternalKnowledgeConsolidator(memory=memory)
    executor = AutonomousTaskExecutor(
        bridge,
        planner=HeuristicTaskPlanner(),
        max_replans=1,
        knowledge_consolidator=consolidator,
    )

    goal_text = f"Explain {token} algorithm"
    goal = f"""
    ```json
    {{
      "goal": "{goal_text}",
      "steps": [
        {{"title": "fail", "action": {{"type": "unknown_action_type"}}}}
      ]
    }}
    ```
    """

    report = executor.run(
        goal,
        context={
            "knowledge_acquisition": {
                "enabled": True,
                "code_root": str(repo_root),
                "max_files": 50,
                "top_k": 5,
                "max_chars_per_hit": 400,
                "max_reference_chars": 400,
            }
        },
    )

    assert report.success is False
    kc = report.meta.get("knowledge_consolidation")
    assert isinstance(kc, dict)
    assert kc.get("stored") is True

    records = memory.query_similar(token, top_k=3)
    assert records
    assert any(token in (record.text or "") for record in records)


def test_consolidation_archives_unverified_web_payload_on_failure(tmp_path):
    from modules.knowledge.knowledge_consolidation import KnowledgeConsolidationConfig

    token = "ALGO_TOKEN_CONSOLIDATE_UNVERIFIED_888"
    memory = LongTermMemoryCoordinator(storage_root=tmp_path / "ltm")
    consolidator = ExternalKnowledgeConsolidator(
        memory=memory,
        config=KnowledgeConsolidationConfig(ingest_graph=False),
    )

    payload = {
        "query": token,
        "channels_used": ["documentation_tool", "web_search"],
        "references": [
            {
                "url": "https://93.184.216.34/doc",
                "title": "Doc",
                "source": "web_search",
                "host": "93.184.216.34",
                "trust": "low",
                "trust_score": 0.2,
            }
        ],
        "web_search": {
            "info": {
                "avg_trust": 0.2,
                "unique_hosts": 1,
                "trust_counts": {"high": 0, "medium": 0, "low": 1},
            }
        },
        "web": {
            "info": {
                "consensus": {
                    "level": "low",
                    "similarity_avg": 0.0,
                    "avg_trust": 0.2,
                    "unique_hosts": 1,
                    "needs_verification": True,
                    "warnings": ["single_host", "low_consensus", "low_trust_source"],
                }
            }
        },
        "retrieval_context": "stub",
    }

    result = consolidator.consolidate(
        goal=f"Explain {token}",
        knowledge_acquisition=[payload],
        success=False,
    )
    assert result.get("stored") is True

    records = list(memory.vector_store.iter_records())
    assert len(records) == 1
    stored = records[0]
    meta = stored.get("metadata") or {}
    assert meta.get("needs_verification") is True
    assert meta.get("archived") is True
    assert meta.get("archived_reason") == "needs_verification_on_failure"
    assert isinstance(meta.get("source_quality"), dict)
    assert meta["source_quality"].get("needs_verification") is True
    assert "SourceQuality" in (stored.get("text") or "")

    # Archived entries are not returned by default similarity recall.
    assert memory.query_similar(token, top_k=3) == []
