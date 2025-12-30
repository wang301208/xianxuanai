from __future__ import annotations

import time

from modules.environment.environment_adapter import (
    EnvironmentAdapter,
    EnvironmentSnapshot,
    choose_task_adapter_mode,
    format_environment_prompt,
)


def test_environment_adapter_evaluate_throttles_on_pressure(monkeypatch) -> None:
    monkeypatch.delenv("EDGE_DEVICE", raising=False)
    adapter = EnvironmentAdapter(
        worker_id="test-env",
        event_bus=None,
        apply_callback=None,
        sustain_samples=1,
        min_concurrency=1,
        max_concurrency=None,
        cpu_high=90.0,
        mem_high=85.0,
    )
    snap = EnvironmentSnapshot(
        timestamp=time.time(),
        cpu_count=8,
        cpu_percent=95.0,
        memory_percent=90.0,
        memory_total_gb=16.0,
        gpu_available=False,
        registry_workers={},
    )
    adj = adapter.evaluate(snap)
    assert adj.concurrency == 4  # base_max=8, throttled to ~50%
    assert "high_cpu" in adj.reason
    assert "high_mem" in adj.reason


def test_environment_adapter_edge_model_recommendation(monkeypatch) -> None:
    monkeypatch.setenv("EDGE_DEVICE", "1")
    monkeypatch.setenv("ENVIRONMENT_ADAPTER_MODEL_EDGE", "edge-small")
    adapter = EnvironmentAdapter(
        worker_id="test-env",
        event_bus=None,
        apply_callback=None,
        sustain_samples=1,
    )
    snap = EnvironmentSnapshot(
        timestamp=time.time(),
        cpu_count=2,
        cpu_percent=10.0,
        memory_percent=10.0,
        memory_total_gb=4.0,
        gpu_available=True,
        registry_workers={},
    )
    adj = adapter.evaluate(snap)
    assert adj.llm_model == "edge-small"
    assert "edge_model" in adj.reason


def test_format_environment_prompt_contains_key_fields() -> None:
    snap = EnvironmentSnapshot(
        timestamp=time.time(),
        cpu_count=4,
        cpu_percent=12.0,
        memory_percent=34.0,
        memory_total_gb=8.0,
        gpu_available=True,
        registry_workers={},
    )
    prompt = format_environment_prompt(snap)
    assert "cpu_cores" in prompt
    assert "memory_total" in prompt
    assert "gpu" in prompt


def test_choose_task_adapter_mode_prefers_explicit_override(monkeypatch) -> None:
    monkeypatch.setenv("TASK_ADAPTER_PREFERRED", "local")
    assert choose_task_adapter_mode() == "local"


def test_environment_adapter_prompt_calls_snapshot(monkeypatch) -> None:
    adapter = EnvironmentAdapter(worker_id="test-env", event_bus=None, apply_callback=None, sustain_samples=1)
    snap = adapter.snapshot()
    prompt = adapter.environment_prompt()
    # Basic sanity: prompt includes cpu core count from the snapshot.
    assert str(snap.cpu_count) in prompt
