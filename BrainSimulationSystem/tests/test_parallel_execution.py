# -*- coding: utf-8 -*-

from __future__ import annotations

import concurrent.futures
import threading
import os
from typing import Dict, Any

import pytest

from BrainSimulationSystem.core.parallel_execution import (
    RegionParallelExecutor,
    RegionUpdateTask,
)
from BrainSimulationSystem.core.physiological_regions import (
    BrainRegion,
    BrainRegionNetwork,
)


def test_region_parallel_executor_thread_mode():
    executor = RegionParallelExecutor(strategy="thread", max_workers=3)
    tasks = [
        RegionUpdateTask(
            name=idx,
            runner=lambda _dt, inputs, base=idx: {"value": inputs["seed"] + base},
            dt=0.1,
            inputs={"seed": 5},
            mode="micro",
        )
        for idx in range(3)
    ]

    results = executor.run(tasks)
    executor.shutdown()

    assert results[0]["value"] == 5
    assert results[2]["value"] == 7
    assert all(payload["mode"] == "micro" for payload in results.values())


def _process_runner(dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
    return {"value": inputs["seed"] + inputs["offset"] + dt}


def _pid_runner(dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
    return {"pid": os.getpid(), "value": inputs["seed"] + dt}


def test_region_parallel_executor_process_mode():
    executor = RegionParallelExecutor(
        strategy="distributed", max_workers=2, distributed={"enabled": True}
    )
    tasks = [
        RegionUpdateTask(
            name=f"task-{idx}",
            runner=_process_runner,
            dt=0.5,
            inputs={"seed": idx, "offset": 1},
            mode="micro",
        )
        for idx in range(3)
    ]

    results = executor.run(tasks)
    executor.shutdown()

    assert results["task-0"]["value"] == pytest.approx(1.5)
    assert results["task-2"]["value"] == pytest.approx(3.5)
    assert all(payload["mode"] == "micro" for payload in results.values())


def test_region_parallel_executor_process_mode_honors_max_workers(monkeypatch):
    observed_max_workers = []

    class RecordingProcessPool(concurrent.futures.ProcessPoolExecutor):
        def __init__(self, *args, **kwargs):
            observed_max_workers.append(kwargs.get("max_workers"))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", RecordingProcessPool)

    executor = RegionParallelExecutor(strategy="process", max_workers=2)
    tasks = [
        RegionUpdateTask(
            name=f"task-{idx}",
            runner=_process_runner,
            dt=0.25,
            inputs={"seed": idx, "offset": 0},
            mode="micro",
        )
        for idx in range(3)
    ]

    results = executor.run(tasks)
    executor.shutdown()

    assert observed_max_workers == [2]
    assert results["task-0"]["value"] == pytest.approx(0.25)
    assert results["task-2"]["value"] == pytest.approx(2.25)


def test_region_parallel_executor_process_mode_does_not_fallback_to_threads(monkeypatch):
    def _fail_thread_run(self, tasks):  # pragma: no cover - ensures path not used
        raise AssertionError("Threaded execution should not be invoked in process mode")

    monkeypatch.setattr(RegionParallelExecutor, "_run_threaded", _fail_thread_run)

    executor = RegionParallelExecutor(strategy="process", max_workers=2)
    tasks = [
        RegionUpdateTask(
            name=f"task-{idx}",
            runner=_pid_runner,
            dt=0.1,
            inputs={"seed": idx},
            mode="micro",
        )
        for idx in range(2)
    ]

    results = executor.run(tasks)
    executor.shutdown()

    parent_pid = os.getpid()
    for payload in results.values():
        assert payload["pid"] != parent_pid
        assert payload["value"] in {0.1, 1.1}


def test_region_parallel_executor_process_mode_rejects_unserializable_inputs():
    executor = RegionParallelExecutor(strategy="process")
    unserializable_lock = threading.Lock()
    task = RegionUpdateTask(
        name="bad",
        runner=_process_runner,
        dt=0.1,
        inputs={"seed": 1, "offset": unserializable_lock},
        mode="micro",
    )

    with pytest.raises(ValueError):
        executor.run([task])


def test_brain_region_network_parallel_configuration():
    class DummyRegion:
        def __init__(self, name: BrainRegion, base: float):
            self.region_name = name
            self.base = base
            self._pending_layer_inputs: Dict[str, float] = {}
            self._pending_modulatory_inputs: Dict[str, float] = {}
            self._pending_inter_region_input = 0.0

        def update(self, dt: float, global_inputs: Dict[str, Any]) -> Dict[str, Any]:
            increment = float(global_inputs.get("increment", 0.0))
            return {"value": self.base + increment}

    network = BrainRegionNetwork()
    region_a = DummyRegion(BrainRegion.PRIMARY_VISUAL_CORTEX, 1.0)
    region_b = DummyRegion(BrainRegion.PRIMARY_MOTOR_CORTEX, 2.0)
    network.add_region(region_a)
    network.add_region(region_b)

    network.configure_parallelism({"mode": "thread", "max_workers": 2})
    results = network.update_network(
        1.0,
        {
            BrainRegion.PRIMARY_VISUAL_CORTEX: {"increment": 0.5},
            BrainRegion.PRIMARY_MOTOR_CORTEX: {"increment": 1.0},
        },
    )
    network.shutdown_parallelism()

    assert results[BrainRegion.PRIMARY_VISUAL_CORTEX]["value"] == 1.5
    assert results[BrainRegion.PRIMARY_MOTOR_CORTEX]["value"] == 3.0
    assert results[BrainRegion.PRIMARY_VISUAL_CORTEX]["mode"] == "micro"
