"""
Complete System Validation Tests

These tests exercise the end-to-end "complete brain" simulation scaffold.

Note: The original version used `@pytest.mark.asyncio`, but this repo does not
ship with `pytest-asyncio` as a dependency. To keep the tests runnable in
minimal environments, we execute async code via `asyncio.run()`.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

import numpy as np
import pytest

from ..core.complete_brain_system import (
    BrainSimulationConfig,
    CompleteBrainSimulationSystem,
    get_full_brain_config,
    get_prototype_config,
)


@asynccontextmanager
async def _managed_system(config: BrainSimulationConfig, *, init_wait: float = 2.0):
    system = CompleteBrainSimulationSystem(config)
    await asyncio.sleep(init_wait)
    try:
        yield system
    finally:
        await system.shutdown()


def _run(coro):
    return asyncio.run(coro)


class TestCompleteBrainSystem:
    def test_system_initialization(self):
        async def _case():
            async with _managed_system(get_prototype_config()) as system:
                assert system.is_initialized
                assert not system.is_running

                status = system.get_system_status()
                component_status = status["component_status"]
                assert component_status["neural_network"]
                assert component_status["synapse_manager"]
                assert component_status["cognitive_architecture"]
                assert component_status["memory_system"]

        _run(_case())

    def test_neural_network_functionality(self):
        async def _case():
            async with _managed_system(get_prototype_config()) as system:
                assert system.neural_network is not None
                assert system.neural_network.total_neurons > 0

                neuron_count = len(system.neural_network.neurons)
                assert neuron_count > 0

                regions = system.neural_network.get_brain_regions()
                assert len(regions) > 0

        _run(_case())

    def test_synapse_system_functionality(self):
        async def _case():
            async with _managed_system(get_prototype_config()) as system:
                assert system.synapse_manager is not None
                synapse_stats = system.synapse_manager.get_statistics()
                assert synapse_stats["total_synapses"] > 0
                nt_distribution = synapse_stats["neurotransmitter_distribution"]
                assert "glutamate" in nt_distribution
                assert "gaba" in nt_distribution

        _run(_case())

    def test_cognitive_architecture_functionality(self):
        async def _case():
            async with _managed_system(get_prototype_config()) as system:
                assert system.cognitive_architecture is not None
                cognitive_state = system.cognitive_architecture.get_cognitive_state()
                assert "attention_state" in cognitive_state
                assert "consciousness_state" in cognitive_state
                assert "neuromodulation" in cognitive_state

                attention_state = cognitive_state["attention_state"]
                assert 0.0 <= attention_state["focus_strength"] <= 1.0

                consciousness_state = cognitive_state["consciousness_state"]
                assert 0.0 <= consciousness_state["awareness_level"] <= 1.0

        _run(_case())

    def test_memory_system_functionality(self):
        async def _case():
            async with _managed_system(get_prototype_config()) as system:
                assert system.memory_system is not None

                from ..models.memory import MemoryType

                episode = {"event": "test_event", "location": "test_location"}
                context = {"time": "morning", "mood": "positive"}

                memory_id = await system.memory_system.encode_memory(
                    episode, MemoryType.EPISODIC, context
                )
                assert memory_id is not None

                query = {"event": "test_event"}
                results = await system.memory_system.retrieve_memory(query, MemoryType.EPISODIC)
                assert len(results) > 0
                assert results[0]["memory_type"] == MemoryType.EPISODIC

        _run(_case())

    def test_biological_components(self):
        async def _case():
            async with _managed_system(get_prototype_config()) as system:
                if system.cell_diversity_system:
                    cell_types = system.cell_diversity_system.get_cell_type_distribution()
                    assert len(cell_types) > 0

                if system.vascular_system:
                    vascular_state = system.vascular_system.get_system_state()
                    assert "total_vessels" in vascular_state
                    assert vascular_state["total_vessels"] > 0

        _run(_case())

    def test_short_simulation_run(self):
        async def _case():
            async with _managed_system(get_prototype_config()) as system:
                duration = 10.0
                results = await system.run_simulation(duration)
                assert "simulation_config" in results
                assert "performance_metrics" in results
                assert "spike_statistics" in results

                sim_config = results["simulation_config"]
                assert sim_config["duration"] == duration

                perf_metrics = results["performance_metrics"]
                assert perf_metrics["neurons_processed"] > 0
                assert perf_metrics["synapses_processed"] > 0

        _run(_case())

    def test_system_health_monitoring(self):
        async def _case():
            async with _managed_system(get_prototype_config()) as system:
                health_status = await system._check_system_health()
                assert "neural_network" in health_status
                assert "synapse_manager" in health_status
                assert "overall_health" in health_status
                assert health_status["overall_health"] in ["healthy", "degraded"]

        _run(_case())

    def test_multi_scale_integration(self):
        async def _case():
            async with _managed_system(get_prototype_config()) as system:
                await system._simulation_step()
                status = system.get_system_status()

                if system.synapse_manager:
                    synapse_stats = system.synapse_manager.get_statistics()
                    assert synapse_stats["total_synapses"] > 0

                assert status["component_status"]["neural_network"]

                if system.cognitive_architecture:
                    cognitive_state = system.cognitive_architecture.get_cognitive_state()
                    assert cognitive_state["active_regions"] > 0

                assert status["component_status"]["cognitive_architecture"]

        _run(_case())

    def test_plasticity_mechanisms(self):
        async def _case():
            async with _managed_system(get_prototype_config()) as system:
                initial_stats = system.synapse_manager.get_statistics()
                initial_weights = initial_stats["weight_statistics"]

                await system.run_simulation(50.0)
                final_stats = system.synapse_manager.get_statistics()
                final_weights = final_stats["weight_statistics"]

                assert final_weights["mean"] != initial_weights["mean"] or final_weights["std"] >= 0.0

        _run(_case())

    def test_learning_and_memory_consolidation(self):
        async def _case():
            async with _managed_system(get_prototype_config()) as system:
                from ..models.memory import MemoryType

                memories = []
                for i in range(5):
                    episode = {"event": f"event_{i}", "location": "test_location"}
                    context = {"time": "morning", "mood": "positive"}
                    memory_id = await system.memory_system.encode_memory(
                        episode, MemoryType.EPISODIC, context
                    )
                    memories.append(memory_id)

                initial_stats = system.memory_system.get_memory_statistics()
                await system.run_simulation(100.0)
                final_stats = system.memory_system.get_memory_statistics()
                assert final_stats["total_memories"] >= initial_stats["total_memories"]

        _run(_case())

    def test_consciousness_emergence(self):
        async def _case():
            async with _managed_system(get_prototype_config()) as system:
                results = await system.run_simulation(200.0)
                cognitive_summary = results["cognitive_summary"]
                if "average_consciousness_level" in cognitive_summary:
                    consciousness_level = cognitive_summary["average_consciousness_level"]
                    assert 0.0 <= consciousness_level <= 1.0
                    assert consciousness_level > 0.1

        _run(_case())


class TestSystemScalability:
    def test_small_scale_system(self):
        async def _case():
            config = BrainSimulationConfig(
                total_neurons=1000,
                total_synapses=10000,
                brain_regions=5,
                simulation_duration=10.0,
            )
            async with _managed_system(config, init_wait=1.0) as system:
                results = await system.run_simulation()
                assert results["simulation_config"]["total_neurons"] == 1000

        _run(_case())

    def test_medium_scale_system(self):
        async def _case():
            config = BrainSimulationConfig(
                total_neurons=100000,
                total_synapses=1000000,
                brain_regions=10,
                simulation_duration=50.0,
            )
            async with _managed_system(config, init_wait=2.0) as system:
                results = await system.run_simulation()
                assert results["simulation_config"]["total_neurons"] == 100000

        _run(_case())


class TestSystemRobustness:
    def test_error_recovery(self):
        async def _case():
            system = CompleteBrainSimulationSystem(get_prototype_config())
            await asyncio.sleep(1.0)
            try:
                original_neural_network = system.neural_network
                system.neural_network = None

                health_status = await system._check_system_health()
                assert health_status["overall_health"] == "degraded"
                assert "neural_network" in health_status.get("failed_components", [])

                system.neural_network = original_neural_network
                health_status = await system._check_system_health()
                assert health_status["overall_health"] == "healthy"
            finally:
                await system.shutdown()

        _run(_case())

    def test_configuration_validation(self):
        invalid_config = BrainSimulationConfig(
            total_neurons=-1000,
            total_synapses=1000,
            dt=0.0,
        )
        assert not invalid_config.validate()

        valid_config = get_prototype_config()
        assert valid_config.validate()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__, "-v"])

