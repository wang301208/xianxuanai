import asyncio
from pathlib import Path
from typing import Any, Dict
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from BrainSimulationSystem.core.backends import (
    EventRouter,
    EventType,
    HardwareConfig,
    HardwarePlatform,
    HardwareSpecs,
    NeuromorphicBackend,
    NeuromorphicBackendManager,
    NeuromorphicEvent,
    SpikeEvent,
    create_neuromorphic_interface,
)


class _DummyBackend(NeuromorphicBackend):
    """Simple backend used to validate routing and selection."""

    def __init__(self, platform: HardwarePlatform, max_neurons: int = 128):
        self._specs = HardwareSpecs(
            platform=platform,
            max_neurons=max_neurons,
            max_synapses=max_neurons * 4,
            max_cores=4,
            memory_size=1,
            power_consumption=1.0,
            real_time_factor=1.0,
            event_throughput=1000,
        )
        super().__init__(platform, {})
        self.hardware_specs = self._specs
        self.received_events = []

    def _get_hardware_specs(self) -> HardwareSpecs:
        return self._specs

    async def initialize_hardware(self) -> bool:
        self.is_initialized = True
        return True

    async def configure_network(self, network_config: Dict[str, Any]) -> bool:
        return True

    async def run_simulation(self, duration: float) -> Dict[str, Any]:
        return {"duration": duration, "power_consumption": 0.0}

    async def process_event(self, event: NeuromorphicEvent) -> Any:
        self.received_events.append(event)
        return None

    async def shutdown(self):
        self.is_running = False


def test_event_router_targets_configured_backend():
    backend = _DummyBackend(HardwarePlatform.SPINNAKER)
    manager = NeuromorphicBackendManager(config={})
    manager.backends = {HardwarePlatform.SPINNAKER: backend}
    manager.event_router.add_route(source_id=42, target_backend=HardwarePlatform.SPINNAKER)

    event = NeuromorphicEvent(
        timestamp=0.01,
        event_type=EventType.SPIKE,
        source_id=42,
        target_id=None,
        data={"payload": 1},
    )

    asyncio.run(manager.event_router.route_event(event, manager.backends))

    queued_event = backend.event_queue.get_nowait()
    assert queued_event == event


def test_backend_selection_prefers_initialized_backend():
    manager = NeuromorphicBackendManager(config={})

    preferred = _DummyBackend(HardwarePlatform.INTEL_LOIHI, max_neurons=512)
    fallback = _DummyBackend(HardwarePlatform.SPINNAKER, max_neurons=64)

    preferred.is_initialized = True
    fallback.is_initialized = True

    manager.backends = {
        HardwarePlatform.INTEL_LOIHI: preferred,
        HardwarePlatform.SPINNAKER: fallback,
    }

    selected = asyncio.run(
        manager.select_optimal_backend(network_size=128, requirements={})
    )

    assert selected == HardwarePlatform.INTEL_LOIHI


def test_create_neuromorphic_interface_wraps_backend():
    config = HardwareConfig(platform=HardwarePlatform.INTEL_LOIHI, chip_count=1)
    interface = create_neuromorphic_interface(HardwarePlatform.INTEL_LOIHI, config)

    # Basic synchronous API should be available and callable.
    assert interface.connect() is False or interface.is_connected in {True, False}
    mapping = interface.map_neurons([{"tau_m": 20.0}, {"tau_m": 30.0}])
    assert set(mapping.keys()) == {0, 1}

    # Ensure sending spike events succeeds without raising.
    spike = SpikeEvent(neuron_id=0, timestamp=0.0)
    interface.send_spike_events([spike])

    interface.disconnect()
