"""Full brain network composition built from modular mixins."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Deque, DefaultDict, Dict, List, Optional, Tuple

from .base import NeuralNetwork
from .dependencies import (
    BrainRegion,
    DetailedNeuron,
    NeuromorphicBackendManager,
    NeuromorphicBridge,
    SynapseManager,
    defaultdict,
    deque,
    np,
)
from .initialization import FullBrainInitializationMixin
from .integration import FullBrainIntegrationMixin
from .runtime import FullBrainRuntimeMixin


class FullBrainNetwork(
    FullBrainIntegrationMixin,
    FullBrainRuntimeMixin,
    FullBrainInitializationMixin,
    NeuralNetwork,
):
    """High level orchestrator for the full brain simulation network."""

    def __init__(self, config: Dict[str, Any], *, auto_initialize: bool = True):
        super().__init__(config)
        self.logger = logging.getLogger("FullBrainNetwork")

        # Core containers are prepared eagerly so delayed initialisation works transparently.
        self.brain_regions: Dict[BrainRegion, Dict[str, Any]] = {}
        self.cortical_columns: Dict[int, Any] = {}
        self.long_range_connections: Dict[int, Dict[str, Any]] = {}
        self.neuromorphic_backends: Dict[str, Any] = {}

        self._pending_synaptic_currents: DefaultDict[int, float] = defaultdict(float)
        self._pending_bridge_events: Deque[Tuple[int, float]] = deque()
        self._last_column_inputs: Dict[int, Dict[int, float]] = {}
        self._last_synapse_currents: Dict[int, float] = {}
        self._last_bridge_inputs: List[Dict[str, float]] = []
        self.total_neurons: int = 0
        self.total_synapses: int = 0
        self._column_neuron_to_global: Dict[Tuple[int, int], int] = {}
        self._global_to_column_neuron: Dict[int, Tuple[int, int]] = {}

        # Lifecycle / runtime state
        self.global_step: int = 0
        self._is_initialized: bool = False
        self.config_warnings: List[str] = []

        # External system references are injected by the orchestrator layer.
        self.synapse_manager: Optional[SynapseManager] = None
        self.cognitive_interface: Optional[Any] = None
        self.cell_diversity_system: Optional[Any] = None
        self.vascular_system: Optional[Any] = None
        self.partition_manager: Optional[Any] = None
        self.backend_manager: Optional[NeuromorphicBackendManager] = None
        self.backend_network_config: Optional[Dict[str, Any]] = None
        self.neuromorphic_bridge: Optional[NeuromorphicBridge] = None
        self.bridge_enabled: bool = False

        # Runtime caches used by analytics and export helpers.
        self.performance_metrics = {
            'update_times': [],
            'memory_usage': [],
            'spike_counts': [],
            'synchronization_times': []
        }
        self._last_neuron_voltages: Dict[int, float] = {}
        self._last_global_activity: Dict[str, Any] = {}
        self._last_bridge_spikes: List[Tuple[int, float]] = []
        self._last_bridge_outputs: List[Tuple[int, float]] = []

        runtime_cfg = self.config.get('runtime', {}) if isinstance(self.config.get('runtime', {}), dict) else {}
        simulation_cfg = runtime_cfg.get('simulation', {}) if isinstance(runtime_cfg.get('simulation', {}), dict) else {}
        self._max_sample_neurons = int(simulation_cfg.get('max_sample_neurons', 512))
        self._baseline_mean = float(simulation_cfg.get('baseline_current_mean', 45.0))
        self._baseline_std = float(simulation_cfg.get('baseline_current_std', 10.0))
        self._noise_std = float(simulation_cfg.get('noise_std', 5.0))
        self._lif_tau = float(simulation_cfg.get('lif_tau', 10.0))
        self._lif_decay = float(simulation_cfg.get('lif_decay', 1.0))
        self._weight_sample_limit = int(simulation_cfg.get('weight_sample_limit', 128))
        self._input_scale = float(simulation_cfg.get('input_scale', 1.0))

        self._runtime_prepared: bool = False
        self._runtime_neurons: List[Tuple[int, int, DetailedNeuron]] = []
        self._runtime_global_keys: List[str] = []
        self._runtime_global_ids: List[Optional[int]] = []
        self._runtime_neuron_index: Dict[Tuple[int, int], int] = {}
        self._runtime_synapses: List[Dict[str, Any]] = []
        self._runtime_synapses_by_pre: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        self._runtime_states: List[Dict[str, float]] = []
        self._baseline_currents: np.ndarray = np.zeros(0, dtype=float)

        if auto_initialize:
            self.initialize_sync()


class FullBrainNeuralNetwork(FullBrainNetwork):
    """Async-friendly wrapper around :class:`FullBrainNetwork`."""

    def __init__(self, config: Dict[str, Any], loop: Optional[asyncio.AbstractEventLoop] = None):
        super().__init__(config, auto_initialize=False)
        self._loop = loop

    async def initialize(self) -> None:
        if self._is_initialized:
            return

        loop = self._loop or asyncio.get_running_loop()
        await loop.run_in_executor(None, self.initialize_sync)

    async def shutdown(self) -> None:
        loop = self._loop or asyncio.get_running_loop()
        await loop.run_in_executor(None, self.shutdown_sync)


def create_full_brain_network(config: Optional[Dict[str, Any]] = None) -> FullBrainNetwork:
    """Factory that mirrors the historic module-level helper."""

    if config is None:
        from ...config.default_config import get_config
        config = get_config()

    return FullBrainNetwork(config)


__all__ = [
    'FullBrainNetwork',
    'FullBrainNeuralNetwork',
    'create_full_brain_network',
]
