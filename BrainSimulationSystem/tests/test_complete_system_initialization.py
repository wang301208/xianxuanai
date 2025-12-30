import asyncio
import importlib
import sys
import types
import unittest

if "networkx" not in sys.modules:
    class _SimpleDiGraph:
        def __init__(self):
            self.nodes = set()
            self.edges = set()

        def add_node(self, node):
            self.nodes.add(node)

        def add_edge(self, source, target):
            self.nodes.add(source)
            self.nodes.add(target)
            self.edges.add((source, target))

    sys.modules["networkx"] = types.SimpleNamespace(DiGraph=_SimpleDiGraph)

cell_div_module = importlib.import_module("BrainSimulationSystem.core.cell_diversity")
if not hasattr(cell_div_module, "CellDiversitySystem"):
    class _FallbackCellDiversitySystem:
        def __init__(self, config):
            self.config = config

        async def initialize(self):  # pragma: no cover - async convenience
            return None

    cell_div_module.CellDiversitySystem = _FallbackCellDiversitySystem
    sys.modules["BrainSimulationSystem.core.cell_diversity"] = cell_div_module

vascular_module = importlib.import_module("BrainSimulationSystem.core.vascular_system")
if not hasattr(vascular_module, "VascularSystem"):
    class _FallbackVascularSystem:
        def __init__(self, config):
            self.config = config

        async def initialize(self):  # pragma: no cover - async convenience
            return None

    vascular_module.VascularSystem = _FallbackVascularSystem
    sys.modules["BrainSimulationSystem.core.vascular_system"] = vascular_module

phys_regions_module = importlib.import_module("BrainSimulationSystem.core.physiological_regions")
if not hasattr(phys_regions_module, "PhysiologicalRegionManager"):
    class _FallbackPhysRegionManager:
        def __init__(self, config):
            self.config = config

        async def initialize(self):  # pragma: no cover - async convenience
            return None

    phys_regions_module.PhysiologicalRegionManager = _FallbackPhysRegionManager
    sys.modules["BrainSimulationSystem.core.physiological_regions"] = phys_regions_module

from BrainSimulationSystem.core.complete_brain_system import (
    BrainSimulationConfig,
    CompleteBrainSimulationSystem,
)


class TestCompleteBrainSystemInitialization(unittest.IsolatedAsyncioTestCase):
    async def test_neural_network_initialization_path(self):
        config = BrainSimulationConfig(
            total_neurons=1_000,
            total_synapses=1_000_000,
            brain_regions=4,
            use_neuromorphic=False,
            distributed_computing=False,
            gpu_acceleration=False,
            detailed_cell_types=False,
            vascular_modeling=False,
            glial_cells=False,
            metabolic_modeling=False,
            consciousness_modeling=False,
            memory_consolidation=False,
            learning_plasticity=False,
        )

        system = CompleteBrainSimulationSystem(config)

        try:
            for _ in range(60):
                if system.is_initialized:
                    break
                await asyncio.sleep(0.05)

            self.assertTrue(system.is_initialized, "System did not finish initialization in time")
            self.assertIsNotNone(system.neural_network)
            self.assertIs(system.neural_network.synapse_manager, system.synapse_manager)
            self.assertTrue(system.neural_network.is_healthy())

            neurons = system.neural_network.get_region_neurons("PREFRONTAL_CORTEX")
            self.assertIsInstance(neurons, list)
        finally:
            if system.neural_network is not None:
                await system.neural_network.shutdown()


if __name__ == "__main__":
    unittest.main()
