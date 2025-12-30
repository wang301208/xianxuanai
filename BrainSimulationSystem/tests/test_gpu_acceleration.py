# -*- coding: utf-8 -*-

import pytest

from BrainSimulationSystem.core.cell_diversity import CellPopulationManager, CellType
from BrainSimulationSystem.core.gpu_acceleration import (
    TORCH_AVAILABLE,
    configure_gpu_acceleration,
    get_gpu_accelerator,
)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is required for GPU acceleration test")
def test_izhikevich_batch_gpu_acceleration():
    configure_gpu_acceleration({"enabled": True, "device": "cpu", "use_cuda": False})
    manager = CellPopulationManager((100.0, 100.0, 100.0))

    # Populate with a few Izhikevich neurons (L5 pyramidal defaults to that model)
    neurons = [
        manager._create_cell(CellType.PYRAMIDAL_L5B, (float(i * 10), 0.0, 0.0))  # type: ignore[attr-defined]
        for i in range(3)
    ]
    for neuron in neurons:
        manager.cells[neuron.cell_id] = neuron

    accelerator = get_gpu_accelerator()
    assert accelerator is not None and accelerator.available

    results = manager.update_all_cells(
        0.5,
        {
            "time": 1.0,
            "synaptic_current": 8.0,
            "external_current": 3.0,
            "noise": 1.0,
        },
    )

    assert len(results) == len(neurons)
    assert all(res["model"] == "izhikevich" for res in results.values())

    # Disable again to avoid impacting other tests
    configure_gpu_acceleration({"enabled": False})
