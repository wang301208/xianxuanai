import os
import sys
import asyncio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.brain.neuromorphic import SpikingNeuralNetwork


def test_run_async_concurrent():
    async def run_networks():
        net1 = SpikingNeuralNetwork(
            n_neurons=1, decay=0.8, threshold=1.0, reset=0.0, weights=[[0.0]]
        )
        net2 = SpikingNeuralNetwork(
            n_neurons=1, decay=0.8, threshold=1.0, reset=0.0, weights=[[0.0]]
        )

        # Disable plasticity to keep weights constant for test determinism
        net1.synapses.adapt = lambda *args, **kwargs: None
        net2.synapses.adapt = lambda *args, **kwargs: None

        inputs1 = [[0.6], [0.6], [0.0], [1.2]]
        inputs2 = [[1.2], [0.0], [0.0], [0.0]]

        results1, results2 = await asyncio.gather(
            net1.run_async(inputs1), net2.run_async(inputs2)
        )

        assert results1 == [(0, [0]), (1, [1]), (2, [0]), (3, [1])]
        assert results2 == [(0, [1]), (1, [0]), (2, [0]), (3, [0])]

    asyncio.run(run_networks())

