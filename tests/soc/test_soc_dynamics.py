import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.brain.soc import SelfOrganizedCriticality


def test_soc_power_law_distribution():
    np.random.seed(0)
    soc = SelfOrganizedCriticality(num_nodes=30, seed=0)

    avalanches = []
    for _ in range(2000):
        activity = np.random.rand(soc.num_nodes) * 0.05
        size = soc.update_network(activity)
        if size > 0:
            avalanches.append(size)

    assert len(avalanches) > 0
    sizes, counts = np.unique(avalanches, return_counts=True)
    log_sizes = np.log10(sizes)
    log_counts = np.log10(counts)
    slope, _ = np.polyfit(log_sizes, log_counts, 1)

    # Slope close to -1 indicates a power-law distribution
    assert -1.5 < slope < -0.5
