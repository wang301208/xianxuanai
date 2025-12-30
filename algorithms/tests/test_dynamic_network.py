import numpy as np

from algorithms.neuro_symbolic import DynamicConfig, DynamicNetwork


def test_dynamic_network_growth_and_pruning():
    cfg = DynamicConfig(max_layers=3, growth_trigger=0.5, shrink_trigger=-0.5, patience=1)
    net = DynamicNetwork(input_size=4, output_size=1, config=cfg)
    # starts with one hidden layer
    assert len(net.layers) == 1

    # insufficient improvement triggers growth
    net.adapt_structure(0.0)
    assert len(net.layers) == 2
    net.adapt_structure(0.0)
    assert len(net.layers) == 3
    # cannot exceed max_layers
    net.adapt_structure(0.0)
    assert len(net.layers) == 3

    # strong negative improvement triggers pruning
    net.adapt_structure(-1.0)
    assert len(net.layers) == 2
