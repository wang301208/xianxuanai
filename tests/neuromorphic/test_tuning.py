import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.brain.neuromorphic import SpikingNetworkConfig
from modules.brain.neuromorphic.tuning import random_search
from concurrent.futures import ThreadPoolExecutor


def test_random_search_updates_neuron_params():
    base = SpikingNetworkConfig(n_neurons=1)
    space = {"neuron_params.threshold": {"type": "uniform", "min": 0.5, "max": 0.5}}

    def evaluator(cfg: SpikingNetworkConfig) -> float:
        return cfg.neuron_params["threshold"]

    results = random_search(base, space, evaluator, trials=3, seed=123)
    assert len(results) == 3
    assert all("neuron_params.threshold" in result.params for result in results)
    assert all(result.config.neuron_params["threshold"] == 0.5 for result in results)


def test_random_search_modifies_top_level_field():
    base = SpikingNetworkConfig(n_neurons=1, learning_rate=0.1)
    space = {"learning_rate": [0.05, 0.2]}

    def evaluator(cfg: SpikingNetworkConfig) -> float:
        return -cfg.learning_rate

    results = random_search(base, space, evaluator, trials=5, seed=1)
    assert len(results) == 5
    best = results[0]
    assert best.config.learning_rate in (0.05, 0.2)


def test_tuning_main_writes_output(tmp_path):
    dataset = tmp_path / 'dataset'
    dataset.mkdir()
    (dataset / 'config.json').write_text('{"n_neurons":1,"neuron":"lif","weights":[[0.0]],"plasticity":"none"}', encoding='utf-8')
    (dataset / 'signal.json').write_text('[[1],[0]]', encoding='utf-8')
    (dataset / 'target.json').write_text('[[1],[0]]', encoding='utf-8')
    param_space = tmp_path / 'space.json'
    param_space.write_text('{"learning_rate": [0.1]}', encoding='utf-8')
    output = tmp_path / 'results.json'

    from modules.brain.neuromorphic.tuning import main as tuning_main

    tuning_main([
        '--dataset',
        str(dataset),
        '--param-space',
        str(param_space),
        '--trials',
        '1',
        '--output',
        str(output),
        '--parallel',
        'thread',
        '--workers',
        '2',
    ])

    import json
    data = json.loads(output.read_text(encoding='utf-8'))
    assert len(data) == 1


def test_random_search_parallel_matches_sequential():
    base = SpikingNetworkConfig(n_neurons=1)
    space = {"learning_rate": [0.1, 0.2]}

    def scorer(cfg: SpikingNetworkConfig) -> float:
        return -cfg.learning_rate

    sequential = random_search(base, space, scorer, trials=4, seed=123)
    with ThreadPoolExecutor(max_workers=2) as pool:
        parallel = random_search(base, space, scorer, trials=4, seed=123, executor=pool)

    assert [r.score for r in parallel] == [r.score for r in sequential]
