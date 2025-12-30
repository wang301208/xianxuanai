import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.brain.neuromorphic import EvaluationMetrics, SpikingNetworkConfig
from modules.brain.neuromorphic.evaluate import evaluate


def test_evaluate_returns_zero_for_empty_signal():
    cfg = SpikingNetworkConfig(n_neurons=1, plasticity=None)
    metrics = evaluate(cfg, [], [])
    assert isinstance(metrics, EvaluationMetrics)
    assert metrics.mse == 0.0
    assert metrics.total_spikes == 0
    assert metrics.avg_rate_diff == 0.0
    assert metrics.first_spike_latency is None
    assert metrics.energy_used == 0
    assert metrics.plugin_metrics == {}



def test_evaluate_computes_mse_and_spikes():
    cfg = SpikingNetworkConfig(n_neurons=1, plasticity=None)
    signal = [[1.0], [0.0]]
    target = [[0.0], [0.0]]
    metrics = evaluate(cfg, signal, target)
    assert metrics.mse >= 0.0
    assert metrics.total_spikes >= 0
    assert metrics.avg_rate_diff >= 0.0
    assert metrics.energy_used >= 0



def test_evaluate_main_dataset_output(tmp_path):
    dataset = tmp_path / 'dataset'
    dataset.mkdir()
    (dataset / 'config.json').write_text('{"n_neurons":1,"neuron":"lif","weights":[[0.0]],"plasticity":"none"}', encoding='utf-8')
    (dataset / 'signal.json').write_text('[[1],[0]]', encoding='utf-8')
    (dataset / 'target.json').write_text('[[1],[0]]', encoding='utf-8')
    output = tmp_path / 'metrics.json'

    from modules.brain.neuromorphic.evaluate import main as evaluate_main

    evaluate_main([
        '--dataset',
        str(dataset),
        '--output',
        str(output),
    ])

    import json
    data = json.loads(output.read_text(encoding='utf-8'))
    assert {'mse', 'total_spikes', 'avg_rate_diff', 'first_spike_latency', 'energy_used'} <= set(data.keys())


def test_evaluate_metrics_option(tmp_path):
    dataset = tmp_path / 'dataset'
    dataset.mkdir()
    (dataset / 'config.json').write_text('{"n_neurons":1,"neuron":"lif","weights":[[0.0]],"plasticity":"none"}', encoding='utf-8')
    (dataset / 'signal.json').write_text('[[1],[0]]', encoding='utf-8')
    (dataset / 'target.json').write_text('[[1],[0]]', encoding='utf-8')
    output = tmp_path / 'metrics.json'

    from modules.brain.neuromorphic.evaluate import main as evaluate_main
    evaluate_main([
        '--dataset',
        str(dataset),
        '--metrics',
        'total_spikes',
        '--output',
        str(output),
    ])
    import json
    data = json.loads(output.read_text(encoding='utf-8'))
    assert list(data.keys()) == ['total_spikes']



def test_evaluate_with_metrics_plugin(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "config.json").write_text('{"n_neurons":1,"neuron":"lif","weights":[[0.0]],"plasticity":"none"}', encoding="utf-8")
    (dataset / "signal.json").write_text('[[1],[0]]', encoding="utf-8")
    (dataset / "target.json").write_text('[[1],[0]]', encoding="utf-8")
    plugin_module = tmp_path / "plugin_metrics.py"
    plugin_module.write_text(
        "def compute(cfg, signal, target, outputs, energy):\n"
        "    return {\'energy_double\': energy * 2}\n",
        encoding="utf-8",
    )
    output = tmp_path / "metrics.json"
    import sys
    sys.path.insert(0, str(tmp_path))
    try:
        from modules.brain.neuromorphic.evaluate import main as evaluate_main
        evaluate_main([
            "--dataset",
            str(dataset),
            "--metrics",
            "energy_used,energy_double",
            "--metrics-plugin",
            "plugin_metrics:compute",
            "--output",
            str(output),
        ])
    finally:
        sys.path.remove(str(tmp_path))
    import json
    data = json.loads(output.read_text(encoding="utf-8"))
    assert "energy_used" in data and "energy_double" in data
    assert data["energy_double"] == data["energy_used"] * 2
