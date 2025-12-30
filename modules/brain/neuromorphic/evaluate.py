from __future__ import annotations

import argparse
import json
from importlib import import_module
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Callable, Optional, Mapping

from .data import DatasetLoader
from .temporal_encoding import latency_encode, rate_encode
from .spiking_network import SpikingNetworkConfig, NeuromorphicBackend, NeuromorphicRunResult


@dataclass
class EvaluationMetrics:
    mse: float
    total_spikes: int
    avg_rate_diff: float
    first_spike_latency: Optional[float]
    energy_used: float
    plugin_metrics: Dict[str, float]


def _load_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_signal(path: str | Path) -> list[list[float]]:
    data = _load_json(path)
    if not isinstance(data, list):
        raise ValueError("Signal file must be a JSON array")
    return [list(map(float, row)) for row in data]


def _load_target(path: str | Path) -> list[list[float]]:
    return _load_signal(path)


def _prepare_signal_for_encoding(
    signal: Sequence[Sequence[float]],
    encoding: str | None,
    *,
    steps: int = 5,
    t_scale: float = 1.0,
) -> Sequence:
    if encoding is None:
        return signal
    mode = encoding.lower()
    prepared: list = []
    if mode == "rate":
        for analog in signal:
            trains = rate_encode([float(v) for v in analog], steps=steps)
            prepared.extend(trains)
        return prepared
    if mode == "latency":
        events: list[tuple[float, list[int]]] = []
        for t, analog in enumerate(signal):
            events.extend(latency_encode([float(v) for v in analog], t_start=float(t), t_scale=t_scale))
        return events
    raise ValueError(f"Unsupported encoding '{encoding}'")
def _select_metrics(metrics: EvaluationMetrics, names: Sequence[str]) -> Dict[str, float]:
    lookup: Dict[str, float | None] = {
        "mse": metrics.mse,
        "total_spikes": float(metrics.total_spikes),
        "avg_rate_diff": metrics.avg_rate_diff,
        "first_spike_latency": metrics.first_spike_latency,
        "energy_used": metrics.energy_used,
    }
    lookup.update(metrics.plugin_metrics)
    if not names:
        return {k: v for k, v in lookup.items() if v is not None}
    selected: Dict[str, float] = {}
    for name in names:
        key = name.strip().lower()
        if not key:
            continue
        if key == "all":
            return {k: v for k, v in lookup.items() if v is not None}
        if key not in lookup:
            raise ValueError(f"Unknown metric {name!r}")
        value = lookup[key]
        if value is not None:
            selected[key] = value
    return selected if selected else {k: v for k, v in lookup.items() if v is not None}



def _load_plugin(spec: str) -> Callable:
    module_name, _, func_name = spec.partition(":")
    if not module_name or not func_name:
        raise ValueError("metrics plugin must be in module:function format")
    module = import_module(module_name)
    plugin = getattr(module, func_name, None)
    if plugin is None or not callable(plugin):
        raise ValueError(f"Callable {func_name!r} not found in module {module_name!r}")
    return plugin

def evaluate(
    cfg: SpikingNetworkConfig,
    signal: Sequence[Sequence[float]],
    target: Sequence[Sequence[float]],
    *,
    encoding: str | None = None,
    encoder_kwargs: Optional[Dict[str, Any]] = None,
    return_run: bool = False,
    custom_metric_fn: Callable[[SpikingNetworkConfig, Sequence[Sequence[float]], Sequence[Sequence[float]], Sequence[Tuple[float, List[int]]], float], Dict[str, float]] | None = None,
) -> EvaluationMetrics | tuple[EvaluationMetrics, NeuromorphicRunResult]:
    encoder_kwargs = dict(encoder_kwargs or {})
    steps = int(encoder_kwargs.get("steps", 5))
    t_scale = float(encoder_kwargs.get("t_scale", 1.0))
    prepared_signal = _prepare_signal_for_encoding(signal, encoding, steps=steps, t_scale=t_scale)

    backend = cfg.create_backend()
    run_result = backend.run_events(prepared_signal)
    outputs = run_result.spike_events
    num_neurons = 0
    if outputs:
        num_neurons = len(outputs[0][1])
    elif target:
        num_neurons = len(target[0])
    grouped: Dict[int, List[int]] = {}
    for time, spikes in outputs:
        index = int(time)
        if index < 0:
            continue
        grouped[index] = [int(s) for s in spikes]
    length = len(target)
    if length and num_neurons == 0:
        num_neurons = len(target[0])
    predicted = [grouped.get(step, [0] * num_neurons) for step in range(length)]

    mse_sum = 0.0
    total_spikes = 0
    num_neurons = len(predicted[0]) if predicted else (len(target[0]) if target else 0)
    predicted_counts = [0] * num_neurons
    target_counts = [0] * num_neurons
    first_pred_times: List[float | None] = [None] * num_neurons
    first_target_times: List[float | None] = [None] * num_neurons

    for step, ((time, pred), tgt) in enumerate(zip(outputs, target)):
        if len(pred) != len(tgt):
            raise ValueError("Prediction and target dimensions mismatch")
        mse_sum += sum((float(p) - float(t_val)) ** 2 for p, t_val in zip(pred, tgt)) / len(pred)
        total_spikes += sum(pred)
        for idx, (p, t_val) in enumerate(zip(pred, tgt)):
            if p:
                predicted_counts[idx] += p
                if first_pred_times[idx] is None:
                    first_pred_times[idx] = float(time)
            if t_val:
                target_counts[idx] += t_val
                if first_target_times[idx] is None:
                    first_target_times[idx] = float(step)

    duration = len(target) if target else 1
    predicted_rates = [count / duration for count in predicted_counts] if duration else []
    target_rates = [count / duration for count in target_counts] if duration else []
    rate_diffs = [abs(a - b) for a, b in zip(predicted_rates, target_rates)]
    avg_rate_diff = sum(rate_diffs) / len(rate_diffs) if rate_diffs else 0.0

    latency_diffs = [
        abs(pred_t - tgt_t)
        for pred_t, tgt_t in zip(first_pred_times, first_target_times)
        if pred_t is not None and tgt_t is not None
    ]
    first_latency = sum(latency_diffs) / len(latency_diffs) if latency_diffs else None

    plugin_metrics: Dict[str, float] = {}
    if custom_metric_fn is not None:
        plugin_metrics = custom_metric_fn(cfg, signal, target, outputs, run_result.energy_used) or {}

    mse = mse_sum / len(predicted) if predicted else 0.0
    metrics = EvaluationMetrics(
        mse=mse,
        total_spikes=total_spikes,
        avg_rate_diff=avg_rate_diff,
        first_spike_latency=first_latency,
        energy_used=run_result.energy_used,
        plugin_metrics=plugin_metrics,
    )
    if return_run:
        return metrics, run_result
    return metrics


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - CLI utility
    parser = argparse.ArgumentParser(description="Evaluate a spiking network against target data")
    parser.add_argument("--config", help="Path to network config JSON")
    parser.add_argument("--signal", help="Path to input signal JSON")
    parser.add_argument("--target", help="Path to target spike JSON")
    parser.add_argument("--dataset", help="Path to dataset directory (config/signal/target)")
    parser.add_argument("--output", help="Write evaluation metrics JSON to this path")
    parser.add_argument(
        "--metrics",
        default="all",
        help="Comma separated list of metrics (mse,total_spikes,avg_rate_diff,first_spike_latency,energy_used,all)",
    )
    parser.add_argument("--metrics-plugin", help="Load additional metric callable via module:function")
    args = parser.parse_args(argv)

    loader = DatasetLoader(Path(args.dataset)) if args.dataset else None
    cfg_path = args.config or (loader.load_config() if loader else None)
    signal_path = args.signal or (loader.load_signal() if loader else None)
    target_path = args.target or (loader.load_target() if loader else None)
    if not (cfg_path and signal_path and target_path):
        raise ValueError("Configuration, signal, and target must be provided directly or via --dataset")

    cfg = SpikingNetworkConfig.from_dict(_load_json(cfg_path))
    signal = _load_signal(signal_path)
    target = _load_target(target_path)
    plugin_fn = _load_plugin(args.metrics_plugin) if args.metrics_plugin else None
    metrics = evaluate(cfg, signal, target, custom_metric_fn=plugin_fn)
    metric_names = [name.strip() for name in (args.metrics.split(",") if args.metrics else [])]
    payload = _select_metrics(metrics, metric_names)
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        print(json.dumps(payload))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
