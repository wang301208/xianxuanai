from __future__ import annotations
import concurrent.futures
from concurrent.futures import Executor

import argparse
import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence, Iterable, Optional

from .spiking_network import SpikingNetworkConfig, NeuromorphicBackend
from .data import DatasetLoader


@dataclass
class TuningResult:
    """Store the outcome of a single hyperparameter trial."""

    trial: int
    params: Dict[str, Any]
    score: float
    config: SpikingNetworkConfig


ParameterSpec = Any


def _sample(spec: ParameterSpec, rng: random.Random) -> Any:
    if callable(spec):
        return spec(rng)
    if isinstance(spec, dict):
        kind = spec.get("type")
        if kind == "uniform":
            return rng.uniform(spec["min"], spec["max"])
        if kind == "loguniform":
            low, high = spec["min"], spec["max"]
            return 10 ** rng.uniform(low, high)
        if kind == "choice":
            return rng.choice(spec["values"])
        if kind == "int":
            return rng.randint(spec["min"], spec["max"])
        raise ValueError(f"Unknown parameter spec type: {kind}")
    if isinstance(spec, (list, tuple)):
        return rng.choice(spec)
    return spec


def _apply_params(
    base: SpikingNetworkConfig, params: Mapping[str, Any]
) -> SpikingNetworkConfig:
    updates: Dict[str, Any] = {}
    neuron_params = dict(base.neuron_params)
    for key, value in params.items():
        if key.startswith("neuron_params."):
            neuron_key = key.split(".", 1)[1]
            neuron_params[neuron_key] = value
        else:
            updates[key] = value
    if updates:
        base = replace(base, **updates)
    if neuron_params != base.neuron_params:
        base = replace(base, neuron_params=neuron_params)
    return base


def random_search(
    base_config: SpikingNetworkConfig,
    param_space: Mapping[str, ParameterSpec],
    evaluator: Callable[[SpikingNetworkConfig], float],
    *,
    trials: int = 20,
    seed: int | None = None,
    executor: Executor | None = None,
) -> list[TuningResult]:
    """Perform random search over spiking network hyperparameters."""

    rng = random.Random(seed)
    trial_indices = list(range(1, trials + 1))
    sampled_params = []
    for trial in trial_indices:
        params = {name: _sample(spec, rng) for name, spec in param_space.items()}
        sampled_params.append((trial, params.copy()))

    def _run_trial(trial_params: tuple[int, Dict[str, Any]]) -> TuningResult:
        trial, params = trial_params
        cfg = _apply_params(base_config, params)
        score = evaluator(cfg)
        return TuningResult(trial=trial, params=params, score=score, config=cfg)

    results: list[TuningResult] = []
    if executor is None:
        for entry in sampled_params:
            results.append(_run_trial(entry))
    else:
        futures = {executor.submit(_run_trial, entry): entry[0] for entry in sampled_params}
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda res: res.score, reverse=True)
    return results


# ---------------------------------------------------------------------------
# CLI utilities
# ---------------------------------------------------------------------------


def _load_param_space(path: str) -> Dict[str, ParameterSpec]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Parameter space must be a JSON object")
    return data


def _default_evaluator(signal: Sequence[Sequence[float]] | None = None):
    def _evaluate(cfg: SpikingNetworkConfig) -> float:
        network = cfg.create()
        net_signal = signal or [[1.0] + [0.0] * (cfg.n_neurons - 1)]
        outputs = network.run(net_signal)
        return sum(sum(spikes) for _, spikes in outputs)

    return _evaluate


def _resolve_signal(
    loader: DatasetLoader | None,
    path: str | None,
    n_neurons: int,
    steps: int,
) -> list[list[float]]:
    if path:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError("Signal file must be a JSON array")
        return [list(map(float, row)) for row in payload]
    if loader is not None:
        return loader.read_json("signal.json")
    return [[1.0] + [0.0] * (n_neurons - 1)] * steps


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - CLI helper
    parser = argparse.ArgumentParser(description="Random search tuner for spiking networks")
    parser.add_argument("--config", help="Path to base network config JSON")
    parser.add_argument("--dataset", help="Path to dataset directory (config/signal/target)")
    parser.add_argument("--param-space", required=True, help="Path to parameter space JSON")
    parser.add_argument("--signal", help="Optional JSON signal for evaluation")
    parser.add_argument("--steps", type=int, default=5, help="Fallback signal length when none provided")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--parallel", choices=["thread", "process"], help="Run trials in parallel using specified executor")
    parser.add_argument("--workers", type=int, help="Number of worker threads/processes when --parallel is used")
    parser.add_argument("--output", help="Write tuning results JSON to this path")
    args = parser.parse_args(argv)

    loader = DatasetLoader(Path(args.dataset)) if args.dataset else None
    cfg_path = args.config or (loader.load_config() if loader else None)
    if cfg_path is None:
        raise ValueError("Either --config or --dataset must be provided")
    cfg = _load_config(cfg_path)
    param_space = _load_param_space(args.param_space)
    signal = _resolve_signal(loader, args.signal, cfg.n_neurons, args.steps)
    evaluator = _default_evaluator(signal)
    if args.parallel:
        max_workers = args.workers
        if args.parallel == "thread":
            pool_cls = concurrent.futures.ThreadPoolExecutor
        else:
            pool_cls = concurrent.futures.ProcessPoolExecutor
        with pool_cls(max_workers=max_workers) as pool:
            results = random_search(
                cfg,
                param_space,
                evaluator,
                trials=args.trials,
                seed=args.seed,
                executor=pool,
            )
    else:
        results = random_search(
            cfg,
            param_space,
            evaluator,
            trials=args.trials,
            seed=args.seed,
        )

    summary = [{"trial": r.trial, "score": r.score, "params": r.params} for r in results]
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    else:
        print(json.dumps(summary))


def _load_config(path: str) -> SpikingNetworkConfig:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return SpikingNetworkConfig.from_dict(payload)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
