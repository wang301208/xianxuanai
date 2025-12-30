from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple


def _ensure_repo_root_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _build_config(args: argparse.Namespace, *, baseline_current_mean: float) -> Dict[str, Any]:
    return {
        "simulation": {
            "backend": "biophysical",
            "dt": float(args.dt_ms),
            "biophysical": {
                "seed": int(args.seed),
                "neurons_per_region": int(args.neurons_per_region),
                "neuron_model": str(args.neuron_model),
                "synapse_model": str(args.synapse_model),
                "baseline_current_mean": float(baseline_current_mean),
                "baseline_current_std": float(args.baseline_current_std),
                "noise_std": float(args.noise_std),
                "intra_connection_prob": float(args.intra_connection_prob),
                "inter_connection_prob": float(args.inter_connection_prob),
                "max_delay_ms": float(args.max_delay_ms),
            },
        }
    }


def _evaluate_rate(args: argparse.Namespace, *, baseline_current_mean: float) -> Tuple[float, int, float]:
    _ensure_repo_root_on_path()
    from BrainSimulationSystem.core.backends import get_backend

    cfg = _build_config(args, baseline_current_mean=baseline_current_mean)
    backend = get_backend("biophysical")
    net = backend.build_network(cfg)

    drive = float(args.input_drive)
    if drive != 0.0:
        net.set_input([drive] * getattr(net, "region_count", 8))
    else:
        net.set_input([0.0] * getattr(net, "region_count", 8))

    dt_ms = float(args.dt_ms)
    for _ in range(int(args.warmup_steps)):
        net.step(dt_ms)

    total_spikes = 0
    for _ in range(int(args.measure_steps)):
        state = net.step(dt_ms)
        total_spikes += int(state.get("spike_count", len(state.get("spikes", []))))

    n_neurons = int(getattr(net, "n_neurons", len(getattr(net, "neurons", {}) or {})) or 0)
    duration_s = float(args.measure_steps) * dt_ms / 1000.0
    rate_hz = float(total_spikes) / max(float(n_neurons) * max(duration_s, 1e-9), 1e-9)
    return rate_hz, total_spikes, duration_s


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate biophysical baseline drive to hit a target firing rate.")
    parser.add_argument("--target-rate-hz", type=float, default=5.0)
    parser.add_argument("--dt-ms", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--neurons-per-region", type=int, default=40)
    parser.add_argument("--neuron-model", type=str, default="izhikevich")
    parser.add_argument("--synapse-model", type=str, default="exp")
    parser.add_argument("--input-drive", type=float, default=0.0, help="Constant region-level drive (set_input).")

    parser.add_argument("--baseline-current-std", type=float, default=0.0)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--intra-connection-prob", type=float, default=0.08)
    parser.add_argument("--inter-connection-prob", type=float, default=0.01)
    parser.add_argument("--max-delay-ms", type=float, default=25.0)

    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--measure-steps", type=int, default=200)

    parser.add_argument("--lo", type=float, default=0.0)
    parser.add_argument("--hi", type=float, default=30.0)
    parser.add_argument("--max-iters", type=int, default=12)
    parser.add_argument("--json", action="store_true", help="Print a JSON summary to stdout.")

    args = parser.parse_args(argv)

    target = float(args.target_rate_hz)
    if not (target > 0.0):
        raise SystemExit("--target-rate-hz must be positive")

    lo = float(args.lo)
    hi = float(args.hi)
    if hi <= lo:
        raise SystemExit("--hi must be > --lo")

    best_mean = None
    best_rate = None

    # Expand search range if needed (simple heuristic).
    rate_hi, _, _ = _evaluate_rate(args, baseline_current_mean=hi)
    expand = 0
    while rate_hi < target and expand < 6:
        hi *= 1.6
        rate_hi, _, _ = _evaluate_rate(args, baseline_current_mean=hi)
        expand += 1

    for _ in range(int(args.max_iters)):
        mid = 0.5 * (lo + hi)
        rate, total_spikes, duration_s = _evaluate_rate(args, baseline_current_mean=mid)
        if best_rate is None or abs(rate - target) < abs(best_rate - target):
            best_rate = float(rate)
            best_mean = float(mid)

        # Bisection direction.
        if rate < target:
            lo = mid
        else:
            hi = mid

    out = {
        "recommended_baseline_current_mean": float(best_mean if best_mean is not None else 0.0),
        "achieved_rate_hz": float(best_rate if best_rate is not None else 0.0),
        "target_rate_hz": float(target),
        "dt_ms": float(args.dt_ms),
        "neurons_per_region": int(args.neurons_per_region),
        "neuron_model": str(args.neuron_model),
        "synapse_model": str(args.synapse_model),
        "input_drive": float(args.input_drive),
        "measure_steps": int(args.measure_steps),
        "warmup_steps": int(args.warmup_steps),
    }

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(
            "baseline_current_mean="
            f"{out['recommended_baseline_current_mean']:.4f}  "
            "rate_hz="
            f"{out['achieved_rate_hz']:.3f}  "
            "target="
            f"{out['target_rate_hz']:.3f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

