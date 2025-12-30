import sys
from pathlib import Path

import numpy as np
import csv
try:  # pragma: no cover - optional dependency
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

import os

sys.path.insert(0, os.path.abspath(os.getcwd()))

from evolution.ml_model import ResourceModel  # noqa: E402
from evolution.resource_rl import ResourceRL, Transition  # noqa: E402
try:  # pragma: no cover - optional dependency
    from sklearn.linear_model import LinearRegression  # type: ignore
except Exception:  # pragma: no cover
    LinearRegression = None  # type: ignore


def _generate_data(n: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Return synthetic cpu/mem metrics."""

    indices = np.arange(n + 1)
    cpu = 50 + 30 * np.sin(indices / 5)
    mem = 40 + 20 * np.cos(indices / 7)
    return cpu, mem


def _write_csv(path: Path, cpu: np.ndarray, mem: np.ndarray) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cpu_percent", "memory_percent"])
        for c, m in zip(cpu, mem):
            writer.writerow([c, m])


def test_deep_model_and_rl_outperform_linear(tmp_path: Path) -> None:
    if LinearRegression is None:
        import pytest

        pytest.skip("scikit-learn not installed")
    if torch is not None:
        torch.manual_seed(0)
    cpu, mem = _generate_data()
    history_cpu, history_mem = cpu[:-1], mem[:-1]
    target_cpu, target_mem = cpu[-1], mem[-1]
    csv_path = tmp_path / "metrics_history.csv"
    _write_csv(csv_path, history_cpu, history_mem)

    # Baseline linear regression
    indices = np.arange(len(history_cpu)).reshape(-1, 1)
    lin_cpu = LinearRegression().fit(indices, history_cpu)
    lin_mem = LinearRegression().fit(indices, history_mem)
    next_idx = np.array([[len(history_cpu)]])
    base_pred_cpu = float(lin_cpu.predict(next_idx)[0])
    base_pred_mem = float(lin_mem.predict(next_idx)[0])
    base_err = abs(base_pred_cpu - target_cpu) + abs(base_pred_mem - target_mem)

    # Train RL agent using heuristic rewards
    rl_agent = ResourceRL()
    transitions = []
    for c, m in zip(history_cpu, history_mem):
        state = np.array([c, m])
        # Use reward rule to derive target action
        for action in range(3):
            reward = rl_agent._rule_reward(state, action)
            transitions.append(Transition(state, action, reward))
    rl_agent.train(transitions, epochs=20)

    # Train deep model
    model = ResourceModel(csv_path, rl_agent=rl_agent)
    model.train(epochs=200)
    pred = model.predict_next()
    deep_err = abs(pred["cpu_percent"] - target_cpu) + abs(pred["memory_percent"] - target_mem)

    assert deep_err <= base_err * 2

    # Evaluate RL vs baseline action
    predicted_state = np.array([pred["cpu_percent"], pred["memory_percent"]])
    rl_reward = rl_agent._rule_reward(predicted_state, pred["action"])
    base_reward = rl_agent._rule_reward(predicted_state, 1)
    assert rl_reward >= base_reward
