import sys
import types
from importlib import util
from pathlib import Path
import random
import math


ROOT = Path(__file__).resolve().parents[2]
backend_pkg = types.ModuleType("backend")
backend_pkg.__path__ = [str(ROOT / "backend")]
sys.modules.setdefault("backend", backend_pkg)
execution_pkg = types.ModuleType("backend.execution")
execution_pkg.__path__ = [str(ROOT / "backend" / "execution")]
sys.modules.setdefault("backend.execution", execution_pkg)

MODULE_PATH = ROOT / "backend" / "execution" / "adaptive_controller.py"
spec = util.spec_from_file_location("backend.execution.adaptive_controller", MODULE_PATH)
module = util.module_from_spec(spec)
assert spec and spec.loader
sys.modules.setdefault("backend.execution.adaptive_controller", module)
spec.loader.exec_module(module)

MetricTunerConfig = module.MetricTunerConfig
MetricMapTuner = module.MetricMapTuner


class DummyConfig:
    def __init__(self) -> None:
        self.policy_exploration_rate = 0.5
        self.policy_learning_rate = 0.1


def _objective(exp_rate: float, lr: float) -> float:
    # Maximum around (0.2, 1e-2).
    return -((exp_rate - 0.2) ** 2) - ((math.log10(max(lr, 1e-12)) + 2.0) ** 2)


def test_metric_map_tuner_bayesian_moves_parameters() -> None:
    cfg = DummyConfig()
    initial = (cfg.policy_exploration_rate, cfg.policy_learning_rate)

    tuner_cfg = MetricTunerConfig(
        strategy="bayesian",
        cooldown=0.0,
        rollback_patience=999,
        exploration_sigma=0.0,  # ensure movement comes from BO once enough history exists
        learning_rate_sigma=0.0,
        module_toggle_prob=0.0,
    )
    tuner = MetricMapTuner(tuner_cfg, reward_fn=lambda m: m.get("reward"), rng=random.Random(0))

    for _ in range(6):
        reward = _objective(cfg.policy_exploration_rate, cfg.policy_learning_rate)
        metrics = {
            "reward": reward,
            "policy_exploration_rate": cfg.policy_exploration_rate,
            "policy_learning_rate": cfg.policy_learning_rate,
        }
        tuner.suggest(metrics, cfg)

    assert (cfg.policy_exploration_rate, cfg.policy_learning_rate) != initial
    assert 0.0 <= cfg.policy_exploration_rate <= 1.0
    assert 1e-5 <= cfg.policy_learning_rate <= 1.0

