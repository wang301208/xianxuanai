from importlib import util
from pathlib import Path
import sys
import types


ROOT = Path(__file__).resolve().parents[2]
backend_pkg = types.ModuleType("backend")
backend_pkg.__path__ = [str(ROOT / "backend")]
sys.modules.setdefault("backend", backend_pkg)
execution_pkg = types.ModuleType("backend.execution")
execution_pkg.__path__ = [str(ROOT / "backend" / "execution")]
sys.modules.setdefault("backend.execution", execution_pkg)

MODULE_PATH = ROOT / "backend" / "execution" / "improvement_experience.py"
spec = util.spec_from_file_location("backend.execution.improvement_experience", MODULE_PATH)
module = util.module_from_spec(spec)
assert spec and spec.loader
sys.modules.setdefault("backend.execution.improvement_experience", module)
spec.loader.exec_module(module)

JsonlExperienceStore = module.JsonlExperienceStore


def test_experience_store_ranks_more_successful_kind(tmp_path: Path) -> None:
    path = tmp_path / "exp.jsonl"
    store = JsonlExperienceStore(path=path, enabled=True, max_records=50, min_trials=1, score_gain_weight=0.2)
    store.record(
        {
            "time": 1.0,
            "metric": "decision_success_rate",
            "kind": "decision_exploration_boost",
            "success": False,
            "evaluated": True,
            "baseline": 0.2,
            "average": 0.21,
            "direction": "increase",
        }
    )
    store.record(
        {
            "time": 2.0,
            "metric": "decision_success_rate",
            "kind": "decision_big_brain",
            "success": True,
            "evaluated": True,
            "baseline": 0.2,
            "average": 0.35,
            "direction": "increase",
        }
    )

    ranked = store.rank_kinds(
        metric="decision_success_rate",
        kinds=["decision_exploration_boost", "decision_big_brain"],
    )
    assert ranked[0] == "decision_big_brain"

    stats = store.stats_for(
        metric="decision_success_rate",
        kinds=["decision_exploration_boost", "decision_big_brain"],
    )
    assert stats["decision_big_brain"].success_rate == 1.0
    assert stats["decision_exploration_boost"].success_rate == 0.0

