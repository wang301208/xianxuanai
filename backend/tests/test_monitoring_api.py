import importlib
import os
import sys
import types
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

sys.modules.setdefault("events", types.SimpleNamespace(EventBus=object))

monitoring_dir = Path(__file__).resolve().parent.parent / "monitoring"
monitoring_pkg = types.ModuleType("monitoring")
monitoring_pkg.__path__ = [str(monitoring_dir)]
sys.modules["monitoring"] = monitoring_pkg

analysis_dir = Path(__file__).resolve().parent.parent / "analysis"
analysis_pkg = types.ModuleType("analysis")
analysis_pkg.__path__ = [str(analysis_dir)]
sys.modules["analysis"] = analysis_pkg

api = importlib.import_module("monitoring.api")
create_app = api.create_app

evaluation = importlib.import_module("monitoring.evaluation")
EvaluationMetrics = evaluation.EvaluationMetrics

storage_mod = importlib.import_module("monitoring.storage")
TimeSeriesStorage = storage_mod.TimeSeriesStorage

analysis_mod = importlib.import_module("analysis.interpretability")
InterpretabilityAnalyzer = analysis_mod.InterpretabilityAnalyzer

from fastapi.testclient import TestClient


def test_dashboard_endpoints(tmp_path: Path) -> None:
    storage = TimeSeriesStorage(tmp_path / "mon.db")
    eval_metrics = EvaluationMetrics()
    eval_metrics.record(True, True, 0.1, "a")
    eval_metrics.record(False, True, 0.2, "b")
    analyzer = InterpretabilityAnalyzer(storage=storage)
    analyzer.log_explanation("because")

    app = create_app(storage=storage, evaluation=eval_metrics)
    client = TestClient(app)

    r = client.get("/metrics/evaluation")
    assert r.status_code == 200

    r2 = client.get("/metrics/explanations")
    assert r2.status_code == 200
