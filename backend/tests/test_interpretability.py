from pathlib import Path
import os
import sys
import importlib.util
import types
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

sys.modules.setdefault("events", types.SimpleNamespace(EventBus=object))

monitoring_dir = Path(__file__).resolve().parent.parent / "monitoring"
spec_storage = importlib.util.spec_from_file_location("monitoring.storage", monitoring_dir / "storage.py")
storage_mod = importlib.util.module_from_spec(spec_storage)
sys.modules["monitoring.storage"] = storage_mod
spec_storage.loader.exec_module(storage_mod)  # type: ignore
sys.modules["monitoring"] = types.ModuleType("monitoring")
sys.modules["monitoring"].storage = storage_mod
TimeSeriesStorage = storage_mod.TimeSeriesStorage

analysis_dir = Path(__file__).resolve().parent.parent / "analysis"
spec_analysis = importlib.util.spec_from_file_location("analysis.interpretability", analysis_dir / "interpretability.py")
analysis_mod = importlib.util.module_from_spec(spec_analysis)
sys.modules["analysis.interpretability"] = analysis_mod
spec_analysis.loader.exec_module(analysis_mod)  # type: ignore
InterpretabilityAnalyzer = analysis_mod.InterpretabilityAnalyzer


def test_interpretability_tools(tmp_path: Path) -> None:
    storage = TimeSeriesStorage(tmp_path / "interp.db")
    analyzer = InterpretabilityAnalyzer(storage=storage)
    curve_path = tmp_path / "curve.png"
    analyzer.generate_learning_curve([0.1, 0.2, 0.3], str(curve_path))
    assert curve_path.exists()

    analyzer.log_failure_case("input", "output", "expected")
    report_path = tmp_path / "failures.csv"
    analyzer.export_failure_cases(str(report_path))
    assert report_path.exists()

    analyzer.log_explanation("test explanation")
    stored = storage.events("analysis.explanations")
    assert stored and stored[0]["text"] == "test explanation"
