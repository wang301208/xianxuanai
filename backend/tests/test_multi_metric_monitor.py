from pathlib import Path
import importlib.util
import os
import time

MODULE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "monitoring", "multi_metric_monitor.py"))
spec = importlib.util.spec_from_file_location("multi_metric_monitor", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
import sys
sys.modules["multi_metric_monitor"] = module
spec.loader.exec_module(module)  # type: ignore
MultiMetricMonitor = module.MultiMetricMonitor


def test_multi_metric_monitor(tmp_path: Path) -> None:
    monitor = MultiMetricMonitor()
    monitor.log_training(0.5)
    monitor.log_inference(0.6, step=1)
    monitor.log_resource()
    # ensure some time passes for plot uniqueness
    time.sleep(0.01)
    paths = monitor.plot(str(tmp_path))
    assert len(paths) == 3
    for p in paths:
        assert Path(p).exists()
