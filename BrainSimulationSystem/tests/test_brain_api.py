import json
import threading
import time
import types
import urllib.error
import urllib.request
from pathlib import Path
import sys
import importlib

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

pytest.importorskip("flask")

from BrainSimulationSystem.api.brain_api import BrainAPI


class DummyBrainSimulation:
    def __init__(self, config=None):
        self.config = config or {"simulation": {"dt": 1.0}}
        self.current_time = 0.0
        self.is_running = False

    def step(self, inputs, dt):
        self.current_time += dt
        return {
            "time": self.current_time,
            "network_state": {"spikes": [1], "voltages": [0.1]},
            "cognitive_state": {"status": "ok"},
        }

    def stop(self):
        self.is_running = False


def _http_get(url: str):
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_post(url: str, payload: dict):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _wait_for_server(url: str, timeout: float = 5.0):
    start = time.time()
    while time.time() - start < timeout:
        try:
            _http_get(url)
            return
        except urllib.error.URLError:
            time.sleep(0.1)
    raise TimeoutError(f"Server {url} did not become available in time")


def test_brain_api_can_run_step_and_shutdown():
    api = BrainAPI(DummyBrainSimulation(), host="127.0.0.1", port=0)
    api.run(block=False, threaded=True)

    try:
        start = time.time()
        while api.port == 0 and time.time() - start < 5.0:
            time.sleep(0.05)

        if api.port == 0:
            raise TimeoutError("API server failed to bind to a port")

        base_url = f"http://{api.host}:{api.port}"
        _wait_for_server(f"{base_url}/api/info")

        info = _http_get(f"{base_url}/api/info")
        assert info["name"] == "大脑模拟系统"
        assert info["brain_simulation"]["is_running"] is False

        step_result = _http_post(
            f"{base_url}/api/brain/step",
            {"inputs": {"foo": 1}, "dt": 0.5},
        )
        assert pytest.approx(step_result["time"], rel=1e-3) == 0.5

        status = _http_get(f"{base_url}/api/brain/status")
        assert pytest.approx(status["current_time"], rel=1e-3) == 0.5
        assert status["api_running"] is True
    finally:
        api.stop()


def test_production_system_api_mode_start_stop(monkeypatch):
    class StubBrainSimulation(DummyBrainSimulation):
        def __init__(self, config):
            super().__init__(config)

    stub_brain_module = types.ModuleType("BrainSimulationSystem.brain_simulation")
    stub_brain_module.BrainSimulation = StubBrainSimulation
    monkeypatch.setitem(sys.modules, "BrainSimulationSystem.brain_simulation", stub_brain_module)

    stub_visualizer_module = types.ModuleType(
        "BrainSimulationSystem.visualization.visualizer"
    )

    class StubVisualizer:
        def __init__(self, *args, **kwargs):
            pass

        def run(self):
            pass

        def stop(self):
            pass

    stub_visualizer_module.BrainVisualizer = StubVisualizer
    monkeypatch.setitem(
        sys.modules, "BrainSimulationSystem.visualization.visualizer", stub_visualizer_module
    )

    stub_config_module = types.ModuleType("BrainSimulationSystem.config.default_config")

    def _get_config():
        return {}

    def _update_config(config):
        return config

    stub_config_module.get_config = _get_config
    stub_config_module.update_config = _update_config
    monkeypatch.setitem(
        sys.modules,
        "BrainSimulationSystem.config.default_config",
        stub_config_module,
    )

    sys.modules.pop("BrainSimulationSystem.production_main", None)
    production_main = importlib.import_module("BrainSimulationSystem.production_main")

    SystemConfig = production_main.SystemConfig
    ProductionBrainSystem = production_main.ProductionBrainSystem

    config = SystemConfig(mode="api", host="127.0.0.1", port=0, enable_cors=False)
    system = ProductionBrainSystem(config)
    system.initialize_components({})

    assert isinstance(system.api_server, BrainAPI)

    thread = threading.Thread(target=system.run_api_mode, daemon=True)
    thread.start()

    try:
        start = time.time()
        while system.api_server.port == 0 and time.time() - start < 5.0:
            time.sleep(0.05)

        if system.api_server.port == 0:
            raise TimeoutError("Production API server failed to bind to a port")

        base_url = f"http://{system.api_server.host}:{system.api_server.port}"
        _wait_for_server(f"{base_url}/api/info")

        info = _http_get(f"{base_url}/api/info")
        assert info["brain_simulation"]["is_running"] is False

        status = _http_get(f"{base_url}/api/brain/status")
        assert status["api_running"] is True
    finally:
        system.shutdown()
        thread.join(timeout=5)

    assert not thread.is_alive()
