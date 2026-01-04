from modules.brain.backends import BrainSimulationSystemAdapter


def test_brain_simulation_adapter_update_config_updates_dt_and_restarts_loop() -> None:
    class DummyBrain:
        def __init__(self) -> None:
            self.is_running = True
            self.update_calls: list[dict] = []
            self.stop_calls = 0
            self.start_calls: list[float] = []

        def update_parameters(self, updates: dict) -> None:
            self.update_calls.append(dict(updates))

        def stop_continuous_simulation(self) -> None:
            self.stop_calls += 1

        def start_continuous_simulation(self, dt: float) -> None:
            self.start_calls.append(float(dt))

    adapter = BrainSimulationSystemAdapter.__new__(BrainSimulationSystemAdapter)
    adapter._brain = DummyBrain()
    adapter._dt = 100.0
    adapter._auto_background = True
    adapter._metadata = {"profile": None, "stage": None, "dt": 100.0}

    adapter.update_config(overrides={"dt": 200.0})

    assert adapter._dt == 200.0
    assert adapter._metadata["dt"] == 200.0
    assert adapter._brain.stop_calls == 1
    assert adapter._brain.start_calls and adapter._brain.start_calls[-1] == 200.0
    assert adapter._brain.update_calls
    assert adapter._brain.update_calls[-1]["simulation"]["dt"] == 200.0
