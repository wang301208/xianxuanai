from BrainSimulationSystem.api.streaming import StreamingHub
from BrainSimulationSystem.core.cognitive_services import CognitiveControllerFactory
from BrainSimulationSystem.core.simulation_orchestrator import SimulationOrchestrator


def test_cognitive_controller_factory_creates_controller():
    factory = CognitiveControllerFactory()
    controller = factory.create()

    assert "attention" in controller.components
    assert "working_memory" in controller.components
    assert hasattr(controller, "process")
    assert controller.state.name

    overrides = {"attention": {"attention_span": 5}}
    controller_with_overrides = factory.create(overrides)
    attention = controller_with_overrides.components["attention"]
    assert getattr(attention, "attention_span") == 5


def test_streaming_hub_broadcast_creates_events():
    hub = StreamingHub()
    queue = hub.register_client()
    hub.broadcast("test.event", {"value": 1})
    message = queue.get()
    assert message["event"] == "test.event"
    assert message["data"] == {"value": 1}
    assert isinstance(message["timestamp"], float)
    hub.unregister_client(queue)


def test_simulation_orchestrator_runs_steps():
    class StubController:
        def __init__(self):
            self.calls = []

        def process(self, payload):
            self.calls.append(payload)
            return {"step": len(self.calls)}

    controller = StubController()
    orchestrator = SimulationOrchestrator(controller)

    result = orchestrator.start(steps=3, interval=0)
    assert result["status"] == "started"

    orchestrator.join(timeout=1)

    status = orchestrator.status()
    assert status["results_count"] == 3
    assert orchestrator.results[-1]["step"] == 3

    stop_result = orchestrator.stop()
    assert stop_result["results_count"] == 3
