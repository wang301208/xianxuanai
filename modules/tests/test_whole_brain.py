import os
import sys
import types


sys.path.insert(0, os.path.abspath(os.getcwd()))


if "fastapi" not in sys.modules:
    fastapi_stub = types.ModuleType("fastapi")

    class _DummyFastAPI:
        def __init__(self, *args, **kwargs) -> None:
            self.routes: dict[str, object] = {}

        def get(self, path: str):  # pragma: no cover - simple stub
            def decorator(func):
                self.routes[f"GET {path}"] = func
                return func

            return decorator

        def post(self, path: str):  # pragma: no cover - simple stub
            def decorator(func):
                self.routes[f"POST {path}"] = func
                return func

            return decorator

    fastapi_stub.FastAPI = _DummyFastAPI
    sys.modules["fastapi"] = fastapi_stub


if "matplotlib" not in sys.modules:
    matplotlib_stub = types.ModuleType("matplotlib")
    pyplot_stub = types.ModuleType("matplotlib.pyplot")

    def _noop(*args, **kwargs):  # pragma: no cover - plotting stub
        return None

    for name in (
        "figure",
        "plot",
        "xlabel",
        "ylabel",
        "title",
        "legend",
        "tight_layout",
        "savefig",
        "close",
    ):
        setattr(pyplot_stub, name, _noop)

    matplotlib_stub.pyplot = pyplot_stub
    sys.modules["matplotlib"] = matplotlib_stub
    sys.modules["matplotlib.pyplot"] = pyplot_stub


if "yaml" not in sys.modules:
    yaml_stub = types.ModuleType("yaml")

    def _safe_load(data):  # pragma: no cover - configuration stub
        return {}

    yaml_stub.safe_load = _safe_load
    sys.modules["yaml"] = yaml_stub


if "psutil" not in sys.modules:
    psutil_stub = types.ModuleType("psutil")

    class _DummyProcess:  # pragma: no cover - minimal monitoring stub
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def cpu_percent(self, interval=None) -> float:
            return 0.0

        def memory_percent(self) -> float:
            return 0.0

    psutil_stub.Process = _DummyProcess
    psutil_stub.NoSuchProcess = RuntimeError
    psutil_stub.AccessDenied = PermissionError
    sys.modules["psutil"] = psutil_stub

from modules.brain.whole_brain import WholeBrainSimulation
from modules.brain.state import BrainCycleResult, BrainRuntimeConfig
from modules.brain.motor_cortex import MotorCortex
from modules.brain.motor.precision import PrecisionMotorSystem
from modules.brain.cerebellum import Cerebellum
from modules.brain.motor.actions import MotorCommand
from modules.brain.neuromorphic.spiking_network import NeuromorphicRunResult
from modules.brain.whole_brain_policy import CognitivePolicy, CognitiveDecision


def test_process_cycle_returns_action_and_broadcast():
    brain = WholeBrainSimulation()
    input_data = {
        "image": [1.0],
        "sound": [1.0],
        "touch": [1.0],
        "text": "good",
        "is_salient": True,
        "context": {"task": "greet", "safety": 0.5},
    }

    result = brain.process_cycle(input_data)

    assert isinstance(result, BrainCycleResult)
    assert result.metadata["executed_action"].startswith("executed")
    assert result.energy_used >= 0
    assert result.thoughts is not None
    assert result.thoughts.summary
    assert result.feeling is not None
    assert result.feeling.descriptor
    assert "curiosity_drive" in result.metrics
    assert "plan_length" in result.metrics
    assert "strategy_bias_approach" in result.metrics
    assert result.intent.plan
    assert 0.0 <= result.intent.confidence <= 1.0
    assert result.metadata["context_task"] == "greet"
    assert brain.get_decision_trace()
    assert brain.precision_motor.basal_ganglia.gating_history
    assert brain.last_motor_result is not None
    assert "motor_energy" in result.metrics
    assert "osc_amplitude" in result.metrics
    assert "novelty" in brain.last_motor_result.metadata.get("modulators", {})


def test_process_cycle_handles_nested_signals():
    brain = WholeBrainSimulation()
    input_data = {
        "image": [[0.1, 0.2], [0.3, 0.4]],
        "sound": [[0.5, 0.6, 0.7]],
        "text": "neutral",
    }

    result = brain.process_cycle(input_data)

    vision = brain.last_perception.modalities.get("vision")
    assert result.intent.intention
    assert result.energy_used >= 0
    assert result.thoughts is not None
    assert result.feeling is not None
    assert vision is not None
    assert len(vision["spike_counts"]) <= brain.max_neurons
    assert "novelty_signal" in result.metrics
    assert "motor_spike_counts" in result.metadata


def test_process_cycle_logs_invalid_signal(caplog):
    brain = WholeBrainSimulation()
    with caplog.at_level("DEBUG", logger="modules.brain.whole_brain"):
        brain.process_cycle({"image": {"bad": "data"}, "text": ""})
    assert "Unsupported sensory signal type" in caplog.text


def test_spiking_cache_respects_limit():
    brain = WholeBrainSimulation()
    brain.max_cache_size = 2
    signals = [[1.0], [1.0, 1.0], [1.0, 1.0, 1.0], [1.0] * 8]
    for signal in signals:
        brain.process_cycle({"image": signal})
    assert len(brain._spiking_cache) <= brain.max_cache_size


def test_process_cycle_latency_encoding():
    brain = WholeBrainSimulation()
    brain.neuromorphic_encoding = "latency"
    brain.encoding_time_scale = 0.5
    result = brain.process_cycle({"image": [0.2, 0.8], "text": ""})
    assert result.energy_used >= 0
    vision = brain.last_perception.modalities.get("vision")
    assert vision is not None
    assert len(vision["spike_counts"]) <= brain.max_neurons
    assert result.metrics.get("cycle_index", 0.0) >= 1.0
    assert brain.precision_motor.basal_ganglia.gating_history


def test_motor_cortex_accepts_neuromorphic_inputs():
    cortex = MotorCortex()
    precision = PrecisionMotorSystem()
    cortex.precision_system = precision
    cortex.basal_ganglia = precision.basal_ganglia
    cortex.cerebellum = Cerebellum()

    run_result = NeuromorphicRunResult(
        spike_events=[(0.0, [1, 0, 1, 0])],
        energy_used=0.42,
        idle_skipped=2,
        spike_counts=[1, 0, 1, 0],
        average_rate=[0.2, 0.0, 0.1, 0.0],
        metadata={"intention": "wave", "channels": ["observe", "approach", "withdraw", "explore"]},
    )

    plan = cortex.plan_movement(
        "wave",
        parameters={
            "neuromorphic_result": run_result,
            "modulators": {"novelty": 0.5},
            "weights": {"approach": 0.7},
        },
    )

    assert isinstance(plan.command, MotorCommand)
    assert plan.command.metadata.get("neuromorphic", {}).get("spike_counts") == [1, 0, 1, 0]

    action = cortex.execute_action(run_result)
    assert isinstance(action, str)
    assert precision.basal_ganglia.gating_history


def test_update_config_disables_metrics():
    brain = WholeBrainSimulation()
    config = BrainRuntimeConfig(metrics_enabled=False, enable_self_learning=False)
    brain.update_config(config)
    result = brain.process_cycle({"text": ""})
    assert result.metrics == {}


def test_idle_cycle_with_flagged_state_schedules_exploration_goal():
    brain = WholeBrainSimulation()
    brain.config.enable_self_learning = True

    class IdlePolicy(CognitivePolicy):
        def select_intention(  # type: ignore[override]
            self,
            perception,
            summary,
            emotion,
            personality,
            curiosity,
            context,
            learning_prediction=None,
            history=None,
        ) -> CognitiveDecision:
            return CognitiveDecision(
                intention="observe",
                confidence=0.2,
                plan=["observe_environment"],
                weights={"observe": 1.0, "approach": 0.0, "withdraw": 0.0, "explore": 0.0},
                tags=["observe"],
                focus=None,
                summary="idle",
                thought_trace=[],
                perception_summary={},
                metadata={"policy": "idle-test"},
            )

    brain.cognition.set_policy(IdlePolicy())
    state_id = "novel-state"
    brain.self_learning.memory[state_id] = {
        "sample": {"state": state_id, "agent_id": "tester", "usage": {}},
        "metadata": {
            "flagged": True,
            "rejections": 1,
            "error_history": [0.7],
            "smoothed_error": 0.7,
            "last_error": 0.7,
        },
    }
    brain.self_learning.exploration_flags.add(state_id)

    result = brain.process_cycle({"text": ""})

    scheduled_goal = f"explore:{state_id}"
    assert result.intent.intention == "explore"
    assert result.intent.plan
    assert result.intent.plan[0] == scheduled_goal
    assert brain._scheduled_exploration_goals.get(scheduled_goal) == state_id
    assert result.metadata["policy_metadata"]["exploration_goal"]["id"] == scheduled_goal


def test_meta_skill_suggestions_surface_in_policy_metadata():
    brain = WholeBrainSimulation()
    brain.config.enable_self_learning = True

    from modules.brain.meta_learning.coordinator import TaskExperience

    experience = TaskExperience(
        state_signature="test-skill",
        intention="explore",
        plan=["investigate", "record"],
        reward=0.7,
        success=True,
        cycle_index=0,
    )
    brain.meta_learning.learned_skills["explore:test-skill"] = {
        "experience": experience,
        "success_rate": 0.9,
        "skill_id": "meta::explore:test-skill",
        "successes": 3,
        "failures": 0,
        "uses": 3,
    }

    result = brain.process_cycle({"text": "observe"})

    suggestions = result.metadata.get("policy_metadata", {}).get("meta_skill_suggestions")
    assert suggestions, "meta-learning suggestions should be exposed in policy metadata"
    assert suggestions[0]["intention"] == "explore"
