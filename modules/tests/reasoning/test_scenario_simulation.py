import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

from modules.brain.reasoning import ScenarioSimulationEngine


def test_forward_and_backward_simulation():
    engine = ScenarioSimulationEngine()

    def step_forward(state):
        s = state.copy()
        s["position"] += 1
        return s

    def step_backward(state):
        s = state.copy()
        s["position"] -= 1
        return s

    engine.add_rule("step", step_forward, step_backward)

    forward_history = engine.simulate_scenario({"position": 0}, ["step", "step"])
    assert forward_history[-1] == [{"position": 2}]

    backward_history = engine.simulate_scenario({"position": 2}, ["step", "step"], reverse=True)
    assert backward_history[-1] == [{"position": 0}]


def test_branching_scenario():
    engine = ScenarioSimulationEngine()

    def branch_forward(state):
        s = state["position"]
        return [{"position": s + 1}, {"position": s - 1}]

    def branch_backward(state):
        s = state["position"]
        return [{"position": s - 1}, {"position": s + 1}]

    def step_forward(state):
        s = state.copy()
        s["position"] += 1
        return s

    def step_backward(state):
        s = state.copy()
        s["position"] -= 1
        return s

    engine.add_rule("branch", branch_forward, branch_backward)
    engine.add_rule("step", step_forward, step_backward)

    outcomes = engine.predict_outcome({"position": 0}, ["branch", "step"])

    assert len(outcomes) == 2
    assert {"position": 2} in outcomes
    assert {"position": 0} in outcomes
