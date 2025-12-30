import os
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.reasoning.planner import ReasoningPlanner
from backend.reasoning.solvers import NeuroSymbolicSolver


class EmptySource:
    def query(self, statement: str):
        return []


def test_neuro_symbolic_solver_merges_neural_and_symbolic():
    def model(statement: str, evidence):
        return {"B": 0.4}

    solver = NeuroSymbolicSolver(model, {"A": "C"})
    conclusion, prob = solver.infer("A", [])

    assert conclusion == "C"
    assert prob == 1.0


def test_planner_selects_solver_via_config(tmp_path):
    weights = {"A": {"B": 0.6}}
    kb = {"A": "C"}

    weights_path = tmp_path / "weights.json"
    kb_path = tmp_path / "kb.json"
    weights_path.write_text(json.dumps(weights))
    kb_path.write_text(json.dumps(kb))

    config = {"name": "neuro_symbolic", "params": {}}
    planner = ReasoningPlanner([EmptySource()], solver_config=config)

    assert isinstance(planner.solver, NeuroSymbolicSolver)

    planner.solver.load_neural_model(str(weights_path))
    planner.solver.load_symbolic_kb(str(kb_path))

    conclusion, prob = planner.infer("A")

    assert conclusion == "C"
    assert prob == 1.0

