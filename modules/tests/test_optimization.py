import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from optimization import log_run, optimize_params


def test_optimization_flow_and_persistence(tmp_path):
    history = tmp_path / "history.csv"
    search_space = {"x": [1, 2, 3], "y": [10, 20]}

    log_run("algo", {"x": 1, "y": 10}, {"score": 0.5}, history)
    log_run("algo", {"x": 2, "y": 20}, {"score": 0.8}, history)

    params = optimize_params("algo", search_space, history_file=history)
    assert params == {"x": 2, "y": 20}

    with history.open() as f:
        lines = f.readlines()
    assert len(lines) == 3  # header + two entries

    # When requesting params for new algorithm it samples from search space
    new_params = optimize_params("new_algo", search_space, history_file=history)
    assert new_params["x"] in search_space["x"]
    assert new_params["y"] in search_space["y"]
