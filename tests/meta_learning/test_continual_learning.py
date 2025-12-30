import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.brain.meta_learning import (
    FewShotTask,
    FewShotAdapter,
    ContinualLearningEngine,
    cross_task_experiment,
    MAMLEngine,
)


def _make_task(slope: float) -> FewShotTask:
    support_x = np.array([[1.0], [2.0]])
    query_x = np.array([[3.0], [4.0]])
    support_y = slope * support_x.squeeze()
    query_y = slope * query_x.squeeze()
    return FewShotTask(support_x, support_y, query_x, query_y)


def test_adapter_incremental_updates():
    adapter = FewShotAdapter(inner_lr=0.1, adapt_steps=5)
    w0 = np.zeros(1)
    task1 = _make_task(1.0)
    first = adapter.adapt(w0, task1)
    task2 = _make_task(2.0)
    second_inc = adapter.adapt(w0, task2, incremental=True)
    adapter._prev = None  # reset memory
    second_scratch = adapter.adapt(w0, task2, incremental=False)
    assert not np.allclose(second_inc, second_scratch)


def test_cross_task_experiment_returns_metrics():
    tasks = [_make_task(1.0), _make_task(2.0)]
    engine = ContinualLearningEngine(input_dim=1)
    transfer, forgetting = cross_task_experiment(engine, tasks)
    assert len(transfer) == len(tasks)
    assert len(forgetting) == len(tasks)
    assert forgetting[0] == 0.0


def test_fast_adapt_uses_previous_experience():
    engine = MAMLEngine(input_dim=1)
    task1 = _make_task(1.0)
    w1, _ = engine.fast_adapt_to_task(task1, steps=5)
    task2 = _make_task(2.0)
    w2, _ = engine.fast_adapt_to_task(task2, steps=0)
    assert np.allclose(w1, w2)
