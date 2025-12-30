import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.brain.meta_learning import (
    FewShotTask,
    MAMLEngine,
    ReptileOptimizer,
)


def _make_task(slope: float) -> FewShotTask:
    """Create a simple linear regression task y = slope * x."""
    support_x = np.array([[1.0], [2.0]])
    query_x = np.array([[3.0], [4.0]])
    support_y = slope * support_x.squeeze()
    query_y = slope * query_x.squeeze()
    return FewShotTask(support_x, support_y, query_x, query_y)


def test_meta_training_and_adaptation():
    tasks = [_make_task(1.0), _make_task(2.0)]
    engine = MAMLEngine(input_dim=1, inner_lr=0.1, meta_lr=0.1, adapt_steps=1)
    history = engine.learn_to_learn(tasks, epochs=5)

    assert len(history) == 5
    assert engine.memory.records  # memory should contain entries

    new_task = _make_task(3.0)
    adapted_w, reflection = engine.fast_adapt_to_task(new_task, steps=5)
    preds = new_task.query_x @ adapted_w
    assert np.allclose(preds, new_task.query_y, atol=0.2)
    assert "loss" in reflection


def test_reptile_optimizer_updates_weights():
    tasks = [_make_task(2.0), _make_task(3.0)]
    reptile = ReptileOptimizer(inner_lr=0.5, meta_lr=0.5, adapt_steps=1)
    w_init = np.zeros(1)
    updated = reptile.meta_update(w_init, tasks)
    assert not np.allclose(updated, w_init)
