import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.ml.meta_learning.maml import MAML, TaskData

def _toy_task():
    return TaskData(
        support_x=np.array([[1.0], [2.0]]),
        support_y=np.array([2.0, 4.0]),
        query_x=np.array([[3.0], [4.0]]),
        query_y=np.array([6.0, 8.0]),
    )

def test_second_order_matches_manual_computation():
    task = _toy_task()
    maml = MAML(input_dim=1, inner_lr=0.1, meta_lr=1.0, adapt_steps=1, second_order=True)
    initial = maml.weights.copy()
    maml.meta_train([task], epochs=1)
    updated = maml.weights.copy()

    Xs, ys = task.support_x, task.support_y
    Xq, yq = task.query_x, task.query_y

    grad_s = 2 * Xs.T @ (Xs @ initial - ys) / len(Xs)
    H_s = 2 * Xs.T @ Xs / len(Xs)
    w_prime = initial - maml.inner_lr * grad_s
    grad_q = 2 * Xq.T @ (Xq @ w_prime - yq) / len(Xq)
    expected_grad = (np.eye(1) - maml.inner_lr * H_s) @ grad_q
    expected = initial - maml.meta_lr * expected_grad

    assert np.allclose(updated, expected)

def test_second_order_differs_from_first_order():
    task = _toy_task()
    maml_first = MAML(input_dim=1, inner_lr=0.1, meta_lr=1.0, adapt_steps=1, second_order=False)
    maml_second = MAML(input_dim=1, inner_lr=0.1, meta_lr=1.0, adapt_steps=1, second_order=True)
    maml_first.meta_train([task], epochs=1)
    maml_second.meta_train([task], epochs=1)
    assert not np.allclose(maml_first.weights, maml_second.weights)
