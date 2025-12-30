from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class TaskData:
    """Container for a single meta-learning task."""

    support_x: np.ndarray
    support_y: np.ndarray
    query_x: np.ndarray
    query_y: np.ndarray


def load_task(
    path: str | Path,
    test_size: float = 0.5,
    random_state: int = 42,
    k_shot: int | None = None,
    n_way: int | None = None,
) -> TaskData:
    """Load a task from ``path``.

    The CSV file must contain numeric feature columns and a ``target`` column. If
    ``k_shot`` and ``n_way`` are provided, a few-shot classification task is
    constructed by sampling ``n_way`` classes and ``k_shot`` support examples per
    class. Remaining examples for those classes form the query set. Otherwise the
    dataset is split randomly for generic regression tasks.
    """

    df = pd.read_csv(path)
    if "target" not in df.columns:
        raise ValueError("Dataset must contain a 'target' column")

    if k_shot is not None and n_way is not None:
        rng = np.random.default_rng(random_state)
        classes = df["target"].unique()
        if len(classes) < n_way:
            raise ValueError("Dataset does not contain enough classes for n_way")
        selected: Sequence[int] = rng.choice(classes, size=n_way, replace=False)

        support_x, support_y, query_x, query_y = [], [], [], []
        for c in selected:
            cls_df = df[df["target"] == c]
            if len(cls_df) <= k_shot:
                raise ValueError(
                    f"Class {c} does not have more than k_shot={k_shot} examples"
                )
            cls_df = cls_df.sample(frac=1.0, random_state=random_state)
            support = cls_df.iloc[:k_shot]
            query = cls_df.iloc[k_shot:]
            support_x.append(support.drop(columns=["target"]).values.astype(float))
            support_y.append(np.full(len(support), c, dtype=float))
            query_x.append(query.drop(columns=["target"]).values.astype(float))
            query_y.append(np.full(len(query), c, dtype=float))
        X_s = np.vstack(support_x)
        y_s = np.concatenate(support_y)
        X_q = np.vstack(query_x)
        y_q = np.concatenate(query_y)
        return TaskData(X_s, y_s, X_q, y_q)

    X = df.drop(columns=["target"]).values.astype(float)
    y = df["target"].values.astype(float)
    X_s, X_q, y_s, y_q = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return TaskData(X_s, y_s, X_q, y_q)


class MAML:
    """Minimal implementation of Model-Agnostic Meta-Learning.

    Parameters
    ----------
    input_dim:
        Dimensionality of the linear model.
    inner_lr:
        Learning rate used for task-specific adaptation.
    meta_lr:
        Learning rate used for the meta-update.
    adapt_steps:
        Number of gradient steps taken during adaptation.
    second_order:
        If ``True``, compute second-order gradients for the meta-update.
        When ``False``, the algorithm reduces to first-order MAML.
    """

    def __init__(
        self,
        input_dim: int,
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
        adapt_steps: int = 1,
        second_order: bool = False,
    ) -> None:
        self.weights = np.zeros(input_dim)
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.adapt_steps = adapt_steps
        self.second_order = second_order

    def _loss_and_grad(
        self,
        w: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        compute_hessian: bool = False,
    ) -> tuple[float, np.ndarray] | tuple[float, np.ndarray, np.ndarray]:
        preds = X @ w
        diff = preds - y
        loss = float(np.mean(diff ** 2))
        grad = 2 * X.T @ diff / len(X)
        if compute_hessian:
            hess = 2 * X.T @ X / len(X)
            return loss, grad, hess
        return loss, grad

    def adapt(self, task: TaskData) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Adapt the model to ``task``.

        Returns
        -------
        np.ndarray | tuple[np.ndarray, np.ndarray]
            The adapted weights. If ``second_order`` is enabled, also return the
            Jacobian of the adapted weights with respect to the initial weights.
        """

        w = self.weights.copy()
        if self.second_order:
            jac = np.eye(len(self.weights))
            for _ in range(self.adapt_steps):
                _, grad, hess = self._loss_and_grad(
                    w, task.support_x, task.support_y, compute_hessian=True
                )
                w -= self.inner_lr * grad
                jac = jac @ (np.eye(len(self.weights)) - self.inner_lr * hess)
            return w, jac

        for _ in range(self.adapt_steps):
            _, grad = self._loss_and_grad(w, task.support_x, task.support_y)
            w -= self.inner_lr * grad
        return w

    def meta_train(self, tasks: List[TaskData], epochs: int = 1) -> List[float]:
        """Run meta-training across all ``tasks`` for ``epochs`` iterations.

        Uses first- or second-order updates depending on ``second_order``.
        """
        history: List[float] = []
        for _ in range(epochs):
            meta_grad = np.zeros_like(self.weights)
            epoch_loss = 0.0
            for task in tasks:
                if self.second_order:
                    adapted, jac = self.adapt(task)
                    loss, grad = self._loss_and_grad(
                        adapted, task.query_x, task.query_y
                    )
                    epoch_loss += loss
                    meta_grad += jac.T @ grad
                else:
                    adapted = self.adapt(task)
                    loss, grad = self._loss_and_grad(
                        adapted, task.query_x, task.query_y
                    )
                    epoch_loss += loss
                    meta_grad += grad
            self.weights -= self.meta_lr * meta_grad / len(tasks)
            history.append(epoch_loss / len(tasks))
        return history
