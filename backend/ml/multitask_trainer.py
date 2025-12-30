from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, List

import pandas as pd
import torch
from torch import nn, optim
from sklearn.model_selection import train_test_split

try:  # Optional dependency for graph parsing
    import networkx as nx
except Exception:  # pragma: no cover - dependency may be missing at runtime
    nx = None  # type: ignore

from . import DEFAULT_TRAINING_CONFIG, TrainingConfig, get_model
from .feature_extractor import (
    FeatureExtractor,
    GraphFeatureExtractor,
    TimeSeriesFeatureExtractor,
)


class MultiTaskTrainer:
    """Train separate models for multiple tasks with a shared encoder.

    Each task dataset must be a CSV file containing ``text`` and ``target`` columns.
    The :class:`FeatureExtractor` is fitted on the union of all task texts and
    reused for every individual model.
    """

    def __init__(
        self,
        tasks: Dict[str, str],
        config: TrainingConfig = DEFAULT_TRAINING_CONFIG,
        feature_type: str = "tfidf",
    ):
        # Map task name to dataset path
        self.tasks = {name: Path(path) for name, path in tasks.items()}
        self.config = config
        self.extractor = self._create_extractor(feature_type)
        self._datasets: Dict[str, Tuple[list[str], list[float]]] = {}

        # Internal flags for testing
        self.adversarial_hook_called = False
        self.curriculum_hook_called = False
        self.ewc_hook_called = False
        self.orthogonal_hook_called = False
        self.optimizer: str | None = None
        self.scheduler: str | None = None
        self.early_stopped = False
        # Store optimizers used per task for testing/inspection
        self.torch_optimizers: Dict[str, optim.Optimizer] = {}

        # Strategy state per task
        self.prev_grads: Dict[str, List[torch.Tensor]] = {}
        self.ewc_prev_params: Dict[str, List[torch.Tensor]] = {}
        self.ewc_fisher: Dict[str, List[torch.Tensor]] = {}
        self._ewc_penalty: Dict[str, torch.Tensor] = {}
        self._curriculum_weights: Dict[str, List[float]] = {}

    def load_datasets(self) -> None:
        """Load datasets and fit the shared feature extractor."""
        texts: list[str] = []
        for name, path in self.tasks.items():
            df = pd.read_csv(path)
            if "text" not in df.columns or "target" not in df.columns:
                raise ValueError(
                    f"Dataset {path} must contain 'text' and 'target' columns"
                )
            raw_texts = df["text"].astype(str).tolist()
            targets = df["target"].values
            if isinstance(self.extractor, TimeSeriesFeatureExtractor):
                data = [list(map(float, t.split())) for t in raw_texts]
            elif isinstance(self.extractor, GraphFeatureExtractor):
                if nx is None:
                    raise ImportError("networkx is required for graph features")
                data = []
                for t in raw_texts:
                    edges = [tuple(e.split("-")) for e in t.split()]
                    g = nx.Graph()
                    g.add_edges_from(edges)
                    data.append(g)
            else:
                data = raw_texts
                texts.extend(raw_texts)
            self._datasets[name] = (data, targets)
        if texts and isinstance(self.extractor, FeatureExtractor):
            # ``FeatureExtractor`` exposes ``fit_transform`` instead of ``fit``
            # for simplicity. We only need the fitted vocabulary here.
            self.extractor.fit_transform(texts)

    def train(self) -> Dict[str, Tuple[nn.Module, float]]:
        """Train a model for each task and return metrics."""
        if not self._datasets:
            self.load_datasets()

        self.optimizer = self._init_optimizer()
        self.scheduler = self.config.lr_scheduler

        results: Dict[str, Tuple[nn.Module, float]] = {}
        for name, (data, targets) in self._datasets.items():
            pairs = list(zip(list(data), list(targets)))
            if self.config.use_curriculum:
                self._apply_curriculum_learning(name, pairs)
            if self.config.use_adversarial:
                self._apply_adversarial_training(name, pairs)
            data, targets = zip(*pairs)
            weights = self._curriculum_weights.get(name, [1.0] * len(data))

            if isinstance(self.extractor, TimeSeriesFeatureExtractor):
                X = self.extractor.fit_transform(list(data))
            elif isinstance(self.extractor, GraphFeatureExtractor):
                X = self.extractor.fit_transform(list(data))
            else:
                X = self.extractor.transform(list(data))
            X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
                X, targets, weights, test_size=0.2, random_state=42
            )

            X_train_t = torch.tensor(
                X_train.toarray() if hasattr(X_train, "toarray") else X_train,
                dtype=torch.float32,
            )
            y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
            w_train_t = torch.tensor(w_train, dtype=torch.float32).unsqueeze(1)
            y_train_t = y_train_t * w_train_t
            X_test_t = torch.tensor(
                X_test.toarray() if hasattr(X_test, "toarray") else X_test,
                dtype=torch.float32,
            )
            y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

            model_type = (
                self.config.task_model_types.get(name)
                if self.config.task_model_types
                else self.config.model_type
            )
            model = get_model(model_type, input_dim=X_train_t.shape[1], output_dim=1)
            optimizer_map = {
                "adam": optim.Adam,
                "adamw": optim.AdamW,
            }
            if hasattr(optim, "Lion"):
                optimizer_map["lion"] = optim.Lion  # type: ignore[attr-defined]
            else:  # pragma: no cover - optional dependency
                try:
                    from lion_pytorch import Lion  # type: ignore

                    optimizer_map["lion"] = Lion
                except Exception:  # pragma: no cover - dependency may be missing
                    pass

            if self.optimizer not in optimizer_map:
                raise ValueError(f"Unsupported optimizer: {self.optimizer}")
            opt_cls = optimizer_map[self.optimizer]
            optimizer = opt_cls(model.parameters(), lr=self.config.initial_lr)
            self.torch_optimizers[name] = optimizer
            criterion = nn.MSELoss()

            patience = self.config.early_stopping_patience
            best_val = float("inf")
            epochs_no_improve = 0

            for _ in range(100):
                model.train()
                preds = model(X_train_t)
                loss = criterion(preds, y_train_t)
                if self.config.use_ewc:
                    self._apply_ewc_regularization(name, model)
                    if name in self._ewc_penalty:
                        loss = loss + self._ewc_penalty[name]
                optimizer.zero_grad()
                loss.backward()
                if self.config.use_orthogonal:
                    self._apply_orthogonal_training(name, model)
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    val_loss = criterion(model(X_test_t), y_test_t).item()
                if val_loss < best_val - 1e-8:
                    best_val = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if patience is not None and epochs_no_improve > patience:
                    self.early_stopped = True
                    break

            if self.config.use_ewc:
                self._apply_ewc_regularization(
                    name, model, X_train_t, y_train_t, update=True
                )

            model.eval()
            with torch.no_grad():
                mse = criterion(model(X_test_t), y_test_t).item()
            results[name] = (model, mse)
        return results

    def _create_extractor(self, feature_type: str):
        if feature_type == "sentence":
            return FeatureExtractor(method="sentence")
        if feature_type == "time_series":
            return TimeSeriesFeatureExtractor()
        if feature_type == "graph":
            return GraphFeatureExtractor()
        return FeatureExtractor()

    # ---- Hooks ---------------------------------------------------------

    def _init_optimizer(self) -> str:
        """Return the configured optimizer name in lowercase."""
        return self.config.optimizer.lower()

    def _apply_adversarial_training(self, name: str, pairs: List[Tuple[Any, float]]) -> None:
        """Augment training pairs with perturbed targets."""
        self.adversarial_hook_called = True
        augmented: List[Tuple[Any, float]] = []
        for d, t in pairs:
            augmented.append((d, t + 0.1))
        pairs.extend(augmented)
        if name in self._curriculum_weights:
            self._curriculum_weights[name].extend(self._curriculum_weights[name])

    def _apply_curriculum_learning(self, name: str, pairs: List[Tuple[Any, float]]) -> None:
        """Sort samples by target magnitude and store weights."""
        self.curriculum_hook_called = True
        pairs.sort(key=lambda p: abs(float(p[1])))
        weights = [(i + 1) / len(pairs) for i in range(len(pairs))]
        self._curriculum_weights[name] = weights

    def _apply_ewc_regularization(
        self,
        name: str,
        model: nn.Module,
        inputs: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        update: bool = False,
    ) -> None:
        """Compute or update EWC penalty for a task."""
        self.ewc_hook_called = True
        if name in self.ewc_prev_params and name in self.ewc_fisher and not update:
            penalty = torch.zeros(1)
            for p, p_old, f in zip(
                model.parameters(), self.ewc_prev_params[name], self.ewc_fisher[name]
            ):
                penalty += (f * (p - p_old) ** 2).sum()
            self._ewc_penalty[name] = penalty.detach()
        if update and inputs is not None and targets is not None:
            model.eval()
            outputs = model(inputs)
            loss = nn.MSELoss()(outputs, targets)
            model.zero_grad()
            loss.backward()
            fisher = [p.grad.detach() ** 2 for p in model.parameters()]
            self.ewc_prev_params[name] = [p.detach().clone() for p in model.parameters()]
            self.ewc_fisher[name] = fisher
            model.zero_grad()

    def _apply_orthogonal_training(self, name: str, model: nn.Module) -> None:
        """Project gradients to be orthogonal to previous ones for a task."""
        self.orthogonal_hook_called = True
        grads = self.prev_grads.get(name)
        if grads is None:
            self.prev_grads[name] = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]
            return
        with torch.no_grad():
            for p, g_prev in zip(model.parameters(), grads):
                if p.grad is None:
                    continue
                proj = (p.grad * g_prev).sum() / (g_prev.norm() ** 2 + 1e-8)
                p.grad -= proj * g_prev
            self.prev_grads[name] = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]
