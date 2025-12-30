"""Meta-learning utilities enabling fast task adaptation with MAML/Reptile."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import torch
    import torch.nn as nn
    from torch import Tensor
except Exception:  # pragma: no cover - torch is optional
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]

from modules.evolution.generic_ga import GAConfig, GeneticAlgorithm


def _require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError(
            "PyTorch is required for meta-learning components. "
            "Install torch to continue or disable meta-learning."
        )


@dataclass
class MetaLearningConfig:
    """Hyper-parameters for gradient-based meta learning."""

    meta_iterations: int = 50
    meta_batch_size: int = 4
    inner_steps: int = 5
    inner_learning_rate: float = 1e-2
    meta_learning_rate: float = 1e-3
    algorithm: str = "maml"
    meta_grad_clip: Optional[float] = None
    device: Optional[str] = None
    report_interval: int = 1
    task_adaptation_steps: int = 5
    seed: Optional[int] = None


@dataclass
class MetaTask:
    """Encapsulates a meta-learning task with support/query samplers."""

    task_id: str
    support_sampler: Callable[[], Tuple[Tensor, Tensor]]
    query_sampler: Callable[[], Tuple[Tensor, Tensor]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def sample_support(self) -> Tuple[Tensor, Tensor]:
        return self.support_sampler()

    def sample_query(self) -> Tuple[Tensor, Tensor]:
        return self.query_sampler()


@dataclass
class MetaTrainingStats:
    """Container for metrics recorded per meta-iteration."""

    iteration: int
    support_loss: float
    query_loss: float
    step_norm: float
    algorithm: str
    additional: Dict[str, Any] = field(default_factory=dict)


class MetaLearner:
    """Implements first-order MAML and Reptile style meta learning."""

    def __init__(
        self,
        model_factory: Callable[[], "nn.Module"],
        loss_fn: Callable[["nn.Module", Tuple[Tensor, Tensor]], Tensor],
        config: Optional[MetaLearningConfig] = None,
        *,
        logger: Optional[Callable[[MetaTrainingStats], None]] = None,
    ) -> None:
        _require_torch()
        self.model_factory = model_factory
        self.loss_fn = loss_fn
        self.config = config or MetaLearningConfig()
        self.logger = logger
        self.device = torch.device(self.config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            random.seed(self.config.seed)

        self.model = self.model_factory().to(self.device)
        self.meta_optimizer: Optional[torch.optim.Optimizer] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def reset_model(self) -> None:
        """Reinitialise the base model and meta-optimizer."""

        self.model = self.model_factory().to(self.device)
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.meta_learning_rate)

    def meta_train(
        self,
        tasks: Sequence[MetaTask],
        *,
        iterations: Optional[int] = None,
        callback: Optional[Callable[[MetaTrainingStats], None]] = None,
    ) -> List[MetaTrainingStats]:
        """Run the meta-training loop on the provided tasks."""

        if not tasks:
            raise ValueError("At least one task is required for meta-learning.")
        if self.meta_optimizer is None:
            self.reset_model()

        history: List[MetaTrainingStats] = []
        total_iterations = iterations or self.config.meta_iterations
        algorithm = self.config.algorithm.lower()

        for iteration in range(1, total_iterations + 1):
            batch = self._sample_task_batch(tasks)
            if algorithm == "reptile":
                stats = self._reptile_step(batch, iteration)
            else:
                stats = self._maml_step(batch, iteration)

            history.append(stats)
            if callback:
                callback(stats)
            if self.logger and (iteration % max(1, self.config.report_interval) == 0):
                self.logger(stats)

        return history

    def adapt_task(
        self,
        task: MetaTask,
        *,
        steps: Optional[int] = None,
        copy_model: bool = True,
    ) -> "nn.Module":
        """Return a task-adapted model using the current meta-parameters."""

        adapted = copy.deepcopy(self.model) if copy_model else self.model
        adapted.to(self.device)
        optimizer = torch.optim.SGD(adapted.parameters(), lr=self.config.inner_learning_rate)
        inner_steps = steps or self.config.task_adaptation_steps

        for _ in range(inner_steps):
            support = self._to_device(task.sample_support())
            loss = self.loss_fn(adapted, support)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return adapted

    def evaluate_task(self, task: MetaTask, *, steps: int = 0) -> float:
        """Evaluate the current (optionally adapted) model on a query batch."""

        if steps > 0:
            model = self.adapt_task(task, steps=steps, copy_model=True)
        else:
            model = self.model
        model.eval()
        with torch.no_grad():
            query = self._to_device(task.sample_query())
            loss = self.loss_fn(model, query)
        return float(loss.item())

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _sample_task_batch(self, tasks: Sequence[MetaTask]) -> List[MetaTask]:
        size = min(len(tasks), self.config.meta_batch_size)
        if len(tasks) <= size:
            return random.sample(tasks, size)
        return random.sample(tasks, size)

    def _clone_model(self) -> "nn.Module":
        clone = copy.deepcopy(self.model)
        clone.to(self.device)
        return clone

    def _to_device(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        x, y = batch
        return x.to(self.device), y.to(self.device)

    # ------------------------------------------------------------------ #
    # MAML step
    # ------------------------------------------------------------------ #
    def _maml_step(self, batch: Sequence[MetaTask], iteration: int) -> MetaTrainingStats:
        assert self.meta_optimizer is not None

        meta_grads = [torch.zeros_like(param) for param in self.model.parameters()]
        support_losses: List[float] = []
        query_losses: List[float] = []

        for task in batch:
            grads, support_loss, query_loss = self._maml_gradients(task)
            support_losses.append(support_loss)
            query_losses.append(query_loss)
            for agg, grad in zip(meta_grads, grads):
                agg.add_(grad)

        batch_size = float(len(batch))
        self.meta_optimizer.zero_grad()
        for param, grad in zip(self.model.parameters(), meta_grads):
            param.grad = grad / batch_size

        if self.config.meta_grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.meta_grad_clip)
        self.meta_optimizer.step()

        avg_support = float(sum(support_losses) / batch_size)
        avg_query = float(sum(query_losses) / batch_size)
        step_norm = float(math.sqrt(sum(torch.sum(param.grad**2).item() for param in self.model.parameters() if param.grad is not None)))

        return MetaTrainingStats(
            iteration=iteration,
            support_loss=avg_support,
            query_loss=avg_query,
            step_norm=step_norm,
            algorithm="maml",
        )

    def _maml_gradients(self, task: MetaTask) -> Tuple[List[Tensor], float, float]:
        model = self._clone_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.config.inner_learning_rate)
        support_losses: List[float] = []

        for _ in range(self.config.inner_steps):
            support = self._to_device(task.sample_support())
            loss = self.loss_fn(model, support)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            support_losses.append(float(loss.item()))

        query = self._to_device(task.sample_query())
        query_loss = self.loss_fn(model, query)
        grads = torch.autograd.grad(query_loss, tuple(model.parameters()))
        detached_grads = [grad.detach() for grad in grads]

        avg_support = float(sum(support_losses) / float(max(1, len(support_losses))))
        return detached_grads, avg_support, float(query_loss.item())

    # ------------------------------------------------------------------ #
    # Reptile step
    # ------------------------------------------------------------------ #
    def _reptile_step(self, batch: Sequence[MetaTask], iteration: int) -> MetaTrainingStats:
        deltas = [torch.zeros_like(param) for param in self.model.parameters()]
        support_losses: List[float] = []
        query_losses: List[float] = []

        for task in batch:
            adapted, support_loss, query_loss = self._reptile_adaptation(task)
            support_losses.append(support_loss)
            query_losses.append(query_loss)
            for delta, adapted_param, base_param in zip(deltas, adapted.parameters(), self.model.parameters()):
                delta.add_(adapted_param.data - base_param.data)

        scale = self.config.meta_learning_rate / float(len(batch))
        step_sq_sum = 0.0
        with torch.no_grad():
            for param, delta in zip(self.model.parameters(), deltas):
                update = scale * delta
                param.add_(update)
                step_sq_sum += torch.sum(update**2).item()

        avg_support = float(sum(support_losses) / float(len(batch)))
        avg_query = float(sum(query_losses) / float(len(batch)))
        step_norm = float(math.sqrt(max(step_sq_sum, 0.0)))

        return MetaTrainingStats(
            iteration=iteration,
            support_loss=avg_support,
            query_loss=avg_query,
            step_norm=step_norm,
            algorithm="reptile",
        )

    def _reptile_adaptation(self, task: MetaTask) -> Tuple["nn.Module", float, float]:
        model = self._clone_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.config.inner_learning_rate)
        support_losses: List[float] = []

        for _ in range(self.config.inner_steps):
            support = self._to_device(task.sample_support())
            loss = self.loss_fn(model, support)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            support_losses.append(float(loss.item()))

        query = self._to_device(task.sample_query())
        with torch.no_grad():
            query_loss = self.loss_fn(model, query)

        avg_support = float(sum(support_losses) / float(max(1, len(support_losses))))
        return model, avg_support, float(query_loss.item())


class MetaLearningGAAdapter:
    """Leverage the genetic algorithm utility to tune meta-learning hyperparameters."""

    def __init__(
        self,
        config_template: MetaLearningConfig,
        search_space: Dict[str, Dict[str, float]],
        evaluator: Callable[[MetaLearningConfig], float],
        *,
        ga_config: Optional[GAConfig] = None,
    ) -> None:
        self.template = config_template
        self.search_space = search_space
        self.evaluator = evaluator
        self.ga_config = ga_config or GAConfig(population_size=32, mutation_sigma=0.25)
        self._keys = list(search_space.keys())
        self._bounds = [(spec["min"], spec["max"]) for spec in search_space.values()]

    def _decode(self, genes: Sequence[float]) -> MetaLearningConfig:
        config_dict = self.template.__dict__.copy()
        for key, value, spec in zip(self._keys, genes, self.search_space.values()):
            if spec.get("type", "float") == "int":
                config_dict[key] = int(round(value))
            else:
                config_dict[key] = float(value)
        return MetaLearningConfig(**config_dict)

    def optimise(self, generations: int = 10) -> Tuple[MetaLearningConfig, float]:
        """Run the GA and return the best configuration and its score."""

        def _fitness(genes: Sequence[float]) -> float:
            config = self._decode(genes)
            score = self.evaluator(config)
            return score

        ga = GeneticAlgorithm(fitness_fn=_fitness, bounds=self._bounds, config=self.ga_config)
        best_genes, best_score = ga.run(generations)
        return self._decode(best_genes), best_score
