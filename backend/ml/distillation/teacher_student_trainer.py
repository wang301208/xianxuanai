from __future__ import annotations

"""Knowledge distillation utilities.

This module defines :class:`TeacherStudentTrainer` which can train a student
network from a fixed teacher network using soft targets and optional feature
matching.  The trainer also exposes helper methods for domain adaptation and
progressive knowledge accumulation.
"""

from dataclasses import dataclass
from typing import Iterable, Sequence

try:  # pragma: no cover - optional runtime dependency
    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader
except Exception:  # pragma: no cover - torch might be missing
    torch = None  # type: ignore
    nn = optim = DataLoader = None  # type: ignore


@dataclass
class DistillationConfig:
    """Configuration for :class:`TeacherStudentTrainer`.

    Attributes:
        temperature: Soften logits by this temperature during distillation.
        alpha: Weight between hard-label loss and distillation loss.
        feature_layers: Names of layers whose outputs should be matched between
            teacher and student for feature based distillation.
        feature_alpha: Weight for the feature matching loss.
    """

    temperature: float = 1.0
    alpha: float = 0.5
    feature_layers: Sequence[str] | None = None
    feature_alpha: float = 0.5


class TeacherStudentTrainer:
    """Train a student model using knowledge distilled from a teacher.

    The trainer supports the classic soft target approach as introduced by
    Hinton et al. and optional feature matching between teacher and student
    layers.  Both models are assumed to be :class:`torch.nn.Module` instances.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        config: DistillationConfig | None = None,
    ) -> None:
        if torch is None:  # pragma: no cover - runtime dependency check
            raise ImportError("torch is required for TeacherStudentTrainer")
        self.teacher = teacher.eval()  # type: ignore[assignment]
        self.student = student
        self.config = config or DistillationConfig()
        self._teacher_features: dict[str, torch.Tensor] = {}
        self._student_features: dict[str, torch.Tensor] = {}
        self._register_feature_hooks()

    # ------------------------------------------------------------------
    # Hook registration
    def _register_feature_hooks(self) -> None:
        layers = self.config.feature_layers or []
        for name in layers:
            if hasattr(self.teacher, name) and hasattr(self.student, name):
                getattr(self.teacher, name).register_forward_hook(
                    self._make_hook(self._teacher_features, name)
                )
                getattr(self.student, name).register_forward_hook(
                    self._make_hook(self._student_features, name)
                )

    @staticmethod
    def _make_hook(storage: dict[str, torch.Tensor], name: str):
        def hook(_module: nn.Module, _inp: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
            storage[name] = out.detach()

        return hook

    # ------------------------------------------------------------------
    def _distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        T = self.config.temperature
        # Soft targets from teacher
        teacher_probs = nn.functional.log_softmax(teacher_logits / T, dim=1)
        student_log_probs = nn.functional.log_softmax(student_logits / T, dim=1)
        loss_kd = nn.functional.kl_div(
            student_log_probs, teacher_probs, reduction="batchmean"
        ) * (T * T)
        if targets is not None:
            loss_ce = nn.functional.cross_entropy(student_logits, targets)
            return self.config.alpha * loss_ce + (1 - self.config.alpha) * loss_kd
        return loss_kd

    def _feature_matching_loss(self) -> torch.Tensor:
        loss = torch.tensor(0.0)
        for name in self.config.feature_layers or []:
            sf = self._student_features.get(name)
            tf = self._teacher_features.get(name)
            if sf is not None and tf is not None:
                loss = loss + nn.functional.mse_loss(sf, tf)
        return loss * self.config.feature_alpha

    # ------------------------------------------------------------------
    def train(
        self, dataloader: Iterable[tuple[torch.Tensor, torch.Tensor]], epochs: int = 1, lr: float = 1e-3
    ) -> None:
        """Train the student to mimic the teacher on a dataset.

        Args:
            dataloader: Iterable yielding ``(inputs, targets)`` batches.
            epochs: Number of training epochs.
            lr: Learning rate for the optimizer.
        """

        optimizer = optim.Adam(self.student.parameters(), lr=lr)
        for _ in range(epochs):
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                with torch.no_grad():
                    teacher_logits = self.teacher(inputs)
                student_logits = self.student(inputs)
                loss = self._distillation_loss(student_logits, teacher_logits, targets)
                if self.config.feature_layers:
                    loss = loss + self._feature_matching_loss()
                loss.backward()
                optimizer.step()

    # ------------------------------------------------------------------
    def fine_tune_on_target(
        self, dataloader: Iterable[tuple[torch.Tensor, torch.Tensor]], epochs: int = 1, lr: float = 1e-3
    ) -> None:
        """Fine-tune the student on labelled target-domain data."""

        optimizer = optim.Adam(self.student.parameters(), lr=lr)
        for _ in range(epochs):
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                logits = self.student(inputs)
                loss = nn.functional.cross_entropy(logits, targets)
                loss.backward()
                optimizer.step()

    def accumulate_knowledge(
        self,
        dataloaders: Sequence[Iterable[tuple[torch.Tensor, torch.Tensor]]],
        epochs: int = 1,
        lr: float = 1e-3,
    ) -> None:
        """Progressively train on a sequence of datasets."""

        for dl in dataloaders:
            self.fine_tune_on_target(dl, epochs=epochs, lr=lr)
