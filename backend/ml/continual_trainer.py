from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch import nn, optim

from . import DEFAULT_TRAINING_CONFIG, TrainingConfig, get_model
from .feature_extractor import (
    FeatureExtractor,
    GraphFeatureExtractor,
    TimeSeriesFeatureExtractor,
)


DEFAULT_LOG_FILE = Path("data") / "new_logs.csv"


class ContinualTrainer:
    """Incrementally fine-tune models on collected experience.

    The trainer keeps track of how many samples have been processed and triggers
    training whenever a configured threshold of new samples has been reached.
    Checkpoints are written after each training run.
    """

    def __init__(
        self,
        config: TrainingConfig = DEFAULT_TRAINING_CONFIG,
        log_file: Path = DEFAULT_LOG_FILE,
        feature_type: str = "tfidf",
    ) -> None:
        self.config = config
        self.log_file = log_file
        self.trained_rows = 0
        if self.log_file.exists():
            # Account for header row
            with self.log_file.open("r", newline="") as f:
                self.trained_rows = sum(1 for _ in f) - 1
        self.pending_samples = 0
        self.step = 0

        # Internal flags for testing hooks
        self.adversarial_hook_called = False
        self.curriculum_hook_called = False
        self.ewc_hook_called = False
        self.orthogonal_hook_called = False
        self.optimizer: str | None = None
        self.scheduler: str | None = None
        self.early_stopped = False

        # State used by training strategies
        self.prev_grads: List[torch.Tensor] | None = None
        self.ewc_prev_params: List[torch.Tensor] | None = None
        self.ewc_fisher: List[torch.Tensor] | None = None
        self._ewc_penalty: torch.Tensor | None = None
        self._curriculum_weights: List[float] | None = None

        # Simple linear model and feature extractor used for fine-tuning
        self.extractor = self._create_extractor(feature_type)
        self.model: nn.Module | None = None
        self.criterion = nn.MSELoss()
        self.torch_optimizer: optim.Optimizer | None = None

        self._load_checkpoint()

    def add_sample(self, sample: Dict[str, Any]) -> None:
        """Register a new sample and trigger training if needed."""
        self.pending_samples += 1
        if self.pending_samples >= self.config.train_after_samples:
            self.train()

    def train(self) -> None:
        """Fine-tune the model on newly collected samples."""
        new_data: List[Dict[str, Any]] = []
        if self.log_file.exists():
            with self.log_file.open("r", newline="") as f:
                reader = list(csv.DictReader(f))
            new_data = reader[self.trained_rows :]
        if not new_data:
            return

        # Select optimizer and scheduler according to config
        self.optimizer = self._init_optimizer()
        self.scheduler = self.config.lr_scheduler

        # Hooks for curriculum learning and adversarial training modify the raw data
        if self.config.use_curriculum:
            self._apply_curriculum_learning(new_data)
        if self.config.use_adversarial:
            self._apply_adversarial_training(new_data)

        rewards = torch.tensor(
            [float(s.get("reward", 0.0)) * float(s.get("curriculum_weight", 1.0)) for s in new_data],
            dtype=torch.float32,
        ).unsqueeze(1)

        if isinstance(self.extractor, TimeSeriesFeatureExtractor):
            series_list = [s.get("series", []) for s in new_data]
            feats = self.extractor.fit_transform(series_list)
            inputs = torch.tensor(feats, dtype=torch.float32)
        elif isinstance(self.extractor, GraphFeatureExtractor):
            graphs = [s.get("graph") for s in new_data]
            feats = self.extractor.fit_transform(graphs)
            inputs = torch.tensor(feats, dtype=torch.float32)
        else:
            texts = []
            for s in new_data:
                joined = " ".join(str(v) for k, v in s.items() if k != "reward")
                if not any(len(tok) > 1 for tok in joined.split()):
                    joined += " filler"
                texts.append(joined)
            try:
                feats = self.extractor.transform(texts)
            except Exception:
                feats = self.extractor.fit_transform(texts)
            arr = feats.toarray() if hasattr(feats, "toarray") else feats
            inputs = torch.tensor(arr, dtype=torch.float32)

        if self.model is None or getattr(self.model, "input_dim", inputs.shape[1]) != inputs.shape[1]:
            self.model = get_model(
                self.config.model_type, input_dim=inputs.shape[1], output_dim=1
            )
            # Map configured optimizer name to implementation class
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
            self.torch_optimizer = opt_cls(
                self.model.parameters(), lr=self.config.initial_lr
            )

        assert self.model is not None
        assert self.torch_optimizer is not None

        # Compute EWC penalty if enabled
        if self.config.use_ewc:
            self._apply_ewc_regularization(inputs, rewards, update=False)

        # Split into train/validation for early stopping
        if rewards.shape[0] > 1:
            split_idx = max(1, int(rewards.shape[0] * 0.8))
        else:
            split_idx = 1
        train_inputs, val_inputs = inputs[:split_idx], inputs[split_idx:]
        train_rewards, val_rewards = rewards[:split_idx], rewards[split_idx:]
        if val_inputs.shape[0] == 0:
            val_inputs, val_rewards = train_inputs, train_rewards

        patience = self.config.early_stopping_patience
        best_val = float("inf")
        epochs_no_improve = 0

        for _ in range(100):
            self.model.train()
            preds = self.model(train_inputs)
            loss = self.criterion(preds, train_rewards)
            if self.config.use_ewc and self._ewc_penalty is not None:
                loss = loss + self._ewc_penalty
            self.torch_optimizer.zero_grad()
            loss.backward()
            if self.config.use_orthogonal:
                self._apply_orthogonal_training()
            self.torch_optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_loss = self.criterion(self.model(val_inputs), val_rewards).item()
            if val_loss < best_val - 1e-8:
                best_val = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if patience is not None and epochs_no_improve > patience:
                self.early_stopped = True
                break

        # Update EWC fisher information after training
        if self.config.use_ewc:
            self._apply_ewc_regularization(train_inputs, train_rewards, update=True)

        self.trained_rows += len(new_data)
        self.pending_samples = 0
        self.step += 1
        self._save_checkpoint()

    def _save_checkpoint(self) -> None:
        if self.model is None:
            return
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = self.config.checkpoint_dir / f"checkpoint_{self.step}.pt"
        input_dim = getattr(self.model, "in_features", getattr(self.model, "input_dim", None))
        state = {
            "model_state": self.model.state_dict(),
            "input_dim": input_dim,
            "step": self.step,
            "trained_rows": self.trained_rows,
        }
        if hasattr(self.extractor, "vectorizer"):
            state["vectorizer"] = self.extractor.vectorizer  # type: ignore[attr-defined]
        torch.save(state, ckpt_path)

    def _load_checkpoint(self) -> None:
        if not self.config.checkpoint_dir.exists():
            return
        ckpts = sorted(self.config.checkpoint_dir.glob("checkpoint_*.pt"))
        if not ckpts:
            return
        state = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
        input_dim = state.get("input_dim")
        if input_dim is not None:
            self.model = get_model(
                self.config.model_type, input_dim=input_dim, output_dim=1
            )
            self.model.load_state_dict(state["model_state"])
            self.torch_optimizer = optim.Adam(
                self.model.parameters(), lr=self.config.initial_lr
            )
        if "vectorizer" in state and hasattr(self.extractor, "vectorizer"):
            self.extractor.vectorizer = state["vectorizer"]  # type: ignore[attr-defined]
        self.step = state.get("step", 0)
        self.trained_rows = state.get("trained_rows", self.trained_rows)

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

    def _apply_adversarial_training(self, data: List[Dict[str, Any]]) -> None:
        """Generate simple adversarial examples by perturbing rewards.

        Each sample is duplicated with a small reward perturbation. This
        increases the diversity of the training signal and results in different
        model updates when the strategy is enabled.
        """
        self.adversarial_hook_called = True
        augmented = []
        for sample in data:
            adv = sample.copy()
            r = float(sample.get("reward", 0.0))
            adv["reward"] = r + 0.1 if r >= 0 else r - 0.1
            augmented.append(adv)
        data.extend(augmented)

    def _apply_curriculum_learning(self, data: List[Dict[str, Any]]) -> None:
        """Order samples by reward magnitude and assign progressive weights."""
        self.curriculum_hook_called = True
        data.sort(key=lambda s: abs(float(s.get("reward", 0.0))))
        weights = [(i + 1) / len(data) for i in range(len(data))]
        for sample, w in zip(data, weights):
            sample["curriculum_weight"] = w
        self._curriculum_weights = weights

    def _apply_ewc_regularization(
        self, inputs: torch.Tensor, targets: torch.Tensor, update: bool = False
    ) -> None:
        """Compute or update the Elastic Weight Consolidation penalty.

        When ``update`` is ``False`` the method calculates the penalty term
        based on previously stored parameters and Fisher information. When
        ``update`` is ``True`` it recomputes and stores the Fisher information
        for the provided data for use in future training iterations.
        """
        self.ewc_hook_called = True
        if self.model is None:
            return

        if not update and self.ewc_prev_params is not None and self.ewc_fisher is not None:
            penalty = torch.zeros(1)
            for p, p_old, f in zip(self.model.parameters(), self.ewc_prev_params, self.ewc_fisher):
                penalty += (f * (p - p_old) ** 2).sum()
            self._ewc_penalty = penalty.detach()

        if update:
            self.model.eval()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            self.model.zero_grad()
            loss.backward()
            fisher = [p.grad.detach() ** 2 for p in self.model.parameters()]
            self.ewc_prev_params = [p.detach().clone() for p in self.model.parameters()]
            self.ewc_fisher = fisher
            self.model.zero_grad()

    def _apply_orthogonal_training(self) -> None:
        """Project gradients to be orthogonal to previous gradients."""
        self.orthogonal_hook_called = True
        if self.model is None:
            return
        if self.prev_grads is None:
            self.prev_grads = [p.grad.detach().clone() for p in self.model.parameters() if p.grad is not None]
            return
        with torch.no_grad():
            for p, g_prev in zip(self.model.parameters(), self.prev_grads):
                if p.grad is None:
                    continue
                proj = (p.grad * g_prev).sum() / (g_prev.norm() ** 2 + 1e-8)
                p.grad -= proj * g_prev
            self.prev_grads = [p.grad.detach().clone() for p in self.model.parameters() if p.grad is not None]
