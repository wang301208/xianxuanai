from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from .config import TransformerBrainConfig
from .transformer_brain import TransformerBrain


def _ensure_dim(values: Iterable[float], dim: int) -> torch.Tensor:
    tensor = torch.as_tensor(list(values), dtype=torch.float32)
    if tensor.numel() < dim:
        pad = torch.zeros(dim - tensor.numel(), dtype=torch.float32)
        tensor = torch.cat([tensor, pad], dim=0)
    elif tensor.numel() > dim:
        tensor = tensor[:dim]
    return tensor


class ObservationActionDataset(Dataset):
    """Dataset mapping observations (and optional memory) to action indices."""

    def __init__(
        self,
        observations: torch.Tensor,
        memory: torch.Tensor,
        actions: torch.Tensor,
    ) -> None:
        if not (len(observations) == len(memory) == len(actions)):
            raise ValueError("Observation, memory and action tensors must align")
        self.observations = observations
        self.memory = memory
        self.actions = actions.long()

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.actions)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.observations[index], self.memory[index], self.actions[index]

    @classmethod
    def random(cls, size: int, dim: int) -> "ObservationActionDataset":
        observations = torch.randn(size, dim)
        memory = torch.zeros(size, dim)
        actions = torch.randint(0, dim, (size,), dtype=torch.long)
        return cls(observations, memory, actions)

    @classmethod
    def from_jsonl(cls, path: Path | str, dim: int) -> "ObservationActionDataset":
        observations: list[torch.Tensor] = []
        memory: list[torch.Tensor] = []
        actions: list[int] = []

        with Path(path).expanduser().open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue

                obs = payload.get("observation")
                mem = payload.get("memory")
                action_index = payload.get("action_index")

                if obs is None or action_index is None:
                    logits = payload.get("action_logits")
                    if logits:
                        action_index = int(torch.as_tensor(logits).argmax().item())
                    else:
                        continue

                observations.append(_ensure_dim(obs or [], dim))
                memory.append(_ensure_dim(mem or [], dim))
                actions.append(int(action_index))

        if not observations:
            raise ValueError("No usable samples found in dataset")

        return cls(
            torch.stack(observations),
            torch.stack(memory),
            torch.tensor(actions, dtype=torch.long),
        )


def save_brain(brain: TransformerBrain, path: str | Path) -> None:
    """Persist model weights to ``path``."""

    torch.save(brain.state_dict(), str(path))


def load_brain(config: TransformerBrainConfig, path: str | Path) -> TransformerBrain:
    """Load model weights from ``path`` into a new ``TransformerBrain`` instance."""

    brain = TransformerBrain(config)
    brain.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    return brain


def train(
    brain: TransformerBrain,
    dataset: Dataset,
    *,
    learning_rate: float | None = None,
    epochs: int | None = None,
) -> TransformerBrain:
    """Train ``brain`` on ``dataset`` using cross-entropy loss."""

    config = brain.config
    lr = learning_rate or config.learning_rate
    num_epochs = epochs or config.epochs
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(brain.parameters(), lr=lr)

    brain.train()
    for _ in range(num_epochs):
        for observations, memory_ctx, target in dataloader:
            optimizer.zero_grad()
            logits = []
            for obs, mem in zip(observations, memory_ctx):
                thought = brain.think(obs, mem)
                logits.append(brain.action_head(thought))
            stacked = torch.stack(logits)
            loss = criterion(stacked, target)
            loss.backward()
            optimizer.step()
    return brain


def build_dataset(args, config: TransformerBrainConfig) -> ObservationActionDataset:
    if args.dataset:
        return ObservationActionDataset.from_jsonl(args.dataset, config.dim)
    return ObservationActionDataset.random(args.synthetic_samples, config.dim)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the internal transformer brain")
    parser.add_argument("--dataset", type=Path, help="JSONL dataset recorded from agent runs")
    parser.add_argument("--output", type=Path, default=Path("transformer_brain.pth"))
    parser.add_argument("--synthetic-samples", type=int, default=256,
                        help="Number of synthetic samples when no dataset is provided")
    parser.add_argument("--dim", type=int, help="Override model dimension")
    parser.add_argument("--layers", type=int, help="Override number of encoder layers")
    parser.add_argument("--heads", type=int, help="Override number of attention heads")
    parser.add_argument("--dropout", type=float, help="Override dropout probability")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, help="Training learning rate")
    parser.add_argument("--batch-size", type=int, help="Batch size to use during training")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - CLI wrapper
    args = parse_args(argv)
    config = TransformerBrainConfig()

    if args.dim:
        config.dim = args.dim
    if args.layers:
        config.layers = args.layers
    if args.heads:
        config.heads = args.heads
    if args.dropout is not None:
        config.dropout = args.dropout
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.epochs:
        config.epochs = args.epochs

    dataset = build_dataset(args, config)
    brain = TransformerBrain(config)
    train(brain, dataset, learning_rate=args.learning_rate, epochs=args.epochs)
    save_brain(brain, args.output)


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
