import pytest

try:  # pragma: no cover - skip tests if torch missing
    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - torch missing
    pytest.skip("torch is required for distillation tests", allow_module_level=True)

from backend.ml.distillation import DistillationConfig, TeacherStudentTrainer


class SimpleNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def make_dataset(n: int = 128) -> DataLoader:
    torch.manual_seed(0)
    x = torch.randn(n, 10)
    y = (x.sum(dim=1) > 0).long()
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=32, shuffle=True)


def train_teacher(model: nn.Module, dataloader: DataLoader) -> None:
    optim_t = optim.SGD(model.parameters(), lr=0.1)
    for _ in range(50):
        for x, y in dataloader:
            optim_t.zero_grad()
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            loss.backward()
            optim_t.step()


def evaluate(model: nn.Module, dataloader: DataLoader) -> float:
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            preds = model(x).argmax(dim=1)
            correct += int((preds == y).sum())
            total += len(x)
    return correct / total


def test_student_improves_with_distillation() -> None:
    dataloader = make_dataset()
    teacher = SimpleNet()
    train_teacher(teacher, dataloader)
    student = SimpleNet()

    baseline_acc = evaluate(student, dataloader)

    cfg = DistillationConfig(temperature=2.0, feature_layers=["fc1"], alpha=0.1)
    trainer = TeacherStudentTrainer(teacher, student, cfg)
    trainer.train(dataloader, epochs=5, lr=0.05)

    distilled_acc = evaluate(student, dataloader)
    assert distilled_acc > baseline_acc
