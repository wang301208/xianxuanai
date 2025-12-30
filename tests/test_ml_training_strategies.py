import csv
import sys
from pathlib import Path

import pytest
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.ml import TrainingConfig
from backend.ml.models import MLP
from backend.ml.continual_trainer import ContinualTrainer
from backend.ml.multitask_trainer import MultiTaskTrainer

try:  # pragma: no cover - optional dependency
    from lion_pytorch import Lion
except Exception:  # pragma: no cover
    Lion = None  # type: ignore


def _write_dataset(path: Path, rows: int = 5) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "target"])
        for i in range(rows):
            writer.writerow([f"sample {i}", i])


def test_continual_trainer_strategy_switch(tmp_path):
    log_file = tmp_path / "logs.csv"
    # initialise log with header
    with log_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "completion"])
        writer.writeheader()

    cfg_off = TrainingConfig(optimizer="adam", checkpoint_dir=tmp_path / "ckpt1")
    trainer = ContinualTrainer(cfg_off, log_file)
    with log_file.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "completion"])
        writer.writerow({"prompt": "q", "completion": "a"})
    trainer.train()
    assert trainer.optimizer == "adam"
    assert trainer.scheduler is None
    assert not trainer.adversarial_hook_called
    assert not trainer.curriculum_hook_called
    assert not trainer.early_stopped

    cfg_on = TrainingConfig(
        optimizer="adamw",
        lr_scheduler="linear",
        early_stopping_patience=1,
        use_adversarial=True,
        use_curriculum=True,
        use_ewc=True,
        use_orthogonal=True,
        checkpoint_dir=tmp_path / "ckpt2",
    )
    trainer_on = ContinualTrainer(cfg_on, log_file)
    with log_file.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "completion"])
        writer.writerow({"prompt": "q2", "completion": "a2"})
    trainer_on.train()
    assert trainer_on.optimizer == "adamw"
    assert trainer_on.scheduler == "linear"
    assert trainer_on.adversarial_hook_called
    assert trainer_on.curriculum_hook_called
    assert trainer_on.ewc_hook_called
    assert trainer_on.orthogonal_hook_called


def test_multitask_trainer_strategy_switch(tmp_path):
    data = tmp_path / "task.csv"
    # constant targets ensure validation loss plateaus for early stopping
    with data.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "target"])
        for _ in range(5):
            writer.writerow(["sample", 1])

    cfg_off = TrainingConfig(optimizer="adam")
    trainer = MultiTaskTrainer({"t": str(data)}, cfg_off)
    trainer.train()
    assert trainer.optimizer == "adam"
    assert trainer.scheduler is None
    assert not trainer.adversarial_hook_called
    assert not trainer.curriculum_hook_called
    assert not trainer.early_stopped

    cfg_on = TrainingConfig(
        optimizer="adam",
        lr_scheduler="cosine",
        early_stopping_patience=0,
        use_adversarial=True,
        use_curriculum=True,
    )
    trainer_on = MultiTaskTrainer({"t": str(data)}, cfg_on)
    trainer_on.train()
    assert trainer_on.optimizer == "adam"
    assert trainer_on.scheduler == "cosine"
    assert trainer_on.adversarial_hook_called
    assert trainer_on.curriculum_hook_called


def test_trainer_model_selection(tmp_path):
    data = tmp_path / "task.csv"
    _write_dataset(data)

    cfg = TrainingConfig(task_model_types={"t": "mlp"})
    trainer = MultiTaskTrainer({"t": str(data)}, cfg)
    results = trainer.train()
    model, _ = results["t"]
    assert isinstance(model, MLP)


@pytest.mark.parametrize(
    "opt_name,opt_cls",
    [
        ("adam", torch.optim.Adam),
        ("adamw", torch.optim.AdamW),
        ("lion", Lion),
    ],
)
def test_continual_trainer_optimizer_instances(tmp_path, opt_name, opt_cls):
    if opt_name == "lion" and Lion is None:
        pytest.skip("lion optimizer not available")
    log_file = tmp_path / f"logs_{opt_name}.csv"
    cfg = TrainingConfig(optimizer=opt_name, checkpoint_dir=tmp_path / f"ckpt_{opt_name}")
    trainer = ContinualTrainer(cfg, log_file)
    with log_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "reward"])
        writer.writeheader()
        writer.writerow({"text": "sample", "reward": "1"})
    trainer.train()
    assert isinstance(trainer.torch_optimizer, opt_cls)


@pytest.mark.parametrize(
    "opt_name,opt_cls",
    [
        ("adam", torch.optim.Adam),
        ("adamw", torch.optim.AdamW),
        ("lion", Lion),
    ],
)
def test_multitask_trainer_optimizer_instances(tmp_path, opt_name, opt_cls):
    if opt_name == "lion" and Lion is None:
        pytest.skip("lion optimizer not available")
    data = tmp_path / f"task_{opt_name}.csv"
    with data.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "target"])
        for i in range(5):
            writer.writerow([f"sample {i}", i])
    cfg = TrainingConfig(optimizer=opt_name)
    trainer = MultiTaskTrainer({"t": str(data)}, cfg)
    trainer.train()
    assert isinstance(trainer.torch_optimizers["t"], opt_cls)


def test_unsupported_optimizer_raises(tmp_path):
    log_file = tmp_path / "logs.csv"
    cfg = TrainingConfig(optimizer="sgd", checkpoint_dir=tmp_path / "ckpt")
    trainer = ContinualTrainer(cfg, log_file)
    with log_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "reward"])
        writer.writeheader()
        writer.writerow({"text": "sample", "reward": "1"})
    with pytest.raises(ValueError):
        trainer.train()


def test_adversarial_training_changes_weights(tmp_path):
    base_file = tmp_path / "adv_base.csv"
    with base_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "reward"])
        writer.writeheader()
    torch.manual_seed(0)
    cfg_base = TrainingConfig(checkpoint_dir=tmp_path / "ckpt_base")
    trainer_base = ContinualTrainer(cfg_base, base_file)
    with base_file.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "reward"])
        writer.writerow({"text": "a", "reward": 1})
        writer.writerow({"text": "b", "reward": 1})
    trainer_base.train()
    base_weights = {
        k: v.detach().clone() for k, v in trainer_base.model.state_dict().items()  # type: ignore
    }

    adv_file = tmp_path / "adv_adv.csv"
    with adv_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "reward"])
        writer.writeheader()
    torch.manual_seed(0)
    cfg_adv = TrainingConfig(use_adversarial=True, checkpoint_dir=tmp_path / "ckpt_adv")
    trainer_adv = ContinualTrainer(cfg_adv, adv_file)
    with adv_file.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "reward"])
        writer.writerow({"text": "a", "reward": 1})
        writer.writerow({"text": "b", "reward": 1})
    trainer_adv.train()
    adv_weights = trainer_adv.model.state_dict()  # type: ignore

    assert any(
        not torch.allclose(base_weights[k], adv_weights[k]) for k in base_weights
    )


def test_curriculum_learning_changes_weights(tmp_path):
    base_file = tmp_path / "cur_base.csv"
    with base_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "reward"])
        writer.writeheader()
    torch.manual_seed(0)
    trainer_base = ContinualTrainer(TrainingConfig(checkpoint_dir=tmp_path / "ckpt_b"), base_file)
    with base_file.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "reward"])
        writer.writerow({"text": "easy", "reward": 0.1})
        writer.writerow({"text": "hard", "reward": 1.0})
    trainer_base.train()
    base_weights = {
        k: v.detach().clone() for k, v in trainer_base.model.state_dict().items()  # type: ignore
    }

    cur_file = tmp_path / "cur_cur.csv"
    with cur_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "reward"])
        writer.writeheader()
    torch.manual_seed(0)
    cfg_cur = TrainingConfig(use_curriculum=True, checkpoint_dir=tmp_path / "ckpt_c")
    trainer_cur = ContinualTrainer(cfg_cur, cur_file)
    with cur_file.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "reward"])
        writer.writerow({"text": "easy", "reward": 0.1})
        writer.writerow({"text": "hard", "reward": 1.0})
    trainer_cur.train()
    cur_weights = trainer_cur.model.state_dict()  # type: ignore

    assert any(
        not torch.allclose(base_weights[k], cur_weights[k]) for k in base_weights
    )


def test_ewc_reduces_weight_change(tmp_path):
    log1 = tmp_path / "ewc1.csv"
    log2 = tmp_path / "ewc2.csv"
    for lf in (log1, log2):
        with lf.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "reward"])
            writer.writeheader()
    torch.manual_seed(0)
    tr_no = ContinualTrainer(TrainingConfig(checkpoint_dir=tmp_path / "ckpt_no"), log1)
    with log1.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "reward"])
        writer.writerow({"text": "a", "reward": 1})
    tr_no.train()
    w_before = {
        k: v.detach().clone() for k, v in tr_no.model.state_dict().items()  # type: ignore
    }
    with log1.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "reward"])
        writer.writerow({"text": "b", "reward": 2})
    tr_no.train()
    w_after = tr_no.model.state_dict()  # type: ignore
    change_no = sum((w_before[k] - w_after[k]).pow(2).sum() for k in w_before)

    # trainer with EWC
    torch.manual_seed(0)
    tr_ewc = ContinualTrainer(
        TrainingConfig(use_ewc=True, checkpoint_dir=tmp_path / "ckpt_ewc"), log2
    )
    with log2.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "reward"])
        writer.writerow({"text": "a", "reward": 1})
    tr_ewc.train()
    w_before_e = {
        k: v.detach().clone() for k, v in tr_ewc.model.state_dict().items()  # type: ignore
    }
    with log2.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "reward"])
        writer.writerow({"text": "b", "reward": 2})
    tr_ewc.train()
    w_after_e = tr_ewc.model.state_dict()  # type: ignore
    change_ewc = sum((w_before_e[k] - w_after_e[k]).pow(2).sum() for k in w_before_e)

    assert change_ewc <= change_no


def test_orthogonal_training_orthogonalizes_gradients(tmp_path):
    # baseline without orthogonal training
    base_file = tmp_path / "ortho_base.csv"
    with base_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "reward"])
        writer.writeheader()
    trainer_no = ContinualTrainer(
        TrainingConfig(checkpoint_dir=tmp_path / "ckpt_no", train_after_samples=1), base_file
    )
    with base_file.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "reward"])
        writer.writerow({"text": "a", "reward": 1})
    trainer_no.train()
    grad1_no = [p.grad.detach().clone() for p in trainer_no.model.parameters() if p.grad is not None]  # type: ignore
    with base_file.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "reward"])
        writer.writerow({"text": "b", "reward": 2})
    trainer_no.train()
    grad2_no = [p.grad.detach().clone() for p in trainer_no.model.parameters() if p.grad is not None]  # type: ignore
    dot_no = sum((g1 * g2).sum() for g1, g2 in zip(grad1_no, grad2_no))

    # orthogonal training
    log_file = tmp_path / "ortho.csv"
    with log_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "reward"])
        writer.writeheader()
    cfg = TrainingConfig(use_orthogonal=True, checkpoint_dir=tmp_path / "ckpt_o", train_after_samples=1)
    torch.manual_seed(0)
    trainer = ContinualTrainer(cfg, log_file)
    with log_file.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "reward"])
        writer.writerow({"text": "a", "reward": 1})
    trainer.train()
    grad1 = [g.clone() for g in trainer.prev_grads]
    with log_file.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "reward"])
        writer.writerow({"text": "b", "reward": 2})
    trainer.train()
    grad2 = trainer.prev_grads
    dot = sum((g1 * g2).sum() for g1, g2 in zip(grad1, grad2))
    assert torch.abs(dot) < torch.abs(dot_no)
