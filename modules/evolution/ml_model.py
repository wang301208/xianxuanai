"""Deep learning model for predicting system resource usage with RL strategy."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
try:  # pragma: no cover - optional torch dependency
    import torch  # type: ignore
    from torch import nn, optim  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = optim = None  # type: ignore

from .resource_rl import ResourceRL

logger = logging.getLogger(__name__)

if nn is not None:
    class _ResourceNet(nn.Module):
        """Simple two-headed network predicting CPU and memory usage."""

        def __init__(self) -> None:
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, 16),
                nn.ReLU(),
            )
            self.cpu_head = nn.Linear(16, 1)
            self.mem_head = nn.Linear(16, 1)

        def forward(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:  # type: ignore[override]
            x = self.shared(x)
            return self.cpu_head(x), self.mem_head(x)


class ResourceModel:
    """Train models on historical metrics and predict future usage."""

    def __init__(
        self,
        data_path: Path | str | None = None,
        rl_agent: Optional[ResourceRL] = None,
    ) -> None:
        self.data_path = (
            Path(data_path) if data_path is not None else Path(__file__).with_name("metrics_history.csv")
        )
        self.model_path = self.data_path.with_suffix(".pt")
        self.net = _ResourceNet() if nn is not None else None
        self.criterion = nn.MSELoss() if nn is not None else None
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.01) if nn is not None else None
        self.rl_agent = rl_agent or ResourceRL()
        self._trained = False
        self._coef_cpu: tuple[float, float] | None = None
        self._coef_mem: tuple[float, float] | None = None

    # ------------------------------------------------------------------
    def _load(self) -> np.ndarray:
        data: list[tuple[float, float]] = []
        try:
            with open(self.data_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data.append((float(row["cpu_percent"]), float(row["memory_percent"])))
        except FileNotFoundError:
            return np.empty((0, 2))
        return np.array(data)

    # ------------------------------------------------------------------
    def train(self, epochs: int = 100) -> None:
        """Train the neural network on historical data."""

        data = self._load()
        if data.size == 0:
            logger.warning("No historical data found for training")
            self._trained = False
            return
        if torch is None or self.net is None or self.criterion is None or self.optimizer is None:
            self._train_numpy(data)
            self._trained = True
            return

        x = torch.arange(len(data), dtype=torch.float32).unsqueeze(1)
        y_cpu = torch.tensor(data[:, 0], dtype=torch.float32).unsqueeze(1)
        y_mem = torch.tensor(data[:, 1], dtype=torch.float32).unsqueeze(1)
        self.net.train()
        loss = None
        for _ in range(epochs):
            cpu_pred, mem_pred = self.net(x)
            loss = self.criterion(cpu_pred, y_cpu) + self.criterion(mem_pred, y_mem)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if loss is not None:
            logger.info("Resource model trained; final loss %.4f", float(loss))
        try:  # pragma: no cover - filesystem dependent
            torch.save(self.net.state_dict(), self.model_path)
        except Exception:
            pass
        self._trained = True

    # ------------------------------------------------------------------
    def load(self) -> None:
        """Load model weights if available."""

        if torch is not None and self.net is not None and self.model_path.exists():
            try:
                self.net.load_state_dict(torch.load(self.model_path, map_location="cpu"))
                self._trained = True
                return
            except Exception:
                pass
        # numpy fallback loads coefficients from a sidecar json if present
        coef_path = self.model_path.with_suffix(".json")
        if coef_path.exists():
            try:
                data = json.loads(coef_path.read_text(encoding="utf-8"))
                cpu = data.get("cpu")
                mem = data.get("mem")
                self._coef_cpu = (float(cpu[0]), float(cpu[1])) if isinstance(cpu, list) and len(cpu) == 2 else None
                self._coef_mem = (float(mem[0]), float(mem[1])) if isinstance(mem, list) and len(mem) == 2 else None
                if self._coef_cpu and self._coef_mem:
                    self._trained = True
            except Exception:
                self._trained = False

    # ------------------------------------------------------------------
    def predict_next(self) -> Dict[str, float]:
        """Predict next CPU and memory usage values and propose action."""

        if not self._trained:
            self.load()
        if not self._trained:
            self.train()
        if not self._trained:
            return {}

        with open(self.data_path, encoding="utf-8") as f:
            data_len = sum(1 for _ in f) - 1  # exclude header

        if torch is not None and self.net is not None:
            x = torch.tensor([[float(data_len)]])
            self.net.eval()
            with torch.no_grad():
                cpu_pred, mem_pred = self.net(x)
            cpu_val = float(cpu_pred.item())
            mem_val = float(mem_pred.item())
        else:
            cpu_val, mem_val = self._predict_numpy(float(data_len))

        action = self.rl_agent.select_action(cpu_val, mem_val)
        logger.info("Predicted CPU %.2f, MEM %.2f, action %s", cpu_val, mem_val, action)
        return {"cpu_percent": cpu_val, "memory_percent": mem_val, "action": action}

    def _train_numpy(self, data: np.ndarray) -> None:
        indices = np.arange(len(data), dtype=np.float32)
        cpu = data[:, 0].astype(np.float32)
        mem = data[:, 1].astype(np.float32)

        self._coef_cpu = _fit_line(indices, cpu)
        self._coef_mem = _fit_line(indices, mem)

        coef_path = self.model_path.with_suffix(".json")
        try:  # pragma: no cover - filesystem dependent
            coef_path.write_text(
                json.dumps(
                    {"cpu": list(self._coef_cpu or (0.0, 0.0)), "mem": list(self._coef_mem or (0.0, 0.0))},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _predict_numpy(self, x: float) -> tuple[float, float]:
        data = self._load()
        if data.size == 0:
            return 0.0, 0.0
        if self._coef_cpu is None or self._coef_mem is None:
            indices = np.arange(len(data), dtype=np.float32)
            self._coef_cpu = _fit_line(indices, data[:, 0].astype(np.float32))
            self._coef_mem = _fit_line(indices, data[:, 1].astype(np.float32))
        cpu_a, cpu_b = self._coef_cpu
        mem_a, mem_b = self._coef_mem
        return float(cpu_a * x + cpu_b), float(mem_a * x + mem_b)


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return slope/intercept for a least-squares line fit."""
    if x.size == 0:
        return 0.0, 0.0
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    denom = float(np.sum((x - x_mean) ** 2))
    if denom <= 0:
        return 0.0, y_mean
    slope = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
    intercept = float(y_mean - slope * x_mean)
    return slope, intercept
