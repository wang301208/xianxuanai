import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class StopCondition:
    """Unified termination logic for optimization algorithms."""

    max_iters: Optional[int] = None
    max_time: Optional[float] = None
    patience: Optional[int] = None

    def __post_init__(self) -> None:
        self.start_time = time.time()
        self.iteration = 0
        self.no_improve = 0

    def keep_running(self) -> bool:
        if self.max_iters is not None and self.iteration >= self.max_iters:
            return False
        if self.max_time is not None and time.time() - self.start_time >= self.max_time:
            return False
        if self.patience is not None and self.no_improve >= self.patience:
            return False
        return True

    def update(self, improved: bool) -> None:
        self.iteration += 1
        if improved:
            self.no_improve = 0
        else:
            self.no_improve += 1
