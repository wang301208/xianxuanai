from __future__ import annotations

import logging
import os
import threading
import time
from typing import Callable, List

logger = logging.getLogger(__name__)


class AutoScheduler:
    """Simple scheduler for periodically executing jobs."""

    def __init__(self) -> None:
        self._jobs: List[tuple[Callable[[], None], float]] = []
        self._stop = threading.Event()

    def add_job(self, func: Callable[[], None], interval: float) -> None:
        """Register ``func`` to run every ``interval`` seconds."""

        self._jobs.append((func, interval))

    def _run_job(self, func: Callable[[], None], interval: float) -> None:
        while not self._stop.wait(interval):
            try:
                func()
            except Exception:  # pragma: no cover - job may raise arbitrary errors
                logger.exception("Scheduled job failed")

    def start(self) -> None:
        """Start all registered jobs and block until stopped."""

        for func, interval in self._jobs:
            thread = threading.Thread(
                target=self._run_job, args=(func, interval), daemon=True
            )
            thread.start()

        try:
            while not self._stop.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self) -> None:
        """Signal all jobs to stop."""

        self._stop.set()


def _env_interval(var: str, default: float) -> float:
    try:
        return float(os.getenv(var, default))
    except ValueError:
        return default


def main() -> None:
    scheduler = AutoScheduler()

    default_interval = _env_interval("AUTO_SCHEDULE_INTERVAL", 0.0)
    retrain_interval = _env_interval("AUTO_RETRAIN_INTERVAL", default_interval)
    self_improve_interval = _env_interval(
        "AUTO_SELF_IMPROVE_INTERVAL", default_interval
    )

    if retrain_interval > 0:
        from ml import retraining_pipeline
        import subprocess
        from pathlib import Path

        def retrain_job() -> None:
            if not retraining_pipeline.main():
                script = (
                    Path(__file__).resolve().parent.parent
                    / "scripts"
                    / "rollback.sh"
                )
                subprocess.run(["bash", str(script)], check=False)

        scheduler.add_job(retrain_job, retrain_interval)

    if self_improve_interval > 0:
        from evolution.self_improvement import SelfImprovement

        scheduler.add_job(lambda: SelfImprovement().run(), self_improve_interval)

    if scheduler._jobs:
        logger.info("Starting AutoScheduler with %d job(s)", len(scheduler._jobs))
        scheduler.start()


if __name__ == "__main__":
    main()
