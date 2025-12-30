import atexit
import json
import queue
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


class ActionLogger:
    """Append structured action logs in JSON Lines format."""

    def __init__(self, log_path: Path | str) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._closed = False
        self._lock = threading.Lock()

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

        atexit.register(self.close)

    def _worker(self) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            while True:
                record = self._queue.get()
                if record is None:
                    break
                with self._lock:
                    json.dump(record, f)
                    f.write("\n")
                    f.flush()

    def log(self, record: Dict[str, Any]) -> None:
        """Append *record* to the log with timestamp and unique id."""
        record.setdefault("id", str(uuid.uuid4()))
        record.setdefault("timestamp", datetime.utcnow().isoformat())
        self._queue.put(record)

    def close(self) -> None:
        if self._closed:
            return
        self._queue.put(None)
        self._thread.join()
        self._closed = True

    def __del__(self) -> None:
        self.close()
