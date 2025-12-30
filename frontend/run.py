#!/usr/bin/env python3
"""Run the frontend application with basic diagnostics logging."""
from __future__ import annotations

import subprocess
from modules.diagnostics import record_error


def main() -> None:
    try:
        subprocess.run("kill $(lsof -t -i :5000)", shell=True, check=False)
        subprocess.run(
            ["flutter", "run", "-d", "chrome", "--web-port", "5000"], check=True
        )
    except Exception as err:  # pragma: no cover - run-time utility
        record_error(err, {"module": "frontend", "action": "run"})
        raise


if __name__ == "__main__":
    main()
