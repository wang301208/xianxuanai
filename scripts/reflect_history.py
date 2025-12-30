#!/usr/bin/env python3
"""Diagnostic CLI to visualize stored reflection histories."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure repository root on path for direct module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.memory.long_term import LongTermMemory
from backend.reflection import load_histories


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect stored reflection histories")
    parser.add_argument("db", help="Path to the LongTermMemory SQLite database")
    parser.add_argument(
        "--category",
        default="reflection",
        help="Memory category used for reflection histories",
    )
    args = parser.parse_args()

    memory = LongTermMemory(Path(args.db))
    try:
        for idx, history in enumerate(load_histories(memory, category=args.category), 1):
            print(f"History {idx}:")
            for j, (evaluation, revision) in enumerate(history, 1):
                print(
                    f"  Pass {j}: confidence={evaluation.confidence:.2f} sentiment={evaluation.sentiment}"
                )
                print(f"    Revision: {revision}")
    finally:
        memory.close()


if __name__ == "__main__":
    main()
