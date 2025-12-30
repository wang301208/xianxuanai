#!/usr/bin/env python3
"""Plot convergence curves from optimization record files.

This script reads one or more CSV record files and plots the best objective
value versus iteration number or function evaluation count. It allows comparing
multiple algorithms on the same figure and supports exporting the result to a
PNG or PDF file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import pandas as pd


BEST_COLUMNS = ["best", "best_value", "fitness", "value"]
ITER_COLUMNS = ["iteration", "iter", "step"]
EVAL_COLUMNS = ["evals", "function_evals", "nfev", "fevals"]


def _locate_column(columns: Iterable[str], options: Iterable[str]) -> str | None:
    """Return the first matching column name from *options* present in *columns*."""
    for opt in options:
        if opt in columns:
            return opt
    return None


def _read_record(path: Path, xaxis: str) -> Tuple[Iterable[float], Iterable[float], str]:
    """Read a CSV record file and return x, y values and x-axis label."""
    df = pd.read_csv(path)

    y_col = _locate_column(df.columns, BEST_COLUMNS)
    if y_col is None:
        raise ValueError(f"No best value column found in {path}")

    if xaxis == "iteration":
        x_col = _locate_column(df.columns, ITER_COLUMNS)
        x = df[x_col] if x_col else df.index
        xlabel = "Iteration"
    else:  # xaxis == 'evals'
        x_col = _locate_column(df.columns, EVAL_COLUMNS)
        if x_col is None:
            raise ValueError(f"No function evaluation column found in {path}")
        x = df[x_col]
        xlabel = "Function Evaluations"

    return x, df[y_col], xlabel


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot convergence curves from optimization record files",
    )
    parser.add_argument(
        "records",
        nargs="+",
        help="CSV record files to plot",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        help="Optional labels for each record; defaults to file names",
    )
    parser.add_argument(
        "--xaxis",
        choices=["iteration", "evals"],
        default="iteration",
        help="Use iteration or function evaluation count for the x-axis",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file (PNG or PDF). If omitted, the plot is shown interactively",
    )
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.records):
        parser.error("Number of labels must match number of records")

    plt.figure()
    xlabel = None
    for idx, record in enumerate(args.records):
        label = args.labels[idx] if args.labels else Path(record).stem
        x, y, xlabel = _read_record(Path(record), args.xaxis)
        plt.plot(x, y, label=label)

    plt.xlabel(xlabel or args.xaxis.title())
    plt.ylabel("Best Value")
    plt.legend()
    plt.grid(True)

    if args.output:
        output_path = Path(args.output)
        ext = output_path.suffix.lower()
        if ext not in {".png", ".pdf"}:
            raise ValueError("Output file must be a PNG or PDF")
        plt.savefig(output_path, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main()
