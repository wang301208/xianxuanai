"""Compare multi-capability benchmark summaries and flag regressions."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple


def load_summary(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text())
    if "tests" not in data:
        raise ValueError(f"Summary file {path} does not contain 'tests' field")
    return data


def compare(
    baseline: Dict[str, Any],
    current: Dict[str, Any],
    threshold: float,
) -> Tuple[List[str], List[str]]:
    """Return (report_lines, regression_lines)."""
    report: List[str] = []
    regressions: List[str] = []

    base_tests: Dict[str, Dict[str, Any]] = baseline.get("tests", {})
    cur_tests: Dict[str, Dict[str, Any]] = current.get("tests", {})

    report.append("Multi-capability benchmark comparison")
    report.append(f"Threshold: {threshold * 100:.1f}% allowed slowdown")
    report.append("")

    for nodeid, base_info in sorted(base_tests.items()):
        cur_info = cur_tests.get(nodeid)
        if cur_info is None:
            msg = f"Missing test in current run: {nodeid}"
            report.append(msg)
            regressions.append(msg)
            continue

        base_outcome = base_info.get("outcome")
        cur_outcome = cur_info.get("outcome")
        base_duration = base_info.get("call_duration") or 0.0
        cur_duration = cur_info.get("call_duration") or 0.0

        report.append(
            f"{nodeid}: baseline={base_duration:.6f}s current={cur_duration:.6f}s "
            f"outcome={cur_outcome}"
        )

        if cur_outcome != "passed":
            msg = f"Test {nodeid} did not pass (outcome={cur_outcome})"
            regressions.append(msg)
            continue

        if base_outcome == "passed" and base_duration > 0:
            allowed = base_duration * (1 + threshold)
            if cur_duration > allowed:
                delta = cur_duration - base_duration
                msg = (
                    f"Regression {nodeid}: baseline {base_duration:.6f}s, "
                    f"current {cur_duration:.6f}s (+{delta:.6f}s)"
                )
                regressions.append(msg)

    # detect unexpected new tests? optionally not necessary
    extra_tests = set(cur_tests) - set(base_tests)
    if extra_tests:
        report.append("")
        report.append("New tests in current run:")
        for nodeid in sorted(extra_tests):
            report.append(f"  - {nodeid}")

    return report, regressions


def main() -> None:
    if len(sys.argv) < 3:
        print(
            "Usage: compare_multicap_benchmarks.py <baseline_summary> <current_summary> [threshold] [report_path]"
        )
        sys.exit(2)

    baseline_path = Path(sys.argv[1])
    current_path = Path(sys.argv[2])
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
    report_path = Path(sys.argv[4]) if len(sys.argv) > 4 else None

    baseline = load_summary(baseline_path)
    current = load_summary(current_path)

    report_lines, regressions = compare(baseline, current, threshold)
    report_text = "\n".join(report_lines)
    print(report_text)

    if report_path is not None:
        report_path.write_text(report_text)

    if regressions:
        print("\nDetected regressions:")
        for line in regressions:
            print(f" - {line}")
        sys.exit(1)


if __name__ == "__main__":
    main()
