"""Utility to extract key metrics from pytest-json-report output."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict, Any


def build_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    tests_summary: Dict[str, Dict[str, Any]] = {}
    for test in report.get("tests", []):
        nodeid = test.get("nodeid", "")
        if not nodeid:
            continue
        call = test.get("call", {}) or {}
        tests_summary[nodeid] = {
            "outcome": test.get("outcome"),
            "call_duration": call.get("duration"),
            "setup_duration": (test.get("setup") or {}).get("duration"),
            "teardown_duration": (test.get("teardown") or {}).get("duration"),
        }

    return {
        "summary": report.get("summary", {}),
        "tests": tests_summary,
    }


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: summarize_pytest_json.py <report_path> <output_path>")
        sys.exit(2)

    report_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    report = json.loads(report_path.read_text())
    summary = build_summary(report)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
