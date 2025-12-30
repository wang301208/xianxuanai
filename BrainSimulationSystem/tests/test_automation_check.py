"""Smoke test for the automation check report."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from BrainSimulationSystem.core.automation_check import run_automated_checks


def test_run_automated_checks_has_no_high_severity_issues():
    """The repository should be free from high-severity modularity problems."""

    report = run_automated_checks()

    assert "高优先级问题" not in report
