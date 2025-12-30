import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "modules")))

import pytest

from metrics.workspace import WorkspaceImpactTracker


def test_average_improvement() -> None:
    tracker = WorkspaceImpactTracker()
    tracker.record("t1", 0.5, 0.8)
    tracker.record("t2", 0.4, 0.6)
    assert tracker.average_improvement() == pytest.approx((0.3 + 0.2) / 2)
