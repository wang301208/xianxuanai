import json
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from monitoring.storage import TimeSeriesStorage


def _insert(cur, ts, topic, data):
    cur.execute(
        "INSERT INTO events (ts, topic, data) VALUES (?, ?, ?)",
        (ts, topic, json.dumps(data)),
    )


def test_index_and_event_filters(tmp_path):
    storage = TimeSeriesStorage(tmp_path / "events.db")

    # index should exist
    cur = storage._conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_events_topic_ts'"
    )
    assert cur.fetchone() is not None

    _insert(cur, 1.0, "a", {"v": 1})
    _insert(cur, 2.0, "a", {"v": 2})
    _insert(cur, 3.0, "a", {"v": 3})
    storage._conn.commit()

    all_events = storage.events("a")
    assert len(all_events) == 3

    limited = storage.events("a", limit=2)
    assert [e["v"] for e in limited] == [1, 2]

    ranged = storage.events("a", start_ts=1.5, end_ts=2.5)
    assert [e["v"] for e in ranged] == [2]


def test_aggregations(tmp_path):
    storage = TimeSeriesStorage(tmp_path / "agg.db")
    cur = storage._conn.cursor()
    events = [
        (1.0, "t", {"status": "success", "blueprint_version": 1}),
        (2.0, "t", {"status": "failure", "stage": "plan", "blueprint_version": 1}),
        (3.0, "t", {"status": "failure", "stage": "plan", "blueprint_version": 2}),
        (4.0, "t", {"status": "failure", "stage": "execute"}),
        (5.0, "t", {"status": "success"}),
    ]
    for ts, topic, data in events:
        _insert(cur, ts, topic, data)
    storage._conn.commit()

    assert storage.success_rate() == pytest.approx(2 / 5)
    assert storage.bottlenecks() == {"plan": 2, "execute": 1}
    assert storage.blueprint_versions() == {"1": 2, "2": 1, "unknown": 2}

