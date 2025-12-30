from __future__ import annotations

from datetime import datetime
from typing import Dict, List


def resolve_conflicts(existing: List[Dict[str, str]], new_items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Merge ``new_items`` into ``existing`` resolving conflicting statements.

    Conflicts are detected when two statements refer to the same ``entity`` but
    provide a different ``value``.  The entry with the more recent ``timestamp``
    wins.  Whenever a replacement happens the ``version`` number is incremented
    to keep track of revisions of a fact.
    """
    updated = list(existing)

    for item in new_items:
        duplicates = [e for e in updated if e.get("entity") == item.get("entity")]
        if not duplicates:
            item.setdefault("version", 1)
            updated.append(item)
            continue

        latest = max(duplicates, key=lambda e: e.get("timestamp", ""))
        if latest.get("value") != item.get("value"):
            # conflicting assertion: keep the newer one
            latest_ts = datetime.fromisoformat(latest.get("timestamp"))
            item_ts = datetime.fromisoformat(item.get("timestamp"))
            if item_ts >= latest_ts:
                item["version"] = latest.get("version", 1) + 1
                updated = [e for e in updated if e.get("entity") != item.get("entity")]
                updated.append(item)
            # else: keep existing latest
        else:
            # same assertion, keep the most recent timestamp
            if item.get("timestamp") > latest.get("timestamp"):
                item["version"] = latest.get("version", 1) + 1
                updated = [e for e in updated if e is not latest]
                updated.append(item)
    return updated

__all__ = ["resolve_conflicts"]
