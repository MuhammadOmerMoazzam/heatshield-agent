"""Shift-override action.

Appends a JSON-lines record for a shift change (break reminders /
shortened shift / halted work) for a crew. A simple file, not a DB
table, per the Phase 6 prompt's own "simple table or file for the
sprint" allowance -- this keeps Phase 2's already-shipped, tested schema
untouched rather than adding a table to it from a later phase.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from agent._shared import REPO_ROOT

DEFAULT_OVERRIDES_PATH = REPO_ROOT / "data" / "shift_overrides.jsonl"

_write_lock = threading.Lock()


def write_shift_override(
    site_id: int, crew_id: int, override_type: str, *, path: str | Path | None = None
) -> None:
    target = Path(path) if path else DEFAULT_OVERRIDES_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "site_id": site_id,
        "crew_id": crew_id,
        "override_type": override_type,
        # Kept timezone-aware (unlike the DB's naive-UTC convention) --
        # this file is meant for external tooling/dashboards to read, not
        # just this codebase, so the offset shouldn't be left implicit.
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    # Concurrent decide_and_act calls (Phase 7's scheduler) can append to
    # this same file; a lock keeps each JSON line from interleaving with
    # another writer's.
    with _write_lock:
        with target.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
