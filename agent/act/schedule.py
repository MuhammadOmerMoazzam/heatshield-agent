"""Shift-override action.

Appends a JSON-lines record for a shift change (break reminders /
shortened shift / halted work) for a crew. A simple file, not a DB
table, per the Phase 6 prompt's own "simple table or file for the
sprint" allowance -- this keeps Phase 2's already-shipped, tested schema
untouched rather than adding a table to it from a later phase.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OVERRIDES_PATH = _REPO_ROOT / "data" / "shift_overrides.jsonl"


def write_shift_override(
    site_id: int, crew_id: int, override_type: str, *, path: Path | None = None
) -> None:
    target = path or DEFAULT_OVERRIDES_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "site_id": site_id,
        "crew_id": crew_id,
        "override_type": override_type,
        "recorded_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
    }
    with target.open("a") as f:
        f.write(json.dumps(record) + "\n")
