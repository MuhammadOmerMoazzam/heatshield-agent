"""Small utilities shared across agent modules.

A single source of truth for the repo root and the naive-UTC "now"
convention, instead of each module re-deriving them independently --
which previously let one module's default path silently depend on the
process's CWD while sibling modules already anchored to the repo root.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def now_naive_utc() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)
