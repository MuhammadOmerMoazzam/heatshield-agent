"""Tests for dashboard.app -- thin per [SS-C.7], since the dashboard is a
viewer, not core logic.

The one contract this file enforces: the render/query functions in
dashboard/app.py must only ever read sites/readings/scores/decisions from
the DB. FortyGuardClient (the credit-spending code) may only be imported
by the one-time background-loop bootstrap added in Phase 9 and tested
separately in test_bootstrap.py -- at Phase 8 there is no bootstrap yet,
so the module must not reference FortyGuardClient at all.
"""

from __future__ import annotations

import ast
import inspect

import dashboard.app as dashboard_app


def test_dedupe_sites_by_name_prefers_the_entry_with_data():
    """Real bug, live-observed: concurrent Streamlit reruns could both
    pass seed_demo_sites_if_empty's "any site exists" check on a still-
    empty database before either committed, inserting the demo sites
    twice -- one copy then accumulates real readings while the other
    sits at "no live reading yet" forever, both visible on the
    dashboard. Now guarded at the source too (agent/models.py: a UNIQUE
    constraint on Site.name), but this keeps the dashboard itself
    correct against rows already duplicated from before that fix, or
    any other future cause.
    """
    empty_entry = {"name": "Phoenix, AZ", "id": 1, "latest_live_ts": None}
    data_entry = {"name": "Phoenix, AZ", "id": 2, "latest_live_ts": "2026-08-30 12:00:00"}
    other_site = {"name": "Houston, TX", "id": 3, "latest_live_ts": None}

    results = dashboard_app._dedupe_sites_by_name([empty_entry, data_entry, other_site])

    assert len(results) == 2
    phoenix = next(r for r in results if r["name"] == "Phoenix, AZ")
    assert phoenix["id"] == 2


def test_dashboard_render_functions_never_call_fortyguard_directly():
    source = inspect.getsource(dashboard_app)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "fortyguard" not in alias.name.lower(), (
                    f"dashboard.app imports {alias.name!r} -- FortyGuardClient must not be "
                    "reachable from Phase 8's render/query functions."
                )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert "fortyguard" not in module.lower(), (
                f"dashboard.app imports from {module!r} -- FortyGuardClient must not be "
                "reachable from Phase 8's render/query functions."
            )
        elif isinstance(node, ast.Name):
            assert node.id != "FortyGuardClient", (
                "dashboard.app references FortyGuardClient by name -- the bootstrap that's "
                "allowed to do this doesn't exist until Phase 9."
            )
