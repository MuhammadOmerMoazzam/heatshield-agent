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
