"""Tests for dashboard.app's Phase 9 bootstrap.

Streamlit Cloud runs a single process with no separate always-on service
for the agent loop, so dashboard/app.py must start agent.loop.run_scheduler()
itself. Streamlit reruns a session's script top-to-bottom on every
interaction, and a single process can serve multiple sessions -- so the
guard must be a module-level flag (shared across every session in this
process), not st.session_state (scoped to one session), or a second
browser tab would start a second competing scheduler thread.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

import dashboard.app as dashboard_app


@pytest.fixture(autouse=True)
def _reset_bootstrap_state():
    """Each test gets a clean slate -- these are module-level globals."""
    dashboard_app._scheduler_started = False
    dashboard_app._scheduler_handle = None
    yield
    dashboard_app._scheduler_started = False
    dashboard_app._scheduler_handle = None


def test_bootstrap_starts_scheduler_exactly_once_across_simulated_reruns(mocker):
    mock_run_scheduler = mocker.patch("dashboard.app.run_scheduler", return_value=MagicMock())

    # Simulate repeated Streamlit reruns (e.g. the user clicking around
    # within one session, or a second session's script execution in the
    # same process).
    dashboard_app.bootstrap_agent_loop()
    dashboard_app.bootstrap_agent_loop()
    dashboard_app.bootstrap_agent_loop()

    mock_run_scheduler.assert_called_once()


def test_bootstrap_returns_the_same_handle_on_every_call(mocker):
    sentinel = MagicMock()
    mocker.patch("dashboard.app.run_scheduler", return_value=sentinel)

    first = dashboard_app.bootstrap_agent_loop()
    second = dashboard_app.bootstrap_agent_loop()

    assert first is sentinel
    assert second is sentinel


def test_bootstrap_guard_holds_under_concurrent_reruns(mocker):
    """Regression-shaped: Streamlit can run multiple sessions' scripts on
    separate threads within the same process, so two sessions could both
    hit an unguarded first call at nearly the same time.
    """
    mock_run_scheduler = mocker.patch("dashboard.app.run_scheduler", return_value=MagicMock())

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda _: dashboard_app.bootstrap_agent_loop(), range(8)))

    mock_run_scheduler.assert_called_once()
