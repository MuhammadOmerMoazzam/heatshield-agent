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

import os
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

import dashboard.app as dashboard_app


class _RaisingSecrets:
    """Stands in for st.secrets when no secrets.toml exists -- the normal
    local-dev case (a .env file is used instead), not an error.
    """

    def items(self):
        raise RuntimeError("No secrets found")


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


def test_load_secrets_into_environ_copies_st_secrets(mocker):
    """Streamlit Cloud secrets (st.secrets) are a separate mechanism from
    os.environ -- not injected automatically -- so the rest of the
    codebase's plain os.getenv(...) calls (the pattern the build plan
    specifies for local dev via python-dotenv) would silently see nothing
    on Cloud even with secrets correctly configured in the Cloud UI.
    """
    mocker.patch.object(
        dashboard_app.st, "secrets", {"FORTYGUARD_API_KEY": "from-cloud-secrets"}
    )
    mocker.patch.dict(os.environ, {}, clear=False)
    os.environ.pop("FORTYGUARD_API_KEY", None)

    dashboard_app._load_secrets_into_environ()

    assert os.environ["FORTYGUARD_API_KEY"] == "from-cloud-secrets"


def test_load_secrets_into_environ_does_not_override_existing_env(mocker):
    """Same precedence already documented for python-dotenv elsewhere in
    this codebase: a value already in os.environ wins.
    """
    mocker.patch.object(dashboard_app.st, "secrets", {"FORTYGUARD_API_KEY": "from-cloud-secrets"})
    mocker.patch.dict(os.environ, {"FORTYGUARD_API_KEY": "from-dot-env"}, clear=False)

    dashboard_app._load_secrets_into_environ()

    assert os.environ["FORTYGUARD_API_KEY"] == "from-dot-env"


def test_load_secrets_into_environ_is_a_no_op_when_no_secrets_file_exists(mocker):
    """Local dev with only a .env file (no .streamlit/secrets.toml) is the
    normal case -- st.secrets raises there, and that must not crash the
    dashboard.
    """
    mocker.patch.object(dashboard_app.st, "secrets", _RaisingSecrets())

    dashboard_app._load_secrets_into_environ()  # must not raise


def test_bootstrap_retries_on_the_next_call_when_run_scheduler_returns_none(mocker):
    """run_scheduler() returns None when it couldn't claim the
    cross-process SchedulerLock (agent/models.py) -- e.g. an orphaned
    process from an earlier reboot still holds it. That must not
    permanently latch this process out of ever running a scheduler: the
    lock is heartbeat-based and can go stale, so retrying on the next
    rerun is what actually lets a live process recover once the orphan
    is truly gone.
    """
    sentinel = MagicMock()
    mock_run_scheduler = mocker.patch(
        "dashboard.app.run_scheduler", side_effect=[None, sentinel]
    )

    first = dashboard_app.bootstrap_agent_loop()
    second = dashboard_app.bootstrap_agent_loop()

    assert first is None
    assert second is sentinel
    assert mock_run_scheduler.call_count == 2


def test_bootstrap_does_not_crash_the_dashboard_when_run_scheduler_raises(mocker):
    """Code-review regression: run_scheduler() can raise before ever
    returning (e.g. FortyGuardClient() raising FortyGuardError on a
    missing/invalid FORTYGUARD_API_KEY -- a real scenario this project
    hit live this session). main() has no try/except of its own around
    bootstrap_agent_loop(), so an uncaught exception here used to crash
    the entire dashboard render on every single rerun -- not even the
    already-seeded site data would show. A misconfigured key should
    degrade to "no scheduler yet", not take the whole dashboard down.
    """
    mocker.patch("dashboard.app.run_scheduler", side_effect=RuntimeError("no API key"))

    result = dashboard_app.bootstrap_agent_loop()  # must not raise

    assert result is None


def test_bootstrap_guard_holds_under_concurrent_reruns(mocker):
    """Regression-shaped: Streamlit can run multiple sessions' scripts on
    separate threads within the same process, so two sessions could both
    hit an unguarded first call at nearly the same time.
    """
    mock_run_scheduler = mocker.patch("dashboard.app.run_scheduler", return_value=MagicMock())

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda _: dashboard_app.bootstrap_agent_loop(), range(8)))

    mock_run_scheduler.assert_called_once()
