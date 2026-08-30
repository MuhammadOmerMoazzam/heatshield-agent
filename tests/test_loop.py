"""Tests for agent.loop's cross-process scheduler lock.

Real production bug, live-observed: dashboard/app.py's in-memory
"started once per process" guard can't stop a second, independent
process (an orphaned thread from an earlier Streamlit Cloud reboot that
wasn't fully killed) from *also* running agent.loop.run_scheduler()
against the same FortyGuard API key -- live logs showed a single site
failing far more times, in a far shorter window, than a single 6-hour-
interval scheduler could produce. SchedulerLock (agent/models.py) is a
singleton database row (the one thing confirmed to be genuinely shared
across however many of "these" processes exist) that only one caller
can hold at a time, with a heartbeat so a truly dead orphan's lock can
still be reclaimed later rather than permanently bricking the scheduler.
"""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock

from agent._shared import now_naive_utc
from agent.loop import (
    _HEARTBEAT_INTERVAL_MINUTES,
    _LOCK_STALE_AFTER,
    _claim_scheduler_lock,
    _renew_scheduler_lock,
    run_scheduler,
)
from agent.models import SchedulerLock, db_session


def test_claim_scheduler_lock_succeeds_when_no_lock_row_exists(tmp_path):
    with db_session(f"sqlite:///{tmp_path / 't.db'}") as session:
        assert _claim_scheduler_lock(session) is True
        assert session.query(SchedulerLock).count() == 1


def test_claim_scheduler_lock_fails_when_an_existing_lock_is_still_fresh(tmp_path):
    database_url = f"sqlite:///{tmp_path / 't.db'}"
    with db_session(database_url) as session:
        assert _claim_scheduler_lock(session) is True

    with db_session(database_url) as session:
        assert _claim_scheduler_lock(session) is False
        # Still just the one row -- a failed claim must not touch it.
        assert session.query(SchedulerLock).count() == 1


def test_claim_scheduler_lock_succeeds_when_the_existing_lock_is_stale(tmp_path):
    database_url = f"sqlite:///{tmp_path / 't.db'}"
    stale_time = now_naive_utc() - _LOCK_STALE_AFTER - timedelta(minutes=1)
    with db_session(database_url) as session:
        session.add(SchedulerLock(id=1, started_at=stale_time, heartbeat_at=stale_time))

    with db_session(database_url) as session:
        assert _claim_scheduler_lock(session) is True
        lock = session.query(SchedulerLock).one()
        assert lock.heartbeat_at > stale_time


def test_renew_scheduler_lock_updates_heartbeat(tmp_path):
    database_url = f"sqlite:///{tmp_path / 't.db'}"
    old_time = now_naive_utc() - timedelta(minutes=1)
    with db_session(database_url) as session:
        session.add(SchedulerLock(id=1, started_at=old_time, heartbeat_at=old_time))

    with db_session(database_url) as session:
        _renew_scheduler_lock(session)

    with db_session(database_url) as session:
        lock = session.query(SchedulerLock).one()
        assert lock.heartbeat_at > old_time


def test_run_scheduler_does_not_start_a_background_scheduler_when_lock_is_held(
    tmp_path, mocker, monkeypatch
):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 't.db'}")
    mocker.patch("agent.loop.FortyGuardClient")
    mock_background_scheduler_cls = mocker.patch(
        "apscheduler.schedulers.background.BackgroundScheduler"
    )
    with db_session() as session:
        _claim_scheduler_lock(session)  # simulate another process already holding it

    result = run_scheduler()

    assert result is None
    mock_background_scheduler_cls.assert_not_called()


def test_run_scheduler_starts_and_schedules_both_the_sensing_and_heartbeat_jobs(
    tmp_path, mocker, monkeypatch
):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 't.db'}")
    mocker.patch("agent.loop.FortyGuardClient")
    mock_scheduler = MagicMock()
    mock_background_scheduler_cls = mocker.patch(
        "apscheduler.schedulers.background.BackgroundScheduler", return_value=mock_scheduler
    )

    result = run_scheduler()

    mock_background_scheduler_cls.assert_called_once()
    assert result is mock_scheduler
    mock_scheduler.start.assert_called_once()
    # Two distinct interval jobs: the expensive sense/score/decide cycle,
    # and a cheap, frequent heartbeat that keeps this process's claim on
    # SchedulerLock fresh so a healthy scheduler is never mistaken for a
    # dead orphan and reclaimed out from under it.
    assert mock_scheduler.add_job.call_count == 2
    interval_minutes_used = [
        call.kwargs.get("minutes") for call in mock_scheduler.add_job.call_args_list
    ]
    assert _HEARTBEAT_INTERVAL_MINUTES in interval_minutes_used


def test_run_scheduler_reclaims_a_stale_lock_left_by_a_dead_orphan(tmp_path, mocker, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 't.db'}")
    mocker.patch("agent.loop.FortyGuardClient")
    mock_background_scheduler_cls = mocker.patch(
        "apscheduler.schedulers.background.BackgroundScheduler", return_value=MagicMock()
    )
    stale_time = now_naive_utc() - _LOCK_STALE_AFTER - timedelta(minutes=1)
    with db_session() as session:
        session.add(SchedulerLock(id=1, started_at=stale_time, heartbeat_at=stale_time))

    result = run_scheduler()

    assert result is not None
    mock_background_scheduler_cls.assert_called_once()
