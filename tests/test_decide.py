"""Tests for agent.decide.decide_and_act -- the agentic core.

Uses a real (tmp_path-scoped) SQLite DB via agent.models.db_session for
Score/Decision persistence, and mocks the three action functions
(notify/schedule/compliance_report) plus the FortyGuardClient, since
this file tests the DECIDE/orchestration logic -- classify, act,
log -- not the action implementations themselves (covered by
test_compliance_report.py and manual review of notify.py/schedule.py).
"""

from __future__ import annotations

from datetime import datetime, time
from unittest.mock import MagicMock

from agent.decide import decide_and_act
from agent.models import Crew, Reading, Score, Site, db_session


def _seed(session):
    site = Site(
        name="Test Site",
        lat=37.33,
        lon=-121.90,
        polygon_geojson={"type": "Polygon", "coordinates": [[[0, 0]]]},
    )
    session.add(site)
    session.flush()

    crew = Crew(
        site_id=site.id,
        work_intensity="moderate",
        ppe_class="Class 1",
        active_shift_start=time(6, 0),
        active_shift_end=time(14, 0),
    )
    session.add(crew)
    session.flush()

    reading = Reading(
        site_id=site.id,
        ts=datetime(2026, 8, 3, 14, 0),
        heat_index=110.0,
        aqi=40.0,
        humidity=55.0,
        solar_irradiance=650.0,
        is_forecast=False,
    )
    session.add(reading)
    session.flush()

    return site, crew, reading


def _patch_actions(mocker, *, notify_return=None, schedule_return=None, report_return=None):
    return (
        mocker.patch("agent.act.notify.notify_slack", return_value=notify_return),
        mocker.patch("agent.act.schedule.write_shift_override", return_value=schedule_return),
        mocker.patch(
            "agent.act.compliance_report.generate_compliance_report", return_value=report_return
        ),
    )


def test_safe_tier_takes_no_action(tmp_path, mocker):
    notify_mock, schedule_mock, report_mock = _patch_actions(mocker)
    client = MagicMock()

    with db_session(f"sqlite:///{tmp_path / 't.db'}") as session:
        site, crew, reading = _seed(session)
        decision = decide_and_act(
            session, client, site, crew, reading, score=80.0, tier="safe", trigger_type="live"
        )
        assert decision.action_taken == "none"

    notify_mock.assert_not_called()
    schedule_mock.assert_not_called()
    report_mock.assert_not_called()


def test_caution_tier_schedules_break_reminders_only(tmp_path, mocker):
    notify_mock, schedule_mock, report_mock = _patch_actions(mocker)
    client = MagicMock()

    with db_session(f"sqlite:///{tmp_path / 't.db'}") as session:
        site, crew, reading = _seed(session)
        decision = decide_and_act(
            session, client, site, crew, reading, score=95.0, tier="caution", trigger_type="live"
        )
        assert decision.action_taken == "break_reminders"

    notify_mock.assert_not_called()
    report_mock.assert_not_called()
    schedule_mock.assert_called_once()
    assert schedule_mock.call_args.args[2] == "break_reminders"


def test_high_tier_notifies_and_shortens_shift(tmp_path, mocker):
    notify_mock, schedule_mock, report_mock = _patch_actions(
        mocker, notify_return="slack", report_return="/outputs/r.pdf"
    )
    client = MagicMock()

    with db_session(f"sqlite:///{tmp_path / 't.db'}") as session:
        site, crew, reading = _seed(session)
        decision = decide_and_act(
            session, client, site, crew, reading, score=115.0, tier="high", trigger_type="live"
        )
        assert decision.action_taken == "shorten_shift_and_notify"
        assert decision.notified_channel == "slack"
        assert decision.report_url == "/outputs/r.pdf"

    notify_mock.assert_called_once()
    schedule_mock.assert_called_once()
    assert schedule_mock.call_args.args[2] == "shorten_shift"
    report_mock.assert_called_once()


def test_extreme_tier_halts_work_notifies_and_triggers_report(tmp_path, mocker):
    notify_mock, schedule_mock, report_mock = _patch_actions(
        mocker, notify_return="slack", report_return="/outputs/r2.pdf"
    )
    client = MagicMock()

    with db_session(f"sqlite:///{tmp_path / 't.db'}") as session:
        site, crew, reading = _seed(session)
        decision = decide_and_act(
            session, client, site, crew, reading, score=140.0, tier="extreme", trigger_type="live"
        )
        assert decision.action_taken == "halt_and_notify_and_report"
        assert decision.notified_channel == "slack"
        assert decision.report_url == "/outputs/r2.pdf"

    notify_mock.assert_called_once()
    schedule_mock.assert_called_once()
    assert schedule_mock.call_args.args[2] == "halt_work"
    report_mock.assert_called_once()


def test_every_decision_is_logged_with_full_audit_fields(tmp_path, mocker):
    _patch_actions(mocker)
    client = MagicMock()

    with db_session(f"sqlite:///{tmp_path / 't.db'}") as session:
        site, crew, reading = _seed(session)
        decision = decide_and_act(
            session, client, site, crew, reading, score=95.0, tier="caution", trigger_type="live"
        )

        assert decision.id is not None
        assert decision.score_id is not None
        assert decision.action_taken is not None
        assert decision.executed_at is not None
        assert decision.trigger_type is not None

        score_row = session.get(Score, decision.score_id)
        assert score_row.reading_id == reading.id
        assert score_row.crew_id == crew.id
        assert score_row.final_score == 95.0
        assert score_row.risk_tier == "caution"


def test_forecast_flag_triggers_proactive_branch_not_live_branch(tmp_path, mocker):
    _patch_actions(mocker)
    client = MagicMock()

    with db_session(f"sqlite:///{tmp_path / 't.db'}") as session:
        site, crew, reading = _seed(session)
        live_decision = decide_and_act(
            session, client, site, crew, reading, score=95.0, tier="caution", trigger_type="live"
        )
        forecast_decision = decide_and_act(
            session,
            client,
            site,
            crew,
            reading,
            score=95.0,
            tier="caution",
            trigger_type="forecast",
        )

        assert live_decision.trigger_type == "live"
        assert forecast_decision.trigger_type == "forecast"
        assert live_decision.id != forecast_decision.id
