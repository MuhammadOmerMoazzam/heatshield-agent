"""Integration tests for agent.loop.

The FortyGuardClient is mocked entirely (no real network), but every
other module -- sense, score, decide, act, models -- is real, exercising
the full sense -> score -> decide -> act chain end-to-end against a
tmp_path-scoped SQLite DB.
"""

from __future__ import annotations

from datetime import time
from unittest.mock import MagicMock

import pytest

from agent.fortyguard_client import TaskFailedError
from agent.loop import _check_credits, run_cycle, run_once, run_scheduler
from agent.models import Crew, Decision, Reading, Score, Site, db_session
from agent.sense import sense_live

HOUSTON_POLYGON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-95.3948, 29.7354],
                    [-95.3448, 29.7354],
                    [-95.3448, 29.7854],
                    [-95.3948, 29.7854],
                    [-95.3948, 29.7354],
                ]],
            },
        }
    ],
}

PHOENIX_POLYGON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-112.10, 33.40],
                    [-112.00, 33.40],
                    [-112.00, 33.50],
                    [-112.10, 33.50],
                    [-112.10, 33.40],
                ]],
            },
        }
    ],
}


def _heatmap_result(mean_temp_c: float = 25.0, max_temp_c: float = 27.0) -> dict:
    return {
        "activity_id": "hm",
        "result": {
            "stats_data": {"temperature_stats": {"mean": mean_temp_c, "maximum": max_temp_c}},
            "map_data": {
                "type": "FeatureCollection",
                "features": [{"properties": {"temperature": mean_temp_c}}],
            },
        },
    }


def _env_params_result(
    heat_index_c: float = 25.0, humidity: float = 40.0, ghi: float = 300.0, aqi: float = 20.0
) -> dict:
    return {
        "activity_id": "env",
        "result": {
            "locations": [
                {
                    "parameters": {
                        "heat_index_celsius": [heat_index_c],
                        "relative_humidity_percent": [humidity],
                        "air_quality:idx": [aqi],
                    },
                    "solar_irradiance": {"clear_sky": {"ghi": ghi}},
                }
            ]
        },
    }


def _usage_result(remaining_credits: float) -> dict:
    return {"remaining_credits": remaining_credits}


def _seed(session) -> tuple[Site, Crew]:
    site = Site(
        name="Phoenix Test Site",
        lat=33.4484,
        lon=-112.0740,
        polygon_geojson=PHOENIX_POLYGON,
        shade_coverage_pct=20.0,
        canopy_pct=10.0,
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

    return site, crew


def _seed_second_site(session) -> tuple[Site, Crew]:
    site = Site(
        name="Houston Test Site",
        lat=29.7604,
        lon=-95.3698,
        polygon_geojson=HOUSTON_POLYGON,
        shade_coverage_pct=15.0,
        canopy_pct=8.0,
    )
    session.add(site)
    session.flush()

    crew = Crew(
        site_id=site.id,
        work_intensity="heavy",
        ppe_class="Class 2",
        active_shift_start=time(6, 0),
        active_shift_end=time(14, 0),
    )
    session.add(crew)
    session.flush()

    return site, crew


def test_full_cycle_safe_reading_produces_no_side_effects(tmp_path, mocker):
    notify_mock = mocker.patch("agent.act.notify.notify_slack")
    schedule_mock = mocker.patch("agent.act.schedule.write_shift_override")
    report_mock = mocker.patch("agent.act.compliance_report.generate_compliance_report")

    client = MagicMock()
    client.create_heatmap.return_value = _heatmap_result(mean_temp_c=25.0, max_temp_c=26.0)
    client.environmental_parameters.return_value = _env_params_result(
        heat_index_c=25.0, humidity=40.0, ghi=300.0, aqi=20.0
    )

    with db_session(f"sqlite:///{tmp_path / 't.db'}") as session:
        _seed(session)
        decisions = run_cycle(session, client, skip_onboarding=True)

        assert len(decisions) == 2  # one live + one forecast, both crew-scored
        for decision in decisions:
            assert decision.action_taken == "none"

        # Phase 9: heatmap_geojson is deliberately never persisted -- a
        # day-level tile FeatureCollection was live-verified as over 1
        # million characters for a single reading, and storing it was
        # directly implicated in recurring "database is locked" failures
        # plus fast, unbounded SQLite growth (see agent/sense.py).
        readings = session.query(Reading).all()
        assert len(readings) == 2
        for reading in readings:
            assert reading.heatmap_geojson is None

    notify_mock.assert_not_called()
    schedule_mock.assert_not_called()
    report_mock.assert_not_called()


def test_full_cycle_extreme_reading_produces_halt_alert_and_report(tmp_path, mocker):
    """The live branch halts work, notifies, and reports. The forecast
    branch (same underlying heatmap data, since the mocked client returns
    the same extreme reading for both the "now" and "now+12h" calls) is
    intentionally different per decide.py's forecast redesign: it flags a
    reschedule and sends a heads-up notification, but never halts work or
    requests a compliance report for a condition that hasn't happened yet
    (Part B.2's proactive-branch design; see decide.py's FORECAST_ACTION_MAP).
    """
    notify_mock = mocker.patch("agent.act.notify.notify_slack", return_value="slack")
    schedule_mock = mocker.patch("agent.act.schedule.write_shift_override")
    report_mock = mocker.patch(
        "agent.act.compliance_report.generate_compliance_report", return_value="/outputs/r.pdf"
    )

    client = MagicMock()
    client.create_heatmap.return_value = _heatmap_result(mean_temp_c=54.0, max_temp_c=56.0)
    client.environmental_parameters.return_value = _env_params_result(
        heat_index_c=54.0, humidity=70.0, ghi=800.0, aqi=90.0
    )

    with db_session(f"sqlite:///{tmp_path / 't.db'}") as session:
        _seed(session)
        decisions = run_cycle(session, client, skip_onboarding=True)

        live_decision = next(d for d in decisions if d.trigger_type == "live")
        forecast_decision = next(d for d in decisions if d.trigger_type == "forecast")

        assert live_decision.action_taken == "halt_and_notify_and_report"
        assert live_decision.report_url == "/outputs/r.pdf"
        assert forecast_decision.action_taken == "flag_for_reschedule_and_notify"
        assert forecast_decision.report_url is None

    # Both branches notify + schedule (once each); only the live branch
    # ever requests a compliance report.
    assert notify_mock.call_count == 2
    assert schedule_mock.call_count == 2
    assert report_mock.call_count == 1


def test_credit_usage_checked_before_cycle_runs_and_throttles_if_low(tmp_path, mocker):
    onboard_mock = mocker.patch("agent.loop.onboard_site")
    mocker.patch("agent.act.notify.notify_slack")
    mocker.patch("agent.act.schedule.write_shift_override")
    mocker.patch("agent.act.compliance_report.generate_compliance_report")

    client = MagicMock()
    client.create_heatmap.return_value = _heatmap_result()
    client.environmental_parameters.return_value = _env_params_result()

    db_url = f"sqlite:///{tmp_path / 't.db'}"
    with db_session(db_url) as session:
        _seed(session)

    # Under the floor -> onboarding is throttled, but the sense/score/
    # decide chain still runs (credits are checked "before the cycle
    # runs", not instead of it).
    client.fetch_api_key_usage.return_value = _usage_result(remaining_credits=100)
    decisions = run_once(client=client, database_url=db_url, credit_floor=10_000)

    client.fetch_api_key_usage.assert_called_once()
    onboard_mock.assert_not_called()
    assert len(decisions) == 2

    # Healthy credits -> onboarding is attempted normally.
    client.fetch_api_key_usage.return_value = _usage_result(remaining_credits=1_000_000)
    run_once(client=client, database_url=db_url, credit_floor=10_000)

    onboard_mock.assert_called_once()


def test_one_site_failing_does_not_skip_the_rest_of_the_cycle(tmp_path, mocker):
    """Regression test: an onboarding or sense failure for one site
    (confirmed live -- a real satellite segmentation task failed
    server-side during Phase 7's integration verification) used to
    propagate out of run_cycle and abort every remaining site. Both
    failure points are now caught per-site so the cycle continues.
    """
    mocker.patch("agent.act.notify.notify_slack")
    mocker.patch("agent.act.schedule.write_shift_override")
    mocker.patch("agent.act.compliance_report.generate_compliance_report")
    mocker.patch(
        "agent.loop.onboard_site",
        side_effect=TaskFailedError("Activity abc failed: {'status': 'Failed'}"),
    )

    client = MagicMock()
    client.create_heatmap.return_value = _heatmap_result(mean_temp_c=25.0, max_temp_c=26.0)
    client.environmental_parameters.return_value = _env_params_result(
        heat_index_c=25.0, humidity=40.0, ghi=300.0, aqi=20.0
    )

    with db_session(f"sqlite:///{tmp_path / 't.db'}") as session:
        site1, _ = _seed(session)
        site2, _ = _seed_second_site(session)

        decisions = run_cycle(session, client, skip_onboarding=False)

        # Onboarding failed for both sites, but sensing/scoring still ran
        # for both -- 2 crews x (live + forecast) = 4 decisions total, not
        # a crash after site1's onboarding failure.
        assert len(decisions) == 4
        site_ids_covered = set()
        for decision in decisions:
            score_row = session.get(Score, decision.score_id)
            reading_row = session.get(Reading, score_row.reading_id)
            site_ids_covered.add(reading_row.site_id)
        assert site_ids_covered == {site1.id, site2.id}


def test_onboarding_flush_failure_for_one_site_does_not_poison_the_rest_of_the_cycle(
    tmp_path, mocker
):
    """Code-review regression: SQLAlchemy 2.0's Session._flush() leaves the
    session INACTIVE on any flush error, requiring an explicit rollback
    before it's usable again -- but a plain session.rollback() would undo
    every *other* site's already-flushed work in the same cycle too, since
    nothing commits until db_session()'s single commit at the very end.
    Forces a genuine flush failure (a NOT NULL violation, not a mocked
    exception) for site 1's onboarding step and confirms site 2 still
    completes -- this is what a naive session.rollback() would get wrong.
    """
    mocker.patch("agent.act.notify.notify_slack")
    mocker.patch("agent.act.schedule.write_shift_override")
    mocker.patch("agent.act.compliance_report.generate_compliance_report")

    client = MagicMock()
    client.create_heatmap.return_value = _heatmap_result()
    client.environmental_parameters.return_value = _env_params_result()

    with db_session(f"sqlite:///{tmp_path / 't.db'}") as session:
        site1, _ = _seed(session)
        site2, _ = _seed_second_site(session)

        def _break_session_flush(client, site):
            session.add(
                Reading(
                    site_id=site.id,
                    ts=None,  # nullable=False -> real IntegrityError on flush
                    heat_index=1.0,
                    humidity=1.0,
                    solar_irradiance=1.0,
                    is_forecast=False,
                )
            )
            session.flush()

        mocker.patch("agent.loop.onboard_site", side_effect=_break_session_flush)

        decisions = run_cycle(session, client, skip_onboarding=False)

        # Both sites' one crew each got a live + forecast decision --
        # site 1's onboarding flush failure didn't take site 2 down too.
        assert len(decisions) == 4
        site_ids = set()
        for decision in decisions:
            score_row = session.get(Score, decision.score_id)
            reading_row = session.get(Reading, score_row.reading_id)
            site_ids.add(reading_row.site_id)
        assert site_ids == {site1.id, site2.id}


def test_sense_flush_failure_for_one_site_does_not_poison_the_rest_of_the_cycle(
    tmp_path, mocker
):
    """Same regression as above, forced inside the sense/score/decide
    except block instead of the onboarding one.
    """
    mocker.patch("agent.act.notify.notify_slack")
    mocker.patch("agent.act.schedule.write_shift_override")
    mocker.patch("agent.act.compliance_report.generate_compliance_report")

    client = MagicMock()
    client.create_heatmap.return_value = _heatmap_result()
    client.environmental_parameters.return_value = _env_params_result()

    with db_session(f"sqlite:///{tmp_path / 't.db'}") as session:
        site1, _ = _seed(session)
        site2, _ = _seed_second_site(session)

        real_sense_live = sense_live

        def _break_on_site1(client_, site):
            if site.id == site1.id:
                session.add(
                    Reading(
                        site_id=site.id,
                        ts=None,  # real IntegrityError on flush
                        heat_index=1.0,
                        humidity=1.0,
                        solar_irradiance=1.0,
                        is_forecast=False,
                    )
                )
                session.flush()
            return real_sense_live(client_, site)

        mocker.patch("agent.loop.sense_live", side_effect=_break_on_site1)

        decisions = run_cycle(session, client, skip_onboarding=True)

        # Site 1 fails entirely (its own flush error), but site 2's crew
        # still gets a live + forecast decision.
        assert len(decisions) == 2
        for decision in decisions:
            score_row = session.get(Score, decision.score_id)
            reading_row = session.get(Reading, score_row.reading_id)
            assert reading_row.site_id == site2.id


def test_one_crews_failure_does_not_erase_another_crews_already_written_live_decision(
    tmp_path, mocker
):
    """Code-review regression, empirically reproduced: unlike the forecast
    branch, the live branch's per-crew loop had no SAVEPOINT/try-except
    isolation of its own -- only the outer per-site one from run_cycle
    covered the whole branch. A later crew's decide_and_act failure would
    silently roll back an earlier crew's already-flushed Score/Decision
    rows too, so a real side effect already sent for that earlier crew
    (a Slack alert, a compliance report) would end up corresponding to no
    Decision row at all -- breaking decide.py's own documented "the
    Decision row still gets written, reflecting what actually happened"
    guarantee.
    """
    mocker.patch("agent.act.notify.notify_slack")
    mocker.patch("agent.act.schedule.write_shift_override")
    mocker.patch("agent.act.compliance_report.generate_compliance_report")

    client = MagicMock()
    client.create_heatmap.return_value = _heatmap_result()
    client.environmental_parameters.return_value = _env_params_result()

    with db_session(f"sqlite:///{tmp_path / 't.db'}") as session:
        site, crew1 = _seed(session)
        crew1_id = crew1.id
        crew2 = Crew(
            site_id=site.id,
            work_intensity="heavy",
            ppe_class="Class 2",
            active_shift_start=time(6, 0),
            active_shift_end=time(14, 0),
        )
        session.add(crew2)
        session.flush()
        crew2_id = crew2.id

        from agent.decide import decide_and_act as real_decide_and_act

        def _fail_second_crew_live(session_, client_, site_, crew, reading, score, tier, trigger_type):
            if crew.id == crew2_id and trigger_type == "live":
                raise RuntimeError("simulated decide_and_act failure for crew 2")
            return real_decide_and_act(session_, client_, site_, crew, reading, score, tier, trigger_type)

        mocker.patch("agent.loop.decide_and_act", side_effect=_fail_second_crew_live)

        run_cycle(session, client, skip_onboarding=True)

    with db_session(f"sqlite:///{tmp_path / 't.db'}") as session:
        live_decisions = [d for d in session.query(Decision).all() if d.trigger_type == "live"]
        crew_ids_with_a_live_decision = {
            session.get(Score, d.score_id).crew_id for d in live_decisions
        }
        # Crew 1's live decision must survive as a real, queryable row --
        # not just live in a Python list this test never even sees --
        # despite crew 2's failure in the very same loop.
        assert crew1_id in crew_ids_with_a_live_decision


def test_forecast_sensing_failure_still_returns_live_branch_decisions(tmp_path, mocker):
    """Code-review regression: sense_forecast raising after the live branch
    already succeeded used to propagate out of _run_site_cycle before its
    `return decisions`, silently discarding the live branch's already-
    decided rows from run_cycle's return value (the DB rows themselves
    were fine -- only the caller-visible count was wrong).
    """
    mocker.patch("agent.act.notify.notify_slack")
    mocker.patch("agent.act.schedule.write_shift_override")
    mocker.patch("agent.act.compliance_report.generate_compliance_report")
    mocker.patch("agent.loop.sense_forecast", side_effect=TaskFailedError("forecast AOI rejected"))

    client = MagicMock()
    client.create_heatmap.return_value = _heatmap_result(mean_temp_c=25.0)
    client.environmental_parameters.return_value = _env_params_result(
        heat_index_c=25.0, humidity=40.0, ghi=300.0, aqi=20.0
    )

    with db_session(f"sqlite:///{tmp_path / 't.db'}") as session:
        _seed(session)

        decisions = run_cycle(session, client, skip_onboarding=True)

        # One crew -> one live decision, even though the forecast branch
        # for the same site failed.
        assert len(decisions) == 1
        assert decisions[0].trigger_type == "live"


def test_run_once_decisions_survive_after_db_session_closes(tmp_path, mocker):
    """Code-review regression: db_session()'s sessionmaker defaulted to
    SQLAlchemy's expire_on_commit=True, so every attribute on the Decision
    rows run_once() returns was expired -- and then unreadable, since the
    session that could refresh them was already closed -- the instant a
    caller touched a field after the `with db_session(...)` block inside
    run_once() exited.
    """
    mocker.patch("agent.act.notify.notify_slack")
    mocker.patch("agent.act.schedule.write_shift_override")
    mocker.patch("agent.act.compliance_report.generate_compliance_report")
    mocker.patch("agent.loop.onboard_site")

    client = MagicMock()
    client.create_heatmap.return_value = _heatmap_result(mean_temp_c=25.0, max_temp_c=26.0)
    client.environmental_parameters.return_value = _env_params_result(
        heat_index_c=25.0, humidity=40.0, ghi=300.0, aqi=20.0
    )
    client.fetch_api_key_usage.return_value = _usage_result(remaining_credits=1_000_000)

    db_url = f"sqlite:///{tmp_path / 't.db'}"
    with db_session(db_url) as session:
        _seed(session)

    decisions = run_once(client=client, database_url=db_url, credit_floor=10_000)

    # No db_session() block open here -- this must not raise
    # DetachedInstanceError.
    assert decisions[0].action_taken == "none"
    assert decisions[0].trigger_type in ("live", "forecast")


def test_check_credits_does_not_throttle_on_unusable_response():
    """Code-review regression: `remaining >= credit_floor` sat outside
    _check_credits' try/except, so a present-but-non-numeric
    remaining_credits value raised an uncaught TypeError instead of
    falling back to "don't throttle" like the docstring promises for any
    unrecognized response.
    """
    client = MagicMock()
    client.fetch_api_key_usage.return_value = {"remaining_credits": "not-a-number"}

    assert _check_credits(client, credit_floor=1000) is True


def test_run_scheduler_rejects_zero_check_credits_every_n_cycles():
    """Code-review regression: state["cycle_count"] % check_credits_every_n_cycles
    raises ZeroDivisionError inside the recurring job (only surfacing once
    the scheduler actually ticks) if this is ever misconfigured as 0.
    """
    with pytest.raises(ValueError):
        run_scheduler(check_credits_every_n_cycles=0)


def test_run_scheduler_uses_utc_timezone(mocker, monkeypatch, tmp_path):
    """Code-review regression: BackgroundScheduler() with no explicit
    timezone defaults to the host's local timezone, so the naive-UTC
    now_naive_utc() passed as next_run_time gets localized as local time
    instead of UTC -- on a host west of UTC, "run immediately at startup"
    silently turns into "run several hours late".

    Isolated DATABASE_URL like every other test here (not the default
    local dev DB): run_scheduler() now claims a cross-process
    SchedulerLock row (agent/models.py) before starting anything, and a
    shared, unisolated DB file could carry lock state left over from a
    previous test/run, incidentally blocking this one.
    """
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 't.db'}")
    mock_scheduler_cls = mocker.patch("apscheduler.schedulers.background.BackgroundScheduler")
    mocker.patch("agent.loop.FortyGuardClient")

    run_scheduler()

    _, kwargs = mock_scheduler_cls.call_args
    assert kwargs.get("timezone") == "UTC"
