"""Integration tests for agent.loop.

The FortyGuardClient is mocked entirely (no real network), but every
other module -- sense, score, decide, act, models -- is real, exercising
the full sense -> score -> decide -> act chain end-to-end against a
tmp_path-scoped SQLite DB.
"""

from __future__ import annotations

from datetime import time
from unittest.mock import MagicMock

from agent.fortyguard_client import TaskFailedError
from agent.loop import run_cycle, run_once
from agent.models import Crew, Reading, Score, Site, db_session

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

        # Phase 8: the dashboard's map panel reads this straight off the
        # reading row -- the loop must persist it, not just sense.py return it.
        readings = session.query(Reading).all()
        assert len(readings) == 2
        for reading in readings:
            assert reading.heatmap_geojson["features"][0]["properties"]["temperature"] == 25.0

    notify_mock.assert_not_called()
    schedule_mock.assert_not_called()
    report_mock.assert_not_called()


def test_full_cycle_extreme_reading_produces_halt_alert_and_report(tmp_path, mocker):
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
        assert forecast_decision.action_taken == "halt_and_notify_and_report"

    assert notify_mock.call_count == 2
    assert schedule_mock.call_count == 2
    assert report_mock.call_count == 2


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
