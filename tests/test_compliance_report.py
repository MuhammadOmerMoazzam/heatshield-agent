"""Tests for agent.act.compliance_report.generate_compliance_report."""

from __future__ import annotations

from datetime import datetime, time
from unittest.mock import MagicMock

import pytest

from agent.act.compliance_report import generate_compliance_report
from agent.models import Crew, Reading, Site, db_session


def _site() -> Site:
    return Site(
        name="S",
        lat=37.33,
        lon=-121.90,
        polygon_geojson={"type": "Polygon", "coordinates": [[[0, 0]]]},
    )


def _reading(ts: datetime = datetime(2026, 8, 3, 14, 0), heat_index: float = 112.0, site_id: int = 1) -> Reading:
    return Reading(
        site_id=site_id,
        ts=ts,
        heat_index=heat_index,
        aqi=40.0,
        humidity=55.0,
        solar_irradiance=650.0,
        is_forecast=False,
    )


@pytest.mark.parametrize(
    "tier, expect_called",
    [("safe", False), ("caution", False), ("high", True), ("extreme", True)],
)
def test_heat_intelligence_called_only_on_high_or_extreme(tier, expect_called):
    client = MagicMock()
    client.heat_intelligence.return_value = "/outputs/report.pdf"

    generate_compliance_report(client, _site(), _reading(), tier)

    assert client.heat_intelligence.called is expect_called


@pytest.mark.parametrize("tier", ["high", "extreme"])
def test_report_url_attached_to_decision_record(tmp_path, mocker, tier):
    # Exercised through the real generate_compliance_report/heat_intelligence
    # call chain (only notify/schedule are mocked) for both report-eligible
    # tiers, not just "extreme" -- a tier check that broke only the "high"
    # path would otherwise go undetected (see decide.py's own docstring:
    # the report fires on both).
    from agent.decide import decide_and_act

    mocker.patch("agent.act.notify.notify_slack", return_value=None)
    mocker.patch("agent.act.schedule.write_shift_override")
    client = MagicMock()
    client.heat_intelligence.return_value = "/outputs/heat_intelligence_xyz.pdf"

    with db_session(f"sqlite:///{tmp_path / 't.db'}") as session:
        site = _site()
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
        reading = _reading(site_id=site.id)
        session.add(reading)
        session.flush()

        decision = decide_and_act(
            session, client, site, crew, reading, score=130.0, tier=tier, trigger_type="live"
        )

        assert decision.report_url == "/outputs/heat_intelligence_xyz.pdf"


def test_report_call_uses_timestamp_matching_the_triggering_heatmap_call():
    client = MagicMock()
    client.heat_intelligence.return_value = "/outputs/r.pdf"
    reading = _reading(ts=datetime(2026, 7, 20, 9, 0))

    generate_compliance_report(client, _site(), reading, "high")

    call = client.heat_intelligence.call_args
    assert call.kwargs["date"] == "2026-07-20"
