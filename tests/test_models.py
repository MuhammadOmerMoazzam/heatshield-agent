"""Tests for agent.models -- schema smoke test plus a concurrency
regression test for the session-factory cache's first-use race.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, time

from agent.models import Crew, Decision, Reading, Score, Site, db_session


def test_create_and_read_one_row_per_table(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'smoke.db'}"

    with db_session(database_url) as session:
        site = Site(
            name="Diridon San Jose",
            lat=37.3297,
            lon=-121.9027,
            polygon_geojson={"type": "Polygon", "coordinates": [[[0, 0]]]},
            shade_coverage_pct=42.0,
            canopy_pct=18.5,
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

        reading = Reading(
            site_id=site.id,
            ts=datetime(2026, 8, 3, 14, 0),
            heat_index=41.2,
            aqi=55.0,
            humidity=38.0,
            solar_irradiance=810.0,
            is_forecast=False,
        )
        session.add(reading)
        session.flush()

        score = Score(
            reading_id=reading.id,
            crew_id=crew.id,
            final_score=0.82,
            risk_tier="high",
        )
        session.add(score)
        session.flush()

        decision = Decision(
            score_id=score.id,
            action_taken="shorten_shift_and_notify",
            executed_at=datetime(2026, 8, 3, 14, 5),
            report_url=None,
            notified_channel="slack",
            trigger_type="live",
        )
        session.add(decision)

    with db_session(database_url) as session:
        sites = session.query(Site).all()
        crews = session.query(Crew).all()
        readings = session.query(Reading).all()
        scores = session.query(Score).all()
        decisions = session.query(Decision).all()

        assert len(sites) == 1
        assert sites[0].name == "Diridon San Jose"
        assert sites[0].polygon_geojson["type"] == "Polygon"

        assert len(crews) == 1
        assert crews[0].site_id == sites[0].id
        assert crews[0].work_intensity == "heavy"
        assert crews[0].active_shift_start == time(6, 0)

        assert len(readings) == 1
        assert readings[0].site_id == sites[0].id
        assert readings[0].heat_index == 41.2
        assert readings[0].is_forecast is False

        assert len(scores) == 1
        assert scores[0].reading_id == readings[0].id
        assert scores[0].crew_id == crews[0].id
        assert scores[0].risk_tier == "high"

        assert len(decisions) == 1
        assert decisions[0].score_id == scores[0].id
        assert decisions[0].trigger_type == "live"
        assert decisions[0].report_url is None


def test_concurrent_first_calls_do_not_race_on_table_creation(tmp_path):
    """Regression test: concurrent first-time db_session() calls against the
    same not-yet-cached URL used to race on Base.metadata.create_all(),
    intermittently raising "table already exists" and dropping rows.
    """
    database_url = f"sqlite:///{tmp_path / 'concurrent.db'}"

    def _write_one(i: int) -> None:
        with db_session(database_url) as session:
            session.add(
                Site(
                    name=f"Site {i}",
                    lat=0.0,
                    lon=0.0,
                    polygon_geojson={"type": "Polygon", "coordinates": [[[0, 0]]]},
                )
            )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(_write_one, range(8)))

    with db_session(database_url) as session:
        assert len(session.query(Site).all()) == 8
