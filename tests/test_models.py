"""Tests for agent.models -- schema smoke test plus a concurrency
regression test for the session-factory cache's first-use race.
"""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, time

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

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


def test_dashboard_media_fields_round_trip(tmp_path):
    """Phase 8 schema extension: sites carry onboarding segmentation image
    paths + legends, readings carry the heatmap tile GeoJSON -- both
    nullable, both needed for the dashboard's image/map panels.
    """
    database_url = f"sqlite:///{tmp_path / 'media.db'}"
    geojson = {"type": "FeatureCollection", "features": [{"properties": {"temperature": 35.0}}]}
    legend = {"tree": "#2e7d32", "building": "#616161"}

    with db_session(database_url) as session:
        site = Site(
            name="Media Test Site",
            lat=37.33,
            lon=-121.90,
            polygon_geojson={"type": "Polygon", "coordinates": [[[0, 0]]]},
            satellite_image_path="outputs/satellite_abc.png",
            satellite_legend=legend,
            streetview_image_path=None,
            streetview_legend=None,
        )
        session.add(site)
        session.flush()

        reading = Reading(
            site_id=site.id,
            ts=datetime(2026, 8, 3, 14, 0),
            heat_index=41.2,
            aqi=None,
            humidity=38.0,
            solar_irradiance=810.0,
            is_forecast=False,
            heatmap_geojson=geojson,
        )
        session.add(reading)

    with db_session(database_url) as session:
        site = session.query(Site).one()
        reading = session.query(Reading).one()

        assert site.satellite_image_path == "outputs/satellite_abc.png"
        assert site.satellite_legend == legend
        assert site.streetview_image_path is None
        assert site.streetview_legend is None
        assert reading.heatmap_geojson == geojson


def test_reading_humidity_and_solar_irradiance_are_nullable(tmp_path):
    """Live-verified (Phase 9): environmental_parameters can independently
    have no data at any attempted timestamp -- sense_live's fallback then
    returns a real temperature-based reading with humidity/solar_irradiance
    genuinely unknown for this cycle, not a guessed 0.0. NULL must be a
    valid, persistable state for both columns, distinguishable from a
    real measured 0.
    """
    database_url = f"sqlite:///{tmp_path / 'nullable.db'}"

    with db_session(database_url) as session:
        site = Site(
            name="Nullable Test Site",
            lat=37.33,
            lon=-121.90,
            polygon_geojson={"type": "Polygon", "coordinates": [[[0, 0]]]},
        )
        session.add(site)
        session.flush()

        reading = Reading(
            site_id=site.id,
            ts=datetime(2026, 8, 3, 14, 0),
            heat_index=95.0,
            aqi=None,
            humidity=None,
            solar_irradiance=None,
            is_forecast=False,
        )
        session.add(reading)

    with db_session(database_url) as session:
        reading = session.query(Reading).one()
        assert reading.humidity is None
        assert reading.solar_irradiance is None


def test_sqlite_engine_uses_wal_mode_and_a_busy_timeout(tmp_path):
    """Regression: a real sqlite3.OperationalError: database is locked
    crashed the deployed app -- the background scheduler's long-running
    writes (many sequential API calls per cycle, holding a transaction
    open the whole time) and the dashboard's own per-rerun queries were
    colliding on the same SQLite file. The default rollback-journal mode
    only allows one writer OR reader at a time; WAL mode allows
    concurrent readers alongside a writer, and a busy_timeout makes a
    connection retry for a while before raising instead of failing
    immediately on the first collision.
    """
    database_url = f"sqlite:///{tmp_path / 'wal.db'}"

    with db_session(database_url) as session:
        journal_mode = session.execute(text("PRAGMA journal_mode")).scalar()
        busy_timeout_ms = session.execute(text("PRAGMA busy_timeout")).scalar()

    assert journal_mode.lower() == "wal"
    assert busy_timeout_ms >= 10_000


def test_rolling_back_a_session_undoes_savepoints_already_released_within_it(tmp_path):
    """Real bug, code-review-caught and empirically reproduced: pysqlite's
    legacy (default) transaction handling implicitly commits when a
    RELEASE SAVEPOINT is issued instead of deferring to the enclosing
    transaction, so a SAVEPOINT that already exited cleanly (e.g.
    agent.loop.run_cycle's per-site `with session.begin_nested():`) was
    durably written to disk immediately -- even though every one of those
    call sites' own comments assert "nothing commits until db_session()'s
    single commit at the very end." A later, unrelated failure in the
    same db_session() block must still be able to undo everything,
    including work from savepoints that already released successfully --
    that's the entire reason those call sites use begin_nested() instead
    of committing directly.
    """
    database_url = f"sqlite:///{tmp_path / 'savepoint_rollback.db'}"

    try:
        with db_session(database_url) as session:
            with session.begin_nested():
                session.add(
                    Site(
                        name="A",
                        lat=0.0,
                        lon=0.0,
                        polygon_geojson={"type": "Polygon", "coordinates": [[[0, 0]]]},
                    )
                )
                session.flush()
            with session.begin_nested():
                session.add(
                    Site(
                        name="B",
                        lat=0.0,
                        lon=0.0,
                        polygon_geojson={"type": "Polygon", "coordinates": [[[0, 0]]]},
                    )
                )
                session.flush()
            raise RuntimeError("simulated later failure outside any savepoint")
    except RuntimeError:
        pass

    with db_session(database_url) as session:
        assert session.query(Site).count() == 0


def test_pre_existing_database_gets_a_unique_index_backfilled_onto_sites_name(tmp_path):
    """Code-review-caught: Site.name's unique=True (added this session to
    fix a live duplicate-site bug) is only enforced by
    Base.metadata.create_all(), which creates *missing* tables but never
    alters an already-existing one -- so a persisted heatshield.db that
    predates this constraint (Streamlit Cloud's filesystem has been
    directly observed to survive at least some reboots this session, not
    just theoretically) would silently keep allowing duplicate site
    names forever, with the duplicate-site fix never actually taking
    effect on that specific database file.

    Simulates that exact pre-existing database (a bare `sites` table
    created without the unique constraint, the shape any deployment from
    before this session's Phase 9 duplicate-site fix would have) and
    confirms opening it through db_session() backfills a real,
    enforced unique index -- not just a fresh from create_all() table.
    """
    db_path = tmp_path / "pre_existing.db"
    raw_conn = sqlite3.connect(str(db_path))
    raw_conn.execute(
        """
        CREATE TABLE sites (
            id INTEGER PRIMARY KEY,
            name VARCHAR NOT NULL,
            lat FLOAT NOT NULL,
            lon FLOAT NOT NULL,
            polygon_geojson TEXT NOT NULL,
            shade_coverage_pct FLOAT,
            canopy_pct FLOAT,
            created_at DATETIME NOT NULL,
            satellite_image_path VARCHAR,
            satellite_legend TEXT,
            streetview_image_path VARCHAR,
            streetview_legend TEXT
        )
        """
    )
    raw_conn.commit()
    raw_conn.close()

    database_url = f"sqlite:///{db_path}"
    with db_session(database_url) as session:
        session.add(
            Site(
                name="Duplicate Name",
                lat=0.0,
                lon=0.0,
                polygon_geojson={"type": "Polygon", "coordinates": [[[0, 0]]]},
            )
        )

    raised = False
    try:
        with db_session(database_url) as session:
            session.add(
                Site(
                    name="Duplicate Name",
                    lat=1.0,
                    lon=1.0,
                    polygon_geojson={"type": "Polygon", "coordinates": [[[1, 1]]]},
                )
            )
    except IntegrityError:
        raised = True

    assert raised, "sites.name should be enforced unique even on a pre-existing database"


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
