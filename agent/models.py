"""SQLAlchemy schema for HeatShield Agent.

Five tables carrying the full sense -> score -> decide -> act loop:
sites, crews, readings, scores, decisions. `decisions` rows are the
project's audit log -- every action the agent takes writes one.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager

from dotenv import load_dotenv
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Time,
    create_engine,
    event,
)
from sqlalchemy.engine import make_url
from sqlalchemy.orm import declarative_base, sessionmaker

from agent._shared import REPO_ROOT, now_naive_utc

logger = logging.getLogger(__name__)

# So DATABASE_URL from .env is actually picked up, not just documented in
# .env.example -- load_dotenv() only sets keys not already in os.environ,
# so a real shell-exported var still wins.
load_dotenv()

Base = declarative_base()

# Anchored to this package's location, not the process's CWD -- a
# scheduler job, `streamlit run`, or pytest launched from a different
# directory would otherwise silently read/write a different data/
# heatshield.db than intended.
DEFAULT_DATABASE_URL = f"sqlite:///{(REPO_ROOT / 'data' / 'heatshield.db').as_posix()}"

_session_factories: dict[str, sessionmaker] = {}
_session_factories_lock = threading.Lock()


class Site(Base):
    __tablename__ = "sites"

    id = Column(Integer, primary_key=True)
    # unique=True: real bug, live-observed -- concurrent Streamlit reruns
    # could each pass agent.seed's "any site exists" check on a still-
    # empty database before either committed, inserting the demo sites
    # twice. This constraint is the authoritative, database-enforced
    # guard against that (an in-process lock alone narrows the race but
    # can't close it, since visibility to other sessions only happens at
    # commit, not at flush) -- agent.seed catches the resulting
    # IntegrityError and treats it as "already seeded," not an error.
    name = Column(String, nullable=False, unique=True)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    polygon_geojson = Column(JSON, nullable=False)
    # Nullable: a site onboarded before satellite/streetview segmentation
    # finishes has no shade data yet (Phase 4).
    shade_coverage_pct = Column(Float, nullable=True)
    canopy_pct = Column(Float, nullable=True)
    created_at = Column(DateTime, nullable=False, default=now_naive_utc)
    # Populated by onboarding (Phase 4/8): decoded, saved-to-disk
    # segmentation overlays + their {class_name: hex_color} legends, for
    # the dashboard's segmentation panels (Phase 8). Nullable like
    # shade_coverage_pct/canopy_pct -- a site onboarded before satellite/
    # streetview finish, or that returned no image, has none of these yet.
    satellite_image_path = Column(String, nullable=True)
    satellite_legend = Column(JSON, nullable=True)
    streetview_image_path = Column(String, nullable=True)
    streetview_legend = Column(JSON, nullable=True)
    # Real bug, live-observed: agent.onboarding.onboard_site() used to key
    # its "already onboarded" check off shade_coverage_pct being non-NULL
    # -- but agent.seed.py deliberately pre-fills that same column with a
    # placeholder estimate (to avoid spending Premium credits on the very
    # first boot before anyone can review it), so a seeded demo site was
    # indistinguishable from a genuinely onboarded one and real
    # satellite/streetview segmentation never ran for it, confirmed live
    # via FortyGuard's own usage breakdown showing zero segmentation calls
    # ever billed. onboarded_at is set only by a real, successful
    # onboard_site() call (never by seeding), so it's the one unambiguous
    # signal of "has this site actually been onboarded".
    onboarded_at = Column(DateTime, nullable=True)


class Crew(Base):
    __tablename__ = "crews"

    id = Column(Integer, primary_key=True)
    site_id = Column(Integer, ForeignKey("sites.id"), nullable=False)
    work_intensity = Column(String, nullable=False)
    ppe_class = Column(String, nullable=False)
    active_shift_start = Column(Time, nullable=False)
    active_shift_end = Column(Time, nullable=False)


class Reading(Base):
    __tablename__ = "readings"

    id = Column(Integer, primary_key=True)
    site_id = Column(Integer, ForeignKey("sites.id"), nullable=False)
    ts = Column(DateTime, nullable=False)
    heat_index = Column(Float, nullable=False)
    aqi = Column(Float, nullable=True)
    # Nullable: environmental_parameters can independently have no data
    # at any of sense_live's attempted timestamps (live-verified, Phase
    # 9) -- NULL means "genuinely unknown this cycle", distinguishable
    # from a real measured 0.
    humidity = Column(Float, nullable=True)
    solar_irradiance = Column(Float, nullable=True)
    # Distinguishes the live reactive branch from the +12h forecast branch
    # (Part B.2) -- Phase 5/6 branch directly on this column.
    is_forecast = Column(Boolean, nullable=False, default=False)
    # The heatmap call's own tile FeatureCollection (Phase 8's dashboard map
    # panel) -- nullable since older readings predate this column and a
    # mocked/degraded heatmap response may carry no map_data.
    heatmap_geojson = Column(JSON, nullable=True)


class Score(Base):
    __tablename__ = "scores"

    id = Column(Integer, primary_key=True)
    reading_id = Column(Integer, ForeignKey("readings.id"), nullable=False)
    crew_id = Column(Integer, ForeignKey("crews.id"), nullable=False)
    final_score = Column(Float, nullable=False)
    risk_tier = Column(String, nullable=False)


class Decision(Base):
    __tablename__ = "decisions"

    id = Column(Integer, primary_key=True)
    score_id = Column(Integer, ForeignKey("scores.id"), nullable=False)
    action_taken = Column(String, nullable=False)
    executed_at = Column(DateTime, nullable=False)
    # Only set on high/extreme tiers (Phase 6 compliance report).
    report_url = Column(String, nullable=True)
    notified_channel = Column(String, nullable=True)
    trigger_type = Column(String, nullable=False)


class SchedulerLock(Base):
    """Singleton row (fixed id=1) guarding against more than one live
    agent.loop.run_scheduler() actually running against this database.

    Real production bug, live-observed: a Streamlit-Cloud-only in-memory
    guard (dashboard/app.py's module-level _scheduler_started flag)
    assumes "one process" -- but live logs showed the same site failing
    far more often than a 6-hour interval allows (5 failures, each
    costing close to a 300s timeout, inside an 11-minute window: not
    enough time for that many *sequential* failures), meaning multiple
    schedulers were running concurrently against the same credits. The
    most likely cause is Streamlit Cloud not always fully killing a prior
    process's threads across the many reboots/redeploys this project has
    gone through -- an in-memory flag in a NEW process can't know about
    an OLD process's still-running scheduler; the database file is the
    one thing confirmed to be genuinely shared across however many of
    "these" exist. heartbeat_at (not a one-time claim-and-forget) lets a
    genuinely dead orphan's lock be reclaimed later rather than
    permanently bricking the scheduler after a clean reboot -- see
    agent.loop._claim_scheduler_lock/_renew_scheduler_lock.
    """

    __tablename__ = "scheduler_lock"

    id = Column(Integer, primary_key=True)
    started_at = Column(DateTime, nullable=False)
    heartbeat_at = Column(DateTime, nullable=False)


def _resolve_database_url() -> str:
    return os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)


def _configure_sqlite_for_concurrent_access(engine) -> None:
    """Real production bug: 'sqlite3.OperationalError: database is
    locked' crashed the deployed app -- the background scheduler's
    run_cycle() holds a write transaction open across many sequential,
    slow API calls, and the dashboard's own per-Streamlit-rerun queries
    collided with it on the same SQLite file. SQLite's default
    rollback-journal mode only allows one writer OR reader at a time.
    WAL mode allows concurrent readers alongside a writer; busy_timeout
    makes a connection retry for a while before raising, rather than
    failing immediately on the first collision.

    Also disables pysqlite's own legacy transaction handling (isolation_
    level=None) and takes over BEGIN ourselves. Second real bug, code-
    review-caught and empirically reproduced: without this, pysqlite
    implicitly commits on RELEASE SAVEPOINT instead of deferring to
    whatever transaction is still open around it -- every begin_nested()
    call site in this codebase (agent/loop.py's onboarding/per-site/
    forecast SAVEPOINTs, agent/seed.py's insert SAVEPOINT) assumes a
    SAVEPOINT that exits cleanly is still undoable by a *later* failure
    in the same db_session() block, which silently wasn't true. This is
    SQLAlchemy's own documented pysqlite workaround (see "Serializable
    isolation / Savepoints / Transactional DDL" in the sqlite dialect
    docs), not a local invention.
    """

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        dbapi_connection.isolation_level = None
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.close()

    @event.listens_for(engine, "begin")
    def _sqlite_begin(conn):
        conn.exec_driver_sql("BEGIN")


def _backfill_sites_name_unique_index(engine) -> None:
    """Code-review-caught: Site.name's unique=True (added this session to
    fix a live duplicate-site bug) is only enforced by
    Base.metadata.create_all() below, which creates *missing* tables but
    never alters an already-existing one -- so a persisted heatshield.db
    predating this constraint would silently keep allowing duplicate
    site names forever, with that fix never actually taking effect on
    that specific database file. Streamlit Cloud's filesystem has been
    directly observed (not just theoretical) to survive at least some
    reboots this session, so this isn't a hypothetical case.

    A standalone UNIQUE INDEX enforces the same guarantee as a UNIQUE
    column constraint without needing SQLite's full rebuild-the-table
    ALTER TABLE dance -- CREATE UNIQUE INDEX works against an
    already-populated table exactly like create_all() works against a
    missing one. If duplicate-named rows already exist on this specific
    database (e.g. from before this session's seeding-race fix), the
    index creation itself fails -- logged, not raised, since crashing
    the whole app on boot over data dashboard/app.py's own
    _dedupe_sites_by_name already papers over would be a worse outcome
    than the status quo.
    """
    try:
        with engine.begin() as conn:
            conn.exec_driver_sql(
                "CREATE UNIQUE INDEX IF NOT EXISTS ix_sites_name_unique ON sites(name)"
            )
    except Exception:
        logger.warning(
            "Could not create a unique index on sites.name -- this database file "
            "likely already has duplicate-named rows predating this session's "
            "duplicate-site fix. New-insert protection is degraded until those "
            "are cleaned up by hand; dashboard/app.py's own dedupe-by-name still "
            "keeps the dashboard itself correct in the meantime.",
            exc_info=True,
        )


def _backfill_missing_nullable_columns(engine) -> None:
    """Generic version of a fix first written narrowly for
    sites.onboarded_at: any nullable column added to a model after its
    table already exists (onboarded_at; Site.satellite_image_path/
    satellite_legend/streetview_image_path/streetview_legend and
    Reading.heatmap_geojson from Phase 8, both predating this generic
    version and never backfilled) is only added by
    Base.metadata.create_all() below, which creates *missing* tables but
    never alters an already-existing one -- so a persisted heatshield.db
    predating any of these columns would raise "no such column" on every
    query that touches it. A plain nullable column with no default is
    exactly what SQLite's own ALTER TABLE ADD COLUMN supports directly --
    no full-table-rebuild workaround needed for this case (see
    _backfill_readings_nullable_environmental_columns below for the case
    that does need one). Runs once per table, for every column declared
    on the model but absent from the live table.
    """
    try:
        with engine.begin() as conn:
            for table in Base.metadata.sorted_tables:
                existing_columns = {
                    row[1]
                    for row in conn.exec_driver_sql(f"PRAGMA table_info({table.name})").fetchall()
                }
                if not existing_columns:
                    continue  # table itself doesn't exist yet -- create_all()'s job
                for column in table.columns:
                    if column.name in existing_columns or not column.nullable:
                        continue
                    column_type = column.type.compile(dialect=conn.dialect)
                    conn.exec_driver_sql(
                        f"ALTER TABLE {table.name} ADD COLUMN {column.name} {column_type}"
                    )
    except Exception:
        logger.warning(
            "Could not backfill missing nullable columns onto an existing "
            "database -- some features may be unavailable on this specific "
            "database file until this is resolved by hand.",
            exc_info=True,
        )


def _backfill_readings_nullable_environmental_columns(engine) -> None:
    """Real bug, deferred earlier this session pending a test proving it
    first (see tests/test_models.py's
    test_pre_existing_database_relaxes_not_null_on_readings_humidity_and_solar_irradiance):
    Reading.humidity/solar_irradiance were originally NOT NULL (Phase 2)
    and later relaxed to nullable=True (Phase 9's live fix, once
    environmental_parameters was confirmed to sometimes return no data at
    all for a given cycle). SQLite bakes NOT NULL into the column itself
    -- changing the Python model doesn't relax an already-existing table,
    and unlike a missing column, SQLite can't ALTER a column's constraint
    in place. The standard workaround is a table rebuild: create the
    table fresh under today's (correct) shape, copy every row across
    using only the columns the old table actually had, then swap it in
    for the original.
    """
    try:
        with engine.begin() as conn:
            columns = conn.exec_driver_sql("PRAGMA table_info(readings)").fetchall()
            if not columns:
                return  # table doesn't exist yet -- create_all()'s job
            by_name = {row[1]: row for row in columns}
            still_not_null = any(
                by_name[name][3] == 1  # PRAGMA table_info's notnull flag
                for name in ("humidity", "solar_irradiance")
                if name in by_name
            )
            if not still_not_null:
                return

            conn.exec_driver_sql("ALTER TABLE readings RENAME TO readings_pre_nullable_migration")
            Reading.__table__.create(conn)
            shared_columns = ", ".join(
                name for name in by_name if name in Reading.__table__.columns.keys()
            )
            conn.exec_driver_sql(
                f"INSERT INTO readings ({shared_columns}) "
                f"SELECT {shared_columns} FROM readings_pre_nullable_migration"
            )
            conn.exec_driver_sql("DROP TABLE readings_pre_nullable_migration")
    except Exception:
        logger.warning(
            "Could not relax NOT NULL on readings.humidity/solar_irradiance "
            "for an existing database -- inserting a genuinely-unknown "
            "humidity/solar_irradiance reading may still fail on this "
            "specific database file until this is resolved by hand.",
            exc_info=True,
        )


def _get_session_factory(database_url: str) -> sessionmaker:
    # Fast path: no lock needed once a URL is cached. Double-checked
    # locking on the slow path -- without it, concurrent first-time calls
    # for the same URL (e.g. two APScheduler jobs starting together) each
    # create their own Engine and race on Base.metadata.create_all(),
    # which can crash with "table already exists" and drop rows.
    if database_url in _session_factories:
        return _session_factories[database_url]
    with _session_factories_lock:
        if database_url not in _session_factories:
            url_obj = make_url(database_url)
            if url_obj.get_backend_name() == "sqlite" and url_obj.database not in (None, ":memory:"):
                parent = os.path.dirname(url_obj.database)
                if parent:
                    os.makedirs(parent, exist_ok=True)
            engine = create_engine(database_url)
            if url_obj.get_backend_name() == "sqlite":
                _configure_sqlite_for_concurrent_access(engine)
            Base.metadata.create_all(engine)
            if url_obj.get_backend_name() == "sqlite":
                _backfill_sites_name_unique_index(engine)
                _backfill_missing_nullable_columns(engine)
                _backfill_readings_nullable_environmental_columns(engine)
            # expire_on_commit=False: db_session() closes its session
            # immediately after commit (see below), and callers like
            # agent.loop.run_once() return ORM rows straight out of that
            # `with` block -- with the default expire_on_commit=True,
            # every attribute on those rows would raise
            # DetachedInstanceError the instant a caller touched a field,
            # since the session that could re-fetch them is already gone.
            _session_factories[database_url] = sessionmaker(bind=engine, expire_on_commit=False)
    return _session_factories[database_url]


@contextmanager
def db_session(database_url: str | None = None):
    """Yield a SQLAlchemy session; commits on success, rolls back on error.

    Defaults to `DATABASE_URL` from the environment (a local SQLite file
    for dev). Pass `database_url` explicitly to target a different
    database, e.g. an isolated file in tests.
    """
    url = database_url or _resolve_database_url()
    session = _get_session_factory(url)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
