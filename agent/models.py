"""SQLAlchemy schema for HeatShield Agent.

Five tables carrying the full sense -> score -> decide -> act loop:
sites, crews, readings, scores, decisions. `decisions` rows are the
project's audit log -- every action the agent takes writes one.
"""

from __future__ import annotations

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
    """

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.close()


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
