"""SQLAlchemy schema for HeatShield Agent.

Five tables carrying the full sense -> score -> decide -> act loop:
sites, crews, readings, scores, decisions. `decisions` rows are the
project's audit log -- every action the agent takes writes one.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime, timezone

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
)
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

DEFAULT_DATABASE_URL = "sqlite:///data/heatshield.db"

_session_factories: dict[str, sessionmaker] = {}


class Site(Base):
    __tablename__ = "sites"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    polygon_geojson = Column(JSON, nullable=False)
    # Nullable: a site onboarded before satellite/streetview segmentation
    # finishes has no shade data yet (Phase 4).
    shade_coverage_pct = Column(Float, nullable=True)
    canopy_pct = Column(Float, nullable=True)
    created_at = Column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
    )


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
    humidity = Column(Float, nullable=False)
    solar_irradiance = Column(Float, nullable=False)
    # Distinguishes the live reactive branch from the +12h forecast branch
    # (Part B.2) -- Phase 5/6 branch directly on this column.
    is_forecast = Column(Boolean, nullable=False, default=False)


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


def _get_session_factory(database_url: str) -> sessionmaker:
    if database_url not in _session_factories:
        if database_url.startswith("sqlite:///"):
            db_path = database_url[len("sqlite:///"):]
            if db_path and db_path != ":memory:":
                parent = os.path.dirname(db_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
        engine = create_engine(database_url)
        Base.metadata.create_all(engine)
        _session_factories[database_url] = sessionmaker(bind=engine)
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
