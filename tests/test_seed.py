"""Tests for agent.seed.seed_demo_sites_if_empty.

Streamlit Cloud's container filesystem starts empty on a fresh deploy
(and may not persist across restarts), so dashboard/app.py's bootstrap
calls this once on every boot to guarantee real, judge-visible sites
exist -- idempotent, so it must be a safe no-op once any site is present.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from agent.models import Crew, Site, db_session
from agent.seed import _DEMO_SITES, seed_demo_sites_if_empty


def test_seed_demo_sites_if_empty_creates_sites_with_crews(tmp_path):
    with db_session(f"sqlite:///{tmp_path / 't.db'}") as session:
        created = seed_demo_sites_if_empty(session)

        assert created > 0
        sites = session.query(Site).all()
        assert len(sites) == created
        for site in sites:
            # Pre-set so a fresh deployment's first automatic cycle senses
            # real live heat data immediately without also silently
            # auto-spending Premium onboarding credits the moment the
            # process boots -- see the module docstring.
            assert site.shade_coverage_pct is not None
            assert site.canopy_pct is not None
            crews = session.query(Crew).filter_by(site_id=site.id).all()
            assert len(crews) == 1


def test_seed_demo_sites_if_empty_is_a_no_op_once_any_site_exists(tmp_path):
    database_url = f"sqlite:///{tmp_path / 't.db'}"

    with db_session(database_url) as session:
        session.add(
            Site(
                name="Pre-existing site",
                lat=0.0,
                lon=0.0,
                polygon_geojson={"type": "Polygon", "coordinates": [[[0, 0]]]},
            )
        )

    with db_session(database_url) as session:
        created = seed_demo_sites_if_empty(session)

        assert created == 0
        assert len(session.query(Site).all()) == 1


def test_seed_demo_sites_if_empty_does_not_duplicate_under_concurrent_calls(tmp_path):
    """Live-observed real bug: concurrent Streamlit reruns could each pass
    the "any site exists" check on a still-empty database before either
    had committed its insert, resulting in duplicate demo sites (e.g. two
    "Phoenix, AZ ..." rows -- one accumulating real readings, the other
    stuck at "no live reading yet" forever, both visible on the
    dashboard). A lock serializes the check-then-insert across threads
    within this process (Streamlit runs each session's script on its own
    thread within one process), closing the race for the common case.
    """
    database_url = f"sqlite:///{tmp_path / 't.db'}"

    def _seed_once(_: int) -> None:
        with db_session(database_url) as session:
            seed_demo_sites_if_empty(session)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(_seed_once, range(8)))

    with db_session(database_url) as session:
        names = [site.name for site in session.query(Site).all()]

    assert len(names) == len(_DEMO_SITES)
    assert len(names) == len(set(names))
