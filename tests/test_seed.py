"""Tests for agent.seed.seed_demo_sites_if_empty.

Streamlit Cloud's container filesystem starts empty on a fresh deploy
(and may not persist across restarts), so dashboard/app.py's bootstrap
calls this once on every boot to guarantee real, judge-visible sites
exist -- idempotent, so it must be a safe no-op once any site is present.
"""

from __future__ import annotations

from agent.models import Crew, Site, db_session
from agent.seed import seed_demo_sites_if_empty


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
