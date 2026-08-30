"""Demo-site seeding for a fresh deployment.

Streamlit Cloud's container filesystem starts empty on a fresh deploy,
and isn't guaranteed to persist across restarts/redeploys -- without
this, a live deployment would show an empty dashboard forever, since
there's no "add site" UI (Phase 8's dashboard is deliberately read-only)
and no shell access to seed one by hand. dashboard/app.py's bootstrap
calls seed_demo_sites_if_empty() once on every boot; it's a no-op the
instant any site exists, so it's safe to call unconditionally.

shade_coverage_pct/canopy_pct are pre-set to representative estimates
here, not derived from a real satellite/streetview segmentation call --
agent.onboarding.onboard_site() treats a site with shade data already
set as already-onboarded and skips it. Without this, the very first
automatic cycle on a fresh boot would silently spend real Premium
credits on satellite/streetview segmentation the moment the process
starts, with no one in the loop to approve it. The live heatmap/
env_params calls (the actual "ingest real FortyGuard data" demonstration)
still fire normally either way.
"""

from __future__ import annotations

import threading
from datetime import time

from sqlalchemy.exc import IntegrityError

from agent.models import Crew, Site

# Real bug, live-observed: concurrent Streamlit reruns (separate threads
# within one process) could each pass seed_demo_sites_if_empty's "any
# site exists" check on a still-empty database before either had
# committed its insert, resulting in duplicate demo sites. This lock
# narrows that race for the common (same-process) case; the Site.name
# UNIQUE constraint (agent/models.py) is what actually guarantees no
# duplicates land regardless -- a session only becomes visible to other
# sessions at commit, not at flush, so a lock alone (held only for the
# duration of this function, not through the caller's later commit)
# can't fully close the window on its own.
_seed_lock = threading.Lock()

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

_DEMO_SITES = [
    {
        "name": "Phoenix, AZ — Warehouse District",
        "lat": 33.4484,
        "lon": -112.0740,
        "polygon_geojson": PHOENIX_POLYGON,
        "shade_coverage_pct": 18.0,
        "canopy_pct": 9.0,
        "crew": {
            "work_intensity": "heavy",
            "ppe_class": "Class 2",
            "active_shift_start": time(6, 0),
            "active_shift_end": time(14, 0),
        },
    },
    {
        "name": "Houston, TX — Port District",
        "lat": 29.7604,
        "lon": -95.3698,
        "polygon_geojson": HOUSTON_POLYGON,
        "shade_coverage_pct": 32.0,
        "canopy_pct": 15.0,
        "crew": {
            "work_intensity": "moderate",
            "ppe_class": "Class 1",
            "active_shift_start": time(6, 0),
            "active_shift_end": time(14, 0),
        },
    },
]


def seed_demo_sites_if_empty(session) -> int:
    """Seed the standard demo sites (+one crew each) if no site exists yet.

    Returns the number of sites created (0 if the table already had at
    least one -- this is what makes repeated calls, e.g. on every
    Streamlit rerun, safe). See module-level _seed_lock and
    Site.name's UNIQUE constraint (agent/models.py) for why a duplicate
    insert from a concurrent caller is caught and treated as "already
    seeded" rather than raising.
    """
    with _seed_lock:
        if session.query(Site).first() is not None:
            return 0

        try:
            with session.begin_nested():
                for spec in _DEMO_SITES:
                    site = Site(
                        name=spec["name"],
                        lat=spec["lat"],
                        lon=spec["lon"],
                        polygon_geojson=spec["polygon_geojson"],
                        shade_coverage_pct=spec["shade_coverage_pct"],
                        canopy_pct=spec["canopy_pct"],
                    )
                    session.add(site)
                    session.flush()
                    session.add(Crew(site_id=site.id, **spec["crew"]))
                session.flush()
        except IntegrityError:
            # A concurrent caller won the race and already committed
            # these sites (by name) between our check above and this
            # insert -- not an error.
            return 0

        return len(_DEMO_SITES)
