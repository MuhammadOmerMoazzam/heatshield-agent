"""Streamlit read-only dashboard for HeatShield Agent (Phase 8) plus the
Phase 9 single-process bootstrap.

Every render/query function below reads sites/readings/scores/decisions
from the DB only -- FortyGuardClient is never imported by name in this
module at all (enforced by tests/test_dashboard.py). The one place
credits get spent from this process is bootstrap_agent_loop() below,
which starts agent.loop.run_scheduler() -- itself the only thing in this
module that reaches (indirectly, via agent.loop) into FortyGuardClient,
tested separately in tests/test_bootstrap.py per [SS-C.7]'s own carve-out.

Query functions pull everything they need into plain dicts/lists *inside*
their db_session() block and return that, rather than handing back ORM
objects -- db_session() commits (and therefore expires attributes) at the
end of its `with` block, so any lazy attribute access after that point
would hit a closed session. Keeping that boundary explicit also keeps the
render functions themselves free of any DB/ORM coupling.

Streamlit Cloud runs a single process with no separate always-on service
for the agent loop (Part E), so this module starts it itself on load. A
module-level flag guards against starting it twice: Streamlit reruns a
session's script top-to-bottom on every interaction, and one process can
serve multiple concurrent sessions -- st.session_state is scoped to a
single session, so it wouldn't stop a *second browser tab* from starting
a second, competing scheduler thread. A module-level flag is shared
across every session in this process, which is the guarantee actually
needed ("once per process," not "once per session").
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

# Streamlit Cloud's launcher does not put this repo's root on sys.path
# the way a plain local `streamlit run` invocation does -- confirmed live:
# this module deployed with "ModuleNotFoundError: No module named 'agent'"
# even though the exact same clone/install/run sequence worked locally.
# agent/ and dashboard/ are source directories here, never actually
# `pip install`-ed as packages (Cloud's build step only runs
# `pip install -r requirements.txt`, never `pip install .`), so their
# importability depends entirely on sys.path -- computed here from
# __file__ rather than relied on implicitly, so it's correct no matter
# what CWD or launch mechanism actually runs this script.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import streamlit as st

from agent.loop import run_scheduler
from agent.models import Decision, Reading, Score, Site, db_session
from agent.seed import seed_demo_sites_if_empty

_bootstrap_lock = threading.Lock()
_scheduler_started = False
_scheduler_handle = None


def _load_secrets_into_environ() -> None:
    """Copy Streamlit Cloud secrets (st.secrets) into os.environ, once per
    call, without overriding anything already set.

    st.secrets is a separate mechanism from os.environ -- Cloud secrets
    configured in its UI are NOT injected into the environment
    automatically, so every other module's plain os.getenv(...) calls
    (agent/fortyguard_client.py, agent/act/notify.py, agent/models.py --
    the pattern the build plan specifies for local dev via python-dotenv)
    would silently see nothing on Cloud even with secrets correctly set.
    Doing this once, here, keeps agent/ itself environment-agnostic -- it
    never needs to know about Streamlit, on Cloud or locally.

    Locally (a .env file, no .streamlit/secrets.toml) st.secrets raises
    rather than returning empty -- that's the normal local-dev case, not
    an error, so it's swallowed: .env's values are already in os.environ
    by the time this runs (python-dotenv's load_dotenv(), called at
    import time in fortyguard_client.py/models.py). Existing os.environ
    values always win, mirroring load_dotenv()'s own documented
    precedence elsewhere in this codebase.
    """
    try:
        secrets_items = list(st.secrets.items())
    except Exception:
        return
    for key, value in secrets_items:
        os.environ.setdefault(key, str(value))


def bootstrap_agent_loop():
    """Start agent.loop.run_scheduler() exactly once per process.

    Safe to call on every rerun/every session -- a no-op after the first
    successful call. Double-checked locking: the lock is only taken on
    the (rare, first-ever) slow path, so repeated reruns within an
    already-bootstrapped process pay no locking cost.
    """
    global _scheduler_started, _scheduler_handle
    if _scheduler_started:
        return _scheduler_handle
    with _bootstrap_lock:
        if not _scheduler_started:
            _scheduler_handle = run_scheduler()
            _scheduler_started = True
    return _scheduler_handle

RISK_TIER_COLORS = {
    "safe": "#2e7d32",
    "caution": "#f9a825",
    "high": "#ef6c00",
    "extreme": "#c62828",
}


def query_site_dashboard_data(session) -> list[dict]:
    """One dict per site: latest live + forecast reading, per-crew scores
    on the latest live reading, and the onboarding segmentation media.
    """
    results = []
    for site in session.query(Site).order_by(Site.name).all():
        latest_live = (
            session.query(Reading)
            .filter_by(site_id=site.id, is_forecast=False)
            .order_by(Reading.ts.desc())
            .first()
        )
        latest_forecast = (
            session.query(Reading)
            .filter_by(site_id=site.id, is_forecast=True)
            .order_by(Reading.ts.desc())
            .first()
        )
        crew_scores = []
        if latest_live is not None:
            for score in session.query(Score).filter_by(reading_id=latest_live.id).all():
                crew_scores.append(
                    {
                        "crew_id": score.crew_id,
                        "final_score": score.final_score,
                        "risk_tier": score.risk_tier,
                    }
                )
        results.append(
            {
                "id": site.id,
                "name": site.name,
                "lat": site.lat,
                "lon": site.lon,
                "shade_coverage_pct": site.shade_coverage_pct,
                "canopy_pct": site.canopy_pct,
                "satellite_image_path": site.satellite_image_path,
                "satellite_legend": site.satellite_legend,
                "streetview_image_path": site.streetview_image_path,
                "streetview_legend": site.streetview_legend,
                "latest_live_ts": latest_live.ts if latest_live else None,
                "latest_live_heat_index": latest_live.heat_index if latest_live else None,
                "latest_live_heatmap_geojson": latest_live.heatmap_geojson if latest_live else None,
                "latest_forecast_ts": latest_forecast.ts if latest_forecast else None,
                "latest_forecast_heat_index": (
                    latest_forecast.heat_index if latest_forecast else None
                ),
                "crew_scores": crew_scores,
            }
        )
    return _dedupe_sites_by_name(results)


def _dedupe_sites_by_name(results: list[dict]) -> list[dict]:
    """Guards against duplicate Site rows sharing the same name -- a real
    bug hit live: concurrent Streamlit reruns could both pass
    seed_demo_sites_if_empty's "any site exists" check on an empty
    database before either committed, inserting the demo sites twice
    (fixed at the source in agent/seed.py, but this keeps the dashboard
    correct against rows already duplicated before that fix, or any
    other future cause). Prefers whichever duplicate actually has a live
    reading; order follows first-seen name (results already arrive
    sorted by name).
    """
    by_name: dict[str, dict] = {}
    for entry in results:
        existing = by_name.get(entry["name"])
        if existing is None or (
            existing["latest_live_ts"] is None and entry["latest_live_ts"] is not None
        ):
            by_name[entry["name"]] = entry
    return list(by_name.values())


def query_decision_log(session, limit: int = 200) -> list[dict]:
    """Most-recent-first decision/audit log, joined out to site + crew for
    display -- this table IS the audit log the README promises (Phase 6).
    """
    rows = (
        session.query(Decision, Score, Site)
        .join(Score, Decision.score_id == Score.id)
        .join(Reading, Score.reading_id == Reading.id)
        .join(Site, Reading.site_id == Site.id)
        .order_by(Decision.executed_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": decision.id,
            "executed_at": decision.executed_at,
            "site_name": site.name,
            "crew_id": score.crew_id,
            "risk_tier": score.risk_tier,
            "action_taken": decision.action_taken,
            "trigger_type": decision.trigger_type,
            "notified_channel": decision.notified_channel,
            "report_url": decision.report_url,
        }
        for decision, score, site in rows
    ]


def render_risk_badge(risk_tier: str) -> str:
    color = RISK_TIER_COLORS.get(risk_tier, "#616161")
    return (
        f"<span style='background:{color};color:white;padding:2px 10px;"
        f"border-radius:10px;font-weight:600;font-size:0.85em'>{risk_tier.upper()}</span>"
    )


def render_legend(legend: dict | None) -> None:
    if not legend:
        return
    swatches = " &nbsp; ".join(
        f"<span style='background:{color};display:inline-block;width:11px;height:11px;"
        f"margin-right:4px;border-radius:2px;vertical-align:middle'></span>{cls}"
        for cls, color in legend.items()
    )
    st.markdown(f"<div style='font-size:0.8em'>{swatches}</div>", unsafe_allow_html=True)


def render_heatmap_panel(heatmap_geojson: dict | None) -> None:
    """Colors each tile by its own temperature and renders the tile
    FeatureCollection a heatmap call already returns (Phase 8) -- pydeck
    ships with streamlit itself, no extra dependency needed.
    """
    features = (heatmap_geojson or {}).get("features") or []
    temps = [
        f["properties"]["temperature"]
        for f in features
        if f.get("properties", {}).get("temperature") is not None
    ]
    if not temps:
        st.caption("No heatmap tile data available for this reading yet.")
        return

    import pydeck as pdk

    lo, hi = min(temps), max(temps)
    colored_features = []
    for feature in features:
        temp = feature.get("properties", {}).get("temperature")
        if temp is None or not feature.get("geometry"):
            continue
        frac = 0.5 if hi == lo else (temp - lo) / (hi - lo)
        colored_features.append(
            {
                **feature,
                "properties": {
                    **feature["properties"],
                    "fill_color": [int(255 * frac), 60, int(255 * (1 - frac)), 180],
                },
            }
        )
    if not colored_features:
        st.caption("Heatmap tiles present but have no geometry to render.")
        return

    first_ring = colored_features[0]["geometry"]["coordinates"][0]
    first_point = first_ring[0] if isinstance(first_ring[0], list) else first_ring
    layer = pdk.Layer(
        "GeoJsonLayer",
        {"type": "FeatureCollection", "features": colored_features},
        get_fill_color="properties.fill_color",
        stroked=False,
        pickable=True,
    )
    view_state = pdk.ViewState(longitude=first_point[0], latitude=first_point[1], zoom=13)
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, map_style=None))


def render_segmentation_images(site_data: dict) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Satellite segmentation (overhead canopy)")
        path = site_data["satellite_image_path"]
        if path and Path(path).is_file():
            st.image(path)
            render_legend(site_data["satellite_legend"])
        else:
            st.caption("Not yet onboarded.")
    with col2:
        st.caption("Street-view segmentation (ground-level shade)")
        path = site_data["streetview_image_path"]
        if path and Path(path).is_file():
            st.image(path)
            render_legend(site_data["streetview_legend"])
        else:
            st.caption("Not yet onboarded.")


def render_site_panel(site_data: dict) -> None:
    st.markdown(f"## {site_data['name']}")

    live_col, forecast_col, map_col = st.columns([1, 1, 2])
    with live_col:
        live_hi = site_data["latest_live_heat_index"]
        st.metric("Live heat index (°F)", f"{live_hi:.1f}" if live_hi is not None else "n/a")
        if site_data["crew_scores"]:
            for crew_score in site_data["crew_scores"]:
                st.markdown(
                    f"Crew {crew_score['crew_id']}: "
                    f"{render_risk_badge(crew_score['risk_tier'])} "
                    f"({crew_score['final_score']:.1f})",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No live reading yet.")
    with forecast_col:
        forecast_hi = site_data["latest_forecast_heat_index"]
        st.metric(
            "Forecast +12h heat index (°F)",
            f"{forecast_hi:.1f}" if forecast_hi is not None else "n/a",
        )
    with map_col:
        render_heatmap_panel(site_data["latest_live_heatmap_geojson"])

    render_segmentation_images(site_data)


def render_decision_log(decisions: list[dict]) -> None:
    st.subheader("Decision / audit log")
    if not decisions:
        st.caption("No decisions recorded yet.")
        return
    st.dataframe(decisions, use_container_width=True, height=400)
    render_compliance_report_downloads(decisions)


def render_compliance_report_downloads(decisions: list[dict]) -> None:
    """The compliance-report PDF only ever lands on this server's own
    local disk (Phase 6: FortyGuard's signed download link is single-use,
    so it's downloaded once rather than persisted/shared) -- Slack's
    incoming webhook can't attach it, so this is the only place it's
    actually reachable. report_url is a local filesystem path, not a
    served URL, so it's read and handed to the browser here rather than
    linked directly.
    """
    reports = [d for d in decisions if d.get("report_url")]
    if not reports:
        return

    st.caption("Compliance reports")
    for entry in reports:
        path = Path(entry["report_url"])
        label = f"{entry['site_name']} — crew {entry['crew_id']} ({entry['executed_at']})"
        if path.is_file():
            st.download_button(
                label=f"Download report — {label}",
                data=path.read_bytes(),
                file_name=path.name,
                mime="application/pdf",
                key=f"report_{entry['id']}",
            )
        else:
            st.caption(f"Report for {label} is no longer available on disk.")


def main() -> None:
    _load_secrets_into_environ()
    st.set_page_config(page_title="HeatShield Agent", layout="wide")

    # Before starting the scheduler: a fresh deployment's container
    # filesystem starts empty, so without this the very first cycle
    # would have zero sites to act on. Idempotent -- a no-op after the
    # first successful call.
    with db_session() as session:
        seed_demo_sites_if_empty(session)

    bootstrap_agent_loop()
    st.title("HeatShield Agent — Site Risk Dashboard")

    with db_session() as session:
        sites_data = query_site_dashboard_data(session)
        decisions = query_decision_log(session)

    if not sites_data:
        st.info("No sites seeded yet.")
    for site_data in sites_data:
        render_site_panel(site_data)
        st.divider()

    render_decision_log(decisions)


if __name__ == "__main__":
    main()
