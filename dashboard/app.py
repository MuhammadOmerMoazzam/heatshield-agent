"""Streamlit read-only dashboard for HeatShield Agent (Phase 8).

Every function below reads sites/readings/scores/decisions from the DB
only -- FortyGuardClient must not be importable from this module at all
(enforced by tests/test_dashboard.py). The background-loop bootstrap that
starts agent.loop.run_scheduler() (the only legitimate place credits get
spent from a Streamlit process) is added separately in Phase 9.

Query functions pull everything they need into plain dicts/lists *inside*
their db_session() block and return that, rather than handing back ORM
objects -- db_session() commits (and therefore expires attributes) at the
end of its `with` block, so any lazy attribute access after that point
would hit a closed session. Keeping that boundary explicit also keeps the
render functions themselves free of any DB/ORM coupling.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from agent.models import Decision, Reading, Score, Site, db_session

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
    return results


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


def main() -> None:
    st.set_page_config(page_title="HeatShield Agent", layout="wide")
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
