"""Sense: the first stage of the sense -> score -> decide -> act loop.

Two deliberately separate functions, per Rule 2: env_params must be
called against the *same* timestamp as a heatmap call, and there is no
guarantee it supports the +12h forecast window the way create_heatmap
does -- so the live (reactive) and forecast (proactive) data paths are
kept provably distinct in code, not merged into one function branching
on a flag.

- ``sense_live`` calls ``create_heatmap`` for "now", then
  ``environmental_parameters`` against that exact same timestamp, and
  returns a fully-scoreable ``RawReading``.
- ``sense_forecast`` calls ``create_heatmap`` for "now + 12h" *only* (no
  env_params call) and returns a lightweight ``ForecastSignal`` carrying
  just the forecasted peak temperature, for Decide's separate proactive
  branch.

Confirmed against the live API docs (docs-api.fortyguard.com/docs/
environmental-parameters): ``environmental_parameters`` requires a
``temperature`` input (°C) -- it isn't purely an output-only endpoint --
so ``sense_live`` uses the heatmap call's own measured mean temperature
for that input, rather than guessing a value. Per-parameter values in the
env_params response are time-aligned arrays, not scalars, and missing
values come back as JSON ``null`` (never to be read as zero) -- both
handled explicitly here rather than assumed away.

Production evidence (real Streamlit Cloud logs, Phase 9): a
``create_heatmap`` call for the literal current hour reliably succeeds
(200 OK) but returns ``stats_data`` with no ``temperature_stats`` --
FortyGuard's pipeline evidently hasn't ingested/aggregated that hour's
data yet. Observed 100% of the time across several hours of real
scheduled cycles, not a one-off flake. ``sense_live`` retries a few
hours further back (``LIVE_LOOKBACK_HOURS``) before giving up, and uses
whichever hour actually returns data -- for both the heatmap call and,
per Rule 2, the matching ``environmental_parameters`` call -- rather
than the originally-requested "now". The returned ``RawReading.ts``
honestly reflects which hour the data is actually from.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from pydantic import BaseModel

from agent._shared import now_naive_utc as _now_naive_utc
from agent.fortyguard_client import FortyGuardClient
from agent.models import Site


class RawReading(BaseModel):
    site_id: int
    ts: datetime
    heat_index: float  # degrees Fahrenheit (converted from the API's Celsius)
    aqi: float | None
    humidity: float  # percent
    solar_irradiance: float  # W/m^2 (clear-sky GHI)
    is_forecast: bool = False
    # The heatmap call's own tile FeatureCollection (Phase 8's dashboard
    # map panel) -- None if the response carried no map_data.
    heatmap_geojson: dict | None = None


class ForecastSignal(BaseModel):
    site_id: int
    ts: datetime
    max_temp_c: float
    heatmap_geojson: dict | None = None


class SenseDataUnavailableError(Exception):
    """A required environmental parameter came back null from the API, or
    no heatmap temperature data was available at any fallback hour tried.
    """


# How many hours back sense_live retries before giving up (0 = the
# literal current hour, tried first). Each entry is a separate,
# credit-metered create_heatmap call, so this stays small and bounded --
# not an unbounded/open-ended search -- rather than a blanket "keep
# retrying forever" policy.
LIVE_LOOKBACK_HOURS: tuple[int, ...] = (0, 1, 2)


def celsius_to_fahrenheit(celsius: float) -> float:
    return celsius * 9 / 5 + 32


def _first_or_raise(values: list | None, field: str) -> float:
    value = values[0] if values else None
    if value is None:
        raise SenseDataUnavailableError(
            f"env_params returned no data for '{field}' at the requested timestamp."
        )
    return value


def sense_live(client: FortyGuardClient, site: Site, *, now: datetime | None = None) -> RawReading:
    """Sense the current moment: a heatmap call for "now", falling back to
    a few hours earlier if that hour has no data yet (see module
    docstring), then env_params against whichever timestamp actually
    succeeded (Rule 2).
    """
    base_now = now or _now_naive_utc()

    mean_temp_c = None
    ts = None
    heatmap_geojson = None
    attempted_timestamps: list[datetime] = []

    for hours_back in LIVE_LOOKBACK_HOURS:
        candidate_ts = base_now - timedelta(hours=hours_back)
        attempted_timestamps.append(candidate_ts)

        heatmap_response = client.create_heatmap(
            site.polygon_geojson,
            start_date=candidate_ts.date().isoformat(),
            start_time=candidate_ts.strftime("%H:%M"),
            filter_type=1,
        )
        temperature_stats = heatmap_response.get("result", {}).get("stats_data", {}).get(
            "temperature_stats"
        )
        if temperature_stats is not None and temperature_stats.get("mean") is not None:
            mean_temp_c = temperature_stats["mean"]
            ts = candidate_ts
            heatmap_geojson = heatmap_response["result"].get("map_data")
            break

    if mean_temp_c is None:
        attempted_str = ", ".join(t.strftime("%Y-%m-%d %H:%M") for t in attempted_timestamps)
        raise SenseDataUnavailableError(
            f"No heatmap temperature data available for site_id={site.id} at any of the last "
            f"{len(LIVE_LOOKBACK_HOURS)} hourly attempts ({attempted_str}) -- FortyGuard's "
            "pipeline may not have ingested data for this recently yet."
        )

    env_response = client.environmental_parameters(
        site.lat,
        site.lon,
        temperature=mean_temp_c,
        reference_ts=ts,
    )
    location = env_response["result"]["locations"][0]
    parameters = location["parameters"]

    heat_index_c = _first_or_raise(parameters.get("heat_index_celsius"), "heat_index_celsius")
    humidity = _first_or_raise(
        parameters.get("relative_humidity_percent"), "relative_humidity_percent"
    )
    ghi = location["solar_irradiance"]["clear_sky"]["ghi"]
    aqi_values = parameters.get("air_quality:idx") or [None]
    aqi = aqi_values[0]

    return RawReading(
        site_id=site.id,
        ts=ts,
        heat_index=celsius_to_fahrenheit(heat_index_c),
        aqi=aqi,
        humidity=humidity,
        solar_irradiance=ghi,
        is_forecast=False,
        heatmap_geojson=heatmap_geojson,
    )


def sense_forecast(
    client: FortyGuardClient, site: Site, *, now: datetime | None = None
) -> ForecastSignal:
    """Sense the +12h forecast: heatmap only, no env_params call, since the
    handbook doesn't guarantee forecast support there.
    """
    ts = (now or _now_naive_utc()) + timedelta(hours=12)

    heatmap_response = client.create_heatmap(
        site.polygon_geojson,
        start_date=ts.date().isoformat(),
        start_time=ts.strftime("%H:%M"),
        filter_type=1,
    )
    max_temp_c = heatmap_response["result"]["stats_data"]["temperature_stats"]["maximum"]
    heatmap_geojson = heatmap_response["result"].get("map_data")

    return ForecastSignal(
        site_id=site.id, ts=ts, max_temp_c=max_temp_c, heatmap_geojson=heatmap_geojson
    )
