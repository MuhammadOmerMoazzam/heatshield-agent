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

Production evidence (real Streamlit Cloud logs, Phase 9, several hours
of real scheduled cycles): a ``create_heatmap`` call for the literal
current hour reliably succeeds (200 OK) but returns ``stats_data`` with
no ``temperature_stats`` -- FortyGuard's pipeline evidently hasn't
ingested/aggregated that hour's data yet, and this can lag more than a
couple of hours behind real-time. ``environmental_parameters`` shows the
exact same "succeeds but empty" behavior, independently, live-verified.
A day-level ``create_heatmap`` query (``filter_type=3``, "today"), by
contrast, has reliably had real data every time this was checked live.

Given that, ``sense_live``'s data-gathering is layered, each layer
falling back only once the one before it comes up genuinely empty:

1. **Temperature** (``create_heatmap``): try a few hours back at hourly
   precision (``LIVE_LOOKBACK_HOURS`` -- freshest, when available), then
   fall back to today's day-level aggregate. Only raises
   ``SenseDataUnavailableError`` if *that* also has no data -- live
   evidence suggests this should be rare.
2. **Humidity/solar/AQI** (``environmental_parameters``): tries the same
   hourly window, against the temperature layer 1 found (Rule 2 --
   "matching timestamp"). Unlike layer 1, there is no day-level
   fallback for this endpoint, so if every attempt comes up empty, this
   layer degrades to ``None`` rather than raising -- a real, current
   temperature reading is worth keeping even without the humidity/solar
   penalties ``compute_raw_stress`` (agent/score.py) normally adds, and
   is a better signal than refusing to produce a reading at all. When
   this happens, ``RawReading.heat_index`` itself falls back to the
   heatmap's own ambient temperature (no humidity adjustment available).

The returned ``RawReading.ts`` honestly reflects which hour the
temperature data is actually from (or "now", for a day-level reading --
there's no single hour to attribute it to).
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
    # None when environmental_parameters had no data at any attempted
    # timestamp (live-verified: this happens) -- a real but incomplete
    # reading, not an error condition.
    humidity: float | None  # percent
    solar_irradiance: float | None  # W/m^2 (clear-sky GHI)
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

# Live-verified (Phase 9): a real environmental_parameters call hit
# TaskTimeoutError at the client's 60s default -- the same class of
# slow-async-task issue Phase 7 fixed for satellite/streetview/
# heat_intelligence, now applied to every call this module makes too.
_SENSE_TIMEOUT = 300.0


def celsius_to_fahrenheit(celsius: float) -> float:
    return celsius * 9 / 5 + 32


def _fetch_live_temperature(
    client: FortyGuardClient, site: Site, base_now: datetime
) -> tuple[float, datetime, dict | None]:
    """Layer 1: a few hours back at hourly precision, then today's
    day-level aggregate. Returns (mean_temp_c, ts, heatmap_geojson).
    Raises SenseDataUnavailableError only if the day-level fallback also
    has no data -- see module docstring.
    """
    attempted: list[str] = []

    for hours_back in LIVE_LOOKBACK_HOURS:
        candidate_ts = base_now - timedelta(hours=hours_back)
        attempted.append(candidate_ts.strftime("%Y-%m-%d %H:%M"))

        heatmap_response = client.create_heatmap(
            site.polygon_geojson,
            start_date=candidate_ts.date().isoformat(),
            start_time=candidate_ts.strftime("%H:%M"),
            filter_type=1,
            timeout=_SENSE_TIMEOUT,
        )
        temperature_stats = heatmap_response.get("result", {}).get("stats_data", {}).get(
            "temperature_stats"
        )
        if temperature_stats is not None and temperature_stats.get("mean") is not None:
            return (
                temperature_stats["mean"],
                candidate_ts,
                heatmap_response["result"].get("map_data"),
            )

    day_response = client.create_heatmap(
        site.polygon_geojson,
        start_date=base_now.date().isoformat(),
        filter_type=3,
        timeout=_SENSE_TIMEOUT,
    )
    day_temperature_stats = day_response.get("result", {}).get("stats_data", {}).get(
        "temperature_stats"
    )
    if day_temperature_stats is not None and day_temperature_stats.get("mean") is not None:
        return day_temperature_stats["mean"], base_now, day_response["result"].get("map_data")

    attempted_str = ", ".join(attempted)
    raise SenseDataUnavailableError(
        f"No heatmap temperature data available for site_id={site.id} at any of the last "
        f"{len(LIVE_LOOKBACK_HOURS)} hourly attempts ({attempted_str}) or today's day-level "
        "aggregate -- FortyGuard's pipeline may not have ingested data for this site recently."
    )


def _fetch_live_environmental_parameters(
    client: FortyGuardClient, site: Site, mean_temp_c: float, ts: datetime
) -> tuple[float | None, float | None, float | None, float | None]:
    """Layer 2: humidity/solar/AQI. Tries ``ts`` first -- the exact
    timestamp layer 1's temperature actually came from (Rule 2:
    "matching timestamp"), not the originally-requested "now" -- then
    widens further back over the same hourly window if that specific
    hour has no data yet. No day-level equivalent exists for this
    endpoint, so this returns all-None (rather than raising) once every
    attempt is empty -- see module docstring for why that's the right
    degradation here.

    Returns (heat_index_c, humidity_pct, ghi, aqi), any of which may be
    None.
    """
    for hours_back in LIVE_LOOKBACK_HOURS:
        candidate_ts = ts - timedelta(hours=hours_back)
        env_response = client.environmental_parameters(
            site.lat,
            site.lon,
            temperature=mean_temp_c,
            reference_ts=candidate_ts,
            timeout=_SENSE_TIMEOUT,
        )
        location = env_response["result"]["locations"][0]
        parameters = location["parameters"]
        heat_index_values = parameters.get("heat_index_celsius") or []
        if heat_index_values and heat_index_values[0] is not None:
            humidity_values = parameters.get("relative_humidity_percent") or [None]
            aqi_values = parameters.get("air_quality:idx") or [None]
            ghi = location.get("solar_irradiance", {}).get("clear_sky", {}).get("ghi")
            return heat_index_values[0], humidity_values[0], ghi, aqi_values[0]

    return None, None, None, None


def sense_live(client: FortyGuardClient, site: Site, *, now: datetime | None = None) -> RawReading:
    """Sense the current moment via the two-layer fallback chain
    described in the module docstring.
    """
    base_now = now or _now_naive_utc()

    mean_temp_c, ts, heatmap_geojson = _fetch_live_temperature(client, site, base_now)
    heat_index_c, humidity, ghi, aqi = _fetch_live_environmental_parameters(
        client, site, mean_temp_c, ts
    )

    if heat_index_c is None:
        # No humidity/solar signal at any attempted hour -- the ambient
        # temperature this cycle's own heatmap call already measured is
        # still a real, current reading, just without the small
        # humidity/solar stress adjustments compute_raw_stress normally
        # adds.
        heat_index_c = mean_temp_c

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

    A forecast necessarily queries a not-yet-elapsed hour, so the same
    "succeeds but empty" lag sense_live's hourly queries show (see module
    docstring) can hit this specific forecasted hour too -- if anything,
    even more likely, since it hasn't happened yet. Falls back to that
    target day's day-level peak rather than raising the raw KeyError this
    used to surface as.
    """
    ts = (now or _now_naive_utc()) + timedelta(hours=12)

    heatmap_response = client.create_heatmap(
        site.polygon_geojson,
        start_date=ts.date().isoformat(),
        start_time=ts.strftime("%H:%M"),
        filter_type=1,
        timeout=_SENSE_TIMEOUT,
    )
    temperature_stats = heatmap_response.get("result", {}).get("stats_data", {}).get(
        "temperature_stats"
    )
    if temperature_stats is not None and temperature_stats.get("maximum") is not None:
        return ForecastSignal(
            site_id=site.id,
            ts=ts,
            max_temp_c=temperature_stats["maximum"],
            heatmap_geojson=heatmap_response["result"].get("map_data"),
        )

    day_response = client.create_heatmap(
        site.polygon_geojson,
        start_date=ts.date().isoformat(),
        filter_type=3,
        timeout=_SENSE_TIMEOUT,
    )
    day_temperature_stats = day_response.get("result", {}).get("stats_data", {}).get(
        "temperature_stats"
    )
    if day_temperature_stats is not None and day_temperature_stats.get("maximum") is not None:
        return ForecastSignal(
            site_id=site.id,
            ts=ts,
            max_temp_c=day_temperature_stats["maximum"],
            heatmap_geojson=day_response["result"].get("map_data"),
        )

    raise SenseDataUnavailableError(
        f"No heatmap temperature data available for site_id={site.id}'s +12h forecast "
        f"({ts.strftime('%Y-%m-%d %H:%M')}) or that day's day-level aggregate -- FortyGuard's "
        "pipeline may not have ingested forecast data for this site recently."
    )
