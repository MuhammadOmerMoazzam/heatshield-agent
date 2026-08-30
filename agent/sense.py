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
of real scheduled cycles across many attempts): an hourly
``create_heatmap`` call (``filter_type=1``) -- for the literal current
hour, or up to a couple of hours back -- reliably succeeds (200 OK) but
returns ``stats_data`` with no ``temperature_stats``. Failed **100% of
the time observed**, on both seeded sites, across many independent
cycles. FortyGuard's pipeline evidently doesn't ingest/aggregate hourly
data on a short enough lag for this to be worth querying at all right
now. A day-level ``create_heatmap`` query (``filter_type=3``, "today"),
by contrast, has reliably had real data every time this was checked
live. Given that -- and given every hourly attempt is a separate,
credit-metered call, and heatmap generation was observed consuming the
large majority of this project's total API credit usage almost entirely
on these failed hourly attempts -- the temperature layer goes straight
to the day-level query rather than trying hourly first. (If FortyGuard's
pipeline ends up catching up on hourly ingestion later, reintroducing an
hourly-first attempt here is a small, self-contained change -- see git
history for the previous hourly-then-day-level version.)

``environmental_parameters`` shows the same "succeeds but empty"
behavior independently, but less reliably absent (it has succeeded on
a first or second attempt in real observed cases) and has no day-level
equivalent to fall back to -- so unlike the temperature layer, it still
retries over ``LIVE_LOOKBACK_HOURS`` before giving up, and degrades to
``None`` (not a raised error) once every attempt is empty: a real,
current temperature reading is worth keeping even without the small
humidity/solar penalties ``compute_raw_stress`` (agent/score.py)
normally adds. When this happens, ``RawReading.heat_index`` itself
falls back to the heatmap's own ambient temperature (no humidity
adjustment available).

The returned ``RawReading.ts`` is "now" -- the day-level query has no
single hour to honestly attribute the reading to.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from pydantic import BaseModel

from agent._shared import now_naive_utc as _now_naive_utc
from agent.fortyguard_client import FortyGuardClient, TaskTimeoutError
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


# How many hours back _fetch_live_environmental_parameters retries
# before giving up (0 = the exact matching timestamp, tried first). Only
# used for environmental_parameters now -- the temperature layer
# (create_heatmap) goes straight to a day-level query instead of an
# hourly search (see module docstring). Each entry is a separate,
# credit-metered call, so this stays small and bounded -- not an
# unbounded/open-ended search -- rather than a blanket "keep retrying
# forever" policy.
LIVE_LOOKBACK_HOURS: tuple[int, ...] = (0, 1, 2)

# Live-verified (Phase 9): a real environmental_parameters call hit
# TaskTimeoutError at the client's 60s default -- the same class of
# slow-async-task issue Phase 7 fixed for satellite/streetview/
# heat_intelligence, now applied to every call this module makes too.
_SENSE_TIMEOUT = 300.0


def celsius_to_fahrenheit(celsius: float) -> float:
    return celsius * 9 / 5 + 32


def _call_with_one_retry_on_timeout(fn):
    """A single retry before giving up on a TaskTimeoutError. Live-
    observed repeatedly this hackathon: Phoenix and Houston each
    independently hit a same-cycle TaskTimeoutError on a FortyGuard call,
    with the identical call often succeeding within seconds on a later
    attempt -- consistent with transient server-side load, not a
    systematic failure. 300s is already a generous per-attempt timeout,
    and this runs in the background scheduler where extra latency costs
    nothing user-facing, so trading a bounded amount of it for materially
    better odds of a reading landing each cycle is worth it. Still
    propagates if the retry also times out -- this is one extra chance,
    not an unbounded loop.
    """
    try:
        return fn()
    except TaskTimeoutError:
        return fn()


def _fetch_live_temperature(
    client: FortyGuardClient, site: Site, base_now: datetime
) -> tuple[float, datetime]:
    """Layer 1: today's day-level aggregate directly -- see module
    docstring for why hourly is skipped entirely here. Returns
    (mean_temp_c, ts). Raises SenseDataUnavailableError if the day-level
    query has no data either.

    The response's own map_data (a day-level tile FeatureCollection) is
    deliberately never read here -- live-verified over 1 million
    characters for a single reading, and persisting it was directly
    implicated in recurring "database is locked" failures (a large
    single-row insert holds the write transaction open longer, widening
    the collision window with the background scheduler's other writes)
    plus fast, unbounded SQLite file growth.
    """
    day_response = _call_with_one_retry_on_timeout(
        lambda: client.create_heatmap(
            site.polygon_geojson,
            start_date=base_now.date().isoformat(),
            filter_type=3,
            timeout=_SENSE_TIMEOUT,
        )
    )
    day_temperature_stats = day_response.get("result", {}).get("stats_data", {}).get(
        "temperature_stats"
    )
    if day_temperature_stats is not None and day_temperature_stats.get("mean") is not None:
        return day_temperature_stats["mean"], base_now

    raise SenseDataUnavailableError(
        f"No heatmap temperature data available for site_id={site.id}'s day-level aggregate "
        f"({base_now.date().isoformat()}) -- FortyGuard's pipeline may not have ingested data "
        "for this site recently."
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
        try:
            env_response = _call_with_one_retry_on_timeout(
                lambda: client.environmental_parameters(
                    site.lat,
                    site.lon,
                    temperature=mean_temp_c,
                    reference_ts=candidate_ts,
                    timeout=_SENSE_TIMEOUT,
                )
            )
        except TaskTimeoutError:
            # Both the original attempt and its retry timed out at this
            # specific hour -- treat it the same as the "succeeded but
            # empty" case below and move on to the next fallback hour,
            # rather than aborting the whole cycle over one slow instant.
            continue
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

    mean_temp_c, ts = _fetch_live_temperature(client, site, base_now)
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
        # heatmap_geojson deliberately never populated -- see
        # _fetch_live_temperature's docstring.
        heatmap_geojson=None,
    )


def sense_forecast(
    client: FortyGuardClient, site: Site, *, now: datetime | None = None
) -> ForecastSignal:
    """Sense the +12h forecast: heatmap only, no env_params call, since the
    handbook doesn't guarantee forecast support there.

    Goes straight to that target day's day-level peak, the same as
    sense_live's temperature layer and for the same reason (see module
    docstring) -- an hourly query for a not-yet-elapsed forecast hour has
    shown the identical "succeeds but empty" behavior in live testing,
    if anything even more reliably than sense_live's, since the hour
    genuinely hasn't happened yet.
    """
    ts = (now or _now_naive_utc()) + timedelta(hours=12)

    day_response = _call_with_one_retry_on_timeout(
        lambda: client.create_heatmap(
            site.polygon_geojson,
            start_date=ts.date().isoformat(),
            filter_type=3,
            # start_time is otherwise unused for a day-level query (the
            # API ignores it per the quickstart docs), but keeping it
            # here is what makes the client's own client-side +12h
            # forecast-window guard (_validate_forecast_window, which
            # only checks start_time when one is given) still fire
            # *before* any network call -- without it, an out-of-window
            # forecast request would go straight to the real API instead
            # of being rejected locally.
            start_time=ts.strftime("%H:%M"),
            timeout=_SENSE_TIMEOUT,
        )
    )
    day_temperature_stats = day_response.get("result", {}).get("stats_data", {}).get(
        "temperature_stats"
    )
    if day_temperature_stats is not None and day_temperature_stats.get("maximum") is not None:
        return ForecastSignal(
            site_id=site.id,
            ts=ts,
            max_temp_c=day_temperature_stats["maximum"],
            # heatmap_geojson deliberately never populated -- see
            # _fetch_live_temperature's docstring.
            heatmap_geojson=None,
        )

    raise SenseDataUnavailableError(
        f"No heatmap temperature data available for site_id={site.id}'s +12h forecast day-level "
        f"aggregate ({ts.date().isoformat()}) -- FortyGuard's pipeline may not have ingested "
        "forecast data for this site recently."
    )
