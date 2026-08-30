"""Tests for agent.sense.sense_live / sense_forecast.

FortyGuardClient is stubbed for the two mocked-call tests; the +12h
window test uses a real client so the assertion exercises the actual
FortyGuardError subclass raised by agent.fortyguard_client, not a stand-in.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from agent.fortyguard_client import ForecastWindowError, FortyGuardClient, TaskTimeoutError
from agent.models import Site
from agent.sense import LIVE_LOOKBACK_HOURS, SenseDataUnavailableError, sense_forecast, sense_live

SMALL_POLYGON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-121.90, 37.33],
                    [-121.89, 37.33],
                    [-121.89, 37.34],
                    [-121.90, 37.34],
                    [-121.90, 37.33],
                ]],
            },
        }
    ],
}


def _make_site(site_id: int = 1) -> Site:
    site = Site(
        name="Test Site",
        lat=37.335,
        lon=-121.895,
        polygon_geojson=SMALL_POLYGON,
    )
    site.id = site_id
    return site


def _heatmap_result(
    mean_temp_c: float = 35.0, max_temp_c: float = 40.0, map_data: dict | None = None
) -> dict:
    result = {
        "stats_data": {"temperature_stats": {"mean": mean_temp_c, "maximum": max_temp_c}}
    }
    if map_data is not None:
        result["map_data"] = map_data
    return {"activity_id": "hm-1", "result": result}


def _env_params_result(
    heat_index_c: float = 38.0,
    humidity: float = 55.0,
    ghi: float = 650.0,
    aqi: float = 42.0,
) -> dict:
    return {
        "activity_id": "env-1",
        "result": {
            "locations": [
                {
                    "parameters": {
                        "heat_index_celsius": [heat_index_c],
                        "relative_humidity_percent": [humidity],
                        "air_quality:idx": [aqi],
                    },
                    "solar_irradiance": {"clear_sky": {"ghi": ghi}},
                }
            ]
        },
    }


def _env_params_result_empty(ghi: float = 0.0) -> dict:
    """Matches live-observed behavior: env_params returns 200 OK with
    empty parameter arrays (not a 4xx/5xx) when it has no data yet for
    the requested timestamp -- the same "succeeds but empty" shape as
    create_heatmap's own current-hour data lag.
    """
    return {
        "activity_id": "env-empty",
        "result": {
            "locations": [
                {
                    "parameters": {
                        "heat_index_celsius": [],
                        "relative_humidity_percent": [],
                        "air_quality:idx": [],
                    },
                    "solar_irradiance": {"clear_sky": {"ghi": ghi}},
                }
            ]
        },
    }


def test_sense_live_calls_both_endpoints_with_matching_timestamps():
    """The temperature layer goes straight to a day-level query (see
    module docstring: hourly create_heatmap queries failed 100% of the
    time in live testing, at real credit cost, while day-level has been
    reliably available) -- confirms it's a single filter_type=3 call
    with no start_time, and that env_params is called against "now"
    (the only timestamp a day-level reading can honestly be attributed
    to) with the day-level mean as its required temperature input.
    """
    client = MagicMock()
    client.create_heatmap.return_value = _heatmap_result(mean_temp_c=35.0)
    client.environmental_parameters.return_value = _env_params_result()
    site = _make_site()
    fixed_now = datetime(2024, 7, 15, 14, 0)

    reading = sense_live(client, site, now=fixed_now)

    assert client.create_heatmap.call_count == 1
    heatmap_call = client.create_heatmap.call_args
    assert heatmap_call.kwargs["start_date"] == "2024-07-15"
    assert heatmap_call.kwargs["filter_type"] == 3
    assert "start_time" not in heatmap_call.kwargs

    env_call = client.environmental_parameters.call_args
    assert env_call.kwargs["reference_ts"] == fixed_now
    # The heatmap's own measured temperature feeds env_params' required
    # `temperature` input -- not an independent/guessed value.
    assert env_call.kwargs["temperature"] == 35.0

    assert reading.site_id == 1
    assert reading.is_forecast is False
    assert reading.ts == fixed_now
    assert reading.humidity == 55.0
    assert reading.solar_irradiance == 650.0
    assert reading.aqi == 42.0
    # heat_index_celsius=38.0 -> Fahrenheit
    assert reading.heat_index == pytest.approx(38.0 * 9 / 5 + 32)


def test_sense_live_raises_clear_error_when_day_level_heatmap_has_no_data():
    """The loud-failure case: the day-level query itself comes back with
    no usable data. This must be an unambiguous, diagnosable error (not
    the raw KeyError this used to surface as) -- caught and logged by
    agent.loop's existing per-site resilience layer same as before, but
    now with a message an operator can actually act on.
    """
    client = MagicMock()
    client.create_heatmap.return_value = {"activity_id": "hm-empty", "result": {"stats_data": {}}}
    site = _make_site()
    fixed_now = datetime(2024, 7, 15, 14, 0)

    with pytest.raises(SenseDataUnavailableError, match="No heatmap temperature data available"):
        sense_live(client, site, now=fixed_now)

    assert client.create_heatmap.call_count == 1


def test_sense_live_falls_back_to_ambient_temperature_when_env_params_has_no_data():
    """Production evidence (Phase 9, live-verified): environmental_parameters
    can independently lag the same way create_heatmap's hourly data does
    -- confirmed live returning empty heat_index_celsius/
    relative_humidity_percent arrays at "now". Rather than discard an
    otherwise-real temperature reading, sense_live falls back to the
    heatmap's own ambient temperature as the heat index, and reports
    humidity/solar_irradiance as genuinely unavailable (None) instead of
    guessing -- compute_raw_stress is designed to handle that (see
    agent/score.py).
    """
    client = MagicMock()
    client.create_heatmap.return_value = _heatmap_result(mean_temp_c=30.0)
    client.environmental_parameters.return_value = _env_params_result_empty()
    site = _make_site()
    fixed_now = datetime(2024, 7, 15, 14, 0)

    reading = sense_live(client, site, now=fixed_now)

    assert client.environmental_parameters.call_count == len(LIVE_LOOKBACK_HOURS)
    assert reading.heat_index == pytest.approx(30.0 * 9 / 5 + 32)
    assert reading.humidity is None
    assert reading.solar_irradiance is None
    assert reading.aqi is None


def test_sense_live_never_stores_heatmap_geojson():
    """Live-verified (Phase 9): a day-level heatmap's tile FeatureCollection
    can be enormous -- over 1 million characters for a single reading in
    real production data -- and persisting it was directly implicated in
    recurring 'database is locked' failures (a large single-row insert
    holds the write transaction open longer, widening the collision
    window) plus fast, unbounded SQLite file growth. Deliberately no
    longer stored, even when the API response carries one, regardless of
    reading.ts.
    """
    map_data = {"type": "FeatureCollection", "features": [{"properties": {"temperature": 35.0}}]}
    client = MagicMock()
    client.create_heatmap.return_value = _heatmap_result(map_data=map_data)
    client.environmental_parameters.return_value = _env_params_result()
    site = _make_site()

    reading = sense_live(client, site)

    assert reading.heatmap_geojson is None


def test_sense_forecast_calls_only_heatmap():
    """Goes straight to the target day's day-level query (see module
    docstring: an hourly query for a not-yet-elapsed forecast hour has
    shown the same "succeeds but empty" lag sense_live's hourly queries
    did). start_time is still passed alongside filter_type=3 -- the API
    itself ignores it for a day-level query, but it's what makes the
    client's own client-side +12h forecast-window guard still fire
    before any network call.
    """
    client = MagicMock()
    client.create_heatmap.return_value = _heatmap_result(max_temp_c=41.5)
    site = _make_site()
    fixed_now = datetime(2024, 7, 15, 14, 0)

    signal = sense_forecast(client, site, now=fixed_now)

    assert client.create_heatmap.call_count == 1
    assert client.environmental_parameters.call_count == 0
    heatmap_call = client.create_heatmap.call_args
    assert heatmap_call.kwargs["start_date"] == "2024-07-16"  # +12h from 14:00 lands the 16th
    assert heatmap_call.kwargs["filter_type"] == 3
    assert heatmap_call.kwargs["start_time"] == "02:00"
    assert signal.site_id == 1
    assert signal.max_temp_c == 41.5


def test_sense_forecast_never_stores_heatmap_geojson():
    """Same reasoning as sense_live -- see
    test_sense_live_never_stores_heatmap_geojson."""
    map_data = {"type": "FeatureCollection", "features": [{"properties": {"temperature": 41.5}}]}
    client = MagicMock()
    client.create_heatmap.return_value = _heatmap_result(max_temp_c=41.5, map_data=map_data)
    site = _make_site()

    signal = sense_forecast(client, site)

    assert signal.heatmap_geojson is None


def test_sense_forecast_raises_clear_error_when_day_level_heatmap_has_no_data():
    client = MagicMock()
    client.create_heatmap.return_value = {"activity_id": "hm-empty", "result": {"stats_data": {}}}
    site = _make_site()
    fixed_now = datetime(2024, 7, 15, 14, 0)

    with pytest.raises(SenseDataUnavailableError, match="No heatmap temperature data available"):
        sense_forecast(client, site, now=fixed_now)

    assert client.create_heatmap.call_count == 1


def test_sense_live_and_sense_forecast_use_an_extended_timeout():
    """Live-verified (Phase 9): a real environmental_parameters call hit
    TaskTimeoutError at the client's 60s default -- the same class of
    slow-async-task issue Phase 7 already fixed for satellite/streetview/
    heat_intelligence, just not applied to sense.py's own calls yet.
    """
    client = MagicMock()
    client.create_heatmap.return_value = _heatmap_result(mean_temp_c=35.0, max_temp_c=41.5)
    client.environmental_parameters.return_value = _env_params_result()
    site = _make_site()

    sense_live(client, site)
    sense_forecast(client, site)

    for call in client.create_heatmap.call_args_list:
        assert call.kwargs.get("timeout") is not None
        assert call.kwargs["timeout"] >= 300.0
    env_call = client.environmental_parameters.call_args
    assert env_call.kwargs.get("timeout") is not None
    assert env_call.kwargs["timeout"] >= 300.0


def test_sense_live_temperature_layer_retries_once_on_timeout_then_succeeds():
    """Live-observed pattern this hackathon: Phoenix and Houston each
    independently hit a same-cycle TaskTimeoutError on a FortyGuard call,
    with the identical call often succeeding within seconds on a later
    attempt -- consistent with transient server-side load, not a
    systematic failure worth giving up on immediately. 300s is already a
    generous per-attempt timeout, and this runs in the background
    scheduler where extra latency costs nothing user-facing, so one retry
    before giving up is a cheap way to materially improve the odds of a
    live reading landing each cycle.
    """
    client = MagicMock()
    client.create_heatmap.side_effect = [
        TaskTimeoutError("Activity abc still 'processing' after timeout"),
        _heatmap_result(mean_temp_c=35.0),
    ]
    client.environmental_parameters.return_value = _env_params_result()
    site = _make_site()

    reading = sense_live(client, site)

    assert client.create_heatmap.call_count == 2
    assert reading.heat_index == pytest.approx(38.0 * 9 / 5 + 32)


def test_sense_live_temperature_layer_still_raises_when_retry_also_times_out():
    client = MagicMock()
    client.create_heatmap.side_effect = [
        TaskTimeoutError("Activity abc still 'processing' after timeout"),
        TaskTimeoutError("Activity def still 'processing' after timeout"),
    ]
    site = _make_site()

    with pytest.raises(TaskTimeoutError):
        sense_live(client, site)

    assert client.create_heatmap.call_count == 2


def test_sense_live_env_params_retries_once_at_the_same_lookback_hour_then_succeeds():
    client = MagicMock()
    client.create_heatmap.return_value = _heatmap_result(mean_temp_c=35.0)
    client.environmental_parameters.side_effect = [
        TaskTimeoutError("Activity abc still 'processing' after timeout"),
        _env_params_result(),
    ]
    site = _make_site()

    reading = sense_live(client, site)

    assert client.environmental_parameters.call_count == 2
    # Both attempts targeted the same (first) lookback hour -- a retry
    # isn't a new fallback hour, it's a second try at the same one.
    first_ts = client.environmental_parameters.call_args_list[0].kwargs["reference_ts"]
    second_ts = client.environmental_parameters.call_args_list[1].kwargs["reference_ts"]
    assert first_ts == second_ts
    assert reading.humidity == 55.0


def test_sense_live_env_params_moves_to_next_lookback_hour_after_timeout_and_retry_both_fail():
    """A timeout is treated the same as the existing "succeeded but
    empty" case once both the original attempt and its retry are
    exhausted -- move on to the next fallback hour rather than aborting
    the whole cycle, mirroring how this loop already degrades gracefully
    for empty data.
    """
    client = MagicMock()
    client.create_heatmap.return_value = _heatmap_result(mean_temp_c=35.0)
    client.environmental_parameters.side_effect = [
        TaskTimeoutError("Activity abc still 'processing' after timeout"),
        TaskTimeoutError("Activity def still 'processing' after timeout"),
        _env_params_result(),
    ]
    site = _make_site()

    reading = sense_live(client, site)

    assert client.environmental_parameters.call_count == 3
    first_ts = client.environmental_parameters.call_args_list[0].kwargs["reference_ts"]
    third_ts = client.environmental_parameters.call_args_list[2].kwargs["reference_ts"]
    assert third_ts == first_ts - timedelta(hours=1)
    assert reading.humidity == 55.0


def test_sense_forecast_retries_once_on_timeout_then_succeeds():
    client = MagicMock()
    client.create_heatmap.side_effect = [
        TaskTimeoutError("Activity abc still 'processing' after timeout"),
        _heatmap_result(max_temp_c=41.5),
    ]
    site = _make_site()

    signal = sense_forecast(client, site)

    assert client.create_heatmap.call_count == 2
    assert signal.max_temp_c == 41.5


def test_sense_forecast_beyond_12h_raises_forecast_window_error():
    client = FortyGuardClient(api_key="test-key", base_url="https://api.fortyguard.com")
    site = _make_site()
    # now + 12h will land well beyond the client's own real "now" + 12h.
    pushed_now = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=2)

    with pytest.raises(ForecastWindowError):
        sense_forecast(client, site, now=pushed_now)
