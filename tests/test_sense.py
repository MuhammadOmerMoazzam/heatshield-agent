"""Tests for agent.sense.sense_live / sense_forecast.

FortyGuardClient is stubbed for the two mocked-call tests; the +12h
window test uses a real client so the assertion exercises the actual
FortyGuardError subclass raised by agent.fortyguard_client, not a stand-in.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from agent.fortyguard_client import ForecastWindowError, FortyGuardClient
from agent.models import Site
from agent.sense import sense_forecast, sense_live

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


def test_sense_live_calls_both_endpoints_with_matching_timestamps():
    client = MagicMock()
    client.create_heatmap.return_value = _heatmap_result(mean_temp_c=35.0)
    client.environmental_parameters.return_value = _env_params_result()
    site = _make_site()
    fixed_now = datetime(2024, 7, 15, 14, 0)

    reading = sense_live(client, site, now=fixed_now)

    heatmap_call = client.create_heatmap.call_args
    assert heatmap_call.kwargs["start_date"] == "2024-07-15"
    assert heatmap_call.kwargs["start_time"] == "14:00"

    env_call = client.environmental_parameters.call_args
    assert env_call.kwargs["reference_ts"] == fixed_now
    # The heatmap's own measured temperature feeds env_params' required
    # `temperature` input -- not an independent/guessed value.
    assert env_call.kwargs["temperature"] == 35.0

    assert reading.site_id == 1
    assert reading.is_forecast is False
    assert reading.humidity == 55.0
    assert reading.solar_irradiance == 650.0
    assert reading.aqi == 42.0
    # heat_index_celsius=38.0 -> Fahrenheit
    assert reading.heat_index == pytest.approx(38.0 * 9 / 5 + 32)


def test_sense_live_captures_heatmap_tile_geojson_for_dashboard():
    """Phase 8: the dashboard's map panel renders the tile FeatureCollection
    a heatmap call already returns -- it must not be discarded after the
    mean/max stats are pulled out of it.
    """
    map_data = {"type": "FeatureCollection", "features": [{"properties": {"temperature": 35.0}}]}
    client = MagicMock()
    client.create_heatmap.return_value = _heatmap_result(map_data=map_data)
    client.environmental_parameters.return_value = _env_params_result()
    site = _make_site()

    reading = sense_live(client, site)

    assert reading.heatmap_geojson == map_data


def test_sense_live_heatmap_geojson_defaults_to_none_when_absent():
    client = MagicMock()
    client.create_heatmap.return_value = _heatmap_result()  # no map_data
    client.environmental_parameters.return_value = _env_params_result()
    site = _make_site()

    reading = sense_live(client, site)

    assert reading.heatmap_geojson is None


def test_sense_forecast_calls_only_heatmap():
    client = MagicMock()
    client.create_heatmap.return_value = _heatmap_result(max_temp_c=41.5)
    site = _make_site()

    signal = sense_forecast(client, site)

    assert client.create_heatmap.call_count == 1
    assert client.environmental_parameters.call_count == 0
    assert signal.site_id == 1
    assert signal.max_temp_c == 41.5


def test_sense_forecast_captures_heatmap_tile_geojson_for_dashboard():
    map_data = {"type": "FeatureCollection", "features": [{"properties": {"temperature": 41.5}}]}
    client = MagicMock()
    client.create_heatmap.return_value = _heatmap_result(max_temp_c=41.5, map_data=map_data)
    site = _make_site()

    signal = sense_forecast(client, site)

    assert signal.heatmap_geojson == map_data


def test_sense_forecast_beyond_12h_raises_forecast_window_error():
    client = FortyGuardClient(api_key="test-key", base_url="https://api.fortyguard.com")
    site = _make_site()
    # now + 12h will land well beyond the client's own real "now" + 12h.
    pushed_now = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=2)

    with pytest.raises(ForecastWindowError):
        sense_forecast(client, site, now=pushed_now)
