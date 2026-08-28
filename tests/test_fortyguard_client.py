"""Tests for agent.fortyguard_client.FortyGuardClient.

All HTTP calls to api.fortyguard.com are mocked via respx -- no real
network calls happen in this file.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from agent.fortyguard_client import (
    ForecastWindowError,
    FortyGuardClient,
    TaskFailedError,
    ValidationError,
)

BASE_URL = "https://api.fortyguard.com"

# A small (~1 km^2), well-closed polygon over lower Manhattan -- comfortably
# under the 50 mi^2 heatmap cap.
SMALL_POLYGON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-74.0170, 40.7050],
                    [-74.0030, 40.7050],
                    [-74.0030, 40.7180],
                    [-74.0170, 40.7180],
                    [-74.0170, 40.7050],
                ]],
            },
        }
    ],
}

# A ~588 km^2 (~227 mi^2) box -- well over the 50 mi^2 Premium heatmap cap
# (confirmed via https://docs-api.fortyguard.com/docs/create-heatmap).
OVERSIZED_POLYGON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-122.20, 37.00],
                    [-121.90, 37.00],
                    [-121.90, 37.20],
                    [-122.20, 37.20],
                    [-122.20, 37.00],
                ]],
            },
        }
    ],
}

PHOENIX = (33.4484, -112.0740)
TORONTO = (43.6532, -79.3832)


@pytest.fixture
def client():
    return FortyGuardClient(api_key="test-key", base_url=BASE_URL)


def _envelope(data: dict) -> dict:
    return {"error": False, "status_code": 200, "message": "OK", "data": data}


def test_auth_check_returns_valid_key_status(respx_mock, client):
    respx_mock.get(f"{BASE_URL}/v1/system/fetch-api-key-usage").mock(
        return_value=httpx.Response(
            200,
            json=_envelope(
                {
                    "plan": "Hackathon",
                    "remaining_credits": 2_000_000,
                    "total_credits": 2_000_000,
                }
            ),
        )
    )

    usage = client.fetch_api_key_usage()

    assert usage["remaining_credits"] == 2_000_000


def test_create_heatmap_submits_and_polls_to_completion(respx_mock, client, mocker):
    mocker.patch("agent.fortyguard_client.time.sleep")
    activity_id = "f52d2453-6a59-4b31-afa3-8fe3bb1ac5df"
    respx_mock.post(f"{BASE_URL}/v1/heatmap").mock(
        return_value=httpx.Response(200, json=_envelope({"activity_id": activity_id}))
    )
    status_route = respx_mock.get(f"{BASE_URL}/v1/status/{activity_id}")
    status_route.side_effect = [
        httpx.Response(
            200,
            json=_envelope({"activity_id": activity_id, "status": "Processing"}),
        ),
        httpx.Response(
            200,
            json=_envelope(
                {
                    "activity_id": activity_id,
                    "status": "Completed",
                    "result": {"map_data": {}, "stats_data": {"Minimum": 20.0}},
                }
            ),
        ),
    ]

    outcome = client.create_heatmap(SMALL_POLYGON, start_date="2024-07-15", filter_type=3)

    assert outcome["activity_id"] == activity_id
    assert outcome["result"]["stats_data"]["Minimum"] == 20.0


def test_create_heatmap_rejects_aoi_over_50mi2(respx_mock, client):
    with pytest.raises(ValidationError):
        client.create_heatmap(OVERSIZED_POLYGON, start_date="2024-07-15", filter_type=3)

    assert len(respx_mock.calls) == 0


def test_create_heatmap_rejects_unclosed_polygon_ring(respx_mock, client):
    unclosed = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-74.0170, 40.7050],
                        [-74.0030, 40.7050],
                        [-74.0030, 40.7180],
                        [-74.0170, 40.7180],
                        # last point deliberately doesn't match the first
                        [-74.0160, 40.7051],
                    ]],
                },
            }
        ],
    }

    with pytest.raises(ValidationError):
        client.create_heatmap(unclosed, start_date="2024-07-15", filter_type=3)

    assert len(respx_mock.calls) == 0


def test_create_heatmap_forecast_window_capped_at_12h(respx_mock, client):
    too_far = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=13)

    with pytest.raises(ForecastWindowError):
        client.create_heatmap(
            SMALL_POLYGON,
            start_date=too_far.date().isoformat(),
            start_time=too_far.strftime("%H:%M"),
            filter_type=1,
        )

    assert len(respx_mock.calls) == 0


def test_env_params_uses_matching_timestamp(respx_mock, client):
    with pytest.raises(TypeError):
        client.environmental_parameters(*PHOENIX, temperature=42.0)  # no reference_ts

    reference_ts = datetime(2024, 7, 15, 14, 0)
    respx_mock.post(f"{BASE_URL}/v1/env_params").mock(
        return_value=httpx.Response(200, json=_envelope({"activity_id": "env-1"}))
    )
    respx_mock.get(f"{BASE_URL}/v1/status/env-1").mock(
        return_value=httpx.Response(
            200,
            json=_envelope({"activity_id": "env-1", "status": "Completed", "result": {}}),
        )
    )

    client.environmental_parameters(*PHOENIX, temperature=42.0, reference_ts=reference_ts)

    submitted_body = json.loads(respx_mock.calls[0].request.content)
    assert submitted_body["date_time"]["start_date"] == "2024-07-15"
    assert submitted_body["date_time"]["start_time"] == "14:00"


def test_status_poll_backoff_sequence(respx_mock, client, mocker):
    sleep_mock = mocker.patch("agent.fortyguard_client.time.sleep")
    activity_id = "poll-seq"
    respx_mock.post(f"{BASE_URL}/v1/heatmap").mock(
        return_value=httpx.Response(200, json=_envelope({"activity_id": activity_id}))
    )
    status_route = respx_mock.get(f"{BASE_URL}/v1/status/{activity_id}")
    status_route.side_effect = [
        httpx.Response(
            200, json=_envelope({"activity_id": activity_id, "status": "Processing"})
        ),
        httpx.Response(
            200, json=_envelope({"activity_id": activity_id, "status": "Processing"})
        ),
        httpx.Response(
            200, json=_envelope({"activity_id": activity_id, "status": "Processing"})
        ),
        httpx.Response(
            200,
            json=_envelope(
                {"activity_id": activity_id, "status": "Completed", "result": {}}
            ),
        ),
    ]

    client.create_heatmap(SMALL_POLYGON, start_date="2024-07-15", filter_type=3)

    assert [call.args[0] for call in sleep_mock.call_args_list] == [3.0, 6.0, 12.0]


def test_failed_task_reports_zero_cost(respx_mock, client, mocker):
    mocker.patch("agent.fortyguard_client.time.sleep")
    activity_id = "will-fail"
    respx_mock.post(f"{BASE_URL}/v1/heatmap").mock(
        return_value=httpx.Response(200, json=_envelope({"activity_id": activity_id}))
    )
    respx_mock.get(f"{BASE_URL}/v1/status/{activity_id}").mock(
        return_value=httpx.Response(
            200,
            json=_envelope(
                {"activity_id": activity_id, "status": "Failed", "message": "bad AOI"}
            ),
        )
    )

    assert client.credits_used_estimate == 0
    with pytest.raises(TaskFailedError):
        client.create_heatmap(SMALL_POLYGON, start_date="2024-07-15", filter_type=3)
    assert client.credits_used_estimate == 0


def test_us_only_coordinate_guard(respx_mock, client):
    with pytest.raises(ValidationError):
        client.environmental_parameters(
            *TORONTO, temperature=25.0, reference_ts=datetime(2024, 7, 15, 14, 0)
        )

    assert len(respx_mock.calls) == 0
