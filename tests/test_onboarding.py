"""Tests for agent.onboarding.onboard_site.

The FortyGuardClient is stubbed (satellite_segmentation/street_view_
segmentation are mocked directly) -- this file tests onboarding logic
(call-once, idempotent, shade-derivation), not the HTTP wrapper itself
(covered by test_fortyguard_client.py).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent.models import Site
from agent.onboarding import onboard_site


def _make_site(**overrides) -> Site:
    defaults = dict(
        name="Test Site",
        lat=37.33,
        lon=-121.90,
        polygon_geojson={"type": "Polygon", "coordinates": [[[0, 0]]]},
    )
    defaults.update(overrides)
    return Site(**defaults)


def _satellite_result(segments: dict) -> dict:
    return {"activity_id": "sat-1", "result": {"segmentation": {"segments": segments}}}


def _streetview_result(segments: dict) -> dict:
    return {"activity_id": "sv-1", "result": {"front": {"segments": segments}}}


def _client_with(satellite_segments: dict, streetview_segments: dict) -> MagicMock:
    client = MagicMock()
    client.satellite_segmentation.return_value = _satellite_result(satellite_segments)
    client.street_view_segmentation.return_value = _streetview_result(streetview_segments)
    return client


def test_new_site_triggers_satellite_and_streetview_exactly_once():
    client = _client_with({"tree": 30.0}, {"tree": 20.0, "building": 10.0})
    site = _make_site()

    onboard_site(client, site)

    assert client.satellite_segmentation.call_count == 1
    assert client.street_view_segmentation.call_count == 1


def test_reonboarding_same_site_does_not_recall_premium_endpoints():
    client = _client_with({"tree": 30.0}, {"tree": 20.0, "building": 10.0})
    site = _make_site()

    onboard_site(client, site)
    onboard_site(client, site)  # re-run, already onboarded, no force

    assert client.satellite_segmentation.call_count == 1
    assert client.street_view_segmentation.call_count == 1


def test_shade_coverage_pct_derived_correctly_from_segmentation_output():
    client = _client_with(
        satellite_segments={"tree": 30.0},
        streetview_segments={
            "tree": 25.0,
            "building": 15.0,
            "sky": 40.0,
            "road, sidewalk": 20.0,
        },
    )
    site = _make_site()

    onboard_site(client, site)

    # tree-canopy + building-shadow classes, summed -- other classes ignored.
    assert site.shade_coverage_pct == pytest.approx(40.0)
    assert site.canopy_pct == pytest.approx(30.0)
