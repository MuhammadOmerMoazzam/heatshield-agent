"""Tests for agent.onboarding.onboard_site.

The FortyGuardClient is stubbed (satellite_segmentation/street_view_
segmentation are mocked directly) -- this file tests onboarding logic
(call-once, idempotent, shade-derivation), not the HTTP wrapper itself
(covered by test_fortyguard_client.py).
"""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent.models import Site
from agent.onboarding import onboard_site

# The smallest possible valid PNG (1x1, transparent) -- real enough to
# round-trip through base64 decode + file write without needing Pillow.
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _make_site(**overrides) -> Site:
    defaults = dict(
        name="Test Site",
        lat=37.33,
        lon=-121.90,
        polygon_geojson={"type": "Polygon", "coordinates": [[[0, 0]]]},
    )
    defaults.update(overrides)
    return Site(**defaults)


def _satellite_result(segments: dict, *, with_image: bool = False) -> dict:
    segmentation = {"segments": segments}
    if with_image:
        segmentation["image_content"] = _TINY_PNG_B64
        segmentation["image_legend"] = {"tree": "#2e7d32"}
    return {"activity_id": "sat-1", "result": {"segmentation": segmentation}}


def _streetview_result(segments: dict, *, with_image: bool = False) -> dict:
    front = {"segments": segments}
    if with_image:
        front["segmented_image"] = _TINY_PNG_B64
        front["image_legend"] = {"tree": "#2e7d32", "building": "#616161"}
    return {"activity_id": "sv-1", "result": {"front": front}}


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


def test_onboarding_persists_segmentation_images_and_legends(tmp_path, monkeypatch):
    """Phase 8: the dashboard renders these images/legends straight from
    disk/DB, so onboarding must decode and save them, not discard them
    after deriving the numeric percentages.
    """
    monkeypatch.setattr("agent.onboarding.REPO_ROOT", tmp_path)

    client = MagicMock()
    client.satellite_segmentation.return_value = _satellite_result(
        {"tree": 30.0}, with_image=True
    )
    client.street_view_segmentation.return_value = _streetview_result(
        {"tree": 20.0, "building": 10.0}, with_image=True
    )
    site = _make_site()

    onboard_site(client, site)

    assert site.satellite_image_path is not None
    assert Path(site.satellite_image_path).is_file()
    assert Path(site.satellite_image_path).read_bytes() == base64.b64decode(_TINY_PNG_B64)
    assert site.satellite_legend == {"tree": "#2e7d32"}

    assert site.streetview_image_path is not None
    assert Path(site.streetview_image_path).is_file()
    assert Path(site.streetview_image_path).read_bytes() == base64.b64decode(_TINY_PNG_B64)
    assert site.streetview_legend == {"tree": "#2e7d32", "building": "#616161"}


def test_onboarding_leaves_image_fields_null_when_api_returns_no_image():
    """Existing behavior (no image_content/segmented_image in the response,
    matching every other test in this file) must not crash or fabricate a
    path -- the dashboard already handles a null image path gracefully.
    """
    client = _client_with({"tree": 30.0}, {"tree": 20.0, "building": 10.0})
    site = _make_site()

    onboard_site(client, site)

    assert site.satellite_image_path is None
    assert site.satellite_legend is None
    assert site.streetview_image_path is None
    assert site.streetview_legend is None
