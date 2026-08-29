"""Site onboarding: satellite + street-view segmentation, run once per site.

Derives two distinct metrics and writes them onto the ``Site`` row:

- ``canopy_pct`` -- overhead tree-canopy extent, from satellite
  segmentation's ``tree`` class (a top-down view of vegetation cover).
- ``shade_coverage_pct`` -- ground-level shade a worker actually stands
  in, from street-view segmentation's ``tree`` (canopy) + ``building``
  (shadow) classes summed together. Street-view is the ground-level,
  person's-eye view, so it's the right source for "is this spot shaded",
  as distinct from satellite's overhead canopy signal.

FortyGuard's segmentation class vocabulary is open and location-dependent
(confirmed against both the quickstart reference notebooks and the live
API docs -- neither publishes a fixed enumerated class list), so classes
are matched by keyword rather than exact name, mirroring the technique
FortyGuard's own use-case notebooks use (``'tree' in cls.lower()``) to
absorb label variations like the composite ``"road, route"`` style keys
the API returns.

Idempotent: re-running for an already-onboarded site (one that already
has shade data) is a no-op unless ``force=True``, since satellite/
streetview are Premium, credit-metered endpoints and a site's physical
shade doesn't change hour to hour -- there's no reason to re-spend
credits on every cycle (Phase 7's loop calls this once per site, not
once per sense cycle).

Phase 8: both segmentation responses also carry a base64 overlay image
(``segmentation.image_content`` for satellite, ``front.segmented_image``
for street-view) plus an ``image_legend`` ({class_name: hex_color}) --
these are decoded and saved to ``outputs/`` (mirroring
``FortyGuardClient.heat_intelligence``'s own PDF-to-disk convention)
rather than discarded after the percentages are derived, so the
dashboard has something to render.
"""

from __future__ import annotations

import base64
from datetime import date
from pathlib import Path

from agent._shared import REPO_ROOT
from agent.fortyguard_client import FortyGuardClient
from agent.models import Site

_CANOPY_KEYWORDS = ("tree",)
_SHADE_KEYWORDS = ("tree", "building")


def _sum_matching_segments(segments: dict, keywords: tuple[str, ...]) -> float:
    total = 0.0
    for cls, pct in segments.items():
        cls_lower = cls.lower()
        if any(keyword in cls_lower for keyword in keywords):
            total += pct
    return total


def _derive_canopy_pct(satellite_result: dict) -> float:
    segments = satellite_result.get("segmentation", {}).get("segments", {})
    return _sum_matching_segments(segments, _CANOPY_KEYWORDS)


def _derive_shade_coverage_pct(streetview_result: dict) -> float:
    segments = streetview_result.get("front", {}).get("segments", {})
    return _sum_matching_segments(segments, _SHADE_KEYWORDS)


def _decode_and_save_image(b64_data, out_path: Path) -> Path | None:
    """Decode a base64 image (or the first of a list of them) and write it
    to ``out_path``. Returns ``None`` without writing anything if the API
    response carried no image -- not every plan/response includes one.
    """
    if isinstance(b64_data, list):
        b64_data = b64_data[0] if b64_data else None
    if not b64_data:
        return None
    if b64_data.startswith("data:"):
        b64_data = b64_data.split(",", 1)[1]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(base64.b64decode(b64_data))
    return out_path


def onboard_site(
    client: FortyGuardClient,
    site: Site,
    *,
    reference_date: str | None = None,
    force: bool = False,
) -> Site:
    """Onboard ``site``: fetch satellite + street-view segmentation exactly
    once, derive ``shade_coverage_pct``/``canopy_pct``, and write them onto
    the row. A no-op if the site already has shade data, unless
    ``force=True``.
    """
    if site.shade_coverage_pct is not None and not force:
        return site

    start_date = reference_date or date.today().isoformat()

    # Segmentation (image processing + ML inference) genuinely takes
    # longer than the client's 60s default -- confirmed via a live call
    # during Phase 7's integration verification, which timed out at the
    # default before completing.
    _SEGMENTATION_TIMEOUT = 300.0

    satellite_response = client.satellite_segmentation(
        latitude=site.lat,
        longitude=site.lon,
        start_date=start_date,
        filter_type=3,
        timeout=_SEGMENTATION_TIMEOUT,
    )
    # A small positive vertical_angle captures more sky/canopy, per the
    # quickstart's own guidance for shade-and-canopy analysis.
    streetview_response = client.street_view_segmentation(
        latitude=site.lat,
        longitude=site.lon,
        vertical_angle=10.0,
        horizontal_angle=0.0,
        back_view=False,
        timeout=_SEGMENTATION_TIMEOUT,
    )

    site.canopy_pct = _derive_canopy_pct(satellite_response["result"])
    site.shade_coverage_pct = _derive_shade_coverage_pct(streetview_response["result"])

    satellite_segmentation = satellite_response["result"].get("segmentation", {})
    satellite_activity_id = satellite_response.get("activity_id", "unknown")
    satellite_image = _decode_and_save_image(
        satellite_segmentation.get("image_content"),
        REPO_ROOT / "outputs" / f"satellite_{satellite_activity_id}.png",
    )
    site.satellite_image_path = str(satellite_image) if satellite_image else None
    site.satellite_legend = satellite_segmentation.get("image_legend") or None

    streetview_front = streetview_response["result"].get("front", {})
    streetview_activity_id = streetview_response.get("activity_id", "unknown")
    streetview_image = _decode_and_save_image(
        streetview_front.get("segmented_image"),
        REPO_ROOT / "outputs" / f"streetview_{streetview_activity_id}.png",
    )
    site.streetview_image_path = str(streetview_image) if streetview_image else None
    site.streetview_legend = streetview_front.get("image_legend") or None

    return site
