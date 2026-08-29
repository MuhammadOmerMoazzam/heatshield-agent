"""Python client for the FortyGuard tOS Enterprise API.

One method per endpoint. Analysis endpoints are async task-based: submit,
get an ``activity_id``, then poll ``/v1/status/{id}`` until it reaches a
terminal state. ``wait_for`` polls on an escalating 3s -> 6s -> 12s backoff
(handbook best practice) rather than hammering the endpoint at a fixed
interval.

This module is HeatShield's own implementation. It was informed by, but
does not import from or copy, FortyGuard's temperature-api-quickstart
reference client, cross-checked against the live API docs at
https://docs-api.fortyguard.com/docs.
"""

from __future__ import annotations

import math
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import httpx
from dotenv import load_dotenv

# So FORTYGUARD_API_KEY/FORTYGUARD_BASE_URL from .env are actually picked
# up, not just documented in .env.example -- load_dotenv() only sets keys
# not already in os.environ, so a real shell-exported var still wins.
load_dotenv()

DEFAULT_BASE_URL = "https://api.fortyguard.com"
_TERMINAL_SUCCESS = {"completed", "succeeded"}
_TERMINAL_FAILURE = {"failed", "error"}
_DEFAULT_BACKOFF: tuple[float, ...] = (3.0, 6.0, 12.0)
_MAX_FORECAST_HOURS = 12

# Confirmed against https://docs-api.fortyguard.com/docs/create-heatmap:
# Basic is capped at 10 mi^2, Premium at 50 mi^2. HeatShield's client
# validates against the Premium cap (the more permissive of the two); the
# API itself is the final authority for a Basic-tier key.
_MAX_AOI_MI2 = 50.0
_MI2_PER_M2 = 1 / 2_589_988.110336

# Coarse US-region guard for client-side pre-validation -- not a geodesic
# border check. The US/Canada border near the Great Lakes doesn't follow
# lines of latitude/longitude (Toronto and Buffalo, ~100 km apart, sit on
# opposite sides of it at similar lat/lon), so no axis-aligned box
# separates every case perfectly. This only needs to catch the common case
# before spending credits; the API is the real authority on borders (see
# the quickstart README's Troubleshooting table).
_US_BOUNDS = (
    (24.0, 49.5, -125.0, -66.0),  # CONUS
    (51.0, 72.0, -170.0, -129.0),  # Alaska
    (18.0, 23.0, -160.0, -154.0),  # Hawaii
    (17.5, 18.6, -67.5, -65.0),  # Puerto Rico
)
_NON_US_CARVEOUT = (42.0, 46.9, -84.5, -75.5)  # southern Ontario / Great Lakes north shore


class FortyGuardError(Exception):
    """Base class for all FortyGuard client errors."""


class ValidationError(FortyGuardError, ValueError):
    """Client-side pre-validation failed before any network call was made."""


class ForecastWindowError(ValidationError):
    """Requested timestamp is beyond the +12h heatmap-only forecast window."""


class TaskFailedError(FortyGuardError):
    """The async task finished with status Failed/Error."""


class TaskTimeoutError(FortyGuardError):
    """The async task did not reach a terminal status within the polling budget."""


def _is_us_coordinate(lat: float, lon: float) -> bool:
    in_region = any(
        min_lat <= lat <= max_lat and min_lon <= lon <= max_lon
        for min_lat, max_lat, min_lon, max_lon in _US_BOUNDS
    )
    if not in_region:
        return False
    c_min_lat, c_max_lat, c_min_lon, c_max_lon = _NON_US_CARVEOUT
    return not (c_min_lat <= lat <= c_max_lat and c_min_lon <= lon <= c_max_lon)


def _iter_rings(polygon_aoi: dict) -> Iterable[list]:
    for feature in polygon_aoi.get("features", []):
        geometry = feature.get("geometry", {})
        gtype = geometry.get("type")
        coords = geometry.get("coordinates", [])
        if gtype == "Polygon":
            yield from coords
        elif gtype == "MultiPolygon":
            for polygon in coords:
                yield from polygon


def _ring_area_m2(ring: list) -> float:
    """Approximate ring area via the shoelace formula on an equirectangular
    projection centered on the ring. Accurate enough for a client-side
    pre-flight size guard at the tens-of-km scale these AOIs live at.
    """
    lats = [pt[1] for pt in ring]
    lons = [pt[0] for pt in ring]
    lat0 = sum(lats) / len(lats)
    lon0 = sum(lons) / len(lons)
    m_per_deg_lat = 110_540.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))
    xy = [((lon - lon0) * m_per_deg_lon, (lat - lat0) * m_per_deg_lat) for lon, lat in ring]
    area = 0.0
    for (x1, y1), (x2, y2) in zip(xy, xy[1:]):
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _validate_polygon_aoi(polygon_aoi: dict) -> None:
    total_area_mi2 = 0.0
    ring_count = 0
    for ring in _iter_rings(polygon_aoi):
        if len(ring) < 4 or ring[0] != ring[-1]:
            raise ValidationError(
                "Polygon ring is not closed: first and last coordinates must match."
            )
        ring_count += 1
        total_area_mi2 += _ring_area_m2(ring) * _MI2_PER_M2
    if ring_count == 0:
        raise ValidationError("polygon_aoi contains no polygon rings.")
    if total_area_mi2 > _MAX_AOI_MI2:
        raise ValidationError(
            f"AOI is approximately {total_area_mi2:.1f} mi², exceeding the "
            f"{_MAX_AOI_MI2:.0f} mi² heatmap cap."
        )


def _validate_forecast_window(start_date: str, start_time: str | None, *, now: datetime) -> None:
    if start_time is None:
        return
    requested = datetime.fromisoformat(f"{start_date}T{start_time}")
    if requested > now + timedelta(hours=_MAX_FORECAST_HOURS):
        raise ForecastWindowError(
            f"Requested time {requested.isoformat()} is more than "
            f"{_MAX_FORECAST_HOURS}h in the future; only heatmap forecasts within "
            f"+{_MAX_FORECAST_HOURS}h are supported."
        )


def _validate_us_coordinate(latitude: float, longitude: float) -> None:
    if not _is_us_coordinate(latitude, longitude):
        raise ValidationError(
            f"({latitude}, {longitude}) is outside FortyGuard's supported U.S. "
            f"coverage area."
        )


class FortyGuardClient:
    """Thin wrapper around the FortyGuard tOS Enterprise API.

    Parameters
    ----------
    api_key:
        FortyGuard API key. Falls back to the ``FORTYGUARD_API_KEY`` env var.
    base_url:
        API root. Falls back to ``FORTYGUARD_BASE_URL`` then the prod default.
    timeout:
        Per-request HTTP timeout in seconds.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key or os.getenv("FORTYGUARD_API_KEY")
        if not self.api_key:
            raise FortyGuardError(
                "No API key provided. Pass api_key=... or set FORTYGUARD_API_KEY in your .env file."
            )
        self.base_url = (base_url or os.getenv("FORTYGUARD_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.credits_used_estimate = 0
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"api-key": self.api_key, "Content-Type": "application/json"},
            timeout=timeout,
        )

    # ------------------------------------------------------------ core

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        resp = self._client.request(method, path, **kwargs)
        if resp.status_code >= 400:
            raise FortyGuardError(f"{method} {path} -> {resp.status_code}: {resp.text[:500]}")
        return resp

    def _submit(self, path: str, payload: dict) -> str:
        body = self._request("POST", path, json=payload).json()
        if body.get("error"):
            raise FortyGuardError(body.get("message", "Submission failed"))
        try:
            return body["data"]["activity_id"]
        except KeyError as exc:
            raise FortyGuardError(f"Unexpected response shape: {body}") from exc

    def get_status(self, activity_id: str) -> dict:
        """Return the raw status-endpoint ``data`` for an activity.

        Right after submission the status endpoint can briefly 404 while the
        activity propagates; that is treated as a transient "pending" state
        so pollers keep retrying instead of failing.
        """
        resp = self._client.get(f"/v1/status/{activity_id}")
        if resp.status_code == 404:
            return {"status": "pending"}
        if resp.status_code >= 400:
            raise FortyGuardError(
                f"GET /v1/status/{activity_id} -> {resp.status_code}: {resp.text[:500]}"
            )
        body = resp.json()
        if body.get("error"):
            raise FortyGuardError(body.get("message", "Status lookup failed"))
        return body["data"]

    def wait_for(
        self,
        activity_id: str,
        *,
        backoff: tuple[float, ...] = _DEFAULT_BACKOFF,
        timeout: float | None = None,
        on_tick: Any = None,
    ) -> dict:
        """Poll the status endpoint until the task terminates.

        Polls on an escalating backoff (default 3s -> 6s -> 12s, then holds
        at the last value) rather than a fixed interval. Returns the
        ``result`` payload on success; raises on failure or timeout.
        """
        deadline = time.monotonic() + (timeout if timeout is not None else self.timeout)
        attempt = 0
        while True:
            data = self.get_status(activity_id)
            status = str(data.get("status", "")).lower()
            if on_tick:
                on_tick(status, data)
            if status in _TERMINAL_SUCCESS:
                self.credits_used_estimate += 1
                return data.get("result", data)
            if status in _TERMINAL_FAILURE:
                raise TaskFailedError(
                    f"Activity {activity_id} failed: {data.get('message') or data}"
                )
            if time.monotonic() >= deadline:
                raise TaskTimeoutError(
                    f"Activity {activity_id} still '{status}' after timeout"
                )
            interval = backoff[min(attempt, len(backoff) - 1)]
            time.sleep(interval)
            attempt += 1

    def _submit_and_wait(self, path: str, payload: dict, *, timeout: float | None = None) -> dict:
        activity_id = self._submit(path, payload)
        result = self.wait_for(activity_id, timeout=timeout)
        return {"activity_id": activity_id, "result": result}

    # ---------------------------------------------------------- analysis API

    def create_heatmap(
        self,
        polygon_aoi: dict,
        start_date: str,
        filter_type: int,
        granularity: int = 100,
        start_time: str | None = None,
        end_time: str | None = None,
        end_date: str | None = None,
        analytic_type: str = "tcm",
        threshold: float | None = None,
        direction: str | None = None,
        *,
        wait: bool = True,
        timeout: float | None = None,
    ) -> dict | str:
        """POST /v1/heatmap -- generate a thermal map over a polygon AOI.

        ``filter_type``: 1=single hour, 2=range of hours, 3=single day,
        4=range of days (pass ``end_date``).
        """
        _validate_polygon_aoi(polygon_aoi)
        _validate_forecast_window(
            start_date, start_time, now=datetime.now(timezone.utc).replace(tzinfo=None)
        )

        date_time: dict[str, Any] = {"start_date": start_date, "filter_type": filter_type}
        if start_time is not None:
            date_time["start_time"] = start_time
        if end_time is not None:
            date_time["end_time"] = end_time
        if end_date is not None:
            date_time["end_date"] = end_date

        payload: dict[str, Any] = {
            "polygon_aoi": polygon_aoi,
            "date_time": date_time,
            "granularity": granularity,
            "analytic_type": analytic_type,
        }
        if threshold is not None:
            payload["threshold"] = threshold
        if direction is not None:
            payload["direction"] = direction

        if not wait:
            return self._submit("/v1/heatmap", payload)
        return self._submit_and_wait("/v1/heatmap", payload, timeout=timeout)

    def environmental_parameters(
        self,
        latitude: float,
        longitude: float,
        temperature: float,
        reference_ts: datetime,
        *,
        analysis: Iterable[str] | None = None,
        wait: bool = True,
        timeout: float | None = None,
    ) -> dict | str:
        """POST /v1/env_params -- heat index, AQI, solar irradiance, and more.

        ``reference_ts`` is a required positional argument with no default.
        Per the handbook (Rule 2), this endpoint must be called against the
        exact same timestamp already used for a ``create_heatmap`` call at
        this location -- there is deliberately no silent "now" fallback, so
        callers cannot call this on an independent clock by omission.
        """
        _validate_us_coordinate(latitude, longitude)

        date_time = {
            "start_date": reference_ts.date().isoformat(),
            "start_time": reference_ts.strftime("%H:%M"),
            "filter_type": 1,
        }
        payload: dict[str, Any] = {
            "latitude": latitude,
            "longitude": longitude,
            "temperature": temperature,
            "date_time": date_time,
        }
        if analysis is not None:
            payload["analysis"] = list(analysis)

        if not wait:
            return self._submit("/v1/env_params", payload)
        return self._submit_and_wait("/v1/env_params", payload, timeout=timeout)

    def satellite_segmentation(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        filter_type: int,
        granularity: int = 100,
        start_time: str | None = None,
        end_time: str | None = None,
        end_date: str | None = None,
        *,
        wait: bool = True,
        timeout: float | None = None,
    ) -> dict | str:
        """POST /v1/satellite -- land-cover segmentation of a satellite tile (Premium)."""
        _validate_us_coordinate(latitude, longitude)

        date_time: dict[str, Any] = {"start_date": start_date, "filter_type": filter_type}
        if start_time is not None:
            date_time["start_time"] = start_time
        if end_time is not None:
            date_time["end_time"] = end_time
        if end_date is not None:
            date_time["end_date"] = end_date

        payload = {
            "sat": {"latitude": latitude, "longitude": longitude},
            "date_time": date_time,
            "granularity": granularity,
        }
        if not wait:
            return self._submit("/v1/satellite", payload)
        return self._submit_and_wait("/v1/satellite", payload, timeout=timeout)

    def street_view_segmentation(
        self,
        latitude: float,
        longitude: float,
        vertical_angle: float = 0.0,
        horizontal_angle: float = 0.0,
        back_view: bool = False,
        *,
        wait: bool = True,
        timeout: float | None = None,
    ) -> dict | str:
        """POST /v1/streetview -- segmentation of a ground-level street view (Premium)."""
        _validate_us_coordinate(latitude, longitude)

        payload = {
            "latitude": latitude,
            "longitude": longitude,
            "vertical_angle": vertical_angle,
            "horizontal_angle": horizontal_angle,
            "back_view": back_view,
        }
        if not wait:
            return self._submit("/v1/streetview", payload)
        return self._submit_and_wait("/v1/streetview", payload, timeout=timeout)

    def heat_intelligence(
        self,
        latitude: float,
        longitude: float,
        temperature: float,
        date: str,
        analysis: Iterable[str] = ("environmental",),
        output_path: str | Path | None = None,
        *,
        timeout: float | None = None,
    ) -> Path:
        """POST /v1/heat_intelligence -- generate a heat-intelligence report.

        Note the underscore in the path (Rule 1) -- there is no
        ``/v1/heat-intelligence`` (dash) endpoint. ``temperature`` is in
        **Fahrenheit** here -- unlike ``environmental_parameters``, which
        takes Celsius -- confirmed against the live docs.

        The completed status response carries a short-lived signed
        ``result.download_link``, not the PDF itself. Per the docs' own
        guidance ("use it immediately... do not log or share the full
        signed URL"), this downloads the PDF right away and returns the
        local file path rather than the URL.
        """
        _validate_us_coordinate(latitude, longitude)

        payload = {
            "latitude": latitude,
            "longitude": longitude,
            "temperature": temperature,
            "date": date,
            "analysis": list(analysis),
        }
        activity_id = self._submit("/v1/heat_intelligence", payload)
        result = self.wait_for(activity_id)
        download_link = result.get("download_link")
        if not download_link:
            raise FortyGuardError(
                f"Activity {activity_id} completed but the status response "
                f"contained no download_link"
            )

        target = Path(output_path) if output_path else Path("outputs") / f"heat_intelligence_{activity_id}.pdf"
        target.parent.mkdir(parents=True, exist_ok=True)
        with httpx.stream("GET", download_link, timeout=self.timeout) as resp:
            if resp.status_code >= 400:
                raise FortyGuardError(f"Downloading report -> {resp.status_code}")
            with target.open("wb") as fh:
                for chunk in resp.iter_bytes():
                    fh.write(chunk)
        return target

    # ------------------------------------------------------------- credits

    def fetch_api_key_usage(self) -> dict:
        """GET /v1/system/fetch-api-key-usage -- current plan status and credit balance."""
        body = self._request("GET", "/v1/system/fetch-api-key-usage").json()
        if body.get("error"):
            raise FortyGuardError(body.get("message", "Usage lookup failed"))
        return body.get("data", {})
