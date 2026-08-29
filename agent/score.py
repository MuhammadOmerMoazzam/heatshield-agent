"""HeatShield risk-scoring model.

Formula
-------
    final_score = raw_stress(heat_index, humidity, solar_irradiance)
                  * exposure_modifier(shade_coverage_pct, work_intensity)

    risk_tier = classify_risk_tier(final_score)

``raw_stress`` starts from the measured heat index (degrees Fahrenheit)
and adds stress that NWS/OSHA heat-index tables -- which are measured in
shade -- don't fully capture on their own for an outdoor crew working in
direct sun:

- **Humidity above 40%.** Heat-index tables already account for
  humidity's effect on evaporative cooling up to a point, but sustained
  humidity above 40% further impairs a working body's ability to cool
  itself, so a small linear penalty is added for every percentage point
  above that threshold.
- **Solar irradiance.** Standard heat index is a shaded-thermometer
  reading; a crew working in direct sun experiences real additional heat
  load proportional to solar irradiance (W/m^2), added as a small linear
  penalty.

``exposure_modifier`` is a multiplier around 1.0 combining two
onboarding- and crew-level factors:

- **Shade coverage** (site-level, from satellite/street-view
  segmentation). More shade reduces the modifier, down to a floor of 0.7
  at 100% canopy/building shade. A site onboarded before segmentation
  finishes has no shade data yet -- rather than crash or silently assume
  full shade (which would understate risk), missing shade data is
  treated as 0% shade, the conservative, higher-risk assumption.
- **Work intensity** (crew-level). Heavier physical work generates more
  internal heat, so "heavy" raises the modifier and "light" lowers it
  relative to "moderate" (also the default for any unrecognized value,
  so a bad onboarding record degrades gracefully instead of halting the
  sense -> score -> decide -> act loop).

Risk tiers
----------
``final_score`` is expressed in the same degrees-Fahrenheit-equivalent
units as heat index, so it classifies directly against the OSHA/NIOSH
heat-index risk categories (NOAA/NWS heat index chart):

=========  ===============================
Tier       final_score (°F HI-equivalent)
=========  ===============================
safe       < 91
caution    91 to < 103
high       103 to < 125
extreme    >= 125
=========  ===============================

These four boundary values (91 / 103 / 125 °F) are the single source of
truth for both this module and the README -- if they ever need to
change, change them here first.

Units
-----
``heat_index_f`` is expected in **Fahrenheit**. FortyGuard's
``env_params`` endpoint returns ``heat_index_celsius``; convert with
``(celsius * 9 / 5) + 32`` before calling ``compute_raw_stress``.
"""

from __future__ import annotations

HUMIDITY_THRESHOLD_PCT = 40.0
HUMIDITY_STRESS_PER_PCT = 0.2
SOLAR_STRESS_PER_WM2 = 0.01

MAX_SHADE_REDUCTION = 0.3  # 100% shade cuts the modifier by up to 30%
WORK_INTENSITY_FACTORS = {
    "light": 0.90,
    "moderate": 1.00,
    "heavy": 1.15,
}

SAFE_MAX_F = 91.0
CAUTION_MAX_F = 103.0
HIGH_MAX_F = 125.0


def compute_raw_stress(
    heat_index_f: float, humidity_pct: float | None, solar_irradiance: float | None
) -> float:
    """Heat index (°F) plus a humidity penalty above 40% and a solar-loading penalty.

    ``humidity_pct``/``solar_irradiance`` of ``None`` (live-verified,
    Phase 9: environmental_parameters can independently have no data for
    a given cycle) contributes no penalty for that factor, the same as a
    measured value at/under its own no-penalty threshold -- a real heat
    index is still worth scoring even without the small additive
    adjustments these two normally contribute.
    """
    stress = heat_index_f
    if humidity_pct is not None and humidity_pct > HUMIDITY_THRESHOLD_PCT:
        stress += (humidity_pct - HUMIDITY_THRESHOLD_PCT) * HUMIDITY_STRESS_PER_PCT
    if solar_irradiance is not None and solar_irradiance > 0:
        stress += solar_irradiance * SOLAR_STRESS_PER_WM2
    return stress


def compute_exposure_modifier(shade_coverage_pct: float | None, work_intensity: str) -> float:
    """Multiplier around 1.0 from site shade coverage and crew work intensity.

    ``shade_coverage_pct=None`` (site not yet onboarded) defaults to 0%
    shade -- the conservative, higher-risk assumption -- rather than
    raising or assuming full coverage.
    """
    if shade_coverage_pct is None:
        shade_coverage_pct = 0.0
    shade_factor = 1.0 - (shade_coverage_pct / 100.0) * MAX_SHADE_REDUCTION
    intensity_factor = WORK_INTENSITY_FACTORS.get(work_intensity, WORK_INTENSITY_FACTORS["moderate"])
    return shade_factor * intensity_factor


def compute_final_score(raw_stress: float, exposure_modifier: float) -> float:
    return raw_stress * exposure_modifier


def classify_risk_tier(final_score: float) -> str:
    """Classify a final_score against the documented OSHA/NIOSH-derived thresholds."""
    if final_score < SAFE_MAX_F:
        return "safe"
    if final_score < CAUTION_MAX_F:
        return "caution"
    if final_score < HIGH_MAX_F:
        return "high"
    return "extreme"
