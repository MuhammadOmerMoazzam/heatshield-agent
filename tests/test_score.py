"""Tests for agent.score -- the risk-scoring model.

See agent/score.py's module docstring for the formula and threshold
rationale (OSHA/NIOSH heat-index guidance).
"""

from __future__ import annotations

import pytest

from agent.score import (
    classify_risk_tier,
    compute_exposure_modifier,
    compute_final_score,
    compute_raw_stress,
)


def test_raw_stress_increases_with_humidity_above_40pct():
    low = compute_raw_stress(heat_index_f=95.0, humidity_pct=30.0, solar_irradiance=0.0)
    at_threshold = compute_raw_stress(heat_index_f=95.0, humidity_pct=40.0, solar_irradiance=0.0)
    above = compute_raw_stress(heat_index_f=95.0, humidity_pct=70.0, solar_irradiance=0.0)

    # No penalty at/under the 40% threshold.
    assert at_threshold == low
    # A real penalty once humidity exceeds it.
    assert above > at_threshold


def test_raw_stress_increases_with_solar_irradiance():
    low = compute_raw_stress(heat_index_f=95.0, humidity_pct=30.0, solar_irradiance=100.0)
    high = compute_raw_stress(heat_index_f=95.0, humidity_pct=30.0, solar_irradiance=800.0)

    assert high > low


def test_raw_stress_handles_missing_humidity_and_solar_gracefully():
    """Live-verified (Phase 9): environmental_parameters can independently
    have no data for a given cycle -- sense_live then reports humidity/
    solar_irradiance as None rather than guessing. compute_raw_stress must
    not crash on that; a None signal simply contributes no penalty for
    that factor, same as a measured value at/under its own no-penalty
    threshold, rather than blocking the score entirely.
    """
    baseline = compute_raw_stress(heat_index_f=95.0, humidity_pct=30.0, solar_irradiance=0.0)
    missing_both = compute_raw_stress(heat_index_f=95.0, humidity_pct=None, solar_irradiance=None)

    assert missing_both == baseline == 95.0


def test_exposure_modifier_reduces_score_with_higher_shade_coverage():
    no_shade = compute_exposure_modifier(shade_coverage_pct=0.0, work_intensity="moderate")
    full_shade = compute_exposure_modifier(shade_coverage_pct=90.0, work_intensity="moderate")

    assert full_shade < no_shade


def test_exposure_modifier_increases_score_with_heavy_work_intensity():
    light = compute_exposure_modifier(shade_coverage_pct=50.0, work_intensity="light")
    heavy = compute_exposure_modifier(shade_coverage_pct=50.0, work_intensity="heavy")

    assert heavy > light


@pytest.mark.parametrize(
    "final_score, expected_tier",
    [
        (50.0, "safe"),
        (90.9, "safe"),
        (91.0, "caution"),
        (95.0, "caution"),
        (102.9, "caution"),
        (103.0, "high"),
        (110.0, "high"),
        (124.9, "high"),
        (125.0, "extreme"),
        (140.0, "extreme"),
    ],
)
def test_risk_tier_boundaries_match_documented_thresholds(final_score, expected_tier):
    assert classify_risk_tier(final_score) == expected_tier


def test_score_is_deterministic_for_same_inputs():
    raw_1 = compute_raw_stress(heat_index_f=100.0, humidity_pct=55.0, solar_irradiance=650.0)
    raw_2 = compute_raw_stress(heat_index_f=100.0, humidity_pct=55.0, solar_irradiance=650.0)
    assert raw_1 == raw_2

    modifier_1 = compute_exposure_modifier(shade_coverage_pct=30.0, work_intensity="heavy")
    modifier_2 = compute_exposure_modifier(shade_coverage_pct=30.0, work_intensity="heavy")
    assert modifier_1 == modifier_2

    score_1 = compute_final_score(raw_1, modifier_1)
    score_2 = compute_final_score(raw_2, modifier_2)
    assert score_1 == score_2
    assert classify_risk_tier(score_1) == classify_risk_tier(score_2)


def test_score_handles_missing_shade_data_gracefully():
    # Should not raise, and must default to the conservative (no-shade,
    # i.e. worst-case / highest-modifier) outcome.
    missing = compute_exposure_modifier(shade_coverage_pct=None, work_intensity="moderate")
    no_shade = compute_exposure_modifier(shade_coverage_pct=0.0, work_intensity="moderate")
    partial_shade = compute_exposure_modifier(shade_coverage_pct=50.0, work_intensity="moderate")

    assert missing == no_shade
    assert missing >= partial_shade
