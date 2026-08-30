# HeatShield Agent — Submission Summary

## Problem

Outdoor crews (construction, warehousing, logistics, event staffing) face heat-illness risk that a plain ambient-temperature reading gets wrong. Two crews in the same city at the same hour face different real risk depending on their site's actual shade and how physically demanding their work is. The standard response — a safety officer checking a weather app and using judgment — doesn't scale, isn't logged for compliance, and isn't proactive about tomorrow's forecast.

## User

A safety/compliance officer responsible for outdoor crews at industrial or commercial sites, who needs a risk signal specific to each site's real exposure conditions and an audit trail proving heat-safety decisions were made and acted on — not just recorded after the fact.

## FortyGuard Usage

HeatShield Agent runs a fully autonomous sense → score → decide → act loop against real FortyGuard data, with no human approval step between classification and action:

- **Sense**: `POST /v1/heatmap` (today + 12h-forecast aggregates) and `POST /v1/env_params` (heat index, humidity, solar irradiance, AQI) called against the *same* timestamp, per the handbook's matching-date/time guidance for these endpoints.
- **Onboarding** (once per site): `POST /v1/satellite` and `POST /v1/streetview` segmentation derive real, site-specific shade coverage — the input a plain heat-index reading has no way to see.
- **Score**: a real model (`agent/score.py`), not a passthrough — `final_score = raw_stress(heat_index, humidity, solar_irradiance) × exposure_modifier(shade_coverage, work_intensity)`, classified against OSHA/NIOSH heat-index tiers.
- **Decide + Act**: a tier maps to a concrete action — nothing, break reminders, a shortened shift with a Slack alert, or a full halt plus a `POST /v1/heat_intelligence` compliance PDF on high/extreme. The +12h forecast branch uses a deliberately smaller action set (flag-for-reschedule only — never a live halt or report over something that hasn't happened yet). Every call writes an audit-log row, and `fetch-api-key-usage` throttles non-essential onboarding calls if credits run low.

The agent runs against four real, climatically distinct US sites (Phoenix AZ, Houston TX, Miami FL, Las Vegas NV) — nothing in the pipeline is hardcoded to them; it works against any US location.

## Measured Result

On a live run today, Phoenix's measured heat index was **96.9°F — "caution" on a plain heat-index lookup.** HeatShield's exposure model, factoring in this crew's heavy work intensity and the site's real (satellite/street-view-derived) shade coverage — near zero in an open warehouse district — raised the score to **111.4, crossing into "high."** The agent autonomously shortened the shift, posted a Slack alert, and generated a compliance-report PDF, with zero human review between the reading and the action. A temperature-only system would have sent a break reminder and moved on; HeatShield caught real exposure risk that "caution" would have missed — and logged the entire decision chain as it happened.
