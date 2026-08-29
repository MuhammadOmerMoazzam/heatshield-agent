"""Compliance report action.

Calls heat_intelligence (underscore, per Rule 1) only on high/extreme
tiers, using the reading's own timestamp -- which already matches the
triggering heatmap call's date, per Sense's Rule-2-compliant design
(agent/sense.py) -- as the report's `date`.
"""

from __future__ import annotations

from agent.fortyguard_client import FortyGuardClient
from agent.models import Reading, Site

_REPORT_TIERS = ("high", "extreme")


def generate_compliance_report(
    client: FortyGuardClient, site: Site, reading: Reading, tier: str
) -> str | None:
    if tier not in _REPORT_TIERS:
        return None

    # reading.heat_index is already in Fahrenheit (converted at the Sense
    # boundary) -- heat_intelligence's `temperature` input is Fahrenheit
    # too (confirmed against the live docs, unlike environmental_parameters
    # which takes Celsius), so no conversion is needed here.
    result = client.heat_intelligence(
        site.lat,
        site.lon,
        temperature=reading.heat_index,
        date=reading.ts.date().isoformat(),
    )
    return str(result)
