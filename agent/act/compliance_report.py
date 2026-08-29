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

# heat_intelligence is the same slow async submit-then-poll pattern that
# made onboarding's satellite/streetview calls time out at the client's
# 60s default (fixed in Phase 7 after a live TaskTimeoutError) -- and it
# fires on every high/extreme decision, not once per site, so it needs
# the same treatment.
_HEAT_INTELLIGENCE_TIMEOUT = 300.0


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
        timeout=_HEAT_INTELLIGENCE_TIMEOUT,
    )
    return str(result)
