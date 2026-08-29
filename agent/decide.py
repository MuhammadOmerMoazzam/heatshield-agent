"""Decide + Act: the agentic core.

No human approves an action between risk classification and execution --
that's what makes this a loop, not a dashboard a human reads. A risk
tier maps to a concrete action via ACTION_MAP, and every call to
decide_and_act() writes a Score row and a Decision row regardless of
tier: "safe" still gets logged, just with action_taken="none" and no
side effects. The decisions table IS the audit log the README promises.

That guarantee holds even when an ACT-phase side effect fails: Slack
being briefly unreachable, a file-write error, or FortyGuard's API
rejecting a report request must not erase the Score row already written
for what may be the single most dangerous heat event of the run. Each
side effect runs through _try_act(), which records the failure and
returns None rather than letting the exception propagate and roll back
the transaction (agent/models.py's db_session rolls back on any
exception) -- the Decision row still gets written, just with
notified_channel/report_url reflecting what actually happened.

Per Part D's Phase 6 prompt and the C.5 test contract (both explicit and
in agreement -- Part B.3's higher-level narrative summary just doesn't
spell out every side effect), the compliance report fires on *both*
high and extreme tiers, not extreme alone -- for a *live* reading.

A forecast (+12h) reading uses a deliberately different, smaller action
set (FORECAST_ACTION_MAP) than a live one at the same tier: a merely
*predicted* "extreme" hasn't happened yet, so it must never fire the
same real halt-work order, emergency-worded Slack alert, and
heat_intelligence report request that an active live "extreme" fires
right now -- that would be, in effect, paging someone for an emergency
that may not occur. Part B.2 frames the forecast branch as proactive/
precautionary ("tomorrow's 6am shift is forecast to cross 'high' --
pre-emptively reschedule tonight"), not a live-equivalent response, and
heat_intelligence itself reports on *measured* conditions -- a
future-dated request against it isn't valid input regardless of tier.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from agent._shared import now_naive_utc
from agent.act import compliance_report, notify, schedule
from agent.fortyguard_client import FortyGuardClient
from agent.models import Crew, Decision, Reading, Score, Site

logger = logging.getLogger(__name__)

ACTION_MAP: dict[str, str] = {
    "safe": "none",
    "caution": "break_reminders",
    "high": "shorten_shift_and_notify",
    "extreme": "halt_and_notify_and_report",
}

# A forecast at "caution" isn't urgent enough to pre-schedule around --
# only a forecasted high/extreme (Part B.2's own "high" example) triggers
# a proactive reschedule flag + heads-up notification. Neither ever halts
# work or requests a compliance report; see the module docstring.
FORECAST_ACTION_MAP: dict[str, str] = {
    "safe": "none",
    "caution": "none",
    "high": "flag_for_reschedule_and_notify",
    "extreme": "flag_for_reschedule_and_notify",
}


def _try_act(func: Callable, *args: Any, **kwargs: Any) -> Any:
    """Run an ACT-phase side effect without letting its failure erase the
    audit trail. Logs and returns None on any exception.
    """
    try:
        return func(*args, **kwargs)
    except Exception:
        logger.exception("Action %s failed", getattr(func, "__name__", func))
        return None


def decide_and_act(
    session,
    client: FortyGuardClient,
    site: Site,
    crew: Crew,
    reading: Reading,
    score: float,
    tier: str,
    trigger_type: str,
) -> Decision:
    """Classify -> act -> log, for one crew's reading against one site."""
    is_forecast = trigger_type == "forecast"
    action_map = FORECAST_ACTION_MAP if is_forecast else ACTION_MAP
    if tier not in action_map:
        raise ValueError(
            f"Unknown risk tier {tier!r} for site_id={site.id} crew_id={crew.id} "
            f"reading_id={reading.id} -- expected one of {tuple(action_map)}."
        )
    action = action_map[tier]

    score_row = Score(
        reading_id=reading.id,
        crew_id=crew.id,
        final_score=score,
        risk_tier=tier,
    )
    session.add(score_row)
    session.flush()

    notified_channel = None
    report_url = None

    if is_forecast:
        if tier in ("high", "extreme"):
            notified_channel = _try_act(
                notify.notify_slack,
                f"[FORECAST +12h] {site.name}: crew {crew.id} predicted to reach "
                f"'{tier}' within 12h -> {action}",
            )
            _try_act(schedule.write_shift_override, site.id, crew.id, "flag_for_reschedule")
        # No compliance report for a forecast -- see module docstring.
    elif tier == "caution":
        _try_act(schedule.write_shift_override, site.id, crew.id, "break_reminders")
    elif tier in ("high", "extreme"):
        # Report generated before the Slack message is composed (not
        # after, as in Phase 6's original ordering) so the message can
        # say whether one is actually available -- the PDF only ever
        # lands on this server's own local disk (see module docstring:
        # FortyGuard's signed download link is single-use), and Slack's
        # incoming webhook can't attach files at all, so a plain "-> the
        # dashboard" pointer is the most Slack can usefully say about it.
        report_url = _try_act(
            compliance_report.generate_compliance_report, client, site, reading, tier
        )
        message = f"[{tier.upper()}] {site.name}: crew {crew.id} -> {action}"
        if report_url:
            message += " (compliance report available on the dashboard)"
        notified_channel = _try_act(notify.notify_slack, message)
        _try_act(
            schedule.write_shift_override,
            site.id,
            crew.id,
            "shorten_shift" if tier == "high" else "halt_work",
        )

    decision = Decision(
        score_id=score_row.id,
        action_taken=action,
        executed_at=now_naive_utc(),
        report_url=report_url,
        notified_channel=notified_channel,
        trigger_type=trigger_type,
    )
    session.add(decision)
    session.flush()
    return decision
