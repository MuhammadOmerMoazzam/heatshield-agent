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
high and extreme tiers, not extreme alone.
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
    if tier not in ACTION_MAP:
        raise ValueError(
            f"Unknown risk tier {tier!r} for site_id={site.id} crew_id={crew.id} "
            f"reading_id={reading.id} -- expected one of {tuple(ACTION_MAP)}."
        )
    action = ACTION_MAP[tier]

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

    if tier == "caution":
        _try_act(schedule.write_shift_override, site.id, crew.id, "break_reminders")
    elif tier in ("high", "extreme"):
        notified_channel = _try_act(
            notify.notify_slack, f"[{tier.upper()}] {site.name}: crew {crew.id} -> {action}"
        )
        _try_act(
            schedule.write_shift_override,
            site.id,
            crew.id,
            "shorten_shift" if tier == "high" else "halt_work",
        )
        report_url = _try_act(
            compliance_report.generate_compliance_report, client, site, reading, tier
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
