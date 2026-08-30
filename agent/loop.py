"""Scheduler + integration loop: the top-level sense -> score -> decide ->
act cycle, run either once (``python -m agent.loop --once``, for a
manual/CI-verifiable single pass) or continuously via APScheduler.

Checks ``fetch_api_key_usage`` at the start of every cycle (and, for the
continuous scheduler, every N cycles thereafter -- the usage check itself
isn't free to poll on every pass) and throttles by skipping onboarding --
the most expensive, least time-sensitive calls (satellite/streetview are
Premium, and a site's shade doesn't change hour to hour) -- once credits
fall under a configurable floor. The live and forecast sense/decide
branches always run regardless of credit level; they're the actual point
of the loop.
"""

from __future__ import annotations

import argparse
import logging
import threading
import time as time_module
from datetime import timedelta

from agent._shared import now_naive_utc
from agent.decide import decide_and_act
from agent.fortyguard_client import FortyGuardClient
from agent.models import Crew, Decision, Reading, SchedulerLock, Site, db_session
from agent.onboarding import onboard_site
from agent.score import (
    classify_risk_tier,
    compute_exposure_modifier,
    compute_final_score,
    compute_raw_stress,
)
from agent.sense import celsius_to_fahrenheit, sense_forecast, sense_live

logger = logging.getLogger(__name__)

DEFAULT_CREDIT_FLOOR = 50_000
DEFAULT_CHECK_CREDITS_EVERY_N_CYCLES = 10
# Live-verified real cost: a single cycle (2 sites x live+forecast, each
# needing a day-level create_heatmap call plus environmental_parameters
# retries) has been observed spending on the order of 10-15% of a
# 2,000,000-credit key. Onboarding's own credit_floor throttle doesn't
# help here -- sensing is never throttled by design (see run_cycle's own
# docstring) -- so the interval itself is the only real lever to keep a
# demo-scale deployment from exhausting its credits within a day of
# continuous operation. 6 hours still demonstrates the scheduler is
# alive and updating on its own; reboot the app any time for an
# immediate fresh cycle on demand (e.g. right before a judge looks).
DEFAULT_INTERVAL_MINUTES = 360

# Real production bug, live-observed: a single site failed far more times
# in an ~11-minute window than a true 6-hour-interval scheduler could
# produce, given each failure took close to the full sense timeout --
# meaning more than one scheduler was running concurrently against the
# same credits. dashboard/app.py's in-memory "started once per process"
# flag can't prevent this: it only knows about its own process, not an
# orphaned thread left behind by an earlier reboot that Streamlit Cloud
# didn't fully kill. SchedulerLock (agent/models.py) is a singleton row
# in the one thing genuinely shared across however many of "these"
# processes exist -- the database file itself. The heartbeat is a
# separate, much more frequent job than the sensing cycle itself so a
# truly dead orphan's lock is detected and reclaimed on the order of
# minutes, not left stuck until the next 6-hour cycle would have renewed
# it (or worse, permanently, if a claim were a one-time affair).
_HEARTBEAT_INTERVAL_MINUTES = 5
_LOCK_STALE_AFTER = timedelta(minutes=_HEARTBEAT_INTERVAL_MINUTES * 4)
_SCHEDULER_LOCK_ID = 1

# Real bug, live-observed in production: multiple near-simultaneous
# Streamlit script reruns from the same fresh boot each called
# _claim_scheduler_lock concurrently. Unlike agent.seed's demo-site
# insert, this had no module-level lock serializing the check-then-
# insert/update region at all -- two callers could both see the lock row
# as absent (or both see the same stale row) and both write, occasionally
# surfacing as a raw sqlite3.OperationalError: database is locked instead
# of being cleanly serialized, or (for the steal-a-stale-lock path, which
# has no IntegrityError safety net at all since it's an UPDATE) both
# believing they'd won and starting a second competing scheduler -- the
# exact failure mode this whole lock exists to prevent.
_scheduler_lock_claim_lock = threading.Lock()


def _check_credits(client: FortyGuardClient, credit_floor: int) -> bool:
    """True if credits are healthy enough to run onboarding this cycle.

    A usage-check failure, an unrecognized response, or a non-numeric
    ``remaining_credits`` value doesn't throttle -- an unrelated billing-
    endpoint hiccup shouldn't stop the agent from sensing real heat risk.
    The comparison is inside the same try/except as the call itself so a
    malformed value can't raise past this function uncaught.
    """
    try:
        usage = client.fetch_api_key_usage()
        remaining = usage.get("remaining_credits")
        if remaining is None:
            return True
        return remaining >= credit_floor
    except Exception:
        logger.exception("fetch_api_key_usage failed or returned an unusable response; not throttling on it")
        return True


def run_cycle(session, client: FortyGuardClient, *, skip_onboarding: bool = False) -> list[Decision]:
    """One full pass: for every site with at least one crew, sense live +
    forecast, then score and decide for every crew at that site.

    Each SAVEPOINT-scoped unit of work below is followed by an explicit
    session.commit() rather than deferring every commit to db_session()'s
    single one at the very end. Real production bug, live-observed:
    'sqlite3.OperationalError: database is locked' reappeared even with
    WAL mode + a busy_timeout already in place, once a single
    transaction spanned a site's *entire* live+forecast branches --
    including their slow, sometimes-300s-timeout API calls. SQLite's
    write lock, once taken by a transaction's first write, is held for
    that whole transaction's remaining lifetime, even through the long
    stretches where nothing is actually being written -- long enough for
    another writer (the scheduler's own heartbeat job, agent/loop.py's
    run_scheduler; a concurrent orphaned scheduler) to exceed even a
    generous busy_timeout. Committing after each small, fast write keeps
    the lock held for milliseconds at a time instead of minutes, and (as
    a bonus) makes every site's/crew's work durable immediately rather
    than only at the very end of a potentially very long cycle.
    """
    decisions: list[Decision] = []

    for site in session.query(Site).all():
        crews = session.query(Crew).filter_by(site_id=site.id).all()
        if not crews:
            continue

        # Onboarding failing (e.g. a real server-side segmentation
        # failure -- confirmed live during Phase 7's integration
        # verification) shouldn't block sensing for this site:
        # compute_exposure_modifier already treats a missing
        # shade_coverage_pct as the conservative 0%-shade case, not an
        # error, so there's a real, correct fallback to fall back to.
        if not skip_onboarding:
            try:
                # A flush failure leaves the SQLAlchemy session in an
                # INACTIVE state (per Session._flush()'s own exception
                # handling) requiring an explicit rollback before the
                # session is usable again -- a SAVEPOINT (begin_nested)
                # scopes that rollback to just this site's onboarding
                # attempt, not any other site's already-committed work.
                with session.begin_nested():
                    onboard_site(client, site)
                    session.flush()
            except Exception:
                logger.exception(
                    "Onboarding failed for site_id=%s (%s); proceeding without shade data.",
                    site.id,
                    site.name,
                )
            session.commit()

        # A failure sensing/scoring THIS site (a bad heatmap, a rejected
        # AOI, etc.) must not crash the whole cycle and skip every other
        # site -- log it and move on. _run_site_cycle handles its own
        # SAVEPOINTs/commits per unit of work internally; this rollback
        # is just a defensive backstop for anything that escapes all of
        # those (a bug, not the expected path) so the session is still
        # usable for the next site.
        try:
            decisions.extend(_run_site_cycle(session, client, site, crews))
        except Exception:
            logger.exception(
                "Sense/score/decide failed for site_id=%s (%s); skipping this site this cycle.",
                site.id,
                site.name,
            )
            session.rollback()

    return decisions


def _run_site_cycle(session, client: FortyGuardClient, site: Site, crews: list[Crew]) -> list[Decision]:
    decisions: list[Decision] = []

    # Live (reactive) branch. Its own SAVEPOINT + commit, separate from
    # the per-crew loop below: a flush failure here (a bad sense_live
    # response) needs the same INACTIVE-session recovery as onboarding's,
    # and committing immediately after releases the write lock before
    # the crew loop's own (Slack/PDF) side effects and well before the
    # forecast branch's own slow sensing call.
    try:
        with session.begin_nested():
            raw_reading = sense_live(client, site)
            reading = Reading(
                site_id=site.id,
                ts=raw_reading.ts,
                heat_index=raw_reading.heat_index,
                aqi=raw_reading.aqi,
                humidity=raw_reading.humidity,
                solar_irradiance=raw_reading.solar_irradiance,
                is_forecast=False,
                heatmap_geojson=raw_reading.heatmap_geojson,
            )
            session.add(reading)
            session.flush()
    except Exception:
        logger.exception(
            "Live sensing failed for site_id=%s (%s); this site is skipped this cycle.",
            site.id,
            site.name,
        )
        session.commit()
        return decisions
    session.commit()

    # Each crew gets its own SAVEPOINT + try/except: code-review-caught,
    # empirically reproduced regression -- with only the outer per-site
    # SAVEPOINT (in run_cycle) covering this whole loop, one crew's
    # decide_and_act failure rolled back *every* crew's already-flushed
    # Score/Decision rows for this reading, not just its own. That broke
    # decide.py's own documented guarantee that a real side effect
    # already sent (a Slack alert, a compliance report) always has a
    # corresponding Decision row -- an earlier crew's real alert would
    # end up matched to no row at all. Same SAVEPOINT reasoning as
    # run_cycle's per-site isolation, just scoped one level deeper.
    for crew in crews:
        try:
            with session.begin_nested():
                raw_stress = compute_raw_stress(
                    reading.heat_index, reading.humidity, reading.solar_irradiance
                )
                modifier = compute_exposure_modifier(site.shade_coverage_pct, crew.work_intensity)
                final_score = compute_final_score(raw_stress, modifier)
                tier = classify_risk_tier(final_score)
                decision = decide_and_act(
                    session, client, site, crew, reading, final_score, tier, "live"
                )
            decisions.append(decision)
        except Exception:
            logger.exception(
                "Live decide/act failed for crew_id=%s at site_id=%s (%s); other crews at "
                "this site still processed.",
                crew.id,
                site.id,
                site.name,
            )
        session.commit()

    # Forecast (proactive) branch -- heatmap only, no env_params call
    # (Rule 2 / Phase 5), so there's no humidity/solar signal to run
    # through compute_raw_stress/compute_exposure_modifier here. The
    # forecasted peak temperature is classified directly against the
    # same thresholds (Part B.2: the forecast flag "doesn't feed the live
    # score directly -- feeds a separate proactive branch").
    #
    # Isolated in its own try/except + SAVEPOINT: a forecast-sensing
    # failure (a rejected AOI, a server-side task failure) must not
    # discard the live branch's already-flushed rows above -- they're
    # already committed by this point (see above), so this SAVEPOINT is
    # really just for the INACTIVE-session recovery a flush failure
    # needs, same as the live branch's own.
    forecast_reading = None
    forecast_temp_f = None
    forecast_tier = None
    try:
        with session.begin_nested():
            forecast_signal = sense_forecast(client, site)
            forecast_temp_f = celsius_to_fahrenheit(forecast_signal.max_temp_c)
            forecast_reading = Reading(
                site_id=site.id,
                ts=forecast_signal.ts,
                heat_index=forecast_temp_f,
                aqi=None,
                # No env_params call for a forecast -- these aren't real
                # measurements. NULL (not a 0.0 sentinel, now that the
                # columns are nullable -- Phase 9) plus is_forecast=True
                # together say so unambiguously.
                humidity=None,
                solar_irradiance=None,
                is_forecast=True,
                heatmap_geojson=forecast_signal.heatmap_geojson,
            )
            session.add(forecast_reading)
            session.flush()
            forecast_tier = classify_risk_tier(forecast_temp_f)
    except Exception:
        logger.exception(
            "Forecast sensing/scoring failed for site_id=%s (%s); live-branch decisions for "
            "this site are still returned.",
            site.id,
            site.name,
        )
    session.commit()

    # Same per-crew isolation as the live branch above, and for the same
    # reason -- a SAVEPOINT around the whole crew loop (the original
    # shape here) let one crew's failure erase an earlier crew's
    # already-flushed forecast Decision too.
    if forecast_reading is not None:
        for crew in crews:
            try:
                with session.begin_nested():
                    decision = decide_and_act(
                        session,
                        client,
                        site,
                        crew,
                        forecast_reading,
                        forecast_temp_f,
                        forecast_tier,
                        "forecast",
                    )
                decisions.append(decision)
            except Exception:
                logger.exception(
                    "Forecast decide/act failed for crew_id=%s at site_id=%s (%s); other "
                    "crews at this site still processed.",
                    crew.id,
                    site.id,
                    site.name,
                )
            session.commit()

    return decisions


def run_once(
    *,
    client: FortyGuardClient | None = None,
    database_url: str | None = None,
    credit_floor: int = DEFAULT_CREDIT_FLOOR,
) -> list[Decision]:
    """Run exactly one sense -> score -> decide -> act cycle over every
    seeded site/crew. Checks credits at the start of this cycle.
    """
    client = client or FortyGuardClient()
    with db_session(database_url) as session:
        credits_ok = _check_credits(client, credit_floor)
        if not credits_ok:
            logger.warning(
                "Credits under floor (%s); skipping onboarding this cycle.", credit_floor
            )
        return run_cycle(session, client, skip_onboarding=not credits_ok)


def _claim_scheduler_lock(session) -> bool:
    """True if this caller now holds the singleton SchedulerLock row --
    either by creating it (no one held it yet) or by stealing it from a
    stale holder (heartbeat older than _LOCK_STALE_AFTER, i.e. presumed
    dead). False if a live holder already exists.

    Race-safe the same way agent.seed's demo-site insert is: the
    fixed-id insert relies on the primary key itself as the uniqueness
    guard, and a concurrent loser catches the IntegrityError rather than
    corrupting the row. _scheduler_lock_claim_lock additionally
    serializes same-process callers (see its own comment) and this
    function commits before returning, so a claim is fully durable and
    visible to any other caller before the lock is released -- mirroring
    the fix applied to agent.seed.seed_demo_sites_if_empty for the exact
    same class of bug.
    """
    from sqlalchemy.exc import IntegrityError

    with _scheduler_lock_claim_lock:
        now = now_naive_utc()
        existing = session.get(SchedulerLock, _SCHEDULER_LOCK_ID)
        if existing is None:
            try:
                with session.begin_nested():
                    session.add(
                        SchedulerLock(id=_SCHEDULER_LOCK_ID, started_at=now, heartbeat_at=now)
                    )
                    session.flush()
                session.commit()
            except IntegrityError:
                return False
            return True

        if now - existing.heartbeat_at < _LOCK_STALE_AFTER:
            return False

        existing.started_at = now
        existing.heartbeat_at = now
        session.flush()
        session.commit()
        return True


def _renew_scheduler_lock(session) -> None:
    """Refresh this holder's heartbeat so a healthy scheduler is never
    mistaken for a dead orphan and reclaimed out from under it.
    """
    existing = session.get(SchedulerLock, _SCHEDULER_LOCK_ID)
    if existing is not None:
        existing.heartbeat_at = now_naive_utc()


def run_scheduler(
    *,
    interval_minutes: int = DEFAULT_INTERVAL_MINUTES,
    credit_floor: int = DEFAULT_CREDIT_FLOOR,
    check_credits_every_n_cycles: int = DEFAULT_CHECK_CREDITS_EVERY_N_CYCLES,
):
    """Start a background APScheduler job running run_cycle() on a fixed
    interval, checking credits at startup and every N cycles thereafter.

    Claims SchedulerLock first (see its docstring and _claim_scheduler_lock
    above) -- if another live process already holds it, this returns None
    without starting anything, rather than running a second scheduler
    concurrently against the same credits.
    """
    if check_credits_every_n_cycles < 1:
        raise ValueError(
            f"check_credits_every_n_cycles must be >= 1, got {check_credits_every_n_cycles!r}"
        )

    with db_session() as session:
        claimed = _claim_scheduler_lock(session)
    if not claimed:
        logger.warning(
            "SchedulerLock already held by a live process; not starting a competing "
            "scheduler here. If the real holder is actually dead, this will be "
            "reclaimed automatically once its heartbeat goes stale."
        )
        return None

    from apscheduler.schedulers.background import BackgroundScheduler

    client = FortyGuardClient()
    state = {"cycle_count": 0, "credits_ok": True}

    def _job() -> None:
        state["cycle_count"] += 1
        if state["cycle_count"] == 1 or state["cycle_count"] % check_credits_every_n_cycles == 0:
            state["credits_ok"] = _check_credits(client, credit_floor)
        with db_session() as session:
            run_cycle(session, client, skip_onboarding=not state["credits_ok"])

    def _heartbeat_job() -> None:
        with db_session() as session:
            _renew_scheduler_lock(session)

    # timezone="UTC": BackgroundScheduler() with no explicit timezone
    # defaults to the host's local tz (apscheduler/schedulers/base.py),
    # which would localize the naive-UTC now_naive_utc() below as if it
    # were already local time -- "run immediately at startup" silently
    # becomes "run several hours late" on any host west of UTC.
    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(_job, "interval", minutes=interval_minutes, next_run_time=now_naive_utc())
    scheduler.add_job(_heartbeat_job, "interval", minutes=_HEARTBEAT_INTERVAL_MINUTES)
    scheduler.start()
    return scheduler


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="HeatShield Agent sense -> score -> decide -> act loop"
    )
    parser.add_argument("--once", action="store_true", help="Run a single cycle and exit.")
    args = parser.parse_args()

    if args.once:
        decisions = run_once()
        print(f"Ran one cycle: {len(decisions)} decision(s) written.")
        return

    scheduler = run_scheduler()
    if scheduler is None:
        print("Another process already holds the scheduler lock; exiting without starting one.")
        return
    try:
        while True:
            time_module.sleep(1)
    except KeyboardInterrupt:
        scheduler.shutdown()


if __name__ == "__main__":
    main()
