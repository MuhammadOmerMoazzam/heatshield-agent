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
import time as time_module

from agent._shared import now_naive_utc
from agent.decide import decide_and_act
from agent.fortyguard_client import FortyGuardClient
from agent.models import Crew, Decision, Reading, Site, db_session
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
                # session is usable again -- but a plain session.rollback()
                # would undo *every* site processed so far in this cycle,
                # since nothing commits until db_session()'s single commit
                # at the very end. A SAVEPOINT (begin_nested) scopes the
                # rollback to just this site's onboarding attempt.
                with session.begin_nested():
                    onboard_site(client, site)
                    session.flush()
            except Exception:
                logger.exception(
                    "Onboarding failed for site_id=%s (%s); proceeding without shade data.",
                    site.id,
                    site.name,
                )

        # A failure sensing/scoring THIS site (a bad heatmap, a rejected
        # AOI, etc.) must not crash the whole cycle and skip every other
        # site -- log it and move on. Same SAVEPOINT reasoning as above.
        try:
            with session.begin_nested():
                decisions.extend(_run_site_cycle(session, client, site, crews))
        except Exception:
            logger.exception(
                "Sense/score/decide failed for site_id=%s (%s); skipping this site this cycle.",
                site.id,
                site.name,
            )

    return decisions


def _run_site_cycle(session, client: FortyGuardClient, site: Site, crews: list[Crew]) -> list[Decision]:
    decisions: list[Decision] = []

    # Live (reactive) branch.
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

    for crew in crews:
        raw_stress = compute_raw_stress(
            reading.heat_index, reading.humidity, reading.solar_irradiance
        )
        modifier = compute_exposure_modifier(site.shade_coverage_pct, crew.work_intensity)
        final_score = compute_final_score(raw_stress, modifier)
        tier = classify_risk_tier(final_score)
        decisions.append(
            decide_and_act(session, client, site, crew, reading, final_score, tier, "live")
        )

    # Forecast (proactive) branch -- heatmap only, no env_params call
    # (Rule 2 / Phase 5), so there's no humidity/solar signal to run
    # through compute_raw_stress/compute_exposure_modifier here. The
    # forecasted peak temperature is classified directly against the
    # same thresholds (Part B.2: the forecast flag "doesn't feed the live
    # score directly -- feeds a separate proactive branch").
    #
    # Isolated in its own try/except + SAVEPOINT: a forecast-sensing
    # failure (a rejected AOI, a server-side task failure) must not
    # discard the live branch's already-flushed rows above -- a plain
    # session.rollback() here would undo the live branch's work too,
    # since both branches share this same uncommitted transaction; a
    # SAVEPOINT scopes the rollback to just the forecast branch.
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
                # measurements. is_forecast=True is the flag that says so.
                humidity=0.0,
                solar_irradiance=0.0,
                is_forecast=True,
                heatmap_geojson=forecast_signal.heatmap_geojson,
            )
            session.add(forecast_reading)
            session.flush()

            forecast_tier = classify_risk_tier(forecast_temp_f)
            for crew in crews:
                decisions.append(
                    decide_and_act(
                        session,
                        client,
                        site,
                        crew,
                        forecast_reading,
                        forecast_temp_f,
                        forecast_tier,
                        "forecast",
                    )
                )
    except Exception:
        logger.exception(
            "Forecast sensing/scoring failed for site_id=%s (%s); live-branch decisions for "
            "this site are still returned.",
            site.id,
            site.name,
        )

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


def run_scheduler(
    *,
    interval_minutes: int = 60,
    credit_floor: int = DEFAULT_CREDIT_FLOOR,
    check_credits_every_n_cycles: int = DEFAULT_CHECK_CREDITS_EVERY_N_CYCLES,
):
    """Start a background APScheduler job running run_cycle() on a fixed
    interval, checking credits at startup and every N cycles thereafter.
    """
    if check_credits_every_n_cycles < 1:
        raise ValueError(
            f"check_credits_every_n_cycles must be >= 1, got {check_credits_every_n_cycles!r}"
        )

    from apscheduler.schedulers.background import BackgroundScheduler

    client = FortyGuardClient()
    state = {"cycle_count": 0, "credits_ok": True}

    def _job() -> None:
        state["cycle_count"] += 1
        if state["cycle_count"] == 1 or state["cycle_count"] % check_credits_every_n_cycles == 0:
            state["credits_ok"] = _check_credits(client, credit_floor)
        with db_session() as session:
            run_cycle(session, client, skip_onboarding=not state["credits_ok"])

    # timezone="UTC": BackgroundScheduler() with no explicit timezone
    # defaults to the host's local tz (apscheduler/schedulers/base.py),
    # which would localize the naive-UTC now_naive_utc() below as if it
    # were already local time -- "run immediately at startup" silently
    # becomes "run several hours late" on any host west of UTC.
    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(_job, "interval", minutes=interval_minutes, next_run_time=now_naive_utc())
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
    try:
        while True:
            time_module.sleep(1)
    except KeyboardInterrupt:
        scheduler.shutdown()


if __name__ == "__main__":
    main()
