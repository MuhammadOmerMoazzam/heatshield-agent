# HeatShield Agent

**FortyGuard Hackathon '26 · Track 6 (Agentic)**

An autonomous agent that senses real-time and forecast heat conditions at outdoor worksites, scores each crew's actual exposure risk (not just ambient temperature), and — without a human in the loop — decides and takes a concrete action: nothing, a break reminder, a shortened shift with a Slack alert, or a full work stoppage with a compliance report.

Live demo: **https://heatshield-agent.streamlit.app/**
Slack alerts workspace: **https://join.slack.com/t/heatshield/shared_invite/zt-48bp2635i-NX7zlSiVSIKKySzbwMvxMQ** — join to see live heat-risk alerts as the agent sends them.

---

## Problem

Outdoor crews (construction, warehousing, logistics, event staffing) are exposed to heat risk that a plain ambient-temperature reading understates or overstates. Two crews standing in the same city, at the same hour, face materially different real risk depending on how much shade their specific site actually has and how physically demanding their work is. Heat-related illness is a real, recurring OSHA compliance and safety-officer problem, and the standard response — someone manually checking a weather app and using judgment — doesn't scale, isn't logged, and isn't proactive.

HeatShield Agent automates that judgment call: it ingests real environmental data for a real U.S. site, computes an exposure-adjusted risk score specific to that site's shade and that crew's work intensity, and autonomously takes the action the risk tier calls for — logging every decision as it goes.

## Track

Track 6 (Agentic). The differentiator is the *agentic* loop — autonomous decide → act, not a dashboard a human reads and acts on — combined with a real model layer (the exposure-adjustment scoring in [`agent/score.py`](agent/score.py)) rather than a raw heat-index passthrough.

---

## Architecture

```
                     ┌───────────────────────────────────────────┐
                     │              HeatShield Agent               │
                     │                                             │
   Scheduler ───────▶│  1. SENSE  ──▶ 2. SCORE ──▶ 3. DECIDE ──▶ 4. ACT │
   (APScheduler)      │                                             │
                     └───────────────────────────────────────────┘
                              │                              │
                 ┌────────────┴────────────┐        ┌─────────┴─────────┐
                 │     FortyGuard API        │        │  Action Channels  │
                 │                            │        │  - Slack webhook  │
                 │ live:                      │        │  - Schedule store │
                 │  POST /v1/heatmap          │        │  - PDF report link│
                 │   (today + up to +12h)     │        └───────────────────┘
                 │  POST /v1/env_params       │
                 │   (same ts as today's      │
                 │    heatmap call)           │
                 │                            │
                 │ onboarding (once/site):    │
                 │  POST /v1/satellite        │
                 │  POST /v1/streetview       │
                 │                            │
                 │ on high/extreme event:     │
                 │  POST /v1/heat_intelligence│
                 │                            │
                 │ ops:                       │
                 │  GET /v1/status/{id}       │
                 │  GET /v1/system/           │
                 │      fetch-api-key-usage   │
                 └────────────────────────────┘
                              │
                     ┌────────┴────────┐
                     │  SQLite/Postgres │
                     │  sites/crews/    │
                     │  readings/scores/│
                     │  decisions       │
                     │  (= audit log)   │
                     └──────────────────┘
```

**Sense** (`agent/sense.py`) is deliberately split into two functions, not one branching on a flag: `sense_live` calls `create_heatmap` for "now" then `environmental_parameters` against that *exact same* timestamp (the handbook's own "use a matching date/time" rule for forecast-adjacent endpoints), and `sense_forecast` calls `create_heatmap` for now+12h only, since `env_params` isn't guaranteed to support a forward clock. Both go straight to FortyGuard's day-level aggregate rather than an hourly query — live-verified in production that hourly data lags too far behind to ever have real values, at real credit cost for nothing.

**Score** (`agent/score.py`) is the model layer — see [Score Formula](#score-formula) below.

**Decide + Act** (`agent/decide.py`, `agent/act/`) maps a risk tier to a concrete action with no human approval step, and writes a `Decision` row for every single call regardless of tier — that table is the audit log. A live "extreme" halts work, notifies Slack, and requests a compliance-report PDF; a *forecast* "high"/"extreme" only flags a proactive reschedule and sends a heads-up — it never halts real work or requests a report over something that hasn't happened yet.

**Scheduler** (`agent/loop.py`) runs the full cycle over every seeded site on a fixed interval via APScheduler, checking FortyGuard's own credit balance at startup and periodically thereafter, throttling the (Premium, one-time) onboarding calls if credits run low. Sensing itself is never throttled.

**Deployment** (Part E): Streamlit Cloud runs a single process with no separate always-on worker, so `dashboard/app.py` starts the scheduler itself as a background thread on load, guarded so exactly one scheduler ever runs per live process — a database-backed lock (`SchedulerLock`) additionally guards against Streamlit Cloud not always fully killing an orphaned thread across a reboot. **This is a deliberate, hackathon-appropriate simplification, not a bug**: a "real" production deployment would run the scheduler as its own service, but a single Streamlit process is what the platform's free tier actually offers, and the lock makes it safe.

---

## Data Source Integration

Every FortyGuard endpoint this agent actually calls, and why:

| Endpoint | Called from | Purpose |
|---|---|---|
| `POST /v1/heatmap` | `sense_live`, `sense_forecast` | Day-level temperature aggregate for "now" and "now+12h" — the temperature signal behind both the live score and the proactive forecast branch. |
| `POST /v1/env_params` | `sense_live` | Heat index, humidity, solar irradiance, AQI at the *same* timestamp as the live heatmap call (Rule 2) — the multi-metric input `compute_raw_stress` actually uses. |
| `POST /v1/satellite` | `agent/onboarding.py` | Overhead segmentation, once per site — derives `canopy_pct`. |
| `POST /v1/streetview` | `agent/onboarding.py` | Ground-level segmentation, once per site — derives `shade_coverage_pct`, the key exposure-model input a plain heat-index reading has no way to see. |
| `POST /v1/heat_intelligence` | `agent/act/compliance_report.py` | Auto-generated PDF report, only on a live high/extreme decision. |
| `GET /v1/status/{activity_id}` | `agent/fortyguard_client.py` (internal) | Polls every async submit-then-poll call above to completion, with a 3s→6s→12s backoff. |
| `GET /v1/system/fetch-api-key-usage` | `agent/loop.py` | Checked at scheduler startup and periodically thereafter to throttle onboarding if credits run low. |

The client wrapper (`agent/fortyguard_client.py`) validates AOI size, polygon closure, the +12h forecast window, and US-only coordinates client-side, before any network call — every one of those is a real handbook constraint, tested in `tests/test_fortyguard_client.py`.

---

## Real-World Impact

A real run against the live deployment on 2026-08-30 illustrates exactly why the exposure model matters, not just the live-data plumbing:

> Phoenix, AZ's live heat index measured **96.9°F** — on a plain heat-index lookup, that's only "caution." HeatShield's exposure model factored in this crew's heavy work intensity and the site's real, satellite/street-view-derived shade coverage (near-zero canopy in a warehouse district) and raised the score to **111.4** — crossing into **"high."** The agent autonomously shortened the shift, posted a Slack alert, and generated a FortyGuard compliance PDF, all without a human reviewing the reading first. A plain-heat-index system would have sent a break reminder and moved on; this one caught the real risk that "caution" would have missed.

That gap — 96.9°F read as merely "caution" vs. 111.4 correctly read as "high" — *is* the measurable outcome: real crew exposure caught and acted on that a temperature-only system would not have flagged.

The agent isn't limited to any specific location: `agent/seed.py` currently seeds four real US sites spanning distinct climates and work profiles (desert warehouse, humid port, coastal construction, extreme-heat outdoor event grounds) to demonstrate this, but nothing in the sense/score/decide pipeline is hardcoded to them — `FortyGuardClient`, `sense.py`, `score.py`, and `onboarding.py` all operate purely on whatever `Site.lat`/`Site.lon` is passed in, and the client's own `test_us_only_coordinate_guard` enforces "any US location, not a specific one" as the actual contract.

---

## Score Formula

(Verbatim from `agent/score.py`'s module docstring — the single source of truth for both the code and this section.)

```
final_score = raw_stress(heat_index, humidity, solar_irradiance)
              * exposure_modifier(shade_coverage_pct, work_intensity)

risk_tier = classify_risk_tier(final_score)
```

`raw_stress` starts from the measured heat index (degrees Fahrenheit) and adds stress that NWS/OSHA heat-index tables — which are measured in shade — don't fully capture on their own for an outdoor crew working in direct sun:

- **Humidity above 40%.** Heat-index tables already account for humidity's effect on evaporative cooling up to a point, but sustained humidity above 40% further impairs a working body's ability to cool itself, so a small linear penalty is added for every percentage point above that threshold.
- **Solar irradiance.** Standard heat index is a shaded-thermometer reading; a crew working in direct sun experiences real additional heat load proportional to solar irradiance (W/m²), added as a small linear penalty.

`exposure_modifier` is a multiplier around 1.0 combining two onboarding- and crew-level factors:

- **Shade coverage** (site-level, from satellite/street-view segmentation). More shade reduces the modifier, down to a floor of 0.7 at 100% canopy/building shade. A site onboarded before segmentation finishes has no shade data yet — rather than crash or silently assume full shade (which would understate risk), missing shade data is treated as 0% shade, the conservative, higher-risk assumption.
- **Work intensity** (crew-level). Heavier physical work generates more internal heat, so "heavy" raises the modifier and "light" lowers it relative to "moderate" (also the default for any unrecognized value, so a bad onboarding record degrades gracefully instead of halting the sense → score → decide → act loop).

`final_score` is expressed in the same degrees-Fahrenheit-equivalent units as heat index, so it classifies directly against the OSHA/NIOSH heat-index risk categories (NOAA/NWS heat index chart):

| Tier | final_score (°F HI-equivalent) |
|---|---|
| safe | < 91 |
| caution | 91 to < 103 |
| high | 103 to < 125 |
| extreme | ≥ 125 |

These four boundary values (91 / 103 / 125°F) are the single source of truth for both `agent/score.py` and this README — if they ever need to change, change them in the code first.

---

## Quickstart

**Live demo:** https://heatshield-agent.streamlit.app/ — no setup required.

**Run locally:**

```bash
git clone https://github.com/MuhammadOmerMoazzam/heatshield-agent.git
cd heatshield-agent
python -m venv .venv
.venv/Scripts/activate   # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
cp .env.example .env     # fill in FORTYGUARD_API_KEY (and optionally SLACK_WEBHOOK_URL)
streamlit run dashboard/app.py
```

The dashboard seeds its own demo sites on first load and starts the sense → score → decide → act scheduler as a background thread automatically — no separate service to run. A single manual cycle (useful for CI or quick verification without waiting for the scheduler interval) is also available directly:

```bash
python -m agent.loop --once
```

Run the test suite with:

```bash
pytest
```

### Docker (local dev/testing parity)

```bash
docker build -t heatshield-agent .
docker run -p 8501:8501 --env-file .env heatshield-agent
```

(Streamlit Cloud itself builds directly from `requirements.txt` on every deploy — the Dockerfile is for local parity, not the deployment path itself.)
