# Server v25.1.0-public.1 — 14 October 2026

The application is **v25.1.0-public.1**, matching local build `2026-10-14-coach-api-removed-v25.1.0`. The device API stays **RLCD schema 1**. The Coach API (1.37.0) is **removed**. Installation and the Afternoon provisioning limit in the [October 8 release guide](SERVER_RELEASE_2026_10_08.md) still apply. I am Claude; I have maintained the server since 13 October.

## What the forecasts come from now

| Output | v24.0.4 | v25.1.0 |
|---|---|---|
| +90 arrival PM2.5 | Ridge v4 | Spatial gradient boosting, `spatial_hgb_point_p1_no_weather_v25_20261014` |
| Ride window +90–210 mean, min and max | Ridge v4 | The same model's three ride heads |
| P(ride mean ≤ 70) | Cycling logistic | `spatial_residual_le70_e1_v25_20261014` |
| ≥20/≥40 change by arrival | Arrival logistic | Unchanged |
| First ≥20 crossing within 90 min | First-crossing HGB v2 | Unchanged |
| Morning / Afternoon sessions | HGB / PatchTST | Unchanged |

[`spatial_point_model.py`](../app/spatial_point_model.py) fits four HistGradientBoosting absolute-error regressors on (target − fresh reference) every Malaysia midnight. It uses a 28-day lookback, the 330-minute embargo, labels completed before the cutoff and 15-minute origins. Its 81 inputs are the arrival logistic's 28 sensor columns, 4 mean-reversion controls, 5 time columns and 44 neighbour columns. The neighbour columns come from six public AirGradient stations around Kuala Lumpur: per-station gap, 60-minute change and age, network medians, and medians weighted upwind by the issued Open-Meteo 10 m and 925 hPa wind. There are no weather or CAMS concentration inputs.

P(ride mean ≤ 70) takes the model's ride-mean forecast and the empirical distribution of its out-of-fold log residuals from the same 28-day window. The cycling logistic now serves only the `/api/cycling-plan` recheck.

The model code is a byte copy of [`research/models/final/`](../research/models/final/) (`spatial_final_models.py`, `spatial_live_features.py`); a test pins both hashes. Inputs are read as they were visible at the issue: TTDI receipt clocks, the neighbour archive the server already collects every 300 s, and the issued wind. A missing fresh reference, fitted model, neighbour archive, wind or every neighbour stale makes every head unavailable, with a reason. One offline station is a missing input, as in training. Outputs are published literally. On the private installation, training rows and live predictions matched the research model exactly on 9 and 13 October.

A new installation needs at least 240 completed 15-minute training origins with neighbour data (about three days of collection) before these heads publish.

## Holdout evidence

These are development results from the [research harness](../research/models/RESULTS.md), on the 8-day holdout of 6–13 October; lower is better.

| Target | Persistence | Ridge v4 / cycling logistic | v25 |
|---|---:|---:|---:|
| +90 MAE, µg/m³ (n = 2,248) | 10.64 | 11.08 | **9.55** |
| Ride-mean MAE (n = 2,199) | 14.09 | 14.06 | **12.88** |
| Ride-min MAE | 15.78 | 15.54 | **14.30** |
| Ride-max MAE | 15.70 | 13.79 | **12.75** |
| P(ride mean ≤ 70) Brier (n = 2,194) | 0.0825 | 0.0953 | **0.0540** |

At the clocks the old models actually issued on 8–13 October, the +90 MAE was 8.64 against 10.15 for the issued Ridge v4 and 9.65 for persistence.

The limits are in [RESULTS.md](../research/models/RESULTS.md#honest-limits) and the [model card](MODEL_CARD.md). In short:

- The holdout is small and its issues overlap.
- I had read results that included holdout days before writing the variant list.
- The pre-registered rule picked a plainer P(≤70) variant; I adopted E1 because only it beat every single candidate on the holdout, so its margin is not independent confirmation.
- The point model is a median regressor that seldom calls a ≥20 change. Advance warning still comes only from the unchanged arrival logistic.

## Other changes since v24.0.4

- **Coach API removed.** The owner retired the coaching application that consumed `/api/v1/mtb/environment-evidence`. Its route, JSON Schema route, guide (`docs/COACH_API.md`) and the code that only built that payload are gone. `/api/openapi.json` now describes only `/api/cycling-plan`. The web page, `/api/analysis`, `/api/rlcd/v1` and the issued records are unchanged.
- **SQLite WAL.** `init_db()` sets `journal_mode=WAL`, so a read snapshot can no longer block the writer and lose an archived publication ("database is locked").
- **Bounded ledger reads.** The arrival logistic seeds its duplicate check from the last 2 MB of its issued ledger instead of the whole file. Event validation no longer refuses large ledgers.
- **Analysis fix.** A rain washout confirming during a dry forming signal no longer raises a `TypeError` in the analysis cycle; the missing humidity rise is reported as unavailable.
- **Log noise.** Two known-benign library warnings (threadpoolctl's OpenMP notice and scikit-learn's `multi_class` FutureWarning) are filtered at start-up.

## Verification

From the repository root, `python -m unittest discover -s tests` ran **567 tests: 565 passed, 2 skipped** (the research-parity tests, which need the private snapshot). It took about 55 s on Python 3.11.9 with the pinned requirements and Node.js 24. The suite now ports 55 of the server's test modules and keeps 3 public-only modules. Five server test modules stay private because they need unpublished assets; the [source manifest](research/2026-10-14-source-release.json) lists them with reasons. These checks show the software does what it says. They do not show forecast skill.

## Packaging

The public copy differs from the private installation only as before:

- loopback by default, with `BUKIT_KIARA_HOST`, `BUKIT_KIARA_PORT`, `BUKIT_KIARA_DATA_DIR`, `BUKIT_KIARA_LOCATION_ID`, `BUKIT_KIARA_LATITUDE` and `BUKIT_KIARA_LONGITUDE`;
- LAN discovery only with `BUKIT_KIARA_DISCOVERY=1`;
- the `-public.1` build suffix;
- no owner-specific laptop-sleep gap annotation;
- tests that import from `app/`.

All other modules are byte copies. `app/` holds the 61-module runtime closure, 7 older modules that the ported tests import, and the 27 research source files the server loads at runtime. Fitted models, the database, the neighbour archive, issued records, freeze and policy files of the retired research protocol, and the PatchTST prefix and weights stay private. The [manifest](research/2026-10-14-source-release.json) lists every file with its SHA-256.

The repeatable publishing script lives in the private workspace. It re-derives each public file from the committed private source, runs a privacy scan and the tests, and never pushes. Original application, test and research code use the scoped [Apache-2.0 license](../app/LICENSE.md).
