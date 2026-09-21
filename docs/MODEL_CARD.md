# Forecast model card

**Public snapshot:** v18.3-public.1, 21 September 2026. **Coach contract:** 1.24.0.

## Purpose and scope

I develop this service to describe observed and estimated local PM2.5 around TTDI/Bukit Kiara. It is an experimental environmental information service, not a personal exposure measurement, medical decision tool or cycling coach. Its geographical scope is one public TTDI sensor and the surrounding model-grid context. A forecast at that location is not a measurement of every trail section.

| Output | Exact meaning |
|---|---|
| Current | Most recent sensor concentration; observation age matters |
| +90 minutes | Arrival-time estimate; replay scoring uses a 15-minute bucket-median proxy with target-clock alignment |
| +90–210 minutes | Mean concentration across a two-hour future interval; uncertainty is for that mean, not in-window min/max |
| Morning / afternoon | Explicitly dated two-hour sessions within broad 09:00–13:00 / 14:00–18:00 windows |

Issue time, retrieval time, observation time, model-source time and target time are different clocks. They must not be substituted for one another. The code uses Malaysia time for session selection and stores absolute timestamps for matching.

## Inputs and competing models

- Local AirGradient PM2.5/PM10, temperature and humidity; derived recent references, momentum, variability and particle mix.
- Archived Open-Meteo weather values available at each issue, including rain timing/context and wind.
- CAMS regional concentrations accessed through Open-Meteo, with local-correction experiments.
- Subang observations as a spatially distinct weather reference, not a forecast or proof of rain at TTDI.

Near-term routing compares learned/analogue alternatives with persistence and retains separate eligibility checks for arrival, mean and peak. When an alternative fails those checks, a recent-sensor reference can remain the main point forecast. That is an explicit fallback, not evidence that the future will be steady.

The ordinary session estimator is an adaptive local ensemble. Its historical expert scores incorporate each issue's own recent-minus-closed sensor adjustment. The final nonnegative floor is applied after mixing components and the current reference adjustment. This aligns historical scoring with live use; the fixed ablation showed negligible aggregate error changes.

The **experimental early-afternoon override** is `afternoon_direction_hgb_v1`: a rise/steady/fall classifier with a within-class historical median change. It is eligible only at issue times **07:00:00–08:00:00 MYT**, for the same day's **14:00–16:00** target. It uses completed earlier training outcomes and available archived external forecasts. Steady predictions can equal the recent reference. Class scores are uncalibrated, not event probabilities. The experiment was specifically authorized despite a failed latest-block safeguard; it is not universally validated.

Dedicated rain-learning and CAMS/local-correction trials are diagnostics unless the returned provenance explicitly says otherwise. Their presence on a page is not proof they changed the main point. The September review did not deploy new magnitude heads, direct-delta alternatives, a rapid-clearing overlay or a new CAMS slope.

## Evaluation and what it does not establish

The journal's numerical table is an exploratory **07:30→14:00–16:00** comparison on 17 known daily outcomes from September 3–20, excluding September 7's incomplete target. It does not measure the morning model or +90-minute skill. Multiple issue clocks share outcomes and are not independent events.

The retained experiment's primary MAE is 22.216 µg/m³, MAPE 29.700% and WAPE 28.311%; matched recent-sensor persistence is 25.862 µg/m³, 35.249% and 32.957%. It detected 2/7 substantial rises and 5/6 substantial falls, with two wrong-way event calls and two change calls on four broadly steady days. Actual changes use ±10 µg/m³ and forecast calls ±5, relative to the matched issue-time reference. These are test definitions, not health thresholds.

Chronological blocks contain 6, 7, 3 and 1 outcomes. The latest case predicted clearing incorrectly. Known-outcome model development and repeated comparisons prevent interpreting these scores as an untouched holdout or guaranteed future performance. Stored sensor readings also lack a complete original acquisition-vintage record, which limits exact reconstruction of what every historical request knew.

The separate ChatGPT reviewer checked supplied source/evidence and reproduced the primary/block percentage arithmetic. It did not independently refit these models or inspect the deployed server. This is AI-assisted critical review, not human scientific peer review.

## Uncertainty, failures and observability

Empirical q10–q90 spans are **uncalibrated**. For the scoped experimental session model, a model-specific interval has not been established; do not borrow the ordinary model's interval. Missing outputs remain missing rather than being manufactured from an unrelated target.

Known weaknesses include missed rises, false clearing, weather timing/location error, slow reaction or lack of anticipation, short regime coverage, sensor gaps, correlated overlapping targets and source/model changes. Correlation between cooling and falling particles does not establish rain or identify the smoke source.

Selection diagnostics now record gate values, thresholds, margins, unavailable evidence and the final published stage. Mean eligibility does not require a peak estimate. Scheduled model handoffs can move the number without new observations; cached outputs keep their original issue and target times. Historical reconstructions are not relabelled as forecasts actually issued at the time.

## Release policy

New candidates should be compared on identical exact targets and information cutoffs, with persistence retained as a baseline. Report absolute error, percentage error, rise/fall detection, false clearing and chronological/event-level behavior. Separate correctness fixes from predictive-skill claims. Freeze a candidate before collecting new prospective evidence; do not silently broaden an experimental scope.

The public synthetic tests check implementation contracts. They do not reproduce the private historical study or certify forecasting accuracy.
