# Forecast model card

**Current published source:** v24.0.3-public.1, 9 October 2026. **Coach contract:** 1.37.0.

The [October 8 source release](SERVER_RELEASE_2026_10_08.md) replaces the September 21 application snapshot. Earlier sections below remain dated research/deployment records; their selectors and fallback behavior do not describe the current publisher. Fitted assets, working data and the private Windows service installation are excluded.

## Source release and arrival distribution — later on 8 October 2026

The public source corresponds to local v24.0.3. The October 9 update changes device presentation and history sampling; the fixed forecast learners below are unchanged. Its fixed Ridge v3 learner publishes the +90 concentration and +90–210 mean/minimum/maximum without numerical selection or persistence substitution. A separately fixed logistic classifier, `receipt_visible_logistic_arrival_change20_40_v1`, predicts a five-class change distribution at exact +90 relative to the issue's trailing five-minute reference. Native class masses supply ≥20/≥40 fall and rise tails and the within-20 outcome. The web focuses on these arrival outcomes; the distinct first-sampled-crossing HGB output remains in the APIs and issue archive.

The arrival learner uses 28 scalar sensor/time features, completed outcomes before Malaysia midnight, five-minute origins, a 28-day history and a 120-minute origin embargo. Weather and neighboring measurements are not learned inputs to this classifier. The categorical learner and numeric concentration model are separate estimators, not a single jointly fitted trajectory distribution.

On 3,232 common historical issues across 34 dates, logistic class log loss was **0.365475** versus a training-prior comparator's **0.371524**; class Brier was worse, **0.155885** versus **0.150380**. At an evaluation-only 10% threshold, five of nine inspected large-fall movement episodes had an earlier signal, but only three selected warnings had positive outcomes at their own +90 endpoint; there were **23 false endpoint-warning clusters**. Rare large rises remained poorly detected, and ≥40 tail scores did not beat the prior comparator. The threshold does not gate publication.

The selection followed inspected failures. Dense issues overlap, historical receipts can be unknown, and these retrospective results are not prospective validation or proof of usable preparation warnings. The classifier can assign small physically unsupported fall probabilities at low references; native values are retained and this limitation is exposed in its evidence. Source/software checks do not resolve that modeling limitation. The numerical arrival learner also missed the inspected October 8 fall's magnitude, as recorded in the earlier v23 account below.

Morning retains a fixed HGB learner. Afternoon's guarded PatchTST source is included, but a source-only clone cannot reproduce its privately certified training prefix and freeze chain; it remains unavailable without compatible locally provisioned assets. The primary fresh learners can fit from a new local database after sufficient completed history accumulates. The ride range describes complete 15-minute median extrema, not a guaranteed instantaneous envelope or a confidence interval.

## Local deployment and decision review — 8 October 2026

The checked local service is build `2026-10-08-fresh-sensor-models-v23.0.0`, Coach API **1.36.0**. The owner's primary decision is whether to begin preparing now for a ride starting approximately 90 minutes later. Advance warning of substantial falls and rises, including **20 and 40 µg/m³** changes, is the priority. The +90 concentration supports the arrival judgment; the +90–210 mean and range provide subsequent ride context.

### Fixed outputs and exact targets

| Output | Current local model / target |
|---|---|
| +90 concentration | `fresh_sequence_ridge_delta_numeric_v3_20261008`; interpolation of complete 15-minute sensor medians at the exact target clock |
| +90–210 mean | Same fixed learner; duration-weighted complete 15-minute median proxy over the ride window |
| Ride minimum and maximum | Same learner's two separate heads; smallest/largest complete 15-minute medians overlapping the ride window, not instantaneous extremes or uncertainty bounds |
| First ≥20 event before +90 | `fresh_sensor_hgb_direct_first20_90min_v2`; first rise/drop crossing at +15, +30, +45, +60, +75 or +90 relative to the fresh issue reference, or no crossing |
| Morning / Afternoon | Separate fixed HGB / PatchTST session models; these results do not validate them |
| Ride-average ≤70 probability | Separate retained classifier; not derived from the numerical ride range |

The numerical Ridge pipeline includes its fitted imputer/scaler and estimator-owned target transform. Native `predict()` returns concentrations. The event HGB's native `predict_proba()` returns the three-class distribution. Publication copies those outputs without a performance selector, persistence substitution, hand-coded rebasing, clipping or range sorting. A missing valid output stays unavailable. The numerical and first-event models use **208 causal sensor/time features**, including a trailing five-minute reference and three hours of sensor sequence. They do not use weather forecasts, neighbors, radar or CAMS inputs.

The freshest single Current reading can differ from the declared five-minute reference. The event target is a first crossing anywhere in the sampled interval, not a change sustained at arrival. It has no learned ≥40 head or crossing-time output. Learned arrival-change probabilities for ≥20/≥40 falls and rises are an identified requirement, not a deployed feature. Ride edge medians can include time outside the exact window; the displayed range is neither a confidence interval nor a guaranteed envelope.

### Development comparisons and adverse results

| Metric and comparison scope | Revised fixed model | Comparator |
|---|---:|---:|
| +90 MAE, 522 paired cases on seven dates | 8.6357 µg/m³ | Previous learner 8.1696 |
| Ride-mean MAE, same cases | 10.9397 µg/m³ | Previous learner 10.6953 |
| Multiclass event Brier, 1,093 issues on four dates | 0.362521 | Training class-prior baseline 0.291692 |
| Drop calls, same four-date event cohort, winning class | 8 detected; 58 missed | 64 false drop calls |
| Event Brier, 558 matched actual October 6 issue clocks | 0.753530 | Original issued model 0.846197 |
| Drop recall, same October 6 matched issues | 6.60% | Original issued model 30.19% |

Lower MAE and Brier are better. The numeric cohort covers September 27, September 30, October 1, October 3, October 4, October 6 and completed October 8 morning targets. The event cohort covers September 30, October 1, October 3 and October 6. Dense overlapping issues are correlated and are not counts of independent episodes. The original October 6 comparator was actually issued; the revised side is a reconstruction. These comparisons do **not** establish improved overall accuracy or sufficient advance-warning skill. The broader event result is worse than a constant training-prior baseline; the matched October 6 probability-score gain comes with substantially lower drop recall.

The local repair used training completed before October 8 midnight MYT, with a 330-minute numerical training embargo. Input receipts/revisions are bounded by the issue where retained; older missing receipt history remains unknown. Model artifacts fitted after a historical issue are rejected in ordinary serving. Offline counterfactual reconstruction is explicitly identified. Development followed inspection of failures, so the recorded studies are not untouched confirmation tests.

### Completed October 8 arrival cases

| Original issue, MYT | Fresh reference | Original issued +90 | Completed +90 proxy | Revised Ridge reconstruction |
|---|---:|---:|---:|---:|
| 16:27:59 | 154.4 | 153.4 | 89.5487 | 152.2757 |
| 16:29:01 | 156.8 | 153.2 | 89.0113 | 150.7203 |

All values are µg/m³. Arrival is exactly issue +90; the proxy interpolates the complete **17:45 median 96.3** and **18:00 median 88.5**, each supported by five observations. The actual changes were **−64.8513 and −67.7887**. Revised numerical reconstruction errors remained **62.7270 and 61.7090**, demonstrating a remaining magnitude miss even after the fresh-input repair.

HGB first-drop replays were **61.69% and 72.24%**, versus originally issued **19.99% and 21.76%**. The new artifacts were fitted after these issues and this episode informed model design. These figures are development evidence, not originally issued successes. They cannot supply an arrival-level or ≥40 probability. The arrival result alone does not score the later two-hour ride targets.

### Evaluation required for the preparation decision

I need separately learned arrival falls/rises of ≥20 and ≥40 relative to a declared issue reference, while retaining the distinct first-crossing target if it remains useful. Evaluation should preserve real publication times and score warning lead time before onset, misses, false calls, probability quality, arrival errors on large movements and persistence of improvement to arrival. It should report independent episodes/dates and chronological purged comparisons, followed by untouched future issued evidence. A minority fall probability can still matter to the rider even when no crossing is the largest class.

The current service forecasts the TTDI sensor's outcomes. Correspondence with Bukit Kiara trail conditions has not been established with paired trail ground truth. A preparation plan retaining the original reference and absolute arrival clock during rechecks is also a requirement identified in the review; it is not implemented in this update.

The repair passed 105 selected Python tests, 27 observation-renderer JavaScript checks and 11 momentum-renderer checks. Live verification checked literal output agreement across APIs and the immutable issue archive. These checks validate implementation behavior, not atmospheric forecasting skill. The [October 8 journal](journal/2026-10-08-warning-before-the-ride.md) describes the investigation and the riding objective in my developer voice.

## Local deployment and research — 5 October 2026

I checked the live deployment at build `2026-10-01-nearterm-input-audit-v19.2.0`, with API **1.28.1**. The dashboard and its collectors run inside the owner's Windows service, independently of Codex. The selected +90-minute point and +90–210-minute mean retain `fresh_sensor_persistence_v1`. Both Morning and Afternoon use the owner-selected experimental `weather_session_change_v1`, which learns signed session changes from local and issued weather features. Its selection is an experimental choice, not a claim of established sudden-change skill.

On October 1, I added an hourly archive of **nine CAMS regional model cells**. They belong to one coarse modeled concentration field, rather than nine independent monitoring stations. I keep network fetch time, database recording time and source checksum; a reconstructed issue can use a run only after both availability clocks. The regional gradient and issued-wind transport context are diagnostic. They do not alter the current primary forecast points.

### Paired regional-input candidates

On October 5, I compared fixed Ridge candidates using the same training and test examples, first with existing local/weather inputs and then with regional additions. These are reconstructed candidates on the short new-input cohort; they are not exact replays of the full-history production estimator or originally issued forecasts.

| Target | Existing-input MAE, µg/m³ | Added-regional-input MAE, µg/m³ | Completed test cases |
|---|---:|---:|---|
| Exact +90-minute point | 5.391 | 5.327 | 24, October 4 |
| +90–210-minute mean | 6.124 | 6.004 | 24, October 4 |
| Morning two-hour mean | 1.49 | 3.19 | 2, October 4–5 |
| Afternoon two-hour mean | 2.40 | 4.33 | 1, October 4 |

The near-term cases overlap; fixed separation leaves six per target. I had only three complete regional collection days, October 2–4. The candidate's October 4 advantage over persistence did not hold in the completed October 5 cases: regional Ridge MAE was **3.049 versus persistence 2.847** for +90 minutes (10 cases), and **3.518 versus 3.071** for the mean (8 cases). Regional inputs still improved the paired basic Ridge errors, from 3.285 to 3.049 and from 3.524 to 3.518 respectively. These partial cases were diagnostic and did not select a model.

October 4 contained five observed +90-minute changes and four mean-window changes reaching **10 µg/m³** in magnitude. No learned candidate correctly called any of them with a predicted change reaching that same threshold. There were no changes reaching 20 µg/m³ on that test day. The session candidate windows contained no changes reaching 10 µg/m³. Matched gradient-boosting comparisons supplied no supporting improvement; the near-term boosting fits had too few training rows to split under their fixed minimum leaf size.

### Actual issued session forecasts

I separately scored one genuine archived publication per target date, selected before fixed decision clocks of **07:30 for Morning** and **12:00 for Afternoon**. I scored each publication's exact stored target window against covered sensor outcomes and paired it with persistence at the same issue.

| Target dates | Issued weather-session model MAE, µg/m³ | Paired persistence MAE, µg/m³ | Distinct completed windows |
|---|---:|---:|---:|
| Morning, October 2–5 | 3.72 | 5.77 | 4 |
| Afternoon, October 2–4 | 6.13 | 5.57 | 3 |

The Morning result shows a benefit in four cases; Afternoon was worse in three. All changes in this issued cohort were below 10 µg/m³, so it does not establish skill at anticipating abrupt clearing or pollution rises. The future Afternoon and next-day Morning targets visible during the review were unscored.

### Limits and decision

I froze the candidate policies before preparing outcomes, fitted only targets completed before each Malaysia midnight cutoff, bounded sensor features by archived publication watermarks, and rejected later or stale external inputs. I did not backfill the new regional inputs into September. Available outcomes were retrospective, not an unseen confirmation set; raw sensor rows also lack a complete first-fetch history for later replacements. The exploratory session comparison used less date support than the production model's usual ten-day requirement.

I did not promote a numerical replacement from this review. Weak or rotating surface winds, different directions aloft and coarse regional pollution fields leave substantial uncertainty about local changes. A northerly forecast does not determine the PM2.5 response. I publish the [aggregate research record](research/2026-10-05-regional-input-review.json) alongside the [October 5 journal](journal/2026-10-05-learning-from-the-forecast-misses.md), which records the forecast misses, implementation work and evaluation correction leading to this result.

## Public snapshot record — 21 September 2026

The following sections preserve the model description and evaluation for **v18.3-public.1 / API 1.24.0**, the bundled source.

### Purpose and scope

I develop this service to describe observed and estimated local PM2.5 around TTDI/Bukit Kiara. It is an experimental environmental information service, not a personal exposure measurement, medical decision tool or cycling coach. Its geographical scope is one public TTDI sensor and the surrounding model-grid context. A forecast at that location is not a measurement of every trail section.

| Output | Exact meaning |
|---|---|
| Current | Most recent sensor concentration; observation age matters |
| +90 minutes | Arrival-time estimate; replay scoring uses a 15-minute bucket-median proxy with target-clock alignment |
| +90–210 minutes | Mean concentration across a two-hour future interval; uncertainty is for that mean, not in-window min/max |
| Morning / afternoon | Explicitly dated two-hour sessions within broad 09:00–13:00 / 14:00–18:00 windows |

Issue time, retrieval time, observation time, model-source time and target time are different clocks. They must not be substituted for one another. The code uses Malaysia time for session selection and stores absolute timestamps for matching.

### Inputs and competing models

- Local AirGradient PM2.5/PM10, temperature and humidity; derived recent references, momentum, variability and particle mix.
- Archived Open-Meteo weather values available at each issue, including rain timing/context and wind.
- CAMS regional concentrations accessed through Open-Meteo, with local-correction experiments.
- Subang observations as a spatially distinct weather reference, not a forecast or proof of rain at TTDI.

Near-term routing compares learned/analogue alternatives with persistence and retains separate eligibility checks for arrival, mean and peak. When an alternative fails those checks, a recent-sensor reference can remain the main point forecast. That is an explicit fallback, not evidence that the future will be steady.

The ordinary session estimator is an adaptive local ensemble. Its historical expert scores incorporate each issue's own recent-minus-closed sensor adjustment. The final nonnegative floor is applied after mixing components and the current reference adjustment. This aligns historical scoring with live use; the fixed ablation showed negligible aggregate error changes.

The **experimental early-afternoon override** is `afternoon_direction_hgb_v1`: a rise/steady/fall classifier with a within-class historical median change. It is eligible only at issue times **07:00:00–08:00:00 MYT**, for the same day's **14:00–16:00** target. It uses completed earlier training outcomes and available archived external forecasts. Steady predictions can equal the recent reference. Class scores are uncalibrated, not event probabilities. The experiment was specifically authorized despite a failed latest-block safeguard; it is not universally validated.

Dedicated rain-learning and CAMS/local-correction trials are diagnostics unless the returned provenance explicitly says otherwise. Their presence on a page is not proof they changed the main point. The September review did not deploy new magnitude heads, direct-delta alternatives, a rapid-clearing overlay or a new CAMS slope.

### Evaluation and what it does not establish

The journal's numerical table is an exploratory **07:30→14:00–16:00** comparison on 17 known daily outcomes from September 3–20, excluding September 7's incomplete target. It does not measure the morning model or +90-minute skill. Multiple issue clocks share outcomes and are not independent events.

The retained experiment's primary MAE is 22.216 µg/m³, MAPE 29.700% and WAPE 28.311%; matched recent-sensor persistence is 25.862 µg/m³, 35.249% and 32.957%. It detected 2/7 substantial rises and 5/6 substantial falls, with two wrong-way event calls and two change calls on four broadly steady days. Actual changes use ±10 µg/m³ and forecast calls ±5, relative to the matched issue-time reference. These are test definitions, not health thresholds.

Chronological blocks contain 6, 7, 3 and 1 outcomes. The latest case predicted clearing incorrectly. Known-outcome model development and repeated comparisons prevent interpreting these scores as an untouched holdout or guaranteed future performance. Stored sensor readings also lack a complete original acquisition-vintage record, which limits exact reconstruction of what every historical request knew.

The separate ChatGPT reviewer checked supplied source/evidence and reproduced the primary/block percentage arithmetic. It did not independently refit these models or inspect the deployed server. This is AI-assisted critical review, not human scientific peer review.

### Uncertainty, failures and observability

Empirical q10–q90 spans are **uncalibrated**. For the scoped experimental session model, a model-specific interval has not been established; do not borrow the ordinary model's interval. Missing outputs remain missing rather than being manufactured from an unrelated target.

Known weaknesses include missed rises, false clearing, weather timing/location error, slow reaction or lack of anticipation, short regime coverage, sensor gaps, correlated overlapping targets and source/model changes. Correlation between cooling and falling particles does not establish rain or identify the smoke source.

Selection diagnostics now record gate values, thresholds, margins, unavailable evidence and the final published stage. Mean eligibility does not require a peak estimate. Scheduled model handoffs can move the number without new observations; cached outputs keep their original issue and target times. Historical reconstructions are not relabelled as forecasts actually issued at the time.

### Release policy

New candidates should be compared on identical exact targets and information cutoffs, with persistence retained as a baseline. Report absolute error, percentage error, rise/fall detection, false clearing and chronological/event-level behavior. Separate correctness fixes from predictive-skill claims. Freeze a candidate before collecting new prospective evidence; do not silently broaden an experimental scope.

The public synthetic tests check implementation contracts. They do not reproduce the private historical study or certify forecasting accuracy.
