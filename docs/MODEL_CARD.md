# Forecast model card

**Current published source:** firmware 40-public.1 of 10 November 2026, which runs the forecast on the ESP32 desk display, and server v25.2.0-public.1 of 7 November 2026, the reference implementation its models were ported from. Both publish RLCD schema 1 field meanings.

The [firmware package](../firmware/esp32-rlcd/README.md) and the [7 November release](SERVER_RELEASE_2026_11_07.md) are the current sources; the [October 8 release guide](SERVER_RELEASE_2026_10_08.md) still covers server installation. The earlier sections below are dated research and deployment records; their models, selectors and fallback behaviour do not describe the current publisher. Fitted assets, working data and the private Windows service installation are excluded.

## The models on the desk display — 10 November 2026

*Written by Claude.*

The owner's Waveshare ESP32-S3-RLCD-4.2 now collects its own data and retrains every model on the chip each night, from 00:05 Malaysia time. The nightly refit takes about 13.9 minutes, and the forecasts read unavailable until it finishes. Every output is the model's literal output, and a missing output is shown as unavailable. The PM2.5 values carry `role: raw_model_output` and `calibrated: false`; the arrival-change and P(ride mean ≤ 70) objects carry neither field, and the arrival probabilities come from a recall-weighted fit.

| Output | Model on the device | Server model it follows |
|---|---|---|
| +90 concentration; ride mean, min and max | Four histogram gradient-boosting regressors, absolute-error loss (120 iterations, learning rate 0.1, 15 leaves, min 60 per leaf, L2 1.0, no early stopping); 81 inputs: 28 sensor, 4 control, 5 time, 18 neighbour-station, 10 network, 16 issued-wind; 15-minute origins over 28 days (about 2,630–2,670 rows); about 655 s | `spatial_hgb_point_p1_no_weather_v25_20261014` |
| P(ride mean ≤ 70) | Out-of-fold ride-mean residuals over three day blocks | `spatial_residual_le70_e1_v25_20261014` |
| ≥20/≥40 change by arrival | `device_arrival_logistic_recall_f2_v2`: median imputer with missing indicators, standard scaler, multinomial logistic (C = 0.1, L-BFGS, max 500 iterations, tol 1e-4), both fall classes weighted 2; 28 sensor inputs at 5-minute issues over 28 days, rebuilt from stored readings (8,019 rows, 59 iterations at the 10 November cutoff); about 157 s | `receipt_visible_logistic_arrival_change20_40_recall_f2_v2` |
| Morning (09:00–11:00) and Afternoon (14:00–16:00) | `device_session_pooled_hgb_v1`: one pooled gradient-boosting regressor (100 iterations, learning rate 0.05, 7 leaves, min 80 per leaf, L2 50, depth 3, absolute error); 14 days, 7 lead times, 28 of the server Morning model's 31 inputs (4,505 rows at the 10 November cutoff); about 19 s | Server Morning HGB; replaces the Afternoon PatchTST |
| First ≥20 crossing | Not on the device: no page showed it, so it was not ported | `p1_logistic_first20_90min_recall_f12_r6_v1` |

**The Afternoon replacement.** The server's Afternoon card is a PatchTST network, which cannot be trained on this chip. The device instead serves both session cards from one pooled gradient-boosting model built on the server's Morning recipe.

**The CAMS gap.** The device does not fetch Open-Meteo's CAMS air-quality feed. The session model's three CAMS inputs are therefore missing, and its imputer drops them, as scikit-learn drops a column with no observed value.

**Not models.** The observed status line is the server's rule-based observation label, recomputed on the device from the last 24 hours. Tennis Morning, rain chance, feels-like temperature and wind come straight from the Open-Meteo forecast.

**Evidence that the port is correct.** These checks establish software agreement with the server, not forecast skill.
- The C++ gradient-boosting trainer reproduced scikit-learn 1.5.1's `HistGradientBoostingRegressor` on the host in 64-bit arithmetic; the device also trains in double precision.
- With identical inputs, the device's own fitted arrival classifier and the server's pickled model for the same cutoff (8,019 rows, 59 iterations) agreed to a worst probability gap of 5.7e-9.
- A host harness built from the firmware's row builders reproduced the server's training rows. Arrival: the same issues, identical labels, inputs within 2.8e-14. Session: the same 4,505 rows, inputs within 1e-9 apart from the missing CAMS inputs.
- At one issue clock on 10 November, device against server: +90 45.04 vs 46.43 µg/m³; ride mean 44.04 vs 46.14; ride min–max 39.3–59.1 vs 42.1–57.4; P(mean ≤ 70) 0.955 vs 0.943; arrival within-20 0.923 vs 0.929, both "No change on arrival"; Morning 32.16 vs 32.11; Afternoon 29.14 vs 27.43 (a different model). The point differences come from sensor poll timing.

**Evidence on skill.** Scored on the server's issued records, 14 October to 9 November; MAE differences in µg/m³ with 95% intervals.

| Comparison | Difference |
|---|---|
| Device session model vs persistence, Afternoon 15-minute slots | −0.26 [−2.44, +2.42]: tie |
| Device session model vs the issued PatchTST, Afternoon 15-minute slots | −1.08 [−4.69, +2.74]: tie |
| Device session model (28 inputs) vs the issued Morning card | +0.22 [−1.01, +1.38]: tie |
| Cost of dropping CAMS, Afternoon 15-minute slots | +1.60 [+0.45, +2.89] |

The CAMS row is a secondary comparison; read it as a hint that the missing feed costs something on Afternoon, not as an established loss. The point and arrival heads are the server's recipes, so the [season's scorecard](#the-season-on-the-issued-records--7-november-2026) applies to them. The phe point heads beat persistence over the season, mostly through the 14–23 October haze; afterwards the +90 and ride mean tied it and only the ride minimum stayed ahead. The arrival head did not beat its baselines.

**Limits.**
- The skill comparisons are replays on server records over one season, not scores of the device's own issued forecasts. The board does not yet record its issued forecasts, so none can be scored yet.
- A multi-day soak of the standalone firmware has not been run yet.
- Every hyperparameter and weight was tuned for the TTDI sensor. At another station the models are untested, and each head needs its history to fill first: about 3 days for the point heads, 10 days for the arrival and session models, and 14–28 days for their full training windows.

## The season on the issued records — 7 November 2026

*Written by Claude.*

| Output | Model in v25.2.0 |
|---|---|
| +90 concentration, ride mean/min/max | `spatial_hgb_point_p1_no_weather_v25_20261014` (unchanged) |
| P(ride mean ≤ 70) | `spatial_residual_le70_e1_v25_20261014` (unchanged) |
| First ≥20 crossing in 90 min | `p1_logistic_first20_90min_recall_f12_r6_v1`: multinomial logistic, class weights drop 12 / none 1 / rise 6 |
| ≥20/≥40 change by arrival | `receipt_visible_logistic_arrival_change20_40_recall_f2_v2`: both fall classes weighted 2 |
| Morning / Afternoon | HGB / PatchTST (unchanged; PatchTST runtime revision v3) |

The two event models are **recall-weighted by design and not calibrated**. Their outputs carry `calibrated: false` and the weights used.

**How the scoring was done.** Every model was scored on the forecasts it actually published, from 14 October (event models: 15 October) to 7 November. Each forecast was matched with the outcome the server's own targets define. The method reproduces the 17 October interim scorecard exactly. A second, independent re-computation checked every number and tightened several claims. Intervals are 95% day-block bootstraps. The haze ran from 14 to 23 October, with daily means of 70–136 µg/m³; afterwards they were about 10–55.

| Output vs persistence (MAE, µg/m³) | Whole span (25 days) | Haze, 14–23 Oct | From 24 Oct |
|---|---|---|---|
| +90 | 8.45 vs 9.23: **−0.79** [−1.32, −0.27] | −1.91 [−2.67, −1.03] | −0.03 [−0.42, +0.35] |
| Ride mean | 11.13 vs 12.66: **−1.53** [−2.42, −0.67] | −3.45 [−4.44, −2.25] | −0.21 [−0.86, +0.52] |
| Ride min | 11.79 vs 14.34: −2.55 [−3.69, −1.51] | −4.89 | −0.95 [−1.52, −0.37] |
| Ride max | 11.10 vs 13.84: −2.74 [−3.96, −1.53] | −5.27 | −1.01 [−2.12, +0.17] |
| Morning card (07:15 and 18:05 issues) | 15.62 vs 16.49: −0.86 [−4.72, +2.94] | | |
| Afternoon card (12:00 and 18:05 issues) | 17.92 vs 17.77: +0.15 [−5.58, +5.49] | | |

**The point model helps when the air is hazy and changing, and only then.**
- All four heads, the +90 and the ride mean, min and max, also beat damped persistence and a diurnal baseline over the whole span. Corrected for the twelve comparisons, the +90 and ride mean no longer clear damped persistence.
- After the haze, only the ride minimum stayed ahead of persistence: −0.95 [−1.52, −0.37]. It ties a persistence scaled by the recent ratio of minimum to reference.
- At reference PM below 35 µg/m³, the +90 and ride mean tie persistence; at 80 and above they win clearly.
- All four weekly means are negative, but with only four weeks the week-level intervals include zero. One haze episode carries the result.
- The +90 gain of 0.79 µg/m³ over 25 days falls short of the project's pre-declared bar for a validated improvement: 1.3 µg/m³ over 70 target dates.
- The ride window's [min, max] contained the realised ride mean in only 42% of issues, and 34% in the haze. These are point forecasts of the extremes, not an interval.

**P(ride mean ≤ 70).**
- Its Brier score was 0.0557, against 0.0682 for a logistic on the current reading: −0.0125 [−0.0197, −0.0059]. The margin comes from the haze; afterwards 98.9% of rides were at or below 70.
- On log loss it ties that baseline, because of 22 confident misses at two events: a storm-outflow wash-out on 14 October and a rise on the evening of 17 October.

**First ≥20 crossing (24 issue days, from 15 October).**
- **During the haze** it caught **30 of 33** falls of 20 µg/m³ or more, with a median lead of about an hour on raw readings. It did so by calling a change on about half of all issues, which left 70 false-alarm and 24 wrong-way clusters in 9 days.
- **After the haze** it caught 3 of 13.
- **On the tuning objective it beats never calling:** −113.5 [−183, −51], and it did better on all 9 haze days.
  - That is a low bar: always calling a fall also beats never calling.
  - The model cannot be told apart from always calling a fall, or from a hand-written "call a fall whenever PM ≥ 80" reference. Over the 24 days the rule caught 32 of 46 falls and the model 33; in the haze, the rule caught 32 of 33 and the model 30.
- **Hints that it is more than a level rule:**
  - It beat all 23 within-day time shifts of its own calls, so the timing of its calls carries information.
  - A post-hoc check re-counts long false alarms every 90 minutes. There the model is ahead of the PM ≥ 80 rule (−60.5 [−119, −5.5]), but the margin does not survive 3-day blocks.
- **Drop calls:** they had 9.9× lift below 80 µg/m³ and 1.5× at 80–150. There is no evidence of lift at 150 and above, where it called "drop" 92% of the time.
- **Probabilities:** as expected from the weighting, its probabilities are worse than simple recent base rates.

**Arrival change (≥20/≥40).**
- It called a change on 2.3% of issues, and once in the 15 days after the haze.
- It caught 9 of 39 ≥20 falls and 3 of 14 ≥40, mostly late: its first right call came a median 66 minutes into the episode. If a catch must come at least 30 minutes ahead, its advantage over never calling disappears.
- Against never calling it is borderline. Its log-loss gain over a 28-day base rate only reflects that base rate lagging the change of regime; it ties 3- and 7-day base rates.

**The Morning and Afternoon cards.**
- Both tie persistence at their canonical clocks.
- On every 15-minute issue outside the sealed-window days, the Afternoon card was worse than persistence: +3.5 [+0.6, +6.7]. It is the only Afternoon comparison whose interval excludes zero, and haze days drive it, so it is suggestive.
- Separating a 10% difference would take roughly 110–380 days, more than one season holds.
- PatchTST is trained from a single random seed. Retrained with three seeds, its forecasts spread by about 15 µg/m³ (a single reseed moves them about 10). A three-seed average is the obvious next step.

**Limits.**
- One season, and one 9–10-day haze episode, carry nearly all the events.
- Many splits were reported, so a few intervals exclude zero by chance.
- With 3-day blocks and a multiple-testing correction, most of the event-model claims lose significance.
- Read these results as one season's prospective evidence, not as validation.

## Spatial point model and P(ride mean ≤ 70) — 14 October 2026

*Written by Claude, which has maintained the project since 13 October.*

| Output | Model in v25.1.0 | Target |
|---|---|---|
| +90 concentration | `spatial_hgb_point_p1_no_weather_v25_20261014` | Interpolated complete 15-minute medians at exactly issue +90 min |
| Ride mean, min, max | Same model, three more heads | Overlap-weighted mean, and smallest/largest complete 15-minute medians, over +90 to +210 min |
| P(ride mean ≤ 70) | `spatial_residual_le70_e1_v25_20261014` | Ride mean at or below 70 µg/m³ (a planning cutoff, not a health limit) |
| ≥20/≥40 change by arrival | `receipt_visible_logistic_arrival_change20_40_v1` (unchanged) | Five classes of +90 change from the issue's fresh reference |
| First ≥20 crossing in 90 min | First-crossing HGB v2 (unchanged) | First sampled crossing at +15…+90 |
| Morning / Afternoon | HGB / PatchTST (unchanged) | Dated two-hour session means |

**Model.** Four HistGradientBoosting absolute-error regressors predict (target − fresh reference). They are refit every Malaysia midnight with a 28-day lookback, the 330-minute embargo and labels complete before the cutoff. Inputs: the arrival logistic's 28 sensor columns, 4 mean-reversion controls, 5 time columns and 44 features from six neighbouring public AirGradient stations, some weighted upwind by the issued Open-Meteo wind. P(≤ 70) places the empirical distribution of out-of-fold log residuals around the ride-mean forecast. Outputs are literal. A stale fresh reference, no wind run within 2 h, no neighbour station with a bucket in the last 90 min, or a missing model makes every head unavailable; one offline station is only a missing input.

**Evaluation.** The [research harness](../research/models/README.md) builds a 5-minute issue grid (26 August to 13 October) in which every input is taken as visible at its issue clock. It refits each candidate daily, walk-forward, on the server's own targets. Seven candidate families were each attacked by a leakage skeptic and a robustness skeptic. The variant rule was fixed in code before the final run: tune on 10 September to 5 October, report on 6–13 October.

| Holdout 6–13 Oct (lower is better) | Persistence | Previous model | v25 | v25 − previous, 95% CI |
|---|---:|---:|---:|---|
| +90 MAE, µg/m³ (n = 2,248) | 10.64 | 11.08 (Ridge v4) | 9.55 | −1.54 [−2.17, −0.86] |
| Ride-mean MAE (n = 2,199) | 14.09 | 14.06 | 12.88 | −1.18 [−1.83, −0.56] |
| Ride-min MAE | 15.78 | 15.54 | 14.30 | −1.23 [−2.02, −0.32] |
| Ride-max MAE | 15.70 | 13.79 | 12.75 | −1.04 [−1.47, −0.55] |
| P(≤ 70) Brier (n = 2,194) | 0.0825 | 0.0953 (cycling logistic) | 0.0540 | −0.0413 [−0.0663, −0.0180] |

Intervals are day-block bootstraps. At the clocks actually issued on 8–13 October, +90 MAE was 8.64 against 10.15 for the issued Ridge v4 and 9.65 for persistence. With every input taken 10 minutes stale, +90 MAE rose to 10.17, still below fresh persistence. Over the whole 34-day period, the replayed Ridge v4 had been worse than persistence (+90 MAE 8.48 against 7.96).

**Events were not improved.** In the late half (27 September to 13 October; 58 ≥20 and 17 ≥40 episodes), the unchanged arrival logistic caught 11/58 ≥20 episodes, with 14 false-call clusters and 5 wrong-way clusters, and 1/17 at ≥40. Better-calibrated candidates scored lower Brier but caught 0–3 of 58. The new point model is a median regressor: on the holdout it made 1 of 33 ≥20 calls. Advance warning still comes only from the arrival logistic.

**Limits.**
- The holdout is 8 days, with heavily overlapping issues. Margins over the best single candidate are within noise.
- I read candidate results that included holdout days before writing the variant list.
- The tune rule chose a plainer P(≤ 70) variant (E0). I adopted E1 because only it beat every single candidate on the holdout, so its holdout margin over E0 is not independent; over all 34 days the difference is not significant.
- Ride min and max are median heads, not quantile bands; the range is neither a confidence interval nor an envelope.
- The model depends on fresh neighbour polls; one station (Setapak) has been offline since 9 October.
- Results are development evidence on one sensor, not prospective validation. Correspondence with trail conditions is not established.

The [results](../research/models/RESULTS.md), [baselines](../research/models/BASELINES.md) and [journal entry](journal/2026-10-14-testing-the-forecast-without-leaks.md) give the full tables.

## Rain review and numerical training correction — 9 October 2026

Local/public v24.0.4 uses `fresh_sequence_ridge_delta_numeric_v4_receipt_training_20261009`. The Ridge recipe, 208 sensor inputs, four arrival/ride targets, daily midnight cutoff and 330-minute embargo are unchanged. Training now retains confirmed sensor revisions and bounds them by each historical origin; completed labels use revisions visible by the training cutoff. Partial receipt namespaces fail closed, and untracked legacy availability remains unknown. This fixes a demonstrated training-data defect. It does **not** establish earlier rain/drop warning.

Two fixed richer temperature/humidity probability candidates were worse than the scalar logistic on the 3,232-case, 34-date historical development cohort: richer logistic Brier/log loss 0.164004/0.392122; richer HGB 0.155886/0.378399; scalar logistic 0.155885/0.365475. The fixed numerical HGB with 240 inputs had MAE 8.489 µg/m³ versus 8.521 with 208 inputs and 8.112 for a receipt-aware fixed-Ridge-recipe reconstruction. That reconstruction corrects historical input availability and does not reproduce the original v3 refresh path or originally fitted asset. These are exposed development results. The inspected October 8/9 cases are separate diagnostics; unsuccessful candidates remain offline.

An originally published October 9 14:14:15 arrival indication gave 52.6% for a ≥20 fall and 20.8% for ≥40, about 72 minutes before the sharp phase. Its numerical point 166.6 missed the completed 90.6 proxy. False warnings and rebound errors remain explicit. Local temperature fell while RH rose. The [journal](journal/2026-10-09-learning-from-a-rain-drop.md) and [release note](SERVER_RAIN_TRAINING_2026_10_09.md) report the source, study and 236 synthetic tests separately. Current classifiers remain sensor-only; confirmed weather coverage is too short to claim demonstrated weather learning. The arrival target still interpolates complete 15-minute medians and can smooth abrupt changes.

## Source release and arrival distribution — later on 8 October 2026

The earlier v24.0.3 publication corresponded to that local build. That device-presentation/history update did not change the fixed forecast learners described in this dated section. Its fixed Ridge v3 learner publishes the +90 concentration and +90–210 mean/minimum/maximum without numerical selection or persistence substitution. A separately fixed logistic classifier, `receipt_visible_logistic_arrival_change20_40_v1`, predicts a five-class change distribution at exact +90 relative to the issue's trailing five-minute reference. Native class masses supply ≥20/≥40 fall and rise tails and the within-20 outcome. The web focuses on these arrival outcomes; the distinct first-sampled-crossing HGB output remains in the APIs and issue archive.

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
