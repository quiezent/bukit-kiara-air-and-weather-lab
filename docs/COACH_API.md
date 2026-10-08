# MTB environmental evidence API

The dashboard exposes one compact, read-only evidence endpoint for a local coaching stack, plus three discoverable contract resources:

```text
GET http://127.0.0.1:8765/api/v1/mtb/environment-evidence
```

Use `http://127.0.0.1:8765` as the origin when the coaching process runs on the dashboard laptop. Use the dashboard laptop's LAN address when the Coach runs on another device.

The response contract is versioned by `schemaVersion` (currently `1.37.0`). It contains current environmental evidence and aggregated analysis, not sensor-history rows, ride-window selection, or a training prescription.

## Arrival changes for the preparation decision (1.37.0; October 8, 2026)

Build `2026-10-08-arrival-preparation-models-v24.0.0` adds `exposureOutlook.arrivalChangeForecast`. Its fixed `receipt_visible_logistic_arrival_change20_40_v1` learner predicts the exact issue +90 concentration proxy minus the issue's declared trailing five-minute sensor reference. It does not predict the first crossing anywhere before arrival. `firstCrossingEventForecast` retains its existing target and model identity.

The new object exposes `probabilityFall20`, `probabilityFall40`, `probabilityRise20`, `probabilityRise40` and `probabilityWithin20`, along with `classProbabilities`, `referencePm`, `forecastIssuedEpoch` and `arrivalEpoch`. Its five learned bins are delta ≤−40; −40<delta≤−20; −20<delta<20; 20≤delta<40; and delta≥40 µg/m³. The estimator's native class distribution supplies coherent tail sums. All four tails remain available irrespective of the largest class. No ≥40 probability is scaled from first20, inferred from a PM2.5 point or range, or masked by a direction decision. The categorical distribution does not identify a concentration expectation; the numerical arrival model remains a separate learner.

For a reference of 100 µg/m³, arrival falls of ≥20 and ≥40 mean arrival proxy ≤80 and ≤60; rises mean ≥120 and ≥140. The reference is a model input and may differ from the latest single Current observation. Arrival uses interpolation of complete 15-minute sensor medians at exact +90. This target does not establish onset time, a continuous-time extreme or conditions on every trail section.

The arrival classifier uses completed outcomes before Malaysia midnight, aligned five-minute origins and a 120-minute origin embargo, independent of the +210 ride labels. Its fixed multinomial logistic recipe uses C=0.1, training-only median imputation with missing indicators and standard scaling, and 28 scalar sensor/time features. A separate HGB development comparison included 285 columns with 77 strictly confirmed issued-weather fields. The coverage audit found **zero confirmed weather training rows before October 8**, and its weather-augmented and sensor-only predictions were identical. The deployed classifier does not use weather or neighbors. Unknown older sensor receipts remain identified, not reconstructed as known availability.

The web shows every arrival tail. Web update `v24.0.2` removes the “Change before arrival” subsection to focus the preparation decision on conditions at arrival; the first-crossing model, API fields and archived forecasts retain their existing semantics. Small positive probabilities below 0.1% display `<0.1%`, retaining full precision in API/archive data. The compact RLCD schema remains **1** and adds optional `forecast.near90.arrival_change`: `available`, `fall20`, `fall40`, `rise20`, `rise40`, `reference_ugm3`, `arrival_epoch` and `model`. Those probabilities are copied literally. The existing `near90.first20` fields keep their original semantics. Full class, source and input provenance stays in Coach and the issue ledger, rather than consuming the device's 8,192-byte budget.

Web wording update `v24.0.1` summarizes the largest of the three disjoint arrival probabilities as “Fall on arrival”, “Rise on arrival” or “No change on arrival”. Exact ties say “Arrival outcome uncertain”. “No change” refers to the displayed within-20 outcome, rather than exact zero movement. This changes presentation only; all four tails and every numerical forecast remain direct model outputs.

Future issues use compact ledger v6, retaining the complete arrival-change object and any separately identified numerical arrival source. Fits run in independent background workers; forecast requests do not train. Actual publication records retain their original reference, absolute arrival time, model/feature identity and publication clock for prospective evaluation. Historical candidate fits made after an event remain counterfactual development evidence; software checks and an encouraging inspected episode do not establish reliable advance warning.

The chronological development comparison includes 3,232 common historical issues across 34 dates; 68 inspected October 8 cases are separate. Logistic class log loss is 0.365475 versus the training-prior comparator's 0.371524, while Brier is worse (0.155885 versus 0.150380). At a 10% evaluation-only threshold, five of nine large-fall movement episodes had an earlier signal, but only three selected warnings also had a positive outcome at their own +90 target; there were 23 false endpoint-warning clusters. The threshold does not gate the display. Rare rises remain poorly detected. The categorical model can also assign small physically unsupported drop probabilities at low references; native values are preserved, and this limitation is retained in `performanceEvidence`. These results do not establish reliable or calibrated preparation warnings.

The dedicated HGB numerical arrival candidate remains offline: its historical +90 MAE was 8.8697 µg/m³ versus the incumbent Ridge's 8.1124 and the fresh-reference comparator's 7.5402. The fixed Ridge continues publishing all four numerical heads without runtime selection or fallback. Its exact issued feature vector is retained with a matching query hash for independent native-output checks. `provenance.forecastPolicy.id` is `fixed_direct_models_with_arrival_change_v24`.

## Fresh sensor sequence forecasts (1.36.0; October 8, 2026)

Build `2026-10-08-fresh-sensor-models-v23.0.0` publishes the fixed `fresh_sequence_ridge_delta_numeric_v3_20261008` learner for the exact +90 point, +90..210 mean and ride minimum/maximum. `fresh_sensor_hgb_direct_first20_90min_v2` learns the separate next-90-minute first-crossing outcome directly. Both use the same causal 3-hour sensor sequence, fresh five-minute reference, short-interval movement and temperature/humidity inputs. Morning and Afternoon retain their fixed session models.

The numerical estimator learns target changes relative to that same fresh reference. Its own `predict` method inverts this training representation and returns physical concentration values; the publication layer copies those outputs without arithmetic. The reference is consistent between training and serving, rather than an adjustment added to a model trained against an older bucket. The event model exposes its directly learned 90-minute class distribution; it does not claim separately learned 15-minute hazard probabilities.

The numerical fields copy `freshSensorForecast.arrivalPoint`, `.trailMeanPoint`, `.trailMinimumPoint` and `.trailMaximumPoint` literally, including full precision. Accuracy results do not select a number, add a recent-sensor offset, clip or widen the range, or substitute a baseline. A missing estimator or invalid causal input gives unavailable. `modelVersion`, `modelOutput`, `performanceEvidence`, input/feature hashes and training metadata identify the new heads in Coach and archived records. New issues use ledger v5 and separate v23 prospective cohorts.

Daily training ends at Malaysia midnight, includes only completed prior outcomes and applies a 330-minute embargo to numerical origins. Training-only median imputation and standard scaling prepare features; missing outcomes are never filled. Historical receipt coverage remains incomplete and is not presented as complete causal replay. Development comparisons and today's previously observed drop do not establish prospective predictive accuracy. The seven-date numerical comparison remains mixed: +90 MAE 8.64 versus 8.17 for the previous learner, and ride-mean MAE 10.94 versus 10.70. The separate event learner still misses many drops. These are input and contract repairs, without a claim that forecasting accuracy is solved.

Current-card observation labels survive independent sensor/history refreshes while their measured-through timestamp remains recent. They describe measured movement. Forecast momentum text displays `No ≥20 µg/m³ change: [value]%`, a rise/fall percentage, or tied outcomes; the page shows the next 90-minute interval and its actual reference. Optional RLCD `display_text` remains schema-1-compatible and uses the same formatter.

## Shared momentum display text (introduced in 1.35.0; updated in 1.36.0)

`/api/rlcd/v1` adds the optional nullable string `forecast.near90.first20.display_text`. The corresponding web/Coach event field is `firstCrossingEventForecast.displayText`. Both are produced by the same server formatter; clients should display the returned text without deriving their own momentum label. Existing RLCD schema version 1 and existing numeric, model, reference, direction and qualification fields remain compatible.

The current possible texts are `No ≥20 µg/m³ change: [value]%`, `[value] % ≥20 µg/m³ rise`, `[value] % ≥20 µg/m³ fall`, and `Momentum outcomes tied`. The former `No large change expected` headline concealed how close the other outcome probabilities could be. The text describes the highest-probability diagnostic first-crossing outcome within the next 90 minutes from the displayed reference, including unqualified model outcomes. It does not change operational qualification or replace the concentration forecast. Probability display rounds to one decimal and omits a trailing `.0`; the probability fields retain full precision.

`display_text` is `null` when the event is unavailable, malformed, or suppressed by the existing RLCD issue/sensor freshness guards. The web retains its existing clock/reference checks. Device clients should retain stale/unavailable guards, accept an absent field from older servers, and render `≥` as `>=` if their font does not support the glyph. The existing 8192-byte budget still preserves forecast fields and reduces only the device graph sample count when needed.

## Historical ride minimum–maximum outputs (1.34.0; October 8, 2026)

Build `2026-10-08-ride-min-max-v22.0.0` adds `exposureOutlook.onTrail.rideMinimumMaximumPm25UgM3`. Its `low` and `high` are the literal paired outputs of `local_extra_trees_joint_extrema90_210_v1`. They predict the minimum and maximum **15-minute PM2.5 bucket medians** overlapping the exact issue +90 to +210 minute ride window. A partial edge bucket summarizes its full observation interval; brief spikes between readings can differ. This target describes variation during the ride, not uncertainty around the two-hour mean or a calibrated probability interval.

The fixed joint estimator learns both endpoints from the same completed targets and causal feature rows. Its leaves average paired observed minima and maxima, preserving their ordering without sorting, widening, clipping, performance selection or recentering around the mean. The existing Ridge `projectedMeanPm25UgM3` remains its own unchanged numeric model output and need not lie within the independently learned extrema. Mean interval properties remain null when unavailable; the extrema never populate quantile/mean interval fields. Missing extrema show unavailable rather than mean, Current or persistence.

The object retains its exact `forecastClock`, `modelParameters`, source feature/target hashes, training cutoff and latest completed training outcome. New compact dashboard issue records preserve it under ledger v4. Device responses expose the same values at `forecast.ride90_210.minimum_maximum`, separately from the mean `pm25_ugm3` and any uncertainty fields. All APIs retain full precision; the page formats the values for readability.

The dashboard removes the word “Experimental” from visible labels. It retains calibration and predictive-accuracy information. This display wording does not change model identities, performance evidence or existing API flags. The first fixed 16-origin chronological development check found minimum MAE 21.573 and maximum MAE 19.760 µg/m³; these errors are informational and do not alter either output. No prospective validation or calibrated coverage is claimed.

## Historical direct model outputs (1.34.0; October 8, 2026)

Build policy `fixed_model_outputs_v21` publishes one fixed model for each numeric target. Statistical performance, event classification, model promotion and persistence comparisons do not select, replace, shrink, rebase or suppress its finite output. There is no alternate-model or persistence fallback. Missing required inputs, an unavailable cached estimator, malformed clocks or an invalid model number produce an unavailable forecast with a null model value.

| Target | Fixed numeric model | Coach numeric property |
| --- | --- | --- |
| Exact issue +90-minute PM2.5 | `local_bounded_analogue_raw_point90_v1` | `exposureOutlook.arrival.projectedPm25UgM3` |
| Exact issue +90..210-minute mean | `local_robust_ridge_alpha100_raw_mean90_210_v1` | `exposureOutlook.onTrail.projectedMeanPm25UgM3` |
| Dated Morning 09:00–11:00 mean | `morning_hgb_basic31_model_direct_v3` | `rideWindows.morning.particleForecast.publishedMeanPm25UgM3` |
| Dated Afternoon 14:00–16:00 mean | `afternoon_patchtst_original_model_direct_v3` | `rideWindows.afternoon.particleForecast.publishedMeanPm25UgM3` |

The near fields copy `shadowForecast.arrivalPoint` and `shadowForecast.trailMeanPoint` without further arithmetic. The session fields copy their fixed estimator's returned mean, including PatchTST's raw prediction without its former numeric deadband. API JSON preserves the returned numeric precision; the dashboard rounds only its displayed text. Numerical equality with the current observation or reference can occur naturally and does not change the model's identity or role.

`pointRole = experimental_model_output` (or `raw_model_output`), `forecastState = model_output`, the model version and the exact issue/target clocks identify a direct forecast. In session objects, `publishedMeanPm25UgM3`, `projectedMeanPm25UgM3`, `planningEstimate.meanPm25UgM3` and `candidateForecast.mean` describe the same returned number. `modelPointApplied` means a model number is published; it does not assert measured skill. `modelSelection` is empty for the fixed sessions. `modelOutputPolicy`, `provenance.forecastPolicy.numericalPerformanceSelection = false` and `numericPersistenceFallback = false` identify the numerical contract.

`baselineMeanPm25UgM3`, near `persistenceAnchorPm25UgM3` and their deprecated aliases are separate sensor-reference comparators. They must not replace a missing model value. The recent reference can be newer than the near model's completed 15-minute feature anchor; it is not added again to the near output. `validated`, `validatedModelPointApplied`, `prospectivelyValidated`, `usedForDecision`, calibration state, development results and issued-performance evidence describe uncertainty or eligibility. They do not control whether a valid numeric model output is exposed. A low score or an unqualified direction does not turn the number into persistence.

The exact dated targets remain authoritative. Morning and Afternoon use canonical 09:00–11:00 and 14:00–16:00 sessions with a 90-minute logistics reference, rolling to the next feasible date when required. A lead outside the model's development/training range is reported as extrapolation in `candidateForecast.modelDomain` and model metadata; it does not route to another predictor. It provides no claim of accuracy at that lead. Actual unavailable causal inputs still make the model unavailable.

Weather age and observation age use response wall clock. Forecast age uses the original issue, and cached targets retain that issue. Each weather window includes `weatherCoverage` and requires complete hourly support for its exact interval and required meteorology, precipitation and precipitation probability. Missing values remain null; missing rain is never a dry zero. Hourly rain/probability and gust maxima describe preceding-hour blocks, with overlap used for interval totals. Legacy summaries without coverage proof are unverified. Model-input weather clocks and the separately displayed weather vintage remain distinct.

New receipt/revision ledgers distinguish observed/source timestamps from network receipt, archive insertion and confirmed post-commit availability. Tracked as-of readers bound availability by the issue; older records without receipt evidence remain explicitly unknown rather than backfilled. The private publication input manifest identifies its actual covered sensor view and materialized PatchTST sources; it does not claim all models were frozen in one transaction. Each model's own source hashes, feature clocks, fit cutoff and artifact identity remain part of its audit metadata. New wrapper/input-policy identities are scored separately from old issued records.

`firstCrossingEventForecast` describes the separate first sampled ≥20 µg/m³ crossing within 90 minutes, not the +90-minute endpoint value. Its highest-probability direction is descriptive; qualification and performance metadata do not select a numerical forecast. The web no longer shows this before-arrival subsection. The numeric arrival model remains independent of event availability, direction and qualification. The cycling ≤70 mean probability is also a separate output and never supplies a PM2.5 value.

The raw near model has no fitted prediction interval or peak output in this contract. Its interval/peak numeric fields are null; older persistence bounds belong only to separate baseline metadata. Session intervals require matched issued evidence for their own version and exact clocks; absent evidence does not borrow another model's residual span or alter the mean. Neither development comparisons nor calibration/qualification labels establish reliable predictive accuracy.

Earlier selectors, fallback rules, deadbands and point overlays described in dated research records do not override this current contract. Ride minimum/maximum are now separate learned outputs described above, not uncertainty bounds borrowed from another model.


## Discoverable resources

| Resource | Route |
|---|---|
| Coach evidence | /api/v1/mtb/environment-evidence |
| OpenAPI | /api/openapi.json |
| JSON Schema | /api/v1/mtb/environment-evidence/schema |
| This guide | /docs/coach-api.md |
| Browser analysis | /api/analysis?days=28 |
| Current sensor | /api/current |
| Device payload, schema 1 | /api/rlcd/v1 |

The schemas generated by this same build are authoritative for field structure, version and nullability. Preserve original issue/reference/target clocks, model identities and unavailable states. The endpoint supplies environmental evidence, not a ride instruction.

For setup, dependency and Afternoon provisioning details, see [the source release guide](SERVER_RELEASE_2026_10_08.md). Earlier contract behavior and research remain in the dated [developer journal](journal/2026-10-08-warning-before-the-ride.md) and [model card](MODEL_CARD.md). Private service launchers and operational archives are not part of this public guide.
