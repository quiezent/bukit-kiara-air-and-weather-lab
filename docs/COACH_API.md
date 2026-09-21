# MTB environmental evidence API

The dashboard exposes one compact, read-only evidence endpoint for a local coaching stack, plus three discoverable contract resources:

```text
GET http://127.0.0.1:8765/api/v1/mtb/environment-evidence
```

Use `http://127.0.0.1:8765` as the origin when the coaching process runs on the dashboard laptop. Use the dashboard laptop's LAN address when the Coach runs on another device.

The response contract is versioned by `schemaVersion` (currently `1.24.0`). It contains current environmental evidence and aggregated analysis, not sensor-history rows, ride-window selection, or a training prescription.

## Reviewed scoring and selection provenance (1.24)

`rideWindows.<window>.particleForecast.modelSelection` identifies the actual selected estimator, scope or fallback reason, and original forecast issue. A cached forecast keeps its original estimator, issue and targets even when delivered after the 08:00 experimental-scope boundary. A scheduled switch of estimator may change the numerical estimate without new observations; it is not measured PM movement. No scope extension or smoothing is introduced.

`directionalModel.diagnostics` distinguishes steady class decisions, ties, insufficient training support/classes and invalid inputs. Class scores are **uncalibrated classifier scores**, not event probabilities. The diagnostics come from the same fit as the point forecast; the classifier and its within-class median magnitude are unchanged.

`provenance.nearTermSelectionDiagnostics` records evaluated gate values, thresholds, outcomes and intermediate points, followed by `publishedStage` after the recent-sensor reference refresh. Unavailable/not-evaluated checks are distinct from failed checks. It carries the issue, build and numerical policy identifiers. These diagnostics are recorded prospectively in dashboard publications; separately reconstructed historical diagnostics are not original issued evidence. The mean no longer requires a peak value when its own eligibility and exact target-clock checks pass. Missing peak estimates are not invented.

The session ensemble now scores each historical expert using that issue's fresh-reference adjustment, while keeping clipping **after** the final weighted mixture. Caches and replay residuals use the revised scoring policy. This is a consistency fix, not a claim of meaningful forecast-accuracy improvement; empirical spans remain uncalibrated. Keep build, model and `provenance.forecastPolicy` identifiers when evaluating forecasts, and do not mix old and new numerical policies silently. Historical archive entries are unchanged.

Neither the tested conditional-magnitude heads nor phase matching, a rapid-trigger overlay, or a new CAMS residual-slope feature is deployed in this release.

## Scoped experimental afternoon direction model (1.23)

The user approved an experimental replacement **only for the same-day 14:00–16:00 PM2.5 mean calculated between 07:00 and 08:00 Malaysia time**. Other afternoon issues, the morning session, +90-minute point and +90–210-minute mean remain on their existing methods. Per-window `particleForecast.modelVersion` identifies the estimator; `directionalModel.applied = true` identifies `afternoon_direction_hgb_v1`. Missing required inputs or an unsuccessful fit retain the existing session forecast, not a fabricated directional result.

This fixed classifier learns rise/steady/fall from completed outcomes over the preceding 14 days, using PM momentum, temperature/humidity, and archived weather/CAMS available at each issue. Its chosen rise/fall is converted to a concentration using the median historical change in that class; a steady classification retains the recent five-minute sensor reference. It can still return the current reference legitimately. It does not apply the older session model's additional post-fit sensor correction. Internal class scores are not published as calibrated probabilities.

The exploratory 07:30 replay across 17 observed days caught 7 of 13 substantial changes versus 1 of 13 for the previous session model, with MAE 22.22 versus 25.84 µg/m³. It made two wrong-way calls and two false change calls on four broadly steady days. On 20 September it incorrectly predicted clearing (69.4 versus observed 101.96 µg/m³), failing the latest-block safeguard. Deployment is **explicitly user-authorized experimentation**, not a claim that every release criterion passed. Surrounding issue times are an extension of that tested 07:30 configuration, not independent validation.

`modelEvidence.retrospectiveReview`, when present, describes that fixed development review; its comparator is the previous adaptive session model, **not persistence**. It is not a continuously updated performance score or a historical issue-time feature. `validationState` remains `experimental_not_validated`. Do not infer a confident session ordering or a training prescription. No model-specific prediction interval is established: range endpoints remain null, `uncertainty.state = collecting`, and the previous model's residual range is not reused. Issued experimental forecasts are archived separately under their own model version for later prospective evaluation. Retrieval alone does not trigger a forecast fit.

## Completed forecast delivery (1.22)

Live requests return the last completed primary analysis immediately while a single background producer maintains the next analysis. During startup, no completed analysis means HTTP `503`; consumers must not treat a newly generated response as a newly issued forecast. `provenance.analysisDelivery` reports delivery state, whether work is underway, the original forecast's age, and the diagnostic stage. A snapshot is available for at most 600 seconds after its original issue. Its observation, issued forecast clocks, target times, points and bounds remain associated with that calculation.

`status.forecastAgeSeconds` is measured at response delivery. After 120 seconds, the response is degraded with `analysis_stale` and `status.fresh = false`; after 600 seconds it is unavailable with `analysis_expired`. These are operating freshness limits, not accuracy or health thresholds. `status.analysisUpdating` and `provenance.analysisDelivery.updating` report ongoing work without implying that the stored evidence has been refreshed. A failed update retains an unexpired completed snapshot with `analysis_update_error`; it does not reset its age.

The primary analysis is published before the separate rain and CAMS correction trials finish. `provenance.analysisDelivery.diagnostics` distinguishes pending, complete and failed diagnostic work. Pending or failed trials provide no substitute forecast. Completed trials retain their own original issue, target and source clocks and remain excluded from the primary estimator and session comparison.

`generatedAt` is response time; `provenance.analysisComputedAt` is primary computation completion; `provenance.analysisForecastIssuedAt` is the frozen calculation issue. Cached window `leadHours`, `active`, `currentConditionsApplicable`, `targetDay`, event state and event-relative ages describe that calculation, not a new assessment at response time. Each window's `startAt`/`endAt` preserves the absolute session date, including across midnight and when weather is unavailable. Near-term minute offsets are measured from `forecastIssuedAt`. Use the absolute target/issue timestamps to interpret a delivered forecast. Observation age and weather/Subang source ages are measured again at delivery without changing their readings, source timestamps or forecast targets. CAMS provenance is bounded by the original issue and cannot be replaced with a later retrieval. Historical replay remains synchronous and omits live `analysisDelivery` metadata.

`evidenceId` identifies the associated sensor observation, not a unique forecast or diagnostic revision. Multiple forecasts or diagnostic stages can share that ID; consumers should also retain the issue and delivery metadata rather than discarding a response solely because its observation ID matches.

## Issued-weather alignment and rain learning (1.21)

Primary session model `local_session_adaptive_v4_issue_weather_alignment` now selects archived weather/CAMS retrieved **at or before the actual calculation time**, not the older completed sensor-bucket time. Historical replay uses the same issue lag. The PM calculation and displayed session weather identify their retrieval clocks separately; `weatherForecast.pmInputSourceAligned` states whether they agree. Sensor inputs still require completed buckets. A weather update between sensor buckets can update the PM calculation immediately. Cached predictions retain their original issue and target clocks.

Hourly precipitation amounts are integrated over their preceding-hour intervals, weighted by overlap with the exact session. Probability is a separate forecast signal: 96% with 0 mm is not treated as a dry forecast. `particleForecast.rainContext` describes `(issue, session start]`, the session itself, and—where archived—the prior three hours of model context. It reports amount, maximum/mean probability, expected wet duration, first wet interval and expected cessation. Missing coverage remains missing; censored cessation is not a known rain-stop time. These are model-grid weather values, not observed rain at TTDI. Rain probability is **not** the probability of PM clearing.

Two separate experimental models learn signed PM changes from these inputs: `rain_nearterm_ridge_v1` predicts +90-minute PM and the +90–210-minute mean; `local_session_adaptive_v4_rain_issue_clock_experimental` predicts exact morning/afternoon session means. Only completed earlier targets enter training or error-based weights. Their feature interactions allow the response to depend on starting PM, rain before/during the target, timing/duration and associated weather changes. Rain coefficients are not forced negative. Both high-rain clearing and high-rain/no-clearing cases are tested.

**The added rain models did not consistently improve paired retrospective error and are not applied to the primary forecasts.** `rainLearning.appliedToPrimaryForecast = false` is authoritative. Session `rainLearning.trialMeanPm25UgM3` is diagnostic only; its own `trialForecastIssuedEpoch` and `trialWeatherFetchedEpoch` must not be confused with the primary issue. The near-term trial is archived separately with its exact targets, training cutoff, source vintage and empirical errors. No automatic promotion occurs. The main session model uses corrected issued-weather timing and accumulated rainfall with its existing component structure; the near-term primary policy is unchanged. This is a data/implementation correction and a testable learning pipeline, not a demonstrated accuracy upgrade.

The compact `weather.rainLearning` field exposes status and weather context, not an alternative coaching prescription. Trial computations can be reused within the same completed sensor bucket and unchanged weather vintage; they preserve their original timestamps. New weather invalidates the trial cache. Previously published forecasts are not rewritten. The collector now retains four prior hours in newly archived weather vintages for future persistence tests; older missing model context and the laptop-sleep sensor gap are not backfilled.

## Recent-reading reference correction (1.20)

The September 9 update addresses response lag **after** PM2.5 changes, not advance prediction of sudden clearing. When a near-term forecast falls back to persistence, its reference is now the median of finite raw readings in `(issue time - 5 minutes, issue time]`. The completed 15-minute feature bucket is still required; missing recent readings retain the closed reference. No missing readings or future outcomes are filled. Arrival remains exactly issue +90 minutes and the mean remains issue +90 to +210 minutes.

Session model `local_session_adaptive_v3_fresh_reference` keeps the closed-feature fits, issued-weather selection and learned component weights, then applies `max(0, closed model mean + recent sensor reference - closed sensor reference)`. This correction applies throughout the 15-minute bucket, including at bucket close, avoiding a clock-boundary reversal to an older reference. Closed fits cache separately from the refreshed point and evidence. A fresh raw reading can therefore update a forecast without replaying all model fits.

Both the near-term persistence bands and session historical residuals use the same recent-reference policy at the same historical issue lag as the published estimate. They remain uncalibrated empirical errors, not within-session minimum/maximum values. Old closed-reference experimental peak points are not mixed into a refreshed persistence mean; the empirical peak upper bound remains available. Separately gated non-persistence models retain their own estimator and evidence.

`freshnessAdjustment` declares whether the correction was applied, its signed `amountUgM3`, `featureAnchorEpoch`, recent `referenceEpoch`/`referenceCount`, and `closedReferencePm25UgM3`. `persistenceAnchorAt` identifies the actual recent reference when applied; it no longer necessarily equals the closed feature clock. Session `componentEstimatesRole = closed_feature_origin_components_before_fresh_reference_adjustment` and `closedReferenceMeanPm25UgM3` explain the arithmetic. Components and their weights precede the correction. `modelEvidence.persistenceMae` now compares the session policy with the same-issue recent-sensor baseline; `closedMainReferenceMae` separately scores the former closed-reference model. Do not compare these unpaired with older displayed error figures.

This is an **experimental reactive operating policy, not a validated model promotion**. Fixed retrospective session tests at five issue phases showed modest pooled MAE improvements (10.82 to 10.64 before September 8; 9.18 to 9.00 on September 8), but day-level, lead-specific and tail errors were mixed. The cases overlap and September 8 was already inspected. New nonlinear, diurnal and weather-interaction candidates did not provide a sufficiently consistent advantage. None has been promoted. `prospectivelyValidated` remains false.

The dashboard leads with the exact modeled session time (for example, 10:15–12:15), with the broad morning/afternoon window secondary. Revisions with different target intervals are not direct comparisons. The finalized dashboard archive retains each published revision and the new model version; previous issued forecasts are not overwritten. Particle-share descriptions now say what was measured rather than implying that rain caused the change.

## Observed movement versus future prediction (1.19)

`particleNowcast.role = observed_recent_movement_not_future_direction` explicitly separates recent measured movement from a future prediction. For example, `Dry clearing forming` reports a detected recent decline, not a guarantee that the decline will continue to +90 minutes. The dashboard now places that event beside current PM2.5. The arrival `headline` describes the actual estimator: a persistence baseline when no directional model passes its checks, or an experimental estimate when a model is applied. Read the final `pointRole`, `forecastState`, target timestamps and evidence; do not infer a forecast direction from the nowcast label.

The September 8 audit matched frozen displayed forecasts against complete target intervals. It tested three new near-term candidates, two session/CAMS variants and two fresher sensor baselines with September 8 kept separate from earlier selection. No replacement had a sufficiently consistent advantage; point-model formulas and historical forecast archives were retained. Repeated persistence values are therefore an explicit fallback, not a stalled calculation. The collapsed dashboard validation details now distinguish arrival replay, window-mean replay and separately recorded issued-mean outcomes. Overlapping forecasts are not independent events.

## Clock alignment, coverage and replay corrections (1.18)

The near-term targets now use one frozen forecast issue clock: arrival is exactly `forecastIssuedAt + 90 minutes`; the two-hour mean covers `+90` through `+210 minutes`. `persistenceAnchorAt` is the older, completed sensor feature bucket, not the issue time. `basedOnObservedAt` identifies the latest raw observation and does not determine target times. API `expectedAt`/`startAt`/`endAt` and the dashboard use the same target clock. A cached response retains its original target times.

`targetVersion = decision_clock_weighted_15min_v1` declares the aggregation: arrival is an interpolated 15-minute bucket-median proxy, not an instantaneous reading. The mean weights complete bucket medians by overlap with the exact two-hour interval. Peak is the largest median of overlapping buckets, whose edge intervals can extend outside that exact window. It is not the raw sensor maximum. These definitions are also returned in `targetDefinition`, `meanTargetDefinition` and `peakTargetDefinition`.

All forecast origin/outcome buckets require at least three finite PM readings, first and last samples within four minutes of their bucket edges, and no internal sampling gap longer than eight minutes. Missing/unfinished target buckets are not filled or scored. Fresh raw readings can display immediately after resume while forecasts wait for a valid completed anchor. `dataCoverage` reports collection gaps separately from model uncertainty; the September 7 15:40–17:06 gap is user-confirmed laptop sleep, not an environmental event. The server cannot collect while Windows is asleep.

Primary session model `local_session_adaptive_v2` fixes a replay defect: prediction issuance no longer depends on whether future target readings are present. Only scoring/training requires completed outcomes. It retains the component formula, speeds up equivalent Ridge calculations, and invalidates source caches by payload content. `modelEvidence.recentCompleted72Hours` separates recent-target errors from whole-history errors; neither is independent prospective validation. Candidates tested on September 7 did not consistently outperform the retained model. The separate correction trial is versioned `cams_local_residual_change_trial_v2_coverage` because it shares the stricter sensor-frame coverage; old trial issues remain unchanged.

Near-term candidate `ridge_particle_met_decision_clock_v2` begins a new prospective record with explicit target clocks. Older origin-relative issues are not relabelled or combined with it. Legacy rapid-event point/range overlays are withheld when their clocks do not match; the detected clearing remains descriptive context. `clearanceEvent.eventConditionedForecast` may therefore describe a diagnostic estimate without applying it to the published forecast; inspect the final exposure fields.

`dashboard_forecast_issues` privately archives the finalized displayed near-term points/bands, session forecasts, clocks and coverage, with actual publication time and build version. It complements, rather than overwrites, older candidate archives and is not a new public endpoint. Historical replays do not write issued forecasts. Collector waits are resume-aware; browser refresh does not control collection. Windows power settings are unchanged.

## CAMS + learned TTDI correction trial (1.17)

Each Morning/Afternoon `particleForecast.regionalCorrectionTrial` now contains a separate live experimental candidate:

`trial TTDI mean = max(0, CAMS regional session mean + learned local correction)`

The correction starts from the last completed TTDI sensor median minus CAMS at that sensor bucket's centre. A regularized regression learns how this gap changes by the target session, using PM movement, variability, local temperature/humidity changes, time of day and the regional trajectory. It trains only on completed earlier sessions, with each residual measured against the **same CAMS forecast retrieved before that historical origin**, not a later revised forecast. This is a statistical local correction, not another smoke-emission estimate. It does not add a separate Sumatra/fire multiplier to CAMS.

The trial does not alter the primary model. Paired development replay found the tested corrections less accurate overall than the deployed model. The `cams_local_residual_change_trial_v2_coverage` candidate is visible only in collapsed dashboard model details and this separate API field. `appliedToPrimaryForecast`, `prospectivelyValidated` and `eligibleForDirectComparison` are all `false`. Do not substitute `meanPm25UgM3` from the trial for `projectedMeanPm25UgM3`, use it to select a session, or interpret the trial as a training recommendation. There is no automatic promotion.

`regionalCorrection` separates `regionalMeanPm25UgM3`, signed `learnedCorrectionPm25UgM3`, `unboundedMeanPm25UgM3`, and `nonnegativeFloorApplied`. The final nonnegative mean is `meanPm25UgM3`; displayed arithmetic agrees to rounding. Current regional concentration and the current local gap are separate reference fields, not future forecasts. `regionalDataRetrievedAt` is retrieval time, **not CAMS model initialization time**; `sensorAnchorAt` and the trial's `forecastIssuedAt` have distinct meanings. Identical cached predictions retain their calculation timestamp.

Missing, stale or incomplete CAMS inputs make the trial unavailable; they never fabricate a CAMS-labelled persistence result and do not disable an otherwise available primary model. Trial history requires at least 48 completed training examples. Its `modelEvidence` scores only ready, CAMS-covered origins and compares against persistence on those same origins. Do not compare its MAE directly with the primary API's MAE, whose origin set can differ. Historical overlapping replay and the inspected later-period results are development evidence, not untouched validation. New trial predictions are recorded in `window_pm_forecast_issues` under their own model version, independently of open browsers, for subsequent evaluation.

The +90-minute and +90–210-minute exposure forecasts are not changed by this trial. No new public ontology endpoint is provided.

## Primary Morning/Afternoon PM model

Morning/Afternoon now leads with **local predicted PM2.5 session means**, not regional haze direction. `particleForecast.pointRole = experimental_window_mean` identifies an experimental session estimate; inspect its `modelVersion` and `directionalModel` for the scoped 1.23 replacement described above. `projectedMeanPm25UgM3` and `planningEstimate.meanPm25UgM3` carry its estimate. Heat and weather are separate secondary observations/forecasts; no outdoor-training recommendation is returned.

The model predicts the mean of eight completed 15-minute sensor medians over the exact dated 120-minute session. It supports session starts 90 minutes to 24 hours after its last completed sensor bucket; unsupported alignments, insufficient history or stale readings return unavailable, not a fabricated persistence prediction. Each lead is replayed separately. The fixed audit tested 1.5, 6.5, 8.5 and 13.5-hour leads; longer leads remain experimental.

Five bounded components are blended: persistence, half-strength local-history Ridge, half-strength issued-weather/CAMS Ridge, half-CAMS change, and horizon-dependent mean reversion. Ridge fits use completed prior targets from the preceding seven days; adaptive weights use only errors whose full target interval had finished, over the preceding three days. Sensor level, recent movement/variability, recent background, temperature/humidity changes and target time are combined with archived CAMS and weather features. Missing/stale model features do not require a regional forecast to replace the local reading. There is no seasonal haze switch, wind-direction penalty or cyclone-category rule.

`componentWeights`, `componentEstimatesPm25UgM3`, `componentEstimatesRole`, `freshnessAdjustment`, `sourceSnapshots` and `modelEvidence` expose the actual calculation. `persistenceAnchorAt` identifies the recent reference or closed fallback, **not the forecast target**; `forecastIssuedAt` is when this cached forecast was calculated. In version 1.21, model snapshots are selected at or before the actual issue time, while sensor features still use the closed origin. Live predictions are logged separately in `window_pm_forecast_issues`, with their actual publication times, even when all browsers are closed.

The model and its ordering are experimental: `validatedModelPointApplied = false` and `planningEstimate.eligibleForDirectComparison = false`. `rideWindows.comparison.pm.differenceMorningMinusAfternoonUgM3` reports the numeric difference; `lowerExperimentalScenarioWindow` merely names its sign, not a validated preference. PM availability and same-day comparison no longer depend on weather availability. No peak forecast is inferred from this mean-only model.

`modelEvidence` is an exact-lead prequential historical replay, not a record of previously issued predictions. Counts overlap and model selection used this short history. Later sensor corrections cannot be reconstructed without ingestion timestamps. Empirical q10–q90 residual spans around the new estimate are uncalibrated **mean forecast errors**, not the minimum and maximum expected during a session. They remain collapsed on the dashboard. Broader improvements over persistence did not hold consistently through the latest rise; do not claim established accuracy or reliable morning/afternoon ranking.

The local +90-minute and +90–210-minute models remain separately defined; the new session model does not replace those models. The legacy CAMS transform and `hazeOutlook` remain optional diagnostic data, not the primary session result. `provenance.particleForecast` describes the new session model; `legacyCamsValidation` preserves the older transform's separate evidence. Regional transport is no longer a permanent dashboard tile.

## Contract discovery and refresh

The evidence response is self-describing. A Coach integration needs only the evidence URL above; it does not need filesystem access, a pasted copy of this file, or GitHub as a runtime dependency. Its top-level `contract` object advertises:

- `openapi`: `/api/openapi.json` — the OpenAPI 3.1 service description.
- `jsonSchema`: `/api/v1/mtb/environment-evidence/schema` — the JSON Schema 2020-12 evidence contract.
- `documentation`: `/docs/coach-api.md` — this living integration guide, served directly from `COACH_API.md`.
- `evidencePollSeconds`: the normal passive evidence polling interval.
- `revisionRole = "contract_and_documentation_only"`: identifies the three contract resources (schema, service description, and guide) and prevents their revision from being mistaken for live evidence freshness.
- `refreshPolicy.resources`: the three contract resources that support `If-None-Match`; the evidence endpoint itself does not.
- `schemaVersion` and `revision`: change indicators that tell the consumer when its cached contract should be revalidated.

Resolve these root-relative URLs against the origin used for the evidence request. `contract.schemaVersion` equals the response `schemaVersion`: a major-version change is incompatible, while a same-major consumer should treat non-required fields as optional, ignore unknown fields, and fetch the current schema whenever the version changes. `contract.revision` is an opaque equality token for the published schema, service description, and guide; it is not an evidence version, observation ID, or freshness timestamp.

Cache each advertised contract resource together with its HTTP `ETag`. On first contact, or whenever `contract.schemaVersion` or `contract.revision` changes, revalidate the cached resources with `If-None-Match`. Keep the cached copy after HTTP `304`; replace it after HTTP `200`. `Cache-Control: no-cache` permits storage but requires revalidation. The dynamic evidence endpoint remains `Cache-Control: no-store` and intentionally has no ETag.

The evidence response also advertises the same resources through HTTP `Link` relations: `service-desc` for OpenAPI, `describedby` for JSON Schema, and `help` for this guide. Polling evidence is passive; a coaching client should not call the dashboard-internal `/api/refresh` route.

## Status semantics

- HTTP `200`, `status.state = "ok"`: current observation and analysis are available.
- HTTP `200`, `status.state = "degraded"`: usable evidence remains, but `status.issues` identifies a forecast older than 120 seconds, an analysis update error, or a stale/unavailable supporting source.
- HTTP `503`, `status.state = "unavailable"`: no completed analysis is available, its original issue is more than 600 seconds old, or the particle observation is missing or more than 15 minutes old.
- HTTP `500`, `status.state = "unavailable"`: evidence generation failed; inspect `status.issues` and retry at the normal cadence.
- `evidenceQuality` is separate from operational status. For example, a fresh response can still have a low-confidence particle outlook while local history is accumulating.

After HTTP `503`, use `provenance.analysisDelivery.retryAfterSeconds` when supplied (5 seconds while an analysis is initializing); otherwise retry no sooner than `contract.evidencePollSeconds`. The Coach endpoint is read-only and does not trigger source collection.

The consumer should retain `evidenceId`, `generatedAt`, `observation.timestamp`, `provenance.analysisForecastIssuedAt`, `provenance.analysisDelivery`, and `status.state` with any downstream record that needs later provenance.

## Main fields

- `observation`: current PM2.5, PM10, heat index, temperature, humidity, and freshness.
- `particleNowcast`: recent particle regime, 30/60-minute movement, sustained-improvement flag, event minimum, and analysis refresh interval.
- `exposureOutlook.arrival`: a replay-screened local-analogue estimate for conditions after the 90-minute trip, alongside the separate sensor-persistence anchor and empirical historical-error bounds. It can alter the point only when the exact target clears at least 96 hourly origins across five dates and 24 horizon-separated origins, with at least 5% MAE improvement and a 55% win fraction in both samples.
- `exposureOutlook.onTrail`: mean and peak PM2.5 for the modeled 120-minute trail interval. Mean and peak are screened independently, so the response can correctly contain a persistence mean together with an experimental bounded-analogue peak. A strongly shrunk robust-Ridge mean is frozen as a prospective shadow candidate and cannot replace persistence from development-history replay alone.
- `projected...Pm25UgM3` reports the point used for that target; inspect `meanValidationState` and `peakValidationState` independently on `onTrail`. `persistenceAnchorRole` is authoritative and `persistenceAnchorAt` gives its timestamp: the generic anchor is the latest completed wall-clock 15-minute median, while a confirmed live event overlay may explicitly use the latest raw sensor reading. `modelFeatureAnchorPm25UgM3` separately reports the closed feature bucket used by the local model. `approximate = true` identifies an experimental mean; `validationState = "mixed_persistence_mean_experimental_peak"` identifies the peak-only mixed state.
- The `likely...Range` fields and `upperPm25UgM3`/`upperMeanPm25UgM3`/`upperPeakPm25UgM3` are raw finite-sample persistence-error quantiles with `calibrated = false`. `forecastRangePm25UgM3` and `forecastMeanRangePm25UgM3` are display envelopes that retain or widen those historical endpoints to contain the persistence anchor and any experimental projection. They deliberately make no quantile or coverage claim. `forecastUpperPm25UgM3` and `forecastPeakUpperPm25UgM3` are the corresponding neutral anchor/scenario-containing high-side references.
- Each exposure `support` distinguishes all eligible origins from the matched origins that actually form its band; `matchedIndependentOriginCount` applies the full forecast horizon.
- `finiteSampleRankCoverage` reports the order-statistic coverage implied by the current matched sample. `calibrated` remains `false` while local history is limited.
- `weather.trailPeriod`: apparent heat, humidity, cloud cover, precipitation probability/amount, and modeled surface/aloft airflow for the same future interval. These fields are neutral weather context and do not alter the particle forecast.
- `weather.regionalTransport`: compact shadow-only regional-wind evidence. `current` reports the issued 925 hPa wind direction, speed, source alignment, and signed source component; `evidence` reports issued-day/origin counts, completed-pair and prospective-scored-origin counts for the fixed 12/18/24-hour lags, plus the validation summary. `usedForParticleForecast = false` means none of these fields alters a PM2.5 point, range, or Morning/Afternoon comparison.
- `weather.subangReference`: corroborating observed temperature, humidity, wind, and visibility from MET Malaysia Subang. It is not an input to PM2.5 point or range forecasts.
- `clearanceEvent`: forming or confirmed dry clearing, rain-consistent washout, rebound timing, or no recent event. `detectionMode = "joint_shock"` is an early nowcast watch; the stronger `"rapid_joint"` or `"rapid_and_sustained"` gate can activate `eventConditionedForecast` only when the current trigger lies inside the prior trigger-level range and at least two post-trigger sensor samples have not rebounded. The event forecast is scale-normalized with `prior outcome / prior event origin`, then faded toward persistence as the signal ages. `lastEvidenceAt` distinguishes continuing evidence from the original onset.
- `rideWindows.comparison`: a descriptive Morning/Afternoon PM and weather summary. `role = "descriptive_air_and_weather_outlook"` and `prescriptiveRecommendation = false` mean the service reports forecast evidence without choosing a ride window. The `pm` object states whether the two modeled particle estimates are eligible for direct comparison; the `weather` object reports each window independently.
- `rideWindows.morning` and `rideWindows.afternoon`: dated window forecasts containing `logisticsReferenceAt`/`logisticsReferenceLabel`, `particleForecast`, `weatherForecast`, `forecastIssuedAt`/`forecastedLabel`, and secondary `observedHistory`; raw history is not returned. The logistics reference is fixed at 90 minutes before the modeled ride, while issuance time only describes when the displayed forecast was calculated.
- `particleForecast.planningEstimate` is the model estimate shown for planning. `available = false` means future PM is unresolved and the persistence anchor is only a current reference. `eligibleForDirectComparison` states whether the estimate can be compared with the other window under the tested model task.
- `particleForecast.uncertainty.typicalMatchedHistoryBandPm25UgM3` is the middle half of matched historical outcomes. `conservativeMatchedHistorySpanPm25UgM3` is the wider 10th–90th historical span. `empiricalPeakQ90Pm25UgM3` carries the raw empirical peak q90; `highSidePeakMarkerPm25UgM3` can be widened to contain the persistence anchor and therefore has `highSidePeakQuantile = null`. These are descriptive and uncalibrated, not prediction intervals, and remain inside the collapsed uncertainty explanation rather than the primary Morning/Afternoon line.
- Each window models a comparable 120-minute session that fits inside 09:00–13:00 or 14:00–18:00 after the 90-minute trip. PM target epochs are generated independently of weather availability. Since 1.16, the adaptive exact-session model owns these means at supported leads; the older near-term bridge/CAMS transform is diagnostic only. The 1.17 correction trial uses the same session targets without changing the primary estimate.
- Older CAMS roles such as `provisional_cams_context`, `aggressive_cams_forecast`, and `validated_forecast` belong to the retained legacy transformation, not the current adaptive model or the new correction trial. Its task-specific comparison intervals and validation must not be transferred to either newer model. The current primary role is `experimental_window_mean`; inspect the current `modelVersion` and `modelEvidence` instead of interpreting legacy validation as approval of the trial.
- `evidenceQuality.limitations`: machine-readable reasons statistical confidence is limited.
- `evidenceQuality.localMeanIssuedValidation`: measured error of the frozen Ridge mean candidate from forecasts recorded before their outcomes. It includes scored/independent counts, dates, mean absolute error, large-error metrics and the promotion result. At least seven scored dates and 20 non-overlapping windows, a 5% and 1 µg/m³ overall MAE improvement, independent improvement, stable daily gains, and no worsening of RMSE or 90th-percentile error are required in addition to the existing replay checks.
- `provenance.analysisComputedAt`: completion time of the shared primary analysis. It stays unchanged while that snapshot is served or enriched with diagnostics. `analysisForecastIssuedAt` preserves its issue; `analysisDelivery` reports response-time age, updating and diagnostic state. `generatedAt` is response time, and `observation.timestamp` remains authoritative for sensor freshness. Historical replays bypass live delivery.
- `provenance`: AirGradient, Open-Meteo weather, Open-Meteo with Copernicus Atmosphere Monitoring Service (CAMS) Global particles, and MET Malaysia source/freshness information.
- `boundary`: confirms that the response is environmental evidence only.

The local analogue arrival/trail forecast, rapid-clearance event overlay, and operational-lead CAMS candidate use a responsive environmental-forecast configuration. Generic candidates are screened target by target against persistence and fall back to persistence whenever the exact estimator does not clear both the hourly and horizon-separated replay tests. Model training, targets, forecast anchoring, and replay use fully completed wall-clock 15-minute buckets; a latest raw sample changes the nowcast state but does not silently recenter a generic future mean. The event overlay uses only completed prior events whose full +210-minute outcome was already known, takes at most one event per local day, requires at least two distinct event-days, rejects trigger levels outside prior support, and cancels on a strong post-trigger rebound. Its ratio normalization prevents a large absolute drop from a high-PM event being copied onto a lower-PM event. These point estimates remain experimental; matched-history error evidence stays separate from the model estimate, and prospective scoring continues.

For 1.x compatibility, `historicalBaseline` remains as a deprecated alias of `observedHistory`; `exposureOutlook.arrival.baselinePm25UgM3` and `exposureOutlook.onTrail.baselineMeanPm25UgM3` remain deprecated aliases of their persistence anchors. `decisionAt`/`decisionLabel` remain deprecated aliases of `logisticsReferenceAt`/`logisticsReferenceLabel`. The deprecated exposure fields `decisionEnvelopePm25UgM3`, `decisionMeanEnvelopePm25UgM3`, `decisionUpperPm25UgM3`, and `decisionPeakRiskMarkerPm25UgM3` remain aliases of the neutral `forecast...` fields. The deprecated ride-window `recheck` field mirrors `forecastedLabel` and contains no future action time. `particleForecast.meanRangePm25UgM3` remains as a deprecated alias of the uncalibrated 10th–90th matched-history span; its `nominalCoverage` is intentionally `null`. `historicalLowerExposureWindow` is not emitted and is not part of the schema; consumers must not infer a preferred window from historical data. Version 1.10 removed the earlier preferred-window and weather-veto fields; consumers should read the neutral `rideWindows.comparison.pm` and `.weather` evidence instead. Version 1.11 adds separate mean/peak validation states, compact regional-transport shadow evidence, neutral forecast/logistics aliases, and corrected interval semantics. Version 1.12 adds closed-bin causal replay, exact peak-only validation, and stronger event-support/rebound checks. Version 1.12.1 removes the unintended public scientific-model endpoint. Version 1.13 separates raw, partial and completed anchors; prevents peak-only activation from moving the mean; adds anchor timestamps; prevents persistence from masquerading as a ride-window model projection; and bridges aligned Morning/Afternoon sessions to the local +90-to-210-minute forecast.

## Python example

Version 1.14 accelerates equivalent numeric calculations and reuses historical fits only when their causal inputs are identical. A background worker maintains analyses and records local forecasts even with every browser closed. The mean model's fixed promotion rule now consumes those issued outcomes instead of remaining permanently disabled. Public scientific-model/ontology endpoints remain absent.

### Regional haze outlook (1.15)

`rideWindows.morning.hazeOutlook` and `rideWindows.afternoon.hazeOutlook` add the CAMS regional particle trajectory and simplified, time-varying air paths. They are separate from each window's `particleForecast` and do not change the local +90-minute or +90–210-minute concentration calculations.

- `regionalMeanPm25UgM3` and `regionalCurrentPm25UgM3` are **regional model values, not TTDI measurements or local predictions**. `regionalChangePct` compares the modeled session mean with the same issued model's current value. The direction rule requires at least 3 µg/m³ or 10% movement; that is a display-resolution rule, not a skill test or health threshold.
- `role = regional_model_outlook_not_local_concentration`, `usedForLocalPmPoint = false`, and `validatedAtTTDI = false` are explicit interpretation boundaries. Do not substitute these regional values into local exposure or training calculations.
- `airPath` traces air backwards from session start, midpoint and end, through a 30-point NOAA GFS wind grid, in half-hour steps for up to 48 hours. 925 hPa is the main level (approximately 2,500 ft; actual model height is provided); 850 hPa is a height-sensitivity check. Directions are meteorological **from** directions and are interpolated as vectors.
- `transitHoursMin/Max` are times since first contact with a coarse Sumatra land sector on those modeled paths. They are **not** countdowns to haze, validated travel-time bounds, probabilities, or proof that smoke was present. An absent crossing, truncated path, or change of wind does not establish clean air. `heightSensitive` marks disagreement in sector crossing between levels. The paths omit vertical motion, dispersion and deposition.
- `surfaceCoupling` describes modeled boundary-layer depth relative to the elevated wind level. A growing layer can entrain elevated smoke or dilute surface pollution; no one-way PM adjustment is applied.
- CAMS already represents atmospheric transport/removal and uses GFAS fire-emission estimates. No independent hotspot feed is collected here. Cyclone intensity, calendar seasons and the air-path diagnostic do not add a second PM multiplier. `sourceAttributionEstablished` remains false.
- `sourceFetchedAt`, `forecastIssuedAt` and `airPath.fetchedAt` have separate meanings. Regional winds refresh every 3 hours and expire after 6 hours; CAMS expires after 2 hours without a successful retrieval. A recent retrieval is not a new native model run. Regional snapshots are compressed and archived for 90 days; replay never uses a snapshot retrieved after its issue time. Retrospectively retrieved wind data is excluded from the issued archive.

In version 1.15, `rideWindows.comparison.regionalHaze` compared regional session means. Version 1.16 restores the local PM model comparison as the headline and no longer emits that regional comparison override. Optional `hazeOutlook` remains regional-only context. No ride is selected. Older `weather.transport` still refers to the separate, unpromoted local-wind regression, not the air-path diagnostic.

CAMS validation now includes compatible `cams_anchor_v1` and `cams_anchor_v2` source archives. Its `causal_issued_model_archive_replay` mode means the current local transformation is replayed against weather-model snapshots actually retrieved before each issue time; it is distinct from the Ridge candidate's record of local predictions actually issued. Archived forecasts fetched after an issue are never used for that issue.

```python
import requests

url = "http://127.0.0.1:8765/api/v1/mtb/environment-evidence"
response = requests.get(url, timeout=10)
evidence = response.json()

if not evidence["status"]["usable"]:
    environmental_state = "unavailable"
else:
    environmental_state = {
        "evidence_id": evidence["evidenceId"],
        "observed_at": evidence["observation"]["timestamp"],
        "pm25_ug_m3": evidence["observation"]["particles"]["pm25UgM3"],
        "particle_state": evidence["particleNowcast"]["state"],
        "arrival": evidence["exposureOutlook"]["arrival"],
        "on_trail": evidence["exposureOutlook"]["onTrail"],
        "trail_weather": evidence["weather"]["trailPeriod"],
        "regional_transport": evidence["weather"].get("regionalTransport"),
        "limitations": evidence["evidenceQuality"]["limitations"],
    }
```

The integration should store the last seen `contract.revision` separately from evidence provenance. Continue retaining `evidenceId`, `generatedAt`, and `observation.timestamp` with the downstream record; those fields describe the evidence itself, while the contract revision describes how to interpret it.

No CORS header is emitted by default. This is intentional: a Python/server-side client can call the endpoint directly, while arbitrary webpages cannot read the LAN service through a browser. CORS is not an authentication mechanism.
