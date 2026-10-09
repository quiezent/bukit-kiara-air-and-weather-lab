# LAN data contract for the ESP32 dashboard

The **31-public.1** board firmware consumes **RLCD schema version 1** over HTTP, with unchanged native arrival-change and current-status fields. V31 changes only the reading footer and version; v30 added local readout-stop controls and removed the displayed/spoken **Observed:** prefix. The [package README](../README.md#v31-verification-9-october-2026) scopes current verification and preserves the historical v30 results. Its public eSpeak NG voice remains unflashed. The [v29 recovery record](VERIFICATION_AUDIO_RECOVERY_2026_10_09.md), [v28 observed-status record](VERIFICATION_OBSERVED_2026_10_09.md), [v27 indoor record](VERIFICATION_INDOOR_2026_10_09.md) and [v26 arrival record](VERIFICATION_2026_10_09.md) remain historical evidence. The example [rlcd-v1-synthetic.json](../examples/rlcd-v1-synthetic.json) uses invented values. [demo_server.py](../tools/demo_server.py) serves that fixture with coherent fresh clocks; it is not a sensor, weather service or trained forecast model.

## Discovery and transport

Advertise DNS-SD service **`_ttdi-weather._tcp.local.`** with an IPv4 address and port, plus these TXT records:

| TXT key | Value |
| --- | --- |
| `schema_version` | `1` |
| `api_path` | `/api/rlcd/v1`, or another path beginning `/api/` |

The board browses on Wi-Fi connection, periodically and after failed requests. It can re-resolve a previously advertised hostname when a browse fails. It polls about every 60 seconds, retries failed requests after 15 seconds and rediscovers periodically. Advertise the server's current LAN interface rather than assuming its DHCP address remains fixed. mDNS multicast must reach both devices on the same LAN.

Serve `GET /api/rlcd/v1` with HTTP **200**, uncompressed, unchunked UTF-8 JSON and an explicit **Content-Length** of **1–16,384 bytes**. The client uses HTTP/1.0, rejects missing/nonpositive or oversized reported lengths, and checks the received byte count. Body size is measured in UTF-8 bytes, not characters. The synthetic server sends `Content-Length` and keeps its body below this limit.

The acceptance check requires `schema_version: 1`, `generated_epoch` later than 1,700,000,000, and object-valued `current`, `weather` and `forecast`. Emit `timezone: "Asia/Kuala_Lumpur"` and use UTC Unix seconds for all epochs; the board's Malaysia display/session clocks use UTC+8. JSON nesting must fit the firmware's limit of 12 levels. Use JSON `null` and availability flags for missing measurements, not zero, NaN or invented replacements. Preserve the original observation, fetch, issue, reference and target times when returning cached data.

`demo: true` activates the public variant's visible **DEMO WEATHER** heading and spoken **Demonstration data** prefix. Its synthetic server also advertises `demo=true` and returns an `X-RLCD-Synthetic-Demo` header; the board's marker comes from the JSON field.

## Preserve the native forecast questions

| Quantity | API location | Native meaning |
| --- | --- | --- |
| First substantial change | `forecast.near90.first20` | Which first ≥20 µg/m³ crossing occurs within the next 90 minutes: rise, drop or neither? |
| Arrival estimate | `forecast.near90.pm25_ugm3` | Primary model's concentration estimate at `target_epoch`, normally issue +90 minutes |
| Arrival change | `forecast.near90.arrival_change` | Native probabilities of concentration change at the exact +90-minute arrival target, relative to the original fresh reference |
| Ride context | `forecast.ride90_210` | Two-hour window from issue +90 to +210 minutes: mean, extrema, weather and chance its mean is ≤70 |

A first crossing can reverse before arrival. Its probabilities are not arrival-change probabilities, and a concentration estimate is not a crossing threshold. The board consumes the server's native arrival-change head; it does not manufacture that head by subtracting point estimates, deriving a probability from first crossings or repurposing ride-window probabilities. If arrival data is unavailable, Page 1 says so rather than substituting `first20`.

The concentration estimate, original reference, within-window extrema, first-crossing probabilities and arrival-change probabilities each retain their native units and target clocks. All values in the synthetic example are illustrative; no prediction model is evaluated by the demonstration server.

## Field semantics

| Object | Fields used by the device |
| --- | --- |
| `current` | `available`, `fresh`, `observed_epoch`, `pm25_ugm3`, `temperature_c`, `humidity_pct`, `display_text` |
| `weather` | `available`, `fresh`, `fetched_epoch`, `current_hour`, `next_hours` |
| `forecast` | `available`, `fresh`, `issued_epoch`, `near90`, `ride90_210`, `sessions` |
| `tennis_morning` | Dated 07:00–09:00 weather window with availability, fetch clock, rain, feels-like temperature and wind |
| `history` | Availability/freshness, source/window epochs, four-column points, collection gaps and full-source PM summary |

Primary PM values require `available: true`, a finite nonnegative `pm25_ugm3` within the display range, and role `raw_model_output`, `experimental_model_output` or legacy `experimental_window_mean`. The device never substitutes `reference_pm25_ugm3` for a missing primary model value. Qualification/calibration metadata stays distinct from numerical output; an eligible raw estimate is not treated as an operational recommendation.

`current.display_text` is the server's nonempty dynamic status for the current sensor concentration. Page 1 removes only its leading `Observed: ` prefix for display, preserving the status body and any old-data marker; the API value remains unchanged. Missing status text is not replaced by a fabricated classification, and invalid/expired current data stays unavailable. This text is independent of the arrival headline.

### Observed-status speech

Page 1 captures `current.display_text` when building its readout snapshot. It speaks the corresponding recorded status immediately **after current outdoor PM2.5 and before outdoor temperature**. Exact case, punctuation and UTF-8 spelling select this finite bank:

| Exact API `current.display_text` | Spoken words |
| --- | --- |
| `Observed: Fast PM2.5 rise detected` | Fast PM2.5 rise detected |
| `Observed: Fast PM2.5 rebound detected` | Fast PM2.5 rebound detected |
| `Observed: Particle rebound may be starting` | Particle rebound may be starting |
| `Observed: Recent PM2.5 medians at or below 35 µg/m³` | Recent PM2.5 medians at or below thirty five micrograms per cubic metre |
| `Observed: Moist-cooling particle clearing forming` | Moist cooling particle clearing forming |
| `Observed: Dry clearing forming` | Dry clearing forming |
| `Observed: PM2.5 reduction forming` | PM2.5 reduction forming |
| `Observed: Rapid PM2.5 reduction detected` | Rapid PM2.5 reduction detected |
| `Latest sensor reading` | Latest sensor reading |
| Unknown, missing, null, empty or nonstring text with otherwise eligible current data | Observed status unavailable |

PM2.5 is pronounced **P M two point five**. V30 speaks the matching fixed body without the recorded **Observed** prefix; the existing audio bank and exact API matching are unchanged. The neutral and unavailable phrases stand alone. There is no runtime arbitrary-text synthesis or derived movement classification. Changing API wording without adding a matching clip causes the unavailable fallback rather than a transcript without corresponding audio.

The status requires the same eligible current observation/concentration as the display, including a valid source epoch and PM value. Expired or future observations, unavailable current data and invalid PM suppress it entirely. Eligible old data retains the single **Outdoor data is old** announcement before current PM, without repeating it for the status. The existing seven-minute old / fifteen-minute expiry rules below are unchanged. Text, clip IDs and values are captured together; later server polling cannot change a running narration.

### Native arrival-change headline and probability

`near90.arrival_change` supplies `available`, `issued_epoch`, `arrival_epoch`, `fresh_reference_epoch`, `reference_ugm3`, `fall20`, `fall40`, `rise20`, `rise40`, `within20`, `outcome`, `outcome_probability` and nonempty `display_text`. All probabilities are native fractions in **[0,1]**. The ≥40 tails are subsets of their corresponding ≥20 tails: `fall40 <= fall20` and `rise40 <= rise20`. The mutually exclusive coarse classes `fall20`, `rise20` and `within20` must sum to one; do not sum all five fields as though the ≥40 subsets were additional classes. `within20` is supplied by the server, not reconstructed on the board.

For a unique largest coarse-class probability, native `outcome` must be `fall20`, `rise20` or `within20`, and `outcome_probability` must match that class's probability. Exact ties require **both fields to be JSON null**. Winner comparisons are exact; coherence tolerances do not turn a unique winner into a tie or normalize its score. The native five-class model meaning and subset tails are retained even though the current card summarizes the three coarse outcomes.

The head's `issued_epoch` must equal the parent forecast issue. `arrival_epoch` must equal both issue **+5,400 seconds** and `near90.target_epoch`, and remain in the future. The issue cannot be future-dated. `fresh_reference_epoch` must be no later than the issue and at most **240 seconds** older; preserve this original reference rather than refreshing it with later observations. Native `reference_ugm3` must be valid. It must agree with `near90.reference_pm25_ugm3` only when that point-reference field is nonnull; an absent or explicitly null point reference does not invalidate an independent arrival head with its own valid native reference. Parent forecast availability and expiry still apply.

The screen uses the native `display_text` plus the validated `outcome_probability × 100`, rounded only for display. A tie keeps the native headline and shows no percentage. Speech uses finite clips for **Fall on arrival**, **Rise on arrival** or **No change on arrival**, followed by that same native winner percentage; `within20` describes its concentration band, not an assertion of exactly zero change. A tie says **Arrival outcome uncertain** without a percentage. Missing, inconsistent or expired arrival data is unavailable; it never falls back to `first20`. The arrival head can remain valid when the separately available +90-minute numerical estimate is missing.

### First-change model summary

`first20` provides `available`, `rise_probability`, `drop_probability`, `none_probability`, `reference_ugm3`, `diagnostic_direction`, `direction` and `display_text`. Probabilities are fractions in **[0,1]**, not percentages, and must sum to one within the classifier's tolerance. `diagnostic_direction` describes the raw unique winner; use `rise`, `drop` or `unresolved`. A unique neither winner and ties both use `unresolved`. Operational `direction` can remain unresolved independently of the raw winner.

This head remains a separate first-crossing target. The Page 1 arrival card and readout do not use it as their arrival outcome. The retained classifier validates raw probabilities and diagnostic direction; a tie must not claim that no ≥20 crossing is most likely. When diagnostic metadata is absent, it supports legacy `direction` semantics. Preserve `display_text` and structured fields consistently for clients that present this separate target.

### Ride range, mean and probability

`ride90_210.start_epoch` and `end_epoch` define a valid two-hour window. Its new `minimum_maximum` head must have:

```json
{
  "available": true,
  "minimum_ugm3": 31.4,
  "maximum_ugm3": 53.8,
  "resolution_minutes": 15,
  "kind": "predicted_window_minimum_maximum"
}
```

These are predicted within-window extrema across 15-minute periods, not measured minima/maxima or confidence bounds on the two-hour mean. Limits must be finite, ordered and within the display range. If this modern object is explicitly null, unavailable or invalid, the device cannot revive legacy quantiles as extrema. It can instead show a valid primary mean labelled **MEAN**.

Legacy payloads omitting `minimum_maximum` can retain `range_kind: "empirical_q10_q90"` with `range_low_ugm3`/`range_high_ugm3`; those limits describe quantiles for the window mean. `mean_le70` independently supplies `available` and fractional `probability`. The value 70 is a planning cutoff, not a health/safety threshold. Mean, extrema, mean-≤70 probability and first-change predictions have independent availability; a warming-up mean model need not suppress available extrema or probability heads.

### Sessions and window weather

`forecast.sessions.morning` and `.afternoon` contain `issued_epoch`, `start_epoch`, `end_epoch`, primary `pm` and `weather`. The example uses 09:00–11:00 and 14:00–16:00 Malaysia time. Speech compares valid primary values at displayed one-decimal precision only when the forecast is fresh, session issues match it and the windows remain valid. It speaks this/tomorrow morning/afternoon from the actual dates; expired or unsupported dates are not silently relabelled. `tennis_morning` is a separate 07:00–09:00 weather-only window; Page 1 labels its displayed wind value in **km/h**.

Modern ride/session weather must provide `coverage_verified: true` and `coverage.complete: true`, plus `available`, `fresh`, `fetched_epoch`, `rain_chance_max_pct`, `rain_mm`, `feels_like_max_c`, `humidity_mean_pct`, `wind_mean_kmh` and `gust_max_kmh`. It must cover the complete target window. Incomplete coverage suppresses the window's weather/readout warnings rather than presenting a partial summary as complete. Genuine legacy payloads without either coverage field retain their earlier availability behavior. Fetch clocks must not postdate the forecast issue.

Hourly rain uses `rain_start_epoch`/`rain_end_epoch` one-hour intervals and `rain_chance_pct` in **[0,100]**. A provider's `current_hour` can describe the preceding hour. Speech selects the interval containing now and the interval containing now +1 hour, rather than trusting an array position. Rain warnings are issued for valid ride/session probabilities strictly **above 75%**.

### Observations, history and stale data

History columns are exactly `["epoch", "pm25_ugm3", "temperature_c", "heat_index_c"]`; row values may be null. The firmware accepts up to **128 points**, within the unchanged **16 KiB complete-response budget**. When the original window contains at most 128 source rows, retain every row that fits the response budget. For larger windows, select balanced original rows preserving the endpoints and extrema of each plotted trace—PM2.5, temperature and heat index—rather than optimizing only the PM curve. Do not interpolate invented observations.

Supply `gap_kind: "source_collection_before_downsampling"` and a `gaps` array describing original collection gaps, including `[]` when collection is known to be continuous. This marker lets the graph distinguish true collection gaps from spacing introduced by thinning. The firmware consumes only the first **128 gap pairs**; keep the published gap description within that limit. `pm25_summary` contains `available`, `average_ugm3`, `lowest_ugm3`, `highest_ugm3` and `sample_count`, computed from all accepted original observations in the same displayed window. The summary and original gap metadata remain based on the unthinned source even when graph rows are reduced; do not recalculate them from the retained graph rows.

Outdoor observations are marked old after seven minutes (or `fresh: false`) and unavailable after 15 minutes. PM forecasts are marked old after two minutes and unavailable after ten. Modeled weather is marked old after 30 minutes and unavailable after two hours. Original clocks, null values and invalid fields control these states; a new HTTP receipt must not make old content fresh. Historical graph data can remain visible with its original dates and **OLD** markers.

Indoor measurements and battery voltage come from the ESP32 board, not this API. Battery percentage is a coarse voltage-derived estimate. The firmware snapshots the selected page's values before playback so later polling/navigation cannot alter a readout mid-sentence.

## Local board indoor correction

This setting belongs to the **ESP32's own HTTP service**, separate from the weather producer's `/api/rlcd/v1`. Send `POST /indoor-correction` to the board with `Content-Type: application/x-www-form-urlencoded` and this form field:

```text
temperature_offset_c=-4.0
```

For example, replace `BOARD_HOST` with the board's current hostname or LAN address:

```sh
curl -X POST "http://BOARD_HOST/indoor-correction" --data "temperature_offset_c=-4.0"
```

The field must parse completely as a finite number in **[−10,+10] °C**; a JSON body is not accepted by this route. Success returns HTTP **200** with `{"saved":true,"temperature_offset_c":-4.00}`. Missing/invalid fields return **400**; a failed NVS save returns **500** without applying the requested change. Accepted settings persist across restarts. The current firmware uses the user-selected **−4.0 °C** fallback when no valid saved setting exists; flashing preserves an existing saved offset. The initial v27 −4.8 °C value and informal comparison evidence remain in the [dated v27 record](VERIFICATION_INDOOR_2026_10_09.md). Select your own offset using a co-located reference under the board's normal operating conditions; this estimate does not replace the SHTC3's factory calibration or establish traceable system calibration.

Let `Traw` and `RHraw` be the manufacturer's converted sensor readings, and `offset` the configured temperature offset. The original firmware helper applies equation 1 of Sensirion's [Design Guide for Humidity and Temperature Sensors](https://sensirion.com/resource/user_guide/sht/design_in/):

```text
Tcorrected = Traw + offset
RHcorrected = RHraw × exp(17.62 × 243.21 × (Traw − Tcorrected)
                         / ((243.21 + Traw) × (243.21 + Tcorrected)))
```

Temperatures are in °C and RH is in percent. RH is limited to **0–100%** after compensation; no independent RH offset is fitted. Offset **0** preserves the exact original temperature and RH. The board's `GET /status` exposes these fields within `indoor`:

| Field | Meaning |
| --- | --- |
| `temperature_c`, `humidity_pct` | Corrected readings used for indoor displays and speech |
| `raw_temperature_c`, `raw_humidity_pct` | Original converted sensor measurements |
| `temperature_offset_c` | Active temperature offset |
| `humidity_temperature_compensated` | Whether the active offset is nonzero |
| `humidity_clamped` | Whether compensation required limiting RH to 0–100% |
| `correction_method` | `board-heat-magnus-v1` |
| `correction_is_estimate` | `true`: this configured system correction is an estimate |
| `correction_saved` | Whether NVS holds the active offset |

Unavailable measurements are returned as JSON `null`. Indoor values and their ranges on every page and in spoken summaries use the corrected readings. Outdoor observations, forecasts and history are unaffected.

Changing the offset resets the corrected temperature and RH minima/maxima. Applying a setting does not take a new measurement or refresh the sample clock; a stale or invalid retained sample remains unavailable until a valid reading arrives.

## Local board readout control

Release physical **KEY** to start reading the selected page, or to request a stop while reading. The newline-terminated USB command **`KEY`** invokes exactly the same toggle; it is not a weather-producer API command. During playback the footer shows **[KEY] Stop Reading** on the left, **WiFi: RSSI** in the center and **[BOOT] Next Page** on the right. There is no **Voice: reading** indicator.

The board's own `GET /status` adds these fields within `readout`:

| Field | Meaning |
| --- | --- |
| `stopped` | Count of intentional stops handled during readout cleanup, separate from natural `completed` streams |
| `stop_requested` | Whether the active readout has a stop request; false after cleanup |

A stop mutes playback and lets the player release the speaker and resume microphone capture through the existing recovery/quiet-tail handoff. Voice read commands and BOOT navigation are unchanged.

## Local board audio diagnostics

These fields belong to the **board's own `GET /status`**, within `voice`; they are not new fields in the weather producer's `/api/rlcd/v1`. V29 adds bounded runtime recovery and diagnostics while retaining the existing speech models, dictionary, 0.80 threshold and recordings. Public build/host and installed-device recovery checks passed, without establishing hours-long reliability or new human listening evidence.

| Field | Meaning |
| --- | --- |
| `ready`, `error` | Current recognition readiness and error |
| `recovery_pending` | Whether a runtime recovery is pending |
| `recovery_count`, `recovery_attempts`, `recovery_failures` | Completed recoveries, attempts and failed attempts |
| `feed_parked`, `detect_parked` | Both acknowledgements are required for pipeline reset; speaker handoff needs feed parking/pause and microphone-channel release |
| `last_fault`, `last_fault_ms` | Preserved most-recent fault description and uptime clock |
| `last_fault_feed_result`, `last_fault_feed_ms`, `last_fault_fetch_ms` | Captured fault-time feed result/progress clocks, retained through recovery |
| `last_feed_result`, `last_feed_ms`, `last_fetch_ms` | Current feed/fetch progress diagnostics |

USB console commands are newline-terminated: **`AUDIORESET`** requests the same bounded recovery used after a fault; **`AUDIOSTALL`** deliberately stops detector consumption for **6,000 ms** while feeding continues, exercising the real ring-saturation/fault path. The reply is an `audio_diagnostic` event with boolean `accepted`; the stall request is rejected during readout. They are diagnostic controls and do not alter command phrases or weather values. No HTTP route exposes them.

Pipeline reset waits for both workers to park outside microphone/AFE/MultiNet calls before resetting the ring/VAD, cleaning the decoder and restarting the microphone. Speaker handoff requires the **feed worker parked or healthy capture paused**, plus **verified microphone-channel release**; it does not require detector parking. This lets KEY speech proceed when recognition remains faulted. Missing required acknowledgement/release reports failure while preserving resources. Pipeline repair is deferred until speaker release. Quiet-tail and stale-event guards apply when capture resumes. These controls help verify ownership/recovery transitions; they do not establish the spontaneous fault's original cause or long-term reliability.
