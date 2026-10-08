# LAN data contract for the ESP32 dashboard

The board consumes **RLCD schema version 1** over HTTP. The example [rlcd-v1-synthetic.json](../examples/rlcd-v1-synthetic.json) documents the field structure using invented values. [demo_server.py](../tools/demo_server.py) serves that fixture with coherent fresh clocks; it is not a sensor, weather service or trained forecast model.

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

## Preserve three different forecast questions

| Quantity | API location | Native meaning |
| --- | --- | --- |
| First substantial change | `forecast.near90.first20` | Which first ≥20 µg/m³ crossing occurs within the next 90 minutes: rise, drop or neither? |
| Arrival estimate | `forecast.near90.pm25_ugm3` | Primary model's concentration estimate at `target_epoch`, normally issue +90 minutes |
| Ride context | `forecast.ride90_210` | Two-hour window from issue +90 to +210 minutes: mean, extrema, weather and chance its mean is ≤70 |

A first crossing can reverse before arrival. Its probabilities are not probabilities of an arrival increase/decrease, and a concentration estimate is not a crossing threshold. The board does not manufacture arrival-change probabilities by subtracting estimates or repurpose ride-window probabilities as arrival probabilities.

In the synthetic example, current/reference concentration is **37.0**, arrival estimate **44.2**, ride mean **41.7** and ride extrema **31.4–53.8 µg/m³**. First-rise/drop/neither probabilities are **0.20/0.10/0.70**; ride mean-≤70 probability is **0.81**. These deliberately different invented quantities must stay separate.

## Field semantics

| Object | Fields used by the device |
| --- | --- |
| `current` | `available`, `fresh`, `observed_epoch`, `pm25_ugm3`, `temperature_c`, `humidity_pct` |
| `weather` | `available`, `fresh`, `fetched_epoch`, `current_hour`, `next_hours` |
| `forecast` | `available`, `fresh`, `issued_epoch`, `near90`, `ride90_210`, `sessions` |
| `tennis_morning` | Dated 07:00–09:00 weather window with availability, fetch clock, rain, feels-like temperature and wind |
| `history` | Availability/freshness, source/window epochs, four-column points, collection gaps and full-source PM summary |

Primary PM values require `available: true`, a finite nonnegative `pm25_ugm3` within the display range, and role `raw_model_output`, `experimental_model_output` or legacy `experimental_window_mean`. The device never substitutes `reference_pm25_ugm3` for a missing primary model value. Qualification/calibration metadata stays distinct from numerical output; an eligible raw estimate is not treated as an operational recommendation.

### First-change model summary

`first20` provides `available`, `rise_probability`, `drop_probability`, `none_probability`, `reference_ugm3`, `diagnostic_direction`, `direction` and `display_text`. Probabilities are fractions in **[0,1]**, not percentages, and must sum to one within the classifier's tolerance. `diagnostic_direction` describes the raw unique winner; use `rise`, `drop` or `unresolved`. A unique neither winner and ties both use `unresolved`. Operational `direction` can remain unresolved independently of the raw winner.

The screen shows the server's nonempty `display_text`; keep it consistent with the structured fields. Speech validates and selects the structured winner, says that change is **most likely**, and reads its probability multiplied by 100. It does not parse a number from `display_text`. A tie says **No directional call**; it must not claim that no ≥20 change is most likely. When diagnostic metadata is absent, the classifier supports legacy `direction` semantics. Missing or inconsistent structured values remain unavailable.

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

`forecast.sessions.morning` and `.afternoon` contain `issued_epoch`, `start_epoch`, `end_epoch`, primary `pm` and `weather`. The example uses 09:00–11:00 and 14:00–16:00 Malaysia time. Speech compares valid primary values at displayed one-decimal precision only when the forecast is fresh, session issues match it and the windows remain valid. It speaks this/tomorrow morning/afternoon from the actual dates; expired or unsupported dates are not silently relabelled. `tennis_morning` is a separate 07:00–09:00 weather-only window.

Modern ride/session weather must provide `coverage_verified: true` and `coverage.complete: true`, plus `available`, `fresh`, `fetched_epoch`, `rain_chance_max_pct`, `rain_mm`, `feels_like_max_c`, `humidity_mean_pct`, `wind_mean_kmh` and `gust_max_kmh`. It must cover the complete target window. Incomplete coverage suppresses the window's weather/readout warnings rather than presenting a partial summary as complete. Genuine legacy payloads without either coverage field retain their earlier availability behavior. Fetch clocks must not postdate the forecast issue.

Hourly rain uses `rain_start_epoch`/`rain_end_epoch` one-hour intervals and `rain_chance_pct` in **[0,100]**. A provider's `current_hour` can describe the preceding hour. Speech selects the interval containing now and the interval containing now +1 hour, rather than trusting an array position. Rain warnings are issued for valid ride/session probabilities strictly **above 75%**.

### Observations, history and stale data

History columns are exactly `["epoch", "pm25_ugm3", "temperature_c", "heat_index_c"]`; row values may be null. `gaps` describe original collection gaps, rather than gaps introduced solely by downsampling. `pm25_summary` contains `available`, `average_ugm3`, `lowest_ugm3`, `highest_ugm3` and `sample_count`, computed from accepted original observations in the same displayed window. Do not calculate that summary from only the reduced graph rows.

Outdoor observations are marked old after seven minutes (or `fresh: false`) and unavailable after 15 minutes. PM forecasts are marked old after two minutes and unavailable after ten. Modeled weather is marked old after 30 minutes and unavailable after two hours. Original clocks, null values and invalid fields control these states; a new HTTP receipt must not make old content fresh. Historical graph data can remain visible with its original dates and **OLD** markers.

Indoor measurements and battery voltage come from the ESP32 board, not this API. Battery percentage is a coarse voltage-derived estimate. The firmware snapshots the selected page's values before playback so later polling/navigation cannot alter a readout mid-sentence.
