# The board's own interface

Since **40-public.1** (10 November 2026) the board collects, trains and forecasts by itself, and this is the interface it offers: an HTTP service on port 80 of its Wi-Fi address and a USB serial console. It no longer reads any server. Earlier client releases (25-public.1 to 32-public.1) consumed a LAN server's `/api/rlcd/v1`; that contract is described in the 32-public.1 version of this file in the repository history and in the [historical verification records](VERIFICATION.md).

The board registers the mDNS name **`ttdi-weather.local`** and advertises `_http._tcp` on port 80. Its DHCP hostname is `waveshare-weather`. All epochs are UTC Unix seconds; the board's day, session and midnight clocks use Malaysia time (UTC+8). A missing value is never zero or an invented replacement: inside an available object it is JSON `null`, and an unavailable object carries `available: false` and may omit its value fields.

The service has no authentication or TLS. Keep the board on a trusted LAN.

## HTTP endpoints

| Method and path | Returns |
| --- | --- |
| `GET /` | The web dashboard: one gzip-encoded page of about 9 KB, built from [web/index.html](../WeatherDashboard/web/index.html) into `WebPage.h` by `tools/build_web.py` |
| `GET /weather` | The forecast payload below (schema 1); `503 {"error":"weather_unavailable"}` before the first one is ready |
| `GET /history?hours=6\|24\|168` | Stored readings for the web charts; any other `hours` gives `400` |
| `GET /status` | Firmware, Wi-Fi, CPU, memory, collection, model and readout state |
| `GET /debug/forecast` | The inputs and raw outputs of the newest forecast |
| `GET /screen.pbm` | The current framebuffer as a 400 × 300 binary PBM image |
| `POST /refresh` | Asks for an immediate collection pass: `202 {"refresh":"requested"}` |
| `POST /view` | Form field `page=0`, `1` or `2` selects a page: `200 {"page":N}`, otherwise `400` |
| `POST /read-info` | Same as releasing KEY when idle: `202 {"started":true}`, `409` while a readout plays, `503` when no readout could start (for example before the first payload) |
| `POST /indoor-correction` | Sets the indoor temperature offset ([below](#local-board-indoor-correction)) |
| `POST /seed`, `POST /seed-done` | History upload; only while the USB `SEED` window is open, otherwise `403 {"error":"seed_closed"}` ([below](#history-seeding)) |

The web page polls `/weather` every 180 s and `/history` for the selected range. **Fetch now** posts `/refresh`, then polls `/weather` until `generated_epoch` changes; a new reading usually arrives in about 20–40 s.

## The forecast payload: `GET /weather`

The payload keeps **schema 1** and the field meanings of the server's `/api/rlcd/v1`, which the pages were written against. The board builds it every 60 s, and at once when a new reading or weather run arrives.

| Top-level field | Meaning |
| --- | --- |
| `schema_version` | `1` |
| `generated_epoch` | When the board built this payload; it is also the forecast issue clock |
| `served_epoch` | When this HTTP response was sent (added by the board, not in the server's payload) |
| `timezone` | `"Asia/Kuala_Lumpur"` |
| `source` | `"device"` |
| `current`, `weather`, `tennis_morning`, `history`, `forecast` | Below |

### `current`

`available` is true when the newest TTDI reading is at most 15 minutes old and has a PM2.5 value; `fresh` when it is at most 7 minutes old. `observed_epoch` is the sensor's own timestamp. When available it carries `pm25_ugm3`, `temperature_c`, `humidity_pct` and `heat_index_c` as the sensor reported them. `display_text` is the observed-status line, such as `Observed: Recent PM2.5 medians at or below 35 µg/m³` or `Latest sensor reading`. It is the server's rule-based observation label recomputed on the board from the last 24 hours, not a forecast.

### Observed-status speech

Read Info on Page 1 speaks the status body right after the current outdoor PM2.5 and before the outdoor temperature, from this finite clip bank. Exact text selects the clip:

| `current.display_text` | Spoken words |
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
| Anything else, with otherwise eligible current data | Observed status unavailable |

PM2.5 is spoken as **P M two point five**. The screen and the speech drop the leading `Observed: `; the payload keeps it. Expired or invalid current data suppresses the status.

### `weather`

Open-Meteo context for Page 1. `available` needs a run fetched within the last two hours; `fresh` one within 30 minutes. `fetched_epoch` is when the board received it. `current_hour.wind_kmh` is the 10 m wind of the hour containing now. `next_hours` lists up to six coming hours as `valid_epoch`, `rain_start_epoch`, `rain_end_epoch` and `rain_chance_pct` (0–100).

### `tennis_morning`

The 07:00–09:00 window, today's until it ends and then tomorrow's: `available`, `fresh`, `start_epoch`, `end_epoch`, `fetched_epoch`, and when available `rain_chance_max_pct`, `feels_like_max_c` and `wind_mean_kmh`. These come straight from the forecast; no model is involved.

### `history`

The last six hours for the on-screen graph: `available`, `fresh`, `observed_epoch`, `start_epoch`, `end_epoch`, `columns` = `["epoch", "pm25_ugm3", "temperature_c", "heat_index_c"]`, and `points` holding up to the newest 128 rows, rounded to one decimal. `gaps` lists `[before, after]` pairs wherever consecutive readings are more than `gap_after_s` (600 s) apart, with `gap_kind: "source_collection_before_downsampling"`. `pm25_summary` gives `available`, `sample_count`, `average_ugm3`, `lowest_ugm3` and `highest_ugm3` over the same rows. At the 3-minute cadence six hours hold about 120 rows, so nothing is thinned.

### `forecast`

`issued_epoch` is the issue clock. `available` and `fresh` are true when the newest reading is at most 15 minutes old and at least one model output was published.

| Object | Fields |
| --- | --- |
| `near90` | `available`, `role`, `calibrated`, `pm25_ugm3` (the +90-minute point), `target_epoch` (issue + 5,400 s), `first20`, `arrival_change` |
| `near90.first20` | Always `{"available": false}`: the first-crossing head is not ported |
| `near90.arrival_change` | `available`, `issued_epoch`, `arrival_epoch` (issue + 5,400 s), `fresh_reference_epoch`, `reference_ugm3`, `fall20`, `fall40`, `rise20`, `rise40`, `within20`, `outcome`, `outcome_probability`, `display_text`, `model` (`device_arrival_logistic_recall_f2_v2`) |
| `ride90_210` | `available`, `role`, `calibrated`, `pm25_ugm3` (the ride mean), `start_epoch` (issue + 5,400 s), `end_epoch` (issue + 12,600 s), `minimum_maximum`, `mean_le70`, `weather` |
| `ride90_210.minimum_maximum` | `available`, `kind: "predicted_window_minimum_maximum"`, `resolution_minutes: 15`, `minimum_ugm3`, `maximum_ugm3` |
| `ride90_210.mean_le70` | `available`, `probability` (absent when unavailable) |
| `sessions.morning`, `sessions.afternoon` | `issued_epoch`, `start_epoch`, `end_epoch`, `pm` (`available`, `role`, `calibrated`, `pm25_ugm3`, `model: "device_session_pooled_hgb_v1"`) and `weather` |

Every PM2.5 output carries `role: "raw_model_output"` and `calibrated: false`. It is the model's literal output: a missing or unfitted model gives `available: false` and `pm25_ugm3: null`, never persistence or another head's value.

- **Point heads.** The +90 value and the ride mean, minimum and maximum come from four gradient-boosting regressors, each predicting the change from the fresh reference. They need today's fitted model, a fresh sensor input row, an Open-Meteo run fetched before the issue and at most two hours old, and at least one neighbour station. Without these, all four are unavailable. The minimum and maximum are point forecasts of the smallest and largest complete 15-minute medians in the window, not a confidence interval.
- **P(ride mean ≤ 70).** `mean_le70.probability` places the out-of-fold residual distribution of the ride-mean head around the current ride-mean forecast. 70 µg/m³ is the owner's planning cutoff, not a health limit.
- **Arrival change.** The five-class classifier's tails: `fall20` and `rise20` are changes of at least 20 µg/m³ from `reference_ugm3` at `arrival_epoch`, and `fall40` and `rise40` the ≥40 subsets of those. `fall20`, `within20` and `rise20` sum to one. `outcome` is `fall20`, `rise20` or `within20` for a unique largest class, with `outcome_probability` its probability and `display_text` **Fall on arrival**, **Rise on arrival** or **No change on arrival**. On an exact tie both are `null` and `display_text` is **Arrival outcome uncertain**. The head needs today's fitted model and a reading from the last five minutes as its reference. The probabilities are recall-weighted, so they are not calibrated.
- **Sessions.** Morning is the next 09:00–11:00 window and Afternoon the next 14:00–16:00 window that still starts at least 90 minutes after the issue. Both come from one pooled model, trained without the server's three CAMS inputs.
- **Window weather.** `ride90_210.weather` and each session's `weather` hold `available`, `fresh`, `fetched_epoch`, `coverage_verified`, `coverage.complete`, and when complete `temperature_max_c`, `feels_like_max_c`, `humidity_mean_pct`, `wind_mean_kmh`, `gust_max_kmh`, `rain_chance_max_pct` and `rain_mm`. Instant fields use the hours the window starts in and interval fields the hours it ends in, as on the server. An incomplete window is unavailable rather than summarised from part of it.

Compared with the server's `/api/rlcd/v1`, the board sends only the fields listed above, with the same meanings. It adds `source` and `served_epoch` and leaves `first20` unavailable. Among the fields it omits are the top-level `status`, `refresh_after_s`, `dashboard_build` and `limitations`; every `age_s`; the point heads' `reference_pm25_ugm3`, `experimental`, `validated`, `qualified`, `used_for_decision`, `performance_status` and `range_*` fields (and `model` on `near90` and `ride90_210`); `model` on `minimum_maximum` and `mean_le70`, and `calibrated` on `mean_le70`; the arrival head's `calibrated` and `recallWeighting`; `sessions.*.logistics_epoch`; and most hourly weather fields. The arrival head carries its own `reference_ugm3`. There is no `demo` field any more.

## Stored history: `GET /history?hours=6|24|168`

Returns `hours`, `stored_days` (the span of readings held in memory), `source_rows` (readings in the requested window), `columns` = `["epoch", "pm25_ugm3", "pm10_ugm3", "temperature_c", "heat_index_c"]`, `points`, `gaps` and `summary`. Up to 480 rows are sent as they are. Above that, the window is cut into 240 time bins, and each bin keeps its lowest and highest PM2.5 rows, so a short spike survives the thinning. `gaps` lists up to 64 `[before, after]` pairs more than 600 s apart. `summary` (`available`, `sample_count`, `average_ugm3`, `lowest_ugm3`, `highest_ugm3`) covers every reading in the window, not only the rows sent.

## Status: `GET /status`

The same JSON is printed for the USB `STATUS` command.

| Field | Meaning |
| --- | --- |
| `firmware`, `uptime_s`, `clock`, `page`, `frames` | Firmware id, seconds since boot, the board's clock, selected page (0–2) and frames drawn |
| `wifi` | `connected`, `rssi`, `ip`, `power_save` (0 none, 1 minimum modem sleep, 2 maximum) |
| `cpu` | `mhz`; per core `busy_last_minute_pct`, `busy_peak_minute_pct` and `busy_since_boot_pct`, from the idle tasks' run time sampled every minute; `measured_s` |
| `heap` | `internal_free`, `internal_min` (lowest since boot) and `psram_free`, in bytes |
| `battery_v` | Battery voltage from the board's divider |
| `engine` | `storage_ready`, `storage_used`, `storage_total`, `storage_formats`; `readings`, `polls`; `ttdi_ok`/`ttdi_fail`, `neighbour_ok`/`neighbour_fail`, `weather_ok`/`weather_fail`, `last_http`, `last_tls_error`; `last_reading`, `last_weather`, `last_issue` (epochs); `task_stack_free`; `model_cutoff`, `model_fitted`, `last_fit_ms`, `arrival_cutoff`, `arrival_fit_ms`, `session_cutoff`, `session_fit_ms`, `fit_stage` (0 when no fit is running); `observe_ms` (time to compute the status line) |
| `readout` | `busy`, `requests`, `completed`, `stopped`, `planned_seconds`, `stack_free_min`, `error` |

A model's `*_cutoff` equal to the latest Malaysia midnight means it was refit for today and can publish.

## Forecast inputs: `GET /debug/forecast`

For checking the board against another implementation: `issue`, `fresh` (the input row was complete enough to forecast), `has_wind`, `x` (the 81 point-model inputs, in the order of [PmFeatures.h](../WeatherDashboard/PmFeatures.h)), `point` (+90, ride mean, ride minimum, ride maximum), `chance` (P(ride mean ≤ 70)), `arrival` (the five class probabilities: fall ≥40, fall 20–40, within 20, rise 20–40, rise ≥40) and `session` (Morning, Afternoon). Missing values in `x`, `point`, `arrival` and `session` are `null`; `chance` is omitted when unavailable, and the body is `null` before the first forecast.

## USB serial commands

115200 baud, one command per line, newline-terminated. Replies are JSON lines. `tools/provision.py` sends them.

| Command | Effect |
| --- | --- |
| `STATUS` | Prints the status JSON above |
| `WIFI <ssid-hex> <password-hex>` | Joins a 2.4 GHz network; both values are hex-encoded UTF-8, and `-` stands for an empty password. They are saved in NVS once connected and never echoed. `provision.py wifi` builds this command from a hidden prompt |
| `VIEW <n>` | Selects page `n` (0 overview, 1 ride forecast, 2 history) |
| `READINFO` | Same as releasing KEY: reads the page aloud, or stops a readout |
| `REFRESH` | Asks for an immediate collection pass |
| `SEED` | Opens the ten-minute history-upload window |

## History seeding

`SEED` opens a ten-minute window. Within it, `POST /seed` takes one day file as the raw request body, with its destination in the `X-Seed-Path` header: `/r/`, `/p/`, `/x/`, `/g/` or `/s/` followed by `yyyymmdd.bin`. The board writes the body to a temporary file and renames it over the day file only when every byte announced in `Content-Length` has arrived, so a cut connection leaves the old file in place. It answers `200 {"opened":true,"bytes":N}` for a complete upload and `500` otherwise. `POST /seed-done` restarts the board so it loads the new history. Outside the window both return `403`. The record layouts are the structs in [Engine.cpp](../WeatherDashboard/Engine.cpp) and [PmFeatures.h](../WeatherDashboard/PmFeatures.h). The owner's seeding script reads a private database and is not published.

## Local board indoor correction

This sets the indoor temperature offset. Send `POST /indoor-correction` with `Content-Type: application/x-www-form-urlencoded` and this form field:

```text
temperature_offset_c=-4.0
```

For example, replace `BOARD_HOST` with the board's hostname or LAN address:

```sh
curl -X POST "http://BOARD_HOST/indoor-correction" --data "temperature_offset_c=-4.0"
```

The field must parse completely as a finite number in **[−10, +10] °C**; a JSON body is not accepted. Success returns `200 {"saved":true,"temperature_offset_c":-4.00}`. A missing or invalid field returns `400`. A failed NVS save returns `500` and leaves the offset unchanged. The setting persists across restarts and reflashes; with no valid saved setting the board uses **−4.0 °C**. Choose your own offset against a co-located reference in the board's normal position. It is an estimate and does not replace the SHTC3's factory calibration.

With `Traw` and `RHraw` the sensor's converted readings and `offset` the configured offset, the board applies equation 1 of Sensirion's [Design Guide for Humidity and Temperature Sensors](https://sensirion.com/resource/user_guide/sht/design_in/):

```text
Tcorrected = Traw + offset
RHcorrected = RHraw × exp(17.62 × 243.21 × (Traw − Tcorrected)
                         / ((243.21 + Traw) × (243.21 + Tcorrected)))
```

RH is limited to **0–100%** after compensation; no separate RH offset is fitted. Offset **0** keeps the original readings. The indoor values on every page and in Read Info use the corrected readings, and changing the offset resets the corrected minima and maxima. In 40-public.1, `/status` no longer includes the indoor fields that earlier releases reported.

## Local board audio diagnostics

Removed in 40-public.1 together with voice recognition and the microphone. The voice-worker recovery fields and the `AUDIORESET` and `AUDIOSTALL` commands that v29 to v32 described here are in the 32-public.1 version of this file in the repository history.
