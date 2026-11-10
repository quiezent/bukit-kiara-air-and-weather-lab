# ESP32 reflective weather dashboard

Firmware for the **Waveshare ESP32-S3-RLCD-4.2** that runs the whole Bukit Kiara PM2.5 forecast on the board itself. The current public variant is **`weather-dashboard-40-public.1`** (10 November 2026), built from the owner's `weather-dashboard-40-standalone`. The previous public firmware, `32-public.1` of 14 October, was a display client that read a LAN server's `/api/rlcd/v1`. This one needs no computer on the network.

What the firmware does:

- **Collects** the public AirGradient TTDI station every 180 s, six neighbouring AirGradient stations every 300 s and an Open-Meteo forecast every 900 s, over HTTPS, and keeps the history in flash.
- **Trains its models on the chip every night,** from 00:05 Malaysia time: four gradient-boosting regressors for the +90-minute PM2.5 and the ride window, a five-class arrival-change classifier, and one session model for the Morning and Afternoon cards. They are C++ ports of the published server's models in [`app/`](../../app), which remains the reference implementation.
- **Publishes the literal model output** on its three pages and in its own JSON payload; its PM2.5 values carry `role: raw_model_output` and `calibrated: false`. A missing output shows as unavailable; nothing is substituted.
- **Serves the web dashboard** from the board at `http://ttdi-weather.local/`: the current reading, +90 and arrival, ride window, history charts, the Morning, Afternoon and Tennis cards, and **Fetch now**.
- **Reads the selected page aloud** when you press KEY (**Read Info**), from prerecorded clips.

What it no longer does: voice commands, wake words and the microphone are gone, and so are the ESP-SR speech models and their flash partition. The server's first-crossing (first20) head was not ported, because no current page shows it; `near90.first20` is always unavailable. It does not fetch Open-Meteo's CAMS air-quality feed, so the session model runs without its three CAMS inputs.

Running correctly is not the same as forecasting well. The [evidence section](#evidence-and-its-limits) separates the two.

## What you need

- A Waveshare ESP32-S3-RLCD-4.2 with its 16 MB QIO flash and 8 MB octal PSRAM. The firmware runs both cores at 240 MHz, with no frequency scaling or light sleep, and Wi-Fi in minimum modem sleep. The board in use has a single-cell battery of unknown capacity; battery life has not been measured.
- A 2.4 GHz Wi-Fi network with internet access. The board sets its clock by NTP and reaches `api.airgradient.com`, `map-data.airgradient.com` and `api.open-meteo.com`.
- To build: Python 3.11 or 3.12 (the pinned numpy 1.26.4 and scikit-learn 1.5.1 have no wheels for later versions, so `pip install -r requirements.txt` fails there), [Arduino CLI 1.5.1](https://github.com/arduino/arduino-cli/releases/tag/v1.5.1), a native [eSpeak NG 1.52.0](https://github.com/espeak-ng/espeak-ng/releases/tag/1.52.0) executable for the speech clips, and a C++17 host compiler such as GCC or Clang for the host tests. Install the native executables separately; none is shipped here.

| Dependency | Version |
| --- | --- |
| Official Arduino-ESP32 core (ESP-IDF) | 3.3.12 (5.5.5) |
| U8g2 / ArduinoJson | 2.36.18 / 7.4.3 |
| Python esptool / pyserial | 5.3.1 / 3.5 |
| Host tests: numpy, scipy, scikit-learn | as pinned in `requirements.txt` |

No other Arduino library is used.

## Build from source

Run these from this directory. Replace the quoted paths with your installations.

```sh
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` on Linux/macOS, then:

```sh
python -m pip install -r requirements.txt
python tools/setup.py --arduino-cli "/path/to/arduino-cli"
python tools/prepare_speech.py --espeak-ng "/path/to/espeak-ng"
python tools/build.py
python tools/test_host.py --cxx g++
python -m unittest discover -s tools/tests -v
python WeatherDashboard/tools/flash_firmware.py --build-dir build/firmware --dry-run
```

- **`setup.py`** checks the CLI version and installs only the pinned core, U8g2 and ArduinoJson into ignored `.state/` directories. There is no speech-model download any more.
- **`prepare_speech.py`** generates the public clip bank from [assets/speech/phrases.json](assets/speech/phrases.json): **120** ordered clips in native `en-us`, as PCM source plus a hash manifest. Speech is finite phrase-and-number composition from recorded clips; eSpeak NG does not run on the board. `--espeak-data PATH` points it at a separately extracted `espeak-ng-data` directory.
- **`build.py`** first runs `tools/build_web.py --check`, which fails if `WeatherDashboard/WebPage.h` does not match `WeatherDashboard/web/index.html`. After editing the page, regenerate the header with `python tools/build_web.py`. The build then compiles with Arduino CLI and fails on any compiler warning. It uses the FQBN `esp32:esp32:esp32s3:USBMode=hwcdc,CDCOnBoot=cdc,FlashSize=16M,FlashMode=qio,PSRAM=opi,PartitionScheme=custom,EraseFlash=none` with `upload.maximum_size=7340032` and `-DWEBSERVER_MAX_LINE_WAIT=1000 -DWEBSERVER_MAX_HEADER_WAIT=2000`. The sketch `static_assert`s those two values, so a plain Arduino IDE build stops with a message. `build/firmware/build-manifest.json` records the build.
- **`test_host.py`** runs the host regressions without a device or network: the web-page check, the readout, arrival-decoder, indoor-compensation, indoor-sensor and page-readout suites, two parity tests that compare the C++ trainers with scikit-learn, and, once the firmware is built, the compiled-image checks. `--skip-flash` runs it before a build. Compiler drivers take repeatable `--cxx-arg`; a Windows Zig example is `python tools/test_host.py --cxx zig --cxx-arg=c++ --cxx-arg=-target --cxx-arg=x86_64-windows-gnu`. `build/host/host-verification.json` records the run.

The [partition table](WeatherDashboard/partitions.csv) has `nvs` at 0x9000 (24 KB), `phy_init` at 0xF000, one 7 MiB `factory` app at 0x10000 and the LittleFS `storage` area at 0x710000 (0x8F0000 bytes, about 9.37 MB usable). Generated speech, SDK caches, build logs, host reports and firmware images are ignored build artifacts.

## Flash and provision

Connect the board by USB and find its serial port. These commands **write firmware** and then Wi-Fi settings; replace `YOUR_SERIAL_PORT` with, for example, `COM5` or `/dev/ttyACM0`:

```sh
python WeatherDashboard/tools/flash_firmware.py --build-dir build/firmware --port YOUR_SERIAL_PORT
python tools/provision.py wifi --port YOUR_SERIAL_PORT
python tools/provision.py status --port YOUR_SERIAL_PORT
```

Use this flasher rather than a generic upload or erase. It checks the image headers, the exact partition layout and checksums, then writes only the bootloader (0x0), partition table (0x8000) and application (0x10000). It refuses any write over NVS or `storage`, so a reflash keeps the Wi-Fi settings, the indoor offset and the collected history. A dry run opens no serial port.

`provision.py wifi` asks for a 2.4 GHz network name and prompts for the password without echoing it; it never reads a password saved on the computer. It sends the USB `WIFI` command, and the board stores the credentials in NVS once it has connected. The same tool sends the other USB commands (`status`, `view --page N`, `readinfo`, `refresh`, `seed`); the [interface guide](docs/API.md#usb-serial-commands) lists them.

## What the device stores

Everything lives in the `storage` partition as LittleFS day files named `yyyymmdd.bin` (Malaysia dates). The device prunes by the date of its newest reading, not the clock, so one wrong time reply cannot delete history. Collection gaps are never backfilled.

| Folder | Contents | Kept |
| --- | --- | --- |
| `/r` | TTDI readings: PM2.5, PM10, temperature, humidity, heat index and the time each was received | 35 days |
| `/x` | 15-minute origin rows: the 81 point-model inputs as issued, including the issued wind | 35 days |
| `/p` | Neighbour-station polls, first receipt of each value only | 3 days |
| `/g` | TTDI particle count, TVOC and CO2, for the observed-status line | 3 days |
| `/s` | Session-model weather inputs as issued at each :00 and :30 | 16 days |
| `/w` | The latest Open-Meteo response | newest only |
| `/m` | The three fitted models | newest only |

Memory holds the newest 14,500 readings, about 30 days at the 3-minute cadence. With 30 days of readings the owner's board used about 3.9 MB of the 9.37 MB.

The issued-wind and session-weather inputs cannot be rebuilt later, so a new board starts empty. A board can be seeded from an existing archive: the USB `SEED` command opens a ten-minute window in which `POST /seed` accepts whole day files. Each upload goes to a temporary file and replaces the day file only when it is complete. The owner's seeding script reads a private database and is not published; the record layouts are the structs in [Engine.cpp](WeatherDashboard/Engine.cpp) and [PmFeatures.h](WeatherDashboard/PmFeatures.h).

## Use another station or place

The firmware is written for the TTDI station and Bukit Kiara. To point it elsewhere, edit these constants and rebuild:

- `kTtdiUrl` in [Engine.cpp](WeatherDashboard/Engine.cpp): the AirGradient public location (86311 for TTDI).
- `kWeatherUrl` in the same file: the Open-Meteo latitude and longitude.
- `kStationTable` in [PmFeatures.h](WeatherDashboard/PmFeatures.h): six neighbouring AirGradient map locations with their coordinates. The 18 station and 10 network inputs assume six stations; a different number changes the input layout.
- `kTtdiLatitude` and `kTtdiLongitude` in the same file: the target sensor's position, from which the neighbours' distances and upwind weights are computed.
- Malaysia time (UTC+8, no daylight saving) is built in with no single setting: the literal `8 * 3600` (or `28800`) in Engine.cpp, PmFeatures.h, SessionFeatures.h, ArrivalLogistic.h, ObservedLabel.h, HistoryGraphs.cpp and PageReadout.cpp; `setenv("TZ", "MYT-8", 1)` and `configTime(8 * 3600, ...)` in WeatherDashboard.ino; and `Asia/Kuala_Lumpur` in Engine.cpp's payload and in web/index.html (then run `python tools/build_web.py`). Together they set the day files, the midnight refit, the session and tennis windows, the models' time-of-day inputs, the observed-status hour rule and the displayed clocks. Elsewhere, change every one, and the UTC+8 dates in `tests/page_readout_test.py`; a zone with daylight saving needs code changes, not new constants.

A new board shows each head as unavailable until its training minimum is met: 240 fifteen-minute origins with neighbour and wind inputs and complete labels for the point heads (about three days of collection), and at least 300 issues over 10 local days for the arrival classifier and 120 rows over 10 target days for the session model. Their training windows fill after 14 days (session) and 28 days (point and arrival). Every hyperparameter and class weight was chosen on TTDI's data. At another station they are untested.

## The models

All three are refit every night from 00:05 Malaysia time. A failed fit is retried hourly, never past the next midnight, and the board restarts once on the first failure of a night, because a fragmented PSRAM cannot be repaired by retrying. The whole refit takes about 13.9 minutes, and the forecasts read unavailable until it finishes.

| Output | Model on the device | Trained on |
| --- | --- | --- |
| +90 PM2.5; ride mean, minimum and maximum over +90 to +210 min | Four histogram gradient-boosting regressors with absolute-error loss, following scikit-learn 1.5's `HistGradientBoostingRegressor` (120 iterations, learning rate 0.1, 15 leaves, at least 60 samples per leaf, L2 1.0, no early stopping). 81 inputs: 28 sensor, 4 control, 5 time, 18 neighbour-station, 10 network and 16 issued-wind | 15-minute origins over 28 days (about 2,630–2,670 rows); about 655 s |
| P(ride mean ≤ 70 µg/m³) | Out-of-fold residuals of the ride-mean head over three day blocks | The same rows |
| ≥20/≥40 change on arrival (five classes) | Multinomial logistic regression: median imputer with missing indicators, standard scaler, C = 0.1, L-BFGS (max 500 iterations, tol 1e-4), recall weights 2 for both fall classes | 28 sensor inputs at 5-minute issues over 28 days, rebuilt from the stored readings (8,019 rows and 59 iterations at the 10 November cutoff); about 157 s |
| Morning (09:00–11:00) and Afternoon (14:00–16:00) mean | One pooled gradient-boosting regressor (100 iterations, learning rate 0.05, 7 leaves, at least 80 per leaf, L2 50, depth 3, absolute error) on 28 of the server's 31 Morning inputs; the three CAMS inputs are missing | 14 days, 7 lead times (4,505 rows at the 10 November cutoff); about 19 s |

The session model replaces two server models: the Afternoon PatchTST, a neural network that cannot be trained on this chip, and the server's Morning model. The **observed status line** under the outdoor number, such as "Recent PM2.5 medians at or below 35 µg/m³", is not a model. It is the server's rule-based observation label, recomputed on the board from the last 24 hours of readings. Tennis Morning (07:00–09:00), rain chance, feels-like temperature and wind come straight from the Open-Meteo forecast.

The point heads need a fresh sensor reading, a wind run issued within the last two hours and at least one neighbour station; without them every point head is unavailable. If Wi-Fi has been up for 30 minutes with no feed answering for 30 minutes, and no fit is running, the board restarts.

## Evidence and its limits

**Software correctness.** These checks show the port computes what the server computed. They say nothing about whether either forecasts well.

- On the host, in 64-bit arithmetic, the C++ gradient-boosting trainer ([HgbTrainer.h](WeatherDashboard/HgbTrainer.h)) reproduced scikit-learn 1.5.1's `HistGradientBoostingRegressor`. The board also trains in double precision.
- Given identical inputs, the board's own fitted arrival classifier and the server's pickled model for the same cutoff (8,019 rows, 59 iterations) agreed to a worst probability gap of 5.7e-9.
- A host harness built from the firmware's own row builders reproduced the server's training rows. Arrival: the same issues, identical labels, inputs within 2.8e-14. Session: the same 4,505 rows, inputs within 1e-9 apart from the missing CAMS inputs. These harnesses read private data and are not published; `test_host.py`'s two trainer tests are the public part.
- At one issue clock on 10 November, device against server: +90 45.04 vs 46.43 µg/m³; ride mean 44.04 vs 46.14; ride min–max 39.3–59.1 vs 42.1–57.4; P(mean ≤ 70) 0.955 vs 0.943; arrival within-20 0.923 vs 0.929, both "No change on arrival"; Morning 32.16 vs 32.11; Afternoon 29.14 vs 27.43, a different model. The point differences come from sensor poll timing.
- An independent review of the integrated firmware confirmed 12 issues, all fixed before v40. Among them: the TTDI 30-second retry was missing; a failed fit could repeat for lack of contiguous PSRAM; the "Forecast ready" flag did not require a published PM2.5 output; Read Info on the history page always said "Data is old"; **Fetch now** did not trigger a collection; and an aborted history upload could leave a torn day file.
- On the board: the unattended midnight refits ran and all three models publish. The web page loads, and **Fetch now** brings a new reading in about 20–40 s. The owner reports that Read Info plays without stutter.

**Forecast skill.** Scored on the server's issued records from 14 October to 9 November (Afternoon on 15-minute slots; MAE differences with 95% intervals):

- The pooled session model **ties** persistence on Afternoon (−0.26 µg/m³ [−2.44, +2.42]) and ties the issued PatchTST (−1.08 [−4.69, +2.74]).
- On Morning, with 28 inputs, it ties the issued Morning card (+0.22 [−1.01, +1.38]).
- Dropping CAMS cost +1.60 [+0.45, +2.89] µg/m³ on Afternoon 15-minute slots. This is a secondary comparison; read it as a hint.
- The point and arrival heads are the server's recipes, so the season's [scorecard](../../docs/MODEL_CARD.md#the-season-on-the-issued-records--7-november-2026) applies: the point heads beat persistence over the season, mostly through the 14–23 October haze; afterwards the +90 and ride mean tied it and only the ride minimum stayed ahead. The arrival head did not beat its baselines.

**Not yet verified:** a multi-day soak of v40, battery life, and any forecast scored on the device's own issued output.

## Resources on the chip

Measured on the owner's board running v40 on 10 November 2026:

- **CPU:** averaged over the first 53 minutes of v40 at 240 MHz, boot included, 4.9% (core 0) and 2.0% (core 1) busy. Core 0 runs the collection: most minutes are under 2.5%, and the minutes with a neighbour-station pass (six TLS handshakes every 5 minutes) reach about 7–11%. The nightly refit runs on core 1 for about 14 minutes; these figures do not include it.
- **Memory:** about 186 KB of internal RAM free (lowest since boot about 119 KB); about 6.9 MB of the 8 MB PSRAM free when idle; about 12 KB free on the collection task's stack.
- **Flash:** the private v40 image is 6,282,595 bytes, 85% of the 7 MiB slot, mostly the recorded speech clips. The public image's size depends on the eSpeak NG clips you generate (see Verification). Storage: about 3.9 MB of 9.37 MB with 30 days of readings.

`GET /status` reports the CPU, memory, storage and collection counters live.

## Buttons and Read Info

Release **BOOT** for the next page. The three pages are **TTDI WEATHER** (current reading, indoor, +90 and arrival, Tennis Morning), **MTB Ride Forecast** (ride window and the Morning and Afternoon sessions) and **OUTDOOR HISTORY** (graphs, battery, indoor ranges with the time since the last restart, and the board's own network address, so a phone on the same network can open the web page). Release **KEY** to read the selected page aloud, and again to stop. While reading, the footer shows **[KEY] Stop Reading** on the left, **WiFi: RSSI** in the center and **[BOOT] Next Page** on the right. USB `READINFO` does the same as KEY. `POST /read-info` only starts a readout; while one plays it answers 409.

- **Page 1** reads the current outdoor PM2.5, the observed-status body, outdoor temperature, current rain, the +90-minute estimate and the native arrival-change winner with its probability, then next-hour rain. An exact arrival tie says **Arrival outcome uncertain**, without a percentage. A status the clip bank does not know says **Observed status unavailable**.
- **Page 2** reads the ride range or mean, P(mean ≤ 70), the ride window's rain chance and the dated morning and afternoon comparison, adding a high-rain-chance sentence for a session above 75%.
- **Page 3** reads the observed PM summary, the indoor readings and the coarse voltage-based battery estimate.

The page's values are captured before playback, so a new reading cannot change a sentence halfway. Old and unavailable states are spoken as such.

## Indoor board-heat correction

The SHTC3 sensor is factory calibrated, but heat from the assembled board can affect its room reading. The current firmware defaults to the user's preferred **−4.0 °C** temperature offset when no valid saved setting exists. Existing NVS settings are retained. The earlier **−4.8 °C** starting value and informal handheld/airflow comparisons belong to the [historical v27 record](docs/VERIFICATION_INDOOR_2026_10_09.md). These settings are placement-dependent estimates without a traceable system-calibration claim. Compare your own unit with a reference in its normal position before choosing an offset.

Relative humidity is adjusted from the raw temperature/RH using equation 1 of Sensirion's [design guide](https://sensirion.com/resource/user_guide/sht/design_in/), with `m = 17.62` and `Tn = 243.21 °C`; there is no independent RH trim. Corrected RH is limited to 0–100%. Every indoor display and page readout uses the corrected values; outdoor readings are unchanged. In 40-public.1, `/status` no longer reports the indoor fields, so the raw sensor measurements are not exposed over HTTP.

The offset accepts finite values from **−10 to +10 °C**, persists in NVS and can be changed using the local [indoor-correction endpoint](docs/API.md#local-board-indoor-correction). Set it to **0** to use the original sensor values without this temperature/RH correction. Changing it resets the corrected temperature and RH minima/maxima so ranges do not mix different settings.

## Verification

### 40-public.1, 10 November 2026

The public `WeatherDashboard/` tree is generated from the private v40 sketch by the firmware publisher, which changes only the firmware id, adds the Waveshare attribution headers and makes the page-readout test take its dependency paths as arguments. The [source manifest](../../docs/research/firmware-weather-dashboard-40-public.1-source-release.json) lists every file with its SHA-256.

- Public host tests: all 11 `tools/test_host.py` steps passed (zig `c++` on Windows), including both trainer parity tests and the four flash-validation checks, and the 24 Python helper tests passed.
- Public build: eSpeak NG was not installed on the build machine, so the build used placeholder audio from a stand-in synthesizer run through the real `tools/prepare_speech.py`. It compiled with 0 warnings to a 5,655,488-byte application image, and the flasher's dry run passed. Real eSpeak clips change the image size, so no public image hash is recorded. The build used an existing installation of Arduino-ESP32 3.3.12, U8g2 2.36.18 and ArduinoJson 7.4.3, not a fresh `tools/setup.py` run, and no GCC, Linux or macOS host ran the host tests.
- The public eSpeak NG voice has not been flashed or listened to. The on-board results above come from the owner's private build, which plays recorded Microsoft David clips that are not distributed.

## History: the LAN client firmware, 8–14 October 2026

From `25-public.1` (8 October) to `32-public.1` (14 October) this package was a display client. It discovered a LAN server by mDNS, read its `/api/rlcd/v1`, recognised a small English vocabulary on the board with Espressif's ESP-SR models, and read pages aloud. Its build, voice-command and demo-server instructions are in the 32-public.1 version of this file in the repository history. The sections below are the dated verification records of those releases, unchanged.

The [October 8 verification record](docs/VERIFICATION.md), [v26 arrival record](docs/VERIFICATION_2026_10_09.md), [v27 indoor-correction record](docs/VERIFICATION_INDOOR_2026_10_09.md), [v28 observed-status record](docs/VERIFICATION_OBSERVED_2026_10_09.md) and [v29 recovery record](docs/VERIFICATION_AUDIO_RECOVERY_2026_10_09.md) retain their historical build, host and installed-device scopes. Public-voice hardware validation remains separate.

### V32 verification, 14 October 2026

I (Claude, maintaining the firmware since 13 October) ported the v32 change into this package with the repository's publishing script, which re-derives every public file from the private source. Applied to the private v31 source, the same transform reproduces all 50 published v31 files byte for byte.

All **15** host-runner steps passed with `tools/test_host.py --skip-flash`, the pinned Arduino-ESP32 3.3.12, ArduinoJson 7.4.3 and ESP-SR headers, and Zig's `c++` driver: the four native executables and the diagnostic-publication, lifecycle (**175** checks), arrival-decoder (**159** assertions), indoor-compensation (**492**), indoor-sensor (**557**) and page-readout suites, including the forced-overflow case. The **23** Python helper tests also passed. The machine had no eSpeak NG, so I could not generate the public clip bank: the page-readout suite ran with a stand-in manifest (the public phrase IDs and text, clip lengths from the installed private voice, a fixed placeholder for **Demonstration data**). For the same reason I did not compile the public image, run the five final-image checks or the flasher dry run, and I record no public image hash for this version.

The separately installed private-voice v32 was built with the same pinned versions and flashed on 13 October. Its recorded checks: host tests passed, `/status` reported the new firmware, the weather API returned HTTP 200, the saved −4 °C indoor offset survived and the framebuffer was inspected. No listening test or recognition-accuracy result is claimed.

### V31 verification, 9 October 2026

The public build compiled without warnings, with all **38** compiled C++/header files matching the final source. All **five final-image validation checks** and the no-port flash dry run passed. The 117-clip speech bank, phrase definitions and upstream model bundle are unchanged. Broader host/audio tests were not repeated for this footer-only change; the v30 results below remain historical evidence.

The separately installed David-audio v31 passed **23 focused checks** in **5.969 seconds**. Normal and reading framebuffers for all three pages were inspected while one overview readout stayed active through page navigation: **[KEY] Stop Reading** remained left, Wi-Fi centered and **[BOOT] Next Page** right, with no **Voice: reading** indicator. USB `KEY`, which uses the physical-button handler, stopped the partial stream without an error; the quiet tail and microphone capture resumed, the normal **[KEY] Read Info** footer returned and overview was restored. No new full-stream, physical-button or human-listening test is claimed.

Public application: **7,413,136 bytes**, SHA-256 `8b98ec526626ca62fc96ac06633bfc5a68ea62cc798827dec3004c6372ed430a`. The public eSpeak image remains unflashed; compiled images and private verification payloads are excluded.

### V30 verification, 9 October 2026

The public build compiled without warnings, with all 38 compiled C++/header files matching the final source. All 16 host-runner steps / eleven native executables passed without host warnings, including **175 checks of the production lifecycle/player** and **1,383 explicitly counted assertions** across four suites. The 23 Python helper tests, five final-image checks and no-port flash dry run passed. The 117-clip public speech bank is unchanged.

The separately installed David-audio v30 passed **64 dashboard and 36 readout-control checks**: all three intentional stops and all three full restarts completed, without AFE repair, recovery failure or reboot. Microphones and the recognition quiet-tail handoff resumed after each, and overview was restored. Normal/reading framebuffers were inspected for the footer and wind units. These checks add no human listening confirmation and cannot validate the unflashed public eSpeak voice.

Public application: **7,413,376 bytes**, SHA-256 `9bcdf9d4e7c27d3e941fe42427d7e510be3010d6b85609b0d918b949ac87133f`. Compiled images and private verification payloads are excluded from this source package.

## Network exposure

The board's web server has no authentication or TLS. Anyone on the same LAN can read its pages and payloads, change the page, start a readout, request a collection pass or change the indoor offset. History uploads are refused unless someone has opened the window over USB. Keep the board on a trusted network and do not forward its port.

## License and scope

Original dashboard firmware, tools and phrase definitions in this package use [Apache-2.0](LICENSE). The Waveshare display-driver and Espressif ES8311 codec adaptations retain their notices. The two trainers and the session imputer follow scikit-learn's and SciPy's implementations; their headers say so, and the BSD-3-Clause notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [LICENSES/](LICENSES/). External libraries, fonts and tools keep their own terms. Read [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [LICENSES/](LICENSES/) before redistribution.

This source publication ships no Microsoft David recordings, generated PCM or WAV files, SDK or tool executables, compiled application binary, fitted model, collected data or seeding script. Other repository components retain their own licenses and snapshot scope.
