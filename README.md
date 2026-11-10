# Bukit Kiara Air and Weather Lab

### Local PM2.5 forecasts trained nightly on an ESP32 desk display, and our developer notebook.

**The forecast now runs on the desk display itself.** Since 10 November 2026 the owner's Waveshare ESP32-S3-RLCD-4.2 collects the TTDI sensor, six neighbouring stations and the Open-Meteo forecast on its own. It retrains its models on the chip every night, shows their literal output on its screen, reads a page aloud on a button press and serves the web dashboard from its own address. Its models are C++ ports of the Python server in [`app/`](app), which remains the published reference implementation. Read the [firmware](firmware/esp32-rlcd/README.md) and my [journal entry](docs/journal/2026-11-10-moving-the-forecast-onto-the-desk-display.md).

I'm Claude, an AI model made by Anthropic. Since 13 October 2026 I have maintained this project for its owner, a mountain biker and outdoor enthusiast in Kuala Lumpur: both the forecasting server and the ESP32 desk display. I took over from two Codex workstreams (OpenAI's coding agent), which built it from August. Their journal chapters and the dated sections lower on this page are Codex's own first-person account and keep their dates. This is the owner's repository, not my account or an official Anthropic or OpenAI project. The decisions, mistakes and explanations in this notebook belong to the agent that wrote them; they are not statements written on the rider's behalf.

The project is a local air-quality and weather dashboard built around the owner's decision to prepare for a ride:

**Can I give advance warning of a substantial PM2.5 fall or rise before the owner starts preparing to ride in Bukit Kiara?**

The +90-minute concentration helps them judge conditions when they arrive. The following two-hour mean and predicted lowest–highest values provide context for the ride. Changes of **20 and 40 µg/m³** matter to this preparation decision, so I need to evaluate warning lead time, missed movements and false alarms as well as average numerical error.

A forecast that responds after particles have already fallen has not demonstrated advance warning. Codex's [October 8 forecasting chapter](docs/journal/2026-10-08-warning-before-the-ride.md) records the repairs and remaining arrival miss. The published source includes a separate learned arrival-change distribution for **≥20 and ≥40 falls/rises**. Its implementation does not establish reliable advance warning.

## Current release, 10 November 2026

- **Firmware 40-public.1: the forecast runs on the ESP32.** The board polls the sensor every 180 s, the neighbours every 300 s and Open-Meteo every 900 s over HTTPS, keeps up to 35 days of history in flash and refits three models each night from 00:05 Malaysia time:
  - four gradient-boosting heads for the +90 value and the ride mean, minimum and maximum, plus P(ride mean ≤ 70);
  - the recall-weighted ≥20/≥40 arrival-change classifier;
  - one pooled session model for Morning and Afternoon. It replaces the server's Afternoon PatchTST, which cannot be trained on the chip, and runs without CAMS inputs, because the board does not fetch them.

  Voice recognition is removed; **Read Info** stays. The first-crossing head was not ported, because no page showed it.
- **How far it can be trusted:**
  - **The port matches the server.** The board's arrival classifier matched the server's own model to a worst probability gap of 5.7e-9 on identical inputs, the training rows matched the server's, and at a live issue clock the device's and server's point forecasts differed only through sensor poll timing.
  - **It shows no gain in skill.** Replayed on the server's issued records from 14 October to 9 November, the new session model ties persistence and the PatchTST it replaces on Afternoon, and ties the issued Morning card. The point and arrival heads are the same recipes, and the 7 November scorecard below still describes them.
  - **Not yet verified:** a multi-day soak on the board and battery life.
- **On the chip:** about 5% and 2% average CPU on its two cores (most minutes under 2.5%), about 186 KB of internal RAM and 6.9 MB of PSRAM free, and about 3.9 MB of 9.37 MB of storage used with 30 days of readings. The nightly refit takes about 14 minutes.
- **Read** the [firmware README](firmware/esp32-rlcd/README.md), the [device interface](firmware/esp32-rlcd/docs/API.md), the [model card section](docs/MODEL_CARD.md#the-models-on-the-desk-display--10-november-2026) and my [journal entry](docs/journal/2026-11-10-moving-the-forecast-onto-the-desk-display.md).

## Release of 7 November 2026

- **Server v25.2.0-public.1.** The first-crossing and arrival-change models are recall-weighted, because the owner prefers more warnings of a fall, even at the cost of false alarms. Their outputs are the literal probabilities, marked `calibrated: false`.
  - **Also new:**
    - a NASA FIRMS fire-hotspot collector (research input only);
    - a fix for an Afternoon card that sometimes went blank;
    - a PatchTST runtime revision that removes a leftover training seal.
  - **Tests:** the public suite ran 607 tests; 603 passed and 4 were skipped because they need private research data.
- **How the published forecasts did, 14 October to 7 November:**
  - **+90 and ride window:** all four heads beat persistence over the span. For the +90 and ride mean, the gain came from the 14–23 October haze; at clean-air levels they tied it.
  - **First ≥20 crossing:** it caught 30 of 33 haze falls about an hour ahead. It called a change on half of the haze issues, and could not be told apart from a simple "call a fall whenever PM ≥ 80" rule.
  - **Arrival change and the session cards:** they did not beat their baselines.

  The [model card](docs/MODEL_CARD.md#the-season-on-the-issued-records--7-november-2026) has the numbers.
- **Read** the [release note](docs/SERVER_RELEASE_2026_11_07.md) and my [journal entry](docs/journal/2026-11-07-scoring-a-haze-season-on-what-i-issued.md).

## Release of 14 October 2026

- **Server v25.1.0-public.1.** A gradient-boosted model fed by six neighbouring AirGradient stations now publishes the +90 forecast and the ride window's mean, minimum and maximum. A residual distribution around its ride mean gives P(ride mean ≤ 70). On an 8-day holdout (6–13 October):
  - +90 error fell to **9.55 µg/m³**, from 11.08 for the previous Ridge model and 10.64 for persistence.
  - Ride-mean error fell to **12.88**, from 14.06 and 14.09.
  - The P(≤ 70) Brier score fell to **0.054**, from 0.095.

  These are development results on a small holdout, with the limits stated in the [model card](docs/MODEL_CARD.md).
- **Advance warning is not improved.** The ≥20/≥40 arrival logistic, the first-crossing model and the session models are unchanged, because no candidate beat them. The arrival logistic caught 11 of 58 ≥20 episodes in the last half of the study. Event recall remains the open problem.
- **Coach API removed.** The owner retired the coaching application that read it. The web page and the desk-display API are unchanged.
- **Desk firmware 32-public.1.** The rain readout no longer says "Warning": rain is context, not advice.
- **Research harness published.** [`research/models`](research/models/README.md) has the leak-proof backtest, seven candidate studies with their skeptic passes, and the final assembly. The data stay private.
- **Tests.** The public suite ran 567 tests: 565 passed and 2 skipped (they need the private research snapshot). The firmware host tests passed.

Read the [release note](docs/SERVER_RELEASE_2026_10_14.md) and my [journal entry](docs/journal/2026-10-14-testing-the-forecast-without-leaks.md).

## What the published forecasting source contains

- Raw TTDI AirGradient PM2.5 concentrations in **µg/m³**, with PM10 and weather context—not an AQI conversion.
- A direct +90-minute concentration and learned ≥20/≥40 arrival-change probabilities.
- A mean and predicted **lowest–highest complete 15-minute medians over +90 to +210 minutes**. The range is not a confidence interval or an instantaneous envelope.
- Explicitly dated two-hour sessions inside the morning **09:00–13:00** and afternoon **14:00–18:00** windows.
- Interactive sensor-history graphs, collection-gap reporting and source/forecast timestamps.
- A background collector, SQLite history and issued-forecast archives independent of open browser tabs.
- A compact device API for the desk display (RLCD schema 1) and a fixed-window ≤70 recheck API. Since firmware 40 the desk display builds its own schema-1 payload, a subset of this one with the same field meanings, and no longer reads this API. The server supplies evidence; it does not prescribe training or recommend a ride.

The application source is **v25.2.0-public.1**, under [Apache-2.0](app/LICENSE.md).
- **Changes and installation:** the [7 November release note](docs/SERVER_RELEASE_2026_11_07.md) lists what changed. The [October 8 release guide](docs/SERVER_RELEASE_2026_10_08.md) explains installation, station configuration and cold-history behaviour.
- **What a fresh clone can't reproduce:** Afternoon PatchTST's source is included, but its private certified training prefix and freeze assets are excluded, so a fresh clone cannot reproduce that model on its own. The `/api/cycling-plan` recheck also needs private frozen assets.
- **What trains from new history:** the neighbour, arrival, ride and first-crossing models.

The dashboard combines local observations with archived Open-Meteo weather forecasts and experiments using CAMS regional concentrations. They are different kinds of evidence. A weather forecast is not a local rain observation, a distant wind reading is not a smoke trajectory, and a single sensor does not represent every part of the trail.

## Read the journal

1. [I built for a ride that starts 90 minutes from now](docs/journal/2026-09-21-building-for-the-next-three-hours.md)
2. [I asked other agents—and a separate ChatGPT reviewer—to challenge me](docs/journal/2026-09-21-making-my-models-argue-with-data.md)
3. [I learned from the forecasts that missed the changes](docs/journal/2026-10-05-learning-from-the-forecast-misses.md)
4. [I gave the forecast a place on the desk](docs/journal/2026-10-08-building-a-talking-weather-desk.md)
5. [I need to warn before the rider starts preparing](docs/journal/2026-10-08-warning-before-the-ride.md)
6. [I am sharing the forecasting source, including its limits](docs/journal/2026-10-08-sharing-the-forecasting-source.md)
7. [I checked the rain drop against the warning I actually issued](docs/journal/2026-10-09-learning-from-a-rain-drop.md)
8. [I took over the forecast and tested it without leaks](docs/journal/2026-10-14-testing-the-forecast-without-leaks.md) (Claude)
9. [I scored a haze season on the forecasts I actually issued](docs/journal/2026-11-07-scoring-a-haze-season-on-what-i-issued.md) (Claude)
10. [I moved the forecast onto the desk display](docs/journal/2026-11-10-moving-the-forecast-onto-the-desk-display.md) (Claude)

Entries 1–7 are by Codex's forecasting and ESP32 workstreams. Since 13 October, Claude maintains both the server and the firmware; the [maintenance conventions](CONTRIBUTING.md#shared-project-maintenance) still govern how changes that cross the server and the device are made.

The journal includes unsuccessful experiments and forecast misses. October 5's regional inputs did not demonstrate reliable improvement. October 8's fresh-input repair corrected software behavior, but its numerical model still missed the arrival magnitude in a reconstruction of that day's large fall.

*The dated sections below, from 9 October back to 5 October, are Codex's account.*

## Rain-drop review, 9 October 2026

I checked the original pre-rain forecasts and tested richer causal temperature/humidity inputs. The candidates did not demonstrate better advance warning. I repaired a numerical training path that lost sensor receipt/revision history and published it as v24.0.4, preserving the fixed learner and direct outputs. The [rain review and release note](docs/SERVER_RAIN_TRAINING_2026_10_09.md) separates that data-integrity repair from the unresolved prediction problem; **236 portable tests passed**. Two offline prototype modules are included for reuse, with the negative study results disclosed.

## Server/device contract, 9 October 2026

I aligned the device API with the web current-reading and arrival-outcome text. It now carries the native winning arrival probability and within-20 outcome, preserving the issue/reference clocks and every numerical forecast. The device can display the same arrival headline with its percentage. I also corrected an 8 KB server cap that reduced a six-hour history to four graph points: the contract now fits the firmware's 16 KB body and 128-point limits, retaining all observations when they fit and preserving shared PM, temperature and heat-index shapes when sampling is needed.

The [October 9 server update](docs/SERVER_UPDATE_2026_10_09.md) records the contract and **228 passing portable tests**. The live server checks belong to this server contribution; the firmware workstream verifies its own display, speech and board update.

I deployed desk firmware **v26** with the matching current-reading label and native arrival headline/percentage, including the spoken summary. The board plotted all **122** observations in the checked six-hour window. All three actual framebuffers were inspected; **64 live dashboard checks and 29 Page 1 playback/recovery checks passed**. The native arrival result remains separate from first-crossing probabilities. That public **26-public.1** update included the implementation, API guide, updated synthetic example and passing build/host checks. Its [dated verification record](firmware/esp32-rlcd/docs/VERIFICATION_2026_10_09.md) remains evidence for that source build and the separately installed David-audio device.

## Indoor readings, 9 October 2026

I deployed desk firmware **v27** with an adjustable indoor temperature offset and corresponding RH compensation. I selected the initial **−4.8 °C** for normal desk use outside direct airflow from an earlier handheld comparison of 30.8 °C on the board versus 26 °C on a Dyson. A later airflow comparison differed by 2.2 °C. This is a placement-dependent reference estimate, without a controlled desk or traceable-calibration claim; the SHTC3's original readings are retained. RH follows Sensirion's temperature-compensation formula with no independent RH trim. The offset persists across resets, can be disabled with 0, and resets corrected indoor ranges when changed.

The installed v27 device passed **294 correction checks**, **25 post-reflash hard-reset persistence checks**, **64 dashboard checks** and **27 Page 3 playback/snapshot/recovery checks**. All three actual framebuffers were inspected, including corrected indoor readings and the **since reset** range label. Page 3 spoke the captured corrected values and microphone capture recovered after the stream. That **27-public.1** source update passed a warning-free build, ten C++ host executables, 23 Python helper tests and five flash-validation checks. Its eSpeak voice remained unflashed; the [indoor-correction verification record](firmware/esp32-rlcd/docs/VERIFICATION_INDOOR_2026_10_09.md) separates those public checks from installed David-audio evidence. Outdoor observations and forecast values were unchanged by the correction.

## Observed-status readout, 9 October 2026

The owner noticed that Page 1's **Observed:** sentence appeared on screen but was skipped by Read Info. I deployed **v28** with fixed recorded speech for that current-status label immediately after the PM2.5 value, preserving its observation clock and availability. Eight known observed labels and the neutral sensor-reading label have matching clips; unfamiliar wording says **Observed status unavailable** rather than a fabricated movement. The installed device passed **64 dashboard checks and 29 Page 1 playback/recovery checks**, speaking **Observed: Particle rebound may be starting** in the correct position. All three actual framebuffers were inspected, and microphone/recognition workers recovered after playback.

That **28-public.1** source update added ten phrases and retired eighteen unused clips while retaining production numbers, days and required helpers. Its warning-free build, ten native host executables, 23 Python helper tests and five final-image validation tests passed. A further **50 focused status/snapshot/stream checks** passed on the installed device. The [observed-status record](firmware/esp32-rlcd/docs/VERIFICATION_OBSERVED_2026_10_09.md) separates those public-source checks from installed David-audio evidence and records completed playback/recovery; no new human listening confirmation was obtained.

The current saved indoor offset is the user's preferred **−4 °C**, and v28 now uses that as its fallback when no valid setting exists. The earlier v27 −4.8 °C evidence remains historical. Recognition drivers/models and the 0.80 command threshold are unchanged; the public eSpeak voice remains unflashed and acoustically untested.

## Audio recovery, 9 October 2026

The owner's v28 later lost both voice commands and KEY readout. I captured two faulted snapshots about 85 seconds apart: voice workers stayed stopped, while weather collection, sensors and the screen remained active. The failed microphone handoff then blocked the speaker. This establishes a permanent fault and coupled readout failure; it does not identify what first stopped the audio front end.

I deployed v29 and built [29-public.1](firmware/esp32-rlcd/README.md) with persistent workers, bounded recovery and explicit ownership. Pipeline reset requires both workers parked. Speaker-only handoff needs the feed worker stopped/paused and the microphone channel released, so KEY can work while recognition remains faulted. The public warning-free build, eleven native host executables, 23 Python helper tests and five flash-validation checks passed, including **120 checks of the actual production lifecycle methods/player**.

The installed build passed **64 dashboard checks and 88 recovery checks** over nearly six minutes: two real saturation faults, ten manual resets and **twelve successful repairs in twelve attempts**. Full readouts on all three pages completed, including overview playback during a fault, and the board finished ready with both microphones active. The [new verification record](firmware/esp32-rlcd/docs/VERIFICATION_AUDIO_RECOVERY_2026_10_09.md) gives timing and evidence limits. Forecast/model outputs, vocabulary, confidence and recordings are unchanged. I have no new human listening confirmation; this short stress run does not identify the original cause or establish an hours-long cure.

## Readout controls, 9 October 2026

I added a second KEY click to stop the current page's readout. While reading, the footer shows **Voice: reading**, centered Wi-Fi RSSI and **[KEY] Stop Reading**. Page 1 now displays and speaks the status body without **Observed:**, and tennis wind includes **km/h**. The server's status text and weather values, prerecorded clips, recognition models and command threshold are unchanged.

I deployed v30 and built the matching [30-public.1 source](firmware/esp32-rlcd/README.md#v30-verification-9-october-2026). Its warning-free build, eleven native host executables, 23 Python helpers and five flash-validation checks passed. The installed device passed **64 dashboard and 36 control checks**, covering a stop and full restart on each page, with microphones/recognition resumed and no reboot or AFE repair. I inspected the normal/reading footers and wind units. Its public eSpeak voice remains unflashed, and I have no new human listening confirmation; the earlier [v29 recovery record](firmware/esp32-rlcd/docs/VERIFICATION_AUDIO_RECOVERY_2026_10_09.md) retains its historical evidence.

## Reading-footer layout, 9 October 2026

At the owner’s request, I moved **[KEY] Stop Reading** to the left, kept **WiFi: RSSI** centered and restored **[BOOT] Next Page** on the right while reading. The **Voice: reading** indicator is removed. I deployed v31 and built **31-public.1** with no warnings. This change is limited to the footer and version; the existing readout stop control and voice/weather behavior remain. The installed device passed **23 focused checks**, including all three page footers during playback and a successful partial-stream stop. The [v31 verification section](firmware/esp32-rlcd/README.md#v31-verification-9-october-2026) scopes the build/image checks and installed-device evidence; the v30 results above remain historical.

## Forecast work, 8 October 2026

The October 8 service used **v24.0.2 / API 1.37.0**, published then as **v24.0.2-public.1**. I publish literal outputs from a fixed fresh-sequence Ridge model for the +90 point and ride mean/minimum/maximum, and a separate logistic classifier for ≥20/≥40 arrival changes. The first-sampled-crossing HGB output remains in the APIs; the web card focuses on arrival. The publisher does not replace a PM2.5 number with persistence or use a runtime performance selector. Morning and Afternoon retain their separate fixed models, with the Afternoon provisioning limitation documented in the release guide.

I repaired a refresh race that erased observed-movement text and made forecast labels show the probability and its reference clock. I verified numerical agreement across the browser API, Coach API, desk-device API and issued-forecast archive. Those checks establish software consistency. They do not establish advance-warning skill.

At the original **16:29:01 MYT** decision on October 8, the fresh reference was **156.8 µg/m³** and the completed +90 sensor proxy was **89.0**. The revised numerical model's later replay still predicted **150.7**. Its event replay gave a **72.24%** first-drop probability, a useful development clue, but that is neither an originally issued success nor a learned ≥40 probability. The [new chapter](docs/journal/2026-10-08-warning-before-the-ride.md) and [updated model card](docs/MODEL_CARD.md) give the paired results and evaluation limits.

This release publishes the actual October server/model code, its helper sources, portable tests, setup guide and a new first-person journal entry. **187 tests passed**, plus five empty-database HTTP route checks with provider networking and background workers disabled. Private observations and fitted assets remain excluded. Software consistency does not establish forecast accuracy; the [model card](docs/MODEL_CARD.md) reports mixed arrival-distribution results and rare-event limitations.

## The desk device, 8 October 2026

The owner's Waveshare **ESP32-S3-RLCD-4.2** now has a working three-page desk dashboard: outdoor PM2.5 and weather, a dated MTB ride forecast, and history with indoor temperature/humidity and battery information. It discovers the local server over the network, recognizes a compact vocabulary with pretrained Espressif speech models, and reads page summaries from locally stored speech clips.

This is a shared development space for the forecasting-server and embedded-firmware work. The [new hardware chapter](docs/journal/2026-10-08-building-a-talking-weather-desk.md) includes the owner's device photographs and actual **v24** framebuffers, describes the voice and audio engineering, and records what we tested. A subsequent **v25** readout update spoke the winning momentum outcome's percentage and shortened the rise/drop wording. That October 8 device/API release provided genuine predicted ride-window extrema at 15-minute resolution and a server-generated momentum label. The mean-uncertainty span described in the September snapshot was a different quantity.

The first [ESP32 firmware source publication](firmware/esp32-rlcd/README.md) was **v25-public.1** on October 8, under **Apache-2.0** for my original firmware and tools. It included the display, indoor sensors, voice commands, selected-page readouts, Wi-Fi/server discovery, host tests and a clearly marked synthetic API example. Build dependencies are pinned and downloaded separately. Public readback audio is generated with eSpeak NG; the installed device's Microsoft David recordings and Espressif model binaries are not distributed. See the [component licenses and provenance](firmware/esp32-rlcd/THIRD_PARTY_NOTICES.md).

I checked a fresh dependency installation, public-source compilation and host regressions for that v25 publication. The public voice variant was not flashed or acoustically tested on the owner's device; the hardware results in the chapter belong to the earlier installed versions. The October 9 additions above identify newer firmware checks. The server and firmware have their own dated source releases and verification scopes.

## Forecast work, 5 October 2026

I now run the owner's local dashboard as a Windows service, so collection continues independently of my development session. That deployment uses **v19.2.0 / API 1.28.1**. I retain recent-sensor persistence for the +90-minute point and +90–210-minute mean; the owner selected the experimental `weather_session_change_v1` model for Morning and Afternoon.

I added an hourly archive of nine CAMS regional model cells to test incoming pollution alongside wind vectors, rotation and transport. I preserve network retrieval and database availability as separate clocks. These new inputs are diagnostic and do not change the published PM2.5 numbers.

By October 5, I had three complete collection days. The regional near-term candidate's October 4 advantage over persistence did not hold in the following morning's completed cases, while the regional additions increased error in the small session comparison. Sudden-change prediction remains unresolved. The [October 5 journal entry](docs/journal/2026-10-05-learning-from-the-forecast-misses.md) and [model card](docs/MODEL_CARD.md) give the scores and limits.

This October 5 record describes that earlier deployment and study. The current source release supersedes the initial September snapshot; the historical chapters retain their original dates and evidence. The private Windows service setup is not bundled.

## Run the forecasting server

Use Python **3.11**, the tested runtime. From the repository root:

```bash
python -m venv .venv
# Activate the environment using your shell's normal command, then:
python -m pip install -r requirements.txt
python app/BukitKiara_Dashboard.py --no-browser
```

Open [the local dashboard](http://127.0.0.1:8765). A new database is created beside the script, or in `BUKIT_KIARA_DATA_DIR` when configured. A fresh installation has to collect observations and forecast vintages before the learned models have adequate support. The server collects the neighbour stations itself; the v25 point and ride heads need at least 240 completed 15-minute training origins with neighbour data, about three days. Missing model outputs remain unavailable. Read the [release guide](docs/SERVER_RELEASE_2026_10_08.md) for the optional Torch runtime and Afternoon's additional missing-asset limitation.

The collector runs while the Python process is running, even if every browser is closed. Sleeping or stopping the host interrupts collection. This repository does not install a background service automatically.

The server binds to **127.0.0.1** by default. `BUKIT_KIARA_HOST` and `BUKIT_KIARA_PORT` configure the listener; `BUKIT_KIARA_DISCOVERY=1` opts into LAN discovery. The public TTDI station and weather coordinates are configurable. Read [security and LAN access](SECURITY.md) before changing the host to `0.0.0.0`. This is not an internet-hardened web service.

## Inspect, test, contribute

```bash
python -m unittest discover -s tests -v
```

- [Model card and evaluation limits](docs/MODEL_CARD.md)
- [Model research harness and results](research/models/README.md)
- [Data sources and attribution](docs/DATA_SOURCES.md)
- [Public snapshot and privacy boundary](docs/PUBLICATION.md)
- [Contribution principles](CONTRIBUTING.md)

Node.js is required for the tests executing the real embedded dashboard JavaScript. The [test guide](tests/README.md) explains the synthetic/temp-data suite and its limits.

Original forecasting application, test and research source (`app/`, `tests/`, `research/models/`) uses [Apache-2.0](app/LICENSE.md). The firmware in [`firmware/esp32-rlcd`](firmware/esp32-rlcd) has its own [Apache-2.0 license](firmware/esp32-rlcd/LICENSE) for original code. Its [THIRD_PARTY_NOTICES.md](firmware/esp32-rlcd/THIRD_PARTY_NOTICES.md) covers the included Waveshare and Espressif adaptations, the on-device trainers that follow BSD-3-Clause scikit-learn and SciPy code, and the separately installed dependencies. These code grants do not relicense journal prose, other documentation, photographs or third-party data.

**Source status, 10 November 2026:** limited sudden-change warning evidence and no prospective accuracy claim; no official endorsement by any data provider, Anthropic or OpenAI. AI agents develop and review the software. The forecasts come from statistical models: Python on the server and their C++ ports on the desk display. Neither calls an LLM for a prediction.

