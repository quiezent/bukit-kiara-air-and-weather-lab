# ESP32 reflective weather dashboard

Source snapshot for the **Waveshare ESP32-S3-RLCD-4.2**: three weather pages, local indoor temperature/humidity, button navigation, offline voice commands and spoken page summaries. The board needs its 16 MB flash and 8 MB OPI PSRAM configuration. The firmware obtains outdoor data from a compatible LAN server; it does not run the forecasting models.

The October 9 public variant is **`weather-dashboard-31-public.1`**. While reading, the footer shows **[KEY] Stop Reading** on the left, **WiFi: RSSI** in the center and **[BOOT] Next Page** on the right, with no **Voice: reading** indicator. The v30 stop control, prefix-free current status and tennis wind **km/h** remain. This v31 change affects only footer presentation and the version; weather data, audio, voice commands/models and the 0.80 threshold are unchanged. Its checks are scoped below; the v30 build/host and installed-device evidence remains historical. The public eSpeak NG voice remains unflashed and acoustically untested; compilation and host tests do not establish recognition accuracy, speaker audibility or forecast skill.

## Build from source

Use **Python 3.11+**, [**Arduino CLI 1.5.1**](https://github.com/arduino/arduino-cli/releases/tag/v1.5.1), a native [**eSpeak NG 1.52.0**](https://github.com/espeak-ng/espeak-ng/releases/tag/1.52.0) executable and a C++17 host compiler such as GCC/Clang. Install the native executables separately; they are not shipped here. Run the commands below from this directory with your Python environment active. Replace the quoted executable paths with your installations.

```sh
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` on Linux/macOS, then:

```sh
python -m pip install -r requirements.txt
python tools/setup.py --arduino-cli "/path/to/arduino-cli"
python tools/prepare_speech.py --espeak-ng "/path/to/espeak-ng"
python tools/build.py
python tools/test_host.py --cxx g++
python -m unittest discover -s tools/tests -v
python WeatherDashboard/tools/flash_firmware.py --build-dir build/firmware --dry-run
```

On Windows, the executable paths can point to `arduino-cli.exe` and `espeak-ng.exe`. The speech generator also accepts `--espeak-data PATH` for a separately extracted `espeak-ng-data` directory. Compiler-driver arguments can be supplied repeatedly; a Windows Zig example is:

```sh
python tools/test_host.py --cxx zig --cxx-arg=c++ --cxx-arg=-target --cxx-arg=x86_64-windows-gnu
```

Before compiling the firmware, host tests can explicitly skip compiled-image checks with `--skip-flash`; run them again without that flag after the build.

`setup.py` checks the CLI version and prepares the pinned dependencies in ignored `.state/` directories:

| Dependency | Version |
| --- | --- |
| Official Arduino-ESP32 | 3.3.12 |
| Bundled ESP-IDF / ESP-SR | 5.5.5 / 2.5.3 |
| Official U8g2 / ArduinoJson | 2.36.18 / 7.4.3 |
| Python esptool / pyserial / zeroconf | 5.3.1 / 3.5 / 0.148.0 |

It also verifies the unchanged official ESP-SR model bundle. Public speech preparation uses **117 ordered clips** in native `en-us`, including the ten observed-status phrases introduced in v28 and **Demonstration data**. Eighteen unused legacy phrases were retired in v28; production number, weekday and required helper clips remain. V29 through v31 change no phrase definitions or recordings. The generator creates matching PCM source and a hash manifest. Retirement changed enum positions, so regenerate speech and rebuild the matching source together rather than reusing a pre-v28 generated file. Speech is finite, prerecorded phrase-and-number composition; eSpeak does not run on the board and this is not arbitrary text-to-speech. Phrase definitions are in [assets/speech/phrases.json](assets/speech/phrases.json).

The build uses the included custom partition table, hardware USB CDC, QIO flash and OPI PSRAM. Generated speech, model data, SDK caches, build logs, host reports and firmware images are ignored build artifacts. `build/firmware/build-manifest.json` records the public build; `build/host/host-verification.json` records host checks.

The [October 8 verification record](docs/VERIFICATION.md), [v26 arrival record](docs/VERIFICATION_2026_10_09.md), [v27 indoor-correction record](docs/VERIFICATION_INDOOR_2026_10_09.md), [v28 observed-status record](docs/VERIFICATION_OBSERVED_2026_10_09.md) and [v29 recovery record](docs/VERIFICATION_AUDIO_RECOVERY_2026_10_09.md) retain their historical build, host and installed-device scopes. Public-voice hardware validation remains separate.

### V31 verification, 9 October 2026

The public build compiled without warnings, with all **38** compiled C++/header files matching the final source. All **five final-image validation checks** and the no-port flash dry run passed. The 117-clip speech bank, phrase definitions and upstream model bundle are unchanged. Broader host/audio tests were not repeated for this footer-only change; the v30 results below remain historical evidence.

The separately installed David-audio v31 passed **23 focused checks** in **5.969 seconds**. Normal and reading framebuffers for all three pages were inspected while one overview readout stayed active through page navigation: **[KEY] Stop Reading** remained left, Wi-Fi centered and **[BOOT] Next Page** right, with no **Voice: reading** indicator. USB `KEY`, which uses the physical-button handler, stopped the partial stream without an error; the quiet tail and microphone capture resumed, the normal **[KEY] Read Info** footer returned and overview was restored. No new full-stream, physical-button or human-listening test is claimed.

Public application: **7,413,136 bytes**, SHA-256 `8b98ec526626ca62fc96ac06633bfc5a68ea62cc798827dec3004c6372ed430a`. The public eSpeak image remains unflashed; compiled images and private verification payloads are excluded.

### V30 verification, 9 October 2026

The public build compiled without warnings, with all 38 compiled C++/header files matching the final source. All 16 host-runner steps / eleven native executables passed without host warnings, including **175 checks of the production lifecycle/player** and **1,383 explicitly counted assertions** across four suites. The 23 Python helper tests, five final-image checks and no-port flash dry run passed. The 117-clip public speech bank is unchanged.

The separately installed David-audio v30 passed **64 dashboard and 36 readout-control checks**: all three intentional stops and all three full restarts completed, without AFE repair, recovery failure or reboot. Microphones and the recognition quiet-tail handoff resumed after each, and overview was restored. Normal/reading framebuffers were inspected for the footer and wind units. These checks add no human listening confirmation and cannot validate the unflashed public eSpeak voice.

Public application: **7,413,376 bytes**, SHA-256 `9bcdf9d4e7c27d3e941fe42427d7e510be3010d6b85609b0d918b949ac87133f`. Compiled images and private verification payloads are excluded from this source package.

## Flash and provision

Connect the board by USB and select its actual serial port. The following **writes firmware**; replace `YOUR_SERIAL_PORT` with, for example, `COM5` or `/dev/ttyACM0`:

```sh
python WeatherDashboard/tools/flash_firmware.py --build-dir build/firmware --port YOUR_SERIAL_PORT
python tools/provision.py wifi --port YOUR_SERIAL_PORT
python tools/provision.py status --port YOUR_SERIAL_PORT
```

Use the validated flasher rather than a generic upload/erase command. It checks the ESP32-S3 image headers, exact partition layout, checksums and model hash, then writes the bootloader, partition table, application and model. Its write ranges preserve existing NVS and PHY data. A dry run opens no serial port.

Provisioning asks for a **2.4 GHz** SSID and prompts for the password without echoing it. It does not retrieve a saved computer password or embed credentials in the source. Credentials are stored in the board's NVS. `scan` lists nearby networks; `discover` requests server discovery:

```sh
python tools/provision.py scan --port YOUR_SERIAL_PORT
python tools/provision.py discover --port YOUR_SERIAL_PORT
```

## Try synthetic data

The checked-in [example](examples/rlcd-v1-synthetic.json) contains invented values, not observations or trained-model predictions. This preview refreshes its clocks without changing that distinction:

```sh
python tools/demo_server.py
```

It binds only `127.0.0.1:8766`; inspect `http://127.0.0.1:8766/api/rlcd/v1` locally. To make the demo discoverable by a board on the same LAN, explicitly supply this computer's LAN IPv4 address:

```sh
python tools/demo_server.py --host 0.0.0.0 --advertise-ip YOUR_LAN_IPV4
```

Stop other services advertising the same service type during this test: the board can choose any compatible discovered server. Synthetic payloads carry `demo: true`, show **DEMO WEATHER** and prepend **Demonstration data** to spoken summaries. Stop the demo with Ctrl+C. It neither queries weather APIs nor evaluates a model.

For your own server, follow [docs/API.md](docs/API.md). DHCP changes are handled through mDNS/DNS-SD; no server address is hardcoded. The HTTP client requires an explicit response length of at most **16 KiB**.

## Buttons, voice and audio

Release **KEY** to read the selected page; release it again during playback to stop reading. Release **BOOT** for the next page. While reading, the footer shows **[KEY] Stop Reading** on the left, **WiFi: RSSI** in the center and **[BOOT] Next Page** on the right, with no **Voice: reading** indicator. The newline-terminated USB command `KEY` uses the same start/stop action as the physical button. Voice recognition runs locally with MultiNet 7 English and VADNet, **without a wake phrase**, using the same **0.80** detector/action threshold.

| Say | Action |
| --- | --- |
| `next`, `next page` | Next page |
| `back` | Previous page |
| `read info`, `read page`, `read` | Read the selected page |
| `overview`, `page one` | Page 1 |
| `forecast`, `page two` | Page 2 |
| `graph`, `page three` | Page 3 |

These are 12 spoken phrases with five internal pronunciation variants. Pause between commands and wait for playback to finish before speaking again.

To customize commands, edit the supplied [VoicePhrases.h](WeatherDashboard/VoicePhrases.h) entries and update its count assertion. Each entry maps text and phonemes to an action; internal variant labels share the same action, and users say the ordinary command words. New actions also need routing in [VoiceNavigation.h](WeatherDashboard/VoiceNavigation.h). This changes the command vocabulary without personal voice enrollment or acoustic-model retraining; test recognition after rebuilding.

The existing phonemes originated from Espressif's [pinned offline G2P tool](https://github.com/espressif/esp-sr/blob/a2bc8a64d9995155024dd871bce2be28dd1794d1/tool/multinet_g2p.py), following its [MultiNet 7 customization guide](https://docs.espressif.com/projects/esp-sr/en/latest/esp32s3/speech_command_recognition/README.html#multinet7-customize-speech-commands). That upstream tool is a separate Espressif-restricted dependency and is not copied into this package. Primary `read` phrases explicitly use imperative **REED** (`RmD`); `read` and `read page` retain **RED** (`RfD`) aliases. Preserve those intentional choices when regenerating phonemes.

Both physical microphones are captured and BSS is enabled. In this pinned SDK's WakeNet-off branch, however, MultiNet receives the original first microphone rather than a separated BSS output. **AEC and NSNET are disabled**. Speaker playback suspends microphone capture, then restores it with a quiet interval; simultaneous echo-cancelled recognition is not implemented.

V29 parks both persistent feed/detection workers before a coordinated in-place pipeline reset, with bounded retries/backoff and stale-event suppression. KEY speaker handoff requires the feed worker parked or healthy capture paused, plus verified microphone-channel release; it can proceed while the detector remains faulted. If the acknowledgements required for an operation are missing, its resources remain intact and failure is reported. The board's [local diagnostics](docs/API.md#local-board-audio-diagnostics) expose recovery and preserved fault state, with USB-only reset/stall controls for testing. Forced-stall recovery tests assess those transitions; they cannot identify or prove a lasting cure for the spontaneous v28 failure.

Page 1 reads current outdoor PM2.5, the matching observed-status body, outdoor temperature, current rain, the +90-minute estimate and native **arrival-change** winner/probability, then next-hour rain. It captures the unchanged `current.display_text` with the page snapshot and speaks its fixed recorded body immediately after the current PM value, omitting only the **Observed:** prefix from display and speech. The [API mapping](docs/API.md#observed-status-speech) covers eight known **Observed:** labels plus **Latest sensor reading**; unknown/malformed text with otherwise eligible current data still says **Observed status unavailable**. Expired/future observations and invalid PM values suppress that status. The existing outdoor-old announcement is spoken once for eligible old data. The tennis wind display includes **km/h**.

The arrival card uses `near90.arrival_change.display_text` and its validated native probability. First-crossing probabilities remain a different target and are never an arrival fallback. An exact arrival tie says **Arrival outcome uncertain**, without a percentage. Page 2 reads the ride range or mean, mean-≤70 probability, rain and dated morning/afternoon comparisons, with rain warnings above 75%. Page 3 reads the observed PM summary, indoor readings/minimum and coarse voltage-based battery estimate. API timestamps and unavailable/stale states are preserved.

## Indoor board-heat correction

The SHTC3 sensor is factory calibrated, but heat from the assembled board can affect its room reading. The current firmware defaults to the user's preferred **−4.0 °C** temperature offset when no valid saved setting exists. Existing NVS settings are retained. The earlier **−4.8 °C** starting value and informal handheld/airflow comparisons belong to the [historical v27 record](docs/VERIFICATION_INDOOR_2026_10_09.md). These settings are placement-dependent estimates without a traceable system-calibration claim. Compare your own unit with a reference in its normal position before choosing an offset.

Relative humidity is adjusted from the raw temperature/RH using equation 1 of Sensirion's [design guide](https://sensirion.com/resource/user_guide/sht/design_in/), with `m = 17.62` and `Tn = 243.21 °C`; there is no independent RH trim. Corrected RH is limited to 0–100%. Every indoor display and page readout uses the corrected values; outdoor API readings are unchanged. Raw sensor measurements remain available through the board's `/status` endpoint.

The offset accepts finite values from **−10 to +10 °C**, persists in NVS and can be changed using the local [indoor-correction endpoint](docs/API.md#local-board-indoor-correction). Set it to **0** to use the original sensor values without this temperature/RH correction. Changing it resets the corrected temperature and RH minima/maxima so ranges do not mix different settings.

## License and scope

Original dashboard firmware, tools and phrase definitions in this package use [Apache-2.0](LICENSE). Included Waveshare/Espressif adaptations retain their notices; external libraries, fonts and tools keep their separate terms. Read [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [LICENSES/](LICENSES/) before redistribution.

ESP-SR's separate **Espressif-hardware-restricted** license applies to its original pretrained weights; this project's Apache license does not relicense them. Preparation downloads/verifies that dependency separately. No Microsoft David recordings, generated PCM/WAVs, model binary, SDK/tool executable or compiled application binary is shipped in this source publication. Other repository components retain their own licenses and snapshot scope.
