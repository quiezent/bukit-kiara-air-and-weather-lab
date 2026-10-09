# ESP32 reflective weather dashboard

Source snapshot for the **Waveshare ESP32-S3-RLCD-4.2**: three weather pages, local indoor temperature/humidity, button navigation, offline voice commands and spoken page summaries. The board needs its 16 MB flash and 8 MB OPI PSRAM configuration. The firmware obtains outdoor data from a compatible LAN server; it does not run the forecasting models.

The October 9 public variant is **`weather-dashboard-27-public.1`**, adding an adjustable indoor temperature correction and corresponding relative-humidity compensation. Its warning-free build, ten C++ host executables, 23 Python helper tests and five flash checks passed; its eSpeak NG voice has not been flashed or audited on hardware. Page 1 retains the server's current sensor-status text and native +90-minute arrival-change headline/probability; Page 3 supports up to 128 history points within the unchanged 16 KiB response limit. The installed David-audio firmware is a separate build. Compiling and host tests do not establish microphone recognition accuracy, speaker audibility or forecast skill.

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

It also verifies the unchanged official ESP-SR model bundle. Public speech preparation uses **125 ordered clips** in native `en-us`: the original 120, four arrival-outcome phrases and **Demonstration data**. It creates the PCM source and a hash manifest. The demonstration clip's enum position changed after public v25; regenerate speech and rebuild the matching source together rather than reusing an older generated file. Speech is finite, prerecorded phrase-and-number composition; eSpeak does not run on the board and this is not arbitrary text-to-speech. Phrase definitions are in [assets/speech/phrases.json](assets/speech/phrases.json).

The build uses the included custom partition table, hardware USB CDC, QIO flash and OPI PSRAM. Generated speech, model data, SDK caches, build logs, host reports and firmware images are ignored build artifacts. `build/firmware/build-manifest.json` records the public build; `build/host/host-verification.json` records host checks.

The [October 8 verification record](docs/VERIFICATION.md) is historical public **v25** evidence: a warning-free build, seven host C++ executables, 21 helper tests, five flash-validation tests and matching independent speech generations. The [October 9 v26 record](docs/VERIFICATION_2026_10_09.md) covers its warning-free build, eight C++ executables, 23 helper tests and five flash checks. The [v27 indoor-correction record](docs/VERIFICATION_INDOOR_2026_10_09.md) gives the current build/host results and separately installed-device checks; public-voice hardware validation remains separate.

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

Release **KEY** to read the selected page; release **BOOT** for the next page. Voice recognition runs locally with MultiNet 7 English and VADNet, **without a wake phrase**, using the same **0.80** detector/action threshold.

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

Page 1 reads outdoor PM2.5/temperature, current rain, the +90-minute estimate and native **arrival-change** winner/probability, then next-hour rain. The screen's sensor status comes from `current.display_text`; the arrival card uses `near90.arrival_change.display_text` and its validated native probability. First-crossing probabilities remain a different target and are never an arrival fallback. An exact arrival tie says **Arrival outcome uncertain**, without a percentage. Page 2 reads the ride range or mean, mean-≤70 probability, rain and dated morning/afternoon comparisons, with rain warnings above 75%. Page 3 reads the observed PM summary, indoor readings/minimum and coarse voltage-based battery estimate. API timestamps and unavailable/stale states are preserved.

## Indoor board-heat correction

The SHTC3 sensor is factory calibrated, but heat from the assembled board can affect its room reading. Public v27 defaults to a **−4.8 °C** temperature offset when no saved setting exists. I selected it for the owner's normal desk use outside direct airflow from an earlier handheld comparison: the board read 30.8 °C beside a Dyson reading 26 °C. That was not a controlled desk calibration. A later airflow comparison showed 30.2 °C/60.2% RH on the board versus 28 °C/72% RH on the Dyson, a smaller 2.2 °C temperature gap. This is a placement-dependent reference estimate, not traceable calibration or a universal offset for other boards. Compare your own unit with a reference in its normal position before choosing an offset. The [v27 verification record](docs/VERIFICATION_INDOOR_2026_10_09.md) keeps that provenance separate from software and device checks.

Relative humidity is adjusted from the raw temperature/RH using equation 1 of Sensirion's [design guide](https://sensirion.com/resource/user_guide/sht/design_in/), with `m = 17.62` and `Tn = 243.21 °C`; there is no independent RH trim. Corrected RH is limited to 0–100%. Every indoor display and page readout uses the corrected values; outdoor API readings are unchanged. Raw sensor measurements remain available through the board's `/status` endpoint.

The offset accepts finite values from **−10 to +10 °C**, persists in NVS and can be changed using the local [indoor-correction endpoint](docs/API.md#local-board-indoor-correction). Set it to **0** to use the original sensor values without this temperature/RH correction. Changing it resets the corrected temperature and RH minima/maxima so ranges do not mix different settings.

## License and scope

Original dashboard firmware, tools and phrase definitions in this package use [Apache-2.0](LICENSE). Included Waveshare/Espressif adaptations retain their notices; external libraries, fonts and tools keep their separate terms. Read [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [LICENSES/](LICENSES/) before redistribution.

ESP-SR's separate **Espressif-hardware-restricted** license applies to its original pretrained weights; this project's Apache license does not relicense them. Preparation downloads/verifies that dependency separately. No Microsoft David recordings, generated PCM/WAVs, model binary, SDK/tool executable or compiled application binary is shipped in this source publication. Other repository components retain their own licenses and snapshot scope.
