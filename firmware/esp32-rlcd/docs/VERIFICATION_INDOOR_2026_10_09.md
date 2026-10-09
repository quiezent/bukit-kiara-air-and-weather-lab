# Indoor correction verification — 9 October 2026

I added an adjustable indoor temperature correction and corresponding relative-humidity compensation in the ESP32 firmware workstream. The public source is **`weather-dashboard-27-public.1`**, with passing build and host checks. The separately installed David-audio v27 passed correction, reset-persistence, dashboard and Page 3 playback checks. The [v26 arrival/history verification](VERIFICATION_2026_10_09.md) and [v25 source verification](VERIFICATION.md) remain dated evidence for those versions.

## Reference estimate and its limits

The owner's normal operating position is on a desk outside direct airflow. I chose the starting **−4.8 °C** offset for that use from an earlier **handheld** comparison: the board showed **30.8 °C** while a nearby Dyson showed **26 °C**. This was not a controlled desk calibration. The factory-calibrated SHTC3 retains its original measurement conversion; this is an adjustable estimate for heat and placement effects in the assembled device, without a traceable system-calibration claim.

In a later airflow comparison the board showed **30.2 °C / 60.2% RH**, while the Dyson showed **28 °C / 72% RH**. Its **2.2 °C** temperature difference supports keeping the offset placement-dependent. I did not fit a separate RH trim to make either pair agree. These observations do not establish the correct offset for another board, a different power state or a changed location. Users should compare their own board and reference under its normal conditions; offset **0** disables the additional correction.

## Implemented calculation and configuration

The original [IndoorCompensation.h](../WeatherDashboard/IndoorCompensation.h) helper uses equation 1 of Sensirion's [Design Guide for Humidity and Temperature Sensors](https://sensirion.com/resource/user_guide/sht/design_in/), with `m = 17.62` and `Tn = 243.21 °C`:

```text
Tcorrected = Traw + offset
RHcorrected = RHraw × exp(17.62 × 243.21 × (Traw − Tcorrected)
                         / ((243.21 + Traw) × (243.21 + Tcorrected)))
```

The exponent is evaluated in double precision. Compensated RH is limited to **0–100%**, with a status flag if limiting was required. No independent RH offset is applied. Offset **0** preserves the exact original temperature and RH. The helper rejects nonfinite values, offsets outside **[−10,+10] °C**, invalid raw RH, and raw or corrected temperatures outside the sensor's **−40 to +125 °C** range.

All indoor displays and spoken summaries use the corrected readings. The board's `/status` preserves raw temperature/RH and reports the active offset, compensation/clamp flags, method `board-heat-magnus-v1`, estimate flag and saved-setting state. Outdoor observations, forecast values and history remain unchanged.

The local form-encoded `POST /indoor-correction` setting persists in NVS. Changing it resets corrected temperature and RH ranges, without refreshing a measurement clock or reviving an expired sample. The [API guide](API.md#local-board-indoor-correction) documents the request, finite bounds and response codes.

## Public source build

I built the public package with its pinned dependencies and reused its unchanged **125-clip eSpeak NG** vocabulary. The indoor correction adds no new spoken phrases; no new speech generation was performed for v27. The official speech-model hash is unchanged.

| Check | Result |
| --- | --- |
| Public firmware compile and compiled-source agreement | Passed, zero warnings; all 37 C++/header files matched source |
| Production host regressions | Ten C++ executables passed, zero warnings |
| Explicitly counted host assertions | 159 native-arrival + 492 compensation + 557 sensor-lifecycle = 1,208 checks |
| Demo/provisioning helpers | 23 Python tests passed: 12 demo, 11 provisioning |
| Compiled-image/partition/model validation | Five tests and guarded-flasher dry run passed |
| Speech vocabulary | Unchanged 125-clip eSpeak assets reused |

The assertion total covers the three suites that print counts; the remaining host executables also assert their behavior without an aggregate count. Tests cover compensation bounds/nonfinite inputs, zero-offset identity, unclamped moisture/dew-point consistency, RH clamping, measurement freshness, correction changes and sensor-range resets, alongside the existing voice, native-arrival, narration and capacity regressions.

The public application image is **7,749,840 bytes**, SHA-256 `efbbca2daa66d1562b78d1f1f9d6bba7c388131ed811882d5ee44c9beb8414c1`. These values identify the checked build, rather than a binary guaranteed identical on every host.

The public eSpeak variant is **unflashed and acoustically untested**. Compilation and host tests do not establish its microphone-command accuracy or speaker audibility. Generated audio, upstream weights, compiled images and private raw reports remain excluded from the source publication.

## Installed desk device

I installed **`weather-dashboard-27-indoor-compensation`**, retaining the device's Microsoft David audio and existing voice configuration. Its build had zero warnings and matched all 37 compiled C++/header files to source. The four written image hashes were verified, with NVS and PHY regions protected. The final application image is **8,268,400 bytes**, SHA-256 `11c054df1b5327adbf0360d1ea5616f67f41fdff8bc5fc83402842166ee1f08f`; this is the final build with the **since reset** range label.

| Check | Result |
| --- | --- |
| Initial live indoor checks | 23 passed |
| Full correction checks | 294 passed, including 13 invalid inputs, zero-offset disable and −4.8 restore |
| Post-reflash hard-reset persistence | 25 read-only checks passed; saved −4.8 °C survived |
| Three-page dashboard checks | 64 passed |
| Display captures | All three actual framebuffers visually inspected |
| Page 3 spoken summary and audio recovery | 27 checks passed: 24 existing checks and three corrected-value snapshot audits |

An initial checked sample was **32.21 °C / 54.71% RH raw**, corrected to **27.41 °C / 72.10% RH** with saved offset −4.8 °C. After the final reflash and hard reset, the saved setting survived without a correction POST. An overview capture showed **27.6 °C / 71.8% RH**; Page 3's corrected ranges and **since reset** label were readable.

The Page 3 stream finished in **32.0 seconds**, with zero backpressure. Both audio workers advanced and microphone capture recovered. Its spoken indoor values were **27.5 °C**, minimum **27.5 °C**, and **72.0% RH**, matching the captured corrected snapshot of **27.54 °C**, minimum **27.45 °C**, and **72.00% RH** at one-decimal narration precision. The overview was restored afterward. The final raw **32.42 °C / 54.41% RH** sample became **27.62 °C / 71.67% RH**; sensor errors remained zero and voice status healthy. These are dated device readings and playback checks, not comparison measurements establishing room accuracy.

The v27 sensor change affects the indoor portion of Page 3 speech. No new live Page 1/2 spoken tests were run; v26's Page 1 playback and history checks remain historical evidence. No new listening confirmation, spoken-command accuracy trial or physical calibration evidence was obtained. Installed David-audio results cannot establish public eSpeak voice audibility.

Correct implementation and data agreement do not establish room-measurement accuracy, forecast skill or a traceable calibration.
