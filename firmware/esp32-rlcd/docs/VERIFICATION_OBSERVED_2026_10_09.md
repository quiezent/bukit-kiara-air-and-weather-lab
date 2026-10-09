# Observed-status speech verification — 9 October 2026

The owner reported that Page 1 displayed an **Observed:** status but its readout skipped that sentence. I added its recorded equivalent immediately after current outdoor PM2.5 and before outdoor temperature. The public source is **`weather-dashboard-28-public.1`**, with passing build and host checks. The separately installed David-audio v28 passed dashboard and Page 1 playback/recovery checks. The [v27 indoor record](VERIFICATION_INDOOR_2026_10_09.md) retains its earlier −4.8 °C default, comparisons and calibration limits as dated evidence.

## What the readout captures

The original [ObservedStatus.h](../WeatherDashboard/ObservedStatus.h) maps eight exact server **Observed:** labels and the neutral **Latest sensor reading** label to fixed PCM phrases. The [API mapping](API.md#observed-status-speech) lists all spellings and spoken words. Unknown, missing or malformed text with otherwise eligible current PM says **Observed status unavailable**. The firmware does not invent a classification or put unsupported server text into a transcript without matching audio.

Source-clock, availability and PM guards remain in the production readout. Expired/future observations and invalid concentrations suppress status speech. Eligible old data retains one **Outdoor data is old** announcement before current PM. The status text and clip IDs are captured with the selected page, so later polling cannot change words while a narration is playing. Native arrival probabilities and first-crossing meanings are unchanged.

## Audio scope and configuration

Ten new phrases cover the eight observed bodies, neutral label and unavailable fallback. Eighteen unused legacy clips were retired after checking production references. Number, weekday and required helper clips remain. The private Microsoft David-audio build vocabulary has **116 clips**, with **106 retained clips byte-identical** to their earlier PCM. Its generation metadata records the retired hashes; retired source/processed WAVs remain local. None of those recordings are published.

I regenerated the public **117-clip** eSpeak NG bank, including **Demonstration data**, totaling **4,088,222 PCM bytes**. Source/manifest/header must be rebuilt together because retirement changes enum positions. Generated audio, upstream model weights, compiled images and private raw reports remain outside this source publication.

The microphone/speaker drivers, speech models, twelve command phrases and confidence **0.80** are unchanged. The current user-selected saved temperature offset is **−4.0 °C**; this version's fallback for boards without a valid saved setting now matches it. NVS settings are retained. The compensation formula and no-independent-RH-trim behavior are unchanged; the new preference does not establish a traceable calibration.

## Public source build

| Check | Result |
| --- | --- |
| Public firmware compile and compiled-source agreement | Passed, zero warnings; all 38 C++/header files matched final source |
| Production observed-status/narration/freshness regressions | Passed in the production readout host tests |
| Existing voice, arrival, indoor and capacity host regressions | All 14 host-runner steps / ten native executables passed |
| Demo/provisioning helper tests | 23 Python tests passed |
| Compiled-image/partition/model validation | Five final-image tests and guarded-flasher dry run passed |
| Speech generation/117-ID consistency | Passed; 117 regenerated clips / 4,088,222 PCM bytes |

The public application image is **7,407,824 bytes**, SHA-256 `6a4b54e47f8798357f302e97f6f24bae63e01fcd7ccaa2b2cd9ff187ce80bc23`. These identify the checked build, rather than a binary guaranteed identical on every host.

The public eSpeak variant remains **unflashed and acoustically untested**. Its host/build results cannot establish speaker audibility or command-recognition accuracy.

## Installed desk device

I installed **`weather-dashboard-28-observed-readout`**. Its build had zero warnings and all 38 compiled C++/header files matched final source. All four flashed image hashes were verified, with NVS/PHY regions protected. The application is **7,866,272 bytes**, SHA-256 `6756c468c1757a4d8e2cf1555fce550f9b5c056c3d665b2ab494840e98d2736d`.

| Check | Result |
| --- | --- |
| Three-page dashboard checks | 64 passed |
| Page 1 playback and recovery checks | 29 passed |
| Saved indoor setting | −4 °C retained across the hard reset |
| Display captures | All three actual framebuffers inspected; labels fit without overlap |
| Focused status/snapshot/stream audit | 50 checks passed |

The captured current status was **Observed: Particle rebound may be starting**. The snapshot began **Current outdoor PM2.5 96.2. Observed: Particle rebound may be starting. Outdoor temperature 28.9 degrees Celsius.** The fixed status played after the current PM value and before outdoor temperature. The complete **435,049-sample** stream finished in **28.235 seconds**, against its **27.19-second** audio plan. The speaker driver completed and both microphone/recognition workers recovered. The overview was restored and status ready afterward. Its observed label wrapped across two lines; the sports and history framebuffers also remained readable.

No new human listening confirmation, command-recognition accuracy trial or physical calibration evidence was obtained. Only Page 1 received a new live readout check for this fix; earlier v26 arrival/history and v27 indoor playback results remain evidence for those versions. Installed David-audio completion/recovery checks cannot establish public eSpeak audibility.
