# Audio recovery verification — 9 October 2026

I built **`weather-dashboard-29-public.1`** and flashed the separate installed-device **`weather-dashboard-29-audio-recovery`** build to recover voice transport and KEY readout after an audio fault. Builds, production-host tests, 64 dashboard checks and 88 live recovery checks passed. The [v28 observed-status record](VERIFICATION_OBSERVED_2026_10_09.md) remains evidence for its initial build/playback checks, which did not establish hours-long audio reliability.

## The failure I captured

Two v28 status captures at uptime **9,822 and 9,907 seconds** showed voice `ready=false` with an audio-feed fault. Accepted-feed and fetched-frame counts stayed frozen while the weather API, indoor sensors and display remained live. KEY readout failed because the stopped voice runtime could not complete the microphone-to-speaker pause. This establishes a permanent audio fault and a coupled speaker handoff failure.

The captures logged **47 zero feed returns**, but they did not preserve the original failing return or fault timestamp. They do not establish why AFE first stopped, that the returns were consecutive, or that playback/ring saturation caused the spontaneous event. The private raw fault captures remain excluded from this publication.

## SDK ownership and recovery

The pinned [ESP-SR 2.5.3 component](https://github.com/espressif/esp-sr/blob/0a6e15d5ea9c7f5402093439f3af4c9250ae0d8d/idf_component.yml) exposes feed, timed fetch, buffer reset and VAD reset in its [AFE interface](https://github.com/espressif/esp-sr/blob/0a6e15d5ea9c7f5402093439f3af4c9250ae0d8d/include/esp32s3/esp_afe_sr_iface.h). That interface supplies no guarantee of safe algorithm reset concurrent with feed/fetch. The audit of the exact installed implementation also found a mutex wait before the fetch read timeout, so a fetch timeout alone is not proof that a worker has left every SDK call.

V29 retains the two worker tasks/model allocations and requires **both** explicit park acknowledgements before an in-place ring/VAD/MultiNet pipeline reset and microphone restart. Recovery uses deadlines, bounded retries and backoff. If a required worker cannot park, its resources remain intact; forcing deletion/reset against an active SDK call is not a valid recovery. Quiet-tail, generation-local state reset and stale-command/diagnostic suppression apply before recognition resumes.

Speaker-only handoff has a narrower ownership requirement: the **feed worker must be parked or healthy capture paused**, and the **microphone channel must be verified released**. The detector need not park for speaker playback. This allows KEY readout when recognition remains faulted/stuck; pipeline reset still waits for both workers and is deferred until the speaker releases ownership.

This resets the existing runtime; it does not reconstruct BSS, revive disabled SDK feed/fetch flags or repair a permanently corrupted/blocked instance. The original spontaneous cause remains unresolved. New last-fault telemetry is preserved through recovery to make a future failure easier to diagnose. The board-local [status and USB controls](API.md#local-board-audio-diagnostics) document the recovery counters/flags and exact diagnostic commands.

## What changes and what is retained

The weather API contract, forecast/model output, observed-status vocabulary, confidence **0.80**, microphone/speaker configuration and prerecorded speech are unchanged. The private David-audio bank stays at **116 clips**; public eSpeak stays at **117**, including the demo announcement. Saved configuration remains in NVS, including the current **−4 °C** indoor offset. Earlier v27's −4.8 °C comparisons remain historical evidence rather than a new calibration claim.

`AUDIORESET` requests bounded runtime recovery over USB. `AUDIOSTALL` stops detector consumption for six seconds while feed continues, testing actual ring saturation rather than merely changing a fault flag. This provides a reproducible recovery condition; it is not a reproduction of the unknown spontaneous root cause. No HTTP recovery/stall control or new weather API field is added.

## Public source checks

| Check | Result |
| --- | --- |
| Final public compile/source agreement | Passed, zero warnings; all 38 compiled C++/header files matched source |
| Actual production lifecycle-method and ReadoutPlayer host tests | 120 checks passed |
| Complete host run | All 16 runner steps / eleven native executables passed, zero host warnings |
| Explicitly counted native checks | 159 arrival + 492 compensation + 557 sensor + 120 lifecycle = 1,328 |
| Demo/provisioning helpers | 23 Python tests passed |
| Compiled-image/model/partition validation | Five final-image tests and guarded dry run passed, without opening a serial port |
| Speech/source consistency | 117-clip eSpeak audio/header/phrases/manifest byte-identical to v28 |

The lifecycle harness compiles the actual production methods and ReadoutPlayer, covering missing park acknowledgement without reset/driver stop, failed reset/restart, retry limits/backoff, clock rollover, faults during playback and stale-event exclusion. The native-check total covers the four suites reporting aggregate counts; the other executables also passed their assertions.

The public application image is **7,412,256 bytes**, SHA-256 `d3d04fe7d22881bcea4c756b035d934a6ef34f1b669af611e6eccb0f82dc30a1`. Its unchanged speech assets were reused, with no audio regeneration. These image facts identify the checked build rather than a binary guaranteed identical on every host. The public eSpeak variant remains **unflashed and acoustically untested**.

## Installed-device checks

| Check | Result |
| --- | --- |
| Final private build/source agreement | Passed, zero warnings; all 38 compiled C++/header files matched source |
| Production lifecycle host checks | 120 passed |
| Image/partition checks and protected flash | Five checks and all four flashed image hashes passed; NVS/PHY protected |
| Actual forced stalls and recovery audit | 88 checks passed over 344.781 seconds |
| Genuine saturation faults | Two feed-result-zero faults: one idle and one during overview playback |
| Paced manual reset requests | Ten; all completed |
| Repair results | Twelve successes in twelve attempts, zero failed repairs |
| Command suppression | No accepted/stale commands across 284 checked suppression intervals |
| Dashboard/layout | 64 checks passed; all three actual v29 framebuffers inspected |
| Settings/final state | Saved −4 °C retained, voice ready, error null, recovery/suppression inactive, overview restored |

The private application image is **7,870,720 bytes**, SHA-256 `339eb5276fbe23642c8f4071cdccc0b68262b37639084fbaec605dff51790075`. It retains the 116-clip David-audio bank; no new recordings were generated.

The audit used **590 snapshots** while uptime advanced from **116 to 460 seconds**, without uptime or counter resets. The USB connection stayed open with **14,383 bytes unread and zero serial reads**. The workers completed repeated repairs and playback handoffs while recognition commands were suppressed during recovery/quiet intervals. Source and production lifecycle tests establish retained model/worker allocations; no per-cycle heap measurements or measured heap-stability claim were made.

All three full speaker streams completed:

| Page | PCM samples | Planned audio | Observed stream |
| --- | ---: | ---: | ---: |
| Overview | 428,051 | 26.75 s | 27.438 s |
| MTB forecast | 440,240 | 27.51 s | 28.141 s |
| History/indoor | 464,737 | 29.05 s | 29.578 s |

During the overview fault, repair waited until speaker release; playback still completed. At the final restored overview, voice was ready with no error/pending recovery/suppression, both microphones were live with RMS **43/119**, the feed result was **6,144**, and indoor sensor errors were zero. Audio/model/configuration/dictionary hashes were unchanged.

No new human listening confirmation or command-recognition accuracy result is claimed. A short forced-stall/stress run can establish these recovery transitions; it cannot prove an hours-long cure or absence of the spontaneous fault. Private payloads, SDK/device reports, identifiers, generated audio, binaries and new photos remain unpublished.
