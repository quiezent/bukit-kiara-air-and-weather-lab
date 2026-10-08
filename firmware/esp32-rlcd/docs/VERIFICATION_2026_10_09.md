# Arrival text and history verification — 9 October 2026

I coordinated this firmware update with the forecasting server maintainer. Page 1 now uses the web dashboard's current-reading label and the native +90-minute arrival headline with its winning probability. Its spoken summary uses the same native arrival probability. The first-crossing forecast remains a different target. The graph problem came from the server's former 8 KiB response cap: it transmitted only four points from 122 observations. The corrected server fits the board's existing 16 KiB body and 128-point limits.

## Installed desk device

The installed version is `weather-dashboard-26-arrival-display`, with the existing Microsoft David recordings and four appended arrival phrases. All 120 previous clips, voice settings, confidence 0.80, speech models, microphone/speaker drivers and graph renderer were preserved. NVS and PHY regions were excluded from the four-image flash.

| Check | Result |
| --- | --- |
| Build and compiled C++/header source agreement | Passed, zero warnings |
| Native arrival decoder | 159 host assertions passed |
| Narration and capacity regressions | Passed for all three pages and overflow |
| Flash-layout validation and written-image hashes | Five checks and all four hashes passed |
| Live dashboard/API/discovery/sensor checks | 64 passed |
| Live Page 1 playback/recovery checks | 29 passed |
| Software display captures | All three actual framebuffers inspected |

At the checked issue, Page 1 showed **Latest sensor reading** and **No change on arrival 95.0%**, from native probability `0.950476302`. Read Info included **No change on arrival, 95.0 percent.** Its complete stream finished in 26.844 seconds against a 25.91-second plan. Both microphones and recognition resumed, with no backpressure or voice error. These are dated values that change with the server's next issue.

The board retained all **122** transmitted observations, with PM2.5 average **110.197541**, minimum **94.2** and maximum **133.2 µg/m³** from the original six-hour samples. PM, temperature and heat-index graphs use the same timestamp rows. Collection gaps and full-source summary statistics remain independent of display sampling.

The installed application image was 8,262,304 bytes, SHA-256 `73965818a945248e7d256e0f99e2358462c0d2fb5068f925b59b6fe0cfd92322`. This image and its Microsoft voice audio are not distributed. No new listening confirmation, physical photograph or spoken-command accuracy trial was obtained for v26; the checks establish data agreement, layout, PCM completion and recovery.

## Public source build

I built `weather-dashboard-26-public.1` using the previously prepared package-specific pinned dependencies on Windows, without using the installed device's generated audio. The native eSpeak NG 1.52.0 `en-us` generator produced **125 clips**, totaling **4,430,932 PCM bytes**: the original 120 phrases, four arrival phrases and the public demo announcement. The example keeps its visible **DEMO WEATHER** heading and spoken **Demonstration data** prefix.

| Check | Result |
| --- | --- |
| Public firmware compile and compiled C++/header source agreement | Passed, zero warnings |
| Production host regressions | Eight C++ executables passed with `-Wall -Wextra -Werror` |
| Native arrival validation | 159 assertions, included in that host run |
| Demo API and provisioning helpers | 23 Python tests passed |
| Compiled-image/partition/model validation | All five tests passed; dry run passed |
| Speech generation | All 125 ordered IDs matched the header; generator validation passed |
| Publication review | No credentials, private network identifiers, machine paths, generated audio or binaries in the source package |

Tests cover native raw winners and exact ties, availability, malformed/stale/future clocks, original-reference consistency, nullable independent point references, nested probability tails, captured readouts, demo disclosure and bounded playlists. The synthetic API rebases issue/arrival/reference clocks coherently and preserves native probabilities separately from its illustrative first-crossing head. Helper tests use loopback or mocks.

The public application image was **7,743,744 bytes**, SHA-256 `d1d1b0425fde252e2e4a642c1135879de32a3a99dcd853348a58fa8b6f0e5e45`. These hashes identify the checked builds, not identical binaries guaranteed on every host. Only one new public v26 speech generation was performed; the [October 8 record](VERIFICATION.md) contains the earlier v25 repeatability check.

The public eSpeak variant was **not flashed or acoustically tested**. Its compilation/host checks do not verify the installed David voice, and installed-board playback checks do not verify eSpeak audibility. Neither establishes forecast accuracy. Generated audio, images, model files and raw reports remain ignored; the [build guide](../README.md) gives the commands to reproduce these checks.
