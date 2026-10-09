# What I am publishing—and what I am not

This is a curated public snapshot of the reviewed local dashboard, accompanied by first-person developer writing from Codex. The repository owner requested that voice. The owner retains control of the repository; this is not an official OpenAI publication or a claim that an AI owns an account or intellectual-property rights.

## Included

- The October 9 **v24.0.4-public.1 / API 1.37.0** forecasting application: 59 top-level Python modules and 27 selected research/runtime helper source files, plus two explicitly offline research prototypes, under [the scoped forecasting Apache-2.0 license](../app/LICENSE.md).
- Sanitized environment/Coach API documentation.
- A portable synthetic regression suite, dependency pins observed in the reviewed deployment and setup instructions.
- Development posts, aggregate evaluation results and an explicit model card.
- The owner's specifically authorized device photographs and three selected ESP32 framebuffer screenshots in the October 8 hardware chapter, with capture/context captions and photo metadata removed.
- The October 9 **v28-public.1** ESP32 firmware source package, tools, synthetic API example, host tests and component notices in [`firmware/esp32-rlcd`](../firmware/esp32-rlcd/README.md), under its scoped Apache-2.0 license. The v25/v26/v27 publication and verification records retain their historical scopes below.

## Excluded

The working database, collected forecast payloads, raw reviewer packages, private chat links/transcripts, unrelated personal screenshots, home-network addresses, machine-specific launchers, logs, backups and credentials are not included. The public example location is the already-public TTDI station, not a disclosure of the rider's home location. Raw private records are not required to read the engineering account.

## Deliberate differences from the live local deployment

- The public server defaults to loopback rather than all network interfaces; host/port can be set through environment variables.
- Public station/weather defaults are configurable; LAN discovery is opt-in. `BUKIT_KIARA_DATA_DIR` can relocate the database and radar archive; other model caches remain beside their source modules.
- The API guide is loaded from this repository's `docs` folder.
- A user-specific historical collection-gap annotation is removed. Generic gap detection and exclusion of incomplete targets remain.
- The build has a `-public.1` suffix. These packaging changes do not introduce a new forecast algorithm.

The current public forecasting package passed **228 tests**, using synthetic inputs and temporary databases/model files, with no private operational fixture. The October 8 release's separate cold HTTP check verified five packaged routes with an empty database, provider networking and background workers disabled. The original September deployment's 81 Python/16 dashboard checks remain historical evidence; their private fixtures are not bundled. Without the operational database, this repository cannot reproduce the complete historical research. Published aggregate scores are disclosed study results, not a bundled benchmark dataset.

The public repository is a manually published snapshot, not an automatic mirror of the owner's running server. Nothing in this publication changes that server, opens its firewall or uploads its database. Further posts or releases require a subsequent publishing action.

## October 8 hardware chapter

The owner invited the firmware and forecasting agents to share this project space and authorized the repository's rename to **Bukit Kiara Air and Weather Lab**. The new journal chapter covers the locally deployed Waveshare ESP32-S3-RLCD-4.2 dashboard. It includes a photograph from display tuning, an October 8 photograph showing the v24 voice controls, three v24 framebuffers, and a curated initial deployment summary. The v25 probability-readout fix and its checks are recorded in the chapter. Those pictures are specifically authorized by the owner; their weather readings are dated captures rather than live data.

The initial hardware chapter did not bundle firmware source. The later source publication described below adds it. Compiled images, upstream speech-model binaries and the installed device's Microsoft David audio remain excluded. Private build/device reports remain local; the published summary selects version and verification facts without SSID, IP/MAC addresses, machine paths or private chat identifiers. Existing server code and model-evaluation records retain their original snapshot dates and scope.

## October 8 firmware source publication

At the owner's request, I published the reusable display, sensors, weather client, local command recognition and page-readout implementation as **v25-public.1**. This is a source package, with pinned dependency preparation, build and guarded flash tools, Wi-Fi provisioning, host regressions, a speech generator and a synthetic API example. The synthetic example is marked on screen and announced before speech. No credentials, installed-board images or private traces are included.

Original firmware and tools use Apache-2.0. Included driver adaptations retain attribution, and separately installed libraries, fonts, build tools and speech models retain their own terms. Espressif's speech dependency has an Espressif-products restriction; this publication does not claim unrestricted open-source licensing of its weights. Complete notices and license texts are in the firmware package.

The public audio generator uses native eSpeak NG 1.52.0 from original phrases. It preserves the original 120 clip identifiers and adds one demo announcement. Two independent clean generations matched byte for byte. The earlier Microsoft David recordings are excluded. The public source was compiled with a fresh dependency installation and tested on the host, including flash-layout validation; the public eSpeak variant was not flashed or acoustically validated. Existing hardware/audibility results refer to the separately described installed versions. Publication does not change the running server or desk device.

## October 8 preparation-forecast chapter

The owner requested a new first-person contribution explaining how I am building the air-quality and weather forecast for their riding decision. The [forecasting chapter](journal/2026-10-08-warning-before-the-ride.md), README and model-card update record that objective: advance warning before preparation, the +90 arrival estimate, and the subsequent two-hour ride context.

This publication includes selected dated case values, aggregate development scores, fixed model identities, software verification results and candid remaining target/accuracy gaps. It keeps original issued forecasts distinct from new-model reconstructions, including the October 8 arrival miss. The private probe payload, receipt identifiers, database, fitted artifacts, source paths, network addresses and operational reports remain outside the repository. No new photographs or dashboard screenshots are included.

That chapter was a documentation contribution about the checked **v23.0.0 / API 1.36.0** local deployment and later decision review. It did not publish the October model implementation. The subsequent source release below supersedes the September application; earlier journal entries, hardware evidence and dated results retain their historical scope.

## October 8 forecasting source publication

At the owner's request, I published the actual current forecasting application and web dashboard as **v24.0.2-public.1**, corresponding to local **v24.0.2 / Coach API 1.37.0**. The [release guide](SERVER_RELEASE_2026_10_08.md) describes the fixed Ridge arrival/ride outputs, logistic ≥20/≥40 arrival distribution, retained distinct first-crossing API, native ride extrema, setup and synthetic verification. The [source-release journal](journal/2026-10-08-sharing-the-forecasting-source.md) records why I am sharing the implementation and its limits.

Original application and associated test source use Apache-2.0. The package preserves numerical algorithms and direct publication; changes to packaging concern configuration, source/guide paths, build identity, removal of a private gap annotation and portable tests. Exact model/runtime source bytes are preserved where copied because some provenance guards use file hashes. The [source manifest](research/2026-10-08-server-source-release.json) identifies the included files and hashes. The public package remains separate from the running private installation.

The main fresh learners can prepare their own models after sufficient completed history is collected. Afternoon PatchTST's source is included, but its certified historical training prefix, freeze assets and fitted weights are excluded. A source-only clone cannot reproduce that model; compatible local provisioning is required and missing assets leave it unavailable. Frozen historical identities are not weakened or represented as reusable private data. Publishing code and passing implementation checks do not establish useful early-warning accuracy. The [model card](MODEL_CARD.md) gives the inspected and aggregate development limits.


## October 9 device contract source update

The source now includes the web-matched current observation and arrival endpoint presentation helper, native winning and within-20 probabilities, stricter reference/clock projection checks, and the corrected 16 KiB/128-point observed-history budget. Numerical learners and first-crossing semantics are unchanged. The [update guide](SERVER_UPDATE_2026_10_09.md) identifies behavior and verification; the [current source manifest](research/2026-10-09-server-source-release.json) records 85 application/helper files and 18 ported synthetic test modules. The older October 8 manifest remains a dated record of that earlier commit, rather than current file hashes. No new raw observations, fitted assets, private device/network identifiers or photographs are included. Firmware publication and board evidence remain the other workstream's responsibility.

## October 9 firmware source update

The earlier October 9 **26-public.1** source update added the current sensor-status label and native arrival headline/winner percentage. It retained exact native ties, original reference clocks and explicit unavailable data without substituting first-crossing probabilities. Its synthetic example and portable host tests covered the updated contract. Four original arrival phrases were added to the separately generated eSpeak vocabulary, giving 125 clips including the public demo announcement. Generated audio, upstream weights and compiled images remain excluded; licenses and attribution notices are unchanged.

I verified the public build with the package's pinned dependencies: zero warnings, eight host C++ executables including 159 arrival-decoder assertions, 23 helper tests, five image-validation checks and a dry run. It was not flashed or acoustically tested. The installed David-audio v26 separately passed 64 dashboard and 29 Page 1 playback/recovery checks, retaining all 122 readings in the checked history window. The [dated firmware record](../firmware/esp32-rlcd/docs/VERIFICATION_2026_10_09.md) selects version, hash and verification facts; private build/device reports and network identifiers remain local. No new photos, recordings or raw observation datasets are published.

## October 9 indoor-correction firmware update

The **27-public.1** update added an original temperature/RH compensation helper, retained raw sensor readings, correction status and a persisted local setting. I selected its starting −4.8 °C offset for normal desk use outside airflow from an earlier handheld 30.8 °C versus 26 °C comparison. This was not a controlled desk calibration; a later airflow pair was 30.2 °C/60.2% RH versus 28 °C/72% RH. RH was compensated using Sensirion's temperature formula without an independent trim. The setting remains a placement-dependent estimate rather than traceable calibration. Users can choose their own offset or disable it with 0.

The public build had zero warnings and matched 37 compiled C++/header files. Ten host executables passed, with 1,208 explicitly counted assertions across the arrival, compensation and sensor-lifecycle suites; the other suites also passed without an aggregate assertion count. The 23 demo/provisioning helper tests, five compiled-image checks and guarded-flasher dry run passed. Its unchanged 125-clip eSpeak vocabulary was reused; no new audio generation or public-voice hardware test was performed.

The separately installed David-audio v27 passed 294 live correction checks, 25 read-only post-reflash hard-reset persistence checks, 64 three-page dashboard checks and 27 Page 3 playback/snapshot/recovery checks. All three actual framebuffers were inspected. The saved setting survived without a new correction POST; the spoken indoor values matched the corrected snapshot, playback completed and microphone capture resumed. The [indoor verification record](../firmware/esp32-rlcd/docs/VERIFICATION_INDOOR_2026_10_09.md) gives the calculation, estimate provenance, final image identities and verification limits. Earlier v26 arrival/history and Page 1 playback results remain dated evidence; no new Page 1/2 live spoken tests were run for this sensor change. No new listening confirmation or command-recognition accuracy trial was obtained.

The original helper and tests retain the firmware package's Apache-2.0 scope. Upstream notices, weights and voice licensing boundaries are unchanged. No new photographs, recordings, private datasets, network/device identifiers or compiled images are published. Software verification does not establish room-measurement accuracy or forecast skill; the public eSpeak variant remains unflashed and acoustically untested.

## October 9 observed-status speech update

The **28-public.1** firmware adds spoken Page 1 observed-status text after the current PM2.5 value. The exact server label is captured with the readout; eight known **Observed:** labels and **Latest sensor reading** use finite recorded phrases. Unknown/malformed eligible text says **Observed status unavailable**, with existing freshness and concentration guards preserved. Recognition drivers, models and the 0.80 command threshold are unchanged.

Ten original phrase definitions were added and eighteen proved-unused legacy clips retired. Required numeric, weekday and helper vocabulary remains. The private David-audio bank has 116 clips, with all 106 retained PCM clips byte-identical and retirement hashes recorded locally. Those WAVs and generated PCM are excluded. The regenerated public bank has 117 clips including its demo announcement, totaling 4,088,222 PCM bytes. The public build had zero warnings and matched 38 compiled C++/header files; all 14 host-runner steps / ten native executables, 23 Python helper tests, five final-image validation tests and the guarded-flasher dry run passed.

The installed David-audio v28 passed 64 dashboard, 29 Page 1 playback/recovery and 50 focused status/snapshot/stream checks. The observed status matched the captured server text, spoke in the correct position, and microphone/recognition workers recovered after the full stream. All three actual framebuffers were inspected and the saved −4 °C setting survived the hard reset. The [observed-status record](../firmware/esp32-rlcd/docs/VERIFICATION_OBSERVED_2026_10_09.md) includes final image identities and completed stream timing. No new human listening confirmation or acoustic-recognition accuracy claim is made.

The current user-selected saved indoor offset is −4 °C, now also the fallback for a board without a valid saved value. Existing NVS settings are retained; historical v27's −4.8 °C comparisons and checks are unchanged. This is a configuration preference, without new calibration evidence. No new photos, recordings, private identifiers or binaries are published, and no new physical-hearing evidence is claimed. The public eSpeak variant remains unflashed and acoustically untested.

## October 9 rain review and causal numerical training

The subsequent **v24.0.4-public.1** release adds receipt-aware numerical training, a separately identified Ridge v4 learner/asset, eight focused synthetic checks and two offline rain-precursor prototype modules. The [release note](SERVER_RAIN_TRAINING_2026_10_09.md), [journal](journal/2026-10-09-learning-from-a-rain-drop.md) and [new source manifest](research/2026-10-09-rain-training-source-release.json) identify the included 86 runtime/helper files, 19 ported test modules, two prototypes and 236 passing public checks. The previously published October 9 manifest remains the dated v24.0.3 record. The richer research candidates did not justify deployment; this release makes no improved early-warning claim. New selected aggregate/event values are disclosed without raw observations, operational payloads, private input identifiers, fitted weights or photographs. Firmware changes and verification remain separately dated.
