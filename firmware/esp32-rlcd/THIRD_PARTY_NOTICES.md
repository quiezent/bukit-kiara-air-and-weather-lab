# ESP32 dashboard licensing and provenance

Copyright 2026 quiezent and contributors.

Original dashboard firmware, project tools and original phrase definitions in this directory are offered under the [Apache License 2.0](LICENSE). This license does not replace the terms of third-party code, libraries, fonts, build tools or data. Retain their copyright notices and license texts when redistributing them. The separate forecasting server and other repository content keep their own licenses.

This publication (**40-public.1**, 10 November 2026) contains source, build instructions and dependency preparation tools. It does not redistribute the Espressif SDK or toolchain, the Arduino CLI or eSpeak NG executables, compiled firmware images, fitted models, collected data, or the installed board's Microsoft David Desktop speech recordings.

## Included adaptations

### Waveshare ST7305 display driver

`WeatherDashboard/ST7305_U8g2.cpp` and `WeatherDashboard/ST7305_U8g2.h` derive from Waveshare's `02_Example/Arduino/10_U8G2_Test` at commit `eb1f63427d735a22b9c30e22fa63ebddae1834d3`. The repository's license grants Apache-2.0 terms and identifies Copyright 2026 Waveshare. Local changes include physical display polarity and dashboard integration; the source files identify the modifications. The upstream driver must not be described as wholly original dashboard code.

- [Pinned driver source](https://github.com/waveshareteam/ESP32-S3-RLCD-4.2/tree/eb1f63427d735a22b9c30e22fa63ebddae1834d3/02_Example/Arduino/10_U8G2_Test).
- [Pinned Waveshare license](https://github.com/waveshareteam/ESP32-S3-RLCD-4.2/blob/eb1f63427d735a22b9c30e22fa63ebddae1834d3/LICENSE); complete copy: [LICENSES/Waveshare-Apache-2.0.txt](LICENSES/Waveshare-Apache-2.0.txt).

### Espressif ES8311 codec adaptation

`WeatherDashboard/BoardSpeaker.cpp` adapts ES8311 register initialization from Espressif's `esp_codec_dev`, as preserved in Waveshare's same pinned commit under `02_Example/Arduino/07_Audio_Test/src/ExternLib/esp_codec_dev`. The copied source identifies Copyright 2023 Espressif Systems (Shanghai) CO LTD and Apache-2.0. The file header retains that attribution and describes the board-specific changes: a fixed 16 kHz DAC-only setup, checked I2C writes, exclusive I2S clock ownership, bounded mono-to-stereo writes and a quiet start.

- [Pinned ES8311 source](https://github.com/waveshareteam/ESP32-S3-RLCD-4.2/blob/eb1f63427d735a22b9c30e22fa63ebddae1834d3/02_Example/Arduino/07_Audio_Test/src/ExternLib/esp_codec_dev/device/es8311/es8311.c).
- Complete upstream license: [LICENSES/Espressif-codec-Apache-2.0.txt](LICENSES/Espressif-codec-Apache-2.0.txt).

These attributions refer to the preserved Apache-licensed versions, not a moving upstream branch whose licensing or implementation may differ. The ES7210 microphone adaptation (`BoardMicrophone.cpp`) shipped up to 32-public.1. It was removed with voice recognition, and no ES7210 code remains in this package.

### Trainers that follow scikit-learn and SciPy

Two headers train the models on the board. Both were written to reproduce the results of specific scikit-learn and SciPy code, step for step and in the same order of arithmetic, so that the board's fitted models match the server's:

- `WeatherDashboard/HgbTrainer.h` follows scikit-learn 1.5's `HistGradientBoostingRegressor` with absolute-error loss: the binning thresholds of `_find_binning_thresholds`, the split-gain arithmetic of `_split_gain`, the node order of the grower's heap and its depth and missing-value rules.
- `WeatherDashboard/ArrivalLogistic.h` follows the scikit-learn pipeline `SimpleImputer(strategy="median", add_indicator=True)`, `StandardScaler` and multinomial `LogisticRegression(solver="lbfgs")`. Its solver takes the iterates of SciPy's L-BFGS-B without bounds: 10 corrections, the same stopping tests and the Moré–Thuente line search (MINPACK-2 `dcsrch`/`dcstep`) built into it. The line-search routine is a close translation of that Fortran code, with the same variable names and branches.
- `WeatherDashboard/SessionFeatures.h` contains a small mean imputer with missing indicators that follows scikit-learn's `SimpleImputer(strategy="mean", add_indicator=True)`.

These are new C++ implementations, not copies of Python, Cython or Fortran files. They were written by reading that source to match its behaviour, though, and the line search follows the Fortran closely. Their independence from the upstream code cannot be established, so this package treats them as derived and keeps both projects' notices:

- **scikit-learn**, BSD-3-Clause, Copyright (c) 2007-2024 The scikit-learn developers. Complete copy: [LICENSES/scikit-learn-BSD-3-Clause.txt](LICENSES/scikit-learn-BSD-3-Clause.txt), from `COPYING` at [tag 1.5.1](https://github.com/scikit-learn/scikit-learn/blob/1.5.1/COPYING).
- **SciPy**, BSD-3-Clause, Copyright (c) 2001-2002 Enthought, Inc. 2003-2024, SciPy Developers. Complete copy: [LICENSES/SciPy-BSD-3-Clause.txt](LICENSES/SciPy-BSD-3-Clause.txt), from `LICENSE.txt` at [tag v1.14.1](https://github.com/scipy/scipy/blob/v1.14.1/LICENSE.txt).
- SciPy's bundled L-BFGS-B 3.0 is by Ciyou Zhu, Richard Byrd, Jorge Nocedal and Jose Luis Morales and is released under the BSD license. Its line search comes from MINPACK-1 (Argonne National Laboratory; Jorge J. Moré and David J. Thuente) and MINPACK-2 (Argonne National Laboratory and the University of Minnesota; Brett M. Averick, Richard G. Carter and Jorge J. Moré). See SciPy's [L-BFGS-B README](https://github.com/scipy/scipy/blob/v1.14.1/scipy/optimize/lbfgsb_src/README) and [LICENSES_bundled.txt](https://github.com/scipy/scipy/blob/v1.14.1/LICENSES_bundled.txt). Its authors ask that publications describing work that uses it, and commercial products that use it, cite at least one of: R. H. Byrd, P. Lu, J. Nocedal and C. Zhu, "A limited memory algorithm for bound constrained optimization", SIAM Journal on Scientific Computing 16(5), 1190–1208 (1995); C. Zhu, R. H. Byrd, P. Lu and J. Nocedal, "Algorithm 778: L-BFGS-B: Fortran subroutines for large-scale bound-constrained optimization", ACM Transactions on Mathematical Software 23(4), 550–560 (1997); J. L. Morales and J. Nocedal, "Remark on 'Algorithm 778: L-BFGS-B: Fortran subroutines for large-scale bound constrained optimization'", ACM Transactions on Mathematical Software 38(1), article 7 (2011).

The host parity tests import numpy, SciPy and scikit-learn as ordinary installed packages. None of their code is redistributed here.

## Embedded data

### ISRG root certificates

`WeatherDashboard/TlsRoots.h` embeds four Let's Encrypt root certificates published by the Internet Security Research Group: ISRG Root X1, ISRG Root X2, ISRG Root YE and ISRG Root YR. They are public trust anchors, included unmodified in PEM form so the board can verify the HTTPS servers it collects from. They are not original work of this project and are not covered by its Apache-2.0 grant. ISRG publishes them at [letsencrypt.org/certificates](https://letsencrypt.org/certificates/).

## External libraries and fonts

### U8g2 2.36.18

U8g2 library code is BSD-2-Clause, Copyright 2016 Oliver Kraus. Its font data has separate terms. The dashboard's Helvetica Bold fonts come from the Adobe/DEC X11 fonts, whose permission and copyright notices are included in the full U8g2 license. The `5x7` font identifies itself as public domain in the upstream font source. Do not apply Apache-2.0 or BSD-2-Clause indiscriminately to every font or example in the library.

The earlier local device build used Waveshare's `01_Arduino_Libraries/U8g2` at the pinned Waveshare commit. The public setup instead installs the official Arduino library-index package with `arduino-cli lib install U8g2@2.36.18`. All 140 source files were checked against that local vendor copy and match after CRLF/LF normalization; no vendor code patch is required. The official `U8g2-2.36.18.zip` package has SHA-256 `8b87fe60bc51508cf5b3c926051a14a46ddaf5644ac4d210f128d47de6c6c732`. The corresponding official U8g2 tag resolves to commit `d66b49af3e48cd0becf95f862353bf94a9c0c2be`. This publication does not bundle a copy of the external library source.

- [Pinned U8g2 license, including Adobe/DEC font terms](https://github.com/olikraus/u8g2/blob/d66b49af3e48cd0becf95f862353bf94a9c0c2be/LICENSE); complete copy: [LICENSES/U8g2-LICENSE.txt](LICENSES/U8g2-LICENSE.txt).
- [Pinned Waveshare library](https://github.com/waveshareteam/ESP32-S3-RLCD-4.2/tree/eb1f63427d735a22b9c30e22fa63ebddae1834d3/01_Arduino_Libraries/U8g2).

### ArduinoJson 7.4.3

ArduinoJson is MIT-licensed, Copyright 2014–2026 Benoit Blanchon. Tag `v7.4.3` resolves to commit `77771d3c07668e01d8f52acb03910c1110bb373f`. The public setup installs it separately with `arduino-cli lib install ArduinoJson@7.4.3`.

- [Pinned upstream license](https://github.com/bblanchon/ArduinoJson/blob/77771d3c07668e01d8f52acb03910c1110bb373f/LICENSE.txt); complete copy: [LICENSES/ArduinoJson-MIT.txt](LICENSES/ArduinoJson-MIT.txt).

### Arduino-ESP32 3.3.12 and its bundled SDK

The documented build uses Espressif's official Arduino-ESP32 core 3.3.12, with ESP-IDF 5.5.5. It is an Arduino CLI build, not a PlatformIO/pioarduino build. Core tag `3.3.12` resolves to commit `94afccf35fb1e401facddbcf9e13bcf7c76a31d8`. Besides the core itself, the firmware uses the core's bundled `WiFi`, `Network`, `NetworkClientSecure`, `HTTPClient`, `WebServer`, `ESPmDNS`, `FS`, `LittleFS`, `Preferences`, `Hash`, `SPI` and `Wire` libraries. Their source files carry their own notices: most are LGPL-2.1; `Preferences`, `LittleFS`, `Hash` and FS's `vfs_api` files are Apache-2.0; `ESPmDNS` and `NetworkUdp.h` are MIT. HTTPS runs on the SDK's TLS stack.

The core's LGPL-2.1 license text and LGPL-2.1-or-later notices, including the Arduino Team notice in `Arduino.h`, remain applicable to that code. The bundled SDK contains components with their own terms; neither this project's Apache license nor the core's LGPL label covers every SDK component. This source publication does not bundle that SDK or a linked firmware executable. Any future binary distribution must also satisfy the applicable dependency licenses, including LGPL source/relinking provisions.

- [Pinned core license](https://github.com/espressif/arduino-esp32/blob/94afccf35fb1e401facddbcf9e13bcf7c76a31d8/LICENSE.md); complete copy: [LICENSES/Arduino-ESP32-LGPL-2.1.md](LICENSES/Arduino-ESP32-LGPL-2.1.md).
- [Pinned Arduino.h notice](https://github.com/espressif/arduino-esp32/blob/94afccf35fb1e401facddbcf9e13bcf7c76a31d8/cores/esp32/Arduino.h).
- [Official 3.3.12 release](https://github.com/espressif/arduino-esp32/releases/tag/3.3.12).

## External build and speech-generation tools

### eSpeak NG

eSpeak NG is an external GPLv3 speech-generation tool. The audited 1.52.0 license source is commit `4870adfa25b1a32b4361592f1be8a40337c58d6c`; its complete [COPYING text](https://github.com/espeak-ng/espeak-ng/blob/4870adfa25b1a32b4361592f1be8a40337c58d6c/COPYING) is retained in [LICENSES/eSpeak-NG-GPL-3.0.txt](LICENSES/eSpeak-NG-GPL-3.0.txt). The executable, source and voice data keep their GPL terms. The public build uses a separately installed tool, not a redistributed executable.

The GPL's output clause applies only when the output's content itself constitutes a covered work. Ordinary recordings synthesized from this project's original weather phrases are a separate output; the firmware does not embed eSpeak NG code or voice data. Public speech assets are generated from those original phrases for this package under the project's Apache-2.0 terms. They are separate from the installed board's Microsoft David Desktop recordings, which are not included. Regeneration records the actual synthesizer version and settings; a version selected by the build instructions remains a tool dependency, not a change to its license.

### Arduino CLI

Arduino CLI is an external GPLv3 build tool. The audited v1.5.1 license source is commit `01f3d4f2ba7c2eaafb5dc710c8a1903af7762fea`; its full [upstream license](https://github.com/arduino/arduino-cli/blob/01f3d4f2ba7c2eaafb5dc710c8a1903af7762fea/LICENSE.txt) is copied to [LICENSES/Arduino-CLI-GPL-3.0.txt](LICENSES/Arduino-CLI-GPL-3.0.txt). The tool is installed or downloaded separately and is not included here. Using it to compile original source does not relicense the source under the tool's GPL.

Other tools and SDK components fetched during setup keep their own upstream licenses. Their omission from the source archive is not a waiver of those terms.

## Earlier releases

- **ESP-SR, 8 October to 14 October 2026 (25-public.1 to 32-public.1).** The client firmware recognised voice commands with Espressif's ESP-SR models (MultiNet 7 English and VADNet) under the ESPRESSIF MIT License, whose grant is limited to use on Espressif products. Its preparation tool downloaded the official model bundle separately. 40-public.1 includes no ESP-SR header, library or model, has no model partition and no `prepare_model.py`. The license copy `LICENSES/ESP-SR-ESPRESSIF-MIT.txt` was removed from this package and remains in the repository history with those releases.
- **ES7210 microphone codec, up to 32-public.1.** See the codec section above.

## License-copy integrity

The files below are complete upstream license texts, copied without text modifications. `LICENSE` is identical to `LICENSES/Apache-2.0.txt`. Component-specific copyright and modification notices remain in source or in the attribution sections above.

| File in `LICENSES/` | SHA-256 |
| --- | --- |
| `Apache-2.0.txt` | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |
| `Waveshare-Apache-2.0.txt` | `bb01037d243b2614f6c0bcf65671eff4a004b65ef687a4fd4925a9ab068bd613` |
| `Espressif-codec-Apache-2.0.txt` | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |
| `U8g2-LICENSE.txt` | `35070a56a2cb59560278c1734c51da94b679bc208c5244625769a0f1df001d30` |
| `ArduinoJson-MIT.txt` | `4a7ee9c96b28cbf30c5bf7c2d211a0ef57179f0328e68ad7b7fa7d754b7da1a2` |
| `Arduino-ESP32-LGPL-2.1.md` | `62e54861b30e953735dc187f036917c588665dbe3c5dd88449e34317608bbf5e` |
| `eSpeak-NG-GPL-3.0.txt` | `8ceb4b9ee5adedde47b31e975c1d90c73ad27b6b165a1dcd80c7c545eb65b903` |
| `Arduino-CLI-GPL-3.0.txt` | `3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986` |
| `scikit-learn-BSD-3-Clause.txt` | `75684d16f4662ab842ed5af9482ac01939aee83c8243d89993616d959d3053f8` |
| `SciPy-BSD-3-Clause.txt` | `c29fb8e5936f33237a8b15256b4c6e93037f05bbf58c3d7b5755aede4c200173` |

The SciPy copy is the first section of the SciPy 1.14.1 wheel's `LICENSE.txt`, which matches the text of `LICENSE.txt` at tag v1.14.1. The scikit-learn copy was transcribed from `COPYING` at tag 1.5.1. Both copies were compared with the files at those tags and match byte for byte.
