# ESP32 dashboard licensing and provenance

Copyright 2026 quiezent and contributors.

Original dashboard firmware, project tools and original phrase definitions in this directory are offered under the [Apache License 2.0](LICENSE). This license does not replace the terms of third-party code, libraries, fonts, build tools or model data. Retain their copyright notices and license texts when redistributing them. The separate forecasting server and other repository content keep their own licenses.

This publication contains source, build instructions and dependency preparation tools. It does not redistribute the ESP-SR model binary, Espressif SDK/toolchain, Arduino CLI or eSpeak NG executables, installed-board firmware images, or Microsoft David Desktop speech recordings. Dependency binaries and compiled firmware remain outside the published source.

## Included adaptations

### Waveshare ST7305 display driver

`WeatherDashboard/ST7305_U8g2.cpp` and `WeatherDashboard/ST7305_U8g2.h` derive from Waveshare's `02_Example/Arduino/10_U8G2_Test` at commit `eb1f63427d735a22b9c30e22fa63ebddae1834d3`. The repository's license grants Apache-2.0 terms and identifies Copyright 2026 Waveshare. Local changes include physical display polarity and dashboard integration; the source files identify the modifications. The upstream driver must not be described as wholly original dashboard code.

- [Pinned driver source](https://github.com/waveshareteam/ESP32-S3-RLCD-4.2/tree/eb1f63427d735a22b9c30e22fa63ebddae1834d3/02_Example/Arduino/10_U8G2_Test).
- [Pinned Waveshare license](https://github.com/waveshareteam/ESP32-S3-RLCD-4.2/blob/eb1f63427d735a22b9c30e22fa63ebddae1834d3/LICENSE); complete copy: [LICENSES/Waveshare-Apache-2.0.txt](LICENSES/Waveshare-Apache-2.0.txt).

### Espressif ES7210 and ES8311 codec adaptations

`WeatherDashboard/BoardMicrophone.cpp` and `WeatherDashboard/BoardSpeaker.cpp` adapt register initialization from Espressif's `esp_codec_dev`, as preserved in Waveshare's same pinned commit under `02_Example/Arduino/07_Audio_Test/src/ExternLib/esp_codec_dev`. The copied source identifies Copyright 2023 Espressif Systems (Shanghai) CO LTD and Apache-2.0. Existing file headers retain attribution and describe the board-specific I2C/I2S, buffering, error handling and playback changes.

- [Pinned ES7210 source](https://github.com/waveshareteam/ESP32-S3-RLCD-4.2/blob/eb1f63427d735a22b9c30e22fa63ebddae1834d3/02_Example/Arduino/07_Audio_Test/src/ExternLib/esp_codec_dev/device/es7210/es7210.c).
- [Pinned ES8311 source](https://github.com/waveshareteam/ESP32-S3-RLCD-4.2/blob/eb1f63427d735a22b9c30e22fa63ebddae1834d3/02_Example/Arduino/07_Audio_Test/src/ExternLib/esp_codec_dev/device/es8311/es8311.c).
- Complete upstream license: [LICENSES/Espressif-codec-Apache-2.0.txt](LICENSES/Espressif-codec-Apache-2.0.txt).

These attributions refer to the preserved Apache-licensed versions, not a moving upstream branch whose licensing or implementation may differ.

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

The documented build uses Espressif's official Arduino-ESP32 core 3.3.12, with ESP-IDF 5.5.5 and bundled ESP-SR 2.5.3. It is an Arduino CLI build; the installed board build was not a PlatformIO/pioarduino build. Core tag `3.3.12` resolves to commit `94afccf35fb1e401facddbcf9e13bcf7c76a31d8`.

The core's LGPL-2.1 license text and LGPL-2.1-or-later notices, including the Arduino Team notice in `Arduino.h`, remain applicable to that code. The bundled SDK contains components with their own terms; neither this project's Apache license nor the core's LGPL label covers every SDK component. This source publication does not bundle that SDK or a linked firmware executable. Any future binary distribution must also satisfy the applicable dependency licenses, including LGPL source/relinking provisions.

- [Pinned core license](https://github.com/espressif/arduino-esp32/blob/94afccf35fb1e401facddbcf9e13bcf7c76a31d8/LICENSE.md); complete copy: [LICENSES/Arduino-ESP32-LGPL-2.1.md](LICENSES/Arduino-ESP32-LGPL-2.1.md).
- [Pinned Arduino.h notice](https://github.com/espressif/arduino-esp32/blob/94afccf35fb1e401facddbcf9e13bcf7c76a31d8/cores/esp32/Arduino.h).
- [Official 3.3.12 release](https://github.com/espressif/arduino-esp32/releases/tag/3.3.12).

## ESP-SR: a separate hardware-restricted dependency

Espressif's upstream license is titled **ESPRESSIF MIT License**. Its permission grant is limited to use on Espressif Systems products. This is not the ordinary unrestricted MIT license, and the pretrained model bundle is not relicensed under this project's Apache-2.0 license. The source-only publication does not claim unrestricted open-source rights over those weights.

The complete applicable text is preserved in [LICENSES/ESP-SR-ESPRESSIF-MIT.txt](LICENSES/ESP-SR-ESPRESSIF-MIT.txt), copied from [upstream commit `0a6e15d5ea9c7f5402093439f3af4c9250ae0d8d`](https://github.com/espressif/esp-sr/blob/0a6e15d5ea9c7f5402093439f3af4c9250ae0d8d/LICENSE). Its Git blob is `8e206f3bc38f5fa84a4956f2d5030ca17daea0d7`, matching the audited model provenance. This license-source commit is separate from the versioned SDK archive that supplies the actual weights.

`WeatherDashboard/tools/prepare_model.py` obtains the unchanged official model through a hash-verified SDK archive or verifies an installed SDK copy. Its generated `WeatherDashboard/flash/model.bin` is ignored by Git. The pinned inputs are:

| Item | Value |
| --- | --- |
| Archive | `esp32s3-libs-3.3.12.zip` |
| Official URL | [Espressif's versioned release asset](https://github.com/espressif/arduino-esp32/releases/download/3.3.12/esp32s3-libs-3.3.12.zip) |
| Archive SHA-256 | `08f4e7292fee97dc011dc64e3c670a9a4e83a6b52a604c4b9b18ab2885cdd092` |
| Archive member | `esp32s3-libs/esp_sr/srmodels.bin` |
| Model bytes | 3,340,296 |
| Model SHA-256 | `c7ea01954baaa1fa89ef6b88ba8819edcbf419b7aba1e093950f8badf1a6c328` |
| Models/resources | `mn7_en`, `vadnet1_medium`, `fst`, `wn9_hiesp` |

Only the English command and VAD models are instantiated by the dashboard; WakeNet is disabled. The model binary, prebuilt ESP-SR libraries and SDK archives must remain outside the published source tree. Their download and use retain Espressif's terms.

## External build and speech-generation tools

### eSpeak NG

eSpeak NG is an external GPLv3 speech-generation tool. The audited 1.52.0 license source is commit `4870adfa25b1a32b4361592f1be8a40337c58d6c`; its complete [COPYING text](https://github.com/espeak-ng/espeak-ng/blob/4870adfa25b1a32b4361592f1be8a40337c58d6c/COPYING) is retained in [LICENSES/eSpeak-NG-GPL-3.0.txt](LICENSES/eSpeak-NG-GPL-3.0.txt). The executable, source and voice data keep their GPL terms. The public build uses a separately installed tool, not a redistributed executable.

The GPL's output clause applies only when the output's content itself constitutes a covered work. Ordinary recordings synthesized from this project's original weather phrases are a separate output; the firmware does not embed eSpeak NG code or voice data. Public speech assets are generated from those original phrases for this package under the project's Apache-2.0 terms. They are separate from the earlier installed board's Microsoft David Desktop recordings, which are not included. Regeneration records the actual synthesizer version and settings; a version selected by the build instructions remains a tool dependency, not a change to its license.

### Arduino CLI

Arduino CLI is an external GPLv3 build tool. The audited v1.5.1 license source is commit `01f3d4f2ba7c2eaafb5dc710c8a1903af7762fea`; its full [upstream license](https://github.com/arduino/arduino-cli/blob/01f3d4f2ba7c2eaafb5dc710c8a1903af7762fea/LICENSE.txt) is copied to [LICENSES/Arduino-CLI-GPL-3.0.txt](LICENSES/Arduino-CLI-GPL-3.0.txt). The tool is installed or downloaded separately and is not included here. Using it to compile original source does not relicense the source under the tool's GPL.

Other tools and SDK components fetched during setup keep their own upstream licenses. Their omission from the source archive is not a waiver of those terms.

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
| `ESP-SR-ESPRESSIF-MIT.txt` | `7d916fb00bc0742c47cafb0d0144b67f826d76779730b1cb8796045ea6ba1b9a` |
| `eSpeak-NG-GPL-3.0.txt` | `8ceb4b9ee5adedde47b31e975c1d90c73ad27b6b165a1dcd80c7c545eb65b903` |
| `Arduino-CLI-GPL-3.0.txt` | `3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986` |
