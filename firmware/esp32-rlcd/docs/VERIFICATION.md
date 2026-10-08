# Public source verification — 8 October 2026

I verified `weather-dashboard-25-public.1` on Windows with Python 3.11, Arduino CLI 1.5.1 and the dependency versions in the build guide. `setup.py` installed into fresh package-specific data and library directories, reusing official Arduino archive downloads where available. The public build did not use the private sketch's installed core or library directories.

| Check | Result |
| --- | --- |
| Public firmware compile | Passed, zero warnings |
| Production host regressions | Seven C++ executables passed with `-Wall -Wextra -Werror` |
| Demo API and provisioning helpers | 21 Python tests passed |
| Compiled-image/partition/model validation | All five tests passed; dry run passed |
| Independent clean speech generations | 121 clips, manifests, C++ and WAV files matched byte for byte |
| Publication review | No credentials, private network identifiers, machine paths, generated audio or binaries in the source package |

Host regressions cover navigation, command sessions, bounded diagnostics, selected-page readouts, unavailable/stale data, probability winners, ride extrema, rain windows, date-aware comparisons and playlist overflow. Synthetic disclosure is tested on all three pages: it prepends its announcement while retaining the same numeric content and following clips. Provisioning tests use mocked serial ports, including quiet-port retries and hidden-password handling. The demo tests include real loopback HTTP responses and mocked mDNS registration; they do not contact a weather provider.

The final local public application image was 7,593,472 bytes; its SHA-256 was `d3047cfed0360e725949641731bad742518352ac1acd17be2413043ecd84862b`. This is a record of that build, not a promise of identical binaries on every host. Generated manifests record engine/data, phrase and tool hashes so differences can be investigated. Only the speech repeatability check used two independent clean generations with identical inputs.

These checks did not flash the public variant, measure its acoustic recognition or speaker audibility, or evaluate forecast skill. The installed Microsoft David v25 hardware results in the journal describe a different audio build. Raw local reports remain ignored; rerun the commands in the [build guide](../README.md) to generate your own results.
