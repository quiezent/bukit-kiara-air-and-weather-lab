#!/usr/bin/env python3
"""Run production host regressions without opening a device or using the network.

Use the package setup first to install the pinned Arduino libraries and generate
speech/model assets. Example: python tools/test_host.py --cxx g++
Pass repeatable --cxx-arg=ARG for compiler drivers such as `zig c++`.
Reports and generated harnesses are written under build/host by default.
"""

import argparse
import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys


PACKAGE = Path(__file__).resolve().parents[1]
FLAGS = ["-std=c++17", "-Wall", "-Wextra", "-Werror"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", required=True, help="Host C++ compiler executable")
    parser.add_argument("--cxx-arg", action="append", default=[],
                        help="Extra compiler driver argument; use --cxx-arg=ARG for values beginning with '-' ")
    parser.add_argument("--arduino-data-dir", type=Path, default=PACKAGE / ".state/arduino-data")
    parser.add_argument("--arduino-user-dir", type=Path, default=PACKAGE / ".state/arduino-user")
    parser.add_argument("--arduino-json-src", type=Path,
                        help="Override the ArduinoJson 7.4.3 src directory")
    parser.add_argument("--esp32-core-dir", type=Path,
                        help="Override the Arduino-ESP32 3.3.12 cores/esp32 directory")
    parser.add_argument("--esp-sr-include", type=Path,
                        help="Override the ESP-SR target directory containing esp_mn_iface.h")
    parser.add_argument("--speech-manifest", type=Path, default=PACKAGE / "build/speech/manifest.json")
    parser.add_argument("--build-dir", type=Path, default=PACKAGE / "build/host",
                        help="Generated test sources, executables and reports")
    parser.add_argument("--firmware-build-dir", type=Path, default=PACKAGE / "build/firmware",
                        help="Matching compiled images for all five flash checks")
    parser.add_argument("--skip-flash", action="store_true",
                        help="Explicitly run C++ checks only before a firmware image set has been built")
    args = parser.parse_args()

    data = args.arduino_data_dir.resolve()
    user = args.arduino_user_dir.resolve()
    arduino_json = (args.arduino_json_src or user / "libraries/ArduinoJson/src").resolve()
    core = (args.esp32_core_dir or data / "packages/esp32/hardware/esp32/3.3.12/cores/esp32").resolve()
    sdk = (args.esp_sr_include or data / "packages/esp32/tools/esp32s3-libs/3.3.12/include/espressif__esp-sr/include/esp32s3").resolve()
    manifest = args.speech_manifest.resolve()
    build = args.build_dir.resolve()
    firmware = PACKAGE / "WeatherDashboard"
    tests = firmware / "tests"
    requirements = [arduino_json / "ArduinoJson.h", core / "stdlib_noniso.c",
                    sdk / "esp_mn_iface.h", manifest]
    if not args.skip_flash:
        requirements.append(firmware / "flash/model.bin")
        if not args.firmware_build_dir.is_dir():
            parser.error("Build the firmware first, specify --firmware-build-dir, or explicitly use --skip-flash")
    for required in requirements:
        if not required.is_file():
            parser.error(f"Required input not found: {required}; run package setup or supply a dependency override")

    build.mkdir(parents=True, exist_ok=True)
    report = {
        "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "compiler": args.cxx,
        "compiler_arguments": args.cxx_arg,
        "required_compile_flags": FLAGS,
        "dependencies": {"arduino_json_src": str(arduino_json), "esp32_core_dir": str(core),
                         "esp_sr_include": str(sdk), "speech_manifest": str(manifest)},
        "flash_checks": "explicitly skipped" if args.skip_flash else "required",
        "scope": "Host C++ and compiled-file checks only; no network or device access.",
        "steps": [],
        "passed": False,
    }
    compiler = [args.cxx, *args.cxx_arg]
    suffix = ".exe" if sys.platform == "win32" else ""
    compiler_arguments = [f"--cxx-arg={argument}" for argument in args.cxx_arg]

    with (build / "host-verification.log").open("w", encoding="utf-8") as log:
        def run(label, command):
            heading = f"\n{label}\n{shlex.join(map(str, command))}\n"
            print(heading, end="", flush=True)
            log.write(heading)
            log.flush()
            result = subprocess.run(command, cwd=PACKAGE, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", check=False)
            output = result.stdout + result.stderr
            print(output, end="" if output.endswith("\n") else "\n", flush=True)
            log.write(output + ("" if output.endswith("\n") else "\n"))
            log.flush()
            report["steps"].append({"name": label, "command": list(map(str, command)),
                                    "exit_code": result.returncode})
            if result.returncode:
                raise subprocess.CalledProcessError(result.returncode, command)

        try:
            run("Host compiler", [*compiler, "--version"])
            for name in ("voice_navigation_test", "voice_session_test", "weather_readout_test",
                         "voice_diagnostic_test"):
                executable = build / (name + suffix)
                command = [*compiler, *FLAGS, "-I", str(firmware)]
                if name == "voice_diagnostic_test":
                    command += ["-I", str(sdk)]
                command += [str(tests / (name + ".cpp")), "-o", str(executable)]
                run("Compile " + name, command)
                run("Run " + name, [str(executable)])

            run("Production voice diagnostic publication", [
                sys.executable, str(tests / "voice_diagnostic_publish_test.py"),
                "--cxx", args.cxx, *compiler_arguments,
                "--esp-sr-include", str(sdk), "--build-dir", str(build / "diagnostic_publish"),
            ])
            run("Production page readout and capacity guard", [
                sys.executable, str(tests / "page_readout_test.py"),
                "--cxx", args.cxx, *compiler_arguments,
                "--arduino-json-src", str(arduino_json), "--esp32-core-dir", str(core),
                "--speech-manifest", str(manifest), "--build-dir", str(build / "page_readout"),
            ])
            if not args.skip_flash:
                run("Five flash validation checks", [
                    sys.executable, str(tests / "test_flash_validation.py"),
                    "--build-dir", str(args.firmware_build_dir.resolve()), "-v",
                ])
            report["passed"] = True
        except (OSError, subprocess.CalledProcessError) as error:
            report["error"] = str(error)
            print(f"Host verification failed: {error}", file=sys.stderr)
            log.write(f"Host verification failed: {error}\n")
        finally:
            report["finished_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            (build / "host-verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["passed"]:
        return 1
    print(f"PASS: host verification; reports in {build}")
    if args.skip_flash:
        print("Flash checks were explicitly skipped; rerun after building the firmware to validate all five.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
