#!/usr/bin/env python3
"""Run the firmware's host regressions without opening a device or using the network.

Use the package setup first to install the pinned Arduino libraries and to
generate the speech clips. Example: python tools/test_host.py --cxx g++
Pass repeatable --cxx-arg=ARG for compiler drivers such as `zig c++`.
The two trainer parity tests need numpy, scipy and scikit-learn (requirements.txt).
Reports and generated harnesses are written under build/host by default.
"""

import argparse
import datetime
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys


PACKAGE = Path(__file__).resolve().parents[1]
FLAGS = ["-std=c++17", "-Wall", "-Wextra", "-Werror"]
PARITY_PACKAGES = {"numpy": "numpy", "scipy": "scipy", "sklearn": "scikit-learn"}


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
    parser.add_argument("--speech-manifest", type=Path, default=PACKAGE / "build/speech/manifest.json")
    parser.add_argument("--build-dir", type=Path, default=PACKAGE / "build/host",
                        help="Generated test sources, executables and reports")
    parser.add_argument("--firmware-build-dir", type=Path, default=PACKAGE / "build/firmware",
                        help="Matching compiled images for the flash checks")
    parser.add_argument("--skip-flash", action="store_true",
                        help="Explicitly run the host checks only, before a firmware image set has been built")
    args = parser.parse_args()

    data = args.arduino_data_dir.resolve()
    user = args.arduino_user_dir.resolve()
    arduino_json = (args.arduino_json_src or user / "libraries/ArduinoJson/src").resolve()
    core = (args.esp32_core_dir or data / "packages/esp32/hardware/esp32/3.3.12/cores/esp32").resolve()
    manifest = args.speech_manifest.resolve()
    build = args.build_dir.resolve()
    firmware = PACKAGE / "WeatherDashboard"
    tests = firmware / "tests"
    trainer_tests = ("hgb_trainer_test.py", "arrival_logistic_test.py")
    sources = [PACKAGE / "tools/build_web.py"] + [tests / name for name in (
        "weather_readout_test.cpp", "arrival_change_test.py", "indoor_compensation_test.py",
        "indoor_sensor_test.py", "page_readout_test.py", *trainer_tests, "test_flash_validation.py")]
    for required in sources:
        if not required.is_file():
            parser.error(f"Package test file not found: {required}; use the complete firmware source package")
    if not args.skip_flash and not args.firmware_build_dir.is_dir():
        parser.error("Build the firmware first, specify --firmware-build-dir, or explicitly use --skip-flash")
    for required in (arduino_json / "ArduinoJson.h", core / "stdlib_noniso.c", manifest):
        if not required.is_file():
            parser.error(f"Required input not found: {required}; run package setup or supply a dependency override")
    missing = [package for module, package in PARITY_PACKAGES.items() if importlib.util.find_spec(module) is None]
    if missing:
        parser.error(f"The trainer parity tests need {', '.join(missing)}; "
                     "install them with python -m pip install -r requirements.txt")

    build.mkdir(parents=True, exist_ok=True)
    report = {
        "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "compiler": args.cxx,
        "compiler_arguments": args.cxx_arg,
        "required_compile_flags": FLAGS,
        "python": sys.version.split()[0],
        "python_packages": {package: importlib.metadata.version(package) for package in PARITY_PACKAGES.values()},
        "dependencies": {"arduino_json_src": str(arduino_json), "esp32_core_dir": str(core),
                         "speech_manifest": str(manifest)},
        "flash_checks": "explicitly skipped" if args.skip_flash else "required",
        "scope": "Host C++, Python parity and compiled-file checks only; no network or device access.",
        "steps": [],
        "passed": False,
    }
    compiler = [args.cxx, *args.cxx_arg]
    suffix = ".exe" if sys.platform == "win32" else ""
    compiler_arguments = [f"--cxx-arg={argument}" for argument in args.cxx_arg]

    with (build / "host-verification.log").open("w", encoding="utf-8", newline="\n") as log:
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
            run("Web page header matches web/index.html", [
                sys.executable, str(PACKAGE / "tools/build_web.py"), "--check",
            ])
            executable = build / ("weather_readout_test" + suffix)
            run("Compile weather_readout_test", [*compiler, *FLAGS, "-I", str(firmware),
                                                 str(tests / "weather_readout_test.cpp"), "-o", str(executable)])
            run("Run weather_readout_test", [str(executable)])
            run("Arrival-change decoder", [
                sys.executable, str(tests / "arrival_change_test.py"),
                "--cxx", args.cxx, *compiler_arguments,
                "--arduino-json-src", str(arduino_json), "--build-dir", str(build / "arrival_change"),
            ])
            run("Indoor temperature and RH compensation", [
                sys.executable, str(tests / "indoor_compensation_test.py"),
                "--cxx", args.cxx, *compiler_arguments,
                "--build-dir", str(build / "indoor_compensation"),
            ])
            run("Indoor sensor and correction lifecycle", [
                sys.executable, str(tests / "indoor_sensor_test.py"),
                "--cxx", args.cxx, *compiler_arguments,
                "--build-dir", str(build / "indoor_sensor"),
            ])
            run("Page readout and capacity guard", [
                sys.executable, str(tests / "page_readout_test.py"),
                "--cxx", args.cxx, *compiler_arguments,
                "--arduino-json-src", str(arduino_json), "--esp32-core-dir", str(core),
                "--speech-manifest", str(manifest), "--build-dir", str(build / "page_readout"),
            ])
            run("Gradient-boosting trainer parity with scikit-learn", [
                sys.executable, str(tests / "hgb_trainer_test.py"),
                "--cxx", args.cxx, *compiler_arguments, "--build-dir", str(build / "hgb_trainer"),
            ])
            run("Arrival logistic trainer parity with scikit-learn", [
                sys.executable, str(tests / "arrival_logistic_test.py"),
                "--cxx", args.cxx, *compiler_arguments, "--build-dir", str(build / "arrival_logistic"),
            ])
            if not args.skip_flash:
                run("Flash validation checks", [
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
            (build / "host-verification.json").write_text(json.dumps(report, indent=2) + "\n",
                                                          encoding="utf-8", newline="\n")
    if not report["passed"]:
        return 1
    print(f"PASS: host verification; reports in {build}")
    if args.skip_flash:
        print("Flash checks were explicitly skipped; rerun after building the firmware to run them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
