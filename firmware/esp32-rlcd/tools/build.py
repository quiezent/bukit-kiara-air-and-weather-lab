#!/usr/bin/env python3
"""Compile the public firmware; never invokes Arduino's generic uploader.

Requires tools/setup.py (pinned toolchain) and tools/prepare_speech.py (the
generated WeatherDashboard/SpeechClips.cpp and build/speech/manifest.json).
Before compiling it checks that WeatherDashboard/WebPage.h matches
WeatherDashboard/web/index.html. Any compiler warning fails the build.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
FIRMWARE_ID = 'weather-dashboard-40-public.1'
FQBN = 'esp32:esp32:esp32s3:USBMode=hwcdc,CDCOnBoot=cdc,FlashSize=16M,FlashMode=qio,PSRAM=opi,PartitionScheme=custom,EraseFlash=none'
# The 7 MiB factory slot of partitions.csv, and the WebServer request-line and
# header waits (1 s and 2 s instead of 5 s and 10 s), which the sketch static_asserts.
BUILD_PROPERTIES = (
    'upload.maximum_size=7340032',
    'compiler.cpp.extra_flags=-DWEBSERVER_MAX_LINE_WAIT=1000 -DWEBSERVER_MAX_HEADER_WAIT=2000',
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--arduino-cli', help='Override the CLI recorded by setup.py')
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    state = ROOT / '.state'
    config = state / 'arduino-cli.yaml'
    if not config.is_file() or not (state / 'toolchain.json').is_file():
        parser.error('Run python tools/setup.py first')
    toolchain = json.loads((state / 'toolchain.json').read_text(encoding='utf-8'))
    cli = args.arduino_cli or toolchain['arduino_cli'] or shutil.which('arduino-cli')
    if not cli or not Path(cli).is_file():
        parser.error('Arduino CLI executable is missing; rerun setup.py or pass --arduino-cli PATH')
    version = subprocess.run([cli, 'version'], check=True, capture_output=True, text=True).stdout
    if not re.search(r'\bVersion:\s*1\.5\.1\b', version):
        parser.error('This source snapshot requires Arduino CLI 1.5.1')
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    sketch = ROOT / 'WeatherDashboard'
    declared = re.findall(r'kFirmwareVersion = "([^"]+)"', (sketch / 'WeatherDashboard.ino').read_text(encoding='utf-8'))
    if declared != [FIRMWARE_ID]:
        parser.error(f'WeatherDashboard.ino declares {declared}; this build script records {FIRMWARE_ID}')
    clips, speech = sketch / 'SpeechClips.cpp', ROOT / 'build/speech/manifest.json'
    if not clips.is_file() or not speech.is_file():
        parser.error('Run python tools/prepare_speech.py --espeak-ng PATH first')
    speech_manifest = json.loads(speech.read_text(encoding='utf-8'))
    if (speech_manifest.get('source_cpp_sha256') != sha256(clips)
            or speech_manifest.get('source_header_sha256') != sha256(sketch / 'SpeechClips.h')):
        parser.error('SpeechClips.cpp, SpeechClips.h and build/speech/manifest.json are not from one '
                     'prepare_speech.py run; rerun python tools/prepare_speech.py')
    web = subprocess.run([sys.executable, str(ROOT / 'tools/build_web.py'), '--check'], check=False)
    if web.returncode:
        parser.error('WeatherDashboard/WebPage.h does not match web/index.html; run python tools/build_web.py')

    build = ROOT / 'build/firmware'
    build.mkdir(parents=True, exist_ok=True)
    command = [cli, '--config-file', str(config), 'compile', '--fqbn', FQBN]
    for prop in BUILD_PROPERTIES:
        command += ['--build-property', prop]
    command += ['--warnings', 'all', '--jobs', str(args.jobs), '--build-path', str(build), str(sketch)]
    result = subprocess.run(command, text=True, encoding='utf-8', errors='replace',
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (build / 'build.log').write_text(result.stdout, encoding='utf-8', newline='\n')
    print(result.stdout, end='')
    if result.returncode:
        raise SystemExit(result.returncode)
    warnings = [line for line in result.stdout.splitlines() if 'warning:' in line.lower()]
    if warnings:
        raise SystemExit('Build contains warnings; inspect build/firmware/build.log')
    images = []
    for name in ('WeatherDashboard.ino.bootloader.bin', 'WeatherDashboard.ino.partitions.bin',
                 'WeatherDashboard.ino.bin'):
        path = build / name
        data = path.read_bytes()
        images.append({'file': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    (build / 'build-manifest.json').write_text(json.dumps({
        'firmware': FIRMWARE_ID, 'built_at_utc': datetime.now(timezone.utc).isoformat(),
        'fqbn': FQBN, 'build_properties': list(BUILD_PROPERTIES),
        'toolchain': {key: value for key, value in toolchain.items() if key != 'arduino_cli'},
        'speech_manifest_sha256': sha256(speech), 'web_page_sha256': sha256(sketch / 'WebPage.h'),
        'warnings': len(warnings), 'images': images,
        'scope': 'Compiled public source with locally generated eSpeak NG clips; this is not the image '
                 'installed on the owner\'s device, which carries different prerecorded clips.'
    }, indent=2) + '\n', encoding='utf-8', newline='\n')
    print('Build complete. Validate and use WeatherDashboard/tools/flash_firmware.py for flashing.')


if __name__ == '__main__':
    main()
