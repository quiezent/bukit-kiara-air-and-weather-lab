#!/usr/bin/env python3
"""Compile the public firmware; never invokes Arduino's generic uploader."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
FQBN = 'esp32:esp32:esp32s3:USBMode=hwcdc,CDCOnBoot=cdc,FlashSize=16M,FlashMode=qio,PSRAM=opi,PartitionScheme=custom,EraseFlash=none'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
    sketch = ROOT / 'WeatherDashboard'
    if not (sketch / 'SpeechClips.cpp').is_file() or not (ROOT / 'build/speech/manifest.json').is_file():
        parser.error('Run python tools/prepare_speech.py --espeak-ng PATH first')
    if not (sketch / 'flash/model.bin').is_file():
        parser.error('Speech model is missing; run setup.py or WeatherDashboard/tools/prepare_model.py')
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    build = ROOT / 'build/firmware'
    build.mkdir(parents=True, exist_ok=True)
    command = [cli, '--config-file', str(config), 'compile', '--fqbn', FQBN,
               '--build-property', 'upload.maximum_size=8388608', '--warnings', 'all',
               '--jobs', str(args.jobs), '--build-path', str(build), str(sketch)]
    result = subprocess.run(command, text=True, encoding='utf-8', errors='replace',
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (build / 'build.log').write_text(result.stdout, encoding='utf-8')
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
        'firmware': 'weather-dashboard-31-public.1', 'built_at_utc': datetime.now(timezone.utc).isoformat(),
        'fqbn': FQBN, 'toolchain': {key: value for key, value in toolchain.items() if key != 'arduino_cli'},
        'warnings': len(warnings), 'images': images,
        'scope': 'Compiled public source with locally generated eSpeak audio; this is not the installed David-audio image.'
    }, indent=2) + '\n', encoding='utf-8')
    print('Build complete. Validate and use WeatherDashboard/tools/flash_firmware.py for flashing.')


if __name__ == '__main__':
    main()
