#!/usr/bin/env python3
"""Install the pinned Arduino dependencies into this package's ignored cache."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CORE = '3.3.12'
CLI = '1.5.1'
INDEX = 'https://espressif.github.io/arduino-esp32/package_esp32_index.json'


def run(command):
    subprocess.run([str(part) for part in command], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arduino-cli', default=shutil.which('arduino-cli'), help='Arduino CLI 1.5.1 executable')
    parser.add_argument('--downloads-dir', type=Path, help='Optional existing Arduino download cache')
    args = parser.parse_args()
    if not args.arduino_cli:
        parser.error('Install Arduino CLI 1.5.1 or pass --arduino-cli PATH')
    cli = Path(args.arduino_cli).resolve()
    version = subprocess.run([str(cli), 'version'], check=True, capture_output=True, text=True).stdout
    if not re.search(r'\bVersion:\s*' + re.escape(CLI) + r'\b', version):
        parser.error('This source snapshot requires Arduino CLI ' + CLI)
    state = ROOT / '.state'
    data, user = state / 'arduino-data', state / 'arduino-user'
    downloads = args.downloads_dir.resolve() if args.downloads_dir else state / 'arduino-downloads'
    for directory in (state, data, user, downloads):
        directory.mkdir(parents=True, exist_ok=True)
    config = state / 'arduino-cli.yaml'
    config.write_text('board_manager:\n  additional_urls:\n    - ' + INDEX + '\ndirectories:\n'
                      + ''.join('  ' + key + ': ' + json.dumps(path.as_posix()) + '\n'
                                for key, path in (('data', data), ('downloads', downloads), ('user', user))),
                      encoding='utf-8')
    command = [cli, '--config-file', config]
    run([*command, 'core', 'update-index'])
    run([*command, 'core', 'install', 'esp32:esp32@' + CORE])
    run([*command, 'lib', 'install', 'U8g2@2.36.18', 'ArduinoJson@7.4.3'])
    sdk = data / 'packages/esp32/tools/esp32s3-libs' / CORE
    run([sys.executable, ROOT / 'WeatherDashboard/tools/prepare_model.py', '--sdk-dir', sdk])
    (state / 'toolchain.json').write_text(json.dumps({
        'arduino_cli': str(cli), 'arduino_cli_version': CLI,
        'arduino_esp32': CORE, 'esp_idf': '5.5.5', 'esp_sr': '2.5.3',
        'u8g2': '2.36.18', 'arduino_json': '7.4.3',
        'config': '.state/arduino-cli.yaml',
    }, indent=2) + '\n', encoding='utf-8')
    print('Pinned toolchain and original speech-model data prepared. Generate speech clips before building.')


if __name__ == '__main__':
    main()
