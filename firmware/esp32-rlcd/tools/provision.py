#!/usr/bin/env python3
"""Provision Wi-Fi over USB with a hidden password prompt; no saved-PC credential lookup."""
import argparse
import getpass
import json
import math
import time
import serial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('status', 'wifi', 'scan', 'discover'))
    parser.add_argument('--port', required=True, help='Serial port such as COM5 or /dev/ttyACM0')
    parser.add_argument('--ssid', help='Network name; requested interactively when omitted')
    parser.add_argument('--open-network', action='store_true')
    parser.add_argument('--timeout', type=float, default=45)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--timeout must be a positive finite number of seconds')
    command = args.action.upper() + '\n'
    if args.action == 'wifi':
        ssid = args.ssid or input('2.4 GHz Wi-Fi network name: ')
        password = '' if args.open_network else getpass.getpass('Wi-Fi password: ')
        if not 1 <= len(ssid.encode('utf-8')) <= 32 or len(password.encode('utf-8')) > 64:
            parser.error('SSID must be 1..32 UTF-8 bytes and password at most 64 bytes')
        command = 'WIFI ' + ssid.encode('utf-8').hex() + ' ' + (password.encode('utf-8').hex() or '-') + '\n'
        del password
    port = serial.Serial(port=None, baudrate=115200, timeout=0.5)
    port.dtr = False
    port.rts = False
    port.port = args.port
    port.open()
    with port:
        time.sleep(2)
        port.reset_input_buffer()
        port.write(command.encode('ascii'))
        del command
        deadline, next_poll = time.monotonic() + args.timeout, time.monotonic() + 5
        while time.monotonic() < deadline:
            # A quiet port (or a non-JSON boot log) must not suppress retries.
            # Poll before reading so every path through the loop schedules it.
            now = time.monotonic()
            if now >= next_poll:
                port.write(b'STATUS\n')
                next_poll = now + 5
            line = port.readline().decode('utf-8', errors='replace').strip()
            try:
                result = json.loads(line)
            except ValueError:
                continue
            if not isinstance(result, dict):
                continue
            if result.get('error'):
                raise SystemExit('Device reported: ' + result['error'])
            if args.action == 'scan' and 'networks' in result:
                print(json.dumps(result, indent=2))
                return
            if 'firmware' in result and 'connected' in result:
                if args.action != 'wifi' or (result['connected'] and result.get('credentials_saved') and result.get('ssid') == ssid):
                    print(json.dumps(result, indent=2))
                    return
        raise SystemExit('Timed out waiting for the device')


if __name__ == '__main__':
    main()
