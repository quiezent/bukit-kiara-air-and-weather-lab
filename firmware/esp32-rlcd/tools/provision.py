#!/usr/bin/env python3
"""Send one USB serial command to the dashboard and print its JSON reply.

Actions and the firmware's newline-terminated commands (115200 baud):
  status    STATUS                  firmware, Wi-Fi, CPU, memory, collection and model state
  wifi      WIFI <ssid> <password>  join a 2.4 GHz network; the password prompt does not echo
  view      VIEW <n>                show page n (--page 0 overview, 1 sports, 2 history)
  readinfo  READINFO                same as releasing KEY: read the page aloud, or stop reading
  refresh   REFRESH                 start a data collection pass now
  seed      SEED                    open the ten-minute history-upload window (POST /seed)

No saved-PC credential lookup is used, and credentials are never printed.
"""
import argparse
import getpass
import json
import math
import time
import serial

PAGES = 3
# The event line the firmware prints once it has handled each command.
EVENTS = {'view': 'view', 'readinfo': 'readout', 'refresh': 'refresh', 'seed': 'seed_open'}


def is_status(result):
    return 'firmware' in result and isinstance(result.get('wifi'), dict) and 'connected' in result['wifi']


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=('status', 'wifi', 'view', 'readinfo', 'refresh', 'seed'))
    parser.add_argument('--port', required=True, help='Serial port such as COM5 or /dev/ttyACM0')
    parser.add_argument('--page', type=int, help='Page for view: 0 overview, 1 sports, 2 history')
    parser.add_argument('--ssid', help='Network name; requested interactively when omitted')
    parser.add_argument('--open-network', action='store_true')
    parser.add_argument('--timeout', type=float, default=60)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--timeout must be a positive finite number of seconds')
    if args.action == 'view':
        if args.page is None or not 0 <= args.page < PAGES:
            parser.error(f'view needs --page 0..{PAGES - 1}')
        command = f'VIEW {args.page}\n'
    elif args.page is not None:
        parser.error('--page applies only to view')
    else:
        command = args.action.upper() + '\n'
    if args.action == 'wifi':
        ssid = args.ssid or input('2.4 GHz Wi-Fi network name: ')
        password = '' if args.open_network else getpass.getpass('Wi-Fi password: ')
        if not 1 <= len(ssid.encode('utf-8')) <= 32 or len(password.encode('utf-8')) > 64:
            parser.error('SSID must be 1..32 UTF-8 bytes and password at most 64 bytes')
        command = 'WIFI ' + ssid.encode('utf-8').hex() + ' ' + (password.encode('utf-8').hex() or '-') + '\n'
        del password
    expected = EVENTS.get(args.action)
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
        connecting = False
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
            event = result.get('event')
            # The readout event carries its own (usually empty) error field.
            if result.get('error') and event is None:
                raise SystemExit('Device reported: ' + str(result['error']))
            if args.action == 'status':
                if is_status(result):
                    print(json.dumps(result, indent=2))
                    return
            elif args.action == 'wifi':
                # Only a status after the device has started joining the new network counts.
                if event == 'wifi_connecting':
                    connecting = True
                elif connecting and is_status(result) and result['wifi']['connected'] is True:
                    print(json.dumps(result, indent=2))
                    return
            elif event == expected:
                print(json.dumps(result, indent=2))
                if args.action == 'readinfo' and result.get('started') is not True:
                    raise SystemExit('Read Info did not start: ' + (result.get('error') or 'no reason given'))
                return
            elif is_status(result):
                # Commands are answered in order, so this one was handled without its event.
                if args.action == 'readinfo':
                    print('No readout started: a readout that was playing has stopped, '
                          'or there is no forecast to read yet.')
                    return
                raise SystemExit(f'The device did not acknowledge {args.action.upper()}')
        raise SystemExit('Timed out waiting for the device')


if __name__ == '__main__':
    main()
