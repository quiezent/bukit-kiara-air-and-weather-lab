# SPDX-License-Identifier: Apache-2.0
"""Mock-only provisioning checks; no device, network, or pyserial install needed."""

from collections import deque
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import Mock, patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "provision.py"
SPEC = importlib.util.spec_from_file_location("rlcd_provision", MODULE_PATH)
provision = importlib.util.module_from_spec(SPEC)
# Production uses pinned pyserial. Tests deliberately substitute the entire
# serial surface before import, so a test can never open a real USB port.
serial_stub = types.ModuleType("serial")
serial_stub.Serial = Mock()
with patch.dict(sys.modules, {"serial": serial_stub}):
    SPEC.loader.exec_module(provision)


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, duration):
        self.sleeps.append(duration)
        self.now += duration


class FakeSerial:
    def __init__(self, clock, lines=(), on_write=None, read_error=None):
        self.clock = clock
        self.lines = deque(lines)
        self.on_write = on_write
        self.read_error = read_error
        self.writes = []
        self.is_open = False
        self.closed = False
        self.reset_count = 0
        self.read_count = 0

    def open(self):
        self.open_settings = (self.port, self.dtr, self.rts)
        self.is_open = True

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()

    def close(self):
        self.closed = True
        self.is_open = False

    def reset_input_buffer(self):
        self.reset_count += 1

    def write(self, data):
        self.writes.append((self.clock.now, data))
        if self.on_write:
            self.on_write(self, data)
        return len(data)

    def readline(self):
        self.read_count += 1
        self.clock.now += 0.5
        if self.read_error:
            raise self.read_error
        return self.lines.popleft() if self.lines else b""


def status(ssid="synthetic-network", **changes):
    result = {"firmware": "synthetic-test", "connected": True,
              "credentials_saved": True, "ssid": ssid}
    result.update(changes)
    return json.dumps(result).encode("utf-8") + b"\n"


class ProvisionTests(unittest.TestCase):
    @contextlib.contextmanager
    def invoke(self, arguments, *, lines=(), password="mock-only-secret",
               entered_ssid="synthetic-network", on_write=None, read_error=None):
        clock = FakeClock()
        port = FakeSerial(clock, lines, on_write, read_error)
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.ExitStack() as stack:
            constructor = stack.enter_context(patch.object(provision.serial, "Serial", return_value=port))
            prompt = stack.enter_context(patch.object(provision.getpass, "getpass", return_value=password))
            network_prompt = stack.enter_context(patch("builtins.input", return_value=entered_ssid))
            stack.enter_context(patch.object(provision.time, "monotonic", clock.monotonic))
            stack.enter_context(patch.object(provision.time, "sleep", clock.sleep))
            stack.enter_context(patch.object(sys, "argv", [str(MODULE_PATH), *arguments]))
            stack.enter_context(contextlib.redirect_stdout(stdout))
            stack.enter_context(contextlib.redirect_stderr(stderr))
            # Any attempt to invoke a saved-credential lookup command fails the
            # test. No host command or environment-specific Wi-Fi tool is used.
            lookups = [stack.enter_context(patch.object(subprocess, name,
                       side_effect=AssertionError("No host credential lookup is allowed")))
                       for name in ("run", "Popen", "check_output", "check_call", "call")]
            lookups += [stack.enter_context(patch.object(os, name,
                        side_effect=AssertionError("No host credential lookup is allowed")))
                        for name in ("system", "popen")]
            result = types.SimpleNamespace(clock=clock, port=port, constructor=constructor,
                                           prompt=prompt, network_prompt=network_prompt,
                                           stdout=stdout, stderr=stderr)
            yield result
            for lookup in lookups:
                lookup.assert_not_called()

    def test_wifi_uses_hidden_password_and_utf8_hex_without_host_lookup(self):
        ssid, password = "山地🌦", "秘密-mock-only"
        with self.invoke(["wifi", "--port", "MOCK", "--ssid", ssid],
                         lines=[status(ssid)], password=password) as run:
            provision.main()
            expected = f"WIFI {ssid.encode('utf-8').hex()} {password.encode('utf-8').hex()}\n".encode("ascii")
            self.assertEqual([data for _, data in run.port.writes], [expected])
            run.prompt.assert_called_once_with("Wi-Fi password: ")
            run.network_prompt.assert_not_called()
            output = run.stdout.getvalue() + run.stderr.getvalue()
            self.assertNotIn(password, output)
            self.assertNotIn(password.encode("utf-8").hex(), output)
            self.assertEqual(json.loads(run.stdout.getvalue())["ssid"], ssid)
            run.constructor.assert_called_once_with(port=None, baudrate=115200, timeout=0.5)
            self.assertEqual(run.port.open_settings, ("MOCK", False, False))
            self.assertEqual(run.port.reset_count, 1)
            self.assertTrue(run.port.closed)

    def test_open_network_uses_empty_password_marker_and_no_password_prompt(self):
        with self.invoke(["wifi", "--port", "/dev/mock", "--open-network"],
                         lines=[status()]) as run:
            provision.main()
            run.prompt.assert_not_called()
            run.network_prompt.assert_called_once_with("2.4 GHz Wi-Fi network name: ")
            expected = b"WIFI " + b"synthetic-network".hex().encode("ascii") + b" -\n"
            self.assertEqual(run.port.writes[0][1], expected)
            self.assertTrue(run.port.closed)

    def test_quiet_wifi_port_polls_status_and_recovers(self):
        def reply_to_poll(port, command):
            if command == b"STATUS\n":
                port.lines.append(status())

        with self.invoke(["wifi", "--port", "MOCK", "--ssid", "synthetic-network", "--timeout", "12"],
                         on_write=reply_to_poll) as run:
            provision.main()
            self.assertEqual(run.port.writes[1], (7.0, b"STATUS\n"))
            self.assertGreaterEqual(run.port.read_count, 11)
            self.assertTrue(json.loads(run.stdout.getvalue())["connected"])
            self.assertTrue(run.port.closed)

    def test_boot_logs_and_non_object_json_do_not_suppress_status_poll(self):
        def reply_to_poll(port, command):
            if command == b"STATUS\n":
                port.lines.append(status())

        noise = [b"boot log\n", b"\xffnot JSON\n", b"[]\n", b"null\n", b"7\n"]
        with self.invoke(["wifi", "--port", "MOCK", "--ssid", "synthetic-network", "--timeout", "12"],
                         lines=noise, on_write=reply_to_poll) as run:
            provision.main()
            self.assertEqual(run.port.writes[1], (7.0, b"STATUS\n"))
            self.assertTrue(run.port.closed)

    def test_quiet_status_timeout_retries_and_closes_port(self):
        with self.invoke(["status", "--port", "MOCK", "--timeout", "12"]) as run:
            with self.assertRaisesRegex(SystemExit, "Timed out waiting for the device"):
                provision.main()
            self.assertEqual(run.port.writes,
                             [(2.0, b"STATUS\n"), (7.0, b"STATUS\n"), (12.0, b"STATUS\n")])
            self.assertEqual(run.clock.now, 14.0)
            self.assertEqual(run.stdout.getvalue(), "")
            self.assertTrue(run.port.closed)

    def test_wifi_waits_for_connected_saved_matching_ssid(self):
        def reply_to_poll(port, command):
            if command == b"STATUS\n":
                port.lines.append(status())

        lines = [status(connected=False), status(credentials_saved=False),
                 status(ssid="other-synthetic-network"),
                 b'{"firmware":"synthetic-test","connected":true}\n']
        with self.invoke(["wifi", "--port", "MOCK", "--ssid", "synthetic-network", "--timeout", "12"],
                         lines=lines, on_write=reply_to_poll) as run:
            provision.main()
            self.assertEqual(run.port.writes[1][1], b"STATUS\n")
            self.assertEqual(json.loads(run.stdout.getvalue())["ssid"], "synthetic-network")
            self.assertTrue(run.port.closed)

    def test_status_scan_and_discover_send_only_the_requested_command(self):
        for action, line in (("status", status()), ("discover", status()),
                             ("scan", b'{"networks":[{"ssid":"synthetic-network"}]}\n')):
            with self.subTest(action=action), self.invoke([action, "--port", "MOCK"], lines=[line]) as run:
                provision.main()
                self.assertEqual([data for _, data in run.port.writes], [action.upper().encode() + b"\n"])
                run.prompt.assert_not_called()
                self.assertTrue(run.port.closed)

    def test_device_error_closes_port_without_printing_credentials(self):
        with self.invoke(["wifi", "--port", "MOCK", "--ssid", "synthetic-network"],
                         lines=[b'{"error":"Synthetic Wi-Fi failure"}\n']) as run:
            with self.assertRaisesRegex(SystemExit, "Device reported: Synthetic Wi-Fi failure"):
                provision.main()
            self.assertEqual(run.stdout.getvalue(), "")
            self.assertTrue(run.port.closed)

    def test_serial_read_error_closes_port(self):
        with self.invoke(["status", "--port", "MOCK"], read_error=OSError("Synthetic port disconnected")) as run:
            with self.assertRaisesRegex(OSError, "Synthetic port disconnected"):
                provision.main()
            self.assertTrue(run.port.closed)

    def test_credential_lengths_are_validated_in_utf8_bytes_before_opening_port(self):
        for ssid, password in (("山" * 11, "mock-only-secret"),
                               ("synthetic-network", "密" * 22)):
            with self.subTest(ssid_bytes=len(ssid.encode("utf-8"))), self.invoke(
                    ["wifi", "--port", "MOCK", "--ssid", ssid], password=password) as run:
                with self.assertRaises(SystemExit) as failure:
                    provision.main()
                self.assertEqual(failure.exception.code, 2)
                run.constructor.assert_not_called()
                self.assertNotIn(password, run.stdout.getvalue() + run.stderr.getvalue())

    def test_invalid_timeouts_are_rejected_before_password_prompt_or_port_open(self):
        for timeout in ("0", "-1", "nan", "inf"):
            with self.subTest(timeout=timeout), self.invoke(
                    ["wifi", "--port", "MOCK", "--ssid", "synthetic-network", "--timeout", timeout]) as run:
                with self.assertRaises(SystemExit) as failure:
                    provision.main()
                self.assertEqual(failure.exception.code, 2)
                run.prompt.assert_not_called()
                run.constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
