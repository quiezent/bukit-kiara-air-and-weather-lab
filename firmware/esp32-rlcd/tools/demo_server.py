#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Serve synthetic RLCD schema-v1 data; no sensor, weather API or trained model.

Local preview (Python standard library only)::

    python tools/demo_server.py

Explicit LAN discovery (install ``zeroconf`` first)::

    python tools/demo_server.py --host 0.0.0.0 --advertise-ip YOUR_LAN_IPV4

The optional advertisement is ``_ttdi-weather._tcp.local.``. Use a computer on
the same LAN as the board, and stop other services advertising that type while
testing: the board may select any compatible discovered service.
"""

from __future__ import annotations

import argparse
import copy
import ipaddress
import json
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import socket
import sys
import time
from typing import Any
from urllib.parse import urlsplit
import uuid


API_PATH = "/api/rlcd/v1"
SERVICE_TYPE = "_ttdi-weather._tcp.local."
MAX_BODY_BYTES = 16384
MALAYSIA = timezone(timedelta(hours=8))
FIXTURE_PATH = Path(__file__).resolve().parents[1] / "examples" / "rlcd-v1-synthetic.json"


def load_fixture(path: Path = FIXTURE_PATH) -> dict[str, Any]:
    """Load the checked-in example without contacting any external service."""
    with path.open(encoding="utf-8") as fixture_file:
        fixture = json.load(fixture_file)
    if (not isinstance(fixture, dict) or fixture.get("schema_version") != 1
            or fixture.get("demo") is not True
            or fixture.get("timezone") != "Asia/Kuala_Lumpur"):
        raise ValueError("Expected the marked synthetic schema-v1 fixture")
    return fixture


def _shift_epoch_fields(value: Any, delta: int) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key.endswith("_epoch") and type(child) is int:
                value[key] = child + delta
            else:
                _shift_epoch_fields(child, delta)
    elif isinstance(value, list):
        for child in value:
            _shift_epoch_fields(child, delta)


def _next_window(now: int, hour: int, duration_hours: int,
                 *, allow_active: bool = False) -> tuple[int, int]:
    local_now = datetime.fromtimestamp(now, MALAYSIA)
    start = local_now.replace(hour=hour, minute=0, second=0, microsecond=0)
    end = start + timedelta(hours=duration_hours)
    if (end <= local_now if allow_active else start <= local_now):
        start += timedelta(days=1)
        end += timedelta(days=1)
    return int(start.timestamp()), int(end.timestamp())


def build_payload(now: int | None = None,
                  fixture: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a new example with coherent clocks, leaving the fixture intact.

    Values remain visibly synthetic and deliberately different: a +90-minute
    point, a two-hour mean, predicted window extrema, and event probabilities
    are separate quantities. No model is evaluated here.
    """
    if now is None:
        now = int(time.time())
    if type(now) is not int or not 1700000000 <= now <= 0xFFFFFFFF - 172800:
        raise ValueError("now must be a supported UTC Unix epoch in seconds")
    template = load_fixture() if fixture is None else fixture
    payload = copy.deepcopy(template)
    delta = now - payload["generated_epoch"]
    _shift_epoch_fields(payload, delta)
    # Row-array timestamps are positional, rather than *_epoch object fields.
    for point in payload["history"]["points"]:
        point[0] += delta
    for gap in payload["history"]["gaps"]:
        gap[0] += delta
        gap[1] += delta

    forecast = payload["forecast"]
    issued = forecast["issued_epoch"]
    near = forecast["near90"]
    ride = forecast["ride90_210"]
    near["target_epoch"] = issued + 90 * 60
    ride["start_epoch"] = near["target_epoch"]
    ride["end_epoch"] = issued + 210 * 60

    # Malaysia has a fixed UTC+8 offset. Hourly rain describes the preceding
    # hour; the first next_hours interval contains the current instant.
    current_hour = now // 3600 * 3600
    weather = payload["weather"]
    for offset, hour in enumerate([weather["current_hour"], *weather["next_hours"]]):
        hour["valid_epoch"] = current_hour + offset * 3600
        hour["rain_start_epoch"] = hour["valid_epoch"] - 3600
        hour["rain_end_epoch"] = hour["valid_epoch"]

    for name, hour in (("morning", 9), ("afternoon", 14)):
        session = forecast["sessions"][name]
        session["issued_epoch"] = issued
        session["start_epoch"], session["end_epoch"] = _next_window(now, hour, 2)
        session["logistics_epoch"] = session["start_epoch"] - 90 * 60
    tennis = payload["tennis_morning"]
    tennis["start_epoch"], tennis["end_epoch"] = _next_window(now, 7, 2, allow_active=True)
    tennis["active"] = tennis["start_epoch"] <= now < tennis["end_epoch"]
    return payload


def encode_payload(payload: dict[str, Any]) -> bytes:
    """Encode strict UTF-8 JSON within the firmware's HTTP body limit."""
    body = json.dumps(payload, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")
    if len(body) >= MAX_BODY_BYTES:
        raise ValueError("Synthetic API body must be smaller than 16384 bytes")
    return body


def make_handler(fixture: dict[str, Any], clock=time.time) -> type[BaseHTTPRequestHandler]:
    class DemoHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            if urlsplit(self.path).path != API_PATH:
                body = b'Synthetic demo API: GET /api/rlcd/v1\n'
                self.send_response(404)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
            else:
                body = encode_payload(build_payload(int(clock()), fixture))
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-RLCD-Synthetic-Demo", "true")
            self.end_headers()
            self.wfile.write(body)

    return DemoHandler


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", choices=("127.0.0.1", "0.0.0.0"), default="127.0.0.1",
                        help="default: local preview; LAN requires --advertise-ip too")
    parser.add_argument("--port", type=int, default=8766,
                        help="HTTP port (default 8766; leaves production port 8765 alone)")
    parser.add_argument("--advertise-ip", metavar="LAN_IPV4",
                        help="explicit local LAN address for optional zeroconf discovery")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if (args.host == "0.0.0.0") != bool(args.advertise_ip):
        parser.error("LAN mode requires both --host 0.0.0.0 and --advertise-ip LAN_IPV4")
    if args.advertise_ip:
        try:
            address = ipaddress.IPv4Address(args.advertise_ip)
        except ipaddress.AddressValueError:
            parser.error("--advertise-ip must be an IPv4 address on this computer's LAN")
        lan_networks = (ipaddress.IPv4Network("10.0.0.0/8"),
                        ipaddress.IPv4Network("172.16.0.0/12"),
                        ipaddress.IPv4Network("192.168.0.0/16"),
                        ipaddress.IPv4Network("169.254.0.0/16"))
        if not any(address in network for network in lan_networks):
            parser.error("--advertise-ip must be a private or link-local LAN IPv4 address")
    return args


def advertise(ip: str, port: int):
    """Register only an explicitly requested LAN address; return cleanup handles."""
    try:
        from zeroconf import IPVersion, ServiceInfo, Zeroconf
    except ImportError as error:
        raise RuntimeError("LAN discovery needs: python -m pip install zeroconf") from error
    suffix = uuid.uuid4().hex[:8]
    service = ServiceInfo(
        SERVICE_TYPE,
        f"RLCD synthetic demo {suffix}.{SERVICE_TYPE}",
        addresses=[socket.inet_aton(ip)],
        port=port,
        properties={"schema_version": "1", "api_path": API_PATH, "demo": "true"},
        server=f"rlcd-synthetic-demo-{suffix}.local.",
    )
    zeroconf = Zeroconf(interfaces=[ip], ip_version=IPVersion.V4Only)
    try:
        zeroconf.register_service(service)
    except Exception:
        zeroconf.close()
        raise
    return zeroconf, service


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    fixture = load_fixture()
    encode_payload(build_payload(fixture=fixture))  # Fail before binding if oversized.
    server = ThreadingHTTPServer((args.host, args.port), make_handler(fixture))
    server.daemon_threads = True
    discovery = service = None
    try:
        if args.advertise_ip:
            discovery, service = advertise(args.advertise_ip, server.server_port)
        host = args.advertise_ip or args.host
        print(f"SYNTHETIC DEMO ONLY — no real sensor readings or trained predictions.\n"
              f"http://{host}:{server.server_port}{API_PATH}", flush=True)
        if discovery:
            print(f"Advertising {SERVICE_TYPE} on the explicitly selected LAN.", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1
    finally:
        if discovery:
            try:
                discovery.unregister_service(service)
            finally:
                discovery.close()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
