# SPDX-License-Identifier: Apache-2.0
"""Run from esp32-rlcd: python -m unittest discover -s tools/tests -v."""

import contextlib
import copy
from datetime import datetime
import importlib.util
import io
import json
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import build_opener, ProxyHandler


MODULE_PATH = Path(__file__).resolve().parents[1] / "demo_server.py"
SPEC = importlib.util.spec_from_file_location("rlcd_demo_server", MODULE_PATH)
demo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(demo)


def local_epoch(year, month, day, hour, minute=0):
    return int(datetime(year, month, day, hour, minute, tzinfo=demo.MALAYSIA).timestamp())


class PayloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = demo.load_fixture()

    def test_fixture_is_synthetic_and_all_models_are_marked(self):
        body = demo.encode_payload(self.fixture)
        self.assertLess(len(body), demo.MAX_BODY_BYTES)
        self.assertTrue(self.fixture["demo"])
        self.assertEqual(self.fixture["site_name"], "SYNTHETIC DEMO")
        self.assertEqual(self.fixture["timezone"], "Asia/Kuala_Lumpur")
        self.assertEqual(self.fixture["schema_version"], 1)
        self.assertEqual(self.fixture["current"]["display_text"], "Latest sensor reading")
        models = []

        def visit(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    if key == "model":
                        models.append(child)
                    if key == "source":
                        self.assertIn("SYNTHETIC DEMO", child)
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(self.fixture)
        self.assertGreaterEqual(len(models), 6)
        self.assertEqual(set(models), {"synthetic-demo-not-trained"})
        for forbidden in ("ssid", "password", "mac_address", "device_id", "sensor_id"):
            self.assertNotIn(f'"{forbidden}"', body.decode("utf-8"))

    def test_clock_shifts_are_coherent_and_do_not_mutate_fixture(self):
        original = copy.deepcopy(self.fixture)
        base = self.fixture["generated_epoch"]
        for now in (base - 7200, base, base + 125, base + 86400 * 366):
            with self.subTest(now=now):
                payload = demo.build_payload(now, self.fixture)
                self.assertEqual(payload["generated_epoch"], now)
                self.assertEqual(now - payload["current"]["observed_epoch"], 45)
                forecast = payload["forecast"]
                issued = forecast["issued_epoch"]
                self.assertEqual(now - issued, forecast["age_s"])
                self.assertEqual(forecast["near90"]["target_epoch"], issued + 5400)
                ride = forecast["ride90_210"]
                self.assertEqual(ride["start_epoch"], issued + 5400)
                self.assertEqual(ride["end_epoch"], issued + 12600)
                self.assertEqual(ride["end_epoch"] - ride["start_epoch"], 7200)
                for session in forecast["sessions"].values():
                    self.assertEqual(session["issued_epoch"], issued)
                    self.assertGreater(session["start_epoch"], issued)
                    self.assertEqual(session["end_epoch"] - session["start_epoch"], 7200)
                    self.assertEqual(session["logistics_epoch"], session["start_epoch"] - 5400)
                window_weather = [ride["weather"], *(s["weather"] for s in forecast["sessions"].values())]
                for weather in [payload["weather"], *window_weather]:
                    self.assertEqual(now - weather["fetched_epoch"], weather["age_s"])
                    self.assertLessEqual(weather["fetched_epoch"], issued)
                for weather in window_weather:
                    self.assertTrue(weather["coverage_verified"])
                    self.assertTrue(weather["coverage"]["complete"])
                self.assertLess(len(demo.encode_payload(payload)), demo.MAX_BODY_BYTES)
        self.assertEqual(self.fixture, original)
        first = demo.build_payload(base, self.fixture)
        first["forecast"]["near90"]["pm25_ugm3"] = -123
        self.assertEqual(self.fixture, original)

    def test_point_mean_extrema_and_first20_are_separate(self):
        payload = demo.build_payload(self.fixture["generated_epoch"], self.fixture)
        near = payload["forecast"]["near90"]
        ride = payload["forecast"]["ride90_210"]
        self.assertNotEqual(near["pm25_ugm3"], near["reference_pm25_ugm3"])
        self.assertNotEqual(near["pm25_ugm3"], ride["pm25_ugm3"])
        extrema = ride["minimum_maximum"]
        self.assertEqual(extrema["kind"], "predicted_window_minimum_maximum")
        self.assertEqual(extrema["resolution_minutes"], 15)
        self.assertLess(extrema["minimum_ugm3"], ride["pm25_ugm3"])
        self.assertGreater(extrema["maximum_ugm3"], ride["pm25_ugm3"])
        self.assertIsNone(ride["range_low_ugm3"])
        self.assertIsNone(ride["range_high_ugm3"])
        self.assertEqual(ride["range_kind"], "unavailable")
        for pm in [near, ride, *(s["pm"] for s in payload["forecast"]["sessions"].values())]:
            self.assertEqual(pm["role"], "experimental_model_output")
            self.assertFalse(pm["validated"])
            self.assertFalse(pm["qualified"])
            self.assertFalse(pm["used_for_decision"])
        event = near["first20"]
        self.assertAlmostEqual(sum(event[key] for key in
                                   ("rise_probability", "drop_probability", "none_probability")), 1)
        self.assertGreater(event["none_probability"], event["rise_probability"])
        self.assertGreater(event["none_probability"], event["drop_probability"])
        self.assertEqual(event["diagnostic_direction"], "unresolved")
        self.assertEqual(event["display_text"], "No large change expected")
        self.assertFalse(event["qualification"]["operational_use_eligible"])
        arrival = near["arrival_change"]
        self.assertEqual(arrival["outcome"], "within20")
        self.assertEqual(arrival["outcome_probability"], .921)
        self.assertEqual(arrival["display_text"], "No change on arrival")
        self.assertNotEqual(arrival["within20"], event["none_probability"])

    def test_native_arrival_clocks_reference_and_identity_survive_rebasing(self):
        base = self.fixture["generated_epoch"]
        original = copy.deepcopy(self.fixture)
        source = original["forecast"]["near90"]["arrival_change"]
        native_values = {key: value for key, value in source.items()
                         if not key.endswith("_epoch")}
        self.assertEqual(source["fall20"], .038)
        self.assertEqual(source["fall40"], .015)
        self.assertEqual(source["rise20"], .041)
        self.assertEqual(source["rise40"], .00004)
        self.assertEqual(source["within20"], .921)
        self.assertAlmostEqual(source["fall20"] + source["rise20"] + source["within20"], 1)
        self.assertLessEqual(source["fall40"], source["fall20"])
        self.assertLessEqual(source["rise40"], source["rise20"])
        for now in (base - 7200, base, base + 125, base + 86400 * 366):
            with self.subTest(now=now):
                payload = demo.build_payload(now, self.fixture)
                forecast = payload["forecast"]
                near = forecast["near90"]
                arrival = near["arrival_change"]
                self.assertIs(arrival["available"], True)
                self.assertEqual(arrival["issued_epoch"], forecast["issued_epoch"])
                self.assertEqual(arrival["arrival_epoch"], near["target_epoch"])
                self.assertEqual(arrival["arrival_epoch"] - arrival["issued_epoch"], 5400)
                self.assertGreater(arrival["arrival_epoch"], now)
                reference_age = arrival["issued_epoch"] - arrival["fresh_reference_epoch"]
                self.assertEqual(reference_age, 15)
                self.assertLessEqual(reference_age, 240)
                self.assertEqual(arrival["reference_ugm3"], near["reference_pm25_ugm3"])
                self.assertEqual({key: value for key, value in arrival.items()
                                  if not key.endswith("_epoch")}, native_values)
                delta = now - base
                for key in ("issued_epoch", "arrival_epoch", "fresh_reference_epoch"):
                    self.assertEqual(arrival[key], source[key] + delta)
                self.assertEqual(near["first20"], original["forecast"]["near90"]["first20"])
                self.assertEqual(len(payload["history"]["points"]),
                                 len(original["history"]["points"]))
        self.assertEqual(self.fixture, original)

    def test_arrival_is_independent_of_crossing_and_numeric_point(self):
        fixture = copy.deepcopy(self.fixture)
        near = fixture["forecast"]["near90"]
        native = copy.deepcopy(near["arrival_change"])
        near["available"] = False
        near["pm25_ugm3"] = None
        near["first20"] = {"available": False, "display_text": "Crossing unavailable"}
        payload = demo.build_payload(fixture["generated_epoch"], fixture)
        arrival = payload["forecast"]["near90"]["arrival_change"]
        self.assertEqual(arrival, native)
        self.assertTrue(arrival["available"])
        self.assertIsNone(payload["forecast"]["near90"]["pm25_ugm3"])
        self.assertFalse(payload["forecast"]["near90"]["first20"]["available"])

    def test_hourly_rain_intervals_contain_now_and_next_hour(self):
        now = local_epoch(2026, 10, 8, 12, 37)
        payload = demo.build_payload(now, self.fixture)
        weather = payload["weather"]
        hours = [weather["current_hour"], *weather["next_hours"]]
        for hour in hours:
            self.assertEqual(hour["valid_epoch"], hour["rain_end_epoch"])
            self.assertEqual(hour["rain_end_epoch"] - hour["rain_start_epoch"], 3600)
            self.assertEqual(hour["valid_epoch"] % 3600, 0)
        first, second = weather["next_hours"][:2]
        self.assertLessEqual(first["rain_start_epoch"], now)
        self.assertGreater(first["rain_end_epoch"], now)
        self.assertLessEqual(second["rain_start_epoch"], now + 3600)
        self.assertGreater(second["rain_end_epoch"], now + 3600)

    def test_sessions_and_tennis_follow_malaysia_calendar(self):
        for hour, minute, morning_day, afternoon_day, tennis_day, active in (
            (6, 59, 8, 8, 8, False), (7, 0, 8, 8, 8, True),
            (8, 30, 8, 8, 8, True), (9, 0, 9, 8, 9, False),
            (13, 59, 9, 8, 9, False), (14, 0, 9, 9, 9, False),
            (23, 59, 9, 9, 9, False),
        ):
            with self.subTest(hour=hour, minute=minute):
                now = local_epoch(2026, 10, 8, hour, minute)
                payload = demo.build_payload(now, self.fixture)
                for name, start_hour, day in (("morning", 9, morning_day),
                                               ("afternoon", 14, afternoon_day)):
                    session = payload["forecast"]["sessions"][name]
                    self.assertEqual(session["start_epoch"], local_epoch(2026, 10, day, start_hour))
                tennis = payload["tennis_morning"]
                self.assertEqual(tennis["start_epoch"], local_epoch(2026, 10, tennis_day, 7))
                self.assertEqual(tennis["end_epoch"], local_epoch(2026, 10, tennis_day, 9))
                self.assertEqual(tennis["active"], active)

    def test_history_points_summary_and_gaps_shift_together(self):
        fixture = copy.deepcopy(self.fixture)
        fixture["history"]["gaps"] = [[fixture["generated_epoch"] - 1200,
                                           fixture["generated_epoch"] - 600]]
        delta = 3700
        payload = demo.build_payload(fixture["generated_epoch"] + delta, fixture)
        history = payload["history"]
        for new, old in zip(history["points"], fixture["history"]["points"]):
            self.assertEqual(new[0], old[0] + delta)
            self.assertEqual(new[1:], old[1:])
        self.assertEqual(history["gaps"], [[stamp + delta for stamp in fixture["history"]["gaps"][0]]])
        self.assertTrue(all(history["start_epoch"] <= p[0] <= history["end_epoch"]
                            for p in history["points"]))
        values = [p[1] for p in history["points"]]
        summary = history["pm25_summary"]
        self.assertAlmostEqual(summary["average_ugm3"], sum(values) / len(values), places=4)
        self.assertEqual(summary["lowest_ugm3"], min(values))
        self.assertEqual(summary["highest_ugm3"], max(values))
        self.assertEqual(summary["sample_count"], len(values))

    def test_invalid_clocks_nonfinite_values_and_oversized_body_fail(self):
        for invalid in (True, 0, 1699999999, 0xFFFFFFFF, 1735718400.5, "1735718400"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                demo.build_payload(invalid, self.fixture)
        with self.assertRaises(ValueError):
            demo.encode_payload({"number": float("nan")})
        with self.assertRaises(ValueError):
            demo.encode_payload({"too_large": "x" * demo.MAX_BODY_BYTES})


class ServerTests(unittest.TestCase):
    def test_loopback_http_length_clock_refresh_and_unknown_path(self):
        clock_value = [local_epoch(2026, 10, 8, 12, 37)]
        server = demo.ThreadingHTTPServer(("127.0.0.1", 0),
                                         demo.make_handler(demo.load_fixture(), lambda: clock_value[0]))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        opener = build_opener(ProxyHandler({}))  # Keep the test entirely on loopback.
        url = f"http://127.0.0.1:{server.server_port}"
        try:
            for offset in (0, 125):
                clock_value[0] += offset
                with opener.open(url + demo.API_PATH + "?preview=1", timeout=3) as response:
                    body = response.read()
                    self.assertEqual(response.status, 200)
                    self.assertEqual(int(response.headers["Content-Length"]), len(body))
                    self.assertLess(len(body), demo.MAX_BODY_BYTES)
                    self.assertEqual(response.headers["Cache-Control"], "no-store")
                    self.assertEqual(response.headers["X-RLCD-Synthetic-Demo"], "true")
                    self.assertEqual(response.headers.get_content_charset(), "utf-8")
                    payload = json.loads(body)
                    self.assertEqual(payload["generated_epoch"], clock_value[0])
                    self.assertTrue(payload["demo"])
            with self.assertRaises(HTTPError) as caught:
                opener.open(url + "/api/not-real", timeout=3)
            with caught.exception as response:
                body = response.read()
                self.assertEqual(response.code, 404)
                self.assertEqual(int(response.headers["Content-Length"]), len(body))
        finally:
            server.shutdown()
            thread.join(timeout=3)
            server.server_close()
        self.assertFalse(thread.is_alive())

    def test_cli_is_loopback_by_default_and_lan_is_explicit(self):
        defaults = demo.parse_args([])
        self.assertEqual(defaults.host, "127.0.0.1")
        self.assertEqual(defaults.port, 8766)
        self.assertIsNone(defaults.advertise_ip)
        lan = demo.parse_args(["--host", "0.0.0.0", "--advertise-ip", "192.168.50.21"])
        self.assertEqual(lan.host, "0.0.0.0")
        self.assertEqual(lan.advertise_ip, "192.168.50.21")
        for argv in (["--host", "0.0.0.0"], ["--advertise-ip", "192.168.50.21"],
                     ["--port", "0"], ["--port", "65536"],
                     ["--host", "localhost"],
                     *(["--host", "0.0.0.0", "--advertise-ip", ip] for ip in
                       ("127.0.0.1", "0.0.0.0", "8.8.8.8", "224.0.0.1", "198.51.100.1", "::1", "hostname"))):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    demo.parse_args(argv)
                self.assertEqual(caught.exception.code, 2)

    def test_optional_mdns_has_matching_type_address_and_txt_without_network(self):
        calls = {}

        class FakeServiceInfo:
            def __init__(self, service_type, name, **kwargs):
                calls["service"] = (service_type, name, kwargs)

        class FakeZeroconf:
            def __init__(self, **kwargs):
                calls["interfaces"] = kwargs

            def register_service(self, service):
                calls["registered"] = service

            def close(self):
                calls["closed"] = True

        fake = types.SimpleNamespace(ServiceInfo=FakeServiceInfo, Zeroconf=FakeZeroconf,
                                     IPVersion=types.SimpleNamespace(V4Only="v4-only"))
        with patch.dict(sys.modules, {"zeroconf": fake}):
            zeroconf, service = demo.advertise("192.168.50.21", 8766)
        service_type, name, metadata = calls["service"]
        self.assertEqual(service_type, "_ttdi-weather._tcp.local.")
        self.assertTrue(name.endswith(service_type))
        self.assertIn("synthetic demo", name)
        self.assertEqual(metadata["addresses"], [bytes([192, 168, 50, 21])])
        self.assertEqual(metadata["port"], 8766)
        self.assertEqual(metadata["properties"],
                         {"schema_version": "1", "api_path": "/api/rlcd/v1", "demo": "true"})
        self.assertEqual(calls["interfaces"], {"interfaces": ["192.168.50.21"], "ip_version": "v4-only"})
        self.assertIs(calls["registered"], service)
        self.assertIsInstance(zeroconf, FakeZeroconf)
        with patch.dict(sys.modules, {"zeroconf": None}), self.assertRaisesRegex(RuntimeError, "pip install zeroconf"):
            demo.advertise("192.168.50.21", 8766)


if __name__ == "__main__":
    unittest.main()
