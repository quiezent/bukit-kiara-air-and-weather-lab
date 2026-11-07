"""FIRMS hotspot collector: parsing, first-receipt de-duplication, key masking, errors.

HTTP is mocked throughout; no test touches the network, the live archive or the
live key file.
"""

from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))
import calendar
import contextlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from unittest import mock
from urllib.error import HTTPError, URLError

import firms_collector as F

FAKE_KEY = "0123456789abcdef0123456789abcdef"
VIIRS_HEADER = ("latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,"
                "confidence,version,bright_ti5,frp,daynight\n")
MODIS_HEADER = ("latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,instrument,"
                "confidence,version,bright_t31,frp,daynight\n")
PAYLOADS = {
    "VIIRS_SNPP_NRT": VIIRS_HEADER +
        "-0.51234,101.45670,330.52,0.39,0.36,2026-10-20,0612,N,VIIRS,n,2.0NRT,290.11,5.21,D\n"
        "1.10000,103.20000,301.00,0.40,0.37,2026-10-20,1842,N,VIIRS,l,2.0NRT,288.00,1.10,N\n",
    "VIIRS_NOAA20_NRT": VIIRS_HEADER +
        "-2.25000,104.75000,345.10,0.50,0.49,2026-10-20,0521,N20,VIIRS,h,2.0NRT,295.30,22.40,D\n",
    "VIIRS_NOAA21_NRT": VIIRS_HEADER +
        "0.75000,109.10000,333.30,0.45,0.39,2026-10-21,0547,N21,VIIRS,n,2.0NRT,291.90,8.80,D\n",
    "MODIS_NRT": MODIS_HEADER +
        "1.23450,102.34560,318.20,1.10,1.00,2026-10-20,315,Terra,MODIS,67,6.1NRT,295.40,12.30,D\n",
}


def fake_fetcher(payloads=PAYLOADS, calls=None):
    def fetch(url, timeout):
        if calls is not None:
            calls.append((url, timeout))
        source = url.split("/")[7]
        return 200, payloads[source].encode("utf-8")
    return fetch


class Workspace(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.db = root / "external_firms" / "firms.sqlite3"
        self.key_path = root / "private" / "firms.json"
        self.key_path.parent.mkdir()
        self.key_path.write_text(json.dumps({"mapKey": FAKE_KEY, "provider": "NASA FIRMS"}), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def rows(self, sql, *args):
        connection = sqlite3.connect(f"file:{self.db.as_posix()}?mode=ro", uri=True)
        try:
            return connection.execute(sql, args).fetchall()
        finally:
            connection.close()

    def archive_bytes(self):
        return b"".join(p.read_bytes() for p in self.db.parent.glob(self.db.name + "*"))

    def poll(self, fetcher):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            results = F.poll_once(self.db, self.key_path, fetcher, spacing=0)
        return results, out.getvalue()


class ParsingTests(unittest.TestCase):
    def test_parses_every_source_with_its_own_columns(self):
        expected = {"VIIRS_SNPP_NRT": ("N", 2), "VIIRS_NOAA20_NRT": ("N20", 1),
                    "VIIRS_NOAA21_NRT": ("N21", 1), "MODIS_NRT": ("Terra", 1)}
        for source, (satellite, count) in expected.items():
            rows, bad = F.parse_csv(source, PAYLOADS[source].encode("utf-8"))
            self.assertEqual((len(rows), bad), (count, 0), source)
            first = rows[0]
            self.assertEqual((first["source"], first["satellite"]), (source, satellite))
            if source == "MODIS_NRT":
                self.assertEqual((first["brightness"], first["bright_t31"]), (318.2, 295.4))
                self.assertIsNone(first["bright_ti4"])
                self.assertEqual((first["instrument"], first["confidence"], first["version"]),
                                 ("MODIS", "67", "6.1NRT"))
                self.assertEqual(first["acq_time"], "0315")      # leading zero restored
                self.assertEqual(first["acq_epoch"], calendar.timegm((2026, 10, 20, 3, 15, 0)))
            else:
                self.assertIsNotNone(first["bright_ti4"])
                self.assertIsNotNone(first["bright_ti5"])
                self.assertIsNone(first["brightness"])
                self.assertEqual((first["instrument"], first["version"]), ("VIIRS", "2.0NRT"))
            for field in ("latitude", "longitude", "scan", "track", "frp"):
                self.assertIsInstance(first[field], float)
            self.assertIn(first["daynight"], ("D", "N"))
        snpp = F.parse_csv("VIIRS_SNPP_NRT", PAYLOADS["VIIRS_SNPP_NRT"].encode())[0][0]
        self.assertEqual((snpp["latitude"], snpp["longitude"], snpp["bright_ti4"], snpp["bright_ti5"],
                          snpp["frp"], snpp["confidence"], snpp["acq_date"], snpp["acq_time"]),
                         (-0.51234, 101.4567, 330.52, 290.11, 5.21, "n", "2026-10-20", "0612"))
        self.assertEqual(snpp["acq_epoch"], calendar.timegm((2026, 10, 20, 6, 12, 0)))   # UTC

    def test_header_only_is_zero_rows_and_text_replies_are_rejected(self):
        self.assertEqual(F.parse_csv("VIIRS_NOAA21_NRT", VIIRS_HEADER.encode()), ([], 0))
        with self.assertRaises(ValueError):
            F.parse_csv("VIIRS_SNPP_NRT", b"Invalid MAP_KEY.")
        with self.assertRaises(ValueError):
            F.parse_csv("MODIS_NRT", VIIRS_HEADER.encode())   # VIIRS columns are not a MODIS reply
        with self.assertRaises(ValueError):
            F.parse_csv("VIIRS_SNPP_NRT", b"")

    def test_bad_rows_are_counted_and_skipped(self):
        body = (VIIRS_HEADER + "x,101,330,0.4,0.4,2026-10-20,0612,N,VIIRS,n,2.0NRT,290,5,D\n"
                "1,101,330,0.4,0.4,2026-10-20,2512,N,VIIRS,n,2.0NRT,290,5,D\n"
                "1,101,330,0.4,0.4,2026-10-20,0612,N,VIIRS,n,2.0NRT,290,5,D\n").encode()
        rows, bad = F.parse_csv("VIIRS_SNPP_NRT", body)
        self.assertEqual((len(rows), bad), (1, 2))

    def test_request_url_and_masking(self):
        url = F.request_url(FAKE_KEY, "MODIS_NRT")
        self.assertEqual(url, F.API_BASE + "/" + FAKE_KEY + "/MODIS_NRT/94,-8,120,8/2")
        self.assertEqual(F.mask(url, FAKE_KEY), F.API_BASE + "/***/MODIS_NRT/94,-8,120,8/2")
        self.assertEqual(F.mask("failed " + url), "failed " + F.API_BASE + "/***/MODIS_NRT/94,-8,120,8/2")
        self.assertNotIn(FAKE_KEY, F.mask(f"echo {FAKE_KEY} here", FAKE_KEY))
        with self.assertRaises(ValueError):
            F.request_url(FAKE_KEY, "LANDSAT_NRT")
        with self.assertRaises(ValueError):
            F.request_url(FAKE_KEY, "MODIS_NRT", day_range=6)


class StorageTests(Workspace):
    def test_poll_stores_requests_detections_and_schema_in_wal_mode(self):
        calls = []
        results, out = self.poll(fake_fetcher(calls=calls))
        self.assertEqual([r["source"] for r in results], list(F.SOURCES))
        self.assertTrue(all(r["ok"] and r["error"] is None for r in results))
        self.assertEqual([c[1] for c in calls], [30] * 4)
        self.assertEqual(self.rows("SELECT value FROM meta WHERE key='schema_version'"), [(F.VERSION,)])
        self.assertEqual(self.rows("PRAGMA journal_mode"), [("wal",)])
        requests = self.rows("SELECT source,bbox,day_range,http_status,row_count,new_rows,bytes,payload_sha256,"
                             "error,url_masked,received_epoch>=requested_epoch FROM requests ORDER BY id")
        self.assertEqual(len(requests), 4)
        for row, source in zip(requests, F.SOURCES):
            self.assertEqual(row[:4], (source, "94,-8,120,8", 2, 200))
            self.assertEqual(row[4], row[5])
            self.assertEqual(row[6], len(PAYLOADS[source].encode()))
            self.assertEqual(len(row[7]), 64)
            self.assertIsNone(row[8])
            self.assertEqual(row[9], f"{F.API_BASE}/***/{source}/94,-8,120,8/2")
            self.assertEqual(row[10], 1)
        self.assertEqual(self.rows("SELECT COUNT(*) FROM detections"), [(5,)])
        self.assertEqual(out, "")

    def test_duplicates_keep_the_first_receipt_and_first_values(self):
        clock = iter([1000.0, 1001.0] + [1002.0 + i for i in range(40)])
        with mock.patch.object(F.time, "time", lambda: next(clock)):
            first, _ = self.poll(fake_fetcher())
        first_snpp = first[0]["requestId"]
        received = self.rows("SELECT received_epoch FROM requests WHERE id=?", first_snpp)[0][0]
        revised = dict(PAYLOADS)
        revised["VIIRS_SNPP_NRT"] = (PAYLOADS["VIIRS_SNPP_NRT"].replace(",5.21,D", ",9.99,D")   # revised FRP
                                     + "2.00000,100.50000,310.00,0.40,0.37,2026-10-21,0606,N,VIIRS,n,"
                                       "2.0NRT,289.00,3.30,D\n")
        second, _ = self.poll(fake_fetcher(revised))
        self.assertEqual(second[0]["rows"], 3)
        self.assertEqual(second[0]["newRows"], 1)
        self.assertEqual([r["newRows"] for r in second[1:]], [0, 0, 0])
        kept = self.rows("SELECT frp,first_request_id,first_received_epoch FROM detections "
                         "WHERE source='VIIRS_SNPP_NRT' AND acq_time='0612'")
        self.assertEqual(kept, [(5.21, first_snpp, received)])
        newest = self.rows("SELECT first_request_id FROM detections WHERE acq_time='0606'")
        self.assertEqual(newest, [(second[0]["requestId"],)])
        self.assertEqual(self.rows("SELECT COUNT(*) FROM detections"), [(6,)])
        self.assertEqual(self.rows("SELECT COUNT(*) FROM requests"), [(8,)])

    def test_concurrent_writers_use_their_own_connections(self):
        errors, results = [], []

        def work(source):
            try:
                for _ in range(5):
                    results.append(F.poll_source(self.db, FAKE_KEY, source, fake_fetcher()))
            except Exception as error:            # pragma: no cover - reported below
                errors.append(error)
        threads = [threading.Thread(target=work, args=(s,)) for s in F.SOURCES for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 40)
        self.assertEqual(self.rows("SELECT COUNT(*) FROM requests"), [(40,)])
        self.assertEqual(self.rows("SELECT COUNT(*) FROM detections"), [(5,)])
        self.assertEqual(sum(r["newRows"] for r in results), 5)


class KeyTests(Workspace):
    def test_key_is_masked_in_requests_errors_stdout_and_absent_from_the_file(self):
        def leaky(url, timeout):
            source = url.split("/")[7]
            if source == "VIIRS_SNPP_NRT":
                raise URLError(f"connect failed for {url}")
            if source == "VIIRS_NOAA20_NRT":
                raise HTTPError(url, 403, "Forbidden", {}, io.BytesIO(f"Invalid MAP_KEY {FAKE_KEY}".encode()))
            if source == "VIIRS_NOAA21_NRT":
                return 200, f"Invalid MAP_KEY: {FAKE_KEY}".encode()
            raise ValueError(f"odd {FAKE_KEY} at {url}")
        results, out = self.poll(leaky)
        self.assertFalse(any(r["ok"] for r in results))
        self.assertEqual([r["httpStatus"] for r in results], [None, 403, 200, None])
        errors = [r[0] for r in self.rows("SELECT error FROM requests ORDER BY id")]
        self.assertEqual(len(errors), 4)
        for error in errors:
            self.assertIn("***", error)
            self.assertNotIn(FAKE_KEY, error)
        self.assertTrue(errors[0].startswith("URLError"))
        self.assertTrue(errors[1].startswith("HTTPError 403"))
        self.assertIn("unexpected payload", errors[2])
        self.assertEqual(out.count("[FIRMS collector]"), 4)
        self.assertIn("***", out)
        self.assertNotIn(FAKE_KEY, out)
        for (url,) in self.rows("SELECT url_masked FROM requests"):
            self.assertIn("/csv/***/", url)
        self.assertNotIn(FAKE_KEY.encode(), self.archive_bytes())

    def test_no_key_fragment_survives_when_the_key_straddles_a_snippet_cut(self):
        # Review 2026-10-21: snippets were cut before masking, leaving a key prefix behind.
        def straddling(url, timeout):
            source = url.split("/")[7]
            pad = {"VIIRS_SNPP_NRT": 150, "VIIRS_NOAA20_NRT": 190, "VIIRS_NOAA21_NRT": 390, "MODIS_NRT": 490}[source]
            body = ("x" * pad + FAKE_KEY + " tail").encode()
            if source in ("VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT"):
                raise HTTPError(url, 400, "Bad Request", {}, io.BytesIO(body))
            return 200, body
        results, out = self.poll(straddling)
        self.assertFalse(any(r["ok"] for r in results))
        fragments = {FAKE_KEY[i:i + 8] for i in range(len(FAKE_KEY) - 7)}
        stored = " ".join(r[0] for r in self.rows("SELECT error FROM requests"))
        for text in (stored, out, self.archive_bytes().decode("latin-1")):
            self.assertFalse(any(f in text for f in fragments))

    def test_clean_run_leaves_no_key_in_the_file_or_output(self):
        out = io.StringIO()
        stop = threading.Event()
        waits = []

        def wait(event, seconds):
            waits.append(seconds)
            return True
        with mock.patch.object(F, "_wait", wait), contextlib.redirect_stdout(out):
            F.collector(self.db, self.key_path, stop, fake_fetcher(), poll_seconds=1800, spacing=0)
        self.assertEqual(waits, [1800])
        self.assertIn("4/4 sources ok, rows 5, new 5", out.getvalue())
        self.assertNotIn(FAKE_KEY, out.getvalue())
        self.assertNotIn(FAKE_KEY.encode(), self.archive_bytes())

    def test_missing_or_invalid_key_logs_one_line_and_requests_nothing(self):
        self.key_path.unlink()
        calls = []
        with self.assertRaises(F.KeyUnavailable):
            F.poll_once(self.db, self.key_path, fake_fetcher(calls=calls), spacing=0)
        waits = []

        def wait(event, seconds):
            waits.append(seconds)
            return len(waits) >= 3
        out = io.StringIO()
        with mock.patch.object(F, "_wait", wait), contextlib.redirect_stdout(out):
            F.collector(self.db, self.key_path, None, fake_fetcher(calls=calls), poll_seconds=1800, spacing=0)
        self.assertEqual(calls, [])
        self.assertEqual(waits, [1800, 1800, 1800])
        self.assertEqual(out.getvalue().count("\n"), 1)
        self.assertIn("firms.json is missing", out.getvalue())
        self.assertFalse(self.db.exists())
        for content in ("{not json", json.dumps({"mapKey": "short"}), json.dumps(["x"]),
                        json.dumps({"mapKey": FAKE_KEY + "/../x"}), json.dumps({"note": FAKE_KEY})):
            self.key_path.write_text(content, encoding="utf-8")
            with self.assertRaises(F.KeyUnavailable) as caught:
                F.read_key(self.key_path)
            self.assertNotIn(FAKE_KEY, str(caught.exception))
            self.assertNotIn("short", str(caught.exception))
        self.key_path.write_text(json.dumps({"mapKey": FAKE_KEY}), encoding="utf-8")
        self.assertEqual(F.read_key(self.key_path), FAKE_KEY)


class HttpErrorTests(Workspace):
    def test_http_and_network_errors_are_recorded_without_rows(self):
        failures = {"VIIRS_SNPP_NRT": HTTPError("u", 500, "Server Error", {}, io.BytesIO(b"busy")),
                    "VIIRS_NOAA20_NRT": HTTPError("u", 429, "Too Many", {}, None),
                    "VIIRS_NOAA21_NRT": TimeoutError("timed out"),
                    "MODIS_NRT": None}

        def fetch(url, timeout):
            source = url.split("/")[7]
            if failures[source] is None:
                return 503, b"maintenance"
            raise failures[source]
        results, out = self.poll(fetch)
        self.assertEqual([r["ok"] for r in results], [False] * 4)
        stored = self.rows("SELECT http_status,row_count,new_rows,received_epoch IS NULL FROM requests ORDER BY id")
        self.assertEqual(stored, [(500, None, None, 0), (429, None, None, 0), (None, None, None, 1),
                                  (503, None, None, 0)])
        errors = [r[0] for r in self.rows("SELECT error FROM requests ORDER BY id")]
        self.assertIn("HTTPError 500: busy", errors[0])
        self.assertIn("HTTPError 429", errors[1])
        self.assertIn("TimeoutError", errors[2])
        self.assertIn("HTTP status 503", errors[3])
        self.assertEqual(self.rows("SELECT COUNT(*) FROM detections"), [(0,)])
        self.assertEqual(out.count("\n"), 4)

    def test_failing_cycles_back_off_and_recover(self):
        state = {"fail": True}

        def fetch(url, timeout):
            if state["fail"]:
                raise URLError("offline")
            return fake_fetcher()(url, timeout)
        waits = []

        def wait(event, seconds):
            waits.append(seconds)
            if len(waits) == 5:
                state["fail"] = False
            return len(waits) >= 6
        with mock.patch.object(F, "_wait", wait), contextlib.redirect_stdout(io.StringIO()):
            F.collector(self.db, self.key_path, None, fetch, poll_seconds=100, spacing=0)
        self.assertEqual(waits, [100, 200, 400, 400, 400, 100])

    def test_unexpected_exceptions_never_end_the_thread(self):
        waits = []

        def wait(event, seconds):
            waits.append(seconds)
            return len(waits) >= 2
        out = io.StringIO()
        with mock.patch.object(F, "store", side_effect=sqlite3.OperationalError(f"disk I/O {FAKE_KEY}")), \
                mock.patch.object(F, "_wait", wait), contextlib.redirect_stdout(out):
            F.collector(self.db, self.key_path, None, fake_fetcher(), poll_seconds=100, spacing=0)
        self.assertEqual(waits, [100, 200])
        self.assertIn("OperationalError", out.getvalue())
        self.assertNotIn(FAKE_KEY, out.getvalue())

    def test_wait_honours_the_stop_event(self):
        stop = threading.Event()
        stop.set()
        self.assertTrue(F._wait(stop, 3600))
        self.assertFalse(F._wait(None, 0))


class DashboardWiringTests(unittest.TestCase):
    def test_dashboard_starts_one_named_daemon_thread(self):
        source = (Path(F.__file__).resolve().parent / "BukitKiara_Dashboard.py").read_text(encoding="utf-8")
        self.assertEqual(source.count("import firms_collector\n"), 1)
        self.assertEqual(source.count("target=firms_collector.collector"), 1)
        self.assertIn('target=firms_collector.collector, daemon=True, name="firms-collector"', source)

    def test_archive_lives_apart_from_the_main_db(self):
        self.assertEqual(F.ARCHIVE_PATH.parts[-3:], ("research", "external_firms", "firms.sqlite3"))
        self.assertEqual(F.KEY_PATH.parts[-2:], ("private", "firms.json"))
        self.assertEqual(F.BBOX, (94, -8, 120, 8))
        self.assertEqual((F.DAY_RANGE, F.POLL_SECONDS, F.TIMEOUT_SECONDS), (2, 1800, 30))
        self.assertEqual(F.SOURCES, ("VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "MODIS_NRT"))


if __name__ == "__main__":
    unittest.main()
