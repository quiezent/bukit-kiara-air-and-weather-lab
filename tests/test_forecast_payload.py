
from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))
import hashlib
import json
import sqlite3
import unittest

import forecast_archive
from forecast_payload import decode_payload_text, encode_payload, loads


class PayloadTests(unittest.TestCase):
    def test_original_utf8_bytes_and_hash_are_preserved(self):
        text = '{"units":"µg/m³", "value":62.50}\n'
        packed = encode_payload(text)
        self.assertEqual(decode_payload_text(packed), text)
        self.assertEqual(hashlib.sha256(decode_payload_text(packed).encode()).digest(),
                         hashlib.sha256(text.encode()).digest())
        self.assertEqual(loads(text), loads(packed))
        self.assertEqual(packed, encode_payload(text))

    def test_old_text_and_plain_bytes_remain_readable(self):
        for value in ('{"old":1}', b'{"old":1}', memoryview(b'{"old":1}')):
            self.assertEqual(loads(value), {"old": 1})

    def test_truncated_payload_is_rejected(self):
        with self.assertRaises((OSError, EOFError)):
            loads(encode_payload('{"x":1}')[:-5])

    def test_writer_uses_compact_gzip_with_correct_identity_and_clocks(self):
        source = {"available": True, "forecastIssuedEpoch": 123456,
                  "current": {"epoch": 123400, "pm02": 62.5},
                  "airWindow": {"arrival": {"point": 62.5, "baselinePoint": 62.5},
                                "nearTermSelectionDiagnostics": {"largeReport": "x" * 20000}},
                  "windows": {}}
        with sqlite3.connect(":memory:") as conn:
            self.assertTrue(forecast_archive.record_dashboard_issue(conn, source, "test", 123460))
            self.assertFalse(forecast_archive.record_dashboard_issue(conn, source, "test", 123460))
            row = conn.execute("SELECT * FROM dashboard_forecast_issues").fetchone()
            self.assertEqual(row[1:5], (123460, 123456, 123400, "test"))
            self.assertIsInstance(row[5], bytes)
            text = decode_payload_text(row[5])
            self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), row[0])
            self.assertEqual(json.loads(text)["airWindow"]["arrival"]["point"], 62.5)
            self.assertNotIn("nearTermSelectionDiagnostics", json.loads(text)["airWindow"])


if __name__ == "__main__":
    unittest.main()
