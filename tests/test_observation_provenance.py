"""Temporary-SQLite causal receipt/revision and materialization contract tests."""

from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest

import observation_provenance as p


def reading(epoch=1000, pm=3.23456789012345):
    return {"epoch": epoch, "timestamp": datetime.fromtimestamp(epoch, timezone.utc).isoformat(),
            "pm02": pm, "atmp": 28.5, "rhum": 64.0, "pm003Count": None}


class ReceiptRevisionTests(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute("CREATE TABLE readings(epoch INTEGER PRIMARY KEY,timestamp TEXT,pm02 REAL)")
        p.init_schema(self.conn)
        self.conn.commit()

    def tearDown(self):
        self.conn.close()

    def confirm(self, epoch):
        self.conn.commit()
        result = p.confirm_receipts_visible(self.conn, clock=lambda: epoch)
        self.conn.commit()
        return result

    def test_legacy_receipts_remain_unknown_and_migration_is_additive(self):
        r = reading()
        self.conn.execute("INSERT INTO readings VALUES(?,?,?)", (r["epoch"], r["timestamp"], r["pm02"]))
        self.conn.commit()
        p.init_schema(self.conn)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0], 1)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM sensor_observation_receipts").fetchone()[0], 0)
        self.assertEqual(p.load_sensor_asof(self.conn, 2000, 500), [])
        result = p.annotate_sensor_rows(self.conn, [r], 2000)[0]
        self.assertEqual(result["pm02"], r["pm02"])
        self.assertFalse(result["_provenance"]["receiptKnown"])
        self.assertIsNone(result["_provenance"]["receivedEpoch"])

    def test_delayed_archive_insert_is_not_available_at_network_receipt(self):
        r = reading()
        receipt = p.record_sensor_receipt(self.conn, r, 1001.125, clock=lambda: 1200.75)
        self.assertEqual(receipt["receivedEpoch"], 1001.125)
        self.assertEqual(receipt["persistedEpoch"], 1200.75)
        self.assertFalse(receipt["receiptKnown"])
        self.assertEqual(p.load_sensor_asof(self.conn, 1300, 900), [])
        self.confirm(1200.75)
        self.assertEqual(p.load_sensor_asof(self.conn, 1200.74, 900), [])
        rows = p.load_sensor_asof(self.conn, 1200.75, 900)
        self.assertEqual(rows[0]["pm02"], r["pm02"])
        self.assertEqual(rows[0]["_provenance"]["availableEpoch"], 1200.75)

    def test_corrections_preserve_original_and_replay_changes_only_after_receipt(self):
        original = p.record_sensor_receipt(self.conn, reading(pm=10.1), 1001, clock=lambda: 1100)
        self.confirm(1100)
        corrected = p.record_sensor_receipt(self.conn, reading(pm=70.2), 1200, clock=lambda: 1300)
        self.confirm(1300)
        self.assertNotEqual(original["revisionId"], corrected["revisionId"])
        self.assertEqual(p.load_sensor_asof(self.conn, 1299, 900)[0]["pm02"], 10.1)
        self.assertEqual(p.load_sensor_asof(self.conn, 1300, 900)[0]["pm02"], 70.2)
        self.assertEqual(p.load_sensor_asof(self.conn, 2000, 900,
            receipt_watermark=original["receiptId"])[0]["pm02"], 10.1)
        annotated = p.annotate_sensor_rows(self.conn, [reading(pm=70.2)], 1299)[0]
        self.assertFalse(annotated["_provenance"]["receiptKnown"])
        self.assertEqual(annotated["pm02"], 70.2)  # Compatibility value is never silently changed.

    def test_identical_poll_deduplicates_revision_but_preserves_both_receipts(self):
        one = p.record_sensor_receipt(self.conn, reading(), 1001, clock=lambda: 1100)
        two = p.record_sensor_receipt(self.conn, reading(), 1200, clock=lambda: 1300)
        self.assertEqual(one["revisionId"], two["revisionId"])
        self.assertNotEqual(one["receiptId"], two["receiptId"])
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM sensor_observation_revisions").fetchone()[0], 1)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM sensor_observation_receipts").fetchone()[0], 2)

    def test_source_exact_bytes_and_late_same_fetch_revision_are_preserved(self):
        original = '{ "fetchedEpoch":1000, "source":"CAMS", "unit":"µg/m³", "point":10.125 }'
        corrected = '{"fetchedEpoch":1000,"source":"CAMS","unit":"µg/m³","point":30.25}'
        first = p.record_source_receipt(self.conn, "air_quality_forecast_runs", original, 1000, 1010, clock=lambda: 1100)
        self.confirm(1100)
        second = p.record_source_receipt(self.conn, "air_quality_forecast_runs", corrected, 1000, 1200, clock=lambda: 1300)
        self.confirm(1300)
        self.assertEqual(first["payloadSha256"], hashlib.sha256(original.encode("utf-8")).hexdigest())
        self.assertEqual(p.load_source_asof(self.conn, "air_quality_forecast_runs", 1099), [])
        self.assertEqual(p.load_source_asof(self.conn, "air_quality_forecast_runs", 1299)[0]["payload"], original)
        self.assertEqual(p.load_source_asof(self.conn, "air_quality_forecast_runs", 1300)[0]["payload"], corrected)
        self.assertEqual(p.load_source_asof(self.conn, "air_quality_forecast_runs", 2000,
            receipt_watermark=first["receiptId"])[0]["payload"], original)
        self.assertNotEqual(first["revisionId"], second["revisionId"])
        annotated = p.annotate_source_runs(self.conn, "air_quality_forecast_runs",
            [{"fetched_epoch": 1000, "payload": corrected}], 1299)
        self.assertFalse(annotated[0]["_provenance"]["receiptKnown"])
        self.assertEqual(annotated[0]["payload"], corrected)

    def test_caller_rollback_rolls_back_compatibility_and_ledger_atomically(self):
        self.conn.execute("BEGIN IMMEDIATE")
        r = reading()
        self.conn.execute("INSERT INTO readings VALUES(?,?,?)", (r["epoch"], r["timestamp"], r["pm02"]))
        p.record_sensor_receipt(self.conn, r, 1001, clock=lambda: 1100)
        self.conn.rollback()
        for table in ("readings", "sensor_observation_revisions", "sensor_observation_receipts"):
            self.assertEqual(self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0], 0)

    def test_materialized_manifest_links_exact_values_without_row_dump(self):
        p.record_sensor_receipt(self.conn, reading(), 1001, clock=lambda: 1100)
        self.confirm(1100)
        rows = p.load_sensor_asof(self.conn, 2000, 900)
        source = '{"fetchedEpoch":1000,"source":"weather","hourly":[]}'
        p.record_source_receipt(self.conn, "weather_forecast_runs", source, 1000, 1001, clock=lambda: 1100)
        self.confirm(1100)
        sources = p.load_source_asof(self.conn, "weather_forecast_runs", 2000)
        manifest = p.build_input_manifest(rows, sources, 2000, metadata={"bucketPolicy": "no_fill"})
        link = p.record_input_snapshot(self.conn, manifest, clock=lambda: 2001)
        self.assertTrue(link["allReceiptsKnown"])
        self.assertTrue(link["reconstructionComplete"])
        self.assertEqual(link["snapshotId"], hashlib.sha256(p._json(manifest).encode("utf-8")).hexdigest())
        self.assertNotIn('"pm02"', p._json(manifest))
        self.assertNotIn('"hourly"', p._json(manifest))
        detached = [dict(rows[0], pm02=20.5)]
        changed = p.build_input_manifest(detached, sources, 2000)
        self.assertNotEqual(manifest["sensorView"]["valuesSha256"], changed["sensorView"]["valuesSha256"])
        self.assertFalse(changed["allReceiptsKnown"])
        unknown = p.build_input_manifest([reading(epoch=1500)], [], 2000)
        self.assertFalse(unknown["allReceiptsKnown"])
        self.assertFalse(unknown["reconstructionComplete"])
        self.assertEqual(unknown["unknownReceiptInputs"], 1)
        self.assertFalse(p.build_input_manifest([], [], 2000)["allReceiptsKnown"])

    def test_ledger_evidence_cannot_be_updated_or_deleted(self):
        p.record_sensor_receipt(self.conn, reading(), 1001, clock=lambda: 1100)
        source = '{"fetchedEpoch":1000}'
        p.record_source_receipt(self.conn, "weather_forecast_runs", source, 1000, 1001, clock=lambda: 1100)
        p.record_input_snapshot(self.conn, p.build_input_manifest([reading()], [], 2000), clock=lambda: 2001)
        self.confirm(2001)
        for table in ("sensor_observation_revisions", "sensor_observation_receipts",
                      "sensor_observation_visibility", "source_forecast_revisions", "source_forecast_receipts",
                      "source_forecast_visibility", "forecast_input_snapshots"):
            with self.subTest(table=table):
                with self.assertRaisesRegex(sqlite3.IntegrityError, "immutable"):
                    self.conn.execute(f"DELETE FROM {table}")
                self.conn.rollback()
                columns = [r[1] for r in self.conn.execute(f"PRAGMA table_info({table})")]
                with self.assertRaisesRegex(sqlite3.IntegrityError, "immutable"):
                    self.conn.execute(f"UPDATE {table} SET {columns[0]}={columns[0]}")
                self.conn.rollback()

    def test_invalid_clocks_or_values_create_no_evidence(self):
        invalid = [dict(reading(), pm02=float("nan")), dict(reading(), timestamp="invalid")]
        for r in invalid:
            with self.assertRaises(ValueError):
                p.record_sensor_receipt(self.conn, r, 1001, clock=lambda: 1100)
        with self.assertRaises(ValueError):
            p.record_sensor_receipt(self.conn, reading(epoch=2000), 1001, clock=lambda: 1100)
        with self.assertRaises(ValueError):
            p.record_source_receipt(self.conn, "weather_forecast_runs", '{"fetchedEpoch":900}', 1000, 1001)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM sensor_observation_receipts").fetchone()[0], 0)

    def test_writer_clock_is_sampled_only_after_an_existing_writer_releases_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "provenance.sqlite"
            holder = sqlite3.connect(path)
            p.init_schema(holder); holder.commit(); holder.execute("BEGIN IMMEDIATE")
            entered = threading.Event(); sampled = threading.Event(); failures = []

            def write():
                try:
                    conn = sqlite3.connect(path, timeout=2)
                    entered.set()
                    def clock():
                        sampled.set()
                        return 1500.5
                    p.record_sensor_receipt(conn, reading(), 1001, clock=clock)
                    conn.commit(); conn.close()
                except Exception as exc:
                    failures.append(exc)

            thread = threading.Thread(target=write)
            thread.start()
            self.assertTrue(entered.wait(1))
            self.assertFalse(sampled.wait(0.05))
            holder.commit()
            thread.join(2)
            self.assertFalse(thread.is_alive())
            self.assertEqual(failures, [])
            self.assertTrue(sampled.is_set())
            self.assertEqual(holder.execute("SELECT persisted_epoch FROM sensor_observation_receipts").fetchone()[0], 1500.5)
            holder.close()

    def test_visibility_requires_first_commit_and_blocks_pre_commit_issue_clocks(self):
        p.record_sensor_receipt(self.conn, reading(), 1001, clock=lambda: 1100)
        with self.assertRaisesRegex(ValueError, "commit receipt transaction"):
            p.confirm_receipts_visible(self.conn, clock=lambda: 1101)
        self.conn.commit()  # Simulate a transaction held until long after insertion.
        self.assertEqual(p.load_sensor_asof(self.conn, 1500, 900), [])
        result = p.confirm_receipts_visible(self.conn, clock=lambda: 1600)
        self.assertEqual(result['confirmedSensorReceipts'], 1)
        self.conn.commit()
        self.assertEqual(p.load_sensor_asof(self.conn, 1599, 900), [])
        row = p.load_sensor_asof(self.conn, 1600, 900)[0]
        self.assertEqual(row['_provenance']['confirmedEpoch'], 1600)
        self.assertEqual(row['_provenance']['availableEpoch'], 1600)
        self.assertEqual(p.confirm_receipts_visible(self.conn, clock=lambda: 1700)['confirmedSensorReceipts'], 0)


if __name__ == "__main__":
    unittest.main()
