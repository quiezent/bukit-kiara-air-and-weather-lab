"""Temporary SQLite fixtures; no network, fitting or production writes."""

from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import fresh_numeric_model_v3 as old
import fresh_numeric_model_v4 as model
import observation_provenance as receipts


class ReceiptTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "fixture.db"
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("CREATE TABLE readings(epoch INTEGER PRIMARY KEY,timestamp TEXT,pm02 REAL,atmp REAL,rhum REAL)")
        receipts.init_schema(self.conn)
        self.conn.commit()
        local = timezone(timedelta(hours=8))
        self.issue = int(datetime(2026, 10, 7, 12, tzinfo=local).timestamp())
        self.cutoff = int(datetime(2026, 10, 8, 0, tzinfo=local).timestamp())
        for epoch in range(self.issue - 4 * 3600, self.issue + 8 * 3600 + 1, 180):
            reading = self.reading(epoch, 20 if epoch <= self.issue else 30)
            self.conn.execute("INSERT INTO readings VALUES(?,?,?,?,?)", tuple(reading.values()))
        self.conn.commit()

    def tearDown(self):
        self.conn.close()
        self.temp.cleanup()

    def reading(self, epoch, pm):
        return {"epoch": epoch, "timestamp": datetime.fromtimestamp(epoch, timezone.utc).isoformat(),
                "pm02": float(pm), "atmp": 30.0, "rhum": 55.0}

    def record(self, epoch, pm, received, *, confirm=True):
        result = receipts.record_sensor_receipt(self.conn, self.reading(epoch, pm),
                                                received, clock=lambda: received + .1)
        self.conn.commit()
        if confirm:
            receipts.confirm_receipts_visible(self.conn, clock=lambda: received + .2)
            self.conn.commit()
        return result

    def correction_history(self):
        # Every sample in this completed 15-minute proxy is corrected later.
        # An earlier forecast must retain 20; the later completed label may use 200.
        for epoch in range(self.issue - 720, self.issue + 1, 180):
            self.record(epoch, 20, epoch + .1)
            self.record(epoch, 200, self.issue + 600)
            self.conn.execute("UPDATE readings SET pm02=200 WHERE epoch=?", (epoch,))
            self.conn.commit()

    def raw_v3(self):
        return [dict(r) for r in self.conn.execute("SELECT epoch,pm02,atmp,rhum FROM readings ORDER BY epoch")]

    def causal(self):
        return model._training_rows(self.conn, self.issue - 4 * 3600, self.cutoff)

    def test_later_overwrite_does_not_replace_original_issue_feature(self):
        self.correction_history()
        legacy_features, legacy_targets, _ = old.training_data(self.raw_v3(), self.cutoff, issue_epochs=[self.issue])
        fixed_features, fixed_targets, _ = model.training_data(self.causal(), self.cutoff, issue_epochs=[self.issue])
        self.assertEqual(legacy_features.loc[self.issue, "freshPm25"], 200)
        self.assertEqual(fixed_features.loc[self.issue, "freshPm25"], 20)
        self.assertEqual(fixed_targets.loc[self.issue, "arrivalPoint"], 30)
        np.testing.assert_array_equal(legacy_targets.to_numpy(), fixed_targets.to_numpy())

    def test_completed_label_uses_latest_available_revision_at_cutoff(self):
        self.correction_history()
        earlier = self.issue - 5400
        features, targets, _ = model.training_data(self.causal(), self.cutoff, issue_epochs=[earlier])
        self.assertEqual(features.loc[earlier, "freshPm25"], 20)
        self.assertEqual(targets.loc[earlier, "arrivalPoint"], 200)

    def test_unconfirmed_and_post_cutoff_values_do_not_fall_back_to_overwrite(self):
        observed = self.issue
        self.record(observed, 20, observed + .1)
        self.record(observed, 900, self.cutoff + 10)
        self.conn.execute("UPDATE readings SET pm02=900 WHERE epoch=?", (observed,))
        unconfirmed = observed - 180
        self.record(unconfirmed, 777, observed + 10, confirm=False)
        self.conn.execute("UPDATE readings SET pm02=777 WHERE epoch=?", (unconfirmed,))
        self.conn.commit()
        selected = self.causal()
        values = [r["pm02"] for r in selected if r["epoch"] == observed]
        self.assertEqual(values, [20])
        self.assertFalse(any(r["epoch"] == unconfirmed for r in selected))
        features, _, _ = model.training_data(selected, self.cutoff, issue_epochs=[self.issue + 1])
        self.assertEqual(features.loc[self.issue + 1, "freshPm25"], 20)

    def test_untracked_legacy_values_remain_explicitly_unknown(self):
        rows = self.causal()
        self.assertTrue(rows)
        self.assertTrue(all(r["_provenance"]["receiptKnown"] is False for r in rows))
        features, _, _ = model.training_data(rows, self.cutoff, issue_epochs=[self.issue])
        self.assertGreater(features.attrs["featureProvenance"][0]["unknownReceiptRows"], 0)

    def test_every_incomplete_receipt_schema_fails_closed(self):
        ledger = ("sensor_observation_receipts", "sensor_observation_revisions", "sensor_observation_visibility")
        for size in (1, 2):
            for subset in combinations(ledger, size):
                with self.subTest(tables=subset):
                    connection = sqlite3.connect(":memory:")
                    connection.row_factory = sqlite3.Row
                    try:
                        connection.execute("CREATE TABLE readings(epoch INTEGER,pm02 REAL,atmp REAL,rhum REAL)")
                        connection.execute("INSERT INTO readings VALUES(?,?,?,?)", (self.issue, 900, 30, 55))
                        for table in subset:
                            connection.execute("CREATE TABLE " + table + "(fixture_marker INTEGER)")
                        with self.assertRaisesRegex(ValueError, "incomplete_sensor_receipt_schema"):
                            model._training_rows(connection, self.issue - 1000, self.cutoff)
                    finally:
                        connection.close()

    def test_refresh_reads_revisions_without_writing_database_or_asset(self):
        self.correction_history()
        captured = {}
        def fitting_stub(rows, cutoff):
            captured.update(rows=rows, cutoff=cutoff)
            return {"metadata": {"trainingCutoffEpoch": cutoff}}
        with patch.object(model, "load_artifact", return_value=None), patch.object(model, "fit", side_effect=fitting_stub), patch.object(model, "save_artifact", return_value=Path("fixture-only.pkl")) as save:
            result = model.refresh_model(self.path, issue_epoch=self.cutoff + 3600)
        self.assertTrue(result["refreshed"])
        self.assertEqual(captured["cutoff"], self.cutoff)
        self.assertEqual({r["pm02"] for r in captured["rows"] if r["epoch"] == self.issue}, {20, 200})
        save.assert_called_once()
        self.assertFalse((Path(self.temp.name) / "fixture-only.pkl").exists())

    def test_metadata_counts_retained_inputs_and_only_valid_training_origins(self):
        self.correction_history()
        captured = {}
        class MatrixSink:
            def fit(self, matrix, responses):
                captured.update(matrix=matrix.copy(), responses=responses.copy())
                return self
        rows = self.causal()
        future = self.reading(self.cutoff + 180, 999)
        future["_provenance"] = {"receiptKnown": False, "availableEpoch": None}
        late = self.reading(self.issue, 888)
        late["_provenance"] = {"receiptKnown": True, "availableEpoch": self.cutoff + 60,
                               "confirmedEpoch": self.cutoff + 60}
        # The first clock precedes the entire fixture and must be dropped;
        # provenance counts must not include it or exceed retained fit origins.
        origins = [self.issue - 4 * 3600 - 60, self.issue]
        with patch.object(model, "MINIMUM_TRAINING_ORIGINS", 1), patch.object(model, "_make_estimator", return_value=MatrixSink()):
            artifact = model.fit(rows + [future, late], self.cutoff, issue_epochs=origins)
        metadata = artifact["metadata"]
        self.assertEqual(metadata["trainingOriginCount"], 1)
        self.assertEqual(metadata["trainingOriginsWithUnknownReceipts"], 1)
        self.assertFalse(metadata["trainingOriginalLiveAvailabilityEstablished"])
        self.assertEqual(metadata["knownReceiptRevisionRows"], 10)
        self.assertEqual(metadata["unknownReceiptRevisionRows"], len(rows) - 10)
        self.assertEqual(metadata["retainedSensorRevisionRows"], len(rows))
        self.assertTrue(metadata["historicalReceiptTimesUnknown"])
        self.assertEqual(captured["matrix"][0, list(model.sensor.FEATURE_COLUMNS).index("freshPm25")], 20)

    def test_recipe_native_outputs_and_cutoff_semantics_remain_explicit(self):
        self.assertEqual(model.MODEL_PARAMETERS, old.MODEL_PARAMETERS)
        self.assertEqual(model.OUTPUT_COLUMNS, old.OUTPUT_COLUMNS)
        self.assertEqual(model.TRAINING_CADENCE_SECONDS, old.TRAINING_CADENCE_SECONDS)
        self.assertEqual(model.EMBARGO_SECONDS, old.EMBARGO_SECONDS)
        self.assertEqual(model.MINIMUM_TRAINING_ORIGINS, old.MINIMUM_TRAINING_ORIGINS)
        identity = model.model_identity()
        self.assertEqual(identity["completedLabelCutoff"], "complete_epoch_at_or_before_training_cutoff")
        self.assertEqual(identity["trainingInputPolicy"], model.TRAINING_INPUT_POLICY)
        self.assertEqual(len(identity["receiptModuleSha256"]), 64)
        self.assertNotEqual(model.MODEL_VERSION, old.MODEL_VERSION)
        self.assertEqual(model.DEFAULT_ARTIFACT_PATH.name, "fresh-numeric-model-v4.pkl")
        native = np.array([[1.25, 2.5, .75, 3.25]])
        class Estimator:
            def predict(self, matrix):
                return native.copy()
        artifact = {"metadata": {"fittedAtEpoch": self.issue - 1,
                    "trainingCutoffEpoch": model.midnight_epoch(self.issue),
                    "latestTrainingOutcomeCompleteEpoch": self.issue - 1}, "estimator": Estimator()}
        result = model.predict(self.raw_v3(), self.issue, artifact=artifact)
        self.assertTrue(result["available"])
        self.assertEqual(list(result["rawOutputUgM3"].values()), native[0].tolist())


if __name__ == "__main__":
    unittest.main()
