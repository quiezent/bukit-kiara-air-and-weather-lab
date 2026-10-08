
from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))
import math
import unittest

import numpy as np
import pandas as pd

import fresh_sensor_features as features


class FreshSensorFeaturesTests(unittest.TestCase):
    def setUp(self):
        self.issue = 1_791_448_800
        # One sample per minute with a known linear fall, plus paired met data.
        self.rows = [{"epoch": self.issue - minute * 60,
                      "pm02": 100 + minute, "atmp": 30 + minute / 100,
                      "rhum": 50 + minute / 20}
                     for minute in range(181)]

    def test_falling_raw_features_arrive_before_closed_bin(self):
        result = features.issue_features(self.rows, self.issue)
        self.assertEqual(result["freshPm25"], 102)
        self.assertEqual(result["rawDelta5"], -5)
        self.assertEqual(result["rawDelta30"], -30)
        self.assertAlmostEqual(result["rawSlope15PerMinute"], -1)
        self.assertAlmostEqual(result["rawHumidityDelta30"], -1.5)
        self.assertAlmostEqual(result["rawTemperatureDelta30"], -.3)

    def test_future_measurements_cannot_change_features_or_provenance(self):
        future = {"epoch": self.issue + 1, "pm02": 5000, "atmp": 100, "rhum": 100}
        before = features.issue_features(self.rows, self.issue)
        after = features.issue_features(self.rows + [future], self.issue)
        np.testing.assert_equal(list(before.values()), list(after.values()))
        self.assertEqual(features.feature_metadata(self.rows, self.issue),
                         features.feature_metadata(self.rows + [future], self.issue))

    def test_post_issue_receipt_cannot_enter_training_query(self):
        delayed = {"epoch": self.issue, "pm02": 9000, "available_epoch": self.issue + 1}
        original = [r for r in self.rows if r["epoch"] != self.issue]
        before = features.issue_features(original, self.issue)
        after = features.issue_features(original + [delayed], self.issue)
        np.testing.assert_equal(list(before.values()), list(after.values()))
        self.assertEqual(features.feature_metadata(original, self.issue),
                         features.feature_metadata(original + [delayed], self.issue))

    def test_as_of_revision_selects_only_latest_available_copy(self):
        base = {"epoch": self.issue - 120, "pm02": 80, "available_epoch": self.issue - 90}
        corrected = {"epoch": self.issue - 120, "pm02": 20, "available_epoch": self.issue + 30}
        before = features.issue_features([corrected, base], self.issue)
        after = features.issue_features([corrected, base], self.issue + 30)
        self.assertEqual(before["freshPm25"], 80)
        self.assertEqual(after["freshPm25"], 20)

    def test_receipt_provenance_enforces_latest_known_availability(self):
        rows = [{"epoch": self.issue - 60, "pm02": 80,
                 "_provenance": {"receiptKnown": True, "availableEpoch": self.issue - 30,
                                 "confirmedEpoch": self.issue + 1}}]
        self.assertFalse(features.feature_metadata(rows, self.issue)["available"])
        rows[0]["_provenance"]["confirmedEpoch"] = self.issue - 10
        self.assertTrue(features.feature_metadata(rows, self.issue)["available"])

    def test_known_receipt_without_available_time_is_excluded(self):
        rows = [{"epoch": self.issue - 60, "pm02": 80,
                 "_provenance": {"receiptKnown": True}}]
        self.assertFalse(features.feature_metadata(rows, self.issue)["available"])

    def test_legacy_receipt_remains_explicitly_unknown(self):
        metadata = features.feature_metadata(self.rows, self.issue)
        self.assertTrue(metadata["available"])
        self.assertFalse(metadata["allInputReceiptsKnown"])
        self.assertEqual(metadata["unknownReceiptRows"], 180)
        self.assertIn("cannot establish", metadata["legacyReceiptCaveat"])

    def test_legacy_null_receipts_are_unknown_while_unconfirmed_known_rows_are_excluded(self):
        row = {"epoch": self.issue, "pm02": 40,
               "_provenance": {"receiptKnown": False, "availableEpoch": None}}
        self.assertTrue(features.feature_metadata([row], self.issue)["available"])
        self.assertFalse(features.feature_metadata([row], self.issue)["allInputReceiptsKnown"])
        row["_provenance"] = {"receiptKnown": True, "availableEpoch": self.issue - 1,
                              "confirmedEpoch": None}
        self.assertFalse(features.feature_metadata([row], self.issue)["available"])

    def test_received_only_row_is_not_assumed_persisted(self):
        row = {"epoch": self.issue - 60, "pm02": 40, "received_epoch": self.issue - 1}
        self.assertFalse(features.feature_metadata([row], self.issue)["available"])
        row["persisted_epoch"] = self.issue - .5
        self.assertTrue(features.feature_metadata([row], self.issue)["available"])

    def test_sequence_has_exact_issue_relative_bounds_and_missing_bins(self):
        rows = [{"epoch": self.issue - 180, "pm02": 80},
                {"epoch": self.issue, "pm02": 40},
                {"epoch": self.issue - 360, "pm02": 1000}]
        sequence = features.sequence_bins(rows, self.issue, lookback_minutes=6, bin_minutes=3)
        self.assertEqual(sequence.index.tolist(), [self.issue - 180, self.issue])
        self.assertEqual(sequence.pm25.tolist(), [80, 40])
        sequence = features.sequence_bins(rows[-2:-1], self.issue, lookback_minutes=6, bin_minutes=3)
        self.assertTrue(math.isnan(sequence.pm25.iloc[0]))
        self.assertEqual(sequence.pm25.iloc[-1], 40)
        self.assertFalse(features.feature_metadata([], self.issue)["available"])

    def test_sequence_flattening_preserves_chronological_order(self):
        sequence = features.sequence_bins(self.rows, self.issue)
        result = features.issue_features(self.rows, self.issue)
        self.assertEqual(result["seq_pm25_177_180min"], sequence.pm25.iloc[0])
        self.assertEqual(result["seq_pm25_0_3min"], sequence.pm25.iloc[-1])
        self.assertEqual(result["seq_temperature_0_3min"], sequence.temperature.iloc[-1])
        self.assertEqual(tuple(result), features.FEATURE_COLUMNS)

    def test_batch_and_serving_features_are_identical_for_dense_offsets(self):
        epochs = [self.issue - 600, self.issue - 300, self.issue, self.issue + 73]
        frame = features.issue_feature_frame(self.rows, epochs)
        self.assertEqual(tuple(frame.columns), features.FEATURE_COLUMNS)
        self.assertEqual(len(frame.attrs["featureProvenance"]), len(epochs))
        for issue in epochs:
            expected = features.issue_features(self.rows, issue)
            np.testing.assert_equal(frame.loc[issue].to_numpy(), list(expected.values()))
        self.assertFalse(frame.attrs["imputationApplied"])

    def test_stale_missing_and_negative_pm_are_not_filled(self):
        row = {"epoch": self.issue - 241, "pm02": 50}
        result = features.issue_features([row], self.issue)
        self.assertTrue(math.isnan(result["freshPm25"]))
        self.assertFalse(features.feature_metadata([row], self.issue)["available"])
        invalid = [{"epoch": self.issue, "pm02": -20}, {"epoch": self.issue - 60, "pm02": float("inf")}]
        self.assertTrue(math.isnan(features.issue_features(invalid, self.issue)["freshPm25"]))
        empty = features.issue_feature_frame([], [])
        self.assertEqual(tuple(empty.columns), features.FEATURE_COLUMNS)
        self.assertEqual(empty.shape, (0, len(features.FEATURE_COLUMNS)))

    def test_feature_hash_changes_for_same_count_corrected_value(self):
        first = [{"epoch": self.issue, "pm02": 50}]
        second = [{"epoch": self.issue, "pm02": 70}]
        self.assertNotEqual(features.feature_metadata(first, self.issue)["inputValuesSha256"],
                            features.feature_metadata(second, self.issue)["inputValuesSha256"])

    def test_issue_time_and_sequence_arguments_are_validated(self):
        for invalid in (None, True, math.nan, -1, pd.Timestamp("2026-10-08")):
            with self.assertRaises(ValueError):
                features.issue_features(self.rows, invalid)
        with self.assertRaises(ValueError):
            features.issue_feature_frame(self.rows, [self.issue, self.issue])
        with self.assertRaises(ValueError):
            features.sequence_bins(self.rows, self.issue, lookback_minutes=5, bin_minutes=3)


if __name__ == "__main__":
    unittest.main()
