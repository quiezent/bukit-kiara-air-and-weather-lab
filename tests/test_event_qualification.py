"""Qualification contracts: raw probability calls survive, operational calls do not."""

from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))
from datetime import datetime
import unittest

import event_qualification as q


class EventQualificationTests(unittest.TestCase):
    def issue(self, hour=12, minute=5, second=0):
        return int(datetime(2026, 10, 8, hour, minute, second, tzinfo=q.MYT).timestamp())

    def test_matching_grid_does_not_claim_prospective_validation(self):
        status = q.qualification_metadata(self.issue())
        self.assertTrue(status["trainingCadenceMatched"])
        self.assertFalse(status["qualification"]["eligible"])
        self.assertFalse(status["prospectivelyValidated"])
        self.assertTrue(status["unvalidatedLiveCadence"])

    def test_nearby_offset_and_wrong_hour_are_outside_training_grid(self):
        for issue in (self.issue(second=1), self.issue(hour=13), self.issue(minute=20)):
            with self.subTest(issue=issue):
                status = q.qualification_metadata(issue)
                self.assertFalse(status["trainingCadenceMatched"])
                self.assertIn("live_issue_outside_fixed_four_hour_plus300_training_grid",
                              status["qualification"]["reasons"])

    def test_probabilities_and_raw_direction_survive_operational_suppression(self):
        for direction in ("rise", "drop", "unresolved"):
            item = {"forecastIssuedEpoch": self.issue(), "available": True,
                    "direction": direction, "probabilityRise": .65,
                    "probabilityDrop": .15, "probabilityNoCrossing": .20,
                    "horizons": [{"leadMinutes": 90, "probabilityRise": .65}]}
            original_horizons = item["horizons"]
            q.apply_event_qualification(item)
            self.assertEqual(item["diagnosticDirection"], direction)
            self.assertEqual(item["direction"], "unresolved")
            self.assertEqual(item["directionStatus"], "unqualified_experimental")
            self.assertIs(item["horizons"], original_horizons)
            self.assertEqual(item["probabilityRise"], .65)

    def test_actual_snapshot_reference_and_unknown_historical_receipt_are_distinct(self):
        item = {"forecastIssuedEpoch": self.issue(), "modelVersion": "m",
                "adapterVersion": "v2", "changeThresholdUgM3": 20,
                "modelIdentity": {"selectedPolicySha256": "abc"}}
        old = q.publication_metadata(item)
        self.assertTrue(old["inputLineage"]["historicalReceiptTimesUnknown"])
        self.assertNotIn("inputSnapshotId", old)
        lineage = {"snapshotId": "real-snapshot", "status": "complete"}
        current = q.publication_metadata(item, lineage)
        self.assertEqual(current["inputSnapshotId"], "real-snapshot")
        self.assertEqual(current["inputLineage"], lineage)
        self.assertIsNot(current["inputLineage"], lineage)
        self.assertEqual(current["issuedCohort"]["issueCadenceRegime"], "fixed_training_grid")


if __name__ == "__main__":
    unittest.main()
