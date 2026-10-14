"""Actual adapter output contracts with fixed estimator predictions, no fitting."""

from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))
from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import event_qualification as q
import first_crossing_live as first
import large_change_live as large


class AdapterQualificationTests(unittest.TestCase):
    def setUp(self):
        self.issue = int(datetime(2026, 10, 8, 12, 7, 11, tzinfo=q.MYT).timestamp())
        cutoff = first.midnight_epoch(self.issue)
        self.metadata = {"available": True, "fittedAtEpoch": self.issue - 60,
                         "maxTrainingOutcomeEpoch": cutoff - 1}
        self.artifact = {"trainingCutoffEpoch": cutoff, "metadata": self.metadata,
                         "identity": {"recipe": "fixed_for_contract"},
                         "hazardModel": object(), "trainingDeltaPaths": None,
                         "estimator": object()}
        self.inputs = (None, {"featureValuesSha256": "exact-feature-hash"})

    def test_large20_raw_rise_and_drop_remain_diagnostic_only(self):
        for mass, expected in (([.1, .2, .7], "rise"), ([.7, .2, .1], "drop")):
            with self.subTest(expected=expected), \
                    patch.object(large, "_load_artifact", return_value=self.artifact), \
                    patch.object(large.source, "_current_inputs", return_value=(self.inputs, None)), \
                    patch.object(large.hazard, "predict_hazards", return_value=None), \
                    patch.object(large.hazard, "cumulative_incidence",
                                 return_value=np.tile(mass, (1, 6, 1))), \
                    patch.object(large.paths, "predict_training_direction_magnitude", return_value={}), \
                    patch.object(large, "_magnitude_estimates", return_value={"raw": "preserved"}):
                result = large.predict_event("unused.db", [], self.issue, self.issue, 100)
            self.assertTrue(result["available"], result)
            self.assertEqual(result["diagnosticDirection"], expected)
            self.assertEqual(result["direction"], "unresolved")
            self.assertFalse(result["qualification"]["eligible"])
            self.assertEqual(result["probabilityRise"], mass[2])
            self.assertEqual(result["magnitudeEstimates"], {"raw": "preserved"})

    def test_first_crossing_has_same_operational_guard_and_frozen_identity(self):
        with patch.object(first, "_load_artifact", return_value=self.artifact), \
                patch.object(first, "_current_inputs", return_value=(self.inputs, None)), \
                patch.object(first.h, "predictions", return_value=None), \
                patch.object(first.h, "cumulative_incidence", return_value=np.tile([.1, .2, .7], (1, 6, 1))):
            result = first.predict_event("unused.db", [], self.issue, self.issue, 100)
        self.assertEqual(result["diagnosticDirection"], "rise")
        self.assertEqual(result["direction"], "unresolved")
        self.assertEqual(result["modelIdentity"], self.artifact["identity"])
        self.assertFalse(result["qualification"]["eligible"])

    def test_original_publication_links_actual_snapshot_and_retains_raw_probabilities(self):
        record = {"forecastIssuedEpoch": self.issue, "modelVersion": large.MODEL_VERSION,
                  "adapterVersion": large.ADAPTER_VERSION, "modelIdentity": self.artifact["identity"],
                  "available": True, "direction": "rise", "probabilityRise": .7,
                  "changeThresholdUgM3": 20}
        q.apply_event_qualification(record)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "issued.jsonl"
            large.record_issued_prediction(record, output_path=output,
                recorded_epoch=self.issue + 10,
                input_lineage={"snapshotId": "immutable-real-input", "status": "complete"})
            archived = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(archived["inputSnapshotId"], "immutable-real-input")
        self.assertEqual(archived["direction"], "unresolved")
        self.assertEqual(archived["diagnosticDirection"], "rise")
        self.assertEqual(archived["probabilityRise"], .7)
        self.assertTrue(archived["isOriginallyIssued"])


if __name__ == "__main__":
    unittest.main()
