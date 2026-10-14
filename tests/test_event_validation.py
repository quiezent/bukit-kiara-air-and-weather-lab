"""Meaningful synthetic contracts for bounded as-issued event validation."""

from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))
from copy import deepcopy
from contextlib import closing, redirect_stderr
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import tracemalloc
import unittest
from unittest.mock import patch

import numpy as np

import event_validation as event


class IssuedValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.db = self.directory / "sensor.db"
        self.ledger = self.directory / "original.jsonl"
        self.issue = int(datetime(2026, 10, 7, 12, 0, tzinfo=event.MYT).timestamp())
        with closing(sqlite3.connect(self.db)) as connection:
            connection.execute("CREATE TABLE readings(epoch INTEGER PRIMARY KEY,pm02 REAL)")

    def tearDown(self):
        self.temporary.cleanup()

    def prediction(self, issue=None, threshold=20, probability=(.6, .3, .1), **updates):
        issue = self.issue if issue is None else issue
        row = {"available": True, "modelVersion": "synthetic_frozen_event_v1", "adapterVersion": "synthetic_adapter_v1",
               "forecastIssuedEpoch": issue, "publishedEpoch": issue + 20, "recordedEpoch": issue + 20,
               "isOriginallyIssued": True, "predictionRole": "genuinely_issued_live_large20_prediction",
               "referencePm": 50., "changeThresholdUgM3": threshold, "thresholdComparison": "at_least",
               "target": "first_sampled_crossing_within90_minutes", "direction": "drop",
               "directionDecision": "three_class_argmax_ties_unresolved", "directionTieTolerance": 0,
               "trainingSupport": {"drop": 10, "none": 30, "rise": 5},
               "trainingCutoffEpoch": issue - 43200, "maxTrainingOutcomeEpoch": issue - 86400,
               "fittedAtEpoch": issue - 60, "featureOriginEpoch": issue // 900 * 900,
               "freshReferenceEpoch": issue - 30, "sensorWatermarkEpoch": issue - 30,
               "inputMaxMeasurementEpoch": issue - 30, "featureSourceMaxEpoch": issue // 900 * 900 - 60,
               "weatherFetchedEpoch": issue - 60,
               "modelIdentity": {"adapterSha256": "original_identity_a"},
               "selectedPolicy": {"trainingCadence": "fixed_four_hour_origins_plus300_seconds"},
               "trainingFitCadence": "fixed_four_hour_origins_plus300_seconds",
               "horizons": [{"leadMinutes": lead, **dict(zip(event.PROBABILITY_KEYS, probability))} for lead in event.LEADS]}
        row.update(updates)
        return row

    def append(self, *rows):
        with self.ledger.open("a", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")

    def buckets(self, values, issue=None, skip=None):
        issue = self.issue if issue is None else issue
        origin = issue // 900 * 900
        records = []
        for index, value in enumerate(values, 1):
            if skip == index:
                continue
            end = origin + index * 900
            records.extend((end - lag * 60, value) for lag in range(15))
        with closing(sqlite3.connect(self.db)) as connection:
            connection.executemany("INSERT OR REPLACE INTO readings(epoch,pm02) VALUES(?,?)", records)
            connection.commit()

    def score(self, cutoff=None, **kwargs):
        return event.score_issued(self.db, [self.ledger], self.issue + 7200 if cutoff is None else cutoff, **kwargs)

    def only_group(self, report):
        self.assertEqual(len(report["groups"]), 1)
        return report["groups"][0]

    def test_first_equal_threshold_cause_absorbs_later_opposite_crossing(self):
        self.buckets([30, 80, 50, 50, 50, 50])
        self.append(self.prediction())
        report = self.score()
        group = self.only_group(report)
        self.assertEqual(report["totals"]["scored"], 1)
        for score in group["leads"].values():
            self.assertEqual(score["actualClassCounts"], {"drop": 1, "none": 0, "rise": 0})
            self.assertAlmostEqual(score["multiclassBrier"], .26)
        self.assertEqual(group["final90Minutes"]["deployedDirection"]["correctDirectionRecall"], 1)

    def test_exact_issue_clock_interpolation_changes_outcome_and_support(self):
        issue = self.issue + 300
        self.buckets([20, 50, 50, 50, 50, 50, 50], issue=issue)
        row = self.prediction(issue=issue)
        self.append(row)
        report = self.score()
        self.assertEqual(self.only_group(report)["final90Minutes"]["actualClassCounts"]["drop"], 1)
        medians, _ = event._sensor_snapshot(self.db, self.issue + 900, self.issue + 7200)
        observed, reason = event._observed(row, medians, self.issue + 7200)
        self.assertIsNone(reason)
        self.assertAlmostEqual(observed["values"][0], 30)
        self.assertEqual(observed["supportEndEpoch"], self.issue + 6300)

    def test_interpolated_target_waits_for_last_touched_bucket_not_nominal90(self):
        issue = self.issue + 300
        self.buckets([30] * 7, issue=issue)
        self.append(self.prediction(issue=issue))
        group = self.only_group(self.score(cutoff=issue + 5400))
        self.assertEqual(group["scoredCount"], 0)
        self.assertEqual(group["unscorableReasons"], {"target_buckets_not_closed_at_cutoff": 1})

    def test_gap_after_observed_early_crossing_invalidates_all_six(self):
        self.buckets([30, 50, 50, 50, 50, 50], skip=6)
        self.append(self.prediction())
        group = self.only_group(self.score())
        self.assertEqual(group["scoredCount"], 0)
        self.assertEqual(group["unscorableReasons"], {"incomplete_covered_target_support": 1})

    def test_future_sensor_rows_cannot_change_current_report_or_be_read(self):
        self.buckets([50] * 6)
        self.append(self.prediction())
        cutoff = self.issue + 5400
        before = self.score(cutoff=cutoff)
        with closing(sqlite3.connect(self.db)) as connection:
            connection.executemany("INSERT INTO readings(epoch,pm02) VALUES(?,?)",
                                   [(cutoff + 60, 1000), (event.SEALED_START_EPOCH, 2000)])
            connection.commit()
        after = self.score(cutoff=cutoff)
        self.assertEqual(before, after)
        self.assertLessEqual(after["manifest"]["sensor"]["maxSensorEpochRead"], cutoff)

    def test_cutoff_future_and_sealed_guards_run_before_any_file_read(self):
        with patch.object(event, "_snapshot_ledgers", side_effect=AssertionError("should not read")):
            with self.assertRaisesRegex(ValueError, "sealed"):
                self.score(cutoff=event.SEALED_START_EPOCH)
            with patch.object(event.time, "time", return_value=self.issue):
                with self.assertRaisesRegex(ValueError, "future"):
                    self.score(cutoff=self.issue + 1)
            with self.assertRaisesRegex(ValueError, "max_age_days"):
                self.score(max_age_days=36)

    def test_required_cutoff_has_no_implicit_now(self):
        with self.assertRaises(TypeError):
            event.score_issued(self.db, [self.ledger])

    def test_original_publication_flag_role_and_delays_are_enforced(self):
        self.buckets([50] * 6)
        self.append(self.prediction(isOriginallyIssued=False),
                    self.prediction(issue=self.issue + 1, predictionRole="replayed_prediction"),
                    self.prediction(issue=self.issue + 2, publishedEpoch=self.issue + 123, recordedEpoch=self.issue + 123),
                    self.prediction(issue=self.issue + 3, publishedEpoch=self.issue + 2))
        report = self.score()
        self.assertEqual(report["totals"]["originallyIssued"], 0)
        rejected = report["manifest"]["ledgers"][0]["rejectedRows"]
        self.assertEqual(rejected["not_originally_issued"], 1)
        self.assertEqual(rejected["not_original_live_prediction_role"], 1)
        self.assertEqual(rejected["publication_delay_outside_0_to120_seconds"], 2)

    def test_future_publication_and_future_feature_epochs_are_rejected(self):
        self.append(self.prediction(issue=self.issue + 5380),
                    self.prediction(weatherFetchedEpoch=self.issue + 1))
        report = self.score(cutoff=self.issue + 5390)
        self.assertEqual(report["totals"]["originallyIssued"], 0)
        rejected = report["manifest"]["ledgers"][0]["rejectedRows"]
        self.assertEqual(rejected["publication_not_known_at_cutoff"], 1)
        self.assertEqual(rejected["input_after_issue_weatherFetchedEpoch"], 1)

    def test_future_ledger_values_are_not_json_decoded(self):
        self.append(self.prediction(issue=event.SEALED_START_EPOCH))
        with patch.object(event.json, "loads", side_effect=AssertionError("future snapshot decoded")):
            report = self.score()
        self.assertEqual(report["manifest"]["ledgers"][0]["rejectedRows"],
                         {"future_or_sealed_issue_not_decoded": 1})

    def test_large_ledger_streams_in_bounded_memory_and_rejects_incomplete_tail(self):
        old = json.dumps(self.prediction(issue=self.issue - 40 * 86400, pad="x" * 10000)) + "\n"
        with self.ledger.open("w", encoding="utf-8") as stream:
            stream.write(old * 800)
        self.append(self.prediction())
        with self.ledger.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(self.prediction(issue=self.issue + 60))[:-5])
        self.assertGreater(self.ledger.stat().st_size, 8_000_000)
        cutoff = self.issue + 7200
        tracemalloc.start()
        try:
            rows, manifests = event._snapshot_ledgers([self.ledger], cutoff, cutoff - 35 * 86400)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        self.assertLess(peak, 2_000_000)
        self.assertEqual([row["forecastIssuedEpoch"] for row in rows], [self.issue])
        self.assertEqual(manifests[0]["rejectedRows"], {"incomplete_snapshot_tail": 1,
                                                        "issue_before_cutoff_window_not_decoded": 800})

    def test_unavailable_is_retained_even_when_model_was_fitted_after_issue(self):
        self.append(self.prediction(available=False, reason="model_not_yet_fitted_at_issue",
                                    fittedAtEpoch=self.issue + 10, horizons=[]))
        report = self.score()
        group = self.only_group(report)
        self.assertEqual(report["totals"]["unavailable"], 1)
        self.assertEqual(group["unavailableReasons"], {"model_not_yet_fitted_at_issue": 1})
        self.assertEqual(report["manifest"]["sensor"]["sensorRowsRead"], 0)

    def test_identity_policy_threshold_adapter_and_clock_changes_split_groups(self):
        self.buckets([50] * 7)
        base = self.prediction()
        self.append(base,
                    self.prediction(issue=self.issue + 1, modelIdentity={"adapterSha256": "identity_b"}),
                    self.prediction(issue=self.issue + 2, selectedPolicy={"trainingCadence": "new_policy"}),
                    self.prediction(issue=self.issue + 3, threshold=10),
                    self.prediction(issue=self.issue + 4, adapterVersion="v2"),
                    self.prediction(issue=self.issue + 300),
                    self.prediction(issue=self.issue + 600))
        report = self.score()
        self.assertEqual(len(report["groups"]), 7)
        self.assertEqual(report["totals"]["scored"], 7)
        regimes = [group["metadata"]["actualIssueClockRegime"]["regime"] for group in report["groups"]]
        self.assertEqual(regimes.count("fixed_four_hour_origin_plus300_seconds"), 1)

    def test_missing_legacy_identity_is_not_given_current_adapter_identity(self):
        row = self.prediction()
        del row["modelIdentity"]
        self.append(row)
        self.buckets([50] * 6)
        group = self.only_group(self.score())
        self.assertEqual(group["metadata"]["modelIdentity"], {"provenance": "missing_archived_model_identity"})

    def test_duplicate_and_conflicting_publications_do_not_manufacture_support(self):
        row = self.prediction()
        self.append(row, row)
        self.buckets([50] * 6)
        report = self.score()
        self.assertEqual(report["totals"]["originallyIssued"], 1)
        self.assertEqual(report["manifest"]["ledgers"][0]["duplicateRows"], 1)
        conflict = deepcopy(row)
        conflict["referencePm"] = 40
        self.append(conflict)
        report = self.score()
        self.assertEqual(report["totals"]["originallyIssued"], 0)
        self.assertEqual(report["manifest"]["ledgers"][0]["rejectedRows"]["conflicting_original_publication"], 2)

    def test_overlapping_supports_count_one_episode_not_many_predictions(self):
        self.buckets([30] * 7)
        self.append(self.prediction(), self.prediction(issue=self.issue + 60), self.prediction(issue=self.issue + 120))
        group = self.only_group(self.score())
        self.assertEqual(group["scoredCount"], 3)
        self.assertEqual(group["support"]["distinctTargetDates"], 1)
        self.assertEqual(group["support"]["connectedOverlappingSupportEpisodes"], 1)
        self.assertEqual(group["support"]["byFinalCause"]["drop"]["connectedOverlappingSupportEpisodes"], 1)
        self.assertEqual(event._episode_count([(0, 10), (10, 20), (50, 60)]), 2)

    def test_class_order_brier_and_issued_support_prior_are_correctly_paired(self):
        self.buckets([70] * 6)
        row = self.prediction(probability=(.1, .2, .7), direction="rise", trainingSupport={"drop": 1, "none": 2, "rise": 4})
        self.append(row)
        final = self.only_group(self.score())["final90Minutes"]
        self.assertAlmostEqual(final["multiclassBrier"], .14)
        self.assertEqual(final["actualClassCounts"], {"drop": 0, "none": 0, "rise": 1})
        self.assertAlmostEqual(final["causewiseBrier"]["rise"], .09)
        prior = final["asIssuedTrainingSupportPrior"]
        # Counts1,2,4 with pseudocount1 become0.2,0.3,0.5.
        self.assertAlmostEqual(prior["multiclassBrier"], .38)
        self.assertAlmostEqual(prior["pairedMulticlassBrierDifference"]["meanModelMinusBaseline"], -.24)
        self.assertEqual(final["alwaysNone"]["multiclassBrier"], 2.)
        self.assertEqual(final["reliabilityFiveBins"]["rise"][3]["count"], 1)
        self.assertEqual(final["reliabilityFiveBins"]["rise"][3]["observedFrequency"], 1.)

    def test_unqualified_deployed_and_original_raw_direction_can_differ(self):
        self.buckets([30] * 6)
        self.append(self.prediction(probability=(.6, .3, .1), direction="unresolved",
                                   diagnosticDirection="drop", diagnosticDirectionAvailable=True,
                                   operationalDecisionRule="no_direction_until_actual_cadence_prospective_qualification",
                                   directionDecision="archived_cause_probability_threshold", directionCallProbability=.5))
        final = self.only_group(self.score())["final90Minutes"]
        self.assertEqual(final["deployedDirection"]["correctDirectionRecall"], 0.)
        self.assertEqual(final["diagnosticRawDirection"]["correctDirectionRecall"], 1.)
        group = self.only_group(self.score())
        for score in group["leads"].values():
            self.assertEqual(score["deployedDirection"]["directionCalls"], 0)

    def test_legacy10_diagnostic_preserves_original_threshold_not_argmax(self):
        self.buckets([30] * 6)
        self.append(self.prediction(threshold=10, probability=(.45, .4, .15), direction="unresolved",
                                   directionDecision="archived_cause_probability_threshold", directionCallProbability=.5))
        final = self.only_group(self.score())["final90Minutes"]
        self.assertEqual(final["diagnosticRawDirection"]["directionCalls"], 0)

    def test_false_and_wrong_way_direction_counts_have_distinct_denominators(self):
        metrics = event._direction_metrics(np.array([0, 2, 1, 0]), np.array([2, 0, 0, 1]))
        self.assertEqual(metrics["actualEvents"], 3)
        self.assertEqual(metrics["falseEventCalls"], 1)
        self.assertEqual(metrics["falseDirectionCalls"], 3)
        self.assertEqual(metrics["wrongWayCalls"], 2)
        self.assertAlmostEqual(metrics["eventRecall"], 2 / 3)
        self.assertEqual(metrics["correctDirectionRecall"], 0.)
        self.assertEqual(metrics["causes"]["drop"]["falseDirectionCalls"], 2)

    def test_bootstrap_resamples_date_clusters_and_is_deterministic(self):
        deltas = np.array([-1., -1., 1., 1.])
        dates = ["2026-10-06", "2026-10-06", "2026-10-07", "2026-10-07"]
        first = event._paired_interval(deltas, dates)
        self.assertEqual(first, event._paired_interval(deltas, dates))
        self.assertEqual(first["dayClusters"], 2)
        self.assertEqual(first["pairedCases"], 4)
        self.assertEqual(first["lower"], -1.)
        self.assertEqual(first["upper"], 1.)
        self.assertIsNone(event._paired_interval(deltas, ["same"] * 4)["lower"])

    def test_invalid_probability_mass_is_counted_without_scoring_or_renormalizing(self):
        self.append(self.prediction(probability=(.6, .6, .2)))
        report = self.score()
        group = self.only_group(report)
        self.assertEqual(report["totals"]["available"], 1)
        self.assertEqual(report["totals"]["unscorableAvailable"], 1)
        self.assertEqual(group["unscorableReasons"], {"invalid_probability_mass": 1})

    def test_source_database_and_ledger_are_unchanged_and_markdown_is_generated(self):
        self.buckets([30] * 6)
        self.append(self.prediction())
        before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in (self.db, self.ledger)]
        report = self.score()
        after = [hashlib.sha256(path.read_bytes()).hexdigest() for path in (self.db, self.ledger)]
        self.assertEqual(before, after)
        markdown = event.render_markdown(report)
        self.assertIn("# Issued event validation", markdown)
        self.assertIn("not an untouched test set", markdown)
        self.assertIn("| 90 | 1 |", markdown)
        self.assertFalse(report["evaluationPolicy"].get("qualified", False))

    def test_cli_refuses_txt_and_source_overwrite(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            event.main(["--db", str(self.db), "--ledger", str(self.ledger), "--cutoff-epoch", str(self.issue),
                        "--markdown-output", str(self.directory / "report.txt")])
        source_named_json = self.directory / "source.json"
        source_named_json.write_bytes(self.db.read_bytes())
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            event.main(["--db", str(source_named_json), "--ledger", str(self.ledger), "--cutoff-epoch", str(self.issue),
                        "--json-output", str(source_named_json)])


if __name__ == "__main__":
    unittest.main()
