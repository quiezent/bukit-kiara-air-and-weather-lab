"""Focused checks for fixed session clocks and shadow-only direction trials."""

from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd

import afternoon_direction_features as features
import afternoon_direction_forecast as direction


LOCAL = timezone(timedelta(hours=8))


def epoch(day, hour, minute=0, second=0):
    return int(datetime(2026, 9, day, hour, minute, second, tzinfo=LOCAL).timestamp())


class DirectionAlignmentTests(unittest.TestCase):
    def test_fixed_target_is_same_afternoon_for_different_morning_origins(self):
        index = pd.date_range(
            pd.Timestamp("2026-09-27", tz="Asia/Kuala_Lumpur"),
            periods=3 * 96, freq="15min")
        frame = pd.DataFrame({
            "pm02": np.arange(len(index), dtype=float),
            "atmp": np.full(len(index), 28.0),
            "rhum": np.full(len(index), 80.0),
        }, index=index)
        origins = pd.DatetimeIndex([
            pd.Timestamp(epoch(27, 7), unit="s", tz="UTC").tz_convert("Asia/Kuala_Lumpur"),
            pd.Timestamp(epoch(27, 7, 15), unit="s", tz="UTC").tz_convert("Asia/Kuala_Lumpur"),
            pd.Timestamp(epoch(28, 7), unit="s", tz="UTC").tz_convert("Asia/Kuala_Lumpur"),
        ])
        rows = [
            {"epoch": int(stamp.timestamp()), "pm02": float(value)}
            for stamp, value in frame.pm02.items()
        ]
        weather = object()
        description = {"featureValues": {}, "metadata": {"errors": []}, "fetchedEpoch": None}
        with patch.object(features, "regional", return_value=(np.nan, np.nan, None)), \
             patch.object(features.rain, "describe_window", return_value=description):
            design = features.design(
                frame, rows, [], weather, epoch(29, 0), 420, 120,
                lag_seconds=0, only_origins=origins,
                target_clock=(14, 0, 16, 0))
        self.assertEqual(design["start"].tolist(),
                         [epoch(27, 14), epoch(27, 14), epoch(28, 14)])
        self.assertEqual(design["end"].tolist(),
                         [epoch(27, 16), epoch(27, 16), epoch(28, 16)])
        self.assertEqual(design["complete"].tolist(),
                         [epoch(27, 16), epoch(27, 16), epoch(28, 16)])
        self.assertAlmostEqual(design["labels"][0], design["labels"][1])
        self.assertAlmostEqual(design["x"]["targetLeadHours"].iloc[0], 7.0)
        self.assertAlmostEqual(design["x"]["targetLeadHours"].iloc[1], 6.75)

    def test_training_uses_only_morning_issues_with_live_lag(self):
        issue = epoch(30, 7, 33)
        index = pd.date_range(
            pd.Timestamp("2026-09-15", tz="Asia/Kuala_Lumpur"),
            pd.Timestamp(epoch(30, 7, 30), unit="s", tz="UTC").tz_convert("Asia/Kuala_Lumpur"),
            freq="15min")
        frame = pd.DataFrame({
            "pm02": np.full(len(index), 30.0),
            "atmp": np.full(len(index), 28.0),
            "rhum": np.full(len(index), 80.0),
        }, index=index)
        rows = [{"epoch": int(stamp.timestamp()), "pm02": 30.0} for stamp in index]
        calls = []
        original_design = features.design

        def record_design(*args, **kwargs):
            value = original_design(*args, **kwargs)
            calls.append((kwargs, value))
            return value

        description = {"featureValues": {}, "metadata": {"errors": []}, "fetchedEpoch": None}
        with patch.object(features, "load", return_value=(rows, frame, [], object())), \
             patch.object(features, "design", side_effect=record_design), \
             patch.object(features, "regional", return_value=(np.nan, np.nan, None)), \
             patch.object(features.rain, "describe_window", return_value=description):
            diagnostics = {}
            result = direction._training_and_query(
                "unused.db", rows, issue, epoch(30, 14), epoch(30, 16),
                preparation_diagnostics=diagnostics)
        self.assertIsNone(result)
        self.assertEqual(diagnostics["reason"], "insufficient_aligned_training_rows")
        self.assertEqual(len(calls), 2)
        training_kwargs, training = calls[0]
        self.assertEqual(training_kwargs["target_clock"], (14, 0, 16, 0))
        self.assertEqual(training_kwargs["lag_seconds"], 180)
        self.assertTrue(all(stamp.hour == 7 for stamp in training["origins"]))
        self.assertTrue(all(stamp.minute in (0, 15, 30, 45)
                            for stamp in training["origins"]))
        self.assertTrue(all(pd.Timestamp(t, unit="s", tz="UTC").tz_convert(
            "Asia/Kuala_Lumpur").hour == 14 for t in training["start"]))

    def test_selected_forecast_and_selection_metadata_survive_shadow_attempt(self):
        issue = epoch(30, 7, 30)
        start, end = epoch(30, 14), epoch(30, 16)
        old = {"available": True, "mean": 22.0, "prediction": 22.0,
               "modelVersion": "selected_session_v6",
               "forecastIssuedEpoch": issue,
               "modelSelection": {"selectedPolicy": "persistence"}}
        windows = {"afternoon": {"startEpoch": start, "endEpoch": end}}
        with patch.object(direction, "references", return_value=(
                np.array([24.0]), np.array([issue]), np.array([1]))), \
             patch.object(direction, "_training_and_query", return_value=(
                object(), object(), object(), issue // 900 * 900, 0)), \
             patch.object(direction, "_replacement", return_value={
                "mean": 10.0, "modelEvidence": {"historicalTargetClock": "same_day_14_to_16"}}):
            actual = direction.apply_experimental_afternoon(
                "unused.db", [], windows, issue, {"afternoon": old})["afternoon"]
        self.assertEqual(actual["mean"], 22.0)
        self.assertEqual(actual["modelVersion"], "selected_session_v6")
        self.assertEqual(actual["modelSelection"], old["modelSelection"])
        self.assertFalse(actual["directionalModel"]["applied"])
        self.assertEqual(actual["directionalModel"]["attempt"]["candidateMeanPm25UgM3"], 10.0)


if __name__ == "__main__":
    unittest.main()
