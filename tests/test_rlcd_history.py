"""Synthetic tests for bounded shared observed history; no database or models."""
from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))

import math
import unittest

import rlcd_history


NOW = 1_800_000_000
START = NOW - rlcd_history.WINDOW_S


def dense_points():
    points = [[START + i * 10, 100, 28, 32] for i in range(2161)]
    points[500][2] = 41
    points[800][2] = 20
    points[1200][3] = 29
    points[1490][3] = 55
    return points


def sensor_rows(points):
    return [dict(epoch=p[0], pm02=p[1], atmp=p[2], heatindex=p[3]) for p in points]


class SharedHistorySelectionTests(unittest.TestCase):
    def assert_native_selection(self, source, selected, maximum):
        by_epoch = {point[0]: point for point in source}
        self.assertEqual(len(selected), min(len(source), maximum))
        self.assertEqual([p[0] for p in selected], sorted({p[0] for p in selected}))
        if source:
            self.assertIs(selected[0], source[0])
            self.assertIs(selected[-1], source[-1])
        for point in selected:
            self.assertIs(point, by_epoch[point[0]])

    def test_at_most_128_rows_are_returned_without_sampling(self):
        for count in (0, 1, 64, 122, 128):
            with self.subTest(count=count):
                source = [[START + i * 150, 100, 28, 32] for i in range(count)]
                selected, method = rlcd_history._select(source, START)
                self.assertIs(selected, source)
                self.assertEqual(method, "actual_rows")

    def test_dense_flat_pm_retains_temperature_and_heat_index_extrema(self):
        source = dense_points()
        for maximum in (8, 16, 64, 128):
            with self.subTest(maximum=maximum):
                selected, method = rlcd_history._select(source, START, maximum)
                self.assert_native_selection(source, selected, maximum)
                for field in (1, 2, 3):
                    self.assertEqual(min(p[field] for p in selected), min(p[field] for p in source))
                    self.assertEqual(max(p[field] for p in selected), max(p[field] for p in source))
                self.assertIn("pm25_temperature_heat_index", method)

    def test_extrema_in_each_time_bin_are_preserved_for_all_channels(self):
        source = dense_points()
        source[300][1], source[700][1] = 75, 150
        source[1400][1], source[2000][1] = 80, 160
        source[1600][2], source[1700][2] = 19, 43
        source[400][3], source[600][3] = 27, 54
        selected, _ = rlcd_history._select(source, START, 16)
        epochs = {p[0] for p in selected}
        for bucket in (source[:1080], source[1080:]):
            for field in (1, 2, 3):
                low = min(bucket, key=lambda p: (p[field], p[0]))
                high = max(bucket, key=lambda p: (p[field], p[0]))
                self.assertIn(low[0], epochs)
                self.assertIn(high[0], epochs)

    def test_constant_channels_fill_entire_budget_with_actual_rows(self):
        source = [[START + i * 10, 100, 28, 32] for i in range(2161)]
        for maximum in (4, 8, 16, 64, 128):
            with self.subTest(maximum=maximum):
                selected, _ = rlcd_history._select(source, START, maximum)
                self.assert_native_selection(source, selected, maximum)

    def test_all_missing_channels_still_fill_with_actual_rows(self):
        source = [[START + i * 10, None, None, None] for i in range(2161)]
        selected, _ = rlcd_history._select(source, START, 128)
        self.assert_native_selection(source, selected, 128)
        self.assertTrue(all(p[1:] == [None, None, None] for p in selected))

    def test_tiny_budgets_explicitly_prioritize_pm(self):
        source = dense_points()
        source[400][1], source[1600][1] = 70, 170
        for maximum in (4, 5, 6, 7):
            with self.subTest(maximum=maximum):
                selected, method = rlcd_history._select(source, START, maximum)
                self.assert_native_selection(source, selected, maximum)
                self.assertEqual(min(p[1] for p in selected), 70)
                self.assertEqual(max(p[1] for p in selected), 170)
                self.assertIn("pm25_priority", method)

    def test_nonuniform_rows_are_balanced_in_time_and_never_interpolated(self):
        source = [[START + i * 2, 100, 28, 32] for i in range(1000)]
        source += [[START + 3600 + i * 1500, 100, 28, 32] for i in range(12)]
        source.append([NOW, 100, 28, 32])
        selected, _ = rlcd_history._select(source, START, 16)
        self.assert_native_selection(source, selected, 16)
        self.assertGreaterEqual(sum(p[0] >= START + 3600 for p in selected), 6)

    def test_selection_does_not_mutate_native_points(self):
        source = dense_points()
        before = [list(point) for point in source]
        rlcd_history._select(source, START, 16)
        self.assertEqual(source, before)

    def test_invalid_limits_are_rejected_even_for_empty_history(self):
        for maximum in (True, False, 3, 129, 8.0, "8", None):
            with self.subTest(maximum=maximum):
                with self.assertRaises(ValueError):
                    rlcd_history._select([], START, maximum)


class SharedHistoryProjectionTests(unittest.TestCase):
    def test_122_real_rows_are_all_projected_at_default_capacity(self):
        points = [[START + i * 170, 100, 28, 32] for i in range(122)]
        history = rlcd_history.build_history(sensor_rows(points), NOW)
        self.assertEqual(history["points"], points)
        self.assertEqual(history["source_rows"], 122)
        self.assertEqual(history["method"], "actual_rows")

    def test_summary_is_full_source_precision_and_independent_of_graph_budget(self):
        points = dense_points()
        for i, point in enumerate(points):
            point[1] = 100.13 + (i % 7) / 100
        rows = sensor_rows(points)
        expected = {"available": True,
                    "average_ugm3": math.fsum(p[1] for p in points) / len(points),
                    "lowest_ugm3": min(p[1] for p in points),
                    "highest_ugm3": max(p[1] for p in points),
                    "sample_count": len(points)}
        for maximum in (4, 8, 16, 64, 128):
            with self.subTest(maximum=maximum):
                history = rlcd_history.build_history(rows, NOW, max_points=maximum)
                self.assertEqual(history["pm25_summary"], expected)
                self.assertEqual(history["source_rows"], 2161)

    def test_dense_collection_gaps_keep_original_boundaries_for_every_budget(self):
        points = dense_points()
        points = points[:1000] + points[1081:]
        rows = sensor_rows(points)
        expected_gap = [[START + 9990, START + 10810]]
        for maximum in (4, 8, 16, 64, 128):
            with self.subTest(maximum=maximum):
                history = rlcd_history.build_history(rows, NOW, max_points=maximum)
                self.assertEqual(history["gaps"], expected_gap)
                self.assertEqual(history["gap_kind"], "source_collection_before_downsampling")

    def test_missing_field_on_selected_native_extreme_is_not_filled(self):
        points = dense_points()
        points[1490][2] = None
        history = rlcd_history.build_history(sensor_rows(points), NOW, max_points=8)
        selected = {point[0]: point for point in history["points"]}
        self.assertEqual(selected[points[1490][0]], points[1490])
        self.assertIsNone(selected[points[1490][0]][2])

    def test_nulls_ranges_and_one_decimal_projection_remain_explicit(self):
        rows = [dict(epoch=NOW - 20, pm02=100.13, atmp=28.24, heatindex=32.24),
                dict(epoch=NOW - 10, pm02=-1, atmp=float("nan"), heatindex=251),
                dict(epoch=NOW, pm02=None, atmp=29, heatindex=None)]
        history = rlcd_history.build_history(rows, NOW)
        self.assertEqual(history["points"], [[NOW - 20, 100.1, 28.2, 32.2],
                                              [NOW - 10, None, None, None],
                                              [NOW, None, 29, None]])
        self.assertEqual(history["pm25_summary"]["sample_count"], 1)
        self.assertEqual(history["pm25_summary"]["average_ugm3"], 100.13)

    def test_window_filtering_and_duplicate_row_identity_are_unchanged(self):
        rows = [dict(epoch=START - 1, pm02=1, atmp=20, heatindex=20),
                dict(epoch=START, pm02=None, atmp=25, heatindex=None),
                dict(epoch=START, pm02=99, atmp=None, heatindex=99),
                dict(epoch=NOW, pm02=10, atmp=28, heatindex=32),
                dict(epoch=NOW + 1, pm02=999, atmp=99, heatindex=99),
                dict(epoch=True, pm02=999, atmp=99, heatindex=99)]
        history = rlcd_history.build_history(rows, NOW)
        self.assertEqual(history["points"], [[START, None, 25, None], [NOW, 10, 28, 32]])
        self.assertEqual(history["source_rows"], 2)
        self.assertEqual(history["pm25_summary"]["sample_count"], 1)

    def test_age_comes_from_latest_accepted_source_value_before_selection(self):
        points = dense_points()
        for point in points[-60:]:
            point[1:] = [None, None, None]
        rows = sensor_rows(points)
        for maximum in (4, 8, 16, 64, 128):
            with self.subTest(maximum=maximum):
                history = rlcd_history.build_history(rows, NOW, max_points=maximum)
                self.assertEqual(history["observed_epoch"], NOW - 600)
                self.assertEqual(history["age_s"], 600)
                self.assertFalse(history["fresh"])
                self.assertTrue(history["available"])

    def test_empty_history_has_no_fabricated_observations(self):
        history = rlcd_history.build_history([], NOW)
        self.assertFalse(history["available"])
        self.assertFalse(history["fresh"])
        self.assertEqual(history["points"], [])
        self.assertEqual(history["gaps"], [])
        self.assertIsNone(history["observed_epoch"])
        self.assertEqual(history["pm25_summary"]["sample_count"], 0)


if __name__ == "__main__":
    unittest.main()
