"""Synthetic tests for immutable, as-issued regional PM input snapshots.

These use no network, sensor database, or production archive.  An affine
concentration field makes spatial and temporal interpolation independently
checkable rather than merely repeating the interpolation implementation.
"""

from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))

from contextlib import closing
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import regional_pm_inputs as regional


ISSUE = 1_790_700_000
FIRST_HOUR = ISSUE // 3600 * 3600 - 3 * 3600
TIMES = tuple(FIRST_HOUR + i * 3600 for i in range(10))


def concentration(latitude, longitude, epoch):
    return (20.0 + 2.0 * latitude + 3.0 * (longitude - 101.0)
            + 4.0 * (epoch - FIRST_HOUR) / 3600.0)


def raw_response():
    result = []
    for latitude, longitude in regional.requested_grid():
        pm = [concentration(latitude, longitude, hour) for hour in TIMES]
        result.append({
            "latitude": latitude,
            "longitude": longitude,
            "timezone": "GMT",
            "utc_offset_seconds": 0,
            "hourly_units": {"time": "unixtime", "pm2_5": "μg/m³", "pm10": "μg/m³"},
            "hourly": {"time": list(TIMES), "pm2_5": pm,
                       "pm10": [value + 10.0 for value in pm]},
        })
    return result


class RegionalPMInputTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.db = self.directory / "regional_pm.db"

    def snapshot(self, fetched=ISSUE, raw=None):
        return regional.build_snapshot(raw if raw is not None else raw_response(), fetched)

    def add(self, fetched=ISSUE, raw=None):
        payload = self.snapshot(fetched, raw)
        regional.init_archive(self.db)
        self.assertTrue(regional.save_snapshot(self.db, payload, recorded_epoch=payload["fetchedEpoch"]))
        return payload

    def selected(self, issue=ISSUE, cutoff=None):
        archive = regional.load_runs(self.db, issue if cutoff is None else cutoff)
        return archive, regional.latest_run(archive, issue)

    def test_grid_is_unique_row_major_and_snaps_center_to_model_spacing(self):
        coordinates = regional.requested_grid()
        self.assertIsInstance(coordinates, tuple)
        self.assertEqual(len(coordinates), 9)
        self.assertEqual(len(set(coordinates)), 9)
        self.assertEqual(coordinates, tuple(sorted(coordinates)))
        self.assertAlmostEqual(coordinates[4][0], 3.2)
        self.assertAlmostEqual(coordinates[4][1], 101.6)
        for axis in (0, 1):
            unique = sorted({point[axis] for point in coordinates})
            self.assertEqual(len(unique), 3)
            self.assertAlmostEqual(unique[1] - unique[0], .4)
            self.assertAlmostEqual(unique[2] - unique[1], .4)

    def test_valid_snapshot_preserves_exact_source_response_and_coordinates(self):
        raw = raw_response()
        expected = deepcopy(raw)
        payload = self.snapshot(raw=raw)
        self.assertEqual(payload["version"], "regional_cams_pm_grid_v1")
        self.assertEqual(payload["fetchedEpoch"], ISSUE)
        self.assertEqual(payload["rawResponse"], expected)
        self.assertEqual(len(payload["requestedLocations"]), 9)
        self.assertEqual(len(payload["returnedLocations"]), 9)
        json.dumps(payload, allow_nan=False)
        raw[0]["hourly"]["pm2_5"][0] = 999.0
        self.assertEqual(payload["rawResponse"], expected)

    def test_invalid_units_or_absent_required_field_are_rejected(self):
        for field in ("pm2_5", "pm10"):
            for change in ("wrong_unit", "missing_unit", "missing_field"):
                with self.subTest(field=field, change=change):
                    raw = raw_response()
                    if change == "wrong_unit":
                        raw[4]["hourly_units"][field] = "mg/m³"
                    elif change == "missing_unit":
                        del raw[4]["hourly_units"][field]
                    else:
                        del raw[4]["hourly"][field]
                    with self.assertRaises(ValueError):
                        self.snapshot(raw=raw)

    def test_negative_boolean_and_nonfinite_concentrations_are_rejected(self):
        for value in (-.1, True, float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=repr(value)):
                raw = raw_response()
                raw[2]["hourly"]["pm2_5"][1] = value
                with self.assertRaises(ValueError):
                    self.snapshot(raw=raw)

    def test_missing_values_remain_missing_instead_of_becoming_zero(self):
        raw = raw_response()
        raw[4]["hourly"]["pm2_5"][2] = None
        payload = self.snapshot(raw=raw)
        self.assertIsNone(payload["rawResponse"][4]["hourly"]["pm2_5"][2])
        regional.init_archive(self.db)
        regional.save_snapshot(self.db, payload, recorded_epoch=payload["fetchedEpoch"])
        _, run = self.selected()
        latitude, longitude = regional.requested_grid()[4]
        self.assertIsNone(regional.interpolate(run, TIMES[2], latitude, longitude))
        self.assertIsNone(regional.interpolate(run, TIMES[2] + 1800, *regional.TARGET))

    def test_hourly_epochs_must_increase_and_match_across_cells(self):
        changes = ("duplicate", "reverse", "nonhourly", "mismatch", "too_short")
        for change in changes:
            with self.subTest(change=change):
                raw = raw_response()
                if change == "duplicate":
                    raw[0]["hourly"]["time"][2] = TIMES[1]
                elif change == "reverse":
                    raw[0]["hourly"]["time"].reverse()
                elif change == "nonhourly":
                    raw[0]["hourly"]["time"][2] += 1
                elif change == "mismatch":
                    raw[0]["hourly"]["time"] = [hour + 3600 for hour in TIMES]
                else:
                    for cell in raw:
                        for name in ("time", "pm2_5", "pm10"):
                            cell["hourly"][name] = cell["hourly"][name][:2]
                with self.assertRaises(ValueError):
                    self.snapshot(raw=raw)

    def test_duplicate_irregular_or_distant_model_cells_are_rejected(self):
        for change in ("duplicate", "irregular", "distant", "reordered", "missing_cell"):
            with self.subTest(change=change):
                raw = raw_response()
                if change == "duplicate":
                    raw[1]["latitude"] = raw[0]["latitude"]
                    raw[1]["longitude"] = raw[0]["longitude"]
                elif change == "irregular":
                    raw[8]["longitude"] += .03
                elif change == "distant":
                    for cell in raw:
                        cell["latitude"] += .3
                elif change == "reordered":
                    raw[0], raw[1] = raw[1], raw[0]
                else:
                    raw.pop()
                with self.assertRaises(ValueError):
                    self.snapshot(raw=raw)

    def test_latest_run_uses_exact_issue_and_excludes_future_or_stale_sources(self):
        self.add(ISSUE - 60)
        self.add(ISSUE + 60)
        archive, run = self.selected(cutoff=ISSUE + 60)
        self.assertIsNotNone(run)
        self.assertEqual(regional.decoded_payload(run)["fetchedEpoch"], ISSUE - 60)
        current = regional.latest_run(archive, ISSUE + 60)
        self.assertEqual(regional.decoded_payload(current)["fetchedEpoch"], ISSUE + 60)
        self.assertIsNone(regional.latest_run(archive, ISSUE - 61))
        stale_archive = regional.load_runs(self.db, ISSUE + 60 + 7201)
        self.assertIsNone(regional.latest_run(stale_archive, ISSUE + 60 + 7201))

    def test_archive_cutoff_cannot_establish_latest_vintage_for_a_later_issue(self):
        self.add(ISSUE)
        self.add(ISSUE + 1)
        archive = regional.load_runs(self.db, ISSUE)
        self.assertIsNone(regional.latest_run(archive, ISSUE + 1))

    def test_fractional_fetch_completion_rounds_up_to_prevent_early_availability(self):
        payload = regional.fetch_snapshot(
            fetcher=lambda url, **kwargs: raw_response(), clock=lambda: ISSUE + .25)
        self.assertEqual(payload["fetchedEpoch"], ISSUE + 1)
        regional.init_archive(self.db)
        self.assertTrue(regional.save_snapshot(self.db, payload, recorded_epoch=payload["fetchedEpoch"]))
        archive = regional.load_runs(self.db, ISSUE + 1)
        self.assertIsNone(regional.latest_run(archive, ISSUE))
        selected = regional.latest_run(archive, ISSUE + 1)
        self.assertIsNotNone(selected)
        self.assertEqual(regional.decoded_payload(selected)["fetchedEpoch"], ISSUE + 1)

    def test_timestamp_is_insert_only_and_changed_payload_cannot_overwrite(self):
        payload = self.add()
        self.assertFalse(regional.save_snapshot(self.db, deepcopy(payload),
                                                recorded_epoch=payload["fetchedEpoch"]))
        changed = raw_response()
        changed[0]["hourly"]["pm2_5"][0] += 1.0
        with self.assertRaises(ValueError):
            regional.save_snapshot(self.db, self.snapshot(raw=changed), recorded_epoch=ISSUE)
        _, run = self.selected()
        self.assertEqual(regional.decoded_payload(run)["rawResponse"], payload["rawResponse"])

    def test_late_archive_receipt_is_unavailable_before_recording_and_duplicate_preserves_receipt(self):
        payload = self.snapshot()
        regional.init_archive(self.db)
        self.assertTrue(regional.save_snapshot(self.db, payload, recorded_epoch=ISSUE + 120))
        early_archive = regional.load_runs(self.db, ISSUE + 119)
        self.assertIsNone(regional.latest_run(early_archive, ISSUE + 119))

        # Loading later must not retrospectively expose the delayed insert.
        later_archive = regional.load_runs(self.db, ISSUE + 999)
        self.assertIsNone(regional.latest_run(later_archive, ISSUE + 119))
        selected = regional.latest_run(later_archive, ISSUE + 120)
        self.assertIsNotNone(selected)
        self.assertEqual(selected.fetched_epoch, ISSUE)
        self.assertEqual(selected.recorded_epoch, ISSUE + 120)
        self.assertEqual(selected.available_epoch, ISSUE + 120)

        self.assertFalse(regional.save_snapshot(self.db, deepcopy(payload),
                                                recorded_epoch=ISSUE + 999))
        reloaded = regional.load_runs(self.db, ISSUE + 999)
        self.assertEqual(regional.latest_run(reloaded, ISSUE + 120).recorded_epoch, ISSUE + 120)
        with closing(sqlite3.connect(self.db)) as connection:
            recorded = connection.execute(
                "SELECT recorded_epoch FROM regional_pm_runs WHERE fetched_epoch=?", (ISSUE,)).fetchone()[0]
        self.assertEqual(recorded, ISSUE + 120)

    def test_archive_payload_is_deterministic_gzip_and_hash_matches_content(self):
        payload = self.add()
        with closing(sqlite3.connect(self.db)) as connection:
            stored_hash, stored_blob = connection.execute(
                "SELECT payload_sha256,payload FROM regional_pm_runs WHERE fetched_epoch=?",
                (ISSUE,)).fetchone()
        self.assertEqual(stored_blob[:2], b"\x1f\x8b")
        # Gzip's timestamp is zero, so publication-clock-independent serialization
        # is deterministic and the exact source snapshot can be verified later.
        self.assertEqual(stored_blob[4:8], b"\x00\x00\x00\x00")
        serialized = gzip.decompress(stored_blob)
        self.assertEqual(json.loads(serialized), payload)
        self.assertEqual(stored_hash, hashlib.sha256(serialized).hexdigest())
        second_path = self.directory / "second.db"
        regional.init_archive(second_path)
        regional.save_snapshot(second_path, payload, recorded_epoch=payload["fetchedEpoch"])
        with closing(sqlite3.connect(second_path)) as connection:
            second_hash, second_blob = connection.execute(
                "SELECT payload_sha256,payload FROM regional_pm_runs WHERE fetched_epoch=?",
                (ISSUE,)).fetchone()
        self.assertEqual((stored_hash, stored_blob), (second_hash, second_blob))

    def test_corrupt_archived_hash_is_rejected_without_reader_writes(self):
        self.add()
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("UPDATE regional_pm_runs SET payload_sha256=?", ("0" * 64,))
        before = self.db.read_bytes()
        archive, run = self.selected()
        self.assertIsNone(run)
        self.assertEqual(archive.invalid_run_count, 1)
        self.assertEqual(self.db.read_bytes(), before)

    def test_poll_once_deduplicates_from_persisted_archive_after_restart(self):
        calls = []

        def fetcher(url, *, timeout):
            calls.append((url, timeout))
            return raw_response()

        first = regional.poll_once(self.db, fetcher=fetcher, clock=lambda: ISSUE)
        self.assertTrue(first["inserted"])
        self.assertEqual(len(calls), 1)
        self.assertIn("domains=cams_global", calls[0][0])
        self.assertIn("forecast_hours=72", calls[0][0])
        self.assertGreater(calls[0][1], 0)

        # A fresh call has no in-memory state but still skips a recent fetch.
        second = regional.poll_once(self.db, fetcher=fetcher, clock=lambda: ISSUE + 30)
        self.assertFalse(second["inserted"])
        self.assertEqual(second["status"], "recent_snapshot")
        self.assertEqual(second["fetchedEpoch"], ISSUE)
        self.assertEqual(len(calls), 1)

        third = regional.poll_once(self.db, fetcher=fetcher, clock=lambda: ISSUE + 3600)
        self.assertTrue(third["inserted"])
        self.assertEqual(len(calls), 2)
        archive = regional.load_runs(self.db, ISSUE + 3600)
        self.assertEqual(archive.fetched_epochs, (ISSUE, ISSUE + 3600))

    def test_failed_fetch_validation_preserves_previous_immutable_archive(self):
        self.add()
        before = self.db.read_bytes()
        bad = raw_response()
        bad[1]["hourly_units"]["pm2_5"] = "mg/m³"
        with self.assertRaises(ValueError):
            regional.poll_once(self.db, fetcher=lambda url, **kwargs: bad,
                               clock=lambda: ISSUE + 3600)
        self.assertEqual(self.db.read_bytes(), before)
        archive = regional.load_runs(self.db, ISSUE + 3600)
        self.assertEqual(archive.fetched_epochs, (ISSUE,))

    def test_describe_requires_valid_weather_known_at_the_issue(self):
        self.add()
        archive, _ = self.selected(cutoff=ISSUE + 60)
        weather = {
            "fetchedEpoch": ISSUE - 60, "source": "Open-Meteo Best Match",
            "latitude": regional.TARGET[0], "longitude": regional.TARGET[1],
            "hourlyUnits": {"wind_speed_10m": "km/h", "wind_direction_10m": "°"},
            "hourly": [{"epoch": hour, "wind_speed_10m": 10.0,
                        "wind_direction_10m": 270.0} for hour in TIMES],
        }
        for change in ("missing", "future", "stale", "bad_unit", "missing_values"):
            with self.subTest(change=change):
                changed = deepcopy(weather)
                if change == "missing":
                    changed = None
                elif change == "future":
                    changed["fetchedEpoch"] = ISSUE + 1
                elif change == "stale":
                    changed["fetchedEpoch"] = ISSUE - 7201
                elif change == "bad_unit":
                    changed["hourlyUnits"]["wind_speed_10m"] = "m/s"
                else:
                    for row in changed["hourly"]:
                        row["wind_speed_10m"] = None
                result = regional.describe(archive, changed, ISSUE, ISSUE + 3600, ISSUE + 7200)
                self.assertFalse(result["available"])
                json.dumps(result, allow_nan=False)
        valid = regional.describe(archive, weather, ISSUE, ISSUE + 3600, ISSUE + 7200)
        self.assertTrue(valid["available"])
        json.dumps(valid, allow_nan=False)

    def test_load_missing_archive_does_not_create_files_and_existing_load_is_readonly(self):
        archive = regional.load_runs(self.db, ISSUE)
        self.assertFalse(self.db.exists())
        self.assertIsNone(regional.latest_run(archive, ISSUE))
        self.add()
        before = self.db.read_bytes()
        before_files = sorted(path.name for path in self.directory.iterdir())
        loaded, run = self.selected()
        self.assertIsNotNone(run)
        self.assertEqual(self.db.read_bytes(), before)
        self.assertEqual(sorted(path.name for path in self.directory.iterdir()), before_files)
        copy = regional.decoded_payload(run)
        copy["rawResponse"][0]["hourly"]["pm2_5"][0] = 999.0
        self.assertNotEqual(regional.decoded_payload(run)["rawResponse"][0]["hourly"]["pm2_5"][0], 999.0)

    def test_affine_field_interpolates_in_space_and_time_and_preserves_pm10(self):
        self.add()
        _, run = self.selected()
        epoch = TIMES[4] + 1800
        latitude, longitude = regional.TARGET
        self.assertAlmostEqual(regional.interpolate(run, epoch, latitude, longitude),
                               concentration(latitude, longitude, epoch), places=8)
        self.assertAlmostEqual(regional.interpolate(run, epoch, latitude, longitude, field="pm10"),
                               concentration(latitude, longitude, epoch) + 10.0, places=8)
        for latitude, longitude in regional.requested_grid():
            self.assertAlmostEqual(regional.interpolate(run, TIMES[3], latitude, longitude),
                                   concentration(latitude, longitude, TIMES[3]), places=8)

    def test_interpolation_uses_actual_returned_centers_not_requested_coordinates(self):
        raw = raw_response()
        for cell in raw:
            cell["latitude"] += .1
            cell["longitude"] += .1
            values = [concentration(cell["latitude"], cell["longitude"], hour) for hour in TIMES]
            cell["hourly"]["pm2_5"] = values
            cell["hourly"]["pm10"] = [value + 10.0 for value in values]
        self.add(raw=raw)
        _, run = self.selected()
        epoch = TIMES[4] + 1800
        self.assertAlmostEqual(regional.interpolate(run, epoch, *regional.TARGET),
                               concentration(*regional.TARGET, epoch), places=8)

    def test_outside_spatial_or_temporal_bounds_returns_missing(self):
        self.add()
        _, run = self.selected()
        latitude, longitude = regional.TARGET
        for hour, lat, lon in (
                (TIMES[0] - 1, latitude, longitude),
                (TIMES[-1] + 1, latitude, longitude),
                (TIMES[4], 2.0, longitude),
                (TIMES[4], latitude, 103.0)):
            with self.subTest(hour=hour, latitude=lat, longitude=lon):
                self.assertIsNone(regional.interpolate(run, hour, lat, lon))


if __name__ == "__main__":
    unittest.main()
