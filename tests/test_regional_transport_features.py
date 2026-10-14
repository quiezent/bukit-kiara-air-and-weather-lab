"""Issue-vintage, missingness and transport-convention checks."""

from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))
from contextlib import closing
from copy import deepcopy
import gzip
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import haze_transport as transport
import regional_transport_features as features

ISSUE = 1_790_700_000


def payload(fetched, direction=180., speed=10., directions=None):
    times = [ISSUE-72*3600+i*3600 for i in range(121)]
    latitudes, longitudes = [-4., 5.], [99., 105.]
    points = []
    for lat in latitudes:
        for lon in longitudes:
            point = {"latitude": lat, "longitude": lon,
                     "modelLatitude": lat, "modelLongitude": lon,
                     "geopotential_height_925hPa": [800.]*len(times)}
            for level in (925, 850):
                point[f"wind_speed_{level}hPa"] = [speed]*len(times)
                point[f"wind_direction_{level}hPa"] = [
                    (directions or {}).get(t, direction) for t in times]
            points.append(point)
    return {"version": transport.VERSION, "fetchedEpoch": fetched,
            "source": "Open-Meteo / NOAA GFS", "dataRole": "issued_regional_wind_snapshot",
            "times": times, "latitudes": latitudes, "longitudes": longitudes,
            "points": points}


class RegionalTransportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.db = Path(temporary.name)/"regional.db"
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute("CREATE TABLE regional_wind_runs "
                         "(fetched_epoch INTEGER PRIMARY KEY,model_version TEXT,payload BLOB)")

    def add(self, fetched, data):
        blob = gzip.compress(json.dumps(data, allow_nan=False).encode())
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute("INSERT OR REPLACE INTO regional_wind_runs VALUES (?,?,?)",
                         (fetched, transport.VERSION, blob))

    def describe(self, issue=ISSUE, cutoff=None, start=None, end=None):
        archive = features.load_runs(self.db, cutoff if cutoff is not None else issue)
        return features.describe_window(archive, issue,
                                        start if start is not None else issue+2*3600,
                                        end if end is not None else issue+4*3600)

    def test_exact_issue_selects_known_vintage_and_ignores_future(self):
        first, later = ISSUE-60, ISSUE+60
        self.add(first, payload(first, direction=180))
        self.add(later, payload(later, direction=0))
        before = self.describe(ISSUE, cutoff=later)
        after = self.describe(later, cutoff=later)
        self.assertTrue(before["available"])
        self.assertEqual(before["fetchedEpoch"], first)
        self.assertEqual(after["fetchedEpoch"], later)
        self.assertGreater(before["featureValues"]["regional925VMeanKmh"], 9.9)
        self.assertLess(after["featureValues"]["regional925VMeanKmh"], -9.9)
        self.assertLessEqual(before["fetchedEpoch"], before["metadata"]["forecastIssuedEpoch"])

    def test_stale_missing_and_after_archive_cutoff_stay_missing(self):
        missing = self.describe()
        self.assertFalse(missing["available"])
        self.assertTrue(all(v is None for v in missing["featureValues"].values()))
        old = ISSUE-features.MAX_AGE_SECONDS-1
        self.add(old, payload(old))
        stale = self.describe()
        self.assertFalse(stale["available"])
        self.assertIn("stale_regional_run", stale["metadata"]["errors"])
        self.assertTrue(all(v is None for v in stale["featureValues"].values()))
        self.add(ISSUE, payload(ISSUE))
        out = self.describe(ISSUE+1, cutoff=ISSUE)
        self.assertFalse(out["available"])
        self.assertIn("issue_after_regional_archive_cutoff", out["metadata"]["errors"])

    def test_latest_invalid_or_stitched_snapshot_does_not_use_older_run(self):
        self.add(ISSUE-60, payload(ISSUE-60))
        bad = payload(ISSUE)
        bad["dataRole"] = "retrospective_stitched_model_diagnostic"
        self.add(ISSUE, bad)
        result = self.describe()
        self.assertFalse(result["available"])
        self.assertEqual(result["fetchedEpoch"], ISSUE)
        self.assertIn("not_issued_regional_snapshot", result["metadata"]["errors"])
        self.assertIsNone(result["featureValues"]["regional925VMeanKmh"])

    def test_southerly_transport_and_backpath_contact_have_correct_sign(self):
        self.add(ISSUE, payload(ISSUE, direction=180))
        result = self.describe()
        values = result["featureValues"]
        self.assertGreater(values["regional925SourceComponentKmh"], 9.9)
        self.assertGreater(values["regional925SourceAlignment"], .99)
        self.assertAlmostEqual(values["regional925SourceTravelHoursApprox"],
                               features.SOURCE_DISTANCE_KM/values["regional925SourceComponentKmh"])
        self.assertEqual(values["regional925SumatraPathFraction"], 1.)
        self.assertGreater(values["regional925TransitMinHours"], 0)
        self.assertEqual(values["regionalHeightAgreementFraction"], 1.)
        self.assertFalse(result["metadata"]["sourceAttributionEstablished"])
        self.assertFalse(result["metadata"]["upwindPmAvailable"])

    def test_calm_is_not_missing_but_has_no_direction_or_travel_time(self):
        self.add(ISSUE, payload(ISSUE, speed=0))
        result = self.describe()
        self.assertTrue(result["available"])
        self.assertEqual(result["featureValues"]["regional925VMeanKmh"], 0)
        for name in ("regional925SourceAlignment", "regional925DirectionChangeDeg",
                     "regional925SourceTravelHoursApprox"):
            self.assertIsNone(result["featureValues"][name])
        self.assertEqual(result["featureValues"]["regional925PathCoverageHours"], 48)

    def test_wind_wrap_interpolates_vectors_not_angles(self):
        self.add(ISSUE, payload(ISSUE, direction=350,
                                directions={ISSUE+3600: 10.}))
        result = self.describe(start=ISSUE+1800, end=ISSUE+1800)
        self.assertTrue(result["available"])
        self.assertLess(result["featureValues"]["regional925VMeanKmh"], -9.8)
        self.assertLess(abs(result["featureValues"]["regional925UMeanKmh"]), 1e-8)

    def test_missing_wind_cells_and_bad_coordinate_axis_stay_unavailable(self):
        bad = payload(ISSUE)
        for point in bad["points"]:
            point["wind_speed_925hPa"] = [None]*len(bad["times"])
        self.add(ISSUE, bad)
        result = self.describe()
        self.assertFalse(result["available"])
        self.assertIsNone(result["featureValues"]["regional925VMeanKmh"])
        self.assertIsNone(result["featureValues"]["regional925SumatraPathFraction"])
        bad = payload(ISSUE)
        for point in bad["points"]:
            point["modelLatitude"] = None
        self.add(ISSUE, bad)
        axis = self.describe()
        self.assertFalse(axis["available"])
        self.assertIn("invalid_regional_coordinate_axes", axis["metadata"]["errors"])

    def test_content_revision_changes_for_corrected_same_timestamp(self):
        self.add(ISSUE, payload(ISSUE, direction=0))
        first = features.load_runs(self.db, ISSUE)
        old = features.describe_window(first, ISSUE, ISSUE+7200, ISSUE+14400)
        self.add(ISSUE, payload(ISSUE, direction=180))
        second = features.load_runs(self.db, ISSUE)
        updated = features.describe_window(second, ISSUE, ISSUE+7200, ISSUE+14400)
        self.assertNotEqual(first.revision, second.revision)
        self.assertLess(old["featureValues"]["regional925VMeanKmh"], 0)
        self.assertGreater(updated["featureValues"]["regional925VMeanKmh"], 0)

    def test_compact_integration_matches_existing_windgrid_paths(self):
        data = payload(ISSUE, direction=184, speed=15)
        grid = transport.WindGrid(deepcopy(data))
        expected = grid.trace(ISSUE+7200, level=925)
        actual = features._compact_path(grid, ISSUE+7200, level=925)
        for field in ("sectorContacts", "tracedHours", "stopReason"):
            self.assertEqual(actual[field], expected[field])


if __name__ == "__main__":
    unittest.main()
