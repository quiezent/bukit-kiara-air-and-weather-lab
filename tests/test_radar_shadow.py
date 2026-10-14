"""Offline contract tests for the isolated RainViewer shadow collector."""

from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))

from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
import unittest

try:
    from PIL import Image
except ImportError:  # The standalone collector's image dependency is optional.
    Image = None

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from radar_shadow import (  # noqa: E402
    API_URL, ATTRIBUTION_TEXT, ATTRIBUTION_URL, PollTooSoon,
    _echo_features, poll_once,
)


def png(pixels=()):
    image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    for x, y, color in pixels:
        image.putpixel((x, y), color)
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


class RadarShadowTests(unittest.TestCase):
    @unittest.skipIf(Image is None, "Pillow is required for PNG feature tests")
    def test_features_are_bounded_and_exclude_uncovered_pixels(self):
        radar = png([
            (256, 256, (0, 100, 200, 255)),  # TTDI core
            (256, 242, (0, 100, 200, 255)),  # nearby north
            (266, 256, (0, 100, 200, 255)),  # nearby east, but no coverage
            (300, 256, (0, 100, 200, 255)),  # beyond 15 km
        ])
        coverage = png([(266, 256, (0, 0, 0, 255))])
        features = _echo_features(radar, coverage)
        self.assertEqual(features["extent_radius_km"], 15)
        self.assertGreater(features["echo_fraction_3km"], 0)
        self.assertGreater(features["sector_echo_fraction_3to15km"]["N"], 0)
        self.assertEqual(features["sector_echo_fraction_3to15km"]["E"], 0)
        self.assertLess(features["coverage_fraction_15km"], 1)
        self.assertLess(features["nearest_echo_km_within_15km"], 1)
        self.assertLess(features["echo_fraction_15km"], 0.01)

    @unittest.skipIf(Image is None, "Pillow is required for PNG feature tests")
    def test_no_radar_coverage_is_missing_not_dry(self):
        no_coverage = Image.new("RGBA", (512, 512), (0, 0, 0, 255))
        stream = BytesIO()
        no_coverage.save(stream, format="PNG")
        features = _echo_features(png(), stream.getvalue())
        self.assertEqual(features["coverage_fraction_15km"], 0)
        self.assertIsNone(features["echo_fraction_15km"])
        self.assertIsNone(features["echo_fraction_3km"])

    @unittest.skipIf(Image is None, "Pillow is required for PNG feature tests")
    def test_poll_uses_only_past_frames_and_persists_features_not_tiles(self):
        now = [1_000_000.0]
        metadata = {
            "version": "2.0", "generated": 999_980,
            "host": "https://tilecache.rainviewer.com",
            "radar": {
                "past": [
                    {"time": 999_200, "path": "/v2/radar/older"},
                    {"time": 999_800, "path": "/v2/radar/latest"},
                ],
                "nowcast": [{"time": 1_000_600, "path": "/v2/radar/future"}],
            },
        }
        radar = png([(256, 256, (0, 100, 200, 255))])
        requested_urls = []

        def fake_fetch(url):
            requested_urls.append(url)
            if url == API_URL:
                return json.dumps(metadata).encode("utf-8")
            if "/coverage/" in url:
                return png()
            if "/radar/" in url:
                return radar
            raise AssertionError(f"unexpected URL {url}")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.jsonl"
            record = poll_once(path, fetch_bytes=fake_fetch, clock=lambda: now[0])
            self.assertEqual(record["status"], "observed")
            self.assertEqual(record["fetched_at_epoch"], now[0])
            self.assertEqual(record["api_generated_at_epoch"], 999_980)
            self.assertEqual(record["latest_frame_age_seconds"], 200)
            self.assertEqual([f["frame_generated_at_epoch"] for f in record["frames"]],
                             [999_200, 999_800])
            self.assertTrue(record["observation_only"])
            self.assertEqual(record["attribution_text"], ATTRIBUTION_TEXT)
            self.assertEqual(record["attribution_url"], ATTRIBUTION_URL)
            self.assertFalse(any("future" in url for url in requested_urls))
            self.assertEqual(len(requested_urls), 4)  # metadata, coverage, two tiles
            saved = path.read_text(encoding="utf-8")
            self.assertEqual(len(saved.splitlines()), 1)
            self.assertEqual(json.loads(saved)["frames"], record["frames"])
            self.assertNotIn("PNG", saved)
            self.assertNotIn("forecast", saved.lower())

            now[0] += 599
            with self.assertRaises(PollTooSoon):
                poll_once(path, fetch_bytes=fake_fetch, clock=lambda: now[0])
            self.assertEqual(len(requested_urls), 4)
            now[0] += 1
            poll_once(path, fetch_bytes=fake_fetch, clock=lambda: now[0])
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 2)

    def test_future_dated_past_frame_fails_closed_and_rate_limits_retry(self):
        now = [2_000_000.0]
        metadata = {
            "generated": 1_999_990, "host": "https://tilecache.rainviewer.com",
            "radar": {"past": [{"time": 2_000_001, "path": "/v2/radar/wrong"}]},
        }
        calls = []

        def fake_fetch(url):
            calls.append(url)
            return json.dumps(metadata).encode("utf-8")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.jsonl"
            with self.assertRaisesRegex(ValueError, "future"):
                poll_once(path, fetch_bytes=fake_fetch, clock=lambda: now[0])
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "fetch_error")
            self.assertNotIn("frames", saved)
            with self.assertRaises(PollTooSoon):
                poll_once(path, fetch_bytes=fake_fetch, clock=lambda: now[0])
            self.assertEqual(calls, [API_URL])


if __name__ == "__main__":
    unittest.main()
