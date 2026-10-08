"""Opt-in, observation-only RainViewer shadow collector for TTDI/Bukit Kiara.

This module is deliberately separate from the dashboard and PM predictors. A
single ``poll_once(output_path)`` call requests RainViewer's documented past
Weather Maps frames, summarizes echoes inside 15 km of the public TTDI point,
and appends one JSON record. Schedule calls no more often than every 10 minutes;
the JSONL log also enforces that spacing across process restarts. PNGs are held
in memory only. No tiles, maps, forecasts, or model outputs are archived here.

RainViewer permits personal/educational API use, asks for visible linked credit,
and provides no availability guarantee. Any future display using these features
must show ``Weather data by RainViewer`` linked to https://www.rainviewer.com/.
The public API ceased providing future radar frames in 2026. Its frame ``time``
is composite generation time, not necessarily the local radar scan time; the
features below are reflectivity echoes, not measured ground rainfall or a
15–30 minute forecast. Source documentation:
https://www.rainviewer.com/api/weather-maps-api.html
https://www.rainviewer.com/api/transition-faq.html
https://www.rainviewer.com/api.html

Runtime dependency for PNG decoding: Pillow. The dashboard may schedule this
collector as a separate observation-only thread; it must not feed numeric PM
forecasts before prospective skill and data-latency validation. It can also be
invoked once with ``python app/radar_shadow.py --output PATH.jsonl``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from io import BytesIO
import json
import math
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen


API_URL = "https://api.rainviewer.com/public/weather-maps.json"
TILE_HOST = "https://tilecache.rainviewer.com"
ATTRIBUTION_TEXT = "Weather data by RainViewer"
ATTRIBUTION_URL = "https://www.rainviewer.com/"
SCHEMA_VERSION = "ttdi_radar_shadow_v1"
LATITUDE = 3.1411106257487
LONGITUDE = 101.62749852676
TILE_SIZE = 512
ZOOM = 7
COLOR_SCHEME = 2  # The only public API palette after the 2026 transition.
MAX_FRAMES = 3
POLL_INTERVAL_SECONDS = 600
FRESH_FRAME_SECONDS = 900
LOCAL_RADIUS_KM = 15.0
CORE_RADIUS_KM = 3.0
MAX_RESPONSE_BYTES = 2_000_000
_RADAR_PATH = re.compile(r"^/v2/radar/[A-Za-z0-9_-]+$")


class PollTooSoon(ValueError):
    """A successful or failed attempt is already logged within 10 minutes."""


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _epoch(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer Unix timestamp")
    return value


def _http_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "TTDI-Radar-Shadow/1.0 (personal research)"})
    with urlopen(request, timeout=15) as response:
        content = response.read(MAX_RESPONSE_BYTES + 1)
    if len(content) > MAX_RESPONSE_BYTES:
        raise ValueError("RainViewer response exceeded the size limit")
    return content


def _past_frames(payload: object, requested_at_epoch: float) -> tuple[int, list[dict]]:
    if not isinstance(payload, dict) or payload.get("host") != TILE_HOST:
        raise ValueError("unexpected RainViewer metadata or tile host")
    generated = _epoch(payload.get("generated"), "generated")
    if generated > requested_at_epoch:
        raise ValueError("RainViewer metadata is dated after this request")
    radar = payload.get("radar")
    past = radar.get("past") if isinstance(radar, dict) else None
    if not isinstance(past, list) or not past:
        raise ValueError("RainViewer returned no past radar frames")
    frames = []
    seen = set()
    for frame in past:
        if not isinstance(frame, dict):
            raise ValueError("invalid radar frame")
        stamp = _epoch(frame.get("time"), "past frame time")
        path = frame.get("path")
        if stamp > generated or stamp > requested_at_epoch:
            raise ValueError("a past radar frame is dated in the future")
        if not isinstance(path, str) or not _RADAR_PATH.fullmatch(path):
            raise ValueError("invalid radar tile path")
        if stamp in seen:
            raise ValueError("duplicate past radar frame time")
        seen.add(stamp)
        frames.append({"time": stamp, "path": path})
    frames.sort(key=lambda frame: frame["time"])
    return generated, frames[-MAX_FRAMES:]


def _coordinate_tile_url(path: str) -> str:
    return (
        f"{TILE_HOST}{path}/{TILE_SIZE}/{ZOOM}/"
        f"{LATITUDE:.7f}/{LONGITUDE:.7f}/{COLOR_SCHEME}/0_0.png"
    )


def _coverage_tile_url() -> str:
    return (
        f"{TILE_HOST}/v2/coverage/0/{TILE_SIZE}/{ZOOM}/"
        f"{LATITUDE:.7f}/{LONGITUDE:.7f}/0/0_0.png"
    )


def _png_pixels(content: bytes):
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to decode RainViewer PNG tiles") from exc
    with Image.open(BytesIO(content)) as image:
        if image.format != "PNG" or image.size != (TILE_SIZE, TILE_SIZE):
            raise ValueError("unexpected RainViewer tile format or dimensions")
        converted = image.convert("RGBA")
        converted.load()
        return converted


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _echo_features(radar_png: bytes, coverage_png: bytes) -> dict:
    radar_image = _png_pixels(radar_png)
    coverage_image = _png_pixels(coverage_png)
    radar = radar_image.load()
    coverage = coverage_image.load()
    # A coordinate tile is centered on the requested lat/lon. Web Mercator's
    # local ground scale is proportional to cos(latitude).
    km_per_pixel = (
        40075.016686 * math.cos(math.radians(LATITUDE)) / (2**ZOOM * TILE_SIZE)
    )
    center = (TILE_SIZE - 1) / 2
    reach = math.ceil(LOCAL_RADIUS_KM / km_per_pixel)
    total = covered = core_total = core_echo = outer_total = outer_echo = 0
    all_echo = 0
    nearest = None
    sectors = {name: [0, 0] for name in ("N", "E", "S", "W")}
    for y in range(int(center) - reach, int(center) + reach + 2):
        for x in range(int(center) - reach, int(center) + reach + 2):
            dx, dy = x - center, y - center
            distance = math.hypot(dx, dy) * km_per_pixel
            if distance > LOCAL_RADIUS_KM:
                continue
            total += 1
            # RainViewer documents transparent coverage-mask pixels as covered;
            # black opaque pixels denote no radar coverage.
            if coverage[x, y][3] != 0:
                continue
            covered += 1
            echo = radar[x, y][3] != 0
            all_echo += int(echo)
            if echo and (nearest is None or distance < nearest):
                nearest = distance
            if distance <= CORE_RADIUS_KM:
                core_total += 1
                core_echo += int(echo)
            else:
                outer_total += 1
                outer_echo += int(echo)
                sector = (
                    ("E" if dx > 0 else "W") if abs(dx) > abs(dy)
                    else ("S" if dy > 0 else "N")
                )
                sectors[sector][0] += 1
                sectors[sector][1] += int(echo)
    return {
        "pixel_scale_km": round(km_per_pixel, 4),
        "extent_radius_km": LOCAL_RADIUS_KM,
        "coverage_fraction_15km": _ratio(covered, total),
        "echo_fraction_3km": _ratio(core_echo, core_total),
        "echo_fraction_15km": _ratio(all_echo, covered),
        "echo_fraction_3to15km": _ratio(outer_echo, outer_total),
        "nearest_echo_km_within_15km": round(nearest, 3) if nearest is not None else None,
        "sector_echo_fraction_3to15km": {
            sector: _ratio(echo, count) for sector, (count, echo) in sectors.items()
        },
    }


def _last_attempt_epoch(output_path: Path) -> float | None:
    if not output_path.exists():
        return None
    last = None
    with output_path.open("r", encoding="utf-8") as archive:
        for line in archive:
            if line.strip():
                last = json.loads(line)
    if last is None:
        return None
    return float(last["requested_at_epoch"])


def _append(output_path: Path, record: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as archive:
        archive.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")


def poll_once(output_path: str | Path, *, fetch_bytes=None, clock=None) -> dict:
    """Append bounded features once, respecting a 10-minute attempt interval.

    ``fetch_bytes`` and ``clock`` allow offline tests. This call has no hidden
    scheduler and only reads public, documented RainViewer URLs. A failed poll
    is logged too, so immediate retries cannot hammer the API.
    """
    output_path = Path(output_path)
    fetch_bytes = fetch_bytes or _http_bytes
    clock = clock or time.time
    requested_at = float(clock())
    if not math.isfinite(requested_at) or requested_at < 0:
        raise ValueError("clock must return a finite Unix timestamp")
    previous = _last_attempt_epoch(output_path)
    if previous is not None and requested_at - previous < POLL_INTERVAL_SECONDS:
        raise PollTooSoon("RainViewer shadow polls must be at least 10 minutes apart")

    base = {
        "schema_version": SCHEMA_VERSION,
        "source": "RainViewer Weather Maps API",
        "source_url": API_URL,
        "attribution_text": ATTRIBUTION_TEXT,
        "attribution_url": ATTRIBUTION_URL,
        "observation_only": True,
        "requested_at_epoch": requested_at,
        "requested_at_utc": _iso(requested_at),
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
    }
    try:
        metadata = json.loads(fetch_bytes(API_URL))
        generated, frames = _past_frames(metadata, requested_at)
        coverage_png = fetch_bytes(_coverage_tile_url())
        frame_rows = []
        for frame in frames:
            radar_png = fetch_bytes(_coordinate_tile_url(frame["path"]))
            frame_rows.append({
                "frame_generated_at_epoch": frame["time"],
                "frame_generated_at_utc": _iso(frame["time"]),
                "frame_id": frame["path"].rsplit("/", 1)[-1],
                "features": _echo_features(radar_png, coverage_png),
            })
        fetched_at = float(clock())
        if not math.isfinite(fetched_at) or fetched_at < requested_at:
            raise ValueError("clock moved backward during RainViewer fetch")
        latest_age = fetched_at - frames[-1]["time"]
        coverage = frame_rows[-1]["features"]["coverage_fraction_15km"]
        status = (
            "stale_frame" if latest_age > FRESH_FRAME_SECONDS
            else "partial_coverage" if coverage is None or coverage < 0.95
            else "observed"
        )
        record = {
            **base,
            "status": status,
            "fetched_at_epoch": fetched_at,
            "fetched_at_utc": _iso(fetched_at),
            "api_generated_at_epoch": generated,
            "api_generated_at_utc": _iso(generated),
            "latest_frame_age_seconds": latest_age,
            "frames": frame_rows,
        }
    except Exception as exc:
        _append(output_path, {**base, "status": "fetch_error", "error_type": type(exc).__name__})
        raise
    _append(output_path, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="feature-only JSONL archive")
    args = parser.parse_args()
    result = poll_once(args.output)
    print(json.dumps({
        "status": result["status"],
        "fetched_at_utc": result["fetched_at_utc"],
        "latest_frame_age_seconds": result["latest_frame_age_seconds"],
        "frame_count": len(result["frames"]),
    }))


if __name__ == "__main__":
    main()
