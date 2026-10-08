"""Receipt-stamped wider radar echo features for prospective event research.

Only documented RainViewer *past* frames are fetched. One 512px zoom-7 tile
already spans about 312 km across at TTDI, so 60 km fits inside its bounds.
Existing 3/15 km summaries and their public JSON shape are preserved. New
features are copied to a compact sidecar archive. Images stay in memory.

Echo masks use nonzero alpha in the documented Universal Blue palette and a
separate coverage mask (transparent=covered). They are reflectivity-presence
proxies, not dBZ, measured surface rain, gust fronts, or guaranteed PM changes.
Translation is a bounded, non-wrapping overlap search on two past echo masks;
uncertain, static, inconsistent and stale motion is explicitly unavailable.

Source: https://www.rainviewer.com/api/weather-maps-api.html
Transition: https://www.rainviewer.com/api/transition-faq.html
Terms/credit: https://www.rainviewer.com/api.html
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

try:
    import radar_shadow as legacy
except ModuleNotFoundError:
    # Allows direct invocation from outside the app's working directory.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import radar_shadow as legacy

VERSION = "ttdi_radar_event_precursors_v1"
MAX_RADIUS_KM = 60.0
MAX_TRANSLATION_KM = 15.0
MAX_PAIR_SECONDS = 1200
MIN_PAIR_SECONDS = 300
MIN_COVERAGE = 0.8
MIN_ECHO_PIXELS = 20
MIN_DICE = 0.4
MIN_TRANSLATION_GAIN = 0.08


def geometry(size=legacy.TILE_SIZE, zoom=legacy.ZOOM, latitude=legacy.LATITUDE):
    scale = 40075.016686 * math.cos(math.radians(latitude)) / (2**zoom * size)
    center = (size - 1) / 2
    yy, xx = np.indices((size, size))
    east = (xx - center) * scale
    north = (center - yy) * scale
    radius = np.hypot(east, north)
    if MAX_RADIUS_KM >= center * scale:
        raise ValueError("60 km feature area does not fit inside tile")
    return scale, east, north, radius


def masks(radar_png, coverage_png):
    radar = np.asarray(legacy._png_pixels(radar_png))
    coverage = np.asarray(legacy._png_pixels(coverage_png))
    return radar[:, :, 3] > 0, coverage[:, :, 3] == 0


def expanded_features(echo, covered):
    scale, east, north, radius = geometry()
    expected = (legacy.TILE_SIZE, legacy.TILE_SIZE)
    if echo.shape != expected or covered.shape != expected:
        raise ValueError("Unexpected radar/coverage mask dimensions")
    features = {
        "pixel_scale_km": round(scale, 4), "extent_radius_km": MAX_RADIUS_KM,
        "echo_semantics": "nonzero_alpha_reflectivity_presence_not_dbz_or_ground_rain",
        "ring_echo_fraction": {}, "ring_coverage_fraction": {},
        "sector_ring_echo_fraction": {},
    }
    sector = np.floor(((np.degrees(np.arctan2(east, north)) + 22.5) % 360) / 45).astype(int)
    names = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
    for low, high in ((0, 3), (3, 15), (15, 30), (30, 60)):
        name = f"{low}to{high}km"
        domain = (radius <= high) & (radius > low if low else radius >= 0)
        available = domain & covered
        coverage = float(available.sum() / domain.sum())
        features["ring_coverage_fraction"][name] = coverage
        features["ring_echo_fraction"][name] = float((available & echo).sum() / available.sum()) if coverage >= MIN_COVERAGE else None
        features["sector_ring_echo_fraction"][name] = {}
        for idx, compass in enumerate(names):
            area = domain & (sector == idx)
            ok = area & covered
            local_coverage = float(ok.sum() / area.sum()) if area.any() else 0
            features["sector_ring_echo_fraction"][name][compass] = float((ok & echo).sum() / ok.sum()) if local_coverage >= MIN_COVERAGE else None
    area = radius <= MAX_RADIUS_KM
    features["coverage_fraction_60km"] = float((area & covered).sum() / area.sum())
    usable = area & covered & echo
    if features["coverage_fraction_60km"] >= MIN_COVERAGE and usable.any():
        features.update(nearest_echo_km_within_60km=round(float(radius[usable].min()), 3),
                        echo_centroid_east_km=float(east[usable].mean()),
                        echo_centroid_north_km=float(north[usable].mean()),
                        echo_pixels_60km=int(usable.sum()))
    else:
        features.update(nearest_echo_km_within_60km=None, echo_centroid_east_km=None,
                        echo_centroid_north_km=None, echo_pixels_60km=int(usable.sum()))
    return features


def _slices(length, delta):
    # Previous pixel at i is compared with current pixel i+delta: never wrap.
    if delta >= 0:
        return slice(0, length-delta), slice(delta, length)
    return slice(-delta, length), slice(0, length+delta)


def translation(previous, current, covered, seconds):
    if not MIN_PAIR_SECONDS <= seconds <= MAX_PAIR_SECONDS:
        return {"available": False, "reason": "frame_interval_outside_5to20_minutes"}
    scale, east, north, radius = geometry()
    region = radius <= MAX_RADIUS_KM
    if float((region & covered).sum()/region.sum()) < MIN_COVERAGE:
        return {"available": False, "reason": "insufficient_radar_coverage"}
    # Crop for a bounded inexpensive search; retain the circular coverage mask.
    iy, ix = np.where(region)
    crop = np.s_[iy.min():iy.max()+1, ix.min():ix.max()+1]
    a, b, valid = previous[crop], current[crop], (covered & region)[crop]
    if min(int((a & valid).sum()), int((b & valid).sum())) < MIN_ECHO_PIXELS:
        return {"available": False, "reason": "insufficient_echo_support"}
    reach = int(MAX_TRANSLATION_KM / scale)
    scored = []
    for dy in range(-reach, reach+1):
        py, cy = _slices(a.shape[0], dy)
        for dx in range(-reach, reach+1):
            if math.hypot(dx, dy) * scale > MAX_TRANSLATION_KM:
                continue
            px, cx = _slices(a.shape[1], dx)
            ok = valid[py, px] & valid[cy, cx]
            aa, bb = a[py, px] & ok, b[cy, cx] & ok
            support = int(aa.sum()+bb.sum())
            score = 2*int((aa & bb).sum())/support if support else 0
            scored.append((score, dx, dy))
    best = max(scored, key=lambda item: (item[0], -abs(item[1])-abs(item[2])))
    zero = next(score for score, dx, dy in scored if dx == dy == 0)
    score, dx, dy = best
    ambiguous = max((s for s, x, y in scored if math.hypot(x-dx, y-dy) >= 3), default=0)
    details = {"available": False, "overlap_dice": score, "zero_shift_dice": zero,
               "peak_margin_beyond_3pixels": score-ambiguous, "interval_seconds": seconds,
               "search_limit_km": MAX_TRANSLATION_KM}
    if dx == dy == 0 or score-zero < MIN_TRANSLATION_GAIN:
        return dict(details, reason="static_clutter_or_no_supported_translation")
    if score < MIN_DICE or score-ambiguous < 0.03:
        return dict(details, reason="ambiguous_or_low_overlap_translation")
    if math.hypot(dx, dy) >= reach-1:
        return dict(details, reason="translation_at_search_boundary")
    velocity_east = dx*scale / (seconds/3600)
    velocity_north = -dy*scale / (seconds/3600)
    details.update(available=True, reason="past_echo_translation_proxy",
                   east_kmh=velocity_east, north_kmh=velocity_north,
                   speed_kmh=math.hypot(velocity_east, velocity_north))
    return details


def motion_features(frames, mask_rows, covered, receipt_epoch):
    base = {"available": False, "semantics": "past_composite_echo_translation_not_ground_rain_or_outflow_forecast"}
    if len(frames) < 3:
        return dict(base, reason="three_past_frames_required")
    if any(f["time"] > receipt_epoch for f in frames):
        return dict(base, reason="frame_dated_after_receipt")
    if receipt_epoch - frames[-1]["time"] > legacy.FRESH_FRAME_SECONDS:
        return dict(base, reason="latest_frame_stale")
    pairs = [translation(mask_rows[i-1], mask_rows[i], covered, frames[i]["time"]-frames[i-1]["time"])
             for i in range(1, len(frames))]
    base["pairs"] = pairs
    if not all(p["available"] for p in pairs):
        return dict(base, reason="two_supported_frame_pairs_required")
    a, b = pairs[-2:]
    cosine = (a["east_kmh"]*b["east_kmh"] + a["north_kmh"]*b["north_kmh"])/(a["speed_kmh"]*b["speed_kmh"])
    speed_ratio = max(a["speed_kmh"], b["speed_kmh"])/min(a["speed_kmh"], b["speed_kmh"])
    base.update(direction_coherence=cosine, speed_ratio=speed_ratio)
    if cosine < .7 or speed_ratio > 2:
        return dict(base, reason="inconsistent_pair_motion")
    latest = expanded_features(mask_rows[-1], covered)
    x, y = latest["echo_centroid_east_km"], latest["echo_centroid_north_km"]
    vx, vy = b["east_kmh"], b["north_kmh"]
    closest_hours = -(x*vx+y*vy)/(vx*vx+vy*vy)
    distance = math.hypot(x+vx*closest_hours, y+vy*closest_hours)
    base.update(available=True, reason="coherent_past_echo_translation_proxy", east_kmh=vx,
                north_kmh=vy, speed_kmh=b["speed_kmh"],
                approaching_centroid=bool(closest_hours > 0),
                closest_approach_minutes_linear_proxy=closest_hours*60 if 0 < closest_hours <= 3 else None,
                closest_approach_distance_km_linear_proxy=distance if 0 < closest_hours <= 3 else None)
    return base


def poll_once(output_path, *, feature_output_path=None, fetch_bytes=None, clock=None):
    """One service-safe bounded collection; failures logged, no retries or fits.

    The legacy log enforces 10-minute cadence across restart. Five requests per
    successful attempt: metadata, coverage and the latest three past tiles.
    The numeric forecasts are never imported or changed.
    """
    fetch_bytes, clock = fetch_bytes or legacy._http_bytes, clock or time.time
    cache = {}

    def cached_fetch(url):
        response = fetch_bytes(url)
        cache[url] = response
        return response

    # The original owns the legacy archive, poll limit, status and UI shape.
    record = legacy.poll_once(output_path, fetch_bytes=cached_fetch, clock=clock)
    sidecar = {key: record.get(key) for key in ("requested_at_epoch", "requested_at_utc",
               "fetched_at_epoch", "fetched_at_utc", "api_generated_at_epoch", "latest_frame_age_seconds",
               "status", "error", "attribution_text", "attribution_url")}
    sidecar.update(schema_version=VERSION, observation_only=True)
    try:
        if record.get("status") != "observed":
            raise ValueError("Original radar collection unavailable")
        metadata_bytes = cache[legacy.API_URL]
        _, frames = legacy._past_frames(json.loads(metadata_bytes), record["requested_at_epoch"])
        coverage_png = cache[legacy._coverage_tile_url()]
        frame_rows, mask_rows = [], []
        covered = None
        for frame in frames:
            tile_png = cache[legacy._coordinate_tile_url(frame["path"])]
            echo, covered = masks(tile_png, coverage_png)
            mask_rows.append(echo)
            frame_rows.append({"frame_generated_at_epoch": frame["time"],
                               "features": expanded_features(echo, covered),
                               "tile_sha256": hashlib.sha256(tile_png).hexdigest()})
        received = record["fetched_at_epoch"]
        sidecar.update(event_precursors={"version": VERSION, "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
                          "coverage_sha256": hashlib.sha256(coverage_png).hexdigest(),
                          "frames": frame_rows,
                          "motion": motion_features(frames, mask_rows, covered, received),
                          "availability_rule": "max(request_received,archive_recorded)<=forecast_issue"})
    except Exception as error:
        sidecar.update(status="error", error=f"{type(error).__name__}: {error}")
    recorded = float(clock())
    if (not math.isfinite(recorded) or recorded < record["requested_at_epoch"]
            or (record.get("fetched_at_epoch") is not None and recorded < record["fetched_at_epoch"])):
        raise ValueError("invalid archive-record clock")
    sidecar["recorded_at_epoch"] = recorded
    if feature_output_path is not None:
        legacy._append(Path(feature_output_path), sidecar)
    return record


def latest_asof(path, issue_epoch, max_age_seconds=900):
    """Read compact features received AND recorded by issue; no reconstruction."""
    if not Path(path).exists():
        return None
    latest = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        received, recorded = row.get("fetched_at_epoch"), row.get("recorded_at_epoch")
        if row.get("status") != "observed" or received is None or recorded is None:
            continue
        if max(received, recorded) > issue_epoch or issue_epoch-received > max_age_seconds:
            continue
        age = row.get("latest_frame_age_seconds")
        if age is None or age+issue_epoch-received > max_age_seconds:
            continue
        if latest is None or received > latest["fetched_at_epoch"]:
            latest = row
    return latest
