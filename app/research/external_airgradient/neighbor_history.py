"""Private AirGradient neighbor archive and causal research feature loader.

Historical values retrieved now are reconstructed inputs, never originally
available dashboard inputs. Point observations can be used as-issued only once
this archive actually received them. No network calls occur in the loader.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time
import unicodedata
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE = "https://map-data.airgradient.com/map/api/v1"
TARGET_LAT = 3.1411106257486976
TARGET_LON = 101.6274985267597
TARGET_MAP_ID = 956  # Same coordinates/name as the local target; excluded.
STATIONS = {
    "mont_kiara": 99667152,
    "setapak": 109266655,
    "klcc": 111145052,
    "setia_eco_park": 1087,
    "enviro_exceltech": 98982064,
    "cyberjaya": 52426946,
}
FIELDS = ("pm25", "minus_local", "delta30", "delta60", "age_min",
          "wind_alignment", "travel_min", "delta30_aligned")
FEATURE_NAMES = tuple("ag_" + name + "_" + field
                      for name in STATIONS for field in FIELDS)
UNIT = "ug/m3"
HISTORY_BUCKET_SECONDS = 900
DEFAULT_DB = Path(__file__).with_name("neighbors.sqlite3")


def epoch(value: str) -> int:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("UTC offset required; naive timestamps are rejected")
    return int(dt.timestamp())


def iso(value: int | float) -> str:
    return datetime.fromtimestamp(value, timezone.utc).isoformat().replace("+00:00", "Z")


def geometry(latitude: float, longitude: float) -> tuple[float, float, float]:
    # Local tangent-plane approximation is adequate at these <30 km distances.
    east = (longitude - TARGET_LON) * 111.32 * math.cos(math.radians(TARGET_LAT))
    north = (latitude - TARGET_LAT) * 111.32
    return east, north, math.hypot(east, north)


def validate_pm25_catalog(payload: list) -> dict:
    entry = next((x for x in payload if x.get("name") == "pm25"), None)
    if not entry:
        raise ValueError("PM2.5 catalog entry absent")
    default = next((x for x in entry.get("units", []) if x.get("isDefault")), None)
    if (not default or default.get("name") != "ug" or
            unicodedata.normalize("NFKC", default.get("label", "")) != "μg/m3"):
        raise ValueError("PM2.5 unit changed: require catalog-default micrograms/m3")
    return entry


def open_archive(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(db_path)
    c.executescript("""
      CREATE TABLE IF NOT EXISTS requests(
        request_id INTEGER PRIMARY KEY, kind TEXT NOT NULL, url TEXT NOT NULL,
        started_epoch REAL NOT NULL, received_epoch REAL NOT NULL,
        recorded_epoch REAL NOT NULL, http_status INTEGER NOT NULL,
        sha256 TEXT NOT NULL, raw_path TEXT NOT NULL, body_bytes INTEGER NOT NULL,
        headers_json TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS locations(
        location_id INTEGER PRIMARY KEY, feature_name TEXT, name TEXT NOT NULL,
        latitude REAL NOT NULL, longitude REAL NOT NULL, timezone TEXT,
        provider TEXT, data_source TEXT, sensor_type TEXT, licenses_json TEXT,
        metadata_request_id INTEGER NOT NULL REFERENCES requests(request_id));
      CREATE TABLE IF NOT EXISTS location_versions(
        location_id INTEGER NOT NULL, latitude REAL NOT NULL, longitude REAL NOT NULL,
        metadata_request_id INTEGER NOT NULL REFERENCES requests(request_id),
        PRIMARY KEY(location_id, metadata_request_id));
      CREATE TABLE IF NOT EXISTS archive_settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS observations(
        location_id INTEGER NOT NULL, measure TEXT NOT NULL,
        bucket_start_epoch INTEGER NOT NULL, bucket_seconds INTEGER NOT NULL,
        pm25 REAL NOT NULL, unit TEXT NOT NULL, kind TEXT NOT NULL,
        valid INTEGER NOT NULL, invalid_reason TEXT,
        request_id INTEGER NOT NULL REFERENCES requests(request_id),
        PRIMARY KEY(location_id,bucket_start_epoch,kind,request_id));
      CREATE INDEX IF NOT EXISTS observations_lookup ON
        observations(location_id,bucket_start_epoch);
    """)
    c.commit()
    return c


class Collector:
    """Sequential bounded HTTP requests; raw bytes and receipt/record times retained."""
    def __init__(self, db_path: Path = DEFAULT_DB, min_spacing_seconds: float = 0.6,
                 exclude_outliers: bool | None = None):
        self.db_path = Path(db_path)
        self.connection = open_archive(self.db_path)
        self.raw_dir = self.db_path.parent / "private_raw"
        self.raw_dir.mkdir(exist_ok=True)
        self.spacing = max(0.5, float(min_spacing_seconds))
        self.last_request_started = 0.0
        self.catalog_verified = False
        self.exclude_outliers = exclude_outliers
        policy = "provider_default" if exclude_outliers is None else "requested_" + str(exclude_outliers).lower()
        existing = self.connection.execute(
            "SELECT value FROM archive_settings WHERE key='history_outlier_query_policy'").fetchone()
        if existing and existing[0] != policy:
            self.connection.close()
            raise ValueError("Use a separate archive for another historical outlier query policy")
        self.connection.execute("INSERT OR IGNORE INTO archive_settings VALUES(?,?)",
                                ("history_outlier_query_policy", policy))
        self.connection.commit()

    def close(self):
        self.connection.close()

    def get(self, path: str, kind: str) -> tuple[object, int]:
        url = path if path.startswith("https://") else BASE + path
        if not url.startswith(BASE + "/") and not url.startswith("https://api.airgradient.com/public/api/"):
            raise ValueError("Only allowlisted AirGradient API URLs accepted")
        for attempt in range(3):
            wait = self.spacing - (time.monotonic() - self.last_request_started)
            if wait > 0:
                time.sleep(wait)
            self.last_request_started = time.monotonic()
            started = time.time()
            try:
                with urlopen(Request(url, headers={"User-Agent": "TTDI-Air-Lab-research/1.0",
                                                   "Accept": "application/json"}), timeout=40) as response:
                    body = response.read(5_000_001)
                    received = time.time()
                    status = response.status
                    headers = {key: response.headers.get(key) for key in
                               ("Date", "Content-Type", "ETag", "Last-Modified", "Retry-After")
                               if response.headers.get(key) is not None}
            except HTTPError as error:
                if error.code in (429, 500, 502, 503, 504) and attempt < 2:
                    delay = error.headers.get("Retry-After", str(2 ** (attempt + 1)))
                    try:
                        delay = float(delay)
                    except ValueError:
                        delay = 15.0
                    if delay > 60:
                        raise RuntimeError("Provider requested long backoff; stop this collection") from error
                    time.sleep(max(1.0, delay))
                    continue
                raise
            if len(body) > 5_000_000:
                raise ValueError("Response size exceeds bounded 5 MB request")
            digest = hashlib.sha256(body).hexdigest()
            raw_path = self.raw_dir / (digest + ".json.gz")
            if not raw_path.exists():
                raw_path.write_bytes(gzip.compress(body, mtime=0))
            payload = json.loads(body)
            recorded = time.time()
            cur = self.connection.execute(
                "INSERT INTO requests(kind,url,started_epoch,received_epoch,recorded_epoch,"
                "http_status,sha256,raw_path,body_bytes,headers_json) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (kind, url, started, received, recorded, status, digest,
                 str(raw_path.relative_to(self.db_path.parent)), len(body), json.dumps(headers)))
            self.connection.commit()
            return payload, int(cur.lastrowid)
        raise RuntimeError("Unreachable HTTP retry state")

    def metadata(self, include_target_audit: bool = False):
        self.get("/docs-json", "documentation")
        catalogs = {}
        for name in ("measurements", "licenses", "datasources", "coordinate-system"):
            catalogs[name], _ = self.get("/catalog/" + name, "catalog_" + name)
        validate_pm25_catalog(catalogs["measurements"])
        self.catalog_verified = True
        area, _ = self.get("/measurements/current/area?" + urlencode({
            "xmin": 101.35, "ymin": 2.9, "xmax": 101.85, "ymax": 3.4,
            "zoom": 15, "measure": "pm25"}), "location_discovery")
        ids = dict(STATIONS)
        if include_target_audit:
            ids["target_audit_only"] = TARGET_MAP_ID
        discovered = {int(x["locationId"]) for x in area.get("data", [])}
        for feature_name, location_id in ids.items():
            if location_id not in discovered:
                print("Location not currently in discovery; metadata still checked:", location_id, flush=True)
            payload, request_id = self.get("/locations/" + str(location_id), "location_metadata")
            if int(payload["locationId"]) != location_id:
                raise ValueError("Location identifier mismatch")
            lat, lon = float(payload["latitude"]), float(payload["longitude"])
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError("Invalid location geometry")
            if location_id != TARGET_MAP_ID and geometry(lat, lon)[2] < 0.25:
                raise ValueError("Neighbor overlaps target; remove before training")
            self.connection.execute("INSERT OR REPLACE INTO locations VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (location_id, feature_name, payload["locationName"], lat, lon,
                 payload.get("timezone"), payload.get("provider"), payload.get("dataSource"),
                 payload.get("sensorType"), json.dumps(payload.get("licenses", [])), request_id))
            self.connection.execute("INSERT INTO location_versions VALUES(?,?,?,?)",
                                    (location_id, lat, lon, request_id))
        self.connection.commit()

    def ingest_history(self, location_id: int, start: int, end: int):
        if not self.catalog_verified:
            raise RuntimeError("Archive current unit metadata before collection")
        if end <= start or end - start > 7 * 86400:
            raise ValueError("History requests must span >0 and <=7 days")
        query = {"measure": "pm25", "start": iso(start), "end": iso(end), "bucketSize": "15m"}
        if self.exclude_outliers is not None:
            query["excludeOutliers"] = str(self.exclude_outliers).lower()
        payload, request_id = self.get("/locations/" + str(location_id) + "/measures/history?" +
                                       urlencode(query), "history_pm25")
        data = payload.get("data", [])
        if payload.get("total") != len(data):
            raise ValueError("History response truncated or unexpectedly paginated")
        seen = set()
        received = self.connection.execute("SELECT received_epoch FROM requests WHERE request_id=?",
                                           (request_id,)).fetchone()[0]
        for row in data:
            bucket = epoch(row["timebucket"])
            # Provider SQL uses inclusive query-end; a boundary sample may
            # yield a partial last bucket. Preserve raw but never ingest it.
            if bucket == end:
                continue
            value = float(row["value"])
            if bucket in seen:
                raise ValueError("Duplicate bucket within one history response")
            seen.add(bucket)
            if not start <= bucket < end or bucket % HISTORY_BUCKET_SECONDS:
                raise ValueError("History bucket outside request bounds or 15-minute alignment")
            reason = None
            if not math.isfinite(value) or not 0 <= value <= 1000:
                reason = "not_finite_or_outside_catalog_range"
            elif bucket + HISTORY_BUCKET_SECONDS > received:
                reason = "bucket_not_yet_closed_at_receipt"
            self.connection.execute("INSERT INTO observations VALUES(?,?,?,?,?,?,?,?,?,?)",
                (location_id, "pm25", bucket, HISTORY_BUCKET_SECONDS, value, UNIT,
                 "history", int(reason is None), reason, request_id))
        self.connection.commit()
        return len(data)

    def collect_history(self, start: int, end: int, station_ids=None):
        if start % 900 or end % 900:
            raise ValueError("Use aligned UTC quarter-hour history bounds")
        self.metadata(include_target_audit=True)
        for location_id in station_ids or STATIONS.values():
            window = start
            while window < end:
                stop = min(window + 7 * 86400, end)
                count = self.ingest_history(location_id, window, stop)
                print(json.dumps({"location_id": location_id, "start": iso(window),
                                  "end": iso(stop), "returned": count}), flush=True)
                window = stop

    def collect_current(self, include_target_audit=False):
        if not self.catalog_verified:
            self.metadata(include_target_audit)
        ids = list(STATIONS.values()) + ([TARGET_MAP_ID] if include_target_audit else [])
        for location_id in ids:
            payload, request_id = self.get("/locations/" + str(location_id) + "/measures/current",
                                          "current_pm25")
            if int(payload["locationId"]) != location_id:
                raise ValueError("Current location identifier mismatch")
            if payload.get("pm25") is None:
                continue
            measured = epoch(payload["measuredAt"])
            value = float(payload["pm25"])
            received = self.connection.execute("SELECT received_epoch FROM requests WHERE request_id=?",
                                               (request_id,)).fetchone()[0]
            reason = None
            if not math.isfinite(value) or not 0 <= value <= 1000:
                reason = "not_finite_or_outside_catalog_range"
            elif measured > received:
                reason = "measured_in_future_at_receipt"
            self.connection.execute("INSERT INTO observations VALUES(?,?,?,?,?,?,?,?,?,?)",
                (location_id, "pm25", measured, 0, value, UNIT, "current",
                 int(reason is None), reason, request_id))
            self.connection.commit()
            print(json.dumps({"location_id": location_id, "measured_at": iso(measured),
                              "received_at": iso(received), "valid": reason is None}), flush=True)


class NeighborHistory:
    """Load one fixed local archive snapshot. No DB writes and no network calls.

    `mode='reconstructed'`: historical bucket start + full15min bucket + assumed
    reporting lag <= issue. This is an assumption sensitivity test, not proof of
    actual historical availability. Only retrieved history rows participate.
    `mode='issued'`: source completion AND HTTP receipt AND archive record time
    <= issue, including revision visibility. History may then participate only
    after its actual first receipt. No assumed lag is substituted for receipt.
    """
    def __init__(self, db_path: Path | str = DEFAULT_DB):
        self.db_path = Path(db_path)
        c = sqlite3.connect(self.db_path.resolve().as_uri() + "?mode=ro", uri=True)
        c.row_factory = sqlite3.Row
        self.locations = {r["feature_name"]: dict(r) for r in c.execute("SELECT * FROM locations")
                          if r["feature_name"] in STATIONS}
        self.metadata_versions = {}
        for name, location_id in STATIONS.items():
            self.metadata_versions[name] = [dict(r) for r in c.execute(
                "SELECT v.*,r.received_epoch,r.recorded_epoch FROM location_versions v JOIN requests r "
                "ON r.request_id=v.metadata_request_id WHERE v.location_id=? ORDER BY r.received_epoch",
                (location_id,))]
        self.rows = {}
        self.ends = {}
        for name, location_id in STATIONS.items():
            records = [dict(r) for r in c.execute(
                "SELECT o.*,r.received_epoch,r.recorded_epoch FROM observations o JOIN requests r "
                "ON r.request_id=o.request_id WHERE o.location_id=? AND o.valid=1 AND o.unit=? "
                "AND o.measure='pm25' ORDER BY o.bucket_start_epoch+o.bucket_seconds,r.received_epoch",
                (location_id, UNIT))]
            for r in records:
                r["end_epoch"] = r["bucket_start_epoch"] + r["bucket_seconds"]
            self.rows[name] = records
            self.ends[name] = [r["end_epoch"] for r in records]
        c.close()

    def latest_asof(self, station: str, issue_epoch: int, lag_minutes: int = 30,
                    mode: str = "reconstructed", max_age_minutes: int = 90):
        if mode not in ("reconstructed", "issued"):
            raise ValueError("Choose reconstructed or issued availability policy")
        if lag_minutes not in (15, 30, 60):
            raise ValueError("Declared reporting-lag sensitivities are 15,30,60 minutes")
        if station not in STATIONS:
            raise ValueError("Unknown neighbor or excluded target station")
        cutoff = issue_epoch - (lag_minutes * 60 if mode == "reconstructed" else 0)
        index = bisect_right(self.ends[station], cutoff) - 1
        lower = issue_epoch - max_age_minutes * 60
        while index >= 0:
            row = self.rows[station][index]
            if row["end_epoch"] < lower:
                return None
            if mode == "reconstructed":
                eligible = row["kind"] == "history"
            else:
                eligible = max(row["received_epoch"], row["recorded_epoch"]) <= issue_epoch
            if eligible:
                return dict(row)
            index -= 1
        return None

    def features(self, issue_epoch: int, local_pm: float, lag_minutes: int = 30,
                 mode: str = "reconstructed", wind_from_deg: float | None = None,
                 wind_speed_kmh: float | None = None, max_age_minutes: int = 90) -> dict:
        result = {key: float("nan") for key in FEATURE_NAMES}
        for name in STATIONS:
            row = self.latest_asof(name, issue_epoch, lag_minutes, mode, max_age_minutes)
            if row is None:
                continue
            prefix = "ag_" + name + "_"
            result[prefix + "pm25"] = row["pm25"]
            result[prefix + "minus_local"] = row["pm25"] - local_pm
            result[prefix + "age_min"] = (issue_epoch - row["end_epoch"]) / 60.0
            for minutes in (30, 60):
                # Identical lag/receipt policy at the earlier origin; no future interpolation.
                previous = self.latest_asof(name, issue_epoch - minutes * 60, lag_minutes,
                                            mode, max_age_minutes)
                if previous and abs((row["end_epoch"] - previous["end_epoch"]) - minutes * 60) <= 900:
                    result[prefix + "delta" + str(minutes)] = row["pm25"] - previous["pm25"]
            if mode == "issued":
                eligible_metadata = [r for r in self.metadata_versions[name]
                                     if max(r["received_epoch"], r["recorded_epoch"]) <= issue_epoch]
                metadata = eligible_metadata[-1] if eligible_metadata else None
            else:
                # We do not know historical station movement. Earliest retrieved
                # coordinates are a disclosed fixed-site reconstruction assumption.
                metadata = (self.metadata_versions[name][0] if self.metadata_versions[name]
                            else self.locations.get(name))
            if metadata and wind_from_deg is not None and math.isfinite(wind_from_deg):
                east, north, distance = geometry(metadata["latitude"], metadata["longitude"])
                angle = math.radians(wind_from_deg)
                alignment = (east * math.sin(angle) + north * math.cos(angle)) / distance
                result[prefix + "wind_alignment"] = alignment
                result[prefix + "delta30_aligned"] = result[prefix + "delta30"] * max(alignment, 0)
                if wind_speed_kmh is not None and math.isfinite(wind_speed_kmh) and wind_speed_kmh >= 1:
                    result[prefix + "travel_min"] = distance / wind_speed_kmh * 60.0
        return result


def collect_current_forever(db_path: Path | str = DEFAULT_DB, interval: int = 300,
                            stop_event=None, on_error=None):
    """Service-thread entry point; network failures are recorded and retried.

    Owns its SQLite connection inside the calling thread. Never imports the
    dashboard, rewrites history, fetches target data or fits a model. `stop_event`
    can be a threading.Event. Errors are sent to `on_error(str)` if supplied;
    otherwise printed without raw responses. Call in an independent daemon.
    """
    if interval < 300:
        raise ValueError("Current collection interval must be >=300 seconds")
    worker = None
    last_metadata = 0.0
    try:
        while stop_event is None or not stop_event.is_set():
            begun = time.monotonic()
            try:
                if worker is None:
                    worker = Collector(Path(db_path))
                if not worker.catalog_verified or time.monotonic() - last_metadata >= 86400:
                    worker.metadata(include_target_audit=False)
                    last_metadata = time.monotonic()
                worker.collect_current(include_target_audit=False)
            except Exception as error:
                message = json.dumps({"collector": "neighbor_pm", "error": type(error).__name__,
                                      "message": str(error), "at": iso(time.time())})
                if on_error is None:
                    print(message, flush=True)
                else:
                    try:
                        on_error(message)
                    except Exception:
                        print(message, flush=True)
                if worker is not None:
                    try:
                        worker.connection.rollback()
                    except Exception:
                        # Discard a failed DB handle and reopen on the next cycle.
                        try:
                            worker.close()
                        except Exception:
                            pass
                        worker = None
            wait = max(1.0, interval - (time.monotonic() - begun))
            if stop_event is None:
                time.sleep(wait)
            elif stop_event.wait(wait):
                break
    finally:
        if worker is not None:
            worker.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("history", "current"))
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--start", help="Timezone-aware ISO timestamp, quarter-hour aligned")
    parser.add_argument("--end", help="Timezone-aware ISO timestamp, quarter-hour aligned")
    parser.add_argument("--loop", action="store_true", help="Current only: collect indefinitely")
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--include-target-audit", action="store_true")
    parser.add_argument("--exclude-outliers", choices=("true", "false"), default=None,
                        help="History only, explicit requested policy; deployed effect unverified")
    args = parser.parse_args()
    collector = Collector(args.db, exclude_outliers=(None if args.exclude_outliers is None
                                                    else args.exclude_outliers == "true"))
    try:
        if args.command == "history":
            if not args.start or not args.end or args.loop:
                parser.error("History needs --start and --end; no loop")
            collector.collect_history(epoch(args.start), epoch(args.end))
        else:
            if args.interval < 300:
                parser.error("Current collection interval must be >=300 seconds")
            while True:
                begun = time.monotonic()
                try:
                    collector.collect_current(args.include_target_audit)
                except Exception as error:
                    if not args.loop:
                        raise
                    print(json.dumps({"error": type(error).__name__, "message": str(error),
                                      "at": iso(time.time())}), flush=True)
                if not args.loop:
                    break
                time.sleep(max(1, args.interval - (time.monotonic() - begun)))
    finally:
        collector.close()


if __name__ == "__main__":
    main()
