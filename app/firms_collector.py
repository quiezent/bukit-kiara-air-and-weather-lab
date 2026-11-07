"""Receipt-stamped NASA FIRMS fire-hotspot archive for private haze research.

An independent service thread polls the FIRMS area CSV API for four NRT sources
over Sumatra, the Peninsula, the Riau islands and west Borneo. Detections go to
a separate SQLite file; INSERT OR IGNORE keeps each detection's FIRST receipt
clock, so research can later use the archive as issued. It never reads or
writes the main DB, trains a model or changes any published forecast.

The MAP KEY is read from private/firms.json at every poll and never stored,
logged or printed: URLs and error texts are written with the key masked as ***.
"""
from __future__ import annotations

import calendar
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import sqlite3
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

VERSION = "firms_hotspot_receipt_archive_v1"
BASE_DIR = Path(__file__).resolve().parent
ARCHIVE_PATH = BASE_DIR / "research" / "external_firms" / "firms.sqlite3"
KEY_PATH = BASE_DIR / "private" / "firms.json"

API_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
SOURCES = ("VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "MODIS_NRT")
BBOX = (94, -8, 120, 8)                 # W,S,E,N: Sumatra, Peninsula, Riau islands, all Borneo
DAY_RANGE = 2
POLL_SECONDS = 1800                     # 4 sources x 48 polls = 192 transactions a day
TIMEOUT_SECONDS = 30
SOURCE_SPACING_SECONDS = 2.0
MAX_BACKOFF_FACTOR = 4                  # failing cycles wait 1, 2, 4, 4 ... poll intervals
MAX_BODY_BYTES = 20_000_000
MASK = "***"

# Identity columns that make a body a FIRMS CSV; scan, track, frp, confidence and daynight are optional.
REQUIRED_COLUMNS = ("latitude", "longitude", "acq_date", "acq_time", "satellite", "instrument", "version")
BRIGHTNESS_COLUMNS = {"VIIRS": ("bright_ti4", "bright_ti5"), "MODIS": ("brightness", "bright_t31")}
_KEY_SHAPE = re.compile(r"[A-Za-z0-9]{16,128}")
_URL_KEY = re.compile(r"(/api/[A-Za-z_]+/csv/)[^/\s'\"]+")
_db_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS requests(
  id INTEGER PRIMARY KEY,
  source TEXT NOT NULL,
  bbox TEXT NOT NULL,
  day_range INTEGER NOT NULL,
  requested_epoch REAL NOT NULL,
  received_epoch REAL,
  http_status INTEGER,
  row_count INTEGER,
  new_rows INTEGER,
  bytes INTEGER,
  payload_sha256 TEXT,
  error TEXT,
  url_masked TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS detections(
  source TEXT NOT NULL,
  satellite TEXT NOT NULL,
  instrument TEXT NOT NULL,
  version TEXT NOT NULL,
  latitude REAL NOT NULL,
  longitude REAL NOT NULL,
  acq_date TEXT NOT NULL,
  acq_time TEXT NOT NULL,
  acq_epoch INTEGER NOT NULL,
  bright_ti4 REAL,
  bright_ti5 REAL,
  brightness REAL,
  bright_t31 REAL,
  scan REAL,
  track REAL,
  frp REAL,
  confidence TEXT,
  daynight TEXT,
  first_request_id INTEGER NOT NULL REFERENCES requests(id),
  first_received_epoch REAL NOT NULL,
  UNIQUE(source, satellite, acq_date, acq_time, latitude, longitude));
CREATE INDEX IF NOT EXISTS detections_receipt ON detections(first_received_epoch);
CREATE INDEX IF NOT EXISTS detections_acq ON detections(acq_epoch);
CREATE INDEX IF NOT EXISTS requests_time ON requests(requested_epoch);
"""


class KeyUnavailable(Exception):
    """The key file is missing or invalid; the reason never contains the key."""


def mask(text, key=None):
    """Replace the key, and any key-shaped FIRMS URL segment, with ***."""
    text = str(text)
    if key:
        text = text.replace(key, MASK)
    return _URL_KEY.sub(lambda m: m.group(1) + MASK, text)


def _snippet(text, key, limit=200):
    """Mask the whole text first, then shorten it, so no part of the key survives a cut."""
    return " ".join(mask(text, key).split())[:limit]


class UnexpectedPayload(ValueError):
    """The body is not this source's FIRMS CSV (e.g. an 'Invalid MAP_KEY' text reply)."""


def log(message, key=None):
    print("[FIRMS collector] " + mask(message, key), flush=True)


def bbox_text(bbox=BBOX):
    return ",".join(f"{v:g}" for v in bbox)


def request_url(key, source, bbox=BBOX, day_range=DAY_RANGE):
    if source not in SOURCES:
        raise ValueError("Unknown FIRMS source")
    if not 1 <= int(day_range) <= 5:
        raise ValueError("FIRMS day range must be 1..5")
    return f"{API_BASE}/{key}/{source}/{bbox_text(bbox)}/{int(day_range)}"


def read_key(path=KEY_PATH):
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        raise KeyUnavailable(f"key file {path.name} is missing") from None
    except OSError as error:
        raise KeyUnavailable(f"key file {path.name} is unreadable ({type(error).__name__})") from None
    try:
        key = json.loads(raw).get("mapKey")
    except (ValueError, AttributeError):
        raise KeyUnavailable(f"key file {path.name} is not a JSON object") from None
    if not isinstance(key, str) or not _KEY_SHAPE.fullmatch(key.strip()):
        raise KeyUnavailable(f"key file {path.name} has no valid mapKey") from None
    return key.strip()


def connect(db_path=ARCHIVE_PATH):
    """Per-call connection in WAL mode; callers close it. Thread-safe with _db_lock."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, timeout=30)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=30000")
    connection.executescript(SCHEMA)
    existing = connection.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    if existing and existing[0] != VERSION:
        connection.close()
        raise ValueError("FIRMS archive holds another schema version; use a separate file")
    now = repr(time.time())
    connection.executemany("INSERT OR IGNORE INTO meta VALUES(?,?)", [
        ("schema_version", VERSION), ("created_epoch", now), ("bbox_wsen", bbox_text()),
        ("day_range", str(DAY_RANGE)), ("poll_seconds", str(POLL_SECONDS)), ("sources", ",".join(SOURCES)),
        ("clock", "acq_date/acq_time and acq_epoch are UTC; *_epoch receipt clocks are Unix seconds")])
    connection.commit()
    return connection


def _float(value):
    value = (value or "").strip()
    return float(value) if value else None


def acq_epoch(acq_date, acq_time):
    """UTC epoch of a FIRMS acquisition; acq_time is HHMM, leading zeros optional."""
    hhmm = str(acq_time).strip().zfill(4)
    if len(hhmm) != 4 or not hhmm.isdigit() or int(hhmm[:2]) > 23 or int(hhmm[2:]) > 59:
        raise ValueError("acq_time is not HHMM")
    day = time.strptime(str(acq_date).strip(), "%Y-%m-%d")
    return calendar.timegm(day) + int(hhmm[:2]) * 3600 + int(hhmm[2:]) * 60, hhmm


def parse_csv(source, body):
    """Parse one FIRMS area CSV payload into detection dicts.

    Returns (rows, bad_rows). Raises ValueError if the header is not the
    source's FIRMS CSV header (e.g. an 'Invalid MAP_KEY' text reply).
    """
    family = "MODIS" if source.startswith("MODIS") else "VIIRS"
    text = body.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    header = [h.strip() for h in (reader.fieldnames or [])]
    if not set(REQUIRED_COLUMNS + BRIGHTNESS_COLUMNS[family]) <= set(header):
        # No body text here: the caller masks the whole body before shortening it.
        raise UnexpectedPayload("unexpected payload")
    reader.fieldnames = header
    rows, bad = [], 0
    for record in reader:
        try:
            latitude, longitude = float(record["latitude"]), float(record["longitude"])
            if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
                raise ValueError("coordinates out of range")
            epoch, hhmm = acq_epoch(record["acq_date"], record["acq_time"])
            row = {"source": source, "satellite": (record.get("satellite") or "").strip(),
                   "instrument": (record.get("instrument") or "").strip(),
                   "version": (record.get("version") or "").strip(),
                   "latitude": latitude, "longitude": longitude,
                   "acq_date": record["acq_date"].strip(), "acq_time": hhmm, "acq_epoch": epoch,
                   "bright_ti4": None, "bright_ti5": None, "brightness": None, "bright_t31": None,
                   "scan": _float(record.get("scan")), "track": _float(record.get("track")),
                   "frp": _float(record.get("frp")),
                   "confidence": (record.get("confidence") or "").strip() or None,
                   "daynight": (record.get("daynight") or "").strip() or None}
            for column in BRIGHTNESS_COLUMNS[family]:
                row[column] = _float(record.get(column))
            rows.append(row)
        except (ValueError, TypeError, KeyError, AttributeError):
            bad += 1
    return rows, bad


def fetch(url, timeout=TIMEOUT_SECONDS):
    """GET with a bounded body. Returns (status, body); raises HTTPError/URLError."""
    request = Request(url, headers={"User-Agent": "TTDI-Air-Lab-research/1.0", "Accept": "text/csv"})
    with urlopen(request, timeout=timeout) as response:
        body = response.read(MAX_BODY_BYTES + 1)
        if len(body) > MAX_BODY_BYTES:
            raise ValueError("FIRMS response exceeds the bounded body size")
        return int(response.status), body


DETECTION_COLUMNS = ("source", "satellite", "instrument", "version", "latitude", "longitude", "acq_date",
                     "acq_time", "acq_epoch", "bright_ti4", "bright_ti5", "brightness", "bright_t31", "scan",
                     "track", "frp", "confidence", "daynight")


def store(db_path, source, requested, received, status, body, rows, error, url_masked, day_range=DAY_RANGE):
    """Record one request and INSERT OR IGNORE its detections; returns (request_id, new_rows)."""
    with _db_lock:
        connection = connect(db_path)
        try:
            with connection:
                cursor = connection.execute(
                    "INSERT INTO requests(source,bbox,day_range,requested_epoch,received_epoch,http_status,"
                    "row_count,new_rows,bytes,payload_sha256,error,url_masked) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (source, bbox_text(), int(day_range), requested, received, status,
                     None if rows is None else len(rows), None, None if body is None else len(body),
                     None if body is None else hashlib.sha256(body).hexdigest(), error, url_masked))
                request_id = cursor.lastrowid
                new_rows = 0
                if rows:
                    before = connection.total_changes
                    connection.executemany(
                        f"INSERT OR IGNORE INTO detections({','.join(DETECTION_COLUMNS)},first_request_id,"
                        f"first_received_epoch) VALUES({','.join('?' * (len(DETECTION_COLUMNS) + 2))})",
                        [tuple(r[c] for c in DETECTION_COLUMNS) + (request_id, received) for r in rows])
                    new_rows = connection.total_changes - before
                connection.execute("UPDATE requests SET new_rows=? WHERE id=?",
                                   (new_rows if rows is not None else None, request_id))
            return request_id, new_rows
        finally:
            connection.close()


def poll_source(db_path, key, source, fetcher=fetch, day_range=DAY_RANGE):
    """One FIRMS request for one source; never raises for network or payload errors."""
    url_masked = request_url(MASK, source, day_range=day_range)
    requested = time.time()
    status = body = rows = error = received = None
    try:
        status, body = fetcher(request_url(key, source, day_range=day_range), TIMEOUT_SECONDS)
        received = time.time()
        if status != 200:
            raise ValueError(f"HTTP status {status}")
        rows, bad = parse_csv(source, body)
        if bad:
            error = f"{bad} unparsable rows skipped"
    except HTTPError as http_error:
        received = time.time()
        status = http_error.code
        try:
            text = http_error.read(65536).decode("utf-8", "replace")
        except Exception:
            text = ""
        error = f"HTTPError {http_error.code}: {_snippet(text, key)}"
    except UnexpectedPayload:
        error = "unexpected payload: " + _snippet(body.decode("utf-8-sig", "replace"), key)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    if error is not None:
        error = mask(error, key)[:500]
    request_id, new_rows = store(db_path, source, requested, received, status, body, rows, error, url_masked,
                                 day_range)
    ok = rows is not None
    return {"source": source, "ok": ok, "requestId": request_id, "httpStatus": status,
            "rows": None if rows is None else len(rows), "newRows": new_rows, "error": error}


def poll_once(db_path=ARCHIVE_PATH, key_path=KEY_PATH, fetcher=fetch, spacing=SOURCE_SPACING_SECONDS):
    """Read the key, then poll every source in turn. Raises KeyUnavailable without a key."""
    key = read_key(key_path)
    results = []
    for index, source in enumerate(SOURCES):
        if index and spacing:
            time.sleep(spacing)
        result = poll_source(db_path, key, source, fetcher)
        if result["error"]:
            log(f"{source}: {result['error']}", key)
        results.append(result)
    return results


def _wait(stop_event, seconds):
    """Wall-clock wait in <=5 s ticks (resumes promptly after laptop sleep); True if stopped."""
    deadline = time.time() + seconds
    previous = time.time()
    while True:
        now = time.time()
        if now >= deadline or now < previous - 5:
            return stop_event is not None and stop_event.is_set()
        previous = now
        tick = min(5.0, deadline - now)
        if stop_event is not None:
            if stop_event.wait(tick):
                return True
        else:
            time.sleep(tick)


def collector(db_path=ARCHIVE_PATH, key_path=KEY_PATH, stop_event=None, fetcher=fetch,
              poll_seconds=POLL_SECONDS, spacing=SOURCE_SPACING_SECONDS):
    """Service-thread entry point: poll every 30 min, back off on failing cycles, never raise."""
    failures = 0
    key_notice = None
    while stop_event is None or not stop_event.is_set():
        try:
            results = poll_once(db_path, key_path, fetcher, spacing)
            key_notice = None
            succeeded = sum(r["ok"] for r in results)
            failures = 0 if succeeded else failures + 1
            log(f"{succeeded}/{len(results)} sources ok, rows "
                f"{sum(r['rows'] or 0 for r in results)}, new {sum(r['newRows'] or 0 for r in results)}")
        except KeyUnavailable as reason:
            failures = 0                        # nothing was requested; re-check at the poll interval
            if str(reason) != key_notice:       # one line per distinct reason, not per poll
                key_notice = str(reason)
                log(f"{key_notice}; not polling, will re-check every {poll_seconds:g} s")
        except Exception as error:              # never crash the thread
            failures += 1
            log(f"{type(error).__name__}: {error}", _key_or_none(key_path))
        delay = poll_seconds * min(2 ** max(failures - 1, 0), MAX_BACKOFF_FACTOR)
        if _wait(stop_event, delay):
            return


def _key_or_none(key_path):
    try:
        return read_key(key_path)
    except Exception:
        return None
