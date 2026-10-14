"""Paths, clocks and fixed harness constants (no I/O at import)."""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

# The laptop serves production and is thermally throttled.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "2")

ROOT = Path(__file__).resolve().parents[3]          # repository root
# Private inputs are not published: point these at your own server and data snapshot.
SERVER = Path(os.environ.get("BUKIT_KIARA_SERVER_DIR", str(ROOT / "app")))   # read only
DATA = Path(os.environ.get("BUKIT_KIARA_RESEARCH_DATA", str(ROOT / "data")))
SNAPSHOT_DB = DATA / "snapshot_20261013.db"
NEIGHBOR_DB = DATA / "neighbors_20261013.sqlite3"
ISSUED_JSONL_DIR = SERVER / "private"                # append-only live files: read only
OUT = DATA / "models"                               # git-ignored large outputs
PROD_MIRROR = OUT / "prod_src_20261013"             # frozen copy of production code
DATASET_PATH = OUT / "dataset.parquet"
DATASET_META_PATH = OUT / "dataset_meta.json"

MYT = timezone(timedelta(hours=8))


def myt_epoch(text: str) -> int:
    """'YYYY-MM-DD[ HH:MM[:SS]]' in MYT -> UTC epoch seconds."""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return int(datetime.strptime(text, fmt).replace(tzinfo=MYT).timestamp())
        except ValueError:
            continue
    raise ValueError(text)


def myt(epoch) -> datetime:
    return datetime.fromtimestamp(int(epoch), MYT)


def midnight(epoch) -> int:
    return int(myt(epoch).replace(hour=0, minute=0, second=0, microsecond=0).timestamp())


# Snapshot taken 2026-10-13 21:52 MYT; the issue grid ends at the last 5-min
# clock with a visible reading (computed from the data at build time).
SNAPSHOT_TAKEN_EPOCH = myt_epoch("2026-10-13 21:52")
GRID_START_EPOCH = myt_epoch("2026-08-26")
GRID_SECONDS = 300
# Sensor data starts 2026-08-25 12:29 MYT and issued weather 2026-08-27 00:51.
# 14 full days of sensor AND weather training history exist from 2026-09-10.
EVAL_START_EPOCH = myt_epoch("2026-09-10")
# Receipt ledger (true receipt clocks) starts here.
RECEIPT_START_EPOCH = myt_epoch("2026-10-08 13:12:20")

# Visibility of rows that predate the receipt ledger. Production treats legacy
# rows as visible at their observation epoch (zero latency). The harness is
# stricter: measured receipt latency (observed->available, since Oct 8) has
# median 35 s, p95 57 s, p99 60 s, max 459 s, so a legacy reading is assumed
# visible at observation + 60 s. Weather/CAMS receipts lag fetch by <= 5 s.
LEGACY_SENSOR_VISIBILITY_LAG_SECONDS = 60
LEGACY_SOURCE_VISIBILITY_LAG_SECONDS = 5
# Neighbour history was backfilled on 2026-10-05; only live 'current' polls
# carry real receipt clocks. The reconstructed group assumes bucket end + 30 min.
NEIGHBOR_RECONSTRUCTED_LAG_SECONDS = 30 * 60
NEIGHBOR_MAX_AGE_SECONDS = 90 * 60
WEATHER_MAX_AGE_SECONDS = 2 * 3600                  # production MAX_WEATHER_AGE_SECONDS
CAMS_MAX_AGE_SECONDS = 3 * 3600

ARRIVAL_SECONDS = 5400
RIDE_END_SECONDS = 12600
CANONICAL_CLOCKS = {"07:15": "morning_same_day", "12:00": "afternoon_same_day", "18:05": "next_day"}
SESSION_HOURS = {"morning": (9, 11), "afternoon": (14, 16)}   # production next_feasible_ride_period
PLANNING_CUTOFF = 70.0
THRESHOLDS = (20.0, 40.0)
