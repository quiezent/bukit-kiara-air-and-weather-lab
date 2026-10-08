"""Additive, immutable receipt/revision evidence; no legacy-clock backfills.

Call init_schema once in a caller-owned transaction. Recording helpers acquire a
SQLite writer lock before sampling persisted_epoch, which means archive insertion
time, not post-COMMIT durability time. They never commit, fetch, fit or mutate the
compatibility tables. received_epoch must be captured by the network collector.
After caller commit, explicitly call confirm_receipts_visible and commit its
separate confirmation transaction. Tracked readers require this post-commit proof.

Strict as-of readers use BOTH receipt clocks. Legacy compatibility rows without
receipt evidence remain explicitly unknown. Compact forecast manifests hash the
materialized values and reference ledger/query watermarks, not copied row dumps.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import gzip
import hashlib
import json
import math
import time
from forecast_payload import decode_payload_text

VERSION = "observation_receipt_revision_v1"
SOURCE_TABLES = frozenset(("weather_forecast_runs", "air_quality_forecast_runs"))


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _epoch(value, name, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite epoch")
    if not math.isfinite(value) or value <= 0 or (integer and int(value) != value):
        raise ValueError(f"{name} must be a positive finite epoch")
    return int(value) if integer else float(value)


def _reading(value):
    if not isinstance(value, Mapping):
        raise TypeError("validated reading must be a mapping")
    result = {k: v for k, v in value.items() if k != "_provenance"}
    result["epoch"] = _epoch(result.get("epoch"), "observed_epoch", integer=True)
    if not isinstance(result.get("timestamp"), str):
        raise ValueError("validated reading must preserve its timestamp")
    try:
        timestamp = datetime.fromisoformat(result["timestamp"].replace("Z", "+00:00"))
        if timestamp.tzinfo is None or int(timestamp.timestamp()) != result["epoch"]:
            raise ValueError("timestamp/observed epoch mismatch")
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError("timestamp/observed epoch mismatch") from error
    _json(result)  # Reject non-JSON and non-finite values; never round values.
    return result


def init_schema(conn):
    """Create only new tables/indexes/triggers; never invent legacy receipts."""
    statements = (
        """CREATE TABLE IF NOT EXISTS observation_provenance_metadata(
            key TEXT PRIMARY KEY,value TEXT NOT NULL)""",
        """CREATE TABLE IF NOT EXISTS sensor_observation_revisions(
            revision_id TEXT PRIMARY KEY,observed_epoch INTEGER NOT NULL,
            payload_sha256 TEXT NOT NULL,payload_json TEXT NOT NULL)""",
        """CREATE TABLE IF NOT EXISTS sensor_observation_receipts(
            receipt_id INTEGER PRIMARY KEY AUTOINCREMENT,
            revision_id TEXT NOT NULL REFERENCES sensor_observation_revisions(revision_id),
            observed_epoch INTEGER NOT NULL,received_epoch REAL NOT NULL,
            persisted_epoch REAL NOT NULL,available_epoch REAL NOT NULL,
            CHECK(available_epoch>=received_epoch AND available_epoch>=persisted_epoch))""",
        """CREATE INDEX IF NOT EXISTS sensor_receipts_observed_available
            ON sensor_observation_receipts(observed_epoch,available_epoch DESC,receipt_id DESC)""",
        """CREATE INDEX IF NOT EXISTS sensor_receipts_revision_available
            ON sensor_observation_receipts(revision_id,available_epoch DESC,receipt_id DESC)""",
        """CREATE TABLE IF NOT EXISTS sensor_observation_visibility(
            receipt_id INTEGER PRIMARY KEY REFERENCES sensor_observation_receipts(receipt_id),
            confirmed_epoch REAL NOT NULL CHECK(confirmed_epoch>0))""",
        """CREATE TABLE IF NOT EXISTS source_forecast_revisions(
            revision_id TEXT PRIMARY KEY,source_table TEXT NOT NULL,
            source_fetched_epoch INTEGER NOT NULL,payload_sha256 TEXT NOT NULL,
            payload_blob BLOB NOT NULL)""",
        """CREATE TABLE IF NOT EXISTS source_forecast_receipts(
            receipt_id INTEGER PRIMARY KEY AUTOINCREMENT,
            revision_id TEXT NOT NULL REFERENCES source_forecast_revisions(revision_id),
            source_table TEXT NOT NULL,source_fetched_epoch INTEGER NOT NULL,
            received_epoch REAL NOT NULL,persisted_epoch REAL NOT NULL,
            available_epoch REAL NOT NULL,
            CHECK(available_epoch>=received_epoch AND available_epoch>=persisted_epoch))""",
        """CREATE INDEX IF NOT EXISTS source_receipts_source_available
            ON source_forecast_receipts(source_table,source_fetched_epoch DESC,
                                       available_epoch DESC,receipt_id DESC)""",
        """CREATE TABLE IF NOT EXISTS source_forecast_visibility(
            receipt_id INTEGER PRIMARY KEY REFERENCES source_forecast_receipts(receipt_id),
            confirmed_epoch REAL NOT NULL CHECK(confirmed_epoch>0))""",
        """CREATE TABLE IF NOT EXISTS forecast_input_snapshots(
            snapshot_id TEXT PRIMARY KEY,forecast_epoch REAL NOT NULL,
            persisted_epoch REAL NOT NULL,manifest_json TEXT NOT NULL)""",
        """CREATE INDEX IF NOT EXISTS forecast_input_snapshots_issue
            ON forecast_input_snapshots(forecast_epoch)""",
    )
    for statement in statements:
        conn.execute(statement)
    conn.execute("INSERT OR IGNORE INTO observation_provenance_metadata VALUES('schema_version',?)", (VERSION,))
    version = conn.execute("SELECT value FROM observation_provenance_metadata WHERE key='schema_version'").fetchone()[0]
    if version != VERSION:
        raise ValueError("unsupported observation provenance schema")
    for table in ("sensor_observation_revisions", "sensor_observation_receipts", "sensor_observation_visibility",
                  "source_forecast_revisions", "source_forecast_receipts", "source_forecast_visibility", "forecast_input_snapshots"):
        for operation in ("UPDATE", "DELETE"):
            conn.execute(f"""CREATE TRIGGER IF NOT EXISTS {table}_immutable_{operation.lower()}
                BEFORE {operation} ON {table} BEGIN
                SELECT RAISE(ABORT,'immutable provenance evidence'); END""")


def _writer_clock(conn, clock):
    if not conn.in_transaction:
        conn.execute("BEGIN IMMEDIATE")
    # A pre-existing transaction may be read-only/deferred. This update acquires
    # the writer lock before the clock is sampled; metadata is not evidence.
    conn.execute("UPDATE observation_provenance_metadata SET value=value WHERE key='schema_version'")
    return _epoch(clock(), "persisted_epoch")


def confirm_receipts_visible(conn, *, clock=time.time):
    """Append post-first-commit proof; caller must commit this second transaction.

    Calling while a transaction remains open is an error: an insertion timestamp
    must never masquerade as post-commit availability. Confirmations can safely
    cover previously committed unconfirmed receipts visible on this connection.
    """
    if conn.in_transaction:
        raise ValueError("commit receipt transaction before confirming visibility")
    tables = (("sensor_observation_receipts", "sensor_observation_visibility"),
              ("source_forecast_receipts", "source_forecast_visibility"))
    if not all(_ledger_present(conn, name) for pair in tables for name in pair):
        return {"confirmedSensorReceipts": 0, "confirmedSourceReceipts": 0, "confirmedEpoch": None}
    # Receipt IDs are append-only, globally serialized by SQLite's writer lock.
    # Each confirmation covers the full visible tail, so indexed MAX watermarks
    # avoid rescanning years of already-confirmed polls on every dashboard commit.
    pending = any(conn.execute(f"SELECT 1 FROM {receipts} WHERE receipt_id>COALESCE((SELECT MAX(receipt_id) FROM {visible}),0) LIMIT 1").fetchone() for receipts, visible in tables)
    if not pending:
        return {"confirmedSensorReceipts": 0, "confirmedSourceReceipts": 0, "confirmedEpoch": None}
    conn.execute("BEGIN IMMEDIATE")
    confirmed = _epoch(clock(), "confirmed_epoch")  # Both first commit and writer-lock wait precede this.
    counts = []
    for receipts, visible in tables:
        cursor = conn.execute(f"""INSERT OR IGNORE INTO {visible}(receipt_id,confirmed_epoch)
            SELECT r.receipt_id,? FROM {receipts} r
            WHERE r.receipt_id>COALESCE((SELECT MAX(receipt_id) FROM {visible}),0)""", (confirmed,))
        counts.append(cursor.rowcount)
    return {"confirmedSensorReceipts": counts[0], "confirmedSourceReceipts": counts[1], "confirmedEpoch": confirmed}


def record_sensor_receipt(conn, reading, received_epoch, *, clock=time.time):
    reading = _reading(reading)
    received = _epoch(received_epoch, "received_epoch")
    if reading["epoch"] > received + 60:
        raise ValueError("sensor observation is more than 60 seconds after receipt")
    text = _json(reading)
    checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
    persisted = _writer_clock(conn, clock)
    conn.execute("INSERT OR IGNORE INTO sensor_observation_revisions VALUES(?,?,?,?)",
                 (checksum, reading["epoch"], checksum, text))
    existing = conn.execute("SELECT observed_epoch,payload_json FROM sensor_observation_revisions WHERE revision_id=?", (checksum,)).fetchone()
    if tuple(existing) != (reading["epoch"], text):
        raise ValueError("conflicting immutable sensor revision")
    available = max(received, persisted)
    cursor = conn.execute("""INSERT INTO sensor_observation_receipts
        (revision_id,observed_epoch,received_epoch,persisted_epoch,available_epoch)
        VALUES(?,?,?,?,?)""", (checksum, reading["epoch"], received, persisted, available))
    return {"revisionId": checksum, "payloadSha256": checksum,
            "materializedValuesSha256": checksum, "receiptId": cursor.lastrowid,
            "observedEpoch": reading["epoch"], "receivedEpoch": received,
            "persistedEpoch": persisted, "availableEpoch": None, "confirmedEpoch": None,
            "insertReceiptKnown": True, "receiptKnown": False, "receiptPolicy": VERSION}


def record_source_receipt(conn, source_table, payload_json, fetched_epoch, received_epoch, *, clock=time.time):
    if source_table not in SOURCE_TABLES:
        raise ValueError("unsupported source forecast table")
    fetched = _epoch(fetched_epoch, "source_fetched_epoch", integer=True)
    received = _epoch(received_epoch, "received_epoch")
    if not isinstance(payload_json, str):
        raise TypeError("source payload must be the exact archived JSON string")
    payload = json.loads(payload_json)
    _json(payload)  # Validate finite JSON while retaining original bytes below.
    if not isinstance(payload, dict) or payload.get("fetchedEpoch") != fetched:
        raise ValueError("source payload/fetch clock mismatch")
    raw = payload_json.encode("utf-8")
    checksum = hashlib.sha256(raw).hexdigest()
    revision = hashlib.sha256(_json([source_table, fetched, checksum]).encode()).hexdigest()
    persisted = _writer_clock(conn, clock)
    conn.execute("INSERT OR IGNORE INTO source_forecast_revisions VALUES(?,?,?,?,?)",
                 (revision, source_table, fetched, checksum, gzip.compress(raw, mtime=0)))
    existing = conn.execute("SELECT source_table,source_fetched_epoch,payload_sha256,payload_blob FROM source_forecast_revisions WHERE revision_id=?", (revision,)).fetchone()
    if tuple(existing[:3]) != (source_table, fetched, checksum) or gzip.decompress(existing[3]) != raw:
        raise ValueError("conflicting immutable source revision")
    available = max(received, persisted)
    cursor = conn.execute("""INSERT INTO source_forecast_receipts
        (revision_id,source_table,source_fetched_epoch,received_epoch,persisted_epoch,available_epoch)
        VALUES(?,?,?,?,?,?)""", (revision, source_table, fetched, received, persisted, available))
    return {"revisionId": revision, "payloadSha256": checksum, "receiptId": cursor.lastrowid,
            "sourceTable": source_table, "sourceFetchedEpoch": fetched,
            "receivedEpoch": received, "persistedEpoch": persisted,
            "availableEpoch": None, "confirmedEpoch": None, "insertReceiptKnown": True,
            "receiptKnown": False, "receiptPolicy": VERSION}


def load_sensor_asof(conn, issue_epoch, start_epoch, end_epoch=None, *, receipt_watermark=None):
    """Bounded, receipt-aware immutable rows. Unknown legacy rows are excluded."""
    issue = _epoch(issue_epoch, "issue_epoch")
    start = _epoch(start_epoch, "start_epoch")
    end = issue if end_epoch is None else min(issue, _epoch(end_epoch, "end_epoch"))
    watermark = 9223372036854775807 if receipt_watermark is None else int(receipt_watermark)
    if not _ledger_present(conn, "sensor_observation_visibility"):
        return []
    selected = conn.execute("""WITH eligible AS (
        SELECT r.*,v.confirmed_epoch,MAX(r.available_epoch,v.confirmed_epoch) AS confirmed_available_epoch,
            ROW_NUMBER() OVER(PARTITION BY r.observed_epoch
            ORDER BY MAX(r.available_epoch,v.confirmed_epoch) DESC,r.receipt_id DESC) AS choice
        FROM sensor_observation_receipts r JOIN sensor_observation_visibility v ON v.receipt_id=r.receipt_id
        WHERE r.observed_epoch>=? AND r.observed_epoch<=? AND r.available_epoch<=?
            AND v.confirmed_epoch<=? AND r.receipt_id<=?)
        SELECT e.receipt_id,e.revision_id,e.observed_epoch,e.received_epoch,
               e.persisted_epoch,e.confirmed_available_epoch,e.confirmed_epoch,v.payload_json
        FROM eligible e JOIN sensor_observation_revisions v ON v.revision_id=e.revision_id
        WHERE e.choice=1 ORDER BY e.observed_epoch""", (start, end, issue, issue, watermark)).fetchall()
    rows = []
    for receipt, revision, observed, received, persisted, available, confirmed, text in selected:
        reading = json.loads(text)
        reading["_provenance"] = {"receiptId": receipt, "revisionId": revision,
            "observedEpoch": observed, "receivedEpoch": received, "persistedEpoch": persisted,
            "availableEpoch": available, "receiptKnown": True, "receiptPolicy": VERSION,
            "confirmedEpoch": confirmed,
            "materializedValuesSha256": hashlib.sha256(_json(reading).encode("utf-8")).hexdigest()}
        rows.append(reading)
    return rows


def load_source_asof(conn, source_table, issue_epoch, start_fetched_epoch=0, *, receipt_watermark=None, include_revisions=False):
    """One available revision per fetched clock; unknown legacy runs excluded."""
    if source_table not in SOURCE_TABLES:
        raise ValueError("unsupported source forecast table")
    issue = _epoch(issue_epoch, "issue_epoch")
    watermark = 9223372036854775807 if receipt_watermark is None else int(receipt_watermark)
    if not _ledger_present(conn, "source_forecast_visibility"):
        return []
    choice = "" if include_revisions else "WHERE e.choice=1"
    selected = conn.execute("""WITH eligible AS (
        SELECT r.*,v.confirmed_epoch,MAX(r.available_epoch,v.confirmed_epoch) AS confirmed_available_epoch,
            ROW_NUMBER() OVER(PARTITION BY r.source_fetched_epoch
            ORDER BY MAX(r.available_epoch,v.confirmed_epoch) DESC,r.receipt_id DESC) AS choice
        FROM source_forecast_receipts r JOIN source_forecast_visibility v ON v.receipt_id=r.receipt_id
        WHERE r.source_table=? AND r.source_fetched_epoch>=? AND r.source_fetched_epoch<=?
            AND r.available_epoch<=? AND v.confirmed_epoch<=? AND r.receipt_id<=?)
        SELECT e.receipt_id,e.revision_id,e.source_fetched_epoch,e.received_epoch,
               e.persisted_epoch,e.confirmed_available_epoch,e.confirmed_epoch,v.payload_sha256,v.payload_blob
        FROM eligible e JOIN source_forecast_revisions v ON v.revision_id=e.revision_id
        """ + choice + " ORDER BY e.source_fetched_epoch,e.confirmed_available_epoch,e.receipt_id",
        (source_table, start_fetched_epoch, issue, issue, issue, watermark)).fetchall()
    return [{"fetched_epoch": fetched, "payload": gzip.decompress(blob).decode("utf-8"),
             "_provenance": {"receiptId": receipt, "revisionId": revision,
                 "sourceTable": source_table, "sourceFetchedEpoch": fetched,
                 "receivedEpoch": received, "persistedEpoch": persisted,
                 "availableEpoch": available, "payloadSha256": checksum,
                 "confirmedEpoch": confirmed,
                 "receiptKnown": True, "receiptPolicy": VERSION}}
            for receipt, revision, fetched, received, persisted, available, confirmed, checksum, blob in selected]


def _ledger_present(conn, name):
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def compatible_sensor_rows(conn, issue_epoch, start_epoch, end_epoch=None, *, strict=False):
    """Tracked values use receipts; only untracked legacy values remain unknown."""
    issue = _epoch(issue_epoch, "issue_epoch")
    end = issue if end_epoch is None else min(issue, _epoch(end_epoch, "end_epoch"))
    present = _ledger_present(conn, "sensor_observation_receipts")
    known = load_sensor_asof(conn, issue, start_epoch, end) if present else []
    if strict:
        return known
    tracked = {r[0] for r in conn.execute("SELECT DISTINCT observed_epoch FROM sensor_observation_receipts WHERE observed_epoch>=? AND observed_epoch<=?", (start_epoch, end))} if present else set()
    cursor = conn.execute("SELECT * FROM readings WHERE epoch>=? AND epoch<=? ORDER BY epoch", (start_epoch, end))
    columns = [d[0] for d in cursor.description]
    legacy = []
    for raw in cursor.fetchall():
        row = dict(zip(columns, raw))
        if row["epoch"] in tracked:
            continue  # Never fall back to an overwritten value with a late receipt.
        row["_provenance"] = {"observedEpoch": row["epoch"], "receiptKnown": False,
            "receivedEpoch": None, "persistedEpoch": None, "availableEpoch": None,
            "receiptPolicy": "legacy_receipt_unknown"}
        legacy.append(row)
    return sorted(known + legacy, key=lambda r: r["epoch"])


def compatible_source_runs(conn, source_table, issue_epoch, start_fetched_epoch=0, *, strict=False, include_revisions=False):
    """Receipt-aware source rows plus explicitly unknown untracked legacy rows.

    Results are dictionaries with fetched_epoch, source, model_version, payload
    (exact JSON text), and _provenance. include_revisions retains all available
    receipt vintages for archives reused across multiple historical issue clocks.
    Readers must select again by availableEpoch for each historical issue.
    """
    if source_table not in SOURCE_TABLES:
        raise ValueError("unsupported source forecast table")
    issue = _epoch(issue_epoch, "issue_epoch")
    present = _ledger_present(conn, "source_forecast_receipts")
    known = load_source_asof(conn, source_table, issue, start_fetched_epoch,
        include_revisions=include_revisions) if present else []
    for row in known:
        payload = json.loads(row["payload"])
        row["source"] = payload.get("source", "")
        row["model_version"] = payload.get("modelVersion")
    if strict:
        return known
    tracked = {r[0] for r in conn.execute("SELECT DISTINCT source_fetched_epoch FROM source_forecast_receipts WHERE source_table=? AND source_fetched_epoch>=? AND source_fetched_epoch<=?", (source_table, start_fetched_epoch, issue))} if present else set()
    columns = "fetched_epoch,source,payload" + (",model_version" if source_table == "air_quality_forecast_runs" else "")
    legacy = []
    for raw in conn.execute(f"SELECT {columns} FROM {source_table} WHERE fetched_epoch>=? AND fetched_epoch<=? ORDER BY fetched_epoch,payload", (start_fetched_epoch, issue)):
        fetched, source, payload = raw[:3]
        if fetched in tracked:
            continue
        legacy.append({"fetched_epoch": fetched, "source": source, "payload": decode_payload_text(payload),
            "model_version": raw[3] if len(raw) > 3 else None,
            "_provenance": {"sourceTable": source_table, "sourceFetchedEpoch": fetched,
                "receiptKnown": False, "receivedEpoch": None, "persistedEpoch": None,
                "availableEpoch": None, "receiptPolicy": "legacy_receipt_unknown"}})
    return sorted(known + legacy, key=lambda r: (r["fetched_epoch"], r["_provenance"].get("availableEpoch") or r["fetched_epoch"], r["_provenance"].get("receiptId") or 0))


def annotate_sensor_rows(conn, rows, issue_epoch):
    """Preserve compatibility values; known only when immutable evidence matches."""
    rows = [dict(r) for r in rows]
    if not rows:
        return []
    known = {r["epoch"]: r for r in load_sensor_asof(conn, issue_epoch,
             min(r["epoch"] for r in rows), max(r["epoch"] for r in rows))}
    output = []
    for row in rows:
        observed = known.get(row["epoch"])
        matches = observed is not None and all(observed.get(k) == v for k, v in row.items() if k != "_provenance")
        row["_provenance"] = (dict(observed["_provenance"]) if matches else
            {"observedEpoch": row["epoch"], "receiptKnown": False,
             "receivedEpoch": None, "persistedEpoch": None, "availableEpoch": None,
             "receiptPolicy": "legacy_or_unmatched_receipt_unknown"})
        if matches:
            row["_provenance"]["materializedValuesSha256"] = hashlib.sha256(
                _json({k: v for k, v in row.items() if k != "_provenance"}).encode("utf-8")).hexdigest()
        output.append(row)
    return output


def annotate_source_runs(conn, source_table, runs, issue_epoch):
    """Keep detached legacy source bytes while explicitly marking unknown clocks."""
    runs = [dict(run) for run in runs]
    if not runs:
        return []
    known = {r["fetched_epoch"]: r for r in load_source_asof(conn, source_table,
        issue_epoch, min(r["fetched_epoch"] for r in runs))}
    output = []
    for run in runs:
        observed = known.get(run["fetched_epoch"])
        matches = observed is not None and observed["payload"] == run["payload"]
        run["_provenance"] = (dict(observed["_provenance"]) if matches else
            {"sourceTable": source_table, "sourceFetchedEpoch": run["fetched_epoch"],
             "receiptKnown": False, "receivedEpoch": None, "persistedEpoch": None,
             "availableEpoch": None, "receiptPolicy": "legacy_or_unmatched_receipt_unknown"})
        output.append(run)
    return output


def build_input_manifest(sensor_rows, source_runs, forecast_epoch, *, metadata=None):
    """Hash actual detached inputs, retaining compact reconstruction references.

    Pass the rows used by inference and only the source runs it actually uses.
    Unknown legacy values are hashed, not certified as historically available.
    metadata must include view/filter/bucketing/feature policy when relevant.
    """
    issue = _epoch(forecast_epoch, "forecast_epoch")
    sensor_rows = sorted((dict(r) for r in sensor_rows), key=lambda r: r["epoch"])
    values_hash = hashlib.sha256()
    receipt_hash = hashlib.sha256()
    known = 0
    receipts = []
    for row in sensor_rows:
        data = {k: v for k, v in row.items() if k != "_provenance"}
        raw = _json(data).encode("utf-8")
        values_hash.update(len(raw).to_bytes(8, "big")); values_hash.update(raw)
        provenance = row.get("_provenance") or {}
        valid = (provenance.get("receiptKnown") is True and
                 isinstance(provenance.get("confirmedEpoch"), (int, float)) and
                 provenance["confirmedEpoch"] <= issue and
                 provenance.get("materializedValuesSha256") == hashlib.sha256(raw).hexdigest() and
                 isinstance(provenance.get("availableEpoch"), (int, float)) and
                 provenance["availableEpoch"] <= issue and data["epoch"] <= issue)
        known += int(valid)
        receipt_raw = _json(provenance).encode("utf-8")
        receipt_hash.update(len(receipt_raw).to_bytes(8, "big")); receipt_hash.update(receipt_raw)
        if valid:
            receipts.append(provenance["receiptId"])
    source_refs = []
    for run in source_runs:
        provenance = dict(run.get("_provenance") or {})
        text = run.get("payload")
        if not isinstance(text, str):
            raise TypeError("materialized source run must retain exact payload text")
        checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
        known_source = (provenance.get("receiptKnown") is True and
            isinstance(provenance.get("confirmedEpoch"), (int, float)) and
            provenance["confirmedEpoch"] <= issue and
            provenance.get("payloadSha256") == checksum and
            isinstance(provenance.get("availableEpoch"), (int, float)) and
            provenance["availableEpoch"] <= issue and run["fetched_epoch"] <= issue)
        source_refs.append({**provenance, "sourceFetchedEpoch": run["fetched_epoch"],
            "payloadSha256": checksum, "receiptKnown": known_source})
    source_refs.sort(key=lambda r: (str(r.get("sourceTable")), r["sourceFetchedEpoch"], r["payloadSha256"]))
    unknown = len(sensor_rows)-known
    source_unknown = sum(r["receiptKnown"] is not True for r in source_refs)
    has_inputs = bool(sensor_rows or source_refs)
    return {"version": VERSION, "forecastEpoch": issue,
        "sensorView": {"rowCount": len(sensor_rows), "knownReceiptRows": known,
            "unknownReceiptRows": unknown, "observedStartEpoch": sensor_rows[0]["epoch"] if sensor_rows else None,
            "observedEndEpoch": sensor_rows[-1]["epoch"] if sensor_rows else None,
            "receiptWatermarkId": max(receipts, default=0), "valuesSha256": values_hash.hexdigest(),
            "receiptSelectionSha256": receipt_hash.hexdigest(), "asOfEpoch": issue},
        "sourceRevisions": source_refs, "unknownReceiptInputs": unknown+source_unknown,
        "hasInputEvidence": has_inputs,
        "allReceiptsKnown": has_inputs and unknown+source_unknown == 0,
        "reconstructionComplete": has_inputs and unknown+source_unknown == 0,
        "availabilityPolicy": "received_inserted_and_post_commit_confirmed_at_or_before_issue_legacy_unknown",
        "metadata": dict(metadata or {})}


def record_input_snapshot(conn, manifest, *, clock=time.time):
    """Store a compact immutable manifest and return the publication linkage."""
    text = _json(manifest)
    snapshot = hashlib.sha256(text.encode("utf-8")).hexdigest()
    forecast = _epoch(manifest.get("forecastEpoch"), "forecast_epoch")
    persisted = _writer_clock(conn, clock)
    conn.execute("INSERT OR IGNORE INTO forecast_input_snapshots VALUES(?,?,?,?)",
                 (snapshot, forecast, persisted, text))
    stored = conn.execute("SELECT manifest_json,persisted_epoch FROM forecast_input_snapshots WHERE snapshot_id=?", (snapshot,)).fetchone()
    if stored[0] != text:
        raise ValueError("conflicting immutable input snapshot")
    return {"version": VERSION, "snapshotId": snapshot, "snapshotSha256": snapshot,
            "forecastEpoch": forecast, "persistedEpoch": stored[1],
            "allReceiptsKnown": manifest.get("allReceiptsKnown") is True,
            "unknownReceiptInputs": manifest.get("unknownReceiptInputs"),
            "reconstructionComplete": manifest.get("reconstructionComplete") is True}
