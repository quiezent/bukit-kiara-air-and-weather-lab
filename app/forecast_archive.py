"""Immutable snapshots of the final published forecast, not reconstructed fits."""
import hashlib
import json
from compact_forecast_record import compact_dashboard_record
from forecast_payload import encode_payload


def record_dashboard_issue(conn, result, build, actual_issued_epoch):
    if not result.get("available"):
        return False
    payload = compact_dashboard_record(result, build)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    issue_id = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    conn.execute("""CREATE TABLE IF NOT EXISTS dashboard_forecast_issues(
        issue_id TEXT PRIMARY KEY, issued_epoch INTEGER NOT NULL,
        forecast_epoch INTEGER NOT NULL, sensor_epoch INTEGER,
        dashboard_build TEXT NOT NULL, payload TEXT NOT NULL)""")
    # Both the prospective scorers and the short-horizon research models select
    # archived issues by their actual publication time. Without this index a
    # daily fit scans every large JSON snapshot in the archive.
    conn.execute("""CREATE INDEX IF NOT EXISTS dashboard_forecast_issues_issued_epoch
        ON dashboard_forecast_issues(issued_epoch)""")
    cursor = conn.execute("""INSERT OR IGNORE INTO dashboard_forecast_issues
        VALUES(?,?,?,?,?,?)""", (issue_id, int(actual_issued_epoch),
        int(result["forecastIssuedEpoch"]), (result.get("current") or {}).get("epoch"), build, encode_payload(encoded)))
    return bool(cursor.rowcount)
