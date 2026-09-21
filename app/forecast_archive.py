"""Immutable snapshots of the final published forecast, not reconstructed fits."""
import hashlib
import json


def record_dashboard_issue(conn, result, build, actual_issued_epoch):
    if not result.get("available"):
        return False
    keys = ("forecastIssuedEpoch", "current", "airWindow", "comparison", "forecastPolicy", "dataCoverage", "rainLearning")
    payload = {key: result.get(key) for key in keys}
    payload["windows"] = {
        key: {field: value.get(field) for field in
              ("startEpoch", "endEpoch", "forecastIssuedEpoch", "modeledSession", "particleForecast", "weatherForecast")}
        for key, value in result.get("windows", {}).items()
    }
    payload["dashboardBuild"] = build
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    issue_id = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    conn.execute("""CREATE TABLE IF NOT EXISTS dashboard_forecast_issues(
        issue_id TEXT PRIMARY KEY, issued_epoch INTEGER NOT NULL,
        forecast_epoch INTEGER NOT NULL, sensor_epoch INTEGER,
        dashboard_build TEXT NOT NULL, payload TEXT NOT NULL)""")
    cursor = conn.execute("""INSERT OR IGNORE INTO dashboard_forecast_issues
        VALUES(?,?,?,?,?,?)""", (issue_id, int(actual_issued_epoch),
        int(result["forecastIssuedEpoch"]), (result.get("current") or {}).get("epoch"), build, encoded))
    return bool(cursor.rowcount)
