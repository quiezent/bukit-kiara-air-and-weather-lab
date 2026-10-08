"""Read-only, as-issued PM2.5 backtest for the TTDI forecast archive.

This scores stored dashboard and session issue payloads. It never refits a model
or replays today's code against old observations. Observations are medians of
right-closed 15-minute buckets, with the production coverage rule and no fill.

Run ``python forecast_backtest.py --help`` for sampling and output controls.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path
import sqlite3
from statistics import median, mean
from typing import Any
from forecast_payload import loads as archive_loads


BUCKET = 900
LOCAL = timezone(timedelta(hours=8))
TARGET_VERSION = "decision_clock_weighted_15min_v1"
DEFAULT_DB = Path(__file__).with_name("bukit_kiara_air_history.db")


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def closed_label(epoch: int) -> int:
    """A sample exactly on a boundary belongs to the bucket ending there."""
    return ((int(epoch) - 1) // BUCKET + 1) * BUCKET


def covered_bucket_medians(readings: list[tuple[int, float]]) -> dict[int, float]:
    grouped: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for epoch, value in readings:
        value = finite(value)
        if value is not None:
            grouped[closed_label(epoch)].append((int(epoch), value))
    medians = {}
    for label, samples in grouped.items():
        samples.sort()
        epochs = [epoch for epoch, _ in samples]
        if (len(samples) < 3 or epochs[0] - (label - BUCKET) > 240
                or label - epochs[-1] > 240
                or any(right - left > 480 for left, right in zip(epochs, epochs[1:]))):
            continue
        medians[label] = float(median(value for _, value in samples))
    return medians


def point_weights(epoch: int) -> dict[int, float]:
    lower = (int(epoch) // BUCKET) * BUCKET
    fraction = (int(epoch) - lower) / BUCKET
    return {lower: 1.0} if fraction == 0 else {lower: 1 - fraction, lower + BUCKET: fraction}


def window_weights(start: int, end: int) -> dict[int, float]:
    if end <= start:
        raise ValueError("window end must follow start")
    first = start // BUCKET + 1
    last = (end + BUCKET - 1) // BUCKET
    return {
        label * BUCKET: (min(end, label * BUCKET) - max(start, (label - 1) * BUCKET)) / (end - start)
        for label in range(first, last + 1)
    }


def weighted_target(buckets: dict[int, float], weights: dict[int, float], as_of: int) -> float | None:
    if not weights or any(label > as_of or label not in buckets for label in weights):
        return None
    return sum(buckets[label] * weight for label, weight in weights.items())


def score_target(buckets: dict[int, float], kind: str, start: int, end: int | None, as_of: int) -> float | None:
    weights = point_weights(start) if kind == "arrival" else window_weights(start, int(end))
    return weighted_target(buckets, weights, as_of)


def rapid_episodes(buckets: dict[int, float], threshold: float, quiet_minutes: int = 90) -> list[dict[str, Any]]:
    """First onset in each same-direction cluster of complete 15-minute changes."""
    candidates = []
    for label in sorted(buckets):
        previous = buckets.get(label - BUCKET)
        if previous is None:
            continue
        change = buckets[label] - previous
        if abs(change) >= threshold:
            candidates.append({"onset_epoch": label - BUCKET, "label_epoch": label,
                               "change": change, "direction": "drop" if change < 0 else "rise"})
    episodes = []
    for event in candidates:
        if (episodes and event["direction"] == episodes[-1]["direction"]
                and event["onset_epoch"] - episodes[-1]["last_onset_epoch"] < quiet_minutes * 60):
            episodes[-1]["last_onset_epoch"] = event["onset_epoch"]
            if abs(event["change"]) > abs(episodes[-1]["largest_15min_change"]):
                episodes[-1]["largest_15min_change"] = event["change"]
            continue
        episodes.append({"onset_epoch": event["onset_epoch"],
                         "last_onset_epoch": event["onset_epoch"],
                         "direction": event["direction"],
                         "largest_15min_change": event["change"]})
    return episodes


def local_date(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, LOCAL).date().isoformat()


def iso_local(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, LOCAL).isoformat(timespec="seconds")


def choose_spaced(rows: list[tuple], spacing_seconds: int) -> list[tuple]:
    """One first completed issue per elapsed interval, globally across builds."""
    selected = []
    last = -10**30
    for row in sorted(rows, key=lambda row: (row[1], row[0])):
        if row[1] >= last + spacing_seconds:
            selected.append(row)
            last = row[1]
    return selected


def choose_daily_clock(rows: list[tuple], clock_minute: int, tolerance_seconds: int,
                       epoch_index: int = 1, group_key=None) -> list[tuple]:
    """Closest issue to the declared local decision clock for each group/day."""
    selected = {}
    for row in rows:
        epoch = int(row[epoch_index])
        local = datetime.fromtimestamp(epoch, LOCAL)
        minute = local.hour * 60 + local.minute + local.second / 60
        distance = abs(minute - clock_minute) * 60
        if distance > tolerance_seconds:
            continue
        group = group_key(row) if group_key else None
        key = (local.date().isoformat(), group)
        rank = (distance, epoch, str(row[0]))
        if key not in selected or rank < selected[key][0]:
            selected[key] = (rank, row)
    return [item[1] for _, item in sorted(selected.items())]


def nearest_preonset(rows: list[tuple], onset: int, lookback_seconds: int) -> tuple | None:
    """Rows are sorted by actual recording epoch; never select after onset."""
    from bisect import bisect_left
    epochs = [int(row[1]) for row in rows]
    position = bisect_left(epochs, onset) - 1
    return rows[position] if position >= 0 and epochs[position] >= onset - lookback_seconds else None


def _number(*values):
    for value in values:
        number = finite(value)
        if number is not None:
            return number
    return None


def _int(value):
    number = finite(value)
    return int(number) if number is not None else None


def extract_dashboard(row: tuple, payload: dict, scope: str) -> list[dict[str, Any]]:
    issue_id, recorded, forecast_epoch, build, _ = row
    if _int(payload.get("forecastIssuedEpoch")) != int(forecast_epoch) or forecast_epoch > recorded:
        return []
    policy_value = payload.get("forecastPolicy") or "unknown"
    policy = str(policy_value.get("id") or "unknown") if isinstance(policy_value, dict) else str(policy_value)
    results = []
    if scope in ("regular", "rapid"):
        air = payload.get("airWindow") or {}
        clock = air.get("forecastClock") or {}
        if air.get("available") and clock.get("targetVersion") == TARGET_VERSION:
            start = _int(clock.get("arrivalTargetEpoch"))
            end = _int(clock.get("windowEndEpoch"))
            if (start is not None and end is not None and
                    start == _int(clock.get("windowStartEpoch")) and start > recorded and end > start):
                for horizon, data, kind in (("arrival90", air.get("arrival") or {}, "arrival"),
                                            ("mean90to210", air.get("trail") or {}, "mean")):
                    if not data.get("available", True):
                        continue
                    point = _number(data.get("point"))
                    anchor_role = str(data.get("persistenceAnchorRole") or "")
                    if data.get("pointRole") == "persistence_anchor":
                        inferred_baseline = point
                    elif "closed_15_minute" in anchor_role:
                        inferred_baseline = _number(air.get("current15"))
                    else:
                        inferred_baseline = _number(air.get("currentPartial15"), air.get("current15"))
                    baseline = _number(data.get("baselinePoint"), inferred_baseline)
                    if point is None or baseline is None:
                        continue
                    results.append({"source": "dashboard", "scope": scope, "issue_id": issue_id,
                                    "recorded_epoch": recorded, "forecast_epoch": forecast_epoch,
                                    "build": build, "policy": policy,
                                    "model": str(data.get("modelVersion") or data.get("method") or data.get("pointRole") or "unknown"),
                                    "horizon": horizon, "kind": kind, "start_epoch": start,
                                    "end_epoch": end if kind == "mean" else None,
                                    "point": point, "baseline": baseline,
                                    "low": _number(data.get("rangeLow")), "high": _number(data.get("rangeHigh"))})
    elif scope == "dashboard_session":
        for name, window in (payload.get("windows") or {}).items():
            prediction = window.get("particleForecast") or {}
            start, end = _int(window.get("startEpoch")), _int(window.get("endEpoch"))
            point = _number(prediction.get("point"), prediction.get("mean"))
            baseline = _number(prediction.get("baselinePoint"), prediction.get("sensorAnchor"),
                               (payload.get("airWindow") or {}).get("currentPartial15"),
                               (payload.get("airWindow") or {}).get("current15"))
            if (prediction.get("available") and point is not None and baseline is not None
                    and start is not None and end is not None and start > recorded and end > start):
                results.append({"source": "dashboard", "scope": scope, "issue_id": issue_id,
                                "recorded_epoch": recorded, "forecast_epoch": forecast_epoch,
                                "build": build, "policy": policy,
                                "model": str(prediction.get("modelVersion") or "unknown"),
                                "horizon": str(name), "kind": "mean", "start_epoch": start,
                                "end_epoch": end, "point": point, "baseline": baseline,
                                "low": _number(prediction.get("rangeLow"), prediction.get("rawRangeLow")),
                                "high": _number(prediction.get("rangeHigh"), prediction.get("rawRangeHigh"))})
    return results


def extract_session_archive(row: tuple, payload: dict) -> dict[str, Any] | None:
    version, origin, start, end, issued, window_key, point, baseline, _ = row
    payload_issue = _int(payload.get("forecastedAtEpoch"))
    if (int(issued) >= int(start) or int(end) <= int(start)
            or _int(payload.get("startEpoch")) != int(start)
            or _int(payload.get("endEpoch")) != int(end)
            or (payload_issue is not None and payload_issue > int(issued))
            or (payload.get("originEpoch") is not None and _int(payload.get("originEpoch")) != int(origin))
            or (payload.get("modelVersion") is not None and payload.get("modelVersion") != version)):
        return None
    return {"source": "session_archive", "scope": "session_archive", "issue_id": f"{version}:{origin}:{start}:{end}",
            "recorded_epoch": int(issued), "forecast_epoch": payload_issue or int(issued),
            "build": "archive", "policy": str(payload.get("pointRole") or "unknown"),
            "model": str(version), "horizon": str(window_key), "kind": "mean",
            "start_epoch": int(start), "end_epoch": int(end),
            "point": finite(point), "baseline": finite(baseline),
            "low": _number(payload.get("rangeLow"), payload.get("rawRangeLow")),
            "high": _number(payload.get("rangeHigh"), payload.get("rawRangeHigh"))}


def score_forecasts(forecasts: list[dict], buckets: dict[int, float], as_of: int) -> tuple[list[dict], Counter]:
    scored = []
    excluded = Counter()
    for forecast in forecasts:
        point, baseline = forecast["point"], forecast["baseline"]
        if point is None or baseline is None:
            excluded["missing_point_or_baseline"] += 1
            continue
        actual = score_target(buckets, forecast["kind"], forecast["start_epoch"], forecast["end_epoch"], as_of)
        if actual is None:
            excluded["incomplete_or_unclosed_target"] += 1
            continue
        low, high = forecast["low"], forecast["high"]
        timing = {}
        onset = forecast.get("event_onset_epoch")
        if onset is not None:
            # Recording time is when the forecast was available, even if its
            # internal calculation clock was earlier. Targets remain unchanged.
            before_onset = forecast["recorded_epoch"] < onset
            timing = {
                "event_timing_role": "pre_onset_target_forecast" if before_onset else "continuation_forecast",
                "recorded_seconds_relative_to_onset": forecast["recorded_epoch"] - onset,
            }
        scored.append({**forecast, **timing, "actual": actual, "error": point - actual,
                       "baseline_error": baseline - actual,
                       "predicted_change": point - baseline,
                       "actual_change": actual - baseline,
                       "range_hit": None if low is None or high is None or low > high else low <= actual <= high})
    return scored, excluded


def summarize(rows: list[dict], change_threshold: float) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    errors = sorted(abs(row["error"]) for row in rows)
    baseline_errors = [abs(row["baseline_error"]) for row in rows]
    mae = mean(errors)
    baseline_mae = mean(baseline_errors)
    ranges = [row["range_hit"] for row in rows if row["range_hit"] is not None]
    actual_change = [row for row in rows if abs(row["actual_change"]) >= change_threshold]
    flagged = [row for row in rows if abs(row["predicted_change"]) >= change_threshold]
    true_flag = sum(abs(row["actual_change"]) >= change_threshold for row in flagged)
    def direction(change):
        return "drop" if change <= -change_threshold else "rise" if change >= change_threshold else "stable"

    directions = ("drop", "stable", "rise")
    confusion = {actual: {predicted: 0 for predicted in directions} for actual in directions}
    for row in rows:
        confusion[direction(row["actual_change"])][direction(row["predicted_change"])] += 1
    directional = {}
    for label in ("drop", "rise"):
        actual_count = sum(confusion[label].values())
        predicted_count = sum(confusion[actual][label] for actual in directions)
        hits = confusion[label][label]
        directional[label] = {
            "actual_count": actual_count, "predicted_count": predicted_count,
            "correct_sign_hits": hits, "false_alarms": predicted_count - hits,
            "misses": actual_count - hits,
            "precision": round(hits / predicted_count, 3) if predicted_count else None,
            "recall": round(hits / actual_count, 3) if actual_count else None,
        }
    correct_sign_hits = sum(directional[label]["correct_sign_hits"] for label in ("drop", "rise"))
    return {"count": len(rows), "days": len({local_date(row["recorded_epoch"]) for row in rows}),
            "mae": round(mae, 3), "persistence_mae": round(baseline_mae, 3),
            "mae_gain_pct": round(100 * (baseline_mae - mae) / baseline_mae, 2) if baseline_mae else None,
            "bias": round(mean(row["error"] for row in rows), 3),
            "p90_abs_error": round(errors[max(0, math.ceil(0.9 * len(errors)) - 1)], 3),
            "win_fraction": round(sum(abs(row["error"]) < abs(row["baseline_error"]) for row in rows) / len(rows), 3),
            "range_count": len(ranges), "range_hit_fraction": round(mean(ranges), 3) if ranges else None,
            "large_actual_change_count": len(actual_change), "large_predicted_change_count": len(flagged),
            "large_change_hits": true_flag,
            "large_change_recall": round(true_flag / len(actual_change), 3) if actual_change else None,
            "large_change_false_alarms": len(flagged) - true_flag,
            "legacy_large_change_metrics_definition": "Direction-agnostic magnitude detection; an opposite-sign forecast can count as a hit.",
            "large_change_correct_sign_hits": correct_sign_hits,
            "large_change_correct_sign_false_alarms": len(flagged) - correct_sign_hits,
            "large_change_correct_sign_precision": round(correct_sign_hits / len(flagged), 3) if flagged else None,
            "large_change_correct_sign_recall": round(correct_sign_hits / len(actual_change), 3) if actual_change else None,
            "large_change_wrong_sign_count": confusion["drop"]["rise"] + confusion["rise"]["drop"],
            "large_change_by_direction": directional,
            "direction_confusion": confusion,
            "direction_confusion_definition": "Rows are actual direction, columns predicted direction; drop <= -threshold, rise >= threshold, stable otherwise.",
            "event_timing_counts": dict(Counter(row.get("event_timing_role", "unclassified") for row in rows))}


def grouped_summary(rows: list[dict], threshold: float) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        key = (row["source"], row["scope"], row["horizon"], row["build"], row["policy"], row["model"])
        grouped[key].append(row)
    output = []
    for key, items in sorted(grouped.items()):
        output.append({"source": key[0], "scope": key[1], "horizon": key[2],
                       "build": key[3], "policy": key[4], "model": key[5],
                       **summarize(items, threshold)})
    return output


def _dashboard_metadata(conn: sqlite3.Connection) -> list[tuple]:
    return [
        (row[0], int(row[1]), int(row[2]), row[3], None)
        for row in conn.execute("SELECT issue_id,issued_epoch,forecast_epoch,dashboard_build FROM dashboard_forecast_issues ORDER BY issued_epoch,issue_id")]


def _fetch_dashboard(conn: sqlite3.Connection, rows: list[tuple]) -> dict[str, dict]:
    payloads = {}
    for issue_id in {row[0] for row in rows}:
        raw = conn.execute("SELECT payload FROM dashboard_forecast_issues WHERE issue_id=?", (issue_id,)).fetchone()
        if raw is not None:
            try:
                payloads[issue_id] = archive_loads(raw[0])
            except (TypeError, ValueError):
                pass
    return payloads


def run_backtest(db_path: Path, spacing_minutes: int = 210, rapid_threshold: float = 20,
                 rapid_lookback_minutes: int = 30, session_clock: str = "07:30",
                 session_tolerance_minutes: int = 30) -> dict[str, Any]:
    hour, minute = (int(part) for part in session_clock.split(":"))
    if not 0 <= hour < 24 or not 0 <= minute < 60:
        raise ValueError("session clock must be HH:MM")
    if spacing_minutes <= 0 or rapid_threshold <= 0 or rapid_lookback_minutes <= 0 or session_tolerance_minutes < 0:
        raise ValueError("spacing and rapid thresholds must be positive")
    uri = db_path.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True, timeout=30)) as conn:
        conn.execute("PRAGMA query_only=ON")
        conn.execute("BEGIN")
        readings = [(int(epoch), pm) for epoch, pm in conn.execute("SELECT epoch,pm02 FROM readings ORDER BY epoch")]
        as_of = max((epoch for epoch, _ in readings), default=0)
        buckets = covered_bucket_medians(readings)
        metadata = _dashboard_metadata(conn)
        regular = choose_spaced(metadata, spacing_minutes * 60)
        daily = choose_daily_clock(metadata, hour * 60 + minute, session_tolerance_minutes * 60)
        episodes = rapid_episodes(buckets, rapid_threshold)
        event_rows = [(event, nearest_preonset(metadata, event["onset_epoch"], rapid_lookback_minutes * 60))
                      for event in episodes]
        selected = regular + daily + [row for _, row in event_rows if row is not None]
        payloads = _fetch_dashboard(conn, selected)
        extracted = []
        for row in regular:
            if row[0] in payloads:
                extracted.extend(extract_dashboard(row, payloads[row[0]], "regular"))
        for row in daily:
            if row[0] in payloads:
                extracted.extend(extract_dashboard(row, payloads[row[0]], "dashboard_session"))
        for event, row in event_rows:
            if row is not None and row[0] in payloads:
                forecasts = extract_dashboard(row, payloads[row[0]], "rapid")
                for forecast in forecasts:
                    forecast["event_onset_epoch"] = event["onset_epoch"]
                    forecast["event_direction"] = event["direction"]
                    forecast["event_largest_15min_change"] = event["largest_15min_change"]
                extracted.extend(forecasts)
        archive_meta = list(conn.execute("SELECT model_version,origin_epoch,start_epoch,end_epoch,issued_epoch,window_key,predicted_mean,anchor_pm25,NULL FROM window_pm_forecast_issues"))
        archive_selected = choose_daily_clock(
            archive_meta, hour * 60 + minute, session_tolerance_minutes * 60,
            epoch_index=4, group_key=lambda row: (row[0], row[5], row[2]))
        for row in archive_selected:
            raw = conn.execute("SELECT payload FROM window_pm_forecast_issues WHERE model_version=? AND origin_epoch=? AND start_epoch=? AND end_epoch=?", row[:4]).fetchone()
            if raw:
                try:
                    forecast = extract_session_archive(row, json.loads(raw[0]))
                except (TypeError, ValueError):
                    forecast = None
                if forecast:
                    extracted.append(forecast)
    # Regular/session issues during a known rapid-change interval are reactive
    # continuation checks, not evidence that the model anticipated its onset.
    for forecast in extracted:
        if "event_onset_epoch" in forecast:
            continue
        active = next((event for event in reversed(episodes)
                       if event["onset_epoch"] <= forecast["recorded_epoch"] < event["last_onset_epoch"] + BUCKET), None)
        if active is not None:
            forecast.update(event_onset_epoch=active["onset_epoch"],
                            event_direction=active["direction"],
                            event_largest_15min_change=active["largest_15min_change"])
    scored, excluded = score_forecasts(extracted, buckets, as_of)
    by_scope = {scope: summarize([row for row in scored if row["scope"] == scope], rapid_threshold)
                for scope in ("regular", "rapid", "dashboard_session", "session_archive")}
    by_horizon = [{"scope": scope, "horizon": horizon,
                   **summarize([row for row in scored if row["scope"] == scope and row["horizon"] == horizon], rapid_threshold)}
                  for scope, horizon in sorted({(row["scope"], row["horizon"]) for row in scored})]
    rapid_details = []
    for event, row in event_rows:
        event_scored = [item for item in scored if item["scope"] == "rapid" and item.get("event_onset_epoch") == event["onset_epoch"]]
        rapid_details.append({"onset_local": iso_local(event["onset_epoch"]),
                              "direction": event["direction"],
                              "largest_15min_change": round(event["largest_15min_change"], 1),
                              "pre_onset_issue_local": iso_local(row[1]) if row else None,
                              "scored": [{"horizon": item["horizon"], "build": item["build"],
                                          "point": round(item["point"], 2), "persistence": round(item["baseline"], 2),
                                          "actual": round(item["actual"], 2), "error": round(item["error"], 2),
                                          "event_timing_role": item["event_timing_role"],
                                          "range_hit": item["range_hit"]} for item in event_scored]})
    return {"method": {"database": str(db_path.resolve()), "read_only": True,
                       "as_of_latest_reading_local": iso_local(as_of) if as_of else None,
                       "target": "as-issued point versus complete coverage-qualified right-closed 15-minute medians",
                       "regular_spacing_minutes": spacing_minutes,
                       "session_issue_clock_local": session_clock,
                       "session_tolerance_minutes": session_tolerance_minutes,
                       "rapid_definition": f"absolute 15-minute median change >= {rapid_threshold} ug/m3; consecutive buckets only; same-direction episodes separated by 90 minutes",
                       "rapid_onset_semantics": "Onset is the start of the first changed 15-minute bucket, a bucket-scale proxy rather than measured instantaneous onset.",
                       "event_timing_semantics": "Recording time before bucket-proxy onset is a pre-onset target forecast, not a prediction of onset time. Issues during the observed rapid-change interval are continuation checks, not anticipation.",
                       "rapid_pre_onset_lookback_minutes": rapid_lookback_minutes,
                       "large_target_change_threshold_ug_m3": rapid_threshold,
                       "range_note": "Stored ranges may be experimental and uncalibrated; range hit is descriptive."},
            "counts": {"readings": len(readings), "covered_buckets": len(buckets),
                       "dashboard_issues": len(metadata), "regular_issues_selected": len(regular),
                       "daily_dashboard_issues_selected": len(daily),
                       "session_archive_issues_selected": len(archive_selected),
                       "rapid_episodes_observed": len(episodes),
                       "rapid_episodes_with_pre_onset_issue": sum(row is not None for _, row in event_rows),
                       "forecasts_extracted": len(extracted), "forecasts_scored": len(scored),
                       "excluded": dict(excluded)},
            "summary_by_scope": by_scope, "summary_by_horizon": by_horizon,
            "groups": grouped_summary(scored, rapid_threshold),
            "rapid_episodes": rapid_details}


def print_report(result: dict[str, Any]) -> None:
    method, counts = result["method"], result["counts"]
    print(f"As-issued TTDI forecast backtest | latest reading {method['as_of_latest_reading_local']}")
    print(f"Coverage-qualified 15-minute medians: {counts['covered_buckets']} from {counts['readings']} readings; "
          f"dashboard issues: {counts['dashboard_issues']}")
    print(f"Selected issues: regular {counts['regular_issues_selected']} at >= {method['regular_spacing_minutes']} min; "
          f"daily dashboard {counts['daily_dashboard_issues_selected']}; "
          f"session archive {counts['session_archive_issues_selected']} at {method['session_issue_clock_local']} local")
    print(f"Rapid episodes: {counts['rapid_episodes_observed']} observed, "
          f"{counts['rapid_episodes_with_pre_onset_issue']} with pre-onset issue; "
          f"scored forecasts {counts['forecasts_scored']}; excluded {counts['excluded']}")
    print("\nScope                  n days  MAE   persist  gain%  range n/hit  large actual/pred/correct-sign/false")
    for name, summary in result["summary_by_scope"].items():
        print(f"{name:21} {summary.get('count', 0):3} {summary.get('days', 0):4} "
              f"{str(summary.get('mae', '-')):>5} {str(summary.get('persistence_mae', '-')):>8} "
              f"{str(summary.get('mae_gain_pct', '-')):>6} "
              f"{summary.get('range_count', 0):>4}/{str(summary.get('range_hit_fraction', '-')):<5} "
              f"{summary.get('large_actual_change_count', 0):>3}/"
              f"{summary.get('large_predicted_change_count', 0):>3}/"
              f"{summary.get('large_change_correct_sign_hits', 0):>3}/"
              f"{summary.get('large_change_correct_sign_false_alarms', 0):>3}")
        for direction, metrics in summary.get("large_change_by_direction", {}).items():
            print(f"  {direction}: correct-sign {metrics['correct_sign_hits']}/{metrics['actual_count']} actual; "
                  f"precision={metrics['precision']} recall={metrics['recall']} false={metrics['false_alarms']}")
        if summary.get("count"):
            print(f"  event timing: {summary['event_timing_counts']}")
    print("\nBy horizon:")
    for summary in result["summary_by_horizon"]:
        print(f"{summary['scope']}/{summary['horizon']}: n={summary['count']} "
              f"MAE={summary.get('mae')} persistence={summary.get('persistence_mae')} "
              f"range={summary.get('range_hit_fraction')} "
              f"correct-sign large-change hits={summary.get('large_change_correct_sign_hits')}/{summary.get('large_actual_change_count')}")
    print("Session archive totals pool distinct experimental models; compare their separate group rows below.")
    print("\nGroups (source/scope/horizon/build/policy/model):")
    for group in result["groups"]:
        print(f"{group['source']} | {group['scope']} | {group['horizon']} | {group['build']} | "
              f"{group['policy']} | {group['model']} : n={group['count']} "
              f"MAE={group.get('mae')} persistence={group.get('persistence_mae')} "
              f"range={group.get('range_hit_fraction')}")
    print("\nRapid episodes (pre-onset issue scored at its actual target times):")
    print(method["event_timing_semantics"])
    for event in result["rapid_episodes"]:
        horizons = ", ".join(f"{x['horizon']} err={x['error']:+.1f} range={x['range_hit']}"
                             for x in event["scored"])
        print(f"{event['onset_local']} {event['direction']} "
              f"{event['largest_15min_change']:+.1f}; issue {event['pre_onset_issue_local'] or '-'}; {horizons or 'no complete target'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="SQLite history database")
    parser.add_argument("--spacing-minutes", type=int, default=210)
    parser.add_argument("--rapid-threshold", type=float, default=20, help="15-minute and target-change threshold in ug/m3")
    parser.add_argument("--rapid-lookback-minutes", type=int, default=30)
    parser.add_argument("--session-clock", default="07:30", help="Daily as-issued local decision time HH:MM")
    parser.add_argument("--session-tolerance-minutes", type=int, default=30)
    parser.add_argument("--json", action="store_true", help="Write full machine-readable report to stdout")
    args = parser.parse_args()
    result = run_backtest(args.db, args.spacing_minutes, args.rapid_threshold,
                          args.rapid_lookback_minutes, args.session_clock, args.session_tolerance_minutes)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    else:
        print_report(result)


if __name__ == "__main__":
    main()
