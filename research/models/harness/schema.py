"""Write research/models/DATASET.md (the dataset schema) from the built dataset."""
from __future__ import annotations

from pathlib import Path

from . import config, dataset

PATH = Path(__file__).resolve().parents[1] / "DATASET.md"

TARGETS = {
    "y_ref": "Fresh 5-min reference at the issue (median of visible readings in (t-300 s, t], newest <= 240 s old). Persistence forecast for every concentration target.",
    "y_arrival": "+90 arrival: interpolated 15-min bucket-median proxy at exactly t+90 min (fresh_numeric_model_v4.targets_for_issues / arrival_change_model). NaN if either bucket is missing.",
    "y_arrival_complete": "Epoch when the y_arrival buckets have closed (label usable for training only after this).",
    "y_ride_mean / y_ride_min / y_ride_max": "Ride window [t+90, t+210]: overlap-weighted mean of 15-min bucket medians; min/max over buckets with positive overlap (partial edge buckets summarise their full 15 min). All NaN if any bucket is missing (targets_for_issues).",
    "y_ride_joint_valid": "All four Ridge heads finite (Ridge v4 trains only on these rows).",
    "y_ride_complete": "Epoch when the last ride bucket closes.",
    "y_delta": "y_arrival - y_ref.",
    "y_class5": "Production arrival classes: 0 delta<=-40, 1 (-40,-20], 2 (-20,20), 3 [20,40), 4 >=40; -1 = no label.",
    "y_fall20 / y_rise20 / y_fall40 / y_rise40": "Binary versions of y_class5 (fall20 = class 0 or 1, rise20 = 3 or 4, fall40 = 0, rise40 = 4); NaN without label.",
    "y_ride_le70": "1 if y_ride_mean <= 70 (cycling_window_probability.target_label).",
    "y_first20": "First crossing within 90 min: first of the six +15..+90 proxies with |proxy - y_ref| >= 20; -1 drop, +1 rise, 0 none (fresh_event_model labels).",
    "y_first20_step / y_first20_complete": "Index (0..5) of the first crossing step; epoch when all six proxies are complete.",
    "y_morning_* / y_afternoon_*": "Next feasible session (production next_feasible_ride_period): Morning 09:00-11:00, Afternoon 14:00-16:00 MYT, first window starting >= t+90 min. _mean = mean of its 8 closed 15-min bucket medians (window_pm_predictor._session_actual); _start, _complete (= window end), _lead_min, _same_day.",
}

GROUP_TEXT = {
    "meta": "Issue metadata: m_fresh_available (production freshness rule), m_fresh_ref_epoch, m_sensor_watermark (newest visible observation), m_input_rows_180min, m_receipts_all_known (every visible input row in the 180-min window has a real receipt clock; False while any pre-ledger row is used, i.e. before about 2026-10-08 16:12).",
    "sensor": "fresh_sensor_features.FEATURE_COLUMNS computed by the production function on rows visible at the issue: freshPm25, raw lags/deltas 5..60 min, SDs, ranges, slopes, fresh temperature/RH and 30-min deltas, sample counts, time and issue-offset cycles, and 3-min sequence bins of PM2.5/temperature/RH over 180 min (seq_<channel>_<age>_<age+3>min). The 28 non-sequence columns are the arrival logistic's inputs.",
    "time": "MYT hour, day cycle, day of week, weekend, days since grid start.",
    "weather": "Latest Open-Meteo vintage visible at the issue (age <= 2 h): point values at +0..+360 min (w_<field>_<lead>, precipitation/showers = right-labelled hourly amount), wind u/v at 10 m/180 m/925 hPa, ride-window and next Morning/Afternoon session means and precipitation sums, prior/next 3 h precipitation, max precipitation probability in 6 h, first lead with probability >= 50, and 60-min vintage revisions at +90 (same as production arrival_feature_inputs).",
    "cams": "Latest CAMS (Open-Meteo, cams_anchor_v1/v2) run visible at the issue (age <= 3 h): PM2.5/PM10/AOD/dust at +0/+90/+180/+360, +90 change, ride-window and session means.",
    "regional_pm": "CAMS 3x3 regional grid (2026-10-01 onward): mean, N-S and E-W gradients and max at +0/+90/+180.",
    "regional_wind": "GFS 925/850 hPa winds (2026-09-05 onward): u/v nearest TTDI at +0/+90 and 925 hPa domain-mean u/v.",
    "subang": "Latest Subang airport report visible (first-fetch clock, age <= 3 h): temperature, RH, dew point, visibility, wind speed/u/v, rain/thunder/haze flags, precipitation.",
    "neighbors_strict": "Seven AirGradient stations; value visible by its receipt clock (first request that returned it). The archive starts with a backfill received at 2026-10-05 19:30 MYT, so this group is empty before 19:35 that day.",
    "neighbors_reconstructed": "Same stations, visibility = min(receipt, bucket end + 30 min). An assumption about AirGradient cloud latency; not a proven clock.",
}


def write():
    meta = dataset.meta()
    lines = ["# Dataset schema", "",
             f"`{config.DATASET_PATH}` (parquet, index `issue_epoch` = UTC seconds). One row per 5-min issue clock "
             f"from {meta['gridStartMYT']} to {meta['gridEndMYT']} MYT ({meta['rows']} rows, {meta['columns']} columns). "
             f"`eval` marks issues from {meta['evalStartMYT'][:10]} (14 full days of sensor and weather history before it).",
             "", "Row columns: `issue_myt`, `date_myt`, `canonical` ('07:15', '12:00', '18:05' or ''), `eval`.", "",
             "## Availability rule", "",
             "Every feature value was visible at the issue time:",
             f"- sensor readings with a receipt (from 2026-10-08 13:12): max(available, confirmed) of the first receipt; older readings: observation + {meta['legacySensorVisibilityLagSeconds']} s (measured receipt latency p99 = 60 s; on post-ledger grid issues this rule never admits a reading before its real receipt, see `harness.audit`; production itself assumes 0 s);",
             f"- weather / CAMS runs: receipt clock when present, else fetched + {meta['legacySourceVisibilityLagSeconds']} s (receipts lag fetch by <= 5 s);",
             "- regional PM: recorded_epoch; regional wind: fetched + 5 s; Subang: first_fetched_epoch;",
             f"- neighbours: receipt clock (strict group) or bucket end + {meta['neighborReconstructedLagSeconds'] // 60} min (reconstructed group).",
             "", "## Targets (production definitions)", ""]
    lines += [f"- `{name}`: {text}" for name, text in TARGETS.items()]
    lines += ["", "## Feature groups", ""]
    for name, columns in meta["groups"].items():
        lines.append(f"### {name} ({len(columns)} columns)")
        lines.append("")
        lines.append(GROUP_TEXT.get(name, ""))
        lines.append("")
        if name != "sensor":
            lines.append("`" + "`, `".join(columns) + "`")
            lines.append("")
    PATH.write_text("\n".join(lines), encoding="utf-8")
    return PATH


if __name__ == "__main__":
    print(write())
