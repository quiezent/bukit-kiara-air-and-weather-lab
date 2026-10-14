"""Walk-forward replay of weather_session_forecast (Morning HGB 'basic31').

Production recipe: weather_session_forecast._trained_model (cutoff = issue-day
MYT midnight; pooled :00/:30 origins x leads 105..1380 min over 14 days; labels
complete < cutoff; HGB absolute-error on change from the fresh reference;
>= 120 rows, >= 10 target dates). Queries call the production predict_windows
with allow_fit=True on the snapshot DB, with the sensor rows visible at the
issue (harness clock) and the production Morning target (next 09:00-11:00 MYT
window starting >= issue + 90 min). Issues: canonical clocks 07:15 (same-day),
12:00 and 18:05 (next day).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config, prod, sources, targets

NAME = "morning_hgb"
CLOCKS = ("07:15", "12:00", "18:05")


def replay(ds: pd.DataFrame, days=None, log=print) -> pd.DataFrame:
    forecast = prod.module("weather_session_forecast")
    canonical = ds.loc[ds.canonical.isin(CLOCKS) & ds["eval"]]
    if days is not None:
        canonical = canonical.loc[[config.midnight(i) in set(days) for i in canonical.index]]
    records = []
    table = sources.sensor_table()
    for issue in canonical.index.to_numpy(np.int64):
        rows = sources.plain_rows(table.loc[(table.epoch >= issue - 17 * 86400) & (table.epoch <= issue)],
                                  visible_by=int(issue))
        start, end = targets.next_session(int(issue), "morning")
        try:
            result = forecast.predict_windows(config.SNAPSHOT_DB, rows, {"morning": {"startEpoch": start, "endEpoch": end}},
                                              int(issue), allow_fit=True)["morning"]
            mean = result.get("mean") if result.get("available") else np.nan
            reason = result.get("reason") or "available"
        except Exception as error:  # production returns unavailable on failure
            mean, reason = np.nan, f"{type(error).__name__}: {error}"
        records.append({"issue_epoch": int(issue), f"{NAME}__pred_morning": mean, f"{NAME}__start": start,
                        f"{NAME}__reason": reason})
        log(f"{NAME} {config.myt(issue):%Y-%m-%d %H:%M} -> {config.myt(start):%m-%d %H:%M} {mean}")
    return pd.DataFrame(records).set_index("issue_epoch")
