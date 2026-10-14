"""Session analogs: Morning 09:00-11:00 and Afternoon 14:00-16:00 means (production session definitions).

Fixed config, chosen a priori and NOT tuned (canonical-clock samples are 16-18
per head per split): for a query at clock c, the library is every past issue
within +-60 min of c's time of day whose next session has the same same-day /
next-day status and whose session label closed before the day's cutoff.
Distance = mean squared standardised difference over level, 3 h trajectory
shape, issued session-weather means and the CAMS session ratio (equal block
weights). K = 25 nearest, uniform weights, ratio transfer to the current
reference, weighted median.

    python -B -m analogs.sessions        # writes data/models/analogs/sessions_pred.parquet and prints MAE tables
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from harness import config, dataset, report, scoring, walkforward

from . import analog, common

K = 25
WINDOW_MIN = 60
BLOCK_WEIGHTS = {"level": 1.0, "traj": 1.0, "wx": 1.0, "cams": 1.0}
SESSION_START = {"morning": 9 * 60, "afternoon": 14 * 60}


def _minute_of_day(index) -> np.ndarray:
    local = pd.to_datetime(np.asarray(index, np.int64), unit="s", utc=True).tz_convert("Asia/Kuala_Lumpur")
    return np.asarray(local.hour * 60 + local.minute)


def session_frame(ds: pd.DataFrame, session: str) -> pd.DataFrame:
    ref = ds.s_freshPm25.where(ds.m_fresh_available).to_numpy(float)
    base = analog.analog_frame(ds)
    out = {"b_level__logref": base["a_level__logref"]}
    for c in [c for c in base.columns if c.startswith("a_traj__")]:
        out["b_traj__" + c.split("__")[1]] = base[c]
    s = session
    out[f"b_wx__temp"] = ds[f"w_{s}_temp_mean"]
    out[f"b_wx__rh"] = ds[f"w_{s}_rh_mean"]
    out[f"b_wx__pprob"] = ds[f"w_{s}_pprob_mean"] / 100
    out[f"b_wx__wspd"] = ds[f"w_{s}_wspd10_mean"]
    out[f"b_wx__u"] = ds[f"w_{s}_wind10_u_mean"]
    out[f"b_wx__v"] = ds[f"w_{s}_wind10_v_mean"]
    out[f"b_wx__logblh"] = np.log(ds[f"w_{s}_blh_mean"].clip(lower=10))
    out[f"b_wx__cloud"] = ds[f"w_{s}_cloud_mean"] / 100
    out[f"b_wx__precip"] = np.log1p(ds[f"w_{s}_precip_sum"])
    out["b_cams__session_ratio"] = analog._lr(ds[f"c_pm25_{s}_mean"], ds.c_pm25_0)
    out["b_cams__gap"] = analog._lr(ds.c_pm25_0, ref)
    frame = pd.DataFrame({k: np.asarray(v, float) for k, v in out.items()}, index=ds.index)
    minute = _minute_of_day(ds.index)
    frame["b_minute"] = minute
    # same-day status of the next feasible session is a function of the issue clock only
    frame["b_same_day"] = (minute + 90 <= SESSION_START[s]).astype(float)
    keep = ["eval", "m_fresh_available", "s_freshPm25"] + [c for c in ds.columns if c.startswith("y_")]
    return ds[keep].join(frame)


def make_fit_predict(session: str):
    def fit_predict(train, test, cutoff):
        cols = [c for c in train.columns if c.startswith("b_") and "__" in c]
        blocks = np.array([c.split("__")[0][2:] for c in cols])
        colw = np.array([BLOCK_WEIGHTS[b] / (blocks == b).sum() for b in blocks])
        lib = train.loc[train.m_fresh_available & train.y_ref.notna() & train[f"y_{session}_mean"].notna()]
        X = lib[cols].to_numpy(float)
        center = np.nan_to_num(np.nanmedian(X, 0))
        scale = np.nanstd(X, 0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, 1.0)
        mult = np.sqrt(colw) / scale

        def z(frame):
            out = (frame[cols].to_numpy(float) - center) * mult
            out[~np.isfinite(out)] = 0.0
            return out

        L = z(lib)
        lib_min, lib_same = lib.b_minute.to_numpy(), lib.b_same_day.to_numpy()
        lib_ref = lib.y_ref.to_numpy(float)
        lib_y = lib[f"y_{session}_mean"].to_numpy(float)
        query = test.loc[test.m_fresh_available & test.s_freshPm25.notna()]
        pred = pd.Series(np.nan, index=test.index)
        Q = z(query)
        for i, (epoch, row) in enumerate(query.iterrows()):
            gap = np.abs(lib_min - row.b_minute)
            gap = np.minimum(gap, 1440 - gap)
            ok = (gap <= WINDOW_MIN) & (lib_same == row.b_same_day)
            if ok.sum() < K:
                continue
            d2 = ((L[ok] - Q[i]) ** 2).sum(1)
            nn = np.argsort(d2)[:K]
            qref = row.s_freshPm25
            values = (qref + analog.OFFSET) * (lib_y[ok][nn] + analog.OFFSET) / (lib_ref[ok][nn] + analog.OFFSET) - analog.OFFSET
            pred[epoch] = float(np.median(values))
        return pd.DataFrame({f"pred_{session}": pred})

    return fit_predict


HEADS = [("Morning same day @07:15", "07:15", "morning", True),
         ("Morning next day @18:05", "18:05", "morning", False),
         ("Morning next day @12:00", "12:00", "morning", False),
         ("Afternoon same day @12:00", "12:00", "afternoon", True),
         ("Afternoon next day @18:05", "18:05", "afternoon", False),
         ("Afternoon next day @20:00", "20:00", "afternoon", False)]


def run():
    ds = dataset.load()
    clocks = {h[1] for h in HEADS}
    minute = _minute_of_day(ds.index)
    hhmm = pd.Series([f"{m // 60:02d}:{m % 60:02d}" for m in minute], index=ds.index)
    test_mask = hhmm.isin(clocks)
    preds = []
    for session in ("morning", "afternoon"):
        frame = session_frame(ds, session)
        pred = walkforward.walk_forward(frame, make_fit_predict(session), targets=(session,), lookback_days=60,
                                        embargo_minutes=common.EMBARGO_MINUTES, test_mask=test_mask,
                                        log=lambda *_: None)
        preds.append(pred)
    pred = pd.concat(preds, axis=1)
    pred.to_parquet(common.OUT / "sessions_pred.parquet")
    return pred


def score(pred=None):
    pred = pd.read_parquet(common.OUT / "sessions_pred.parquet") if pred is None else pred
    ds = dataset.load()
    replay = report.standard_replay(ds)
    ev = ds.loc[ds["eval"]]
    clock = pd.Series(pd.to_datetime(ev.index, unit="s", utc=True).tz_convert("Asia/Kuala_Lumpur").strftime("%H:%M"),
                      index=ev.index)
    rows = []
    for period in ("tune", "holdout", "all"):
        idx = ev.index.to_numpy(np.int64)
        in_period = {"tune": idx < common.HOLDOUT_START, "holdout": idx >= common.HOLDOUT_START,
                     "all": np.ones(len(idx), bool)}[period]
        for label, at, key, same in HEADS:
            col = f"pred_{key}"
            base = pd.Series(in_period, index=ev.index) & (clock == at) & ev[f"y_{key}_mean"].notna() \
                & (ev[f"y_{key}_same_day"] == same) & pred[col].reindex(ev.index).notna()
            inc_mask = base & replay[col].reindex(ev.index).notna()
            for name, mask, with_inc in (("all rows", base, False), ("incumbent rows", inc_mask, True)):
                if not mask.any():
                    continue
                f = ev.loc[mask]
                c = scoring.point_metrics(f[f"y_{key}_mean"], pred.loc[f.index, col], f.y_ref)
                r = scoring.point_metrics(f[f"y_{key}_mean"], replay.loc[f.index, col], f.y_ref) if with_inc else {}
                rows.append({"period": period, "head": label, "rows": name, "n": c["n"],
                             "persistence": c["persistence_mae"], "analog": c["mae"],
                             "incumbent": r.get("mae", np.nan)})
    table = pd.DataFrame(rows)
    table.to_csv(common.OUT / "sessions_scores.csv", index=False)
    return table


if __name__ == "__main__":
    pred = run()
    with pd.option_context("display.width", 200):
        print(score(pred).round(2).to_string())
