"""Walk-forward tuning and final evaluation of the session candidates.

    cd research/models
    & $py -B -m sessions.run grid        # all configs on the tune days only (cached per config)
    & $py -B -m sessions.run report      # select on the tune split, run the selected config on every
                                         # eval day, write data/models/sessions/results.json

Split (fixed before any run): tune = issues 2026-09-10 .. 09-26 whose session
label completed before 09-27 00:00 MYT; test = issues 2026-09-27 .. 10-13.
The configuration is chosen on the tune split only (all-grid session MAE,
both sessions). Test numbers are reported for that single configuration.
"""
from __future__ import annotations

import itertools
import json
import sys
import time

import numpy as np
import pandas as pd

from harness import config, dataset, report, scoring, walkforward
from . import features, model

OUT = config.OUT / "sessions"
TUNE_END = config.myt_epoch("2026-09-27")
TRAIN_STEP = 900                 # training origins every 15 min (grid rows are 5-min and near-duplicates)
HEADS = [  # label, clock, session, same_day, incumbent column
    ("Morning same day @07:15", "07:15", "morning", True, "pred_morning"),
    ("Morning next day @12:00", "12:00", "morning", False, "pred_morning"),
    ("Morning next day @18:05", "18:05", "morning", False, "pred_morning"),
    ("Afternoon same day @07:15", "07:15", "afternoon", True, None),
    ("Afternoon same day @12:00", "12:00", "afternoon", True, "pred_afternoon"),
    ("Afternoon next day @18:05", "18:05", "afternoon", False, "pred_afternoon"),
    ("Afternoon next day @20:00", "20:00", "afternoon", False, "pred_afternoon"),
]


def configs():
    out = {}
    full = ("base", "weather", "cams")
    # sess7 (mean of the last 7 sessions) was dropped as an anchor after its naive tune MAE (31.2 vs
    # persistence 18.0); three of its runs were never started, none were scored.
    grid = [(a, "log", g, lb) for a, g, lb in itertools.product(
        ("ref", "diurnal"), (("base",), ("base", "weather"), full), (14, 28))]
    grid += [(a, "diff", full, lb) for a, lb in itertools.product(("ref", "diurnal"), (14, 28))]
    for anchor, space, groups, lookback in grid:
        key = f"hgb_{anchor}_{space}_{'+'.join(groups)}_{lookback}d"
        out[key] = (model.hgb(anchor, space, groups), lookback)
    for anchor, groups, hours in itertools.product(("ref", "diurnal"), (("base", "weather"), full),
                                                   (None, (6, 18))):
        key = f"win_{anchor}_log_{'+'.join(groups)}_{'all' if hours is None else 'day'}_14d"
        out[key] = (model.hgb_windows(anchor, "log", groups, hours), 14)
    # added after the 14-day window runs: the 28-day lookback of the best tune config
    out["win_diurnal_log_base+weather_all_28d"] = (model.hgb_windows("diurnal", "log", ("base", "weather")), 28)
    # post-selection diagnostics (run on all days, never in the tune table): selected window model without weather
    for lookback in (14, 28):
        out[f"diag_win_diurnal_log_base_all_{lookback}d"] = (model.hgb_windows("diurnal", "log", ("base",)), lookback)
    for lookback in (14, 28):
        out[f"linear_{lookback}d"] = (model.linear(1.0), lookback)
    for column in ("x_sess_last", "x_sess_mean7", "x_anchor_diurnal", "x_c_sess_pm25"):
        out[f"naive_{column[2:]}"] = (model.naive(column), 1)
    return out


def load_stack(ds):
    path = OUT / "stacked.parquet"
    if path.exists():
        return pd.read_parquet(path)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    stack = features.stacked(ds, features.history_features(ds))
    stack.to_parquet(path)
    print(f"stacked frame {stack.shape} in {time.time() - started:.0f}s")
    return stack


def walk_forward(train_frame, test_frame, fit_predict, lookback_days, days):
    """Harness walk-forward semantics per row: refit at each MYT midnight D on training rows with issue
    in (D - lookback, D) whose label completed before D; predict the test rows issued on day D.
    Session configs train and test on the stacked frame; window configs train on generic windows."""
    issue = train_frame.index.get_level_values(0).to_numpy(np.int64)
    complete = train_frame.complete.to_numpy(np.int64)
    labelled = np.isfinite(train_frame.y.to_numpy(float))
    test_issue = test_frame.index.get_level_values(0).to_numpy(np.int64)
    outputs = []
    for cutoff in days:
        train_mask = ((issue > cutoff - lookback_days * 86400) & (issue < cutoff) & (complete < cutoff)
                      & labelled & (issue % TRAIN_STEP == 0))
        test_mask = (test_issue >= cutoff) & (test_issue < cutoff + 86400)
        if not test_mask.any():
            continue
        test = test_frame.loc[test_mask].drop(columns=["y", "complete"])
        outputs.append(fit_predict(train_frame.loc[train_mask], test))
    return pd.concat(outputs).rename("pred")


def run_grid(ds, stack, names=None, full=False):
    """Tuning runs cover the tune days only (pred_<key>); ``full`` runs every eval day (final_<key>)."""
    days = walkforward.eval_days(ds)
    if not full:
        days = [d for d in days if d < TUNE_END]
    for key, (fn, lookback) in configs().items():
        if names and key not in names:
            continue
        path = OUT / f"{'final' if full else 'pred'}_{key}.parquet"
        if path.exists():
            continue
        started = time.time()
        if getattr(fn, "windows", False):
            from . import windows
            train_frame, test_frame = windows.load(ds, stack)
        else:
            train_frame = test_frame = stack
        pred = walk_forward(train_frame, test_frame, fn, lookback, days)
        pred.to_frame().to_parquet(path)
        print(f"{key}: {time.time() - started:.0f}s", flush=True)


def load_pred(key, final=True):
    return pd.read_parquet(OUT / f"{'final' if final else 'pred'}_{key}.parquet")["pred"]


# ----------------------------------------------------------------------------- scoring
def _clock(index):
    return pd.to_datetime(index, unit="s", utc=True).tz_convert("Asia/Kuala_Lumpur").strftime("%H:%M")


def grid_mae(stack, pred, rows):
    """All-grid session MAE over the given (issue, session) rows where candidate and persistence exist."""
    y = stack.loc[rows, "y"].to_numpy(float)
    p = pred.reindex(rows).to_numpy(float)
    ref = stack.loc[rows, "x_freshPm25"].to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(p) & np.isfinite(ref)
    return float(np.mean(np.abs(p[ok] - y[ok]))), float(np.mean(np.abs(ref[ok] - y[ok]))), int(ok.sum())


def split_rows(stack, split):
    issue = stack.index.get_level_values(0).to_numpy(np.int64)
    complete = stack.complete.to_numpy(np.int64)
    evaluate = issue >= config.EVAL_START_EPOCH
    if split == "tune":
        mask = evaluate & (issue < TUNE_END) & (complete < TUNE_END)
    elif split == "test":
        mask = issue >= TUNE_END
    else:
        mask = evaluate
    return stack.index[mask & np.isfinite(stack.y.to_numpy(float))]


def head_rows(ds, stack, preds: dict, incumbent, split):
    """Per canonical head: identical rows for persistence, candidates and (where present) the incumbent."""
    rows = []
    rows_all = split_rows(stack, split)
    issues = rows_all.get_level_values(0).to_numpy(np.int64)
    clock = _clock(issues)
    for label, at, session, same, inc_col in HEADS:
        sel = rows_all[(clock == at) & (rows_all.get_level_values(1) == session)]
        sub = stack.loc[sel]
        sel = sel[(sub.x_same_day.to_numpy() == float(same)) & np.isfinite(sub.x_freshPm25.to_numpy())]
        sub = stack.loc[sel]
        y, ref = sub.y.to_numpy(float), sub.x_freshPm25.to_numpy(float)
        inc = (incumbent[inc_col].reindex(sel.get_level_values(0)).to_numpy(float) if inc_col
               else np.full(len(sel), np.nan))
        ok_all = np.all([np.isfinite(p.reindex(sel).to_numpy(float)) for p in preds.values()], axis=0)
        record = {"head": label, "split": split, "n": int(ok_all.sum()),
                  "persistence": float(np.mean(np.abs(ref - y)[ok_all])) if ok_all.any() else np.nan}
        for name, p in preds.items():
            v = p.reindex(sel).to_numpy(float)
            record[name] = float(np.mean(np.abs(v - y)[ok_all])) if ok_all.any() else np.nan
            record[f"{name}_bias"] = float(np.mean((v - y)[ok_all])) if ok_all.any() else np.nan
        both = ok_all & np.isfinite(inc)
        record["n_inc"] = int(both.sum())
        if both.any():
            record["inc_incumbent"] = float(np.mean(np.abs(inc - y)[both]))
            record["inc_persistence"] = float(np.mean(np.abs(ref - y)[both]))
            for name, p in preds.items():
                record[f"inc_{name}"] = float(np.mean(np.abs(p.reindex(sel).to_numpy(float) - y)[both]))
            # paired: cases where the candidate (first entry of preds) is closer than the incumbent
            cand = next(iter(preds.values())).reindex(sel).to_numpy(float)
            record["inc_wins"] = int(np.sum(np.abs(cand - y)[both] < np.abs(inc - y)[both]))
        rows.append(record)
    return rows


def lead_rows(stack, preds, split):
    rows_all = split_rows(stack, split)
    out = []
    lead = stack.loc[rows_all, "x_lead_min"].to_numpy(float)
    bins = [(90, 240), (240, 480), (480, 960), (960, 1440)]
    for session in features.SESSIONS:
        for lo, hi in bins:
            sel = rows_all[(rows_all.get_level_values(1) == session) & (lead >= lo) & (lead < hi)]
            if not len(sel):
                continue
            sub = stack.loc[sel]
            y, ref = sub.y.to_numpy(float), sub.x_freshPm25.to_numpy(float)
            ok = np.isfinite(ref) & np.all([np.isfinite(p.reindex(sel).to_numpy(float)) for p in preds.values()], axis=0)
            record = {"session": session, "lead": f"{lo}-{hi}", "n": int(ok.sum()),
                      "days": len({config.midnight(i) for i in sel.get_level_values(0)[ok]}),
                      "persistence": float(np.mean(np.abs(ref - y)[ok]))}
            for name, p in preds.items():
                record[name] = float(np.mean(np.abs(p.reindex(sel).to_numpy(float) - y)[ok]))
            out.append(record)
    return out


def head_frame(ds, stack, preds: dict, incumbent, split):
    """One row per canonical (head, issue): truth, reference, incumbent and every prediction."""
    rows_all = split_rows(stack, split)
    issues = rows_all.get_level_values(0).to_numpy(np.int64)
    clock = _clock(issues)
    parts = []
    for label, at, session, same, inc_col in HEADS:
        sel = rows_all[(clock == at) & (rows_all.get_level_values(1) == session)]
        sub = stack.loc[sel]
        sel = sel[(sub.x_same_day.to_numpy() == float(same)) & np.isfinite(sub.x_freshPm25.to_numpy())]
        frame = stack.loc[sel, ["y", "x_freshPm25", "start"]].rename(columns={"x_freshPm25": "ref"})
        frame["head"] = label
        frame["incumbent"] = (incumbent[inc_col].reindex(sel.get_level_values(0)).to_numpy(float) if inc_col
                              else np.nan)
        for name, p in preds.items():
            frame[name] = p.reindex(sel).to_numpy(float)
        parts.append(frame)
    return pd.concat(parts)


def session_events(frame, names, threshold):
    """Each canonical (head, issue) is one case: truth = sign(y - ref) if |y - ref| >= T; a point forecast
    calls when |pred - ref| >= T. Rows: where every listed model has an output."""
    ok = np.all([np.isfinite(frame[n].to_numpy(float)) for n in names], axis=0)
    sub = frame.loc[ok]
    truth = scoring.truth_direction(sub.y - sub.ref, threshold)
    out = {"n": int(ok.sum()), "falls": int((truth == -1).sum()), "rises": int((truth == 1).sum())}
    for name in names:
        call = scoring.calls_from_point(sub[name], sub.ref, threshold)
        event = truth != 0
        out[name] = {"hits": int((event & (call == truth)).sum()), "misses": int((event & (call == 0)).sum()),
                     "wrong_way": int((event & (call == -truth)).sum()), "false": int((~event & (call != 0)).sum())}
    return out


def issued_rows(ds, stack, pred):
    """Genuinely issued incumbent (Oct 8+, first publication within 120 s of each canonical clock) vs the
    candidate at the canonical grid tick; persistence = the issued baselinePoint (report.issued_session_rows)."""
    from harness import issued
    ledger = issued.ledger()
    canonical = issued.canonical_sample(ledger.loc[ledger.issue_epoch >= config.myt_epoch("2026-10-08")])
    out = []
    for label, clock, key, same in (("Morning same day @07:15", "07:15", "morning", True),
                                    ("Morning next day @12:00", "12:00", "morning", False),
                                    ("Morning next day @18:05", "18:05", "morning", False),
                                    ("Afternoon same day @12:00", "12:00", "afternoon", True),
                                    ("Afternoon next day @18:05", "18:05", "afternoon", False)):
        part = canonical.loc[(canonical.canonical == clock) & (canonical[f"{key}_model"] == issued.INCUMBENT_VERSIONS[key])]
        records = []
        for _, row in part.iterrows():
            tick = int(row.canonical_tick)
            if (tick, key) not in stack.index:
                continue
            target = stack.loc[(tick, key)]
            if int(target.start) != int(row[f"{key}_start"]) or bool(target.x_same_day) != same:
                continue
            records.append((target.y, row[f"{key}_point"], row[f"{key}_baseline"], pred.get((tick, key), np.nan)))
        frame = pd.DataFrame(records, columns=["y", "issued", "baseline", "candidate"]).dropna()
        if len(frame):
            out.append({"head": label, "n": len(frame),
                        "persistence": float((frame.baseline - frame.y).abs().mean()),
                        "issued": float((frame.issued - frame.y).abs().mean()),
                        "candidate": float((frame.candidate - frame.y).abs().mean())})
    return out


def tune_table(stack, keys):
    rows = split_rows(stack, "tune")
    out = []
    for key in keys:
        pred = load_pred(key, final=False)
        mae, pers, n = grid_mae(stack, pred, rows)
        record = {"config": key, "tune_grid_mae": mae, "tune_grid_persistence": pers, "n": n}
        for session in features.SESSIONS:
            sel = rows[rows.get_level_values(1) == session]
            record[f"tune_{session}_mae"] = grid_mae(stack, pred, sel)[0]
        out.append(record)
    return pd.DataFrame(out).sort_values("tune_grid_mae")


def main(argv):
    command = argv[0] if argv else "report"
    ds = dataset.load()
    stack = load_stack(ds)
    if command in ("grid", "final"):
        run_grid(ds, stack, set(argv[1:]) or None, full=command == "final")
        return
    keys = [k for k in configs() if (OUT / f"pred_{k}.parquet").exists()]
    table = tune_table(stack, keys)
    pd.set_option("display.width", 250)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    fitted = table.loc[~table.config.str.startswith(("naive", "diag"))]
    best = fitted.iloc[0].config
    session_only = fitted.loc[fitted.config.str.startswith("hgb_")].iloc[0].config
    lookback = "14d" if best.endswith("14d") else "28d"
    diagnostics = {"session_only": session_only, "linear": f"linear_{lookback}",
                   "sess_last": "naive_sess_last", "sess_mean7": "naive_sess_mean7", "cams": "naive_c_sess_pm25"}
    if best == f"win_diurnal_log_base+weather_all_{lookback}":
        diagnostics["no_weather"] = f"diag_win_diurnal_log_base_all_{lookback}"
    print("selected on tune:", best, "| diagnostics:", diagnostics)
    run_grid(ds, stack, {best, *diagnostics.values()}, full=True)
    incumbent = report.standard_replay(ds)
    preds = {"candidate": load_pred(best), **{name: load_pred(key) for name, key in diagnostics.items()}}
    results = {"selected": best, "diagnostics": diagnostics, "tune_table": table.to_dict("records")}
    for split in ("tune", "test", "all"):
        results[f"heads_{split}"] = head_rows(ds, stack, preds, incumbent, split)
        results[f"leads_{split}"] = lead_rows(stack, preds, split)
        rows = split_rows(stack, split)
        results[f"grid_{split}"] = {name: grid_mae(stack, p, rows) for name, p in preds.items()}
    for split in ("tune", "test"):
        frame = head_frame(ds, stack, preds, incumbent, split)
        frame.to_parquet(OUT / f"heads_{split}.parquet")
        for threshold in (20, 40):
            results[f"events{threshold}_{split}_with_incumbent"] = session_events(
                frame, ["incumbent", "candidate", "linear"], threshold)
            results[f"events{threshold}_{split}_all_heads"] = session_events(frame, ["candidate", "linear"], threshold)
        big = frame.loc[(frame.y - frame.ref).abs() >= 40, ["head", "ref", "y", "incumbent", "candidate", "linear"]]
        big.insert(0, "issue_myt", [config.myt(i).strftime("%m-%d %H:%M") for i in big.index.get_level_values(0)])
        results[f"big40_{split}"] = big.reset_index(drop=True).to_dict("records")
        print(f"\n== events {split}", {k: v for k, v in results.items() if k.startswith("events") and split in k})
        print(big.to_string(index=False, float_format=lambda v: f"{v:.0f}"))
    results["issued"] = issued_rows(ds, stack, preds["candidate"])
    print("\n== issued (Oct 8+)\n", pd.DataFrame(results["issued"]).to_string(index=False))
    (OUT / "results.json").write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
    for split in ("tune", "test"):
        print(f"\n== heads {split}")
        print(pd.DataFrame(results[f"heads_{split}"]).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        print(f"== leads {split}")
        print(pd.DataFrame(results[f"leads_{split}"]).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        print(f"== grid {split}", results[f"grid_{split}"])


if __name__ == "__main__":
    main(sys.argv[1:])
