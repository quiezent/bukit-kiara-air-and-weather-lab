"""Candidate at the exact clocks of genuinely issued forecasts (Oct 8+), vs the issued incumbent.

Same daily refit as the grid walk-forward (training rows from the 5-min grid dataset with the
walk-forward mask of harness.walkforward), but each day's model predicts the harness
issued-clock frame (features rebuilt at the exact issue clock, one issue per 5-min slot).

    python -B weather-inputs/issued_eval.py      # appends nothing; prints Markdown and writes issued_preds_*.parquet
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from harness import config, issued, scoring, walkforward   # noqa: E402
from harness.incumbents import run as runs   # noqa: E402

import numpy as np   # noqa: E402
import pandas as pd   # noqa: E402

import candidate as cand   # noqa: E402
import wxfeatures   # noqa: E402
from evaluate import ev_cell, f   # noqa: E402

START = config.myt_epoch("2026-10-08")


def train_frame(ds, cutoff, head, cfg):
    """Copy of harness.walkforward.walk_forward's training mask for one cutoff."""
    index = ds.index.to_numpy(np.int64)
    complete = ds[walkforward.COMPLETE_COLUMNS[cand.TARGET_KEY[head]]].to_numpy(np.int64) < cutoff
    mask = ((index > cutoff - cfg.get("lookback", 28) * 86400) & (index < cutoff - cand.EMBARGO[head] * 60) & complete)
    return ds.loc[mask]


def predict_issued(ds, frame, cols, chosen):
    parts = []
    days = sorted({config.midnight(i) for i in frame.index})
    for cutoff in days:
        test = frame.loc[(frame.index >= cutoff) & (frame.index < cutoff + 86400)]
        test = test.drop(columns=[c for c in test.columns if c.startswith(("y_", "issued__"))])
        out = []
        for head in ("class5", "arrival", "ride_mean", "ride_min", "ride_max", "le70"):
            cfg = chosen["ride_mean" if head in ("ride_min", "ride_max") else head]
            if head == "class5":
                fp = cand.class5_fit_predict(cols, cfg)
            elif head == "le70":
                fp = cand.le70_fit_predict(cols, cfg)
            else:
                fp = cand.point_fit_predict(head, cols, cfg)
            out.append(fp(train_frame(ds, cutoff, head, cfg), test, cutoff))
        parts.append(pd.concat(out, axis=1))
        print(f"{config.myt(cutoff):%Y-%m-%d} issued={len(test)}", flush=True)
    return pd.concat(parts)


def main():
    ds, sensor, weather = cand.load()
    frame = runs.issued_clock_frame()
    frame = frame.loc[frame.index >= START]
    frame = frame.join(wxfeatures.build(frame).add_prefix("x_"))
    chosen = json.loads((HERE / "final_config.json").read_text())
    preds = {}
    for name, cols in (("S", sensor), ("SW", sensor + weather)):
        path = cand.OUT / f"issued_preds_{name}.parquet"
        preds[name] = predict_issued(ds, frame, cols, chosen)
        preds[name].to_parquet(path)
    replay = runs.load("issued_clocks")
    L = ["## Genuinely issued clocks, Oct 8-13 (one issue per 5-min slot; rows where the issued model is the current incumbent)", "",
         "| Target | Metric | Persistence / prior | Issued incumbent | Same model, sensor only | Candidate (sensor + weather) | n |",
         "|---|---|---|---|---|---|---|"]
    ridge_ok = (frame.issued__arrival_model == issued.INCUMBENT_VERSIONS["ridge"]) & (frame.index >= config.myt_epoch("2026-10-09 17:44"))
    for key, col, target, icol in (("arrival", "pred_arrival", "y_arrival", "issued__arrival_point"),
                                   ("ride_mean", "pred_ride_mean", "y_ride_mean", "issued__ride_mean_point"),
                                   ("ride_min", "pred_ride_min", "y_ride_min", "issued__ride_min_point"),
                                   ("ride_max", "pred_ride_max", "y_ride_max", "issued__ride_max_point")):
        mask = ridge_ok & frame[icol].notna() & frame[target].notna() & preds["SW"][col].notna() & preds["S"][col].notna()
        sub = frame.loc[mask]
        i = scoring.point_metrics(sub[target], sub[icol], sub.y_ref)
        s = scoring.point_metrics(sub[target], preds["S"].loc[mask, col], sub.y_ref)
        w = scoring.point_metrics(sub[target], preds["SW"].loc[mask, col], sub.y_ref)
        L.append(f"| {key} | MAE | {f(i['persistence_mae'])} | {f(i['mae'])} | {f(s['mae'])} | {f(w['mae'])} | {i['n']} |")
    ac_ok = (frame.issued__ac_model == issued.INCUMBENT_VERSIONS["arrival_logistic"]) & frame.issued__ac_p_fall20.notna()
    mask = ac_ok & frame.y_delta.notna() & preds["SW"].p_fall20.notna() & preds["S"].p_fall20.notna()
    sub = frame.loc[mask]
    issues = sub.index.to_numpy(np.int64)
    day = pd.Series([config.midnight(i) for i in sub.index], index=sub.index)
    prior = scoring.walk_forward_prior(sub, "y_class5", range(5), "y_arrival_complete", history=ds)
    for t in (20, 40):
        truth = scoring.truth_direction(sub.y_delta, t)
        cells = [ev_cell(scoring.event_confusion(issues, truth, np.zeros(len(sub)))),
                 ev_cell(scoring.event_confusion(issues, truth, scoring.calls_from_probs(sub[f"issued__ac_p_fall{t}"], sub[f"issued__ac_p_rise{t}"])))]
        for name in ("S", "SW"):
            p = preds[name].loc[mask]
            cells.append(ev_cell(scoring.event_confusion(issues, truth, scoring.calls_from_probs(p[f"p_fall{t}"], p[f"p_rise{t}"]))))
        L.append(f"| change >={t} | events (argmax) | " + " | ".join(cells) + f" | {len(sub)} |")
        cells = [ev_cell(scoring.event_confusion(issues, truth, np.zeros(len(sub)))),
                 ev_cell(scoring.event_confusion(issues, truth, scoring.calls_from_probs(sub[f"issued__ac_p_fall{t}"], sub[f"issued__ac_p_rise{t}"], "0.2")))]
        for name in ("S", "SW"):
            p = preds[name].loc[mask]
            cells.append(ev_cell(scoring.event_confusion(issues, truth, scoring.calls_from_probs(p[f"p_fall{t}"], p[f"p_rise{t}"], "0.2"))))
        L.append(f"| change >={t} | events (0.2) | " + " | ".join(cells) + f" | {len(sub)} |")
        classes = {("fall", 20): [0, 1], ("rise", 20): [3, 4], ("fall", 40): [0], ("rise", 40): [4]}
        for side in ("fall", "rise"):
            pr = prior[[f"prior_{k}" for k in classes[(side, t)]]].sum(axis=1)
            y = sub[f"y_{side}{t}"]
            L.append(f"| P({side} >={t}) | Brier | {f(scoring.binary_brier(pr, y)['brier'], 4)} | "
                     f"{f(scoring.binary_brier(sub[f'issued__ac_p_{side}{t}'], y)['brier'], 4)} | "
                     f"{f(scoring.binary_brier(preds['S'].loc[mask, f'p_{side}{t}'], y)['brier'], 4)} | "
                     f"{f(scoring.binary_brier(preds['SW'].loc[mask, f'p_{side}{t}'], y)['brier'], 4)} | {len(sub)} |")
    cy_ok = (frame.issued__cy_model == issued.INCUMBENT_VERSIONS["cycling70"]) & frame.issued__cy_p_le70.notna()
    mask = cy_ok & frame.y_ride_le70.notna() & preds["SW"].p_ride_le70.notna() & preds["S"].p_ride_le70.notna()
    sub = frame.loc[mask]
    y = sub.y_ride_le70
    L.append(f"| P(ride mean <=70) | Brier (persistence 0/1) | {f(scoring.binary_brier((sub.y_ref <= 70).astype(float), y)['brier'], 4)} | "
             f"{f(scoring.binary_brier(sub.issued__cy_p_le70, y)['brier'], 4)} | "
             f"{f(scoring.binary_brier(preds['S'].loc[mask, 'p_ride_le70'], y)['brier'], 4)} | "
             f"{f(scoring.binary_brier(preds['SW'].loc[mask, 'p_ride_le70'], y)['brier'], 4)} | {len(sub)} |")
    text = "\n".join(L) + "\n"
    (HERE / "RESULTS_ISSUED.md").write_text("# Weather-inputs candidate at genuinely issued clocks\n\n"
                                            "Generated by `python -B weather-inputs/issued_eval.py`. Oct 8-13 lies inside the held-out test half. "
                                            "Small sample: 5-6 days.\n\n" + text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
