"""Final walk-forward evaluation of the tuned analog configs.

    python -B -m analogs.evaluate     # writes analogs/RESULTS.md and data/models/analogs/{pred_*.parquet, results.json}

Configs come from data/models/analogs/chosen_{point,prob,event}.json (tuned on
issue days 09-10..09-27 only). Every config is then run walk-forward over the
whole evaluation period (daily refit, library = label-complete rows before the
day's cutoff minus the 330 min embargo). The HOLDOUT (09-28..10-13) is the
result; the tuning split is shown for reference only (hyperparameters were
selected on it).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from harness import config, report, scoring

from . import analog, common

RESULTS_PATH = Path(__file__).resolve().parent / "RESULTS.md"


def load_config(mode):
    raw = json.loads((common.OUT / f"chosen_{mode}.json").read_text(encoding="utf-8"))
    cfg = analog.AnalogConfig(weights=raw["weights"], k=int(raw["k"]), bandwidth=float(raw["bandwidth"]),
                              transfer=raw["transfer"], alpha=float(raw["alpha"]), traj_tau=float(raw["traj_tau"]),
                              point_stat=raw["point_stat"])
    return cfg, int(raw["lookback"]), raw


def run_mode(mode, slim):
    cfg, lookback, raw = load_config(mode)
    started = time.time()
    pred = common.run(cfg, "all", lookback_days=lookback, slim=slim)
    seconds = time.time() - started
    pred.to_parquet(common.OUT / f"pred_{mode}.parquet")
    return pred, cfg, lookback, seconds


def compose(preds, le70_from):
    point, prob, event = preds["point"], preds["prob"], preds["event"]
    main = point[common.POINT_COLS + common.RIDE_COLS].join(prob[common.CLASS_COLS + common.FIRST_COLS])
    main["p_ride_le70"] = preds[le70_from]["p_ride_le70"]
    ev = event[common.CLASS_COLS + common.FIRST_COLS].copy()
    return main, ev


# ----------------------------------------------------------------------------- tables
def _f(v, d=2):
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"


def _ev(e):
    if not e or not e.get("n_issues"):
        return "–"
    return (f"{e['episodes_caught']}/{e['episodes']} ep; hits {e['hits']}, miss {e['misses']}, "
            f"wrong {e['wrong_way']} ({e['wrong_way_clusters']} cl), false {e['false_calls']} ({e['false_clusters']} cl)")


def results_rows(main, event, period):
    a = common.score_all(main, period)
    b = common.score_all(event, period, others=False)
    rows = []

    def add(target, metric, base, inc, cand, cand_ev, n):
        rows.append((target, metric, base, inc, cand, cand_ev, n))

    p, i, c = a[("point", "persistence")], a[("point", "incumbent")], a[("point", "candidate")]
    add("+90 arrival", "MAE", _f(p["arrival"]["mae"]), _f(i["arrival"]["mae"]), _f(c["arrival"]["mae"]), "", c["arrival"]["n"])
    add("+90 arrival", "skill vs persistence", "0", _f(i["arrival"]["skill_vs_persistence"], 3),
        _f(c["arrival"]["skill_vs_persistence"], 3), "", c["arrival"]["n"])
    add("+90 arrival", "MAE where |true change| >= 20", _f(p["arrival"]["mae_change20"]), _f(i["arrival"]["mae_change20"]),
        _f(c["arrival"]["mae_change20"]), "", c["arrival"]["n_change20"])
    for t in (20, 40):
        add(f"change >={t} (point forecast)", "events", _ev(p[f"events{t}_point"]), _ev(i[f"events{t}_point"]),
            _ev(c[f"events{t}_point"]), "", c[f"events{t}_point"]["n_issues"])
    p, i, c = a[("ride", "persistence")], a[("ride", "incumbent")], a[("ride", "candidate")]
    for head in ("ride_mean", "ride_min", "ride_max"):
        add(head.replace("_", " "), "MAE", _f(p[head]["mae"]), _f(i[head]["mae"]), _f(c[head]["mae"]), "", c[head]["n"])
    add("ride min", "pinball q0.1", _f(p["ride_min_pinball10"]), _f(i["ride_min_pinball10"]), _f(c["ride_min_pinball10"]), "", c["ride_min"]["n"])
    add("ride max", "pinball q0.9", _f(p["ride_max_pinball90"]), _f(i["ride_max_pinball90"]), _f(c["ride_max_pinball90"]), "", c["ride_max"]["n"])
    q, i, c, e = a[("class", "prior")], a[("class", "incumbent")], a[("class", "candidate")], b[("class", "candidate")]
    for t in (20, 40):
        for rule in ("argmax", "0.2"):
            add(f"change >={t} (probabilities, {rule})", "events", "persistence: 0 calls",
                _ev(i[f"events{t}_prob_{rule}"]), _ev(c[f"events{t}_prob_{rule}"]), _ev(e[f"events{t}_prob_{rule}"]),
                c[f"events{t}_prob_{rule}"]["n_issues"])
        for side in ("fall", "rise"):
            k = f"brier_{side}{t}"
            add(f"P({side} >={t})", "Brier (prior)", _f(q[k]["brier"], 4), _f(i[k]["brier"], 4), _f(c[k]["brier"], 4),
                _f(e[k]["brier"], 4), c[k]["n"])
    for metric, key, d in (("Brier (prior)", "brier", 4), ("log loss (prior)", "log_loss", 4)):
        add("5-class arrival change", metric, _f(q["class5"][key], d), _f(i["class5"][key], d), _f(c["class5"][key], d),
            _f(e["class5"][key], d), c["class5"]["n"])
    q, i, c, e = a[("first", "prior")], a[("first", "incumbent")], a[("first", "candidate")], b[("first", "candidate")]
    add("first crossing >=20 in 90 min", "events (argmax)", "none-call: 0 calls", _ev(i["first20_events"]),
        _ev(c["first20_events"]), _ev(e["first20_events"]), c["first20_events"]["n_issues"])
    for metric, key in (("Brier (prior)", "brier"), ("log loss (prior)", "log_loss")):
        add("first crossing >=20 in 90 min", metric, _f(q["first20"][key], 4), _f(i["first20"][key], 4),
            _f(c["first20"][key], 4), _f(e["first20"][key], 4), c["first20"]["n"])
    p, q, i, c = a[("le70", "persistence")], a[("le70", "prior")], a[("le70", "incumbent")], a[("le70", "candidate")]
    add("P(ride mean <=70)", "Brier (persistence 0/1 call)", _f(p["ride_le70"]["brier"], 4), _f(i["ride_le70"]["brier"], 4),
        _f(c["ride_le70"]["brier"], 4), "", c["ride_le70"]["n"])
    add("P(ride mean <=70)", "Brier (prior)", _f(q["ride_le70"]["brier"], 4), _f(i["ride_le70"]["brier"], 4),
        _f(c["ride_le70"]["brier"], 4), "", c["ride_le70"]["n"])
    return rows, a, b


def threshold_sweep(frames, period, thresholds=(0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)):
    """Diagnostic: >=20 episodes caught vs false/wrong-way clusters when calling at P >= tau (both models)."""
    ds, ev, replay, persist, prior = common.references()
    mask = common.masks(ev, replay, period)["class"]
    sub = ev.loc[mask]
    truth = scoring.truth_direction(sub.y_delta, 20)
    issues = sub.index.to_numpy(np.int64)
    rows = []
    for name, frame in frames.items():
        f = frame.reindex(sub.index)
        for tau in thresholds:
            e = scoring.event_confusion(issues, truth, scoring.calls_from_probs(f.p_fall20, f.p_rise20, tau))
            rows.append({"model": name, "tau": tau, "caught": e["episodes_caught"], "episodes": e["episodes"],
                         "false_cl": e["false_clusters"], "wrong_cl": e["wrong_way_clusters"],
                         "false_issues": e["false_calls"], "hits": e["hits"]})
    return pd.DataFrame(rows)


def _row_losses(ev, frame, group):
    """Per-row loss of one model for a head group (NaN where it has no output)."""
    f = frame.reindex(ev.index)
    if group == "arrival":
        return (f.pred_arrival - ev.y_arrival).abs()
    if group in ("ride_mean", "ride_min"):
        return (f[f"pred_{group}"] - ev[f"y_{group}"]).abs()
    if group == "class5":
        p = f[[f"p_class5_{k}" for k in range(5)]].to_numpy(float)
        onehot = (ev.y_class5.to_numpy(float)[:, None] == np.arange(5)[None, :]).astype(float)
        return pd.Series(((p - onehot) ** 2).sum(1), index=ev.index)
    if group == "first20":
        p = f[["p_first_drop", "p_first_none", "p_first_rise"]].to_numpy(float)
        onehot = (ev.y_first20.to_numpy(float)[:, None] == np.array([-1, 0, 1])[None, :]).astype(float)
        return pd.Series(((p - onehot) ** 2).sum(1), index=ev.index)
    if group == "fall20":
        return (f.p_fall20 - ev.y_fall20) ** 2
    if group == "le70":
        return (f.p_ride_le70 - ev.y_ride_le70) ** 2
    raise KeyError(group)


def daily_paired(main, period="holdout", n_boot=2000, seed=0):
    """Analog minus baseline loss on identical rows: days won and a day-block bootstrap 95% CI of the mean difference."""
    ds, ev, replay, persist, prior = common.references()
    m = common.masks(ev, replay, period)
    checks = [("arrival", "point", "persistence", persist), ("arrival", "point", "incumbent", replay),
              ("ride_mean", "ride", "persistence", persist), ("ride_mean", "ride", "incumbent", replay),
              ("ride_min", "ride", "incumbent", replay),
              ("fall20", "class", "prior", prior), ("fall20", "class", "incumbent", replay),
              ("class5", "class", "prior", prior), ("class5", "class", "incumbent", replay),
              ("first20", "first", "prior", prior), ("first20", "first", "incumbent", replay),
              ("le70", "le70", "persistence", persist), ("le70", "le70", "incumbent", replay)]
    rng = np.random.default_rng(seed)
    rows = []
    for head, group, name, base in checks:
        diff = (_row_losses(ev, main, head) - _row_losses(ev, base, head))[m[group]].dropna()
        day = pd.Series([config.midnight(i) for i in diff.index], index=diff.index)
        per_day = diff.groupby(day).agg(["sum", "count"])
        sums, counts = per_day["sum"].to_numpy(), per_day["count"].to_numpy()
        boot = []
        for _ in range(n_boot):
            pick = rng.integers(0, len(sums), len(sums))
            boot.append(sums[pick].sum() / counts[pick].sum())
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"head": head, "vs": name, "mean_diff": diff.mean(), "ci_lo": lo, "ci_hi": hi,
                     "days_better": int((sums < 0).sum()), "days": len(sums), "n": len(diff)})
    return pd.DataFrame(rows)


def predict_latency(repeats=20):
    """Seconds for one daily library build + one single-issue prediction (prob config, last day)."""
    from harness import walkforward
    slim = common.load_slim()
    cfg, lookback, _ = load_config("prob")
    cutoff = common.days("all")[-1]
    index = slim.index.to_numpy(np.int64)
    complete = np.ones(len(slim), bool)
    for column in sorted({walkforward.COMPLETE_COLUMNS[t] for t in common.TARGETS}):
        complete &= slim[column].to_numpy(np.int64) < cutoff
    train = slim.loc[(index > cutoff - lookback * 86400) & (index < cutoff - common.EMBARGO_MINUTES * 60) & complete]
    test = slim.loc[(index >= cutoff)].drop(columns=[c for c in slim.columns if c.startswith("y_")])
    fp = analog.make_fit_predict(cfg)
    started = time.time()
    for _ in range(repeats):
        fp(train, test.iloc[:1], cutoff)
    one = (time.time() - started) / repeats
    started = time.time()
    fp(train, test, cutoff)
    return {"library_rows": len(train), "single_issue_seconds": round(one, 4),
            "full_day_seconds": round(time.time() - started, 3), "full_day_issues": len(test)}


def _md(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def build():
    slim = common.load_slim()
    preds, info = {}, {}
    for mode in ("point", "prob", "event"):
        pred, cfg, lookback, seconds = run_mode(mode, slim)
        preds[mode] = pred
        days = len(common.days("all"))
        info[mode] = {"config": cfg.label(), "lookback": lookback, "walkforward_seconds": round(seconds, 1),
                      "seconds_per_daily_refit_and_predict": round(seconds / days, 2),
                      "ms_per_issue": round(1000 * seconds / pred.index.size, 2)}
        print(mode, info[mode], flush=True)
    # P(ride mean <= 70): pick the config with the better TUNING Brier (one binary choice, tuning days only)
    tune_le70 = {m: common.score_all(preds[m], "tune", others=False)[("le70", "candidate")]["ride_le70"]["brier"]
                 for m in ("point", "prob")}
    le70_from = min(tune_le70, key=tune_le70.get)
    info["le70_from"] = {"chosen": le70_from, "tune_brier": tune_le70}
    main, event = compose(preds, le70_from)
    main.to_parquet(common.OUT / "pred_analog.parquet")
    event.to_parquet(common.OUT / "pred_analog_event.parquet")

    ds, ev, replay, persist, prior = common.references()
    sections, payload = [], {"info": info}
    header = ["Target", "Metric", "Persistence / prior", "Incumbent replay", "Analog", "Analog-event", "n"]
    for period, title in (("holdout", "HOLDOUT 2026-09-28 .. 10-13 (untouched by tuning)"),
                          ("tune", "Tuning split 2026-09-10 .. 09-27 (hyperparameters chosen here; in-sample for selection)")):
        rows, a, b = results_rows(main, event, period)
        sections.append(f"## {title}\n\n" + _md(header, rows))
        payload[period] = rows
    paired = daily_paired(main, "holdout")
    sections.append("## Holdout paired by day: analog loss minus baseline loss (negative = analog better)\n\n"
                    "Mean difference on identical rows; 95% CI from a day-block bootstrap (2,000 resamples of the "
                    "16 holdout days); days_better = days whose summed difference is negative.\n\n"
                    + _md(list(paired.columns), [[r["head"], r["vs"], f"{r['mean_diff']:+.4f}", f"{r['ci_lo']:+.4f}",
                                                  f"{r['ci_hi']:+.4f}", r["days_better"], r["days"], r["n"]]
                                                 for r in paired.to_dict("records")]))
    payload["paired"] = paired.to_dict("records")
    latency = predict_latency()
    info["latency"] = latency
    sections.append("## Runtime\n\n" + "\n".join(f"- {k}: {v}" for k, v in latency.items()))
    sweep = threshold_sweep({"incumbent logistic": replay, "analog": main, "analog-event": event}, "holdout")
    sections.append("## Diagnostic: >=20 episodes vs call threshold (holdout; call when P(fall20) or P(rise20) >= tau)\n\n"
                    + _md(list(sweep.columns), sweep.values.tolist()))
    payload["sweep"] = sweep.to_dict("records")
    cases_inc = report.case_table(ds, replay)
    cases_main = report.case_table(ds, main)
    cases_ev = report.case_table(ds, event)
    keep = ["case", "onset", "ref -> +90 at onset", "peak change"]
    cases = cases_main[keep].copy()
    cases["incumbent logistic P onset-30 / onset; first call"] = (cases_inc["logistic P(dir>=20) onset-30 / onset"] + "; "
                                                                  + cases_inc["logistic first argmax call"])
    cases["analog P onset-30 / onset; first call"] = (cases_main["logistic P(dir>=20) onset-30 / onset"] + "; "
                                                      + cases_main["logistic first argmax call"])
    cases["analog-event P onset-30 / onset; first call"] = (cases_ev["logistic P(dir>=20) onset-30 / onset"] + "; "
                                                            + cases_ev["logistic first argmax call"])
    cases["analog +90 at onset; first point call"] = (cases_main["Ridge replay at onset"] + "; "
                                                      + cases_main["Ridge first right call (min vs onset, from -60)"])
    cases["first-crossing P at onset; first call (incumbent | analog | analog-event)"] = (
        cases_inc["first-crossing P at onset"] + "; " + cases_inc["first-crossing first argmax call"] + " | "
        + cases_main["first-crossing P at onset"] + "; " + cases_main["first-crossing first argmax call"] + " | "
        + cases_ev["first-crossing P at onset"] + "; " + cases_ev["first-crossing first argmax call"])
    sections.append("## Big-event cases (all in the holdout; first call = minutes vs onset, searched from onset-60, argmax)\n\n"
                    + _md(list(cases.columns), cases.values.tolist()))
    payload["cases"] = cases.to_dict("records")
    configs = "\n".join(f"- **{m}**: `{info[m]['config']}` lookback {info[m]['lookback']} d; "
                        f"walk-forward {info[m]['walkforward_seconds']} s for {len(common.days('all'))} days "
                        f"({info[m]['seconds_per_daily_refit_and_predict']} s per daily refit+predict, "
                        f"{info[m]['ms_per_issue']} ms per issue)" for m in ("point", "prob", "event"))
    head = ("# Analog forecaster: generated results\n\nGenerated by `python -B -m analogs.evaluate`. "
            "Identical rows per head group: where the incumbent replay produced an output and the label exists "
            "(the analog outputs on every evaluation row). Event cells: episodes caught / episodes; issue counts; "
            "`cl` = clusters. Analog = point config for +90/ride heads, prob config for probabilities, "
            f"P(ride<=70) from the {le70_from} config (better tuning Brier: {tune_le70}). "
            "Analog-event = the event-tuned config (probability heads only).\n\nConfigs:\n" + configs)
    RESULTS_PATH.write_text(head + "\n\n" + "\n\n".join(sections) + "\n", encoding="utf-8")
    (common.OUT / "results.json").write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    print(RESULTS_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    build()
