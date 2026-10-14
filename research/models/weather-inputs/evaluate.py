"""Score the weather-inputs candidate against persistence/prior, the incumbent replay and
its own sensor-only ablation, on identical rows.

    python -B weather-inputs/evaluate.py tune      # tuning grid on 2026-09-10..09-26 only
    python -B weather-inputs/evaluate.py final     # writes weather-inputs/RESULTS.md + data/models/weather-inputs/results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from harness import config, dataset, report, scoring   # noqa: E402

import numpy as np   # noqa: E402
import pandas as pd   # noqa: E402

from candidate import OUT, TUNE_END   # noqa: E402

P5 = [f"p_class5_{k}" for k in range(5)]
TAILS = ["p_fall20", "p_rise20", "p_fall40", "p_rise40"]
POINTS = [("arrival", "pred_arrival", "y_arrival"), ("ride_mean", "pred_ride_mean", "y_ride_mean"),
          ("ride_min", "pred_ride_min", "y_ride_min"), ("ride_max", "pred_ride_max", "y_ride_max")]
PERIODS = {"test": (TUNE_END, None), "tune": (config.EVAL_START_EPOCH, TUNE_END), "all": (config.EVAL_START_EPOCH, None)}


def base(ds):
    evaluate = ds.loc[ds["eval"] & ds.m_fresh_available]
    replay = report.standard_replay(ds).reindex(evaluate.index)
    prior = scoring.walk_forward_prior(evaluate, "y_class5", range(5), "y_arrival_complete", history=ds)
    prior.columns = P5
    prior["p_fall20"] = prior.p_class5_0 + prior.p_class5_1
    prior["p_rise20"] = prior.p_class5_3 + prior.p_class5_4
    prior["p_fall40"], prior["p_rise40"] = prior.p_class5_0, prior.p_class5_4
    le70 = scoring.walk_forward_prior(evaluate, "y_ride_le70", (0.0, 1.0), "y_ride_complete", history=ds)
    prior["p_ride_le70"] = le70["prior_1.0"]
    return evaluate, replay, prior, scoring.persistence_predictions(evaluate)


def in_period(index, period):
    a, b = PERIODS[period]
    idx = np.asarray(index, np.int64)
    return pd.Series((idx >= a) & ((idx < b) if b else True), index=index)


def ev_cell(r):
    if not r or not r.get("n_issues"):
        return "-"
    return (f"{r['episodes_caught']}/{r['episodes']} ep; hits {r['hits']}, miss {r['misses']}, "
            f"wrong {r['wrong_way']}, false {r['false_calls']} ({r['false_clusters']} cl)")


def f(x, d=2):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def score_heads(evaluate, models: dict, period: str):
    """models: name -> standard-column frame. Masks require every model with that head to have output."""
    out = {}
    per = in_period(evaluate.index, period)
    have = lambda col: [m for m, frame in models.items() if col in frame]   # noqa: E731
    # class / tails
    if any("p_fall20" in fr for fr in models.values()):
        mask = per & evaluate.y_delta.notna()
        for m in have("p_fall20"):
            mask &= models[m]["p_fall20"].reindex(evaluate.index).notna()
        out["class"] = {m: scoring.score_candidate(evaluate, models[m][[c for c in P5 + TAILS if c in models[m]]], mask=mask)
                        for m in have("p_fall20")}
    for key, col, target in POINTS:
        if not have(col):
            continue
        mask = per & evaluate[target].notna()
        for m in have(col):
            mask &= models[m][col].reindex(evaluate.index).notna()
        out[key] = {m: scoring.score_candidate(evaluate, models[m][[col]], mask=mask) for m in have(col)}
        if key == "ride_min":
            for m in have(col):
                out[key][m]["pinball"] = scoring.pinball(evaluate.y_ride_min[mask], models[m][col].reindex(evaluate.index)[mask], .1)
        if key == "ride_max":
            for m in have(col):
                out[key][m]["pinball"] = scoring.pinball(evaluate.y_ride_max[mask], models[m][col].reindex(evaluate.index)[mask], .9)
    if have("p_ride_le70"):
        mask = per & evaluate.y_ride_le70.notna()
        for m in have("p_ride_le70"):
            mask &= models[m]["p_ride_le70"].reindex(evaluate.index).notna()
        out["le70"] = {m: scoring.score_candidate(evaluate, models[m][["p_ride_le70"]], mask=mask) for m in have("p_ride_le70")}
    return out


# ----------------------------------------------------------------------------- tuning
def tune():
    ds = dataset.load()
    evaluate, replay, prior, persistence = base(ds)
    runs = json.loads((OUT / "tune_runs.json").read_text())
    rows = []
    for key, info in runs.items():
        head, _, fs = key.split("__")
        pred = pd.read_parquet(OUT / f"tune_{key}.parquet")
        models = {"cand": pred}
        if head == "class5":
            models["inc"] = replay[P5 + TAILS]
        elif head == "le70":
            models["inc"] = replay[["p_ride_le70"]]
        else:
            models["inc"] = replay[[f"pred_{head}"]]
        s = score_heads(evaluate, models, "tune")
        if head == "class5":
            c = s["class"]["cand"]
            rows.append((key, info["cfg"], f"logloss {f(c['class5']['log_loss'], 4)} brier {f(c['class5']['brier'], 4)} "
                         f"bF20 {f(c['brier_fall20']['brier'], 4)} bR20 {f(c['brier_rise20']['brier'], 4)} | "
                         f"20: {ev_cell(c['events20_prob_argmax'])} | 40: {ev_cell(c['events40_prob_argmax'])}"))
        elif head == "le70":
            rows.append((key, info["cfg"], f"brier {f(s['le70']['cand']['ride_le70']['brier'], 4)} (inc {f(s['le70']['inc']['ride_le70']['brier'], 4)})"))
        else:
            c = s[head]["cand"][head]
            rows.append((key, info["cfg"], f"MAE {f(c['mae'])} (pers {f(c['persistence_mae'])}, inc {f(s[head]['inc'][head]['mae'])}) "
                         f"MAE|chg|>=20 {f(c['mae_change20'])}"
                         + (f" | 20pt: {ev_cell(s[head]['cand']['events20_point'])}" if head == "arrival" else "")))
    # reference rows: incumbent and prior on the tune half
    s = score_heads(evaluate, {"inc": replay[P5 + TAILS], "prior": prior[P5 + TAILS]}, "tune")
    for m in ("inc", "prior"):
        c = s["class"][m]
        rows.append((f"class5 {m}", {}, f"logloss {f(c['class5']['log_loss'], 4)} brier {f(c['class5']['brier'], 4)} "
                     f"bF20 {f(c['brier_fall20']['brier'], 4)} bR20 {f(c['brier_rise20']['brier'], 4)} | "
                     f"20: {ev_cell(c['events20_prob_argmax'])} | 40: {ev_cell(c['events40_prob_argmax'])}"))
    for key, cfg, text in rows:
        print(f"{key:24s} {json.dumps(cfg):55s} {text}")


# ----------------------------------------------------------------------------- final
def final():
    ds = dataset.load()
    evaluate, replay, prior, persistence = base(ds)
    sw = pd.read_parquet(OUT / "final_SW.parquet")
    so = pd.read_parquet(OUT / "final_S.parquet")
    chosen = json.loads((HERE / "final_config.json").read_text())
    results, md = {}, []
    names = {"persistence": "Persistence / prior", "incumbent": "Incumbent replay",
             "sensor": "Same model, sensor only", "weather": "Candidate (sensor + weather)"}
    for period in ("test", "tune", "all"):
        models_cls = {"prior": prior[P5 + TAILS], "persistence0": persistence[TAILS], "incumbent": replay[P5 + TAILS],
                      "sensor": so[P5 + TAILS], "weather": sw[P5 + TAILS]}
        models_pt = {"persistence": persistence, "incumbent": replay, "sensor": so, "weather": sw}
        sc = score_heads(evaluate, models_cls, period)
        sp = score_heads(evaluate, {m: fr.drop(columns=[c for c in fr.columns if c.startswith("p_")], errors="ignore")
                                    for m, fr in models_pt.items()}, period)
        sl = score_heads(evaluate, {"persistence": persistence[["p_ride_le70"]], "prior": prior[["p_ride_le70"]],
                                    "incumbent": replay[["p_ride_le70"]], "sensor": so[["p_ride_le70"]],
                                    "weather": sw[["p_ride_le70"]]}, period)
        results[period] = {"class": sc, "points": sp, "le70": sl}
        md.append(render_period(period, sc, sp, sl))
    cases = {"weather": report.case_table(ds, sw), "sensor": report.case_table(ds, so)}
    md.append(render_cases(cases))
    md.append(render_oct13(ds, replay, so, sw))
    header = ["# Weather-inputs candidate: generated results",
              "",
              "Generated by `python -B weather-inputs/evaluate.py final`. Narrative and caveats: `REPORT.md`.",
              f"Frozen configuration (chosen on the tune half only): `{json.dumps(chosen)}`.",
              "Periods: **test** = 2026-09-27..10-13 MYT (not used for any choice), **tune** = 2026-09-10..09-26, "
              "**all** = both. Identical rows inside each table: every model listed must have produced an output.",
              "Event cells: episodes caught / episodes; issue hits, misses, wrong-way, false calls (false clusters).",
              ""]
    (HERE / "RESULTS.md").write_text("\n".join(header + md) + "\n", encoding="utf-8")
    (OUT / "results.json").write_text(json.dumps(results, indent=1, default=float))
    print("\n".join(md))


def render_period(period, sc, sp, sl):
    title = {"test": "Test half 2026-09-27..10-13 (held out)", "tune": "Tune half 2026-09-10..09-26 (used for selection)",
             "all": "Whole eval 2026-09-10..10-13"}[period]
    L = [f"## {title}", ""]
    c = sc["class"]
    n = c["weather"]["class5"]["n"]
    L += [f"### Arrival change classes (n={n}; embargo 120 min)", "",
          "| Metric | Prior / persistence | Incumbent logistic | Same model, sensor only | Candidate (sensor + weather) |",
          "|---|---|---|---|---|"]
    for t in (20, 40):
        for rule in ("argmax", "0.2"):
            k = f"events{t}_prob_{rule}"
            L.append(f"| >={t} events ({rule}) | {ev_cell(c['persistence0'][k])} | {ev_cell(c['incumbent'][k])} | "
                     f"{ev_cell(c['sensor'][k])} | {ev_cell(c['weather'][k])} |")
        for side in ("fall", "rise"):
            k = f"brier_{side}{t}"
            L.append(f"| P({side} >={t}) Brier | {f(c['prior'][k]['brier'], 4)} | {f(c['incumbent'][k]['brier'], 4)} | "
                     f"{f(c['sensor'][k]['brier'], 4)} | {f(c['weather'][k]['brier'], 4)} |")
    for k in ("brier", "log_loss"):
        L.append(f"| 5-class {k.replace('_', ' ')} | {f(c['prior']['class5'][k], 4)} | {f(c['incumbent']['class5'][k], 4)} | "
                 f"{f(c['sensor']['class5'][k], 4)} | {f(c['weather']['class5'][k], 4)} |")
    L += ["", "### +90 arrival and ride window (embargo 330 min)", "",
          "| Target | Metric | Persistence | Incumbent Ridge v4 | Same model, sensor only | Candidate (sensor + weather) | n |",
          "|---|---|---|---|---|---|---|"]
    for key, col, _ in POINTS:
        s = sp[key]
        L.append(f"| {key} | MAE | {f(s['persistence'][key]['mae'])} | {f(s['incumbent'][key]['mae'])} | "
                 f"{f(s['sensor'][key]['mae'])} | {f(s['weather'][key]['mae'])} | {s['weather'][key]['n']} |")
        L.append(f"| {key} | skill vs persistence | 0 | {f(s['incumbent'][key]['skill_vs_persistence'], 3)} | "
                 f"{f(s['sensor'][key]['skill_vs_persistence'], 3)} | {f(s['weather'][key]['skill_vs_persistence'], 3)} | |")
        L.append(f"| {key} | RMSE | {f(s['persistence'][key]['rmse'])} | {f(s['incumbent'][key]['rmse'])} | "
                 f"{f(s['sensor'][key]['rmse'])} | {f(s['weather'][key]['rmse'])} | |")
        L.append(f"| {key} | MAE where abs(true change) >= 20 | {f(s['persistence'][key]['mae_change20'])} | "
                 f"{f(s['incumbent'][key]['mae_change20'])} | {f(s['sensor'][key]['mae_change20'])} | "
                 f"{f(s['weather'][key]['mae_change20'])} | {s['weather'][key]['n_change20']} |")
        if key in ("ride_min", "ride_max"):
            q = "q0.1" if key == "ride_min" else "q0.9"
            L.append(f"| {key} | pinball {q} | {f(s['persistence']['pinball'])} | {f(s['incumbent']['pinball'])} | "
                     f"{f(s['sensor']['pinball'])} | {f(s['weather']['pinball'])} | |")
        if key == "arrival":
            for t in (20, 40):
                k = f"events{t}_point"
                L.append(f"| arrival point | >={t} events | {ev_cell(s['persistence'][k])} | {ev_cell(s['incumbent'][k])} | "
                         f"{ev_cell(s['sensor'][k])} | {ev_cell(s['weather'][k])} | |")
    s = sl["le70"]
    L += ["", f"### P(ride mean <= 70) (n={s['weather']['ride_le70']['n']}; incumbent exists from 09-18)", "",
          "| Persistence 0/1 call | Walk-forward prior | Incumbent cycling logistic | Same model, sensor only | Candidate (sensor + weather) |",
          "|---|---|---|---|---|",
          f"| {f(s['persistence']['ride_le70']['brier'], 4)} | {f(s['prior']['ride_le70']['brier'], 4)} | "
          f"{f(s['incumbent']['ride_le70']['brier'], 4)} | {f(s['sensor']['ride_le70']['brier'], 4)} | "
          f"{f(s['weather']['ride_le70']['brier'], 4)} |", ""]
    return "\n".join(L)


def render_cases(cases):
    L = ["## Big-event cases (harness case table, same onsets as BASELINES.md)", ""]
    for name, frame in cases.items():
        label = "Candidate (sensor + weather)" if name == "weather" else "Same model, sensor only"
        keep = [c for c in frame.columns if c not in ("first-crossing P at onset", "first-crossing first argmax call",
                                                       "issued at onset (Oct 8+)")]
        frame = frame[keep].rename(columns={"Ridge replay at onset": "point +90 at onset",
                                            "Ridge first right call (min vs onset, from -60)": "point first right call",
                                            "logistic P(dir>=20) onset-30 / onset": "P(dir>=20) onset-30 / onset",
                                            "logistic first argmax call": "first argmax call"})
        L += [f"### {label}", "", "| " + " | ".join(frame.columns) + " |", "|" + "---|" * len(frame.columns)]
        L += ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
        L.append("")
    return "\n".join(L)


def render_oct13(ds, replay, so, sw):
    a, b = config.myt_epoch("2026-10-13 15:30"), config.myt_epoch("2026-10-13 18:00")
    idx = ds.index[(ds.index >= a) & (ds.index <= b)][::3]
    L = ["## Oct 13 rain drop, issue by issue (every 15 min)", "",
         "| Issue | ref | true +90 change | forecast P(rain) +90 | incumbent P(fall>=20) | sensor-only P(fall>=20) | candidate P(fall>=20) | candidate P(fall>=40) | candidate +90 point |",
         "|---|---|---|---|---|---|---|---|---|"]
    for i in idx:
        r = ds.loc[i]
        L.append(f"| {str(config.myt(i))[11:16]} | {f(r.y_ref, 0)} | {f(r.y_delta, 0)} | {f(r.w_pprob_90, 0)} | "
                 f"{f(replay.p_fall20.get(i, np.nan))} | {f(so.p_fall20.get(i, np.nan))} | {f(sw.p_fall20.get(i, np.nan))} | "
                 f"{f(sw.p_fall40.get(i, np.nan))} | {f(sw.pred_arrival.get(i, np.nan), 0)} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    {"tune": tune, "final": final}[sys.argv[1]]()
