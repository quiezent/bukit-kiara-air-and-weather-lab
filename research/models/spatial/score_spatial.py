"""Score spatial variants against persistence/prior and the incumbent replay on identical rows.

    python -B -m spatial.score_spatial <class_variant> <class_base> <reg_variant> <reg_base> [--lag 300] [--split tune|final|all]

Splits (fixed before any scoring): tune = 2026-09-10 .. 2026-10-05 (26 days), final =
2026-10-06 .. 2026-10-13 (8 days). On final days every neighbour input has a real receipt clock.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from harness import config, dataset, report, scoring

OUT = config.OUT / "spatial"
SPLIT_EPOCH = config.myt_epoch("2026-10-06")


def load_pred(name, lag):
    return pd.read_parquet(OUT / f"pred_{name}_lag{lag}.parquet")


def ev_frame(ds, split):
    ev = ds.loc[ds["eval"] & ds.m_fresh_available]
    if split == "tune":
        ev = ev.loc[ev.index < SPLIT_EPOCH]
    elif split == "final":
        ev = ev.loc[ev.index >= SPLIT_EPOCH]
    return ev


def _ev(r):
    if not r or not r.get("n_issues"):
        return "–"
    return (f"{r['episodes_caught']}/{r['episodes']} ep; hits {r['hits']}, miss {r['misses']}, "
            f"wrong {r['wrong_way']}, false {r['false_calls']} ({r['false_clusters']} cl)")


def f2(x, d=2):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def score(ds, split, preds: dict, lag):
    """preds: {label: frame}. Returns markdown rows and a dict of raw scores."""
    ev = ev_frame(ds, split)
    inc = report.standard_replay(ds)
    per = scoring.persistence_predictions(ev)
    rows, raw = [], {}

    # --- class probabilities: identical rows = incumbent logistic output & every candidate output
    cls = {k: v for k, v in preds.items() if "p_fall20" in v}
    mask_c = inc.p_fall20.notna()
    for v in cls.values():
        mask_c &= v.p_fall20.reindex(ds.index).notna()
    mask_c &= ds.y_class5.isin(range(5))
    prior = scoring.walk_forward_prior(ev, "y_class5", list(range(5)), "y_arrival_complete", history=ds)
    prior_frame = pd.DataFrame({f"p_class5_{k}": prior[f"prior_{k}"] for k in range(5)}, index=ev.index)
    prior_frame["p_fall20"] = prior_frame.p_class5_0 + prior_frame.p_class5_1
    prior_frame["p_rise20"] = prior_frame.p_class5_3 + prior_frame.p_class5_4
    prior_frame["p_fall40"], prior_frame["p_rise40"] = prior_frame.p_class5_0, prior_frame.p_class5_4
    models = {"prior/persistence": prior_frame, "incumbent logistic": inc, **cls}
    sc = {k: scoring.score_candidate(ev, m[[c for c in m.columns if c.startswith("p_")]], mask=mask_c)
          for k, m in models.items()}
    sc["prior/persistence"]["events20_prob_argmax"] = scoring.score_candidate(
        ev, per[["p_fall20", "p_rise20", "p_fall40", "p_rise40"]], mask=mask_c)["events20_prob_argmax"]
    sc["prior/persistence"]["events40_prob_argmax"] = scoring.score_candidate(
        ev, per[["p_fall20", "p_rise20", "p_fall40", "p_rise40"]], mask=mask_c)["events40_prob_argmax"]
    names = list(models)
    n = sc[names[1]]["class5"]["n"]
    rows.append(("**Arrival change classes**", "", *[f"**{k}**" for k in names], "n"))
    for t in (20, 40):
        for rule in ("argmax", "0.2"):
            key = f"events{t}_prob_{rule}"
            rows.append((f"change >={t} ({rule})", "events", *[_ev(sc[k].get(key)) for k in names], n))
    for b in ("fall20", "rise20", "fall40", "rise40"):
        rows.append((f"P({b})", "Brier", *[f2(sc[k][f'brier_{b}'].get('brier'), 4) for k in names], n))
    rows.append(("5-class", "Brier", *[f2(sc[k]["class5"].get("brier"), 4) for k in names], n))
    rows.append(("5-class", "log loss", *[f2(sc[k]["class5"].get("log_loss"), 4) for k in names], n))
    raw["class"] = sc

    # --- points: identical rows = Ridge replay output & every candidate output
    reg = {k: v for k, v in preds.items() if "pred_arrival" in v}
    if reg:
        mask_r = inc.pred_arrival.notna() & inc.pred_ride_mean.notna()
        for v in reg.values():
            mask_r &= v.pred_arrival.reindex(ds.index).notna() & v.pred_ride_mean.reindex(ds.index).notna()
        rmodels = {"persistence": per, "incumbent Ridge v4": inc, **reg}
        rs = {k: scoring.score_candidate(ev, m[[c for c in m.columns if c.startswith("pred_")
                                                and c not in ("pred_morning", "pred_afternoon")]], mask=mask_r)
              for k, m in rmodels.items()}
        rn = list(rmodels)
        rows.append(("**Concentrations**", "", *[f"**{k}**" for k in rn], "n"))
        a = {k: rs[k]["arrival"] for k in rn}
        rows.append(("+90 arrival", "MAE", *[f2(a[k]["mae"]) for k in rn], a[rn[0]]["n"]))
        rows.append(("+90 arrival", "skill vs persistence", *[f2(a[k]["skill_vs_persistence"], 3) for k in rn], a[rn[0]]["n"]))
        rows.append(("+90 arrival", "MAE where |change|>=20", *[f2(a[k]["mae_change20"]) for k in rn], a[rn[0]]["n_change20"]))
        rows.append(("change >=20 (point)", "events", *[_ev(rs[k]["events20_point"]) for k in rn], a[rn[0]]["n"]))
        rows.append(("change >=40 (point)", "events", *[_ev(rs[k]["events40_point"]) for k in rn], a[rn[0]]["n"]))
        for head in ("ride_mean", "ride_min", "ride_max"):
            h = {k: rs[k][head] for k in rn}
            rows.append((head.replace("_", " "), "MAE", *[f2(h[k]["mae"]) for k in rn], h[rn[0]]["n"]))
            rows.append((head.replace("_", " "), "skill vs persistence", *[f2(h[k]["skill_vs_persistence"], 3) for k in rn], h[rn[0]]["n"]))
        rows.append(("ride min", "pinball q0.1", *[f2(rs[k]["ride_min_pinball10"]) for k in rn], ""))
        rows.append(("ride max", "pinball q0.9", *[f2(rs[k]["ride_max_pinball90"]) for k in rn], ""))
        raw["points"] = rs
        # P(ride mean <= 70): rows where the cycling replay and every candidate exist
        mask_p = inc.p_ride_le70.notna() & ds.y_ride_mean.notna()
        for v in reg.values():
            mask_p &= v.p_ride_le70.reindex(ds.index).notna()
        ps = {k: scoring.score_candidate(ev, m[["p_ride_le70"]], mask=mask_p)["ride_le70"] for k, m in rmodels.items()}
        rows.append(("P(ride mean<=70)", "Brier (persistence = 0/1 call)", *[f2(ps[k].get("brier"), 4) for k in rn], ps[rn[0]].get("n")))
        raw["ride_le70"] = ps
    return rows, raw


def table(rows):
    width = max(len(r) for r in rows)
    out = []
    for r in rows:
        r = list(r) + [""] * (width - len(r))
        out.append("| " + " | ".join(str(x) for x in r) + " |")
        if len(out) == 1:
            out.append("|" + "---|" * width)
    return "\n".join(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("variants", nargs="+")
    p.add_argument("--lag", type=int, default=300)
    p.add_argument("--split", default="tune")
    p.add_argument("--cases", action="store_true")
    a = p.parse_args()
    ds = dataset.load()
    preds = {v: load_pred(v, a.lag) for v in a.variants}
    rows, raw = score(ds, a.split, preds, a.lag)
    print(f"split={a.split} lag={a.lag}")
    print(table(rows))
    (OUT / f"scores_{a.split}_lag{a.lag}_{'_'.join(a.variants)}.json").write_text(json.dumps(raw, default=float, indent=1))
    if a.cases:
        for v, frame in preds.items():
            if "p_fall20" in frame:
                print(f"\ncases: {v}")
                print(report.case_table(ds, frame.reindex(ds.index)).drop(columns=["issued at onset (Oct 8+)"]).to_markdown(index=False))
