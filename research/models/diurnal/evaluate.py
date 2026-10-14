"""Scoring helpers for the diurnal candidate (extends harness.report without changing it).

Every comparison uses identical rows: evaluation issues with a fresh reference,
inside the period, where the incumbent replay produced an output, the label
exists and the candidate produced an output (the candidate covers every such
row; coverage is reported).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from harness import config, report, scoring

TUNE = (config.myt_epoch("2026-09-10"), config.myt_epoch("2026-09-27"))
FINAL = (config.myt_epoch("2026-09-27"), config.myt_epoch("2026-10-14"))
SESSION_HEADS = [("Morning same day @07:15", "07:15", "morning", True),
                 ("Morning next day @18:05", "18:05", "morning", False),
                 ("Morning next day @12:00", "12:00", "morning", False),
                 ("Afternoon same day @12:00", "12:00", "afternoon", True),
                 ("Afternoon next day @18:05", "18:05", "afternoon", False),
                 ("Afternoon next day @20:00", "20:00", "afternoon", False)]


def evaluation_frame(ds, period):
    ev = ds.loc[ds["eval"] & ds.m_fresh_available]
    return ev.loc[(ev.index >= period[0]) & (ev.index < period[1])]


def _mask(ev, *series):
    m = pd.Series(True, index=ev.index)
    for s in series:
        m &= s.reindex(ev.index).notna()
    return m


def _events(r):
    if not r or not r.get("n_issues"):
        return "–"
    return (f"{r['episodes_caught']}/{r['episodes']} ep; hits {r['hits']}, miss {r['misses']}, "
            f"wrong {r['wrong_way']}, false {r['false_calls']} ({r['false_clusters']} cl)")


def _f(v, d=2):
    return "–" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


def class5_prior(ev, ds):
    prior = scoring.walk_forward_prior(ev, "y_class5", range(5), "y_arrival_complete", history=ds)
    prior.columns = [f"p_class5_{k}" for k in range(5)]
    prior["p_fall20"] = prior.p_class5_0 + prior.p_class5_1
    prior["p_rise20"] = prior.p_class5_3 + prior.p_class5_4
    prior["p_fall40"], prior["p_rise40"] = prior.p_class5_0, prior.p_class5_4
    return prior


def first_prior(ev, ds):
    prior = scoring.walk_forward_prior(ev, "y_first20", (-1, 0, 1), "y_first20_complete", history=ds)
    prior.columns = ["p_first_drop", "p_first_none", "p_first_rise"]
    return prior


def le70_prior(ev, ds):
    prior = scoring.walk_forward_prior(ev, "y_ride_le70", (0.0, 1.0), "y_ride_complete", history=ds)
    return prior[["prior_1.0"]].rename(columns={"prior_1.0": "p_ride_le70"})


def quick(ds, pred, period, replay=None) -> dict:
    """Compact metric set used for tuning."""
    replay = report.standard_replay(ds) if replay is None else replay
    ev = evaluation_frame(ds, period)
    out = {}
    if "pred_arrival" in pred:
        m = _mask(ev, replay.pred_arrival, ev.y_arrival, pred.pred_arrival)
        s = scoring.score_candidate(ev, pred, mask=m)
        out["arrival_mae"] = s["arrival"]["mae"]
        out["arrival_skill"] = s["arrival"]["skill_vs_persistence"]
        e = s["events20_point"]
        out["ev20_point"] = f"{e['episodes_caught']}/{e['episodes']} f{e['false_calls']}"
        m = _mask(ev, replay.pred_ride_mean, ev.y_ride_mean, pred.pred_ride_mean)
        s = scoring.score_candidate(ev, pred, mask=m)
        for h in ("ride_mean", "ride_min", "ride_max"):
            out[f"{h}_mae"] = s[h]["mae"]
    if "p_class5_0" in pred:
        m = _mask(ev, replay.p_fall20, ev.y_delta, pred.p_fall20)
        s = scoring.score_candidate(ev, pred, mask=m)
        out["class5_ll"] = s["class5"]["log_loss"]
        out["class5_brier"] = s["class5"]["brier"]
        out["fall20_brier"] = s["brier_fall20"]["brier"]
        out["rise20_brier"] = s["brier_rise20"]["brier"]
        e = s["events20_prob_argmax"]
        out["ev20_argmax"] = f"{e['episodes_caught']}/{e['episodes']} f{e['false_calls']} w{e['wrong_way']}"
        e = s["events20_prob_0.2"]
        out["ev20_p02"] = f"{e['episodes_caught']}/{e['episodes']} f{e['false_calls']} w{e['wrong_way']}"
        m = _mask(ev, replay.p_first_drop, ev.y_first20, pred.p_first_drop)
        s = scoring.score_candidate(ev, pred, mask=m)
        out["first_ll"] = s["first20"]["log_loss"]
        e = s["first20_events"]
        out["first_ev"] = f"{e['episodes_caught']}/{e['episodes']} f{e['false_calls']}"
    if "p_ride_le70" in pred:
        m = _mask(ev, replay.p_ride_le70, ev.y_ride_le70, pred.p_ride_le70)
        out["le70_brier"] = scoring.score_candidate(ev, pred, mask=m)["ride_le70"].get("brier", np.nan)
    for key in ("morning", "afternoon"):
        if f"pred_{key}" in pred:
            for label, at, k, same in SESSION_HEADS:
                if k != key:
                    continue
                r = session_metrics(ev, pred, replay, at, key, same)
                if r:
                    out[f"{key}@{at}"] = r["mae"]
    return out


def session_metrics(ev, pred, replay, at, key, same, require_incumbent=True):
    clock = pd.to_datetime(ev.index, unit="s", utc=True).tz_convert("Asia/Kuala_Lumpur").strftime("%H:%M")
    m = pd.Series(clock == at, index=ev.index) & ev[f"y_{key}_mean"].notna() & (ev[f"y_{key}_same_day"] == same)
    m &= pred[f"pred_{key}"].reindex(ev.index).notna()
    if require_incumbent:
        m &= replay[f"pred_{key}"].reindex(ev.index).notna()
    if not m.any():
        return None
    frame = ev.loc[m]
    cand = scoring.point_metrics(frame[f"y_{key}_mean"], pred.loc[frame.index, f"pred_{key}"], frame.y_ref)
    inc = scoring.point_metrics(frame[f"y_{key}_mean"], replay.loc[frame.index, f"pred_{key}"], frame.y_ref)
    return {"n": cand["n"], "mae": cand["mae"], "persistence": cand["persistence_mae"], "incumbent": inc["mae"]}


def full_table(ds, pred, period, replay=None, label="candidate") -> list[tuple]:
    """Rows (target, metric, persistence/prior, incumbent, candidate, n) on identical rows."""
    replay = report.standard_replay(ds) if replay is None else replay
    ev = evaluation_frame(ds, period)
    pers = scoring.persistence_predictions(ev)
    rows = []
    heads = [("+90 arrival", "pred_arrival", "arrival", "y_arrival"),
             ("ride mean +90..210", "pred_ride_mean", "ride_mean", "y_ride_mean"),
             ("ride min", "pred_ride_min", "ride_min", "y_ride_min"),
             ("ride max", "pred_ride_max", "ride_max", "y_ride_max")]
    for name, col, key, target in heads:
        if col not in pred:
            continue
        m = _mask(ev, replay[col], ev[target], pred[col])
        cov = float(pred[col].reindex(ev.index)[_mask(ev, replay[col], ev[target])].notna().mean())
        p = scoring.score_candidate(ev, pers[[col]], mask=m)
        r = scoring.score_candidate(ev, replay[[col]], mask=m)
        c = scoring.score_candidate(ev, pred[[col]], mask=m)
        n = c[key]["n"]
        rows.append((name, "MAE", _f(p[key]["mae"]), _f(r[key]["mae"]), _f(c[key]["mae"]), n))
        rows.append((name, "skill vs persistence", "0", _f(r[key]["skill_vs_persistence"], 3),
                     _f(c[key]["skill_vs_persistence"], 3), n))
        rows.append((name, "RMSE", _f(p[key]["rmse"]), _f(r[key]["rmse"]), _f(c[key]["rmse"]), n))
        rows.append((name, "MAE where |true change| >= 20", _f(p[key]["mae_change20"]), _f(r[key]["mae_change20"]),
                     _f(c[key]["mae_change20"]), c[key]["n_change20"]))
        if key == "ride_min":
            y = ev.y_ride_min[m]
            rows.append((name, "pinball q0.1", _f(scoring.pinball(y, pers.pred_ride_min[m], .1)),
                         _f(scoring.pinball(y, replay.pred_ride_min.reindex(ev.index)[m], .1)),
                         _f(scoring.pinball(y, pred.pred_ride_min.reindex(ev.index)[m], .1)), int(m.sum())))
        if key == "ride_max":
            y = ev.y_ride_max[m]
            rows.append((name, "pinball q0.9", _f(scoring.pinball(y, pers.pred_ride_max[m], .9)),
                         _f(scoring.pinball(y, replay.pred_ride_max.reindex(ev.index)[m], .9)),
                         _f(scoring.pinball(y, pred.pred_ride_max.reindex(ev.index)[m], .9)), int(m.sum())))
        if key == "arrival":
            for t in (20, 40):
                k = f"events{t}_point"
                rows.append((f"change >={t} from the +90 point", "events", _events(p[k]), _events(r[k]),
                             _events(c[k]), c[k]["n_issues"]))
        if cov < 1:
            rows.append((name, "candidate coverage of incumbent rows", "", "", _f(cov, 3), ""))
    if "p_fall20" in pred:
        m = _mask(ev, replay.p_fall20, ev.y_delta, pred.p_fall20)
        prior = class5_prior(ev, ds)
        cols = ["p_fall20", "p_rise20", "p_fall40", "p_rise40"] + [f"p_class5_{k}" for k in range(5)]
        q = scoring.score_candidate(ev, prior, mask=m)
        r = scoring.score_candidate(ev, replay[cols], mask=m)
        c = scoring.score_candidate(ev, pred[cols], mask=m)
        for t in (20, 40):
            for rule in ("argmax", "0.2"):
                k = f"events{t}_prob_{rule}"
                rows.append((f"change >={t} (probabilities, {rule})", "events", "0 calls (prior/persistence)",
                             _events(r[k]), _events(c[k]), c[k]["n_issues"]))
            for side in ("fall", "rise"):
                k = f"brier_{side}{t}"
                rows.append((f"P({side} >={t})", "Brier", _f(q[k]["brier"], 4), _f(r[k]["brier"], 4),
                             _f(c[k]["brier"], 4), c[k]["n"]))
        rows.append(("5-class arrival change", "Brier", _f(q["class5"]["brier"], 4), _f(r["class5"]["brier"], 4),
                     _f(c["class5"]["brier"], 4), c["class5"]["n"]))
        rows.append(("5-class arrival change", "log loss", _f(q["class5"]["log_loss"], 4),
                     _f(r["class5"]["log_loss"], 4), _f(c["class5"]["log_loss"], 4), c["class5"]["n"]))
    if "p_first_drop" in pred:
        cols = ["p_first_drop", "p_first_none", "p_first_rise"]
        m = _mask(ev, replay.p_first_drop, ev.y_first20, pred.p_first_drop)
        q = scoring.score_candidate(ev, first_prior(ev, ds), mask=m)
        r = scoring.score_candidate(ev, replay[cols], mask=m)
        c = scoring.score_candidate(ev, pred[cols], mask=m)
        rows.append(("first crossing >=20 in 90 min", "events (argmax)", _events(q["first20_events"]),
                     _events(r["first20_events"]), _events(c["first20_events"]), c["first20_events"]["n_issues"]))
        rows.append(("first crossing >=20 in 90 min", "Brier", _f(q["first20"]["brier"], 4),
                     _f(r["first20"]["brier"], 4), _f(c["first20"]["brier"], 4), c["first20"]["n"]))
        rows.append(("first crossing >=20 in 90 min", "log loss", _f(q["first20"]["log_loss"], 4),
                     _f(r["first20"]["log_loss"], 4), _f(c["first20"]["log_loss"], 4), c["first20"]["n"]))
    if "p_ride_le70" in pred:
        m = _mask(ev, replay.p_ride_le70, ev.y_ride_le70, pred.p_ride_le70)
        p = scoring.score_candidate(ev, pers[["p_ride_le70"]], mask=m)
        q = scoring.score_candidate(ev, le70_prior(ev, ds), mask=m)
        r = scoring.score_candidate(ev, replay[["p_ride_le70"]], mask=m)
        c = scoring.score_candidate(ev, pred[["p_ride_le70"]], mask=m)
        rows.append(("P(ride mean <=70)", "Brier (persistence 0/1 call)", _f(p["ride_le70"]["brier"], 4),
                     _f(r["ride_le70"]["brier"], 4), _f(c["ride_le70"]["brier"], 4), c["ride_le70"]["n"]))
        rows.append(("P(ride mean <=70)", "Brier (walk-forward prior)", _f(q["ride_le70"]["brier"], 4),
                     _f(r["ride_le70"]["brier"], 4), _f(c["ride_le70"]["brier"], 4), c["ride_le70"]["n"]))
        rows.append(("P(ride mean <=70)", "log loss (prior)", _f(q["ride_le70"]["log_loss"], 4),
                     _f(r["ride_le70"]["log_loss"], 4), _f(c["ride_le70"]["log_loss"], 4), c["ride_le70"]["n"]))
    for name, at, key, same in SESSION_HEADS:
        if f"pred_{key}" not in pred or f"pred_{key}" not in replay:
            continue
        r = session_metrics(ev, pred, replay, at, key, same)
        if r:
            rows.append((name, "MAE", _f(r["persistence"]), _f(r["incumbent"]), _f(r["mae"]), r["n"]))
    return rows


def markdown(rows, header=("Target", "Metric", "Persistence / prior", "Incumbent replay", "Candidate", "n")):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    return "\n".join(out)


def _direction_counts(issues, truth, call, d):
    ok = np.isfinite(truth) & np.isfinite(call)
    issues, truth, call = issues[ok], truth[ok], call[ok]
    event = truth == d
    lookup = dict(zip(issues, call))
    runs = scoring._runs(issues, event)
    caught = sum(any(lookup[e] == d for e in run) for run in runs)
    false = (~event) & (call == d)
    return (f"{caught}/{len(runs)} ep; hits {int((event & (call == d)).sum())}, miss {int((event & (call == 0)).sum())}, "
            f"wrong {int((event & (call == -d)).sum())}, false {int(false.sum())} ({len(scoring._runs(issues, false))} cl)")


def direction_table(ds, pred, period, replay=None) -> list[tuple]:
    """Falls and rises separately at >=20 and >=40, from the +90 point and from the probabilities (argmax, 0.2).

    A false call is a call in that direction when that direction's event did not happen; a wrong-way
    issue is an event of that direction met by the opposite call.
    """
    replay = report.standard_replay(ds) if replay is None else replay
    ev = evaluation_frame(ds, period)
    m = _mask(ev, replay.pred_arrival, replay.p_fall20, ev.y_delta, pred.pred_arrival, pred.p_fall20)
    ev = ev.loc[m]
    issues, ref = ev.index.to_numpy(np.int64), ev.y_ref.to_numpy(float)
    rows = []
    for t in (20, 40):
        truth = scoring.truth_direction(ev.y_delta, t)
        for d, side in ((-1.0, "fall"), (1.0, "rise")):
            point_inc = scoring.calls_from_point(replay.pred_arrival.reindex(ev.index), ref, t)
            point_can = scoring.calls_from_point(pred.pred_arrival.reindex(ev.index), ref, t)
            rows.append((f"{side} >={t}", "+90 point (Ridge | candidate)", _direction_counts(issues, truth, point_inc, d),
                         _direction_counts(issues, truth, point_can, d), len(ev)))
            for rule in ("argmax", "0.2"):
                p_inc = scoring.calls_from_probs(replay[f"p_fall{t}"].reindex(ev.index),
                                                 replay[f"p_rise{t}"].reindex(ev.index), rule)
                p_can = scoring.calls_from_probs(pred[f"p_fall{t}"].reindex(ev.index),
                                                 pred[f"p_rise{t}"].reindex(ev.index), rule)
                rows.append((f"{side} >={t}", f"probabilities {rule} (logistic | candidate)",
                             _direction_counts(issues, truth, p_inc, d), _direction_counts(issues, truth, p_can, d),
                             len(ev)))
    return rows
