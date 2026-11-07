"""Live first >=20 crossing within 90 minutes: recall-weighted multinomial logistic on P1 inputs (v25.2).

Replaces fresh_event_model_v2 (HGB v2) as the source of the published firstCrossingEventForecast
and the RLCD first20 fields. fresh_event_model_v2 stays importable for its tests and pickles;
first_crossing_live / research/first_crossing_hazard are untouched (cycling_window_live uses them).

Research and decision (2026-10-14)
* research/backtest/SELECTION_GUIDE.md, deploy candidate 1: the C = 0.001 multinomial logistic on
  the P1 inputs (research/backtest/linear) replaces HGB v2.
* research/backtest/eager/frozen.json, variant first20_p1_f12_r6. The owner asked the event heads to
  warn more eagerly ("a right prediction in PM2.5 drop is an opportunity to go cycling outdoors").
  On the tune folds (09-10..09-23) the objective 5 x missed falls + 2.5 x missed rises + false-call
  clusters + wrong-way clusters chose training-loss class weights drop 12, no crossing 1, rise 6
  (divided by their mean). The eagerness lives in the training loss only: the published outcome
  is the literal argmax of the published probabilities, with no threshold or gate. The
  probabilities are therefore recall-weighted, NOT calibrated (calibrated: false and
  recallWeighting are in every output).

Recipe (the research code, reproduced; parity: research/first20_logistic_v25_2/parity_check.py)
* Inputs (82, research linear.core groups base/controls/time/spatial): the 28 TTDI sensor columns
  and log(ref + 5); the 4 TTDI mean-reversion controls; 1st and 2nd daily harmonics of the MYT
  clock and a weekend flag; the 44 neighbour columns of P1. Rows come from spatial_point_model's
  feature builder (same receipt clocks, same cached rows, same live path).
* Preprocessing fit on the training window only: median fill, clip to the training 0.5..99.5
  percentiles, standardise.
* LogisticRegression(C=0.001, lbfgs, max_iter=2000) with the recall sample weights; probability
  floor (1 - 3e-4) p + 1e-4 so an unseen class never gets probability 0.
* Training (nightly, background only): cutoff = Malaysia midnight of the issue day. Every usable
  5-minute grid issue in the 28 days before the cutoff fits the preprocessing; those whose
  first-crossing label closed before the cutoff train the classifier.
* Label: production fresh_event_model's first crossing (six interpolated 15-minute median proxies
  at +15..+90 min; the first with |proxy - ref| >= 20 gives the sign; none -> 0), ref = the fresh
  5-minute reference s_freshPm25.

Unavailable with a reason, never substituted: no fresh reference, a reference that differs from
the dashboard's, an unreadable neighbour archive, no neighbour bucket within 90 minutes, no issued
wind within 2 hours, or no model fitted for the issue day.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import closing
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import pickle
import threading
import time
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

import fresh_numeric_model_v4 as numeric_targets
from forecast_clock import point_weights
import spatial_final_models as M
import spatial_point_model as S

ROOT = Path(__file__).resolve().parent
PRIVATE = ROOT / "private"
MYT = S.MYT
MODEL_VERSION = "p1_logistic_first20_90min_recall_f12_r6_v1"
ADAPTER_VERSION = "p1_logistic_first20_live_adapter_v1"
ARTIFACT_NAME = "first20-logistic-model-v1.pkl"
STATUS_NAME = "first20-logistic-status-v1.json"
FEATURE_CACHE_NAME = "first20-logistic-features-v1.pkl"
ISSUE_NAME = "first20-logistic-issued-v1.jsonl"
RESEARCH_VARIANT = "first20_p1_f12_r6"
TRAINING_CADENCE = "aligned_five_minute_issue_grid_v1"

CLASSES = (-1, 0, 1)                      # drop, no crossing, rise (research targets.SPECS['first20'])
CLASS_NAMES = ("drop", "none", "rise")
CLASS_WEIGHTS = {-1: 12.0, 0: 1.0, 1: 6.0}
LOGISTIC_C = 0.001
MAX_ITER = 2000
PROBABILITY_FLOOR = 1e-4
CLIP_QUANTILES = (0.005, 0.995)
REF_OFFSET = 5.0
THRESHOLD = 20.0
LEADS_MINUTES = (15, 30, 45, 60, 75, 90)
ORIGIN_SECONDS = 300
LOOKBACK_DAYS = 28
MINIMUM_TRAINING_ISSUES = 300
FIT_CPU_THREADS = 1
VALID_SECONDS = 5400

TIME_COLUMNS = ("lin_h1s", "lin_h1c", "lin_h2s", "lin_h2c", "lin_weekend")
SPATIAL_COLUMNS = tuple(M.SP_STATION + M.SP_NET + M.SP_WIND)
INPUT_COLUMNS = tuple([*M.SENSOR28, "lin_logref", *M.CONTROLS, *TIME_COLUMNS, *SPATIAL_COLUMNS])
IDENTITY_FILES = ("first_crossing_logistic_model.py", "spatial_point_model.py", "spatial_final_models.py",
                  "spatial_live_features.py", "fresh_sensor_features.py", "fresh_numeric_model_v4.py",
                  "forecast_clock.py", "collection_quality.py", "forecast_payload.py")

RECALL_WEIGHTING = {
    "classWeights": {"fall": CLASS_WEIGHTS[-1], "noChange": CLASS_WEIGHTS[0], "rise": CLASS_WEIGHTS[1]},
    "normalisation": "sample weights divided by their mean over the training rows",
    "appliedIn": "training_loss_only",
    "selectionObjective": "5 x missed >=20 fall episodes + 2.5 x missed >=20 rise episodes + false-call clusters "
                          "+ wrong-way clusters (lower is better), tune folds 2026-09-10..09-23",
    "researchVariant": RESEARCH_VARIANT,
    "evidence": "research/backtest/eager/frozen.json (2026-10-14)",
    "note": "probabilities are recall-weighted, not calibrated; the published outcome is the literal argmax",
}

PERFORMANCE_EVIDENCE = {
    "state": "development_only", "prospectivelyValidated": False, "modelVersion": MODEL_VERSION,
    "evidenceSource": "research/backtest/eager (frozen.json, final_scores.md, issued_scores.md), 2026-10-14",
    "objective": RECALL_WEIGHTING["selectionObjective"],
    "selection": "class weights chosen on the 2026-09-10..09-23 tune folds; fall weight 12 is the top edge of the grid",
    "finalSpanPostSelection": {"dates": ["2026-09-24", "2026-10-13"], "caseCount": 5553,
                               "fallsCaught": "27/38", "risesCaught": "9/26", "falseCallClusters": 66,
                               "wrongWayClusters": 25, "objective": 188.5, "previousHgbV2Objective": 241.0,
                               "neverCallObjective": 255.0, "alwaysFallObjective": 128.0,
                               "episodeErrors": 119, "previousHgbV2EpisodeErrors": 81, "neverCallEpisodeErrors": 64,
                               "logLoss": 0.5625, "previousHgbV2LogLoss": 0.5738, "classPriorLogLoss": 0.435,
                               "callShare": 0.22, "callShareOct6To13": 0.40, "previousHgbV2CallShareOct6To13": 0.06},
    "issuedClocks": {"dates": ["2026-10-08", "2026-10-13"], "caseCount": 1438, "fallsCaught": "7/11",
                     "risesCaught": "3/8", "objective": 64.5, "previousHgbV2AsIssuedObjective": 72.5,
                     "neverCallObjective": 75.0, "callShare": 0.26},
    "overlappingIssues": True,
    "summary": ("Recall-weighted (drop 12, rise 6): it called a crossing on 22% of Sep 24-Oct 13 issues and 40% of "
                "Oct 6-13 issues (HGB v2: 6%). On Sep 24-Oct 13 it caught 27 of 38 falls (HGB v2: 7) with 66 "
                "false-call clusters; the owner's objective improved from 241 to 188.5, but a permanent fall alarm "
                "scores 128 on the same objective, so this is the chosen trade-off, not proven warning skill. "
                "Probabilities are not calibrated."),
}

_LOCK = threading.RLock()
_FIT_LOCK = threading.Lock()
_RECORD_LOCK = threading.Lock()
_LOADED = {}
_IDENTITY = {}
_RECORDED = OrderedDict()
_ISSUE_PATHS_READ = set()


# ----------------------------------------------------------------------------- recipe
def selected_policy():
    return {"modelVersion": MODEL_VERSION, "researchVariant": RESEARCH_VARIANT,
            "learner": "LogisticRegression(C=0.001, lbfgs, max_iter=2000), multinomial over drop/none/rise",
            "parameters": {"C": LOGISTIC_C, "max_iter": MAX_ITER, "solver": "lbfgs"},
            "inputColumnCount": len(INPUT_COLUMNS), "inputColumnsSha256": S._sha_text(list(INPUT_COLUMNS)),
            "preprocessing": "median fill, clip to the training 0.5-99.5 percentiles, standardise (training window only)",
            "probabilityFloor": PROBABILITY_FLOOR, "recallWeighting": dict(RECALL_WEIGHTING),
            "target": "first_sampled_crossing_within90_minutes", "changeThresholdUgM3": THRESHOLD,
            "thresholdComparison": "at_least", "targetSampleLeadsMinutes": list(LEADS_MINUTES),
            "learnedProbabilityHorizonsMinutes": [90],
            "probabilityMethod": "direct_three_class_first90_outcome_predict_proba_with_uniform_floor",
            "trainingCadence": TRAINING_CADENCE, "trainingLookbackDays": LOOKBACK_DAYS,
            "trainingCutoff": "strict_completed_labels_before_issue_day_midnight_Asia_Kuala_Lumpur",
            "directionDecision": "three_class_argmax_ties_unresolved", "calibrated": False,
            "numericalSelectionApplied": False, "publishingThresholdApplied": False}


def class_sample_weights(y):
    """Recall weights by class, divided by their mean (research eager.heads.sample_weights)."""
    y = np.asarray(y)
    w = np.ones(len(y))
    for cls, weight in CLASS_WEIGHTS.items():
        w[y == cls] = weight
    return w / w.mean() if len(w) else w


class Prep:
    """Median fill, percentile clip and standardisation fit on training inputs only.

    research/backtest/linear/core.Prep, reproduced operation for operation (column statistics are
    computed per column, so fitting on the 82 used columns equals fitting on the research's wider
    design frame).
    """

    def __init__(self, x: pd.DataFrame, lo=CLIP_QUANTILES[0], hi=CLIP_QUANTILES[1]):
        self.columns = list(x.columns)
        v = x.to_numpy(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(v, axis=0)
        self.med = np.where(np.isfinite(med), med, 0.0)
        filled = np.where(np.isfinite(v), v, self.med)
        self.lo = np.quantile(filled, lo, axis=0)
        self.hi = np.quantile(filled, hi, axis=0)
        clipped = np.clip(filled, self.lo, self.hi)
        self.mu = clipped.mean(axis=0)
        sd = clipped.std(axis=0)
        self.sd = np.where(sd > 1e-12, sd, 1.0)

    def __call__(self, x: pd.DataFrame) -> np.ndarray:
        v = x[self.columns].to_numpy(float)
        v = np.where(np.isfinite(v), v, self.med)
        return (np.clip(v, self.lo, self.hi) - self.mu) / self.sd


def time_block(issues) -> pd.DataFrame:
    """Daily harmonics of the MYT clock and a weekend flag (research linear.core.time_block)."""
    local = np.asarray(issues, np.int64) + 8 * 3600
    hour = (local % 86400) / 3600.0
    dow = (local // 86400 + 3) % 7                    # 1970-01-01 was a Thursday (Monday = 0)
    w = 2 * np.pi * hour / 24
    return pd.DataFrame({"lin_h1s": np.sin(w), "lin_h1c": np.cos(w), "lin_h2s": np.sin(2 * w),
                         "lin_h2c": np.cos(2 * w), "lin_weekend": (dow >= 5).astype(float)})


def design(frame: pd.DataFrame) -> pd.DataFrame:
    """The 82 raw model inputs for feature rows indexed by issue epoch (cache or live columns)."""
    if any(c not in frame for c in M.CONTROLS):
        frame = M.add_controls(frame)
    issues = frame.index.to_numpy(np.int64)
    raw = frame[[*M.SENSOR28, *M.CONTROLS, *SPATIAL_COLUMNS]].astype(float).reset_index(drop=True)
    ref = frame["s_freshPm25"].to_numpy(float)
    raw["lin_logref"] = np.log(np.maximum(ref, 0) + REF_OFFSET)
    out = pd.concat([raw, time_block(issues)], axis=1)
    out.index = frame.index
    return out[list(INPUT_COLUMNS)]


def fit_classifier(x: np.ndarray, y: np.ndarray):
    y = np.asarray(y).astype(int)
    lr = LogisticRegression(C=LOGISTIC_C, max_iter=MAX_ITER)
    with threadpool_limits(limits=FIT_CPU_THREADS), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lr.fit(x, y, sample_weight=class_sample_weights(y))
    return lr, [{"category": w.category.__name__, "message": str(w.message)[:200]} for w in caught]


def predict_frame(model: dict, x_raw: pd.DataFrame) -> np.ndarray:
    """(n, 3) probabilities in CLASSES order, floored exactly as the research predict_logistic."""
    z = model["prep"](x_raw)
    lr = model["classifier"]
    out = np.zeros((len(z), len(CLASSES)))
    with threadpool_limits(limits=FIT_CPU_THREADS):
        proba = lr.predict_proba(z)
    for j, cls in enumerate(lr.classes_):
        out[:, CLASSES.index(int(cls))] = proba[:, j]
    return (1 - PROBABILITY_FLOOR * len(CLASSES)) * out + PROBABILITY_FLOOR


# ----------------------------------------------------------------------------- labels
def first_crossing_labels(buckets: dict, issues, references) -> pd.DataFrame:
    """Production first-crossing labels (fresh_event_model / research harness.targets.first_crossing).

    ``buckets`` maps a 15-minute bucket end epoch to its eligible right-closed median. The label is
    NaN when the reference or any of the six proxies is missing; ``y_first20_complete`` is the latest
    bucket it needs.
    """
    issues = np.asarray(issues, dtype=np.int64)
    references = np.asarray(references, dtype=float)
    cause = np.full(len(issues), np.nan)
    complete = np.zeros(len(issues), dtype=np.int64)
    for index, (issue, reference) in enumerate(zip(issues, references)):
        origin = int(issue) // 900 * 900
        values, completions = [], []
        for lead in LEADS_MINUTES:
            weights = point_weights(int(issue - origin + lead * 60))
            labels = [origin + offset * 900 for offset in weights]
            completions.append(max(labels))
            if any(label not in buckets for label in labels):
                values.append(np.nan)
            else:
                values.append(sum(buckets[label] * weight for label, weight in zip(labels, weights.values())))
        complete[index] = max(completions)
        if not np.isfinite(reference) or reference < 0 or not np.isfinite(values).all():
            continue
        delta = np.asarray(values) - reference
        crossed = np.flatnonzero(np.abs(delta) >= THRESHOLD)
        cause[index] = int(np.sign(delta[crossed[0]])) if len(crossed) else 0
    return pd.DataFrame({"y_first20": cause, "y_first20_complete": complete},
                        index=pd.Index(issues, name="issue_epoch"))


def bucket_map(rows) -> dict:
    series = numeric_targets.observed_series(rows)
    return {int(t): float(v) for t, v in zip(series.index.asi8 // 10**9, series.to_numpy(float)) if np.isfinite(v)}


# ----------------------------------------------------------------------------- identity and files
def _signature(names):
    return tuple((name, (ROOT / name).stat().st_mtime_ns, (ROOT / name).stat().st_size) for name in names)


def model_identity():
    """Identity of the recipe; any source change refits once at the next start."""
    signature = _signature(IDENTITY_FILES)
    with _LOCK:
        if _IDENTITY.get("model", (None,))[0] == signature:
            return _IDENTITY["model"][1]
    identity = {"modelVersion": MODEL_VERSION, "adapterVersion": ADAPTER_VERSION,
                "selectedPolicy": selected_policy(), "minimumTrainingIssues": MINIMUM_TRAINING_ISSUES,
                "featureIdentity": S.feature_identity(), "scikitLearnVersion": sklearn.__version__,
                "sourceSha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in IDENTITY_FILES}}
    identity["identitySha256"] = S._sha_text(identity)
    with _LOCK:
        _IDENTITY["model"] = (signature, identity)
    return identity


def _paths(model_dir=None):
    directory = Path(model_dir) if model_dir is not None else PRIVATE
    return directory / ARTIFACT_NAME, directory / STATUS_NAME, directory / FEATURE_CACHE_NAME


# ----------------------------------------------------------------------------- training rows
def training_origins(cutoff_epoch):
    """5-minute grid issues in (cutoff - 28 d, cutoff): the research training batch."""
    cutoff = int(cutoff_epoch)
    lower = cutoff - LOOKBACK_DAYS * 86400
    first = (lower // ORIGIN_SECONDS + 1) * ORIGIN_SECONDS
    return np.arange(first, cutoff, ORIGIN_SECONDS, dtype=np.int64)


def _load_own_cache(path, identity):
    try:
        value = pickle.loads(Path(path).read_bytes())
        if (isinstance(value, dict) and value.get("identity") == identity
                and isinstance(value.get("frame"), pd.DataFrame) and list(value["frame"].columns) == list(S.CACHE_COLUMNS)):
            return value["frame"]
    except (OSError, EOFError, ValueError, TypeError, AttributeError, ImportError, pickle.UnpicklingError):
        pass
    return S._frame_from_rows([], [])


def training_features(db_path, cutoff_epoch, *, archive_path=None, model_dir=None):
    """Feature rows of every 5-minute training issue.

    Rows come from spatial_point_model's builder: first this model's own cache, then rows the
    spatial point model already cached (its 15-minute origins, same builder and identity), and
    only the remainder is computed. A row depends only on inputs visible by its issue, so a row
    computed after the issue equals the row the live path computed at the issue.
    """
    cutoff = int(cutoff_epoch)
    origins = training_origins(cutoff)
    identity = S.feature_identity()
    _, _, path = _paths(model_dir)
    cached = _load_own_cache(path, identity)
    wanted = pd.Index(origins)
    missing = wanted.difference(cached.index)
    seeded = 0
    started = time.perf_counter()
    if len(missing):
        shared = S._load_feature_cache(model_dir)
        reuse = missing.intersection(shared.index)
        if len(reuse):
            cached = pd.concat([cached, shared.loc[reuse]]) if len(cached) else shared.loc[reuse]
            seeded = int(len(reuse))
        missing = wanted.difference(cached.index)
    computed = int(len(missing))
    if computed:
        fresh = S.compute_training_features(db_path, [int(o) for o in missing], archive_path=archive_path)
        cached = pd.concat([cached, fresh]) if len(cached) else fresh
    if computed or seeded:
        cached = cached.loc[cached.index >= cutoff - (LOOKBACK_DAYS + 1) * 86400]
        cached = cached.loc[~cached.index.duplicated(keep="last")].sort_index()
        S._atomic_write(path, pickle.dumps({"identity": identity, "featureVersion": S.FEATURE_VERSION,
                                            "originSeconds": ORIGIN_SECONDS, "frame": cached},
                                           protocol=pickle.HIGHEST_PROTOCOL))
    frame = cached.reindex(origins)
    frame["m_fresh_available"] = frame["m_fresh_available"].fillna(False).astype(bool)
    info = {"origins": int(len(origins)), "cachedOrigins": int(len(origins) - seeded - computed),
            "seededFromSpatialPointCache": seeded, "computedOrigins": computed,
            "computeSeconds": round(time.perf_counter() - started, 3)}
    return frame, info


def training_frame(db_path, cutoff_epoch, *, archive_path=None, model_dir=None):
    """Feature rows plus first-crossing labels (and their closing clocks) for the training window."""
    cutoff = int(cutoff_epoch)
    features, info = training_features(db_path, cutoff, archive_path=archive_path, model_dir=model_dir)
    with closing(S._connect_ro(db_path)) as conn:
        conn.execute("BEGIN")
        rows = S.label_rows(conn, cutoff - LOOKBACK_DAYS * 86400 - S.SENSOR_WINDOW_SECONDS, cutoff)
    labels = first_crossing_labels(bucket_map(rows), features.index.to_numpy(np.int64),
                                   features["s_freshPm25"].to_numpy(float))
    return features.join(labels), info


# ----------------------------------------------------------------------------- fitting
def fit(history: pd.DataFrame, cutoff_epoch):
    """Fit preprocessing and the weighted classifier at the cutoff; refuses thin training support.

    Returns (model or None, metadata).
    """
    cutoff = int(cutoff_epoch)
    idx = history.index.to_numpy(np.int64)
    window = (idx > cutoff - LOOKBACK_DAYS * 86400) & (idx < cutoff) & M.usable(history)
    batch = history.loc[window]
    y = batch["y_first20"].to_numpy(float)
    closed = batch["y_first20_complete"].to_numpy(np.int64) < cutoff
    train = closed & np.isin(y, np.asarray(CLASSES, float)) & np.isfinite(batch["s_freshPm25"].to_numpy(float))
    yy = y[train].astype(int)
    support = {name: int((yy == cls).sum()) for cls, name in zip(CLASSES, CLASS_NAMES)}
    base = {"trainingIssueCount": int(train.sum()), "preprocessingIssueCount": int(len(batch)),
            "trainingSupport": support}
    if int(train.sum()) < MINIMUM_TRAINING_ISSUES or len(set(yy)) < 2:
        return None, {**base, "available": False, "reason": "insufficient_completed_training_issues"}
    x_raw = design(batch)
    prep = Prep(x_raw)
    x = prep(x_raw)[train]
    classifier, fit_warnings = fit_classifier(x, yy)
    used = batch.loc[train]
    weights = class_sample_weights(yy)
    model = {"prep": prep, "classifier": classifier, "columns": list(INPUT_COLUMNS)}
    days = {datetime.fromtimestamp(int(i), MYT).date() for i in used.index}
    return model, {**base, "available": True, "reason": "available",
                   "trainingDistinctIssueDays": len(days),
                   "trainingMinIssueEpoch": int(used.index.min()), "trainingMaxIssueEpoch": int(used.index.max()),
                   "maxTrainingOutcomeEpoch": int(used["y_first20_complete"].max()),
                   "observedTrainingClasses": [int(c) for c in classifier.classes_],
                   "normalisedClassWeights": {name: float(weights[yy == cls][0]) if (yy == cls).any() else None
                                              for cls, name in zip(CLASSES, CLASS_NAMES)},
                   "lbfgsIterations": [int(n) for n in np.ravel(classifier.n_iter_)],
                   "fitWarnings": fit_warnings,
                   "convergenceWarningPresent": any(w["category"] == "ConvergenceWarning" for w in fit_warnings),
                   "independentIssues": False, "overlappingTrainingIssues": True,
                   "trainingDataHash": S._frame_hash(used[[*S.CACHE_COLUMNS, "y_first20", "y_first20_complete"]])}


def load_artifact(model_dir=None):
    path, _, _ = _paths(model_dir)
    try:
        stamp = (path.stat().st_mtime_ns, path.stat().st_size)
    except OSError:
        return None
    identity = model_identity()
    key = str(path.resolve())
    with _LOCK:
        cached = _LOADED.get(key)
        if cached and cached[0] == stamp and cached[1].get("identity") == identity:
            return cached[1]
    try:
        value = pickle.loads(path.read_bytes())
    except (OSError, EOFError, ValueError, TypeError, AttributeError, ImportError, pickle.UnpicklingError):
        return None
    if (not isinstance(value, dict) or value.get("identity") != identity or value.get("modelVersion") != MODEL_VERSION
            or value.get("adapterVersion") != ADAPTER_VERSION):
        return None
    with _LOCK:
        _LOADED[key] = (stamp, value)
    return value


def refresh_model(db_path, issue_epoch=None, *, archive_path=None, model_dir=None):
    """Background-worker entry: one fit per Malaysia day (and after any recipe change)."""
    issue = int(time.time()) if issue_epoch is None else int(issue_epoch)
    cutoff = S.midnight_epoch(issue)
    artifact_path, status_path, _ = _paths(model_dir)
    with _FIT_LOCK:
        prior = load_artifact(model_dir)
        if (prior is not None and prior["trainingCutoffEpoch"] == cutoff
                and prior["metadata"].get("reason") != "background_fit_unavailable"):
            return dict(prior["metadata"], cacheReloaded=True)
        started = time.perf_counter()
        identity = model_identity()
        metadata = {"modelVersion": MODEL_VERSION, "adapterVersion": ADAPTER_VERSION, "trainingCutoffEpoch": cutoff,
                    "modelIdentity": identity, "fitCpuThreads": FIT_CPU_THREADS, "researchVariant": RESEARCH_VARIANT,
                    "trainingFitCadence": TRAINING_CADENCE, "recallWeighting": dict(RECALL_WEIGHTING),
                    "calibrated": False, "prospectivelyValidated": False, "numericalSelectionApplied": False,
                    "weatherFeaturesUsed": True, "issuedWindInputCount": 16, "neighbourPmInputsUsed": True}
        model = None
        try:
            history, cache = training_frame(db_path, cutoff, archive_path=archive_path, model_dir=model_dir)
            metadata["featureCache"] = cache
            model, support = fit(history, cutoff)
            metadata.update(support)
        except Exception as error:
            metadata.update(available=False, reason="background_fit_unavailable",
                            errorType=type(error).__name__, error=str(error)[:300])
            model = None
        metadata["fittedAtEpoch"] = int(time.time())
        metadata["fitElapsedSeconds"] = round(time.perf_counter() - started, 3)
        value = {"modelVersion": MODEL_VERSION, "adapterVersion": ADAPTER_VERSION, "identity": identity,
                 "trainingCutoffEpoch": cutoff, "metadata": metadata, "model": model}
        S._atomic_write(artifact_path, pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        S._atomic_write(status_path, (json.dumps(S._json_value(metadata), indent=2, allow_nan=False) + "\n").encode("utf-8"))
        with _LOCK:
            _LOADED[str(artifact_path.resolve())] = ((artifact_path.stat().st_mtime_ns, artifact_path.stat().st_size), value)
        return dict(metadata, cacheReloaded=False)


# ----------------------------------------------------------------------------- inference
def _qualification(result):
    """No prospective qualification yet: the argmax is a diagnostic direction (as HGB v2's was)."""
    result.update(qualificationVersion="fresh_actual_cadence_event_qualification_v1",
                  prospectivelyValidated=False, trainingFitCadence=TRAINING_CADENCE,
                  trainingCadenceMatched=False, unvalidatedLiveCadence=True,
                  liveIssueOffsetSeconds=int(result["forecastIssuedEpoch"]) % 900,
                  probabilityRole="recall_weighted_uncalibrated_experimental_diagnostic",
                  operationalDecisionRule="no_direction_until_actual_cadence_prospective_qualification",
                  directionStatus="unqualified_experimental",
                  qualification={"state": "unqualified_experimental", "eligible": False,
                                 "operationalUseEligible": False, "actualCadenceValidated": False,
                                 "calibrationValidated": False, "trainingCadenceMatched": False,
                                 "reasons": ["no_actual_cadence_prospective_qualification",
                                             "recall_weighted_probabilities_not_calibrated",
                                             "dense_overlapping_training_issues_are_not_independent_events"],
                                 "evidenceRole": "chronological_development_and_originally_issued_diagnostics"})
    result["diagnosticDirection"] = result.get("direction", "unresolved")
    result["diagnosticDirectionAvailable"] = bool(result.get("available"))
    result["direction"] = "unresolved"
    return result


def _base(issue):
    return {"available": False, "modelVersion": MODEL_VERSION, "adapterVersion": ADAPTER_VERSION,
            "experimental": True, "calibrated": False, "recallWeighting": dict(RECALL_WEIGHTING),
            "forecastIssuedEpoch": issue, "validUntilEpoch": issue + VALID_SECONDS,
            "changeThresholdUgM3": THRESHOLD, "thresholdComparison": "at_least",
            "target": "first_sampled_crossing_within90_minutes", "direction": "unresolved",
            "directionDecision": "three_class_argmax_ties_unresolved", "directionTieTolerance": 0.0,
            "publishingThresholdApplied": False, "horizons": [], "magnitudeEstimates": {},
            "selectedPolicy": selected_policy(), "displayText": None,
            "weatherFeaturesUsed": True, "issuedWindInputCount": 16, "neighbourPmInputsUsed": True,
            "issuedWindRole": "16_inputs_from_issued_wind_upwind_neighbour_pm_advection_and_eta"}


def _unavailable(result, reason, *details):
    return _qualification({**result, "available": False, "reason": reason,
                           "unavailableReasons": [reason, *[d for d in details if d]]})


def _artifact_problem(artifact, issue, allow_reconstruction):
    if artifact is None:
        return "background_model_collecting"
    metadata = artifact.get("metadata") or {}
    if artifact.get("trainingCutoffEpoch") != S.midnight_epoch(issue):
        return "first20_model_not_fitted_for_issue_day"
    if not metadata.get("available") or artifact.get("model") is None:
        return metadata.get("reason") or "first20_model_training_unavailable"
    fitted = S._finite(metadata.get("fittedAtEpoch"))
    if fitted is None:
        return "model_fit_timestamp_unavailable"
    if fitted > issue and not allow_reconstruction:
        return "model_not_yet_fitted_at_issue"
    if (S._finite(metadata.get("maxTrainingOutcomeEpoch")) or math.inf) >= artifact["trainingCutoffEpoch"]:
        return "training_label_cutoff_violation"
    return None


def predict_event(db_path, rows, issue_epoch, sensor_watermark_epoch, reference_pm, *, archive_path=None,
                  model_dir=None, artifact=None, allow_reconstruction=False):
    """Read-only inference for one issue: the literal drop / none / rise probabilities and argmax."""
    issue = int(issue_epoch)
    result = _base(issue)
    try:
        ref = float(reference_pm)
        if not math.isfinite(ref) or ref < 0:
            return _unavailable(result, "fresh_sensor_reference_unavailable")
        result["referencePm"] = ref
        watermark = int(sensor_watermark_epoch)
        if watermark > issue:
            return _unavailable(result, "invalid_sensor_watermark")
        live = S.live_issue_features(db_path, rows, issue, archive_path=archive_path)
        lineage = live["lineage"]
        result["freshFeatureProvenance"] = lineage
        if live["features"] is None:
            return _unavailable(result, live["reason"], live.get("errorType"))
        features = live["features"]
        if not math.isclose(float(features["s_freshPm25"]), ref, rel_tol=0, abs_tol=1e-7):
            return _unavailable(result, "fresh_sensor_reference_mismatch")
        artifact = load_artifact(model_dir) if artifact is None else artifact
        problem = _artifact_problem(artifact, issue, allow_reconstruction)
        if problem:
            return _unavailable(result, problem)
        model = artifact["model"]
        frame = pd.DataFrame([features], index=pd.Index([issue], name="issue_epoch"))
        x_raw = design(frame)
        p = predict_frame(model, x_raw)[0]
        if p.shape != (3,) or not np.isfinite(p).all() or (p < 0).any() or abs(float(p.sum()) - 1) > 1e-9:
            return _unavailable(result, "invalid_learned_probability_distribution")
        ties = np.flatnonzero(p == p.max())
        direction = ("drop" if len(ties) == 1 and ties[0] == 0 else "rise" if len(ties) == 1 and ties[0] == 2
                     else "unresolved")
        values = [float(v) if math.isfinite(v) else None for v in x_raw.to_numpy(float)[0]]
        provenance_max = S._finite(lineage.get("featureSourceMaxEpoch"))
        training = {k: v for k, v in artifact["metadata"].items() if k != "modelIdentity"}
        result.update(training, available=True, reason="available", unavailableReasons=[],
                      calibrated=False, recallWeighting=dict(RECALL_WEIGHTING),
                      direction=direction, modelIdentity=artifact["identity"],
                      trainingCutoffEpoch=artifact["trainingCutoffEpoch"],
                      probabilityDrop=float(p[0]), probabilityNoCrossing=float(p[1]), probabilityRise=float(p[2]),
                      probabilityAnyCrossing=float(p[0] + p[2]),
                      horizons=[{"leadMinutes": 90, "probabilityDrop": float(p[0]),
                                 "probabilityNoCrossing": float(p[1]), "probabilityRise": float(p[2])}],
                      featureOriginEpoch=issue // 900 * 900, sensorWatermarkEpoch=watermark,
                      freshReferenceEpoch=lineage.get("freshReferenceEpoch"),
                      featureSourceMaxEpoch=provenance_max, inputMaxMeasurementEpoch=provenance_max,
                      sourceRole="fresh_sensor_and_neighbour_snapshot_for_publication",
                      impossibleDropSupport=bool(ref < THRESHOLD and p[0] > 0),
                      featureColumns=list(INPUT_COLUMNS), featureValues=values,
                      featureValuesSha256=hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest(),
                      reconstructed=bool(allow_reconstruction))
        return _qualification(result)
    except Exception as error:
        return _unavailable(result, "first20_inference_unavailable", type(error).__name__, str(error)[:200])


def record_issued_prediction(result, *, output_path=None, recorded_epoch=None, input_lineage=None):
    """Append a genuinely published first-crossing forecast once; call only after real publication."""
    recorded = int(time.time()) if recorded_epoch is None else int(recorded_epoch)
    issue = int(result["forecastIssuedEpoch"])
    if not 0 <= recorded - issue <= 120:
        raise ValueError("Only fresh actual publication may be recorded as originally issued")
    path = Path(output_path) if output_path is not None else PRIVATE / ISSUE_NAME
    identity_hash = hashlib.sha256(json.dumps(S._json_value(result.get("modelIdentity")), sort_keys=True,
                                              separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    resolved = str(path.resolve())
    key = (resolved, issue, identity_hash, result.get("featureValuesSha256"), result.get("reason"))
    lineage = dict(input_lineage) if input_lineage is not None else {"status": "not_archived"}
    record = {**S._json_value(result), "recordedEpoch": recorded, "publishedEpoch": recorded, "isOriginallyIssued": True,
              "predictionRole": "genuinely_issued_live_first20_logistic_prediction", "inputLineage": lineage,
              "issuedCohort": {"modelVersion": MODEL_VERSION, "adapterVersion": ADAPTER_VERSION,
                               "modelIdentitySha256": identity_hash, "target": result.get("target"),
                               "changeThresholdUgM3": THRESHOLD, "trainingCadence": TRAINING_CADENCE,
                               "recallWeighting": dict(RECALL_WEIGHTING), "calibrated": False,
                               "issueHourMYT": datetime.fromtimestamp(issue, MYT).hour, "issueOffsetSeconds": issue % 900}}
    snapshot_id = lineage.get("snapshotId", lineage.get("inputSnapshotId"))
    if snapshot_id is not None:
        record["inputSnapshotId"] = snapshot_id
    with _RECORD_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        if resolved not in _ISSUE_PATHS_READ:
            if path.exists():
                # A duplicate is a fresh (<=120 s) issue, so only the ledger tail can hold one.
                with path.open("rb") as stream:
                    stream.seek(max(0, path.stat().st_size - 2_000_000))
                    for line in stream.read().splitlines():
                        try:
                            old = json.loads(line)
                            _RECORDED[(resolved, int(old["forecastIssuedEpoch"]), old["issuedCohort"]["modelIdentitySha256"],
                                       old.get("featureValuesSha256"), old.get("reason"))] = None
                        except (ValueError, KeyError, TypeError):
                            continue
            _ISSUE_PATHS_READ.add(resolved)
        if key in _RECORDED:
            return False
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
        _RECORDED[key] = None
        while len(_RECORDED) > 4096:
            _RECORDED.popitem(last=False)
    return True
