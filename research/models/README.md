# PM2.5 forecast research harness

Shared dataset, walk-forward evaluator and scorer for every candidate model of the Bukit Kiara / TTDI PM2.5 forecasts. It answers one question per candidate: on identical targets and cutoffs, does it beat persistence and the production incumbent?

> **Public copy.** The research data are private and not in this repository: the
> SQLite snapshot, the neighbour archive, the issued-forecast records and every
> parquet/JSON output under `data/models/`. The code, results and dataset schema are
> published so the method can be read, reused and criticised. To rerun it, collect your
> own history with the server in `app/`, then set `BUKIT_KIARA_SERVER_DIR` (default
> `app/`) and `BUKIT_KIARA_RESEARCH_DATA` (default `data/` at the repository root).
> Parquet files also need `pyarrow` (21.0.0 was used), which `requirements.txt` omits.
> Paths in these notes are relative to the repository root; `server\` means `app/` here.

- **Dataset schema:** `DATASET.md`
- **Current baseline numbers and big-event cases:** `BASELINES.md`
- **Final assembled models (2026-10-14) and the decision per target:** `RESULTS.md`; code in `final\`
- **Large outputs:** `data\models\` (git-ignored)

## Ground rules

- **Production is read only.** The harness imports production code from a frozen byte copy in `data\models\prod_src_20261013\`, never from `server\`. It reads the snapshot DB `data\snapshot_20261013.db` read only.
- **Interpreter and threads.** Run everything with the system Python and `-B` (no bytecode):
  ```
  $py = "python"   # Python 3.11 with requirements.txt
  cd research/models
  ```
  `harness\config.py` sets the BLAS thread limits to 2. Use `n_jobs<=2`.
- **Leak-free inputs.** Every feature was visible at its issue clock; `DATASET.md` lists the clock used for each source. Before the receipt ledger (2026-10-08 13:12), a sensor reading counts as visible at observation + 60 s. On the 1,508 post-ledger grid issues that rule never admits a reading before its real receipt; production's 0 s rule would in 14% of them. It hides an already received reading in 20%, so it errs on the strict side.
- **Chronological only.** For each evaluation day D, a candidate is refit on rows with issue in (D − lookback, D − embargo) whose label closed before D's MYT midnight. It then predicts day D. The evaluation period starts on 2026-09-10.
- **Literal outputs.** Score what the model outputs. A missing output is unavailable, never persistence.

## Rebuild (dataset about 4 min, replays about 25 min in three parallel runs, report about 2 min)

```
& $py -B -m harness.dataset                                  # data\models\dataset.parquet + dataset_meta.json
& $py -B -m harness.schema                                   # DATASET.md
& $py -B -m harness.incumbents.run ridge_v4 arrival_logistic first_crossing
& $py -B -m harness.incumbents.run cycling70
& $py -B -m harness.incumbents.run morning_hgb afternoon_patchtst
& $py -B -m harness.incumbents.run issued_clocks             # replays at genuinely issued clocks
& $py -B -m harness.report                                   # BASELINES.md + data\models\baselines.json
& $py -B -m harness.checks 30                                # feature code vs sources cut at the harness clocks
& $py -B -m harness.audit                                    # raw-SQL re-derivation of values, clocks, targets (about 2 min)
```

## Evaluating a candidate

```python
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from harness import dataset, walkforward, scoring, report

ds = dataset.load()                                          # 5-min issue grid, features + targets
cols = walkforward.feature_columns(ds, ("sensor", "time", "weather", "cams"))

def fit_predict(train, test, cutoff):                        # train: label-complete rows only
    x = train[cols].fillna(train[cols].median()).fillna(0)
    ok = train.y_arrival.notna() & train.s_freshPm25.notna()
    model = Ridge(alpha=100).fit(x[ok], (train.y_arrival - train.s_freshPm25)[ok])
    xt = test[cols].fillna(train[cols].median()).fillna(0)
    return pd.DataFrame({"pred_arrival": test.s_freshPm25 + model.predict(xt)}, index=test.index)

pred = walkforward.walk_forward(ds, fit_predict, targets=("arrival",), lookback_days=28, embargo_minutes=330)
evaluate = ds.loc[ds["eval"] & ds.m_fresh_available]
mine = scoring.score_candidate(evaluate, pred)                           # MAE, skill, >=20/>=40 event confusion
incumbent = scoring.score_candidate(evaluate, report.standard_replay(ds)[["pred_arrival"]])
persist = scoring.score_candidate(evaluate, scoring.persistence_predictions(evaluate)[["pred_arrival"]])
```

**Standard prediction columns.** The scorer reads whichever of these a candidate returns: `pred_arrival`, `pred_ride_mean`, `pred_ride_min`, `pred_ride_max`, `p_fall20`, `p_rise20`, `p_fall40`, `p_rise40`, `p_class5_0..4`, `p_first_drop/none/rise`, `p_ride_le70`, `pred_morning`, `pred_afternoon`.

**Scoring rules.**
- **Comparison rows.** Use the same rows for every model. Pass `mask=` to `score_candidate`, for example where the incumbent produced an output.
- **Event calls from probabilities.** `argmax` is the primary rule and is not tuned. `0.2` is a fixed sensitivity rule.
- **Embargo.** `walk_forward(targets=...)` picks the label-completion columns. Set `embargo_minutes` to the production embargo of the head you challenge: 330 for Ridge, 120 for the arrival logistic.
- **Priors.** Use `scoring.walk_forward_prior(evaluate, column, values, complete_col, history=ds)`. Always pass the full dataset as `history`, because an evaluation subset has no labels before its first day.
- **Big events.** `report.case_table(ds, preds)` gives the per-event view for any frame with the standard columns.

## Layout

| File | Role |
|---|---|
| `harness\config.py` | Paths, clocks, visibility lags, evaluation start |
| `harness\prod.py` | Frozen production mirror and read-only imports |
| `harness\sources.py` | Snapshot and neighbour loaders with visibility clocks |
| `harness\features.py` | Feature groups (sensor via production function, weather, CAMS, regional, Subang, neighbours) |
| `harness\targets.py` | Production target definitions |
| `harness\dataset.py` | Build and load the grid dataset |
| `harness\walkforward.py` | Daily-refit evaluator |
| `harness\scoring.py` | Metrics, event confusion, priors |
| `harness\incumbents\` | Walk-forward replays of the production models, each with a `verify` against production |
| `harness\issued.py` | Genuinely issued outputs: ledger, private JSONL, prospective table |
| `harness\report.py` | Baseline table and case table |
| `harness\checks.py` | Leak check of the feature code against sources truncated at the harness clocks |
| `harness\audit.py` | Independent raw-SQL audit: feature values and source clocks at 40 sampled issues, the legacy sensor rule, 50 targets |
| `final\` | Final assembled models: port-ready `models.py` (fit/predict) and `spatial_live.py` (per-issue neighbour features), plus `run.py` (walk-forward, selection, scoring) |
