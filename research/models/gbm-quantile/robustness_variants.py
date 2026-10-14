"""Hyperparameter-sensitivity refits of the GBM-quantile `final` candidate (robustness review).

Each variant changes one setting of `final`, refits daily on the test split
(2026-09-27..10-13) with the arrival and ride-mean heads only (compute), and
saves data/models/gbm-quantile/robustness/<variant>__test.parquet. Scoring is
in robustness.py (action `variants`).

    python -B gbm-quantile/robustness_variants.py <variant> [<variant> ...]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from harness import config as hconfig, walkforward  # noqa: E402
import evaluate as E  # noqa: E402
import gbmq  # noqa: E402

OUT = hconfig.OUT / "gbm-quantile" / "robustness"
FINAL = E.CONFIGS["final"].with_(heads=("arrival", "ride_mean"))
VARIANTS = {
    "v_final2": FINAL.with_(name="v_final2"),                                   # control: must equal final
    "v_d3": FINAL.with_(name="v_d3", max_depth=3),
    "v_d6": FINAL.with_(name="v_d6", max_depth=6, min_child_weight=20.0),
    "v_mcw10": FINAL.with_(name="v_mcw10", min_child_weight=10.0),
    "v_r50": FINAL.with_(name="v_r50", rounds=50),
    "v_eta05": FINAL.with_(name="v_eta05", eta=0.05, rounds=200),
    "v_lb42": FINAL.with_(name="v_lb42", lookback_days=42),
    "v_lb21": FINAL.with_(name="v_lb21", lookback_days=21),
    "v_sub": FINAL.with_(name="v_sub", subsample=0.8, colsample=0.8),
}


def run(name):
    cfg = VARIANTS[name]
    ds = E.load_ds()
    cols = gbmq.columns(ds, cfg)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    pred = walkforward.walk_forward(ds, gbmq.make_fit_predict(cfg, cols), targets=("arrival",),
                                    lookback_days=cfg.lookback_days,
                                    embargo_minutes=gbmq.ARRIVAL_EMBARGO_SECONDS // 60, days=E.split_days("test"),
                                    log=lambda s: print(f"[{name}] {s}", flush=True))
    pred.to_parquet(OUT / f"{name}__test.parquet")
    meta = {"config": {k: v for k, v in cfg.__dict__.items()}, "features": len(cols),
            "seconds": round(time.time() - started, 1)}
    (OUT / f"{name}__test.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    print(f"saved {name} rows={len(pred)} in {meta['seconds']} s", flush=True)


if __name__ == "__main__":
    for variant in sys.argv[1:]:
        run(variant)
