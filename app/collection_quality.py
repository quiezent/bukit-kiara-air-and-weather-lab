"""Coverage checks use only observations within each completed sensor bucket."""
import numpy as np
import pandas as pd


def bucket_coverage(raw, column="pm02", bucket_seconds=900, edge_seconds=240):
    values = pd.to_numeric(raw[column], errors="coerce")
    valid = values.notna() & np.isfinite(values)
    epochs = pd.Series(raw.index.asi8 / 1e9, index=raw.index).where(valid)
    groups = epochs.resample(f"{bucket_seconds}s", label="right", closed="right")
    summary = groups.agg(["count", "min", "max"])
    ends = summary.index.asi8 / 1e9
    # Edge coverage is knowable at bucket closure; no later gap endpoint is used.
    eligible = ((summary["count"] >= 3)
                & (summary["min"] - (ends - bucket_seconds) <= edge_seconds)
                & (ends - summary["max"] <= edge_seconds))
    # Avoid claiming coverage across an interrupted cadence inside one bucket.
    valid_epochs = epochs.dropna()
    gaps = valid_epochs.groupby(pd.Grouper(freq=f"{bucket_seconds}s", label="right", closed="right")).diff()
    largest = gaps.resample(f"{bucket_seconds}s", label="right", closed="right").max()
    eligible &= largest.reindex(summary.index).fillna(0).le(edge_seconds * 2)
    return pd.DataFrame({"readingCount": summary["count"], "forecastEligible": eligible})


def collection_summary(rows, as_of_epoch, gap_seconds=600):
    epochs = sorted({int(row["epoch"]) for row in rows if row["epoch"] <= as_of_epoch})
    gaps = [{"startEpoch": left, "endEpoch": right, "durationMinutes": round((right-left)/60, 2),
             "cause": "not_determined", "imputed": False}
            for left, right in zip(epochs, epochs[1:]) if right-left > gap_seconds]
    # Gaps are measured; the public snapshot does not infer their cause.
    return {"gapThresholdSeconds": gap_seconds, "gapCount": len(gaps),
            "recentGaps": gaps[-10:], "latestObservationEpoch": epochs[-1] if epochs else None,
            "latestObservationAgeSeconds": as_of_epoch-epochs[-1] if epochs else None,
            "missingDataFilled": False,
            "forecastBucketRule": "At least 3 samples, both edges within 4 min, internal gaps <=8 min"}
