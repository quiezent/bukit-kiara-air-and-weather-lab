# SPDX-License-Identifier: Apache-2.0
"""Fit the device's arrival-change classifier (ArrivalLogistic.h) on synthetic data and
compare it with the scikit-learn 1.5 pipeline of the forecasting server's
arrival_logistic_model.py v2:

    make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                  StandardScaler(),
                  LogisticRegression(C=0.1, solver="lbfgs", max_iter=500, tol=1e-4))   # multinomial

fitted with the recall sample weights of ArrivalLogistic.h (both fall classes 2, the
others 1, divided by their mean). The harness compiles the real header with its device
defaults (ARRIVAL_KERNEL double) and calls arrival::fit and arrival::predict as
Engine.cpp does. Two fits: about the size of the device's nightly fit (8,000 rows, all
five classes, NaN gaps, a never-observed input and a constant input), and a smaller
window in which the rise-of-40 class never occurs. The test requires the same classes,
missing-indicator columns and medians, the same L-BFGS iteration and evaluation counts,
and held-out probabilities (rows with NaNs, new NaNs and values outside the training
range) within the tolerances below.

Run from the package: python WeatherDashboard/tests/arrival_logistic_test.py --cxx g++ --build-dir build/host/arrival_logistic
Use repeatable --cxx-arg=ARG for compiler drivers such as zig c++.
Needs numpy, SciPy and scikit-learn 1.5 (the version the forecasting server's recipe uses).
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import warnings


HARNESS = r"""
// Fit with the device's classifier, predict held-out rows, dump the fitted model.
#include "ArrivalLogistic.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

static std::vector<double> slurp(const char *path, size_t count) {
  std::vector<double> out(count);
  FILE *file = std::fopen(path, "rb");
  if (!file || std::fread(out.data(), sizeof(double), count, file) != count) {
    std::fprintf(stderr, "cannot read %zu values from %s\n", count, path);
    std::exit(2);
  }
  std::fclose(file);
  return out;
}

static void spill(const char *path, const std::vector<double> &values) {
  FILE *file = std::fopen(path, "wb");
  if (!file || std::fwrite(values.data(), sizeof(double), values.size(), file) != values.size()) {
    std::fprintf(stderr, "cannot write %s\n", path);
    std::exit(2);
  }
  std::fclose(file);
}

static void list(const char *name, const uint8_t *values, int count) {
  std::printf("\"%s\":[", name);
  for (int i = 0; i < count; ++i) std::printf(i ? ",%u" : "%u", unsigned(values[i]));
  std::printf("],");
}

int main(int argc, char **argv) {
  if (argc != 8) {
    std::fprintf(stderr, "usage: x.f64 labels.f64 rows test.f64 test_rows probabilities.f64 model.f64\n");
    return 2;
  }
  const uint32_t rows = uint32_t(std::strtoul(argv[3], nullptr, 10));
  const size_t testRows = std::strtoul(argv[5], nullptr, 10);
  const std::vector<double> X = slurp(argv[1], size_t(rows) * arrival::kInputs);
  const std::vector<double> classes = slurp(argv[2], rows);
  const std::vector<double> test = slurp(argv[4], testRows * arrival::kInputs);
  std::vector<uint8_t> labels(rows);
  for (uint32_t i = 0; i < rows; ++i) labels[i] = uint8_t(classes[i]);

  arrival::Model model;
  arrival::FitInfo info;
  const auto started = std::chrono::steady_clock::now();
  if (!arrival::fit(X.data(), labels.data(), rows, model, &info)) {
    std::fprintf(stderr, "fit failed\n");
    return 3;
  }
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();

  std::vector<double> probabilities(testRows * arrival::kClasses);
  for (size_t i = 0; i < testRows; ++i)
    if (!arrival::predict(model, test.data() + i * arrival::kInputs, probabilities.data() + i * arrival::kClasses)) {
      std::fprintf(stderr, "predict refused held-out row %zu\n", i);
      return 4;
    }
  spill(argv[6], probabilities);
  // median[28], mean[d], scale[d], then coef[k][0..d] for each class (intercept last).
  std::vector<double> fitted(model.median, model.median + arrival::kInputs);
  fitted.insert(fitted.end(), model.mean, model.mean + model.featureCount);
  fitted.insert(fitted.end(), model.scale, model.scale + model.featureCount);
  for (int k = 0; k < model.classCount; ++k)
    fitted.insert(fitted.end(), model.coef[k], model.coef[k] + model.featureCount + 1);
  spill(argv[7], fitted);
  std::printf("{");
  list("classes", model.classId, model.classCount);
  list("indicators", model.indicatorColumn, model.indicatorCount);
  std::printf("\"features\":%u,\"iterations\":%u,\"evaluations\":%u,\"skipped\":%u,\"restarts\":%u,"
              "\"converged\":%s,\"loss\":%.17g,\"gradient_norm\":%.17g,\"fit_ms\":%.1f}\n",
              unsigned(model.featureCount), unsigned(info.iterations), unsigned(info.evaluations),
              unsigned(info.skippedUpdates), unsigned(info.restarts), info.converged ? "true" : "false",
              info.loss, info.gradientNorm, ms);
  return 0;
}
"""

SEED = 20261110
INPUTS, CLASSES = 28, 5
# The device's solver takes scikit-learn's L-BFGS iterates, so the two fits differ only
# by rounding. Measured on these data (x86-64, zig c++ -O2): held-out probabilities 2.6e-9
# (full) and 2.8e-10 (no rise40), coefficients 3.3e-9 and 1.4e-10; ArrivalLogistic.h
# reports below 3e-8 on real 28-day windows. The tolerances leave room for another
# compiler or libm but not for another solver path: one fewer L-BFGS correction moved the
# probabilities by 4.8e-3 and the iteration count by one.
PROBABILITY_TOLERANCE = 1e-6
COEFFICIENT_TOLERANCE = 1e-6


def label_class(np, delta):
    """arrival::labelClass: inclusive 20/40 boundaries of the +90 change."""
    return np.select([delta <= -40, delta <= -20, delta < 20, delta < 40], [0, 1, 2, 3], 4)


def synthetic(np, rng, rows):
    """28 sensor-like inputs and a +90 change that partly follows them."""
    latent = rng.standard_normal((rows, 5))
    X = latent @ rng.standard_normal((5, INPUTS)) + 0.7 * rng.standard_normal((rows, INPUTS))
    X[:, 0] = np.round(np.exp(3.6 + 0.5 * latent[:, 0]), 1)  # The fresh PM2.5 level to 0.1 ug/m3.
    X[:, 1:8] = np.round(3 * X[:, 1:8], 1)  # Recent changes.
    X[:, 20] = np.round(28 + 2 * X[:, 20], 1)  # Temperature-like.
    X[:, 21] = np.round(np.clip(75 + 8 * X[:, 21], 0, 100), 1)  # Humidity-like.
    X[:, 25] = np.clip(np.round(X[:, 25] + 3), 0, 6)  # A small count.
    X[rng.random(rows) < 0.03, 9:14] = np.nan  # Gaps in the recent window blank several inputs.
    X[rng.random(rows) < 0.02, 20:22] = np.nan
    X[rng.random(rows) < 0.3, 15] = np.nan
    X[:, 26] = 0.0  # Constant.
    X[:, 27] = np.nan  # Never observed.
    delta = (13 * latent[:, 1] - 0.15 * (X[:, 0] - 40) + 5 * latent[:, 2] * latent[:, 3]
             + np.where(np.isnan(X[:, 15]), 3.0, 0.0) + rng.laplace(0, 6, rows))
    return X, label_class(np, np.round(delta, 1))


def recall_weights(np, y):
    weights = np.where(y <= 1, 2.0, 1.0)
    return weights / weights.mean()


def reference(np, X, y):
    """The server's pipeline, with its solver evaluations counted."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model._linear_loss import LinearModelLoss
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                          StandardScaler(),
                          LogisticRegression(C=0.1, solver="lbfgs", max_iter=500, tol=1e-4, multi_class="multinomial"))
    original = LinearModelLoss.loss_gradient
    calls = []

    def counted(self, *args, **kwargs):
        calls.append(None)
        return original(self, *args, **kwargs)

    LinearModelLoss.loss_gradient = counted
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)  # multi_class is deprecated in 1.5; the server sets it.
            model.fit(X, y, logisticregression__sample_weight=recall_weights(np, y))
    finally:
        LinearModelLoss.loss_gradient = original
    return model, len(calls)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", required=True)
    parser.add_argument("--cxx-arg", action="append", default=[])
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import scipy
    import sklearn
    if not sklearn.__version__.startswith("1.5."):
        parser.error(f"needs scikit-learn 1.5 (the server recipe's version), found {sklearn.__version__}")
    firmware = Path(__file__).resolve().parents[1]
    build = args.build_dir.resolve()
    build.mkdir(parents=True, exist_ok=True)
    source = build / "arrival_logistic_test.cpp"
    source.write_text(HARNESS, encoding="utf-8")
    suffix = ".exe" if sys.platform == "win32" else ""
    executable = build / ("arrival_logistic_test" + suffix)
    subprocess.run([args.cxx, *args.cxx_arg, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                    "-I", str(firmware), str(source), "-o", str(executable)], check=True)

    rng = np.random.default_rng(SEED)
    X, y = synthetic(np, rng, 8000)
    test, _ = synthetic(np, rng, 1500)
    for j in (2, 3):  # New NaNs in inputs that were complete at fit.
        test[rng.random(len(test)) < 0.1, j] = np.nan
    test[::13, 0] *= 3  # Beyond the training range.
    small = np.arange(2000)[y[:2000] != 4]
    cases = (("full", X, y), ("no rise40", X[small], y[small]))

    failures, summaries = [], []
    for name, Xf, yf in cases:
        tag = name.replace(" ", "_")
        paths = {part: build / f"{tag}_{part}.f64" for part in ("x", "labels", "test", "probabilities", "model")}
        Xf.astype("<f8").tofile(paths["x"])
        yf.astype("<f8").tofile(paths["labels"])
        test.astype("<f8").tofile(paths["test"])
        run = subprocess.run([str(executable), str(paths["x"]), str(paths["labels"]), str(len(Xf)), str(paths["test"]),
                              str(len(test)), str(paths["probabilities"]), str(paths["model"])],
                             check=True, capture_output=True, text=True)
        info = json.loads(run.stdout)
        model, evaluations = reference(np, Xf, yf)
        imputer, scaler, logistic = (step for _, step in model.steps)
        problems = []
        classes = logistic.classes_.astype(int).tolist()
        indicators = imputer.indicator_.features_.astype(int).tolist()
        if info["classes"] != classes:
            problems.append(f"classes {info['classes']}, scikit-learn {classes}")
        if info["indicators"] != indicators:
            problems.append(f"missing indicators {info['indicators']}, scikit-learn {indicators}")
        d, K = info["features"], len(info["classes"])
        fitted = np.fromfile(paths["model"], dtype="<f8")
        if problems or d != scaler.mean_.size or fitted.size != INPUTS + 2 * d + K * (d + 1):
            failures += [f"{name}: {p}" for p in problems] or [f"{name}: model shape {d} features, {fitted.size} values"]
            continue
        median, mean, scale = fitted[:INPUTS], fitted[INPUTS:INPUTS + d], fitted[INPUTS + d:INPUTS + 2 * d]
        coef = fitted[INPUTS + 2 * d:].reshape(K, d + 1)
        iterations = int(logistic.n_iter_[0])
        if (info["iterations"], info["evaluations"]) != (iterations, evaluations):
            problems.append(f"{info['iterations']} iterations and {info['evaluations']} evaluations, "
                            f"scikit-learn {iterations} and {evaluations}")
        if not np.array_equal(median, imputer.statistics_):
            problems.append(f"medians differ by up to {np.abs(median - imputer.statistics_).max():.3e}")
        scaler_gap = max((np.abs(mean - scaler.mean_) / (1 + np.abs(scaler.mean_))).max(),
                         (np.abs(scale - scaler.scale_) / scaler.scale_).max())
        if scaler_gap > 1e-12:
            problems.append(f"scaler relative gap {scaler_gap:.3e}")
        coef_gap = max(np.abs(coef[:, :d] - logistic.coef_).max(), np.abs(coef[:, d] - logistic.intercept_).max())
        if coef_gap > COEFFICIENT_TOLERANCE:
            problems.append(f"coefficient gap {coef_gap:.3e} > {COEFFICIENT_TOLERANCE:g}")
        expected = np.zeros((len(test), CLASSES))
        expected[:, classes] = model.predict_proba(test)
        cpp = np.fromfile(paths["probabilities"], dtype="<f8").reshape(-1, CLASSES)
        gap = float(np.abs(cpp - expected).max())
        top = int((cpp.argmax(1) != expected.argmax(1)).sum())
        if gap > PROBABILITY_TOLERANCE or top:
            problems.append(f"held-out probability gap {gap:.3e} (tolerance {PROBABILITY_TOLERANCE:g}), top class differs on {top} rows")
        if not info["converged"]:
            problems.append("the device's solver did not converge")
        support = np.bincount(yf, minlength=CLASSES).tolist()
        summaries.append(
            f"{name} ({len(Xf)} rows, class counts {support}, {d} scaled inputs incl. {len(indicators)} indicators): "
            f"{info['iterations']} iterations / {info['evaluations']} evaluations = scikit-learn {iterations} / {evaluations}, "
            f"worst held-out probability gap {gap:.2e} over {len(test)} rows, coefficient gap {coef_gap:.2e}, "
            f"fit {info['fit_ms']:.0f} ms")
        failures += [f"{name}: {p}" for p in problems]
    if failures:
        print(f"FAIL: ArrivalLogistic.h differs from scikit-learn {sklearn.__version__} (SciPy {scipy.__version__})")
        for failure in failures:
            print("  " + failure)
        sys.exit(1)
    print(f"PASS: ArrivalLogistic.h (ARRIVAL_KERNEL double) fits the server's arrival pipeline as scikit-learn "
          f"{sklearn.__version__} / SciPy {scipy.__version__} does on synthetic data (probability tolerance "
          f"{PROBABILITY_TOLERANCE:g}); " + "; ".join(summaries))


if __name__ == "__main__":
    main()
