# SPDX-License-Identifier: Apache-2.0
"""Fit the device's gradient-boosting trainer (HgbTrainer.h) on synthetic data and
compare it with scikit-learn 1.5 HistGradientBoostingRegressor(loss="absolute_error").

The harness compiles the real header with HGB_REAL double, as Engine.cpp builds it on
the device, and fits two configurations that it takes from the firmware itself: the
point heads (the hgb::Params defaults) and the pooled session model (the kIterations,
kLearningRate, kMaxLeaves, kMinSamplesLeaf, kL2 and kMaxDepth constants of
SessionFeatures.h, set field by field as Engine.cpp does). scikit-learn is then fitted
with the same parameters on the same rows, and the test requires:

- the same baseline and, in every tree, the same splits (feature, threshold and the
  direction of missing values) reached in the same order from the root;
- every leaf equal to scikit-learn's leaf rounded to float, because hgb::Node stores
  leaf values as float;
- hgb::predict on held-out rows (with NaNs, new NaNs and values outside the training
  range) equal, bit for bit, to scikit-learn's own trees with float leaves, summed in
  tree order;
- the remaining gap to scikit-learn's predict() within the summed float rounding of the
  leaves on each row's path. That is the only tolerance: the rest must match exactly.

Run from the package: python WeatherDashboard/tests/hgb_trainer_test.py --cxx g++ --build-dir build/host/hgb_trainer
Use repeatable --cxx-arg=ARG for compiler drivers such as zig c++.
Needs numpy and scikit-learn 1.5 (the version the forecasting server's recipe uses).
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys


HARNESS = r"""
// Fit with the device's trainer, predict held-out rows, dump the forest node by node.
#define HGB_REAL double  // As Engine.cpp defines it for the device.
#include "HgbTrainer.h"
#include "SessionFeatures.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

static unsigned yields = 0;
static void countYield() { ++yields; }

int main(int argc, char **argv) {
  if (argc != 10) {
    std::fprintf(stderr, "usage: point|session x.f64 y.f64 rows features test.f64 test_rows predictions.f64 forest.f64\n");
    return 2;
  }
  const bool session = std::strcmp(argv[1], "session") == 0;
  if (!session && std::strcmp(argv[1], "point") != 0) return 2;
  const uint32_t rows = uint32_t(std::strtoul(argv[4], nullptr, 10));
  const uint16_t features = uint16_t(std::strtoul(argv[5], nullptr, 10));
  const size_t testRows = std::strtoul(argv[7], nullptr, 10);
  const std::vector<double> X = slurp(argv[2], size_t(rows) * features);
  const std::vector<double> y = slurp(argv[3], rows);
  const std::vector<double> test = slurp(argv[6], testRows * features);

  hgb::Params params;  // The point heads fit with the defaults.
  if (session) {
    params.maxIter = ses::kIterations;
    params.learningRate = ses::kLearningRate;
    params.maxLeaves = ses::kMaxLeaves;
    params.minSamplesLeaf = ses::kMinSamplesLeaf;
    params.l2 = ses::kL2;
    params.maxDepth = ses::kMaxDepth;
  }
  hgb::Trainer *trainer = new hgb::Trainer();
  hgb::Forest forest;
  const auto started = std::chrono::steady_clock::now();
  if (!trainer->fit(X.data(), y.data(), rows, features, params, forest, countYield)) {
    std::fprintf(stderr, "fit failed\n");
    return 3;
  }
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
  delete trainer;

  std::vector<double> predictions(testRows);
  for (size_t i = 0; i < testRows; ++i) predictions[i] = hgb::predict(forest, test.data() + i * features);
  spill(argv[8], predictions);
  // One record per node in each tree's own order: tree, feature, threshold, missingLeft, left, right, value.
  std::vector<double> nodes;
  for (uint16_t tree = 0; tree < forest.treeCount; ++tree) {
    const uint32_t begin = forest.rootOffsets[tree];
    const uint32_t end = tree + 1 < forest.treeCount ? forest.rootOffsets[tree + 1] : forest.nodeCount;
    for (uint32_t at = begin; at < end; ++at) {
      const hgb::Node &node = forest.nodes[at];
      nodes.insert(nodes.end(), {double(tree), double(node.feature), double(node.threshold), double(node.missingLeft),
                                 double(node.left), double(node.right), double(node.value)});
    }
  }
  spill(argv[9], nodes);
  std::printf("{\"max_iter\":%u,\"learning_rate\":%.17g,\"max_leaf_nodes\":%u,\"min_samples_leaf\":%u,"
              "\"l2_regularization\":%.17g,\"max_depth\":%u,\"trees\":%u,\"nodes\":%u,\"yields\":%u,"
              "\"baseline\":%.17g,\"fit_ms\":%.1f}\n",
              unsigned(params.maxIter), params.learningRate, unsigned(params.maxLeaves), unsigned(params.minSamplesLeaf),
              params.l2, unsigned(params.maxDepth), unsigned(forest.treeCount), unsigned(forest.nodeCount), yields,
              forest.baseline, ms);
  forest.release();
  return 0;
}
"""

SEED = 20261110
# (configuration, training rows, held-out rows, columns): about the size of the device's
# nightly fits (point heads: about 2,650 origins x 81 inputs; session: about 4,500 rows of
# 28 imputed inputs plus their missing indicators).
CONFIGURATIONS = (("point", 2650, 600, 81), ("session", 4500, 800, 34))


def synthetic(np, rng, rows, test_rows, features):
    """Sensor-like columns of several kinds, with gaps, a dead input and a constant input."""
    n = rows + test_rows
    latent = rng.standard_normal((n, 4))
    X = latent @ rng.standard_normal((4, features)) + 0.5 * rng.standard_normal((n, features))
    for j in range(1, features):
        kind = j % 5
        if kind == 1:
            X[:, j] = np.round(X[:, j] * 4) / 4  # Coarse steps: fewer than 255 values, midpoint thresholds.
        elif kind == 2:
            X[:, j] = np.clip(np.round(2 * X[:, j] + 6), 0, 12)  # Small counts.
        elif kind == 3:
            X[:, j] = np.sin(2 * np.pi * rng.integers(0, 96, n) / 96)  # A 15-minute clock.
        elif kind == 4:
            X[:, j] = np.exp(X[:, j])  # Heavy tail; kind 0 stays continuous (percentile thresholds).
    X[:, 1::9] = X[:, 1::9] > 0  # 0/1 flags, like missing indicators.
    X[:, 0] = np.round(np.exp(3.5 + 0.6 * latent[:, 0]), 1)  # A PM2.5 level to 0.1 ug/m3.
    for k, j in enumerate(range(5, features - 2, 7)):  # Scattered gaps of 2 to 30 percent.
        X[rng.random(n) < (0.02, 0.1, 0.3)[k % 3], j] = np.nan
    for start in rng.integers(0, n - 60, 6):  # A neighbour offline for a stretch.
        X[start:start + 60, 6:9] = np.nan
    X[:, features - 1] = np.nan  # Never observed.
    X[:, features - 2] = 3.0  # Constant.
    y = (6 * np.tanh(latent[:, 1]) + 2 * latent[:, 2] * (X[:, 2] > 6) + np.where(np.isnan(X[:, 5]), 4.0, 0.0)
         + rng.laplace(0, 2.5, n))
    y = np.round(y, 1)  # Readings carry 0.1 ug/m3, so ties exercise the median and sign rules.
    train, test = X[:rows], X[rows:].copy()
    for j in (2, 3, 4):  # New NaNs in inputs that were complete at fit.
        test[rng.random(test_rows) < 0.1, j] = np.nan
    test[::17, 0] *= 3  # Beyond the training range.
    return train, y[:rows], test


def float_leaf_forest(np, model, X):
    """scikit-learn's own trees with every leaf rounded to float, summed in tree order as
    hgb::predict sums them; also the summed rounding of the leaves on each row's path."""
    total = np.full(len(X), float(np.ravel(model._baseline_prediction)[0]))
    bound = np.zeros(len(X))
    rows = np.arange(len(X))
    for (predictor,) in model._predictors:
        nodes = predictor.nodes
        at = np.zeros(len(X), dtype=np.intp)
        while True:
            split = nodes["is_leaf"][at] == 0
            if not split.any():
                break
            node = nodes[at[split]]
            value = X[rows[split], node["feature_idx"]]
            left = np.where(np.isnan(value), node["missing_go_to_left"] == 1, value <= node["num_threshold"])
            at[split] = np.where(left, node["left"], node["right"])
        leaf = nodes["value"][at]
        rounded = leaf.astype(np.float32).astype(np.float64)
        total += rounded
        bound += np.abs(rounded - leaf)
    return total, bound


def compare_trees(np, model, info, records):
    """Walk both forests from every root; return the first differences found."""
    problems = []
    if len(model._predictors) != info["trees"]:
        problems.append(f"{info['trees']} trees, scikit-learn {len(model._predictors)}")
        return problems
    tree_of = records[:, 0].astype(int)
    for t, (predictor,) in enumerate(model._predictors):
        sk, cpp = predictor.nodes, records[tree_of == t]
        if len(sk) != len(cpp):
            problems.append(f"tree {t}: {len(cpp)} nodes, scikit-learn {len(sk)}")
            continue
        stack = [(0, 0)]
        while stack:
            a, b = stack.pop()
            s, c = sk[a], cpp[b]
            leaf = c[1] < 0
            if bool(s["is_leaf"]) != leaf:
                problems.append(f"tree {t} node {b}: leaf {leaf}, scikit-learn {bool(s['is_leaf'])}")
                break
            if leaf:
                if c[6] != float(np.float32(s["value"])):
                    problems.append(f"tree {t} node {b}: leaf {c[6]!r}, scikit-learn {s['value']!r}")
                continue
            if (int(c[1]) != int(s["feature_idx"]) or c[2] != float(s["num_threshold"])
                    or bool(c[3]) != bool(s["missing_go_to_left"])):
                problems.append(f"tree {t} node {b}: split (feature {int(c[1])}, threshold {c[2]!r}, missing left "
                                f"{bool(c[3])}), scikit-learn ({int(s['feature_idx'])}, {float(s['num_threshold'])!r}, "
                                f"{bool(s['missing_go_to_left'])})")
                break
            stack.append((int(s["left"]), int(c[4])))
            stack.append((int(s["right"]), int(c[5])))
        if len(problems) >= 5:
            break
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", required=True)
    parser.add_argument("--cxx-arg", action="append", default=[])
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import sklearn
    from sklearn.ensemble import HistGradientBoostingRegressor
    if not sklearn.__version__.startswith("1.5."):
        parser.error(f"needs scikit-learn 1.5 (the server recipe's version), found {sklearn.__version__}")
    firmware = Path(__file__).resolve().parents[1]
    build = args.build_dir.resolve()
    build.mkdir(parents=True, exist_ok=True)
    source = build / "hgb_trainer_test.cpp"
    source.write_text(HARNESS, encoding="utf-8")
    suffix = ".exe" if sys.platform == "win32" else ""
    executable = build / ("hgb_trainer_test" + suffix)
    subprocess.run([args.cxx, *args.cxx_arg, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                    "-I", str(firmware), str(source), "-o", str(executable)], check=True)

    rng = np.random.default_rng(SEED)
    failures, summaries = [], []
    for config, rows, test_rows, features in CONFIGURATIONS:
        X, y, test = synthetic(np, rng, rows, test_rows, features)
        paths = {name: build / f"{config}_{name}.f64" for name in ("x", "y", "test", "predictions", "forest")}
        X.astype("<f8").tofile(paths["x"])
        y.astype("<f8").tofile(paths["y"])
        test.astype("<f8").tofile(paths["test"])
        run = subprocess.run([str(executable), config, str(paths["x"]), str(paths["y"]), str(rows), str(features),
                              str(paths["test"]), str(test_rows), str(paths["predictions"]), str(paths["forest"])],
                             check=True, capture_output=True, text=True)
        info = json.loads(run.stdout)
        model = HistGradientBoostingRegressor(
            loss="absolute_error", early_stopping=False, max_bins=255, max_iter=info["max_iter"],
            learning_rate=info["learning_rate"], max_leaf_nodes=info["max_leaf_nodes"],
            min_samples_leaf=info["min_samples_leaf"], l2_regularization=info["l2_regularization"],
            max_depth=info["max_depth"] or None).fit(X, y)
        cpp = np.fromfile(paths["predictions"], dtype="<f8")
        records = np.fromfile(paths["forest"], dtype="<f8").reshape(-1, 7)
        problems = compare_trees(np, model, info, records)
        if info["baseline"] != float(np.ravel(model._baseline_prediction)[0]):
            problems.append(f"baseline {info['baseline']!r}, scikit-learn {np.ravel(model._baseline_prediction)[0]!r}")
        if info["yields"] != info["trees"] or info["trees"] != info["max_iter"]:
            problems.append(f"{info['trees']} trees, {info['yields']} yields, max_iter {info['max_iter']}")
        expected, bound = float_leaf_forest(np, model, test)
        exact_gap = float(np.max(np.abs(cpp - expected)))
        if exact_gap != 0:
            problems.append(f"hgb::predict differs from scikit-learn's float-leaf trees by up to {exact_gap:.3e}")
        sk = model.predict(test)
        gap = np.abs(cpp - sk)
        # The only tolerance: the float rounding of the leaves on the row's path, plus double
        # rounding of the two 100-odd term sums. Measured on these data (x86-64, zig c++ -O2):
        # worst gaps 8.1e-8 (point) and 4.8e-8 (session), bounds 2.2e-7 and 1.7e-7; the trees
        # and the float-leaf predictions match exactly.
        allowed = bound + 1e-12 * (1 + np.abs(sk))
        if not np.all(gap <= allowed):
            problems.append(f"gap to predict() {gap.max():.3e} exceeds the float-leaf bound on {(gap > allowed).sum()} rows")
        leaves = records[records[:, 1] < 0]
        depth = f"depth {info['max_depth']}" if info["max_depth"] else "no depth limit"
        summaries.append(
            f"{config} ({info['max_iter']} iterations, learning rate {info['learning_rate']:g}, "
            f"{info['max_leaf_nodes']} leaves, {info['min_samples_leaf']} per leaf, L2 {info['l2_regularization']:g}, "
            f"{depth}; {rows}x{features} fit, {test_rows} held out): {info['trees']} trees, {len(records)} nodes "
            f"({len(leaves)} leaves) identical, held-out predictions equal to the float-leaf trees (gap {exact_gap:.0e}), "
            f"worst gap to predict() {gap.max():.2e} (float-leaf bound {bound.max():.2e}), fit {info['fit_ms']:.0f} ms")
        failures += [f"{config}: {p}" for p in problems]
    if failures:
        print("FAIL: HgbTrainer.h differs from scikit-learn " + sklearn.__version__)
        for failure in failures:
            print("  " + failure)
        sys.exit(1)
    print(f"PASS: HgbTrainer.h (HGB_REAL double) grows the same forests as scikit-learn {sklearn.__version__} "
          f"HistGradientBoostingRegressor(absolute_error) on synthetic data; " + "; ".join(summaries))


if __name__ == "__main__":
    main()
