#pragma once
// Follows scikit-learn 1.5 (BSD-3-Clause, Copyright (c) 2007-2024 The scikit-learn developers).
// Notices: THIRD_PARTY_NOTICES.md and LICENSES/ in the public firmware package.
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
//
// Histogram gradient boosting for the on-device forecast models: the recipe of
// scikit-learn 1.5 HistGradientBoostingRegressor(loss="absolute_error",
// early_stopping=False), which the Python server trains nightly.
//
// Portable C++ with no Arduino dependency, so the same file runs in the host
// parity test (WeatherDashboard/tests) and on the ESP32-S3. Absolute
// error makes every gradient +1 or -1 and every hessian 1, so histograms hold
// exact integers. NaN inputs take their own bin, as in scikit-learn.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

#ifndef HGB_ALLOC
#define HGB_ALLOC(bytes) malloc(bytes)
#endif
#ifndef HGB_FREE
#define HGB_FREE(pointer) free(pointer)
#endif

#ifndef HGB_REAL
#define HGB_REAL float
#endif

namespace hgb {

using Real = HGB_REAL;

constexpr uint16_t kMissingBin = 255;
constexpr uint16_t kMaxBins = 255;  // Non-missing bins per feature.

struct Params {
  uint16_t maxIter = 120;
  double learningRate = 0.1;
  uint8_t maxLeaves = 15;
  uint16_t minSamplesLeaf = 60;
  double l2 = 1.0;
  uint8_t maxDepth = 0;  // Deepest allowed leaf (edges from the root); 0 means no limit.
};

struct Node {
  Real threshold;    // x <= threshold goes left; +inf means "split on NaN".
  float value;       // Leaf output (learning rate already applied).
  int16_t feature;   // -1 for a leaf.
  uint16_t left, right;
  uint8_t missingLeft;
  uint8_t reserved;
};

struct Forest {
  double baseline = 0;
  Node *nodes = nullptr;
  uint32_t nodeCount = 0;
  uint16_t *roots = nullptr;  // Offset of each tree's root: tree nodes are contiguous.
  uint32_t *rootOffsets = nullptr;
  uint16_t treeCount = 0;

  void release() {
    HGB_FREE(nodes);
    HGB_FREE(rootOffsets);
    nodes = nullptr;
    rootOffsets = nullptr;
    nodeCount = treeCount = 0;
  }
};

inline double predict(const Forest &forest, const Real *x) {
  double sum = forest.baseline;
  for (uint16_t tree = 0; tree < forest.treeCount; ++tree) {
    const Node *nodes = forest.nodes + forest.rootOffsets[tree];
    uint16_t at = 0;
    while (nodes[at].feature >= 0) {
      const Node &node = nodes[at];
      const Real value = x[node.feature];
      const bool left = std::isnan(value) ? node.missingLeft != 0 : value <= node.threshold;
      at = left ? node.left : node.right;
    }
    sum += nodes[at].value;
  }
  return sum;
}

class Trainer {
 public:
  // X is row-major (rows * features floats). Returns false when memory runs out.
  // yield, when set, runs after each tree so a task can pause or feed a watchdog.
  bool fit(const Real *X, const Real *y, uint32_t rows, uint16_t features,
           const Params &params, Forest &out, void (*yield)() = nullptr) {
    release();
    forestRelease(out);
    rows_ = rows;
    features_ = features;
    params_ = params;
    const uint32_t maxNodes = 2u * params.maxLeaves - 1u;
    bins_ = alloc<uint8_t>(uint64_t(rows) * features);
    thresholds_ = alloc<double>(uint64_t(features) * kMaxBins);
    nonMissing_ = alloc<uint16_t>(features);
    hasMissing_ = alloc<uint8_t>(features);
    hist_ = alloc<Bin>(uint64_t(features) * 256u);
    raw_ = alloc<double>(rows);
    gradient_ = alloc<int8_t>(rows);
    samples_ = alloc<uint16_t>(rows);
    scratch_ = alloc<uint16_t>(rows);
    residual_ = alloc<double>(rows);
    build_ = alloc<Build>(maxNodes);
    sorted_ = alloc<Real>(rows);
    out.nodes = alloc<Node>(uint64_t(maxNodes) * params.maxIter);
    out.rootOffsets = alloc<uint32_t>(params.maxIter);
    if (!bins_ || !thresholds_ || !nonMissing_ || !hasMissing_ || !hist_ || !raw_ || !gradient_
        || !samples_ || !scratch_ || !residual_ || !build_ || !sorted_ || !out.nodes
        || !out.rootOffsets || rows == 0 || rows > 65535) {
      release();
      forestRelease(out);
      return false;
    }
    bin(X);
    for (uint32_t row = 0; row < rows; ++row) residual_[row] = y[row];
    out.baseline = median(residual_, rows);
    for (uint32_t row = 0; row < rows; ++row) raw_[row] = out.baseline;
    for (uint16_t iteration = 0; iteration < params.maxIter; ++iteration) {
      for (uint32_t row = 0; row < rows; ++row) gradient_[row] = raw_[row] > y[row] ? 1 : -1;
      out.rootOffsets[iteration] = out.nodeCount;
      growTree(y, out);
      out.treeCount = iteration + 1;
      if (yield) yield();
    }
    release();
    return true;
  }

  ~Trainer() { release(); }

 private:
  struct Bin {
    int16_t gradient;
    uint16_t count;
  };
  struct Build {
    uint16_t begin, end;       // Range in samples_.
    int32_t gradient;
    double gain;
    int16_t feature;
    uint16_t bin;
    int32_t leftGradient;
    uint8_t missingLeft;
    int16_t left, right;       // Child build indices; -1 for a leaf.
    uint8_t depth;
  };

  template <typename T>
  static T *alloc(uint64_t count) {
    return count ? static_cast<T *>(HGB_ALLOC(count * sizeof(T))) : nullptr;
  }

  static void forestRelease(Forest &forest) { forest.release(); }

  void release() {
    HGB_FREE(bins_); HGB_FREE(thresholds_); HGB_FREE(nonMissing_); HGB_FREE(hasMissing_);
    HGB_FREE(hist_); HGB_FREE(raw_); HGB_FREE(gradient_); HGB_FREE(samples_);
    HGB_FREE(scratch_); HGB_FREE(residual_); HGB_FREE(build_); HGB_FREE(sorted_);
    bins_ = nullptr; thresholds_ = nullptr; nonMissing_ = nullptr; hasMissing_ = nullptr;
    hist_ = nullptr; raw_ = nullptr; gradient_ = nullptr; samples_ = nullptr;
    scratch_ = nullptr; residual_ = nullptr; build_ = nullptr; sorted_ = nullptr;
  }

  // numpy.median: the mean of the two middle values for an even count.
  static double median(double *values, uint32_t count) {
    const uint32_t middle = count / 2;
    std::nth_element(values, values + middle, values + count);
    const double upper = values[middle];
    if (count & 1) return upper;
    return 0.5 * (*std::max_element(values, values + middle) + upper);
  }

  // scikit-learn _find_binning_thresholds, max_bins = 255.
  void bin(const Real *X) {
    for (uint16_t feature = 0; feature < features_; ++feature) {
      uint32_t present = 0;
      for (uint32_t row = 0; row < rows_; ++row) {
        const Real value = X[uint64_t(row) * features_ + feature];
        if (!std::isnan(value)) sorted_[present++] = value;
      }
      std::sort(sorted_, sorted_ + present);
      uint32_t distinct = 0;
      for (uint32_t i = 0; i < present; ++i)
        if (i == 0 || sorted_[i] != sorted_[i - 1]) ++distinct;
      double *thresholds = thresholds_ + uint32_t(feature) * kMaxBins;
      uint16_t count = 0;
      if (distinct <= kMaxBins) {
        double previous = 0;
        bool first = true;
        for (uint32_t i = 0; i < present; ++i) {
          if (i && sorted_[i] == sorted_[i - 1]) continue;
          if (!first) thresholds[count++] = 0.5 * (previous + double(sorted_[i]));
          previous = sorted_[i];
          first = false;
        }
      } else {
        // numpy.percentile(method="midpoint") on linspace(0, 100, 256)[1:-1], in numpy's
        // arithmetic: near-equal values make the rounding of a midpoint choose the bin.
        const double step = 100.0 / kMaxBins;
        for (uint16_t k = 1; k < kMaxBins; ++k) {
          const double position = double(present - 1) * (double(k) * step / 100.0);
          const uint32_t low = uint32_t(std::floor(position));
          const uint32_t high = uint32_t(std::ceil(position));
          const double a = sorted_[low], b = sorted_[high];
          thresholds[count++] = low == high ? a : b - (b - a) * 0.5;
        }
      }
      nonMissing_[feature] = count + 1;
      hasMissing_[feature] = present != rows_;
      for (uint32_t row = 0; row < rows_; ++row) {
        const Real value = X[uint64_t(row) * features_ + feature];
        uint16_t index = kMissingBin;
        if (!std::isnan(value))
          index = uint16_t(std::lower_bound(thresholds, thresholds + count, double(value)) - thresholds);
        bins_[uint64_t(row) * features_ + feature] = uint8_t(index);
      }
    }
  }

  // Same arithmetic, in the same order, as scikit-learn's _split_gain: equal gains
  // are common with integer gradients, so rounding decides which split wins.
  double loss(int32_t gradient, int32_t hessian) const {
    const double value = -double(gradient) / (double(hessian) + params_.l2 + 1e-15);
    return double(gradient) * value;
  }

  // Screening term in single precision (the S3 has a hardware float unit only).
  float approx(int32_t gradient, int32_t hessian) const {
    const float g = float(gradient);
    return g * g / (float(hessian) + float(params_.l2));
  }

  double splitGain(double current, int32_t leftGradient, int32_t leftCount, int32_t rightGradient, int32_t rightCount) const {
    double gain = current;
    gain -= loss(leftGradient, leftCount);
    gain -= loss(rightGradient, rightCount);
    return gain;
  }

  void findSplit(Build &node) {
    const uint32_t count = uint32_t(node.end) - node.begin;
    std::memset(hist_, 0, uint32_t(features_) * 256u * sizeof(Bin));
    for (uint32_t i = node.begin; i < node.end; ++i) {
      const uint16_t row = samples_[i];
      const uint8_t *cells = bins_ + uint64_t(row) * features_;
      const int8_t gradient = gradient_[row];
      for (uint16_t feature = 0; feature < features_; ++feature) {
        Bin &slot = hist_[uint32_t(feature) * 256u + cells[feature]];
        slot.gradient += gradient;
        ++slot.count;
      }
    }
    const double parent = loss(node.gradient, int32_t(count));
    const float parentApprox = approx(node.gradient, int32_t(count));
    // Far wider than the single-precision error of a gain: only candidates that
    // cannot win skip the exact double-precision gain, so results do not change.
    const float margin = 4e-6f * float(count) + 1e-3f;
    const int32_t minimum = params_.minSamplesLeaf;
    double best = -1;
    node.feature = -1;
    for (uint16_t feature = 0; feature < features_; ++feature) {
      const Bin *hist = hist_ + uint32_t(feature) * 256u;
      double featureBest = -1;
      int32_t binBest = -1, gradientBest = 0;
      bool missingLeft = false;
      const int32_t end = int32_t(nonMissing_[feature]) - 1 + (hasMissing_[feature] ? 1 : 0);
      int32_t leftCount = 0, leftGradient = 0;
      for (int32_t index = 0; index < end; ++index) {
        if (!hist[index].count) continue;  // Same candidate as the previous bin.
        leftCount += hist[index].count;
        leftGradient += hist[index].gradient;
        const int32_t rightCount = int32_t(count) - leftCount;
        if (leftCount < minimum) continue;
        if (rightCount < minimum) break;
        if (approx(leftGradient, leftCount) + approx(node.gradient - leftGradient, rightCount) - parentApprox + margin
            <= std::max(float(featureBest), 0.0f)) continue;
        const double gain = splitGain(parent, leftGradient, leftCount, node.gradient - leftGradient, rightCount);
        if (gain > featureBest && gain > 0) {
          featureBest = gain;
          binBest = index;
          gradientBest = leftGradient;
          missingLeft = false;
        }
      }
      if (hasMissing_[feature]) {
        int32_t rightCount = 0, rightGradient = 0;
        for (int32_t index = int32_t(nonMissing_[feature]) - 2; index >= 0; --index) {
          if (!hist[index + 1].count) continue;  // Same candidate as the previous bin.
          rightCount += hist[index + 1].count;
          rightGradient += hist[index + 1].gradient;
          const int32_t leftCountNow = int32_t(count) - rightCount;
          if (rightCount < minimum) continue;
          if (leftCountNow < minimum) break;
          const int32_t leftGradientNow = node.gradient - rightGradient;
          if (approx(leftGradientNow, leftCountNow) + approx(rightGradient, rightCount) - parentApprox + margin
              <= std::max(float(featureBest), 0.0f)) continue;
          const double gain = splitGain(parent, leftGradientNow, leftCountNow, rightGradient, rightCount);
          if (gain > featureBest && gain > 0) {
            featureBest = gain;
            binBest = index;
            gradientBest = leftGradientNow;
            missingLeft = true;
          }
        }
      }
      if (binBest >= 0 && featureBest > best) {
        best = featureBest;
        node.feature = int16_t(feature);
        node.bin = uint16_t(binBest);
        node.leftGradient = gradientBest;
        node.missingLeft = missingLeft;
      }
    }
    node.gain = best;
  }

  // Stable partition of the node's samples; returns the left child's end.
  uint16_t partition(const Build &node) {
    uint32_t left = node.begin, right = 0;
    const uint8_t splitBin = uint8_t(node.bin);
    for (uint32_t i = node.begin; i < node.end; ++i) {
      const uint16_t row = samples_[i];
      const uint16_t cell = bins_[uint64_t(row) * features_ + node.feature];
      const bool goLeft = cell == kMissingBin && hasMissing_[node.feature]
          ? node.missingLeft != 0 : cell <= splitBin;
      if (goLeft) samples_[left++] = row;
      else scratch_[right++] = row;
    }
    std::memcpy(samples_ + left, scratch_, right * sizeof(uint16_t));
    return uint16_t(left);
  }

  // Python's heapq on TreeNode (highest gain first): equal gains are common with
  // integer gradients, and the heap's order decides which node gets the last leaves.
  bool before(int16_t a, int16_t b) const { return build_[a].gain > build_[b].gain; }

  void siftDown(uint16_t start, uint16_t pos) {
    const int16_t item = heap_[pos];
    while (pos > start) {
      const uint16_t parent = (pos - 1) >> 1;
      if (!before(item, heap_[parent])) break;
      heap_[pos] = heap_[parent];
      pos = parent;
    }
    heap_[pos] = item;
  }

  void push(int16_t index) {
    heap_[heapSize_] = index;
    siftDown(0, heapSize_++);
  }

  int16_t pop() {
    const int16_t last = heap_[--heapSize_];
    if (!heapSize_) return last;
    const int16_t top = heap_[0];
    heap_[0] = last;
    uint16_t pos = 0, child = 1;
    while (child < heapSize_) {
      const uint16_t right = child + 1;
      if (right < heapSize_ && !before(heap_[child], heap_[right])) child = right;
      heap_[pos] = heap_[child];
      pos = child;
      child = 2 * pos + 1;
    }
    heap_[pos] = last;
    siftDown(0, pos);
    return top;
  }

  void considerSplit(int16_t index) {
    Build &node = build_[index];
    if (uint32_t(node.end) - node.begin < 2u * params_.minSamplesLeaf) return;
    findSplit(node);
    if (node.gain > 0) push(index);
  }

  void growTree(const Real *y, Forest &out) {
    for (uint32_t row = 0; row < rows_; ++row) samples_[row] = uint16_t(row);
    int32_t gradient = 0;
    for (uint32_t row = 0; row < rows_; ++row) gradient += gradient_[row];
    int16_t built = 1;
    build_[0] = Build{0, uint16_t(rows_), gradient, -1, -1, 0, 0, 0, -1, -1, 0};
    heapSize_ = 0;
    considerSplit(0);
    uint16_t leaves = 1;
    while (heapSize_) {
      Build &node = build_[pop()];
      const uint16_t middle = partition(node);
      const int16_t left = built++, right = built++;
      node.left = left;
      node.right = right;
      // A feature with no missing value in training sends NaN to the larger child (grower.py).
      if (!hasMissing_[node.feature]) node.missingLeft = middle - node.begin > node.end - middle;
      const uint8_t depth = node.depth + 1;
      build_[left] = Build{node.begin, middle, node.leftGradient, -1, -1, 0, 0, 0, -1, -1, depth};
      build_[right] = Build{middle, node.end, node.gradient - node.leftGradient, -1, -1, 0, 0, 0, -1, -1, depth};
      ++leaves;
      if (leaves == params_.maxLeaves) break;
      if (params_.maxDepth && depth == params_.maxDepth) continue;  // Same check, same order, as grower.py.
      considerSplit(left);
      considerSplit(right);
    }
    // Emit nodes; every leaf takes the learning rate times the median residual.
    Node *nodes = out.nodes + out.nodeCount;
    for (int16_t i = 0; i < built; ++i) {
      const Build &item = build_[i];
      Node &node = nodes[i];
      node.reserved = 0;
      if (item.left >= 0) {
        node.feature = item.feature;
        node.left = uint16_t(item.left);
        node.right = uint16_t(item.right);
        node.missingLeft = item.missingLeft;
        node.value = 0;
        node.threshold = item.bin >= nonMissing_[item.feature] - 1
            ? std::numeric_limits<Real>::infinity()
            : Real(thresholds_[uint32_t(item.feature) * kMaxBins + item.bin]);
      } else {
        const uint32_t count = uint32_t(item.end) - item.begin;
        for (uint32_t k = 0; k < count; ++k) {
          const uint16_t row = samples_[item.begin + k];
          residual_[k] = y[row] - raw_[row];
        }
        const double update = params_.learningRate * median(residual_, count);
        node.feature = -1;
        node.left = node.right = 0;
        node.missingLeft = 0;
        node.threshold = 0;
        node.value = float(update);
        for (uint32_t k = 0; k < count; ++k) raw_[samples_[item.begin + k]] += update;
      }
    }
    out.nodeCount += uint32_t(built);
  }

  uint32_t rows_ = 0;
  uint16_t features_ = 0;
  Params params_;
  uint8_t *bins_ = nullptr;
  double *thresholds_ = nullptr;
  uint16_t *nonMissing_ = nullptr;
  uint8_t *hasMissing_ = nullptr;
  Bin *hist_ = nullptr;
  double *raw_ = nullptr;
  int8_t *gradient_ = nullptr;
  uint16_t *samples_ = nullptr;
  uint16_t *scratch_ = nullptr;
  double *residual_ = nullptr;
  Build *build_ = nullptr;
  int16_t heap_[256];
  uint16_t heapSize_ = 0;
  Real *sorted_ = nullptr;
};

}  // namespace hgb
