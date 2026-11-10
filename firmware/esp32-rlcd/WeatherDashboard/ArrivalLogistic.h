#pragma once
// Follows scikit-learn 1.5 (BSD-3-Clause, Copyright (c) 2007-2024 The scikit-learn developers) and the L-BFGS-B 3.0 solver bundled
// with SciPy 1.14 (BSD license; Ciyou Zhu, Richard Byrd, Jorge Nocedal and Jose Luis Morales),
// whose MINPACK-2 dcsrch/dcstep line search detail::LineSearch translates closely.
// Notices: THIRD_PARTY_NOTICES.md and LICENSES/ in the public firmware package.
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
//
// The +90 minute arrival-change classifier of the laptop server
// (arrival_logistic_model.py v2), fitted and evaluated on the device:
//
//   make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
//                 StandardScaler(),
//                 LogisticRegression(C=0.1, solver="lbfgs", max_iter=500, tol=1e-4))   // multinomial
//
// with recall sample weights (both fall classes 2, the others 1, divided by their
// mean) on the 28 sensor inputs (the first 28 columns of PmFeatures.h) at 5-minute
// issue clocks over 28 days.
//
// Portable C++ with no Arduino dependency, so the same file runs in the host parity
// test and on the ESP32-S3. The solver takes the iterates of SciPy's L-BFGS-B without
// bounds (10 corrections, the More-Thuente line search, the same stopping tests).
//
// ARRIVAL_KERNEL is the type of the per-row arithmetic only: the scaled matrix, the
// logits, the softmax and the running gradient. `double` reproduces scikit-learn's
// iterates (host parity, seven 28-day windows: probability gap below 3e-8, the same
// iteration and evaluation counts). `float` is far cheaper on the hardware FPU but
// steers the solver onto a slightly different path (same runs: mean gap up to 4e-4,
// worst row 1.5e-2, top class differs on up to 2 of about 8000 rows). The L-BFGS state,
// the loss total and the final coefficients stay double in both cases, because they
// cost almost nothing and the line search needs them.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>

#ifndef ARRIVAL_ALLOC
#define ARRIVAL_ALLOC(bytes) malloc(bytes)
#endif
#ifndef ARRIVAL_FREE
#define ARRIVAL_FREE(pointer) free(pointer)
#endif
#ifndef ARRIVAL_KERNEL
#define ARRIVAL_KERNEL double
#endif

namespace arrival {

using Kernel = ARRIVAL_KERNEL;

constexpr int kClasses = 5;  // fall40, fall20to40, within20, rise20to40, rise40
constexpr int kInputs = 28;
constexpr int kMaxFeatures = 2 * kInputs;   // Inputs plus one missing-indicator each, at most.
constexpr int kMaxDof = kMaxFeatures + 1;   // Plus the intercept.
constexpr int kLookbackDays = 28;
constexpr int kEmbargoSeconds = 7200;       // Training issues end before cutoff - 2 h.
constexpr int kMinRows = 300, kMinDays = 10;

// ------------------------------------------------------------------- labels
// Inclusive 20/40 boundaries of the change from the fresh reference.
inline int labelClass(double delta) {
  if (delta <= -40) return 0;
  if (delta <= -20) return 1;
  if (delta < 20) return 2;
  if (delta < 40) return 3;
  return 4;
}

struct Label {
  double arrival, delta, completeEpoch;
  int cls;
};

// The exact +90 target of a training issue: the two closed 15-minute medians around
// issue + 90 min, linearly interpolated. medians[k] is the bucket ending bucket0 + k * 900
// (pm::bucketMedian with the rows visible at the cutoff; NaN when not eligible).
// False when a bucket is missing or closes at or after the cutoff.
inline bool arrivalLabel(const double *medians, size_t count, double bucket0, int64_t issue, double reference,
                         int64_t cutoff, Label &out) {
  const int64_t origin = issue / 900 * 900;
  const int64_t lead = issue - origin + 5400;
  const int64_t lower = lead / 900;
  const int64_t remainder = lead - lower * 900;
  const int spans = remainder == 0 ? 1 : 2;
  const double fraction = double(remainder) / 900;
  const double weights[2] = {spans == 1 ? 1.0 : 1 - fraction, fraction};
  const int64_t complete = origin + (lower + spans - 1) * 900;
  if (complete >= cutoff) return false;
  double arrival = 0;
  for (int span = 0; span < spans; ++span) {
    const double at = (double(origin + (lower + span) * 900) - bucket0) / 900;
    if (at < 0 || at >= double(count)) return false;
    const double value = medians[size_t(at)];
    if (!std::isfinite(value)) return false;
    arrival += value * weights[span];
  }
  out.arrival = arrival;
  out.delta = arrival - reference;
  out.completeEpoch = double(complete);
  out.cls = labelClass(out.delta);
  return true;
}

// The server refuses a fit below 300 rows or 10 distinct local days. Issues must be sorted.
inline bool enoughHistory(const double *issues, size_t rows) {
  if (rows < size_t(kMinRows)) return false;
  int days = 0;
  int64_t last = -1;
  for (size_t i = 0; i < rows; ++i) {
    const int64_t day = (int64_t(issues[i]) + 8 * 3600) / 86400;
    if (day != last) { ++days; last = day; }
  }
  return days >= kMinDays;
}

// ------------------------------------------------------------------- the model
struct Params {
  double C = 0.1, tol = 1e-4;
  double ftol = 64 * std::numeric_limits<double>::epsilon();
  uint16_t maxIter = 500;
  uint8_t memory = 10, maxLineSearch = 50;
};

// Plain data: can be written to and read from a file as it is (about 3.4 kB).
struct Model {
  uint8_t classCount, featureCount, indicatorCount, reserved;
  uint8_t classId[kClasses];           // Observed classes, ascending.
  uint8_t indicatorColumn[kInputs];    // Input columns that had a NaN at fit, ascending.
  double median[kInputs];
  double mean[kMaxFeatures], scale[kMaxFeatures];
  double coef[kClasses][kMaxDof];      // coef[k][featureCount] is the intercept.
};

struct FitInfo {
  uint16_t iterations = 0, evaluations = 0, skippedUpdates = 0, restarts = 0;
  bool converged = false;  // False when stopped by maxIter or an abnormal line search.
  double loss = 0, gradientNorm = 0;
};

// Ordered mass over the five classes (zero for a class absent from the training window).
// x holds the 28 inputs; NaN is a missing input. False for an infinite input.
inline bool predict(const Model &m, const double *x, double probability[kClasses]) {
  double z[kMaxFeatures];
  for (int j = 0; j < kInputs; ++j) {
    if (std::isinf(x[j])) return false;
    z[j] = ((std::isnan(x[j]) ? m.median[j] : x[j]) - m.mean[j]) / m.scale[j];
  }
  for (int t = 0; t < m.indicatorCount; ++t) {
    const int at = kInputs + t;
    z[at] = ((std::isnan(x[m.indicatorColumn[t]]) ? 1.0 : 0.0) - m.mean[at]) / m.scale[at];
  }
  double raw[kClasses];
  double top = -std::numeric_limits<double>::infinity();
  for (int k = 0; k < m.classCount; ++k) {
    double sum = m.coef[k][m.featureCount];
    for (int j = 0; j < m.featureCount; ++j) sum += m.coef[k][j] * z[j];
    raw[k] = sum;
    top = std::max(top, sum);
  }
  double total = 0;
  for (int k = 0; k < m.classCount; ++k) { raw[k] = std::exp(raw[k] - top); total += raw[k]; }
  for (int k = 0; k < kClasses; ++k) probability[k] = 0;
  for (int k = 0; k < m.classCount; ++k) probability[m.classId[k]] = raw[k] / total;
  return true;
}

// Tail sums of the one distribution, as the server publishes them.
struct Tails {
  double fall20, fall40, rise20, rise40, within20, within40;
};
inline Tails tails(const double p[kClasses]) {
  return {p[0] + p[1], p[0], p[3] + p[4], p[4], p[2], p[1] + p[2] + p[3]};
}

// ------------------------------------------------------------------- the solver
namespace detail {

// The More-Thuente line search (MINPACK-2 dcsrch / dcstep) as built into L-BFGS-B.
struct LineSearch {
  enum Status { Evaluate, Done };
  static constexpr double kFtol = 1e-3, kGtol = 0.9, kXtol = 0.1, kStepMin = 0.0, kStepMax = 1e10;
  bool bracketed = false;
  int stage = 1;
  double finit = 0, ginit = 0, gtest = 0, width = 0, width1 = 0;
  double stx = 0, fx = 0, gx = 0, sty = 0, fy = 0, gy = 0, stmin = 0, stmax = 0;

  void start(double f, double g, double stp) {
    bracketed = false;
    stage = 1;
    finit = f;
    ginit = g;
    gtest = kFtol * ginit;
    width = kStepMax - kStepMin;
    width1 = width / 0.5;
    stx = 0; fx = finit; gx = ginit;
    sty = 0; fy = finit; gy = ginit;
    stmin = 0;
    stmax = stp + 4.0 * stp;
  }

  static void cstep(double &stx, double &fx, double &dx, double &sty, double &fy, double &dy, double &stp,
                    double fp, double dp, bool &brackt, double stpmin, double stpmax) {
    const double sgnd = dp * (dx / std::fabs(dx));
    double stpf, stpc, stpq, theta, s, gamma, p, q, r;
    if (fp > fx) {
      theta = 3 * (fx - fp) / (stp - stx) + dx + dp;
      s = std::max(std::max(std::fabs(theta), std::fabs(dx)), std::fabs(dp));
      gamma = s * std::sqrt((theta / s) * (theta / s) - (dx / s) * (dp / s));
      if (stp < stx) gamma = -gamma;
      p = (gamma - dx) + theta;
      q = ((gamma - dx) + gamma) + dp;
      r = p / q;
      stpc = stx + r * (stp - stx);
      stpq = stx + ((dx / ((fx - fp) / (stp - stx) + dx)) / 2) * (stp - stx);
      stpf = std::fabs(stpc - stx) < std::fabs(stpq - stx) ? stpc : stpc + (stpq - stpc) / 2;
      brackt = true;
    } else if (sgnd < 0) {
      theta = 3 * (fx - fp) / (stp - stx) + dx + dp;
      s = std::max(std::max(std::fabs(theta), std::fabs(dx)), std::fabs(dp));
      gamma = s * std::sqrt((theta / s) * (theta / s) - (dx / s) * (dp / s));
      if (stp > stx) gamma = -gamma;
      p = (gamma - dp) + theta;
      q = ((gamma - dp) + gamma) + dx;
      r = p / q;
      stpc = stp + r * (stx - stp);
      stpq = stp + (dp / (dp - dx)) * (stx - stp);
      stpf = std::fabs(stpc - stp) > std::fabs(stpq - stp) ? stpc : stpq;
      brackt = true;
    } else if (std::fabs(dp) < std::fabs(dx)) {
      theta = 3 * (fx - fp) / (stp - stx) + dx + dp;
      s = std::max(std::max(std::fabs(theta), std::fabs(dx)), std::fabs(dp));
      gamma = s * std::sqrt(std::max(0.0, (theta / s) * (theta / s) - (dx / s) * (dp / s)));
      if (stp > stx) gamma = -gamma;
      p = (gamma - dp) + theta;
      q = (gamma + (dx - dp)) + gamma;
      r = p / q;
      if (r < 0 && gamma != 0) stpc = stp + r * (stx - stp);
      else if (stp > stx) stpc = stpmax;
      else stpc = stpmin;
      stpq = stp + (dp / (dp - dx)) * (stx - stp);
      if (brackt) {
        stpf = std::fabs(stpc - stp) < std::fabs(stpq - stp) ? stpc : stpq;
        if (stp > stx) stpf = std::min(stp + 0.66 * (sty - stp), stpf);
        else stpf = std::max(stp + 0.66 * (sty - stp), stpf);
      } else {
        stpf = std::fabs(stpc - stp) > std::fabs(stpq - stp) ? stpc : stpq;
        stpf = std::min(stpmax, stpf);
        stpf = std::max(stpmin, stpf);
      }
    } else {
      if (brackt) {
        theta = 3 * (fp - fy) / (sty - stp) + dy + dp;
        s = std::max(std::max(std::fabs(theta), std::fabs(dy)), std::fabs(dp));
        gamma = s * std::sqrt((theta / s) * (theta / s) - (dy / s) * (dp / s));
        if (stp > sty) gamma = -gamma;
        p = (gamma - dp) + theta;
        q = ((gamma - dp) + gamma) + dy;
        r = p / q;
        stpf = stp + r * (sty - stp);
      } else if (stp > stx) {
        stpf = stpmax;
      } else {
        stpf = stpmin;
      }
    }
    if (fp > fx) {
      sty = stp; fy = fp; dy = dp;
    } else {
      if (sgnd < 0) { sty = stx; fy = fx; dy = dx; }
      stx = stp; fx = fp; dx = dp;
    }
    stp = stpf;
  }

  // f and g at stp; returns Done on convergence or a warning, else a new stp to try.
  Status advance(double f, double g, double &stp) {
    const double ftest = finit + stp * gtest;
    if (stage == 1 && f <= ftest && g >= std::min(kFtol, kGtol) * ginit) stage = 2;
    bool stop = false;
    if (bracketed && (stp <= stmin || stp >= stmax)) stop = true;
    if (bracketed && stmax - stmin <= kXtol * stmax) stop = true;
    if (stp == kStepMax && f <= ftest && g <= gtest) stop = true;
    if (stp == kStepMin && (f > ftest || g >= gtest)) stop = true;
    if (f <= ftest && std::fabs(g) <= kGtol * (-ginit)) stop = true;
    if (stop) return Done;
    if (stage == 1 && f <= fx && f > ftest) {
      double fm = f - stp * gtest, fxm = fx - stx * gtest, fym = fy - sty * gtest;
      double gm = g - gtest, gxm = gx - gtest, gym = gy - gtest;
      cstep(stx, fxm, gxm, sty, fym, gym, stp, fm, gm, bracketed, stmin, stmax);
      fx = fxm + stx * gtest;
      fy = fym + sty * gtest;
      gx = gxm + gtest;
      gy = gym + gtest;
    } else {
      cstep(stx, fx, gx, sty, fy, gy, stp, f, g, bracketed, stmin, stmax);
    }
    if (bracketed) {
      if (std::fabs(sty - stx) >= 0.66 * width1) stp = stx + 0.5 * (sty - stx);
      width1 = width;
      width = std::fabs(sty - stx);
      stmin = std::min(stx, sty);
      stmax = std::max(stx, sty);
    } else {
      stmin = stp + 1.1 * (stp - stx);
      stmax = stp + 4.0 * (stp - stx);
    }
    stp = std::max(stp, kStepMin);
    stp = std::min(stp, kStepMax);
    if ((bracketed && (stp <= stmin || stp >= stmax)) || (bracketed && stmax - stmin <= kXtol * stmax)) stp = stx;
    return Evaluate;
  }
};

class Solver {
 public:
  ~Solver() { release(); }

  bool run(const double *X, const uint8_t *labels, uint32_t rows, const Params &params, Model &m, FitInfo &info,
           void (*yield)()) {
    n_ = rows;
    p_ = params;
    yield_ = yield;
    memset(&m, 0, sizeof(m));
    if (rows == 0 || params.memory == 0) return false;
    if (!prepareClasses(labels, m)) return false;
    if (!imputeAndScale(X, m)) return false;
    if (!allocateSolverState()) return false;
    const bool ran = minimise(info);
    if (ran) {
      for (int k = 0; k < K_; ++k)
        for (int j = 0; j < dof_; ++j) m.coef[k][j] = x_[k * dof_ + j];
    }
    release();
    return ran;
  }

 private:
  static int lowestBit(uint32_t bits) {
    int at = 0;
    while (!(bits & 1)) { bits >>= 1; ++at; }
    return at;
  }

  template <typename T>
  static T *get(size_t count) {
    return static_cast<T *>(ARRIVAL_ALLOC(count * sizeof(T)));
  }

  void release() {
    ARRIVAL_FREE(y_); ARRIVAL_FREE(z_); ARRIVAL_FREE(mask_); ARRIVAL_FREE(state_); ARRIVAL_FREE(kw_); ARRIVAL_FREE(ki_);
    ARRIVAL_FREE(acc_); ARRIVAL_FREE(rho_); ARRIVAL_FREE(alpha_);
    y_ = nullptr; z_ = nullptr; mask_ = nullptr; state_ = nullptr; kw_ = nullptr; ki_ = nullptr; acc_ = nullptr;
    rho_ = nullptr; alpha_ = nullptr;
  }

  bool prepareClasses(const uint8_t *labels, Model &m) {
    uint32_t counts[kClasses] = {0, 0, 0, 0, 0};
    for (uint32_t i = 0; i < n_; ++i) {
      if (labels[i] >= kClasses) return false;
      ++counts[labels[i]];
    }
    int8_t encode[kClasses];
    K_ = 0;
    for (int c = 0; c < kClasses; ++c) {
      encode[c] = counts[c] ? int8_t(K_) : int8_t(-1);
      if (counts[c]) m.classId[K_++] = uint8_t(c);
    }
    if (K_ < 2) return false;
    m.classCount = uint8_t(K_);
    y_ = get<uint8_t>(n_);
    if (!y_) return false;
    for (uint32_t i = 0; i < n_; ++i) y_[i] = uint8_t(encode[labels[i]]);
    // Recall weights: fall classes 2, others 1, divided by their mean over the rows.
    double total = 0;
    for (int c = 0; c < kClasses; ++c) total += counts[c] * (c <= 1 ? 2.0 : 1.0);
    const double mean = total / n_;
    swSum_ = 0;
    for (int k = 0; k < K_; ++k) {
      weight_[k] = (m.classId[k] <= 1 ? 2.0 : 1.0) / mean;
      kweight_[k] = Kernel(weight_[k]);
      swSum_ += counts[m.classId[k]] * weight_[k];
    }
    return true;
  }

  bool imputeAndScale(const double *X, Model &m) {
    double *column = get<double>(n_);
    if (!column) return false;
    int indicators = 0;
    for (int j = 0; j < kInputs; ++j) {
      uint32_t used = 0;
      for (uint32_t i = 0; i < n_; ++i) {
        const double v = X[size_t(i) * kInputs + j];
        if (std::isinf(v)) { ARRIVAL_FREE(column); return false; }
        if (!std::isnan(v)) column[used++] = v;
      }
      if (used != n_) m.indicatorColumn[indicators++] = uint8_t(j);
      if (!used) { m.median[j] = 0; continue; }  // keep_empty_features
      std::sort(column, column + used);
      m.median[j] = (used & 1) ? column[used / 2] : (column[used / 2 - 1] + column[used / 2]) / 2;
      tick();
    }
    m.indicatorCount = uint8_t(indicators);
    indicators_ = indicators;
    d_ = kInputs + indicators;
    m.featureCount = uint8_t(d_);
    dof_ = d_ + 1;
    nvar_ = K_ * dof_;
    z_ = get<Kernel>(size_t(n_) * kInputs);
    mask_ = get<uint32_t>(n_);
    if (!z_ || !mask_) { ARRIVAL_FREE(column); return false; }
    for (uint32_t i = 0; i < n_; ++i) {
      uint32_t bits = 0;
      for (int t = 0; t < indicators; ++t)
        if (std::isnan(X[size_t(i) * kInputs + m.indicatorColumn[t]])) bits |= uint32_t(1) << t;
      mask_[i] = bits;
    }
    const auto value = [&](uint32_t i, int f) {
      if (f < kInputs) {
        const double v = X[size_t(i) * kInputs + f];
        return std::isnan(v) ? m.median[f] : v;
      }
      return std::isnan(X[size_t(i) * kInputs + m.indicatorColumn[f - kInputs]]) ? 1.0 : 0.0;
    };
    const double eps = std::numeric_limits<double>::epsilon();
    for (int f = 0; f < d_; ++f) {
      // StandardScaler: the corrected two-pass mean and variance.
      double sum = 0;
      for (uint32_t i = 0; i < n_; ++i) sum += value(i, f);
      const double mean = sum / n_;
      double correction = 0, square = 0;
      for (uint32_t i = 0; i < n_; ++i) {
        const double t = value(i, f) - mean;
        correction += t;
        square += t * t;
      }
      const double variance = (square - correction * correction / n_) / n_;
      const double bound = n_ * eps * variance + (n_ * mean * eps) * (n_ * mean * eps);
      const double scale = variance <= bound ? 1.0 : std::sqrt(variance);
      m.mean[f] = mean;
      m.scale[f] = scale;
      if (f < kInputs) {
        for (uint32_t i = 0; i < n_; ++i) z_[size_t(i) * kInputs + f] = Kernel((value(i, f) - mean) / scale);
      } else {
        // A scaled indicator takes two values: absent[t], or absent[t] + step[t] when missing.
        absent_[f - kInputs] = (0.0 - mean) / scale;
        step_[f - kInputs] = (1.0 - mean) / scale - absent_[f - kInputs];
      }
      tick();
    }
    ARRIVAL_FREE(column);
    return true;
  }

  bool allocateSolverState() {
    const size_t vectors = 5 + 2u * p_.memory;
    state_ = get<double>(vectors * nvar_);
    kw_ = get<Kernel>(size_t(K_) * kInputs);
    ki_ = get<Kernel>(size_t(K_) * (indicators_ ? indicators_ : 1));
    acc_ = get<Kernel>(nvar_);
    rho_ = get<double>(p_.memory);
    alpha_ = get<double>(p_.memory);
    if (!state_ || !kw_ || !ki_ || !acc_ || !rho_ || !alpha_) return false;
    x_ = state_;
    g_ = x_ + nvar_;
    t_ = g_ + nvar_;
    r_ = t_ + nvar_;
    d_v_ = r_ + nvar_;
    s_ = d_v_ + nvar_;
    y_v_ = s_ + size_t(p_.memory) * nvar_;
    return true;
  }

  void tick() { if (yield_) yield_(); }

  double dot(const double *a, const double *b) const {
    double sum = 0;
    for (int c = 0; c < nvar_; ++c) sum += a[c] * b[c];
    return sum;
  }

  double norm() const {
    double top = 0;
    for (int c = 0; c < nvar_; ++c) top = std::max(top, std::fabs(g_[c]));
    return top;
  }

  void flush() {
    const int used = (K_ - 1) * dof_;
    for (int c = 0; c < used; ++c) { g_[c] += acc_[c]; acc_[c] = 0; }
  }

  // Loss and gradient at x_: the weighted mean of the multinomial loss plus the L2 term.
  // Three exact shortcuts keep the arithmetic down: 8 * 28 multiply-adds a row instead of 10 * 41.
  // A scaled indicator is a constant plus a step on the few rows where its input was missing, so
  // it costs nothing on the other rows. Softmax depends only on differences of logits, so the last
  // class needs no logit. The residuals of a row sum to zero, so the last class's data gradient
  // is the negated sum of the others'.
  void evaluate() {
    const int K = K_, dof = dof_, I = indicators_, last = K - 1;
    Kernel base[kClasses];
    const double *lastW = x_ + last * dof;
    for (int k = 0; k < last; ++k) {
      const double *w = x_ + k * dof;
      double constant = w[kInputs + I] - lastW[kInputs + I];
      for (int t = 0; t < I; ++t) {
        const double difference = w[kInputs + t] - lastW[kInputs + t];
        constant += difference * absent_[t];
        ki_[k * I + t] = Kernel(difference * step_[t]);
      }
      base[k] = Kernel(constant);
      for (int j = 0; j < kInputs; ++j) kw_[k * kInputs + j] = Kernel(w[j] - lastW[j]);
    }
    for (int c = 0; c < nvar_; ++c) { g_[c] = 0; acc_[c] = 0; }
    const uint32_t flushEvery = sizeof(Kernel) == sizeof(float) ? 64 : 0xFFFFFFFFu;
    double lossSum = 0;
    uint32_t pending = 0;
    for (uint32_t i = 0; i < n_; ++i) {
      const Kernel *z = z_ + size_t(i) * kInputs;
      const uint32_t mask = mask_[i];
      Kernel raw[kClasses], mass[kClasses];
      raw[last] = 0;
      int best = last;
      for (int k = 0; k < last; ++k) {
        const Kernel *w = kw_ + k * kInputs;
        Kernel sum = base[k];
        for (int j = 0; j < kInputs; ++j) sum += w[j] * z[j];
        for (uint32_t bits = mask; bits; bits &= bits - 1) sum += ki_[k * I + lowestBit(bits)];
        raw[k] = sum;
        if (sum > raw[best]) best = k;
      }
      const Kernel top = raw[best];
      Kernel total = 0;
      for (int k = 0; k < K; ++k) {
        mass[k] = k == best ? Kernel(1) : Kernel(std::exp(raw[k] - top));
        total += mass[k];
      }
      const int label = y_[i];
      lossSum += weight_[label] * (double(top - raw[label]) + double(std::log(total)));
      const Kernel sw = kweight_[label];
      for (int k = 0; k < last; ++k) {
        Kernel residual = mass[k] / total;
        if (k == label) residual -= Kernel(1);
        residual *= sw;
        Kernel *a = acc_ + k * dof;
        for (int j = 0; j < kInputs; ++j) a[j] += residual * z[j];
        for (uint32_t bits = mask; bits; bits &= bits - 1) a[kInputs + lowestBit(bits)] += residual;
        a[kInputs + I] += residual;
      }
      if (++pending == flushEvery) { flush(); pending = 0; }
      if (yield_ && !(i & 2047)) yield_();
    }
    flush();
    for (int k = 0; k < last; ++k) {
      double *a = g_ + k * dof;
      for (int t = 0; t < I; ++t) a[kInputs + t] = absent_[t] * a[kInputs + I] + step_[t] * a[kInputs + t];
    }
    for (int j = 0; j < dof; ++j) {
      double sum = 0;
      for (int k = 0; k < last; ++k) sum += g_[k * dof + j];
      g_[last * dof + j] = -sum;
    }
    const double l2 = 1.0 / (p_.C * swSum_);
    double square = 0;
    for (int k = 0; k < K; ++k) {
      for (int j = 0; j < dof; ++j) {
        const int c = k * dof + j;
        g_[c] /= swSum_;
        if (j < d_) { square += x_[c] * x_[c]; g_[c] += l2 * x_[c]; }
      }
    }
    f_ = lossSum / swSum_ + 0.5 * l2 * square;
    ++evals_;
#ifdef ARRIVAL_TRACE
    ARRIVAL_TRACE(f_, norm());
#endif
  }

  void direction(int col, int head, double theta) {
    const int m = p_.memory;
    const auto pair = [&](int age) { return (head + age) % m; };
    for (int c = 0; c < nvar_; ++c) d_v_[c] = g_[c];
    for (int age = col - 1; age >= 0; --age) {
      const int at = pair(age);
      const double a = rho_[at] * dot(s_ + size_t(at) * nvar_, d_v_);
      alpha_[at] = a;
      const double *y = y_v_ + size_t(at) * nvar_;
      for (int c = 0; c < nvar_; ++c) d_v_[c] -= a * y[c];
    }
    const double gamma = 1.0 / theta;
    for (int c = 0; c < nvar_; ++c) d_v_[c] *= gamma;
    for (int age = 0; age < col; ++age) {
      const int at = pair(age);
      const double b = rho_[at] * dot(y_v_ + size_t(at) * nvar_, d_v_);
      const double *s = s_ + size_t(at) * nvar_;
      const double a = alpha_[at] - b;
      for (int c = 0; c < nvar_; ++c) d_v_[c] += a * s[c];
    }
    for (int c = 0; c < nvar_; ++c) d_v_[c] = -d_v_[c];
  }

  bool minimise(FitInfo &info) {
    const double eps = std::numeric_limits<double>::epsilon();
    const int m = p_.memory;
    evals_ = 0;
    for (int c = 0; c < nvar_; ++c) x_[c] = 0;
    evaluate();
    int iter = 0, col = 0, head = 0, skipped = 0, restarts = 0;
    double theta = 1;
    bool converged = false;
    double gnorm = norm();
    if (gnorm <= p_.tol) converged = true;
    while (!converged && iter < p_.maxIter) {
      if (col == 0) {
        for (int c = 0; c < nvar_; ++c) d_v_[c] = -g_[c];
      } else {
        direction(col, head, theta);
      }
      const double dtd = dot(d_v_, d_v_);
      const double dnorm = std::sqrt(dtd);
      double stp = iter == 0 ? std::min(1.0 / dnorm, LineSearch::kStepMax) : 1.0;
      for (int c = 0; c < nvar_; ++c) { t_[c] = x_[c]; r_[c] = g_[c]; }
      const double fold = f_;
      double gd = dot(g_, d_v_);
      const double gdold = gd;
      bool failed = gd >= 0;
      if (!failed) {
        LineSearch search;
        search.start(f_, gd, stp);
        int ifun = 0, iback = 0;
        for (;;) {
          ++ifun;
          iback = ifun - 1;
          if (iback >= p_.maxLineSearch) { failed = true; break; }
          for (int c = 0; c < nvar_; ++c) x_[c] = stp * d_v_[c] + t_[c];
          evaluate();
          gd = dot(g_, d_v_);
          if (search.advance(f_, gd, stp) == LineSearch::Done) break;
        }
      }
      if (failed) {
        for (int c = 0; c < nvar_; ++c) { x_[c] = t_[c]; g_[c] = r_[c]; }
        f_ = fold;
        if (col == 0) break;  // Abnormal termination, as in L-BFGS-B.
        col = 0; head = 0; theta = 1;
        ++restarts;
        continue;
      }
      ++iter;
      gnorm = norm();
      if (gnorm <= p_.tol) { converged = true; break; }
      if ((fold - f_) <= p_.ftol * std::max(std::max(std::fabs(fold), std::fabs(f_)), 1.0)) { converged = true; break; }
      // Curvature pair.
      for (int c = 0; c < nvar_; ++c) r_[c] = g_[c] - r_[c];
      const double rr = dot(r_, r_);
      double dr, scale;
      if (stp == 1.0) {
        dr = gd - gdold;
        scale = -gdold;
      } else {
        dr = (gd - gdold) * stp;
        scale = -gdold * stp;
        for (int c = 0; c < nvar_; ++c) d_v_[c] *= stp;
      }
      if (dr <= eps * scale) { ++skipped; continue; }
      int at;
      if (col < m) { at = (head + col) % m; ++col; }
      else { at = head; head = (head + 1) % m; }
      memcpy(s_ + size_t(at) * nvar_, d_v_, nvar_ * sizeof(double));
      memcpy(y_v_ + size_t(at) * nvar_, r_, nvar_ * sizeof(double));
      rho_[at] = 1.0 / dr;
      theta = rr / dr;
    }
    info.iterations = uint16_t(iter);
    info.evaluations = uint16_t(evals_);
    info.skippedUpdates = uint16_t(skipped);
    info.restarts = uint16_t(restarts);
    info.converged = converged;
    info.loss = f_;
    info.gradientNorm = gnorm;
    return true;
  }

  Params p_;
  void (*yield_)() = nullptr;
  uint32_t n_ = 0;
  int K_ = 0, d_ = 0, dof_ = 0, nvar_ = 0, indicators_ = 0;
  int evals_ = 0;
  double swSum_ = 0, f_ = 0;
  double weight_[kClasses];
  double absent_[kInputs], step_[kInputs];
  Kernel kweight_[kClasses];
  uint8_t *y_ = nullptr;
  Kernel *z_ = nullptr, *kw_ = nullptr, *ki_ = nullptr, *acc_ = nullptr;
  uint32_t *mask_ = nullptr;
  double *state_ = nullptr, *rho_ = nullptr, *alpha_ = nullptr;
  double *x_ = nullptr, *g_ = nullptr, *t_ = nullptr, *r_ = nullptr, *d_v_ = nullptr, *s_ = nullptr, *y_v_ = nullptr;
};

}  // namespace detail

// X is rows * 28 doubles, row-major, NaN for a missing input; labels are classes 0..4.
// Transient memory: rows * (13 + 28 * sizeof(Kernel)) bytes plus about 26 * 8 * classes *
// (featureCount + 1) bytes; ARRIVAL_ALLOC can point it at PSRAM.
// yield, when set, runs every 2048 rows so a task can pause or feed a watchdog.
inline bool fit(const double *X, const uint8_t *labels, uint32_t rows, Model &out, FitInfo *info = nullptr,
                void (*yield)() = nullptr, const Params &params = Params()) {
  auto *solver = static_cast<detail::Solver *>(ARRIVAL_ALLOC(sizeof(detail::Solver)));
  if (!solver) return false;
  new (solver) detail::Solver();
  FitInfo local;
  const bool ok = solver->run(X, labels, rows, params, out, info ? *info : local, yield);
  solver->~Solver();
  ARRIVAL_FREE(solver);
  return ok;
}

}  // namespace arrival
