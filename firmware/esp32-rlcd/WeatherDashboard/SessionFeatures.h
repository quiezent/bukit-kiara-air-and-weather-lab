#pragma once
// The imputer follows scikit-learn 1.5 (BSD-3-Clause, Copyright (c) 2007-2024 The scikit-learn developers).
// Notices: THIRD_PARTY_NOTICES.md and LICENSES/ in the public firmware package.
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
//
// The inputs and targets of the laptop server's two-hour session model
// (weather_session_model.py, afternoon_direction_features.py,
// rain_weather_features.py, forecast_freshness.py, collection_quality.py) and
// the mean imputer with missing indicators its pipeline puts in front of the
// trees (scikit-learn SimpleImputer(add_indicator=True)).
//
// Portable C++ with no Arduino dependency: the host test compares every row with
// the server's own feature code. A missing input stays NaN.
//
// Inputs 15..17 are the server's CAMS regional columns. The device does not fetch
// CAMS, so they stay NaN here; the imputer then drops them, as scikit-learn drops
// a column with no observed value.
//
// Fit recipe (weather_session_model.build_dataset and _trained_model), refit at each
// local midnight: origins at :00 and :30, issue = origin, one row per lead in kLeads with
// start = origin + lead and end = start + 2 h. Keep rows with inputRow() true, a finite
// sessionLabel(), labelComplete() < cutoff and cutoff - 14 d <= issue < cutoff; require
// kMinTrainingRows rows and kMinTargetDays distinct local dates of labelComplete().
// Target = label - x[kFresh]; published point = max(0, x[kFresh] + prediction).
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include "PmFeatures.h"

#ifndef SESSION_MAX_HOURS
#define SESSION_MAX_HOURS 48
#endif

namespace ses {

using pm::finite;
using pm::kInf;
using pm::kNaN;
using pm::kPi;
using pm::Reading;

constexpr int kInputs = 31;
constexpr int kMaxOutputs = kInputs * 2;  // Imputed columns plus their missing indicators.
constexpr int kMaxHours = SESSION_MAX_HOURS;
constexpr int kMaxBucketRows = 64;
constexpr int kLeadCount = 7;
constexpr int kLeads[kLeadCount] = {105, 120, 240, 420, 750, 1080, 1380};  // Minutes from the origin to the target's start.
constexpr int kLookbackDays = 14;
constexpr size_t kMinTrainingRows = 120;
constexpr int kMinTargetDays = 10;

// The fixed trees: HistGradientBoostingRegressor(loss="absolute_error", early_stopping=False).
constexpr uint16_t kIterations = 100;
constexpr double kLearningRate = 0.05;
constexpr uint8_t kMaxLeaves = 7;
constexpr uint16_t kMinSamplesLeaf = 80;
constexpr double kL2 = 50.0;
constexpr uint8_t kMaxDepth = 3;

enum Column {
  kFresh, kClosedLevel, kFreshOffset, kDelta15, kDelta30, kDelta60, kDelta120, kSd60, kMean3Offset,
  kTemp, kRh, kTempDelta30, kRhDelta30, kSinTarget, kCosTarget,
  kCamsCurrent, kCamsTarget, kCamsDelta,
  kPriorRainMm, kDuringRainMm, kPriorRainProbMean, kDuringRainProbMean, kFirstWetLeadHours,
  kHoursUntilExpectedWetEnd, kWetTempMinDeltaC, kTargetTempDeltaC, kTargetRhDeltaPct,
  kTargetWind10DeltaKmh, kTargetWind925MeanKmh, kTargetLeadHours, kTargetDurationHours
};

// ----------------------------------------------------------------- issued weather
struct Hour {  // One Open-Meteo row: rain and chance belong to the hour ending at epoch.
  double epoch, rainMm, rainChance, temperature, humidity, wind10, wind925;
};

struct Run {
  double fetched = 0;
  bool invalid = false;
  int count = 0;
  Hour hours[kMaxHours];
};

constexpr double kHour = 3600.0, kEpsilon = 1e-6, kMaxWeatherAge = 7200.0;

// The server's per-value range checks: anything out of range is missing, never clipped.
inline void sanitize(Hour &h) {
  if (!finite(h.rainMm) || h.rainMm < 0) h.rainMm = kNaN;
  if (!finite(h.rainChance) || h.rainChance < 0 || h.rainChance > 100) h.rainChance = kNaN;
  if (!finite(h.temperature)) h.temperature = kNaN;
  if (!finite(h.humidity) || h.humidity < 0 || h.humidity > 100) h.humidity = kNaN;
  if (!finite(h.wind10) || h.wind10 < 0) h.wind10 = kNaN;
  if (!finite(h.wind925) || h.wind925 < 0) h.wind925 = kNaN;
}

using Field = double Hour::*;

inline int bisectLeft(const Run &run, double value) {
  int lo = 0, hi = run.count;
  while (lo < hi) { const int mid = (lo + hi) / 2; if (run.hours[mid].epoch < value) lo = mid + 1; else hi = mid; }
  return lo;
}

inline int bisectRight(const Run &run, double value) {
  int lo = 0, hi = run.count;
  while (lo < hi) { const int mid = (lo + hi) / 2; if (run.hours[mid].epoch <= value) lo = mid + 1; else hi = mid; }
  return lo;
}

struct Block {
  int index;
  double a, b;
};

// Overlap of (start, end] with each row's hour (epoch - 3600, epoch].
inline int blocks(const Run &run, double start, double end, Block *out) {
  if (end <= start) return 0;
  int used = 0;
  const int stop = bisectLeft(run, end + kHour);
  for (int i = bisectRight(run, start); i < stop; ++i) {
    const double a = std::max(start, run.hours[i].epoch - kHour), b = std::min(end, run.hours[i].epoch);
    if (b > a) out[used++] = {i, a, b};
  }
  return used;
}

struct Interval {
  double rainMm, chanceMean;
  bool available;
};

inline Interval interval(const Run &run, double start, double end) {
  const double duration = end - start;
  Block list[kMaxHours];
  const int n = blocks(run, start, end, list);
  double amountCovered = 0, chanceCovered = 0, amount = 0, chance = 0;
  for (int k = 0; k < n; ++k) {
    const Hour &h = run.hours[list[k].index];
    const double span = list[k].b - list[k].a;
    if (finite(h.rainMm)) { amountCovered += span; amount += h.rainMm * span / kHour; }
    if (finite(h.rainChance)) { chanceCovered += span; chance += h.rainChance * span; }
  }
  const bool ac = std::max(0.0, duration - amountCovered) <= kEpsilon;
  const bool pc = std::max(0.0, duration - chanceCovered) <= kEpsilon;
  return {ac ? amount : kNaN, pc && duration != 0 ? chance / duration : kNaN, ac && pc};
}

inline double sample(const Run &run, Field field, double t) {
  const int right = bisectLeft(run, t);
  if (right < run.count && run.hours[right].epoch == t) return run.hours[right].*field;
  const int left = right - 1;
  if (left < 0 || right >= run.count) return kNaN;
  const Hour &l = run.hours[left], &r = run.hours[right];
  if (r.epoch - l.epoch > kHour + kEpsilon) return kNaN;
  const double a = l.*field, b = r.*field;
  if (!finite(a) || !finite(b)) return kNaN;
  const double weight = (t - l.epoch) / (r.epoch - l.epoch);
  return a + weight * (b - a);
}

struct PathStats {
  double mean, low;
  bool ok;
};

// Time-weighted mean and minimum over the hourly knots inside (start, end], end points included.
inline PathStats pathStats(const Run &run, Field field, double start, double end) {
  PathStats out{kNaN, kNaN, false};
  double previousTime = start;
  double previous = sample(run, field, start);
  if (!finite(previous)) return out;
  double area = 0, low = previous;
  const int stop = bisectLeft(run, end);
  for (int i = bisectRight(run, start); i <= stop; ++i) {
    const bool last = i == stop;
    const double t = last ? end : run.hours[i].epoch;
    if (last && !(end > start)) break;
    const double value = sample(run, field, t);
    if (!finite(value) || t - previousTime > kHour + kEpsilon) return out;
    area += (t - previousTime) * (previous + value) / 2;
    low = std::min(low, value);
    previousTime = t;
    previous = value;
  }
  out.mean = end > start ? area / (end - start) : previous;
  out.low = low;
  out.ok = true;
  return out;
}

inline bool isWet(const Hour &h) { return h.rainMm >= 0.1 || h.rainChance >= 60.0; }

// Inputs 18..28 for one issue and target window. Returns the server's "available" flag.
inline bool weatherInputs(const Run *run, double issue, double start, double end, double *x) {
  for (int column = kPriorRainMm; column <= kTargetWind925MeanKmh; ++column) x[column] = kNaN;
  if (!run || run->invalid || issue - run->fetched > kMaxWeatherAge) return false;
  const Interval prior = interval(*run, issue, start), during = interval(*run, start, end);
  if (!prior.available || !during.available) return false;
  x[kPriorRainMm] = prior.rainMm;
  x[kDuringRainMm] = during.rainMm;
  x[kPriorRainProbMean] = prior.chanceMean;
  x[kDuringRainProbMean] = during.chanceMean;

  Block all[kMaxHours], wet[kMaxHours];
  const int n = blocks(*run, issue, end, all);
  int wetCount = 0;
  for (int k = 0; k < n; ++k)
    if (isWet(run->hours[all[k].index])) wet[wetCount++] = all[k];
  if (wetCount) {
    x[kFirstWetLeadHours] = (wet[0].a - issue) / kHour;
    for (int index = wet[wetCount - 1].index + 1; index < run->count; ++index) {
      const Hour &h = run->hours[index];
      if (h.epoch - run->hours[index - 1].epoch > kHour + kEpsilon) break;
      if (!finite(h.rainMm) || !finite(h.rainChance)) break;
      if (!isWet(h)) { x[kHoursUntilExpectedWetEnd] = (run->hours[index - 1].epoch - start) / kHour; break; }
    }
    const double baseline = sample(*run, &Hour::temperature, issue);
    double low = kInf;
    bool ok = finite(baseline);
    for (int k = 0; k < wetCount && ok; ++k) {
      const PathStats path = pathStats(*run, &Hour::temperature, wet[k].a, wet[k].b);
      ok = path.ok;
      if (ok) low = std::min(low, path.low);
    }
    if (ok) x[kWetTempMinDeltaC] = low - baseline;
  }

  const struct { Field field; int column; } deltas[3] = {
      {&Hour::temperature, kTargetTempDeltaC}, {&Hour::humidity, kTargetRhDeltaPct}, {&Hour::wind10, kTargetWind10DeltaKmh}};
  for (const auto &item : deltas) {
    const PathStats path = pathStats(*run, item.field, start, end);
    const double baseline = sample(*run, item.field, issue);
    if (path.ok && finite(baseline)) x[item.column] = path.mean - baseline;
  }
  const PathStats wind925 = pathStats(*run, &Hour::wind925, start, end);
  if (wind925.ok) x[kTargetWind925MeanKmh] = wind925.mean;
  return true;
}

// ------------------------------------------------------------------ sensor inputs
// Median of one closed 15-minute bucket (end - 900, end] of a column, or NaN unless
// at least 3 values with both edges within 240 s and no internal gap above 480 s.
inline double closedBucket(const Reading *rows, size_t n, double end, double Reading::*column) {
  const double start = end - 900;
  double values[kMaxBucketRows];
  int used = 0;
  double first = kNaN, last = kNaN, gap = 0;
  for (size_t i = pm::lowerBoundEpoch(rows, n, start); i < n && rows[i].epoch <= end; ++i) {
    const double value = rows[i].*column;
    if (!finite(value)) continue;
    if (used == kMaxBucketRows) return kNaN;
    values[used++] = value;
    if (finite(first)) gap = std::max(gap, rows[i].epoch - last);
    else first = rows[i].epoch;
    last = rows[i].epoch;
  }
  if (used < 3 || first - start > 240 || end - last > 240 || gap > 480) return kNaN;
  return pm::nanMedian(values, used);
}

// Median of the finite, non-negative PM values in (issue - 300, issue]; NaN when none.
inline double freshReference(const Reading *rows, size_t n, double issue) {
  double values[kMaxBucketRows];
  int used = 0;
  for (size_t i = pm::lowerBoundEpoch(rows, n, issue - 300); i < n && rows[i].epoch <= issue; ++i)
    if (finite(rows[i].pm) && rows[i].pm >= 0 && used < kMaxBucketRows) values[used++] = rows[i].pm;
  return used ? pm::nanMedian(values, used) : kNaN;
}

// Inputs 0..14 and the two lead inputs. `rows` must be sorted by epoch.
inline void sensorInputs(const Reading *rows, size_t n, double issue, double start, double end, double *x) {
  const double origin = std::floor(issue / 900) * 900;
  double level[12];
  for (int k = 0; k < 12; ++k) level[k] = closedBucket(rows, n, origin - k * 900.0, &Reading::pm);
  const double closed = level[0];
  const double fresh = freshReference(rows, n, issue);
  x[kFresh] = fresh;
  x[kClosedLevel] = closed;
  x[kFreshOffset] = fresh - closed;
  static constexpr int steps[4] = {1, 2, 4, 8};
  for (int s = 0; s < 4; ++s) {
    bool complete = true;
    for (int k = 0; k <= steps[s]; ++k) complete = complete && finite(level[k]);
    x[kDelta15 + s] = complete ? closed - level[steps[s]] : kNaN;
  }
  bool four = true;
  for (int k = 0; k < 4; ++k) four = four && finite(level[k]);
  if (four) {
    const double mean = (level[0] + level[1] + level[2] + level[3]) / 4;
    double square = 0;
    for (int k = 0; k < 4; ++k) square += (level[k] - mean) * (level[k] - mean);
    x[kSd60] = std::sqrt(square / 4);
  } else {
    x[kSd60] = kNaN;
  }
  bool twelve = true;
  double sum = 0;
  for (int k = 0; k < 12; ++k) { twelve = twelve && finite(level[k]); sum += level[k]; }
  x[kMean3Offset] = twelve ? sum / 12 - closed : kNaN;

  const struct { double Reading::*source; int level, delta; } weather[2] = {
      {&Reading::temperature, kTemp, kTempDelta30}, {&Reading::humidity, kRh, kRhDelta30}};
  for (const auto &item : weather) {
    const double now = closedBucket(rows, n, origin, item.source);
    const double middle = closedBucket(rows, n, origin - 900, item.source);
    const double past = closedBucket(rows, n, origin - 1800, item.source);
    x[item.level] = now;
    x[item.delta] = finite(now) && finite(middle) && finite(past) ? now - past : kNaN;
  }

  // Target position in the day: the window's middle, Kuala Lumpur clock, whole seconds.
  const int64_t middleSecond = int64_t(std::floor((start + end) / 2));
  const int64_t local = (middleSecond + 8 * 3600) % 86400;
  const double hour = double(local / 3600) + double((local % 3600) / 60) / 60 + double(local % 60) / 3600;
  x[kSinTarget] = std::sin(hour * kPi / 12);
  x[kCosTarget] = std::cos(hour * kPi / 12);
  x[kTargetLeadHours] = (start - issue) / 3600;
  x[kTargetDurationHours] = (end - start) / 3600;
}

// One input row for an issue and a two-hour window. `rows` may hold later readings;
// only those at or before the issue are used. `run` is the newest forecast fetched at
// or before the issue. Returns true when the server would accept the row.
inline bool inputRow(const Reading *rows, size_t n, const Run *run, double issue, double start, double end, double *x) {
  const size_t visible = pm::lowerBoundEpoch(rows, n, issue);
  sensorInputs(rows, visible, issue, start, end, x);
  x[kCamsCurrent] = x[kCamsTarget] = x[kCamsDelta] = kNaN;  // Not collected on the device.
  const bool weather = weatherInputs(run, issue, start, end, x);
  return weather && finite(x[kFresh]) && finite(x[kClosedLevel]);
}

// ------------------------------------------------------------------------ targets
// Mean of the eight closed 15-minute medians of a target that starts leadMinutes
// after the origin; NaN when any is missing. medians[k] closes at first + k * 900 and
// medians[originIndex] closes at the origin.
inline double sessionLabel(const double *medians, size_t count, size_t originIndex, int leadMinutes) {
  const size_t offset = originIndex + size_t(leadMinutes) / 15;
  if (offset + 8 >= count) return kNaN;
  double sum = 0;
  for (int k = 1; k <= 8; ++k) {
    const double value = medians[offset + k];
    if (!finite(value)) return kNaN;
    sum += value * 0.125;
  }
  return sum;
}

// Clock of the last bucket a label needs; a training row needs it strictly before the cutoff.
inline double labelComplete(double origin, int leadMinutes) { return origin + (leadMinutes / 15 + 8) * 900.0; }

// ------------------------------------------------------------------------ imputer
// SimpleImputer(strategy="mean", add_indicator=True) over the 31 inputs: kept columns in
// order (a column with no observed value is dropped), then one 0/1 indicator for each
// column that had a missing value in the fit.
struct Imputer {
  double mean[kInputs];
  uint8_t kept[kInputs];
  uint8_t indicator[kInputs];
  uint8_t outputs = 0;

  void fit(const double *X, size_t rows) {
    uint32_t seen[kInputs] = {0};
    double sum[kInputs] = {0};
    for (size_t r = 0; r < rows; ++r)
      for (int c = 0; c < kInputs; ++c) {
        const double value = X[r * kInputs + c];
        if (std::isnan(value)) continue;
        sum[c] += value;
        ++seen[c];
      }
    outputs = 0;
    for (int c = 0; c < kInputs; ++c) {
      kept[c] = seen[c] != 0;
      indicator[c] = seen[c] != rows;
      mean[c] = seen[c] ? sum[c] / seen[c] : kNaN;
      outputs += kept[c] + indicator[c];
    }
  }

  void transform(const double *x, double *out) const {
    int at = 0;
    for (int c = 0; c < kInputs; ++c)
      if (kept[c]) out[at++] = std::isnan(x[c]) ? mean[c] : x[c];
    for (int c = 0; c < kInputs; ++c)
      if (indicator[c]) out[at++] = std::isnan(x[c]) ? 1.0 : 0.0;
  }
};

}  // namespace ses
