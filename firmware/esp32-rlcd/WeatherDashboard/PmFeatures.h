#pragma once
// The 81 model inputs and the training targets of the laptop server's v25
// spatial models (spatial_point_model.py, fresh_sensor_features.py,
// spatial_live_features.py, spatial_final_models.py, fresh_numeric_model_v4.py).
//
// Portable C++ with no Arduino dependency: the host test compares every row with
// the server's own feature cache. Everything here sees only records visible at
// the issue clock, and a missing input stays NaN.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace pm {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kPi = 3.141592653589793;

struct Reading {  // TTDI sensor, one row per observation, sorted by epoch.
  double epoch, pm, temperature, humidity, visible;
  double heatIndex, pm10;  // Display and status label only; not model inputs.
};

struct Poll {  // Neighbour station "current" value, sorted by end.
  int32_t station;
  int32_t reserved;
  double end, value, receipt;
};

struct Wind {  // Toward-vectors at +0, +30, +60, +90 min from the issued run.
  double u10[4], v10[4], u925[4], v925[4];
};

constexpr int kStations = 6;
struct Station {
  int32_t id;
  double latitude, longitude;
};
constexpr Station kStationTable[kStations] = {
    {1087, 3.1113299, 101.476412},      // setia
    {52426946, 2.913, 101.640583},      // cyber
    {98982064, 3.02072, 101.714186},    // enviro
    {99667152, 3.1696, 101.6528},       // montkiara
    {109266655, 3.1922, 101.72318},     // setapak
    {111145052, 3.15665, 101.70853},    // klcc
};
constexpr double kTtdiLatitude = 3.1411106, kTtdiLongitude = 101.6274985;

constexpr int kSensorInputs = 28, kControls = 4, kTimeInputs = 5, kStationInputs = 18, kNetInputs = 10, kWindInputs = 16;
constexpr int kInputCount = kSensorInputs + kControls + kTimeInputs + kStationInputs + kNetInputs + kWindInputs;  // 81
constexpr int kSeqBins = 60;
constexpr int kMaxWindowRows = 400;

inline bool finite(double v) { return std::isfinite(v); }

// numpy.median over the finite values of a small array (values is reordered).
inline double nanMedian(double *values, int count) {
  int used = 0;
  for (int i = 0; i < count; ++i)
    if (finite(values[i])) values[used++] = values[i];
  if (!used) return kNaN;
  std::sort(values, values + used);
  return (used & 1) ? values[used / 2] : 0.5 * (values[used / 2 - 1] + values[used / 2]);
}

inline double nanMean(const double *values, int count) {
  double sum = 0;
  int used = 0;
  for (int i = 0; i < count; ++i)
    if (finite(values[i])) { sum += values[i]; ++used; }
  return used ? sum / used : kNaN;
}

// AirGradient's EPA correction (getEpaCorrectedPM), as used on the public map.
inline double epaCorrect(double pm, double rh) {
  if (!finite(pm)) return kNaN;
  if (pm == 0) return 0;
  if (!finite(rh)) return std::max(pm, 0.0);
  double out;
  if (pm < 30) out = 0.524 * pm - 0.0862 * rh + 5.75;
  else if (pm < 50) { const double a = pm / 20 - 1.5; out = (0.786 * a + 0.524 * (1 - a)) * pm - 0.0862 * rh + 5.75; }
  else if (pm < 210) out = 0.786 * pm - 0.0862 * rh + 5.75;
  else if (pm < 260) {
    const double a = pm / 50 - 21.0 / 5;
    out = (0.69 * a + 0.786 * (1 - a)) * pm - 0.0862 * rh * (1 - a) + 2.966 * a + 5.75 * (1 - a);
  } else out = 2.966 + 0.69 * pm + 8.84e-4 * pm * pm;
  return std::max(out, 0.0);
}

inline double weightedMedian(const double *values, const double *weights, int count) {
  int order[kStations];
  for (int i = 0; i < count; ++i) order[i] = i;
  const auto key = [&](int i) { return finite(values[i]) ? values[i] : kInf; };
  std::stable_sort(order, order + count, [&](int a, int b) { return key(a) < key(b); });
  double total = 0;
  for (int i = 0; i < count; ++i) total += finite(values[i]) ? weights[i] : 0.0;
  if (!(total > 0)) return kNaN;
  double cumulative = 0;
  for (int i = 0; i < count; ++i) {
    const int at = order[i];
    cumulative += finite(values[at]) ? weights[at] : 0.0;
    if (cumulative >= 0.5 * total) return finite(values[at]) ? values[at] : kNaN;
  }
  return kNaN;
}

struct Window {  // Sensor rows visible at the issue, within (issue - 180 min, issue].
  double epoch[kMaxWindowRows], pm[kMaxWindowRows], temperature[kMaxWindowRows], humidity[kMaxWindowRows];
  int count = 0;
};

inline size_t lowerBoundEpoch(const Reading *rows, size_t n, double epoch) {  // First row with epoch > value.
  size_t lo = 0, hi = n;
  while (lo < hi) { const size_t mid = (lo + hi) / 2; if (rows[mid].epoch <= epoch) lo = mid + 1; else hi = mid; }
  return lo;
}

inline void selectWindow(const Reading *rows, size_t n, double issue, Window &w) {
  w.count = 0;
  size_t i = lowerBoundEpoch(rows, n, issue - 10800);
  for (; i < n && rows[i].epoch <= issue && w.count < kMaxWindowRows; ++i) {
    if (!(rows[i].visible <= issue)) continue;
    w.epoch[w.count] = rows[i].epoch;
    w.pm[w.count] = rows[i].pm;
    w.temperature[w.count] = rows[i].temperature;
    w.humidity[w.count] = rows[i].humidity;
    ++w.count;
  }
}

inline double reference(const Window &w, const double *column, double end) {
  double picked[kMaxWindowRows];
  int used = 0;
  double last = -kInf;
  for (int i = 0; i < w.count; ++i) {
    if (w.epoch[i] > end - 300 && w.epoch[i] <= end && finite(column[i])) { picked[used++] = column[i]; last = w.epoch[i]; }
  }
  if (!used || end - last > 240) return kNaN;
  return nanMedian(picked, used);
}

// The 60 three-minute PM bins ending at the issue, oldest first.
inline void pmBins(const Window &w, double issue, double out[kSeqBins]) {
  double firstEnd = issue - 59 * 180.0;
  double values[kMaxWindowRows];
  for (int bin = 0; bin < kSeqBins; ++bin) {
    int used = 0;
    const double end = firstEnd + bin * 180.0;
    for (int i = 0; i < w.count; ++i) {
      const double e = w.epoch[i];
      if (e > issue - 10800 && e <= issue && e <= end && (bin == 0 || e > end - 180.0)) values[used++] = w.pm[i];
    }
    out[bin] = nanMedian(values, used);
  }
}

// Fills the first 37 inputs (sensor, controls, time). Returns true when a fresh reference exists.
inline bool sensorInputs(const Window &w, double issue, double *x) {
  const double fresh = reference(w, w.pm, issue);
  int at = 0;
  x[at++] = fresh;
  static constexpr int lags[5] = {5, 10, 15, 30, 60};
  double lagged[5];
  for (int i = 0; i < 5; ++i) { lagged[i] = reference(w, w.pm, issue - lags[i] * 60); x[at++] = lagged[i]; }
  for (int i = 0; i < 5; ++i) x[at++] = fresh - lagged[i];

  double sd[3] = {kNaN, kNaN, kNaN}, range[2] = {kNaN, kNaN}, slope[2] = {kNaN, kNaN};
  static constexpr int spans[3] = {15, 30, 60};
  for (int s = 0; s < 3; ++s) {
    const double from = issue - spans[s] * 60;
    double sum = 0, epochSum = 0, low = kInf, high = -kInf;
    int n = 0;
    for (int i = 0; i < w.count; ++i)
      if (w.epoch[i] > from && finite(w.pm[i])) { sum += w.pm[i]; epochSum += (w.epoch[i] - issue) / 60; low = std::min(low, w.pm[i]); high = std::max(high, w.pm[i]); ++n; }
    if (n >= 3) {
      const double mean = sum / n;
      double square = 0;
      for (int i = 0; i < w.count; ++i)
        if (w.epoch[i] > from && finite(w.pm[i])) square += (w.pm[i] - mean) * (w.pm[i] - mean);
      sd[s] = std::sqrt(square / n);
    }
    if (s < 2) {
      if (n >= 2) range[s] = high - low;
      if (n >= 3) {
        const double meanX = epochSum / n, meanY = sum / n;
        double numerator = 0, denominator = 0;
        for (int i = 0; i < w.count; ++i)
          if (w.epoch[i] > from && finite(w.pm[i])) {
            const double cx = (w.epoch[i] - issue) / 60 - meanX;
            denominator += cx * cx;
            numerator += cx * (w.pm[i] - meanY);
          }
        if (denominator > 0) slope[s] = numerator / denominator;
      }
    }
  }
  x[at++] = sd[0]; x[at++] = sd[1]; x[at++] = sd[2];
  x[at++] = range[0]; x[at++] = range[1];
  x[at++] = slope[0]; x[at++] = slope[1];

  const double temperature = reference(w, w.temperature, issue);
  const double humidity = reference(w, w.humidity, issue);
  x[at++] = temperature;
  x[at++] = humidity;
  x[at++] = temperature - reference(w, w.temperature, issue - 1800);
  x[at++] = humidity - reference(w, w.humidity, issue - 1800);

  int fresh5 = 0, recent30 = 0;
  double lastFinite = -kInf;
  for (int i = 0; i < w.count; ++i) {
    if (!finite(w.pm[i])) continue;
    if (w.epoch[i] > issue - 300) { ++fresh5; lastFinite = w.epoch[i]; }
    if (w.epoch[i] > issue - 1800) ++recent30;
  }
  x[at++] = fresh5;
  x[at++] = recent30;
  const double daySeconds = std::fmod(issue + 8 * 3600, 86400.0);
  x[at++] = std::sin(2 * kPi * daySeconds / 86400);
  x[at++] = std::cos(2 * kPi * daySeconds / 86400);
  const double offset = std::fmod(issue, 900.0);
  x[at++] = std::sin(2 * kPi * offset / 900);
  x[at++] = std::cos(2 * kPi * offset / 900);

  // TTDI-only mean reversion, from the three-minute bins (ages in minutes).
  double bins[kSeqBins];
  pmBins(w, issue, bins);
  double within90[kSeqBins], old[kSeqBins], recent[kSeqBins];
  int nWithin = 0, nOld = 0, nRecent = 0;
  for (int bin = 0; bin < kSeqBins; ++bin) {
    const int age = 177 - 3 * bin;
    if (age < 90) within90[nWithin++] = bins[bin];
    if (age >= 150) old[nOld++] = bins[bin];
    if (age < 30) recent[nRecent++] = bins[bin];
  }
  const double mean180 = nanMean(bins, kSeqBins);
  x[at++] = fresh - nanMean(within90, nWithin);
  x[at++] = fresh - mean180;
  x[at++] = mean180;
  x[at++] = nanMean(recent, nRecent) - nanMean(old, nOld);

  // Clock inputs: minute resolution, Kuala Lumpur time, Monday = 0.
  const int64_t localSeconds = int64_t(issue) + 8 * 3600;
  const int64_t minuteOfDay = (localSeconds % 86400) / 60;
  const double hour = double(minuteOfDay / 60) + double(minuteOfDay % 60) / 60;
  const int dow = int(((localSeconds / 86400) + 3) % 7);
  x[at++] = hour;
  x[at++] = std::sin(2 * kPi * hour / 24);
  x[at++] = std::cos(2 * kPi * hour / 24);
  x[at++] = dow;
  x[at++] = dow >= 5 ? 1 : 0;
  return finite(lastFinite) && issue - lastFinite <= 240;
}

// A neighbour's 15-minute buckets within the three-hour slice.
struct Buckets {
  double end[40], sum[40], receipt[40];
  int count[40];
  int used = 0;

  void add(double e, double value, double when) {
    for (int i = 0; i < used; ++i)
      if (end[i] == e) { sum[i] += value; count[i] += 1; receipt[i] = std::max(receipt[i], when); return; }
    if (used == 40) return;
    end[used] = e; sum[used] = value; count[used] = 1; receipt[used] = when; ++used;
  }
  double visibleAt(int i) const { return std::max(receipt[i], end[i] + 300); }
  double valueAt(double e, double issue) const {
    if (e <= 0) return kNaN;
    for (int i = 0; i < used; ++i)
      if (end[i] == e) return visibleAt(i) <= issue ? sum[i] / count[i] : kNaN;
    return kNaN;
  }
  double latest(double issue) const {
    double best = -1;
    for (int i = 0; i < used; ++i)
      if (visibleAt(i) <= issue && end[i] > best) best = end[i];
    return best > 0 && issue - best <= 5400 ? best : -1;
  }
};

struct TtdiBuckets {
  double end[16], pmSum[16], rhSum[16], visible[16], epa[16];
  int count[16], rhCount[16];
  int used = 0;

  void add(double e, double pm, double rh, double vis) {
    int i = 0;
    for (; i < used; ++i) if (end[i] == e) break;
    if (i == used) {
      if (used == 16) return;
      end[i] = e; pmSum[i] = 0; rhSum[i] = 0; visible[i] = vis; count[i] = 0; rhCount[i] = 0;
      ++used;
    }
    pmSum[i] += pm;
    if (finite(rh)) { rhSum[i] += rh; ++rhCount[i]; }
    ++count[i];
    visible[i] = std::max(visible[i], vis);
  }
  void finish() {
    for (int i = 0; i < used; ++i)
      epa[i] = count[i] >= 2 ? epaCorrect(pmSum[i] / count[i], rhCount[i] ? rhSum[i] / rhCount[i] : kNaN) : kNaN;
  }
  double at(double e, double issue) const {
    if (e <= 0) return kNaN;
    for (int i = 0; i < used; ++i)
      if (end[i] == e && count[i] >= 2) return visible[i] <= issue ? epa[i] : kNaN;
    return kNaN;
  }
};

inline double ceilBucket(double epoch) { return std::ceil(epoch / 900) * 900; }

// Fills inputs 37..80 (per-station, network and wind-weighted neighbour columns).
inline void spatialInputs(const Reading *rows, size_t rowCount, const Poll *polls, size_t pollCount, double issue,
                          const double *x, const Wind *wind, double *out) {
  static double east[kStations], north[kStations], dist[kStations];
  static bool geometry = false;
  if (!geometry) {
    for (int i = 0; i < kStations; ++i) {
      const double dn = (kStationTable[i].latitude - kTtdiLatitude) * 111.2;
      const double de = (kStationTable[i].longitude - kTtdiLongitude) * 111.2 * std::cos(kTtdiLatitude * kPi / 180);
      const double d = std::hypot(de, dn);
      east[i] = de / d; north[i] = dn / d; dist[i] = d;
    }
    geometry = true;
  }
  const double from = issue - 10800;
  Buckets neighbours[kStations];
  {
    size_t lo = 0, hi = pollCount;
    while (lo < hi) { const size_t mid = (lo + hi) / 2; if (polls[mid].end <= from) lo = mid + 1; else hi = mid; }
    for (size_t i = lo; i < pollCount && polls[i].end <= issue; ++i)
      if (polls[i].station >= 0 && polls[i].station < kStations)
        neighbours[polls[i].station].add(ceilBucket(polls[i].end), polls[i].value, polls[i].receipt);
  }
  TtdiBuckets ttdi;
  for (size_t i = lowerBoundEpoch(rows, rowCount, from); i < rowCount && rows[i].epoch <= issue; ++i)
    if (finite(rows[i].pm)) ttdi.add(ceilBucket(rows[i].epoch), rows[i].pm, rows[i].humidity, rows[i].visible);
  ttdi.finish();

  double diff[kStations], chg30[kStations], chg60[kStations], rel60[kStations], level[kStations], age[kStations];
  int at = 0;
  for (int s = 0; s < kStations; ++s) {
    const Buckets &b = neighbours[s];
    const double latest = b.latest(issue);
    const double now = b.valueAt(latest, issue);
    const double lag30 = b.valueAt(latest > 0 ? latest - 1800 : -1, issue);
    const double lag60 = b.valueAt(latest > 0 ? latest - 3600 : -1, issue);
    const double tNow = ttdi.at(latest, issue);
    const double t60 = ttdi.at(latest > 0 ? latest - 3600 : -1, issue);
    age[s] = latest > 0 ? (issue - latest) / 60 : kNaN;
    level[s] = now;
    diff[s] = now - tNow;
    chg30[s] = now - lag30;
    chg60[s] = now - lag60;
    rel60[s] = (now - lag60) - (tNow - t60);
    out[at++] = diff[s];
    out[at++] = chg60[s];
    out[at++] = age[s];
  }

  double scratch[kStations];
  const auto median = [&](const double *values) { std::copy(values, values + kStations, scratch); return nanMedian(scratch, kStations); };
  int finiteLevels = 0;
  for (int s = 0; s < kStations; ++s) finiteLevels += finite(level[s]);
  out[at++] = finiteLevels;
  const double netLevel = median(level);
  out[at++] = netLevel;
  const double netDiff = median(diff);
  out[at++] = netDiff;
  double absolute[kStations];
  for (int s = 0; s < kStations; ++s) absolute[s] = std::fabs(diff[s] - netDiff);
  out[at++] = median(absolute);
  out[at++] = median(chg30);
  out[at++] = median(chg60);
  double low = kInf, high = -kInf;
  for (int s = 0; s < kStations; ++s) if (finite(chg60[s])) { low = std::min(low, chg60[s]); high = std::max(high, chg60[s]); }
  out[at++] = finite(low) ? low : kNaN;
  out[at++] = finite(high) ? high : kNaN;
  out[at++] = median(rel60);
  out[at++] = netLevel - epaCorrect(x[0], x[19]);  // s_freshPm25, s_freshHumidity

  for (int level925 = 0; level925 < 2; ++level925) {
    double u = kNaN, v = kNaN;
    if (wind) { u = nanMean(level925 ? wind->u925 : wind->u10, 4); v = nanMean(level925 ? wind->v925 : wind->v10, 4); }
    const double speed = finite(u) && finite(v) ? std::hypot(u, v) : kNaN;
    double align[kStations], up[kStations], down[kStations], along[kStations], reachDiff[kStations];
    for (int s = 0; s < kStations; ++s) {
      align[s] = -(east[s] * u + north[s] * v) / (speed > 0 ? speed : kNaN);
      const double a = align[s];
      up[s] = finite(diff[s]) ? (std::isnan(a) ? kNaN : std::max(a, 0.0) * std::max(a, 0.0)) : 0.0;
      down[s] = finite(diff[s]) ? (std::isnan(a) ? kNaN : std::max(-a, 0.0) * std::max(-a, 0.0)) : 0.0;
      along[s] = dist[s] * (std::isnan(a) ? kNaN : std::max(a, 1e-3));
      const double reach = (90 + (finite(age[s]) ? age[s] : 0.0)) / 60 * speed / along[s];
      reachDiff[s] = (std::isnan(reach) ? kNaN : std::min(std::max(reach, 0.0), 1.0)) * diff[s];
    }
    out[at++] = weightedMedian(diff, up, kStations);
    out[at++] = weightedMedian(chg60, up, kStations);
    out[at++] = weightedMedian(chg30, up, kStations);
    out[at++] = weightedMedian(rel60, up, kStations);
    out[at++] = weightedMedian(diff, down, kStations);
    out[at++] = weightedMedian(reachDiff, up, kStations);
    int best = 0;
    double bestValue = -kInf;
    for (int s = 0; s < kStations; ++s) {
      const double candidate = finite(diff[s]) ? align[s] : -9.0;
      if (std::isnan(candidate)) { best = s; break; }
      if (candidate > bestValue) { bestValue = candidate; best = s; }
    }
    const double eta = along[best] / (speed > 0 ? speed : kNaN) * 60 - age[best];
    out[at++] = align[best] > 0.5 ? eta : kNaN;
    out[at++] = align[best];
  }
}

// One input row. Readings and polls must be sorted; returns the fresh-reference flag.
inline bool inputRow(const Reading *rows, size_t rowCount, const Poll *polls, size_t pollCount, double issue,
                     const Wind *wind, double *x, Window &scratch) {
  selectWindow(rows, rowCount, issue, scratch);
  const bool fresh = sensorInputs(scratch, issue, x);
  spatialInputs(rows, rowCount, polls, pollCount, issue, x, wind, x + kSensorInputs + kControls + kTimeInputs);
  return fresh;
}

// ------------------------------------------------------------------- targets
// Closed 15-minute medians with the production coverage rule; NaN when not eligible.
// `epochs` and `pm` hold rows sorted by epoch with finite, non-negative PM only.
inline double bucketMedian(const double *epochs, const double *pm, size_t n, double end, size_t &cursor, double *scratch) {
  const double start = end - 900;
  while (cursor < n && epochs[cursor] <= start) ++cursor;
  size_t i = cursor;
  int used = 0;
  double previous = kNaN, first = kNaN, last = kNaN, largestGap = 0;
  for (; i < n && epochs[i] <= end; ++i) {
    if (used < 64) scratch[used] = pm[i];
    if (!finite(first)) first = epochs[i];
    else largestGap = std::max(largestGap, epochs[i] - previous);
    previous = epochs[i];
    last = epochs[i];
    ++used;
  }
  if (used < 3 || used > 64) return kNaN;
  if (first - start > 240 || end - last > 240 || largestGap > 480) return kNaN;
  return nanMedian(scratch, used);
}

struct Targets {
  double arrival, rideMean, rideMin, rideMax;
};

// Targets of a 15-minute grid origin from medians[k] = bucket ending origin0 + k * 900.
inline Targets gridTargets(const double *medians, size_t count, size_t originIndex) {
  Targets out{kNaN, kNaN, kNaN, kNaN};
  if (originIndex + 6 < count) out.arrival = medians[originIndex + 6];
  if (originIndex + 14 >= count || !finite(out.arrival)) return out;  // The server blanks ride targets too.
  double sum = 0, low = kInf, high = -kInf;
  for (int k = 7; k <= 14; ++k) {
    const double value = medians[originIndex + k];
    if (!finite(value)) return out;
    sum += value * 0.125;
    low = std::min(low, value);
    high = std::max(high, value);
  }
  out.rideMean = sum;
  out.rideMin = low;
  out.rideMax = high;
  return out;
}

}  // namespace pm
