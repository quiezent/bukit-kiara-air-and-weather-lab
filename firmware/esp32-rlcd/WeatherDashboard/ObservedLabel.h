#pragma once
// The line under the outdoor PM2.5 number ("Observed: Fast PM2.5 rise detected").
//
// A port of the laptop server's label chain: passing_shower_signal,
// fast_dry_clearing_signal and dry_clearing_feature_frame (the washout signal),
// air_window_analysis (state and label), separate_observation_from_prediction
// (eventLabel) and device_display.current_display_text, all in
// BukitKiara_Dashboard.py and device_display.py.
//
// Portable C++ with no Arduino dependency, so the host test can compare it with
// the server's own Python clock by clock. Arithmetic is 64-bit double on purpose:
// every branch is a threshold on differences and ratios of decimal readings, and
// the same rounding as Python is what keeps a tie on the same side.
//
// Inputs: pm02, pm10, atmp, rhum, heatindex, pm003Count, tvoc, rco2 per reading.
// A missing value stays NaN and is never filled. Rain, Open-Meteo and Subang are
// used by the server only for the washout "kind" text, which never reaches this
// label, so they are not inputs here.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace obs {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

struct Sample {  // One TTDI reading, sorted by epoch, unique, visible at the issue.
  double epoch, pm02, pm10, count, temperature, humidity, heatIndex, co2, tvoc;
};

enum Column { kPm, kPm10, kCount, kTemperature, kHumidity, kHeat, kCo2, kTvoc, kColumns };
constexpr double Sample::*const kField[kColumns] = {&Sample::pm02,        &Sample::pm10,     &Sample::count,
                                                    &Sample::temperature, &Sample::humidity, &Sample::heatIndex,
                                                    &Sample::co2,         &Sample::tvoc};

struct Bucket3 {  // Right-labelled 3-minute medians and their rolling 3-bucket median.
  double raw[kColumns], smooth[kColumns], last;
};
struct Bucket15 {  // Right-labelled 15-minute medians and their restart-after-gap EMAs.
  double pm, pm10, last, fast, slow, coarse, fine;
};

// Caller-owned work space (PSRAM on the device). A window longer than these holds
// is cut from its oldest end.
struct Scratch {
  Bucket3 *three;
  int threeCapacity;
  Bucket15 *fifteen;
  int fifteenCapacity;
};

// The window the server reads (RIDE_API_ANALYSIS_DAYS) and the one the device needs.
// 24 h is equivalent because the longest forming cluster in all stored history
// spans 2.65 h and the slower EMA forgets its start at (7/9)^96 = 3e-11 per day.
constexpr double kServerLookback = 28 * 86400.0;
constexpr double kDeviceLookback = 24 * 3600.0;
constexpr int kDeviceThreeBuckets = 24 * 20 + 2, kDeviceFifteenBuckets = 24 * 4 + 2;

// ArduinoJson 7 stores a short number as float32, so "16.8" reaches a double as
// 16.799999237. The sensor reports at most two decimals: restore the exact value
// Python parsed before storing a reading.
inline double exactReading(double parsed) { return std::round(parsed * 100.0) / 100.0; }

enum class State : uint8_t {
  Collecting, Stale, FastRise, Rebound, Confirmed, Forming, RapidImprovement, Rising, Improving, StableHigh, Steady
};
enum class Washout : uint8_t { None, Forming, Active, Recent };

inline const char *stateName(State s) {
  switch (s) {
    case State::Collecting: return "collecting";
    case State::Stale: return "stale";
    case State::FastRise: return "fast_rise";
    case State::Rebound: return "rebound";
    case State::Confirmed: return "confirmed";
    case State::Forming: return "forming";
    case State::RapidImprovement: return "rapid_improvement";
    case State::Rising: return "rising";
    case State::Improving: return "improving";
    case State::StableHigh: return "stable_high";
    default: return "steady";
  }
}
inline const char *washoutName(Washout s) {
  switch (s) {
    case Washout::Forming: return "forming";
    case Washout::Active: return "active";
    case Washout::Recent: return "recent";
    default: return "none";
  }
}

constexpr const char *kDefaultText = "Latest sensor reading";

struct Result {
  bool available = false;
  State state = State::Collecting;
  Washout washout = Washout::None;
  const char *label = "Collecting particle history…";
  const char *eventLabel = nullptr;     // Set only for the five displayed states.
  const char *displayText = kDefaultText;  // The line the screen shows.
  double detectedEpoch = kNaN, throughEpoch = kNaN, lastBucket = kNaN;
  double current = kNaN, change30 = kNaN, gap = kNaN, minimumSinceEvent = kNaN, pm10Fraction30 = kNaN;
  double fineShare = kNaN, coarse = kNaN;
};

namespace detail {

constexpr int kStep3 = 180, kStep15 = 900, kMaxPerBucket = 64;

inline bool finite(double v) { return std::isfinite(v); }
inline bool missing(double v) { return v != v; }
inline double pct(double now, double before) { return now / before - 1.0; }  // pandas pct_change, IEEE 0 division.

inline int64_t ceilLabel(double epoch, int step) {
  const int64_t e = int64_t(std::ceil(epoch));
  return ((e + step - 1) / step) * step;
}

// pandas median: NaN skipped, the mean of the two middle values when even.
inline double medianOf(double *values, int count) {
  if (!count) return kNaN;
  for (int i = 1; i < count; ++i) {
    const double v = values[i];
    int at = i;
    while (at > 0 && values[at - 1] > v) { values[at] = values[at - 1]; --at; }
    values[at] = v;
  }
  return (count & 1) ? values[count / 2] : (values[count / 2] + values[count / 2 - 1]) / 2;
}

inline double bucketMedian(const Sample *rows, size_t lo, size_t hi, double Sample::*field) {
  double values[kMaxPerBucket];
  int used = 0;
  // Newest rows win in the impossible case of more than 64 readings in a bucket.
  if (hi - lo > size_t(kMaxPerBucket)) lo = hi - kMaxPerBucket;
  for (size_t i = lo; i < hi; ++i)
    if (finite(rows[i].*field)) values[used++] = rows[i].*field;
  return medianOf(values, used);
}

inline double middleOfThree(double a, double b, double c) {
  if (missing(a) || missing(b) || missing(c)) return kNaN;
  return std::max(std::min(a, b), std::min(std::max(a, b), c));
}

// One line of segmented_ewm: pandas ewm(span, adjust=False), restarted by a gap.
struct Ewm {
  double oldWeight, newWeight, value;
  bool have;
  explicit Ewm(double span) : value(kNaN), have(false) {
    const double com = (span - 1) / 2, alpha = 1.0 / (1.0 + com);
    oldWeight = 1.0 - alpha;
    newWeight = com == 1 ? 1.0 - oldWeight : alpha;
  }
  double update(double x) {
    if (missing(x)) { have = false; return kNaN; }
    if (!have) { value = x; have = true; return value; }
    if (value != x) value = (oldWeight * value + newWeight * x) / (oldWeight + newWeight);
    return value;
  }
};

inline int fillThree(const Sample *rows, size_t lo, size_t hi, Bucket3 *out, int capacity, int64_t &first) {
  const int64_t end = ceilLabel(rows[hi - 1].epoch, kStep3);
  first = ceilLabel(rows[lo].epoch, kStep3);
  if ((end - first) / kStep3 + 1 > capacity) first = end - int64_t(capacity - 1) * kStep3;
  const int count = int((end - first) / kStep3) + 1;
  for (int i = 0; i < count; ++i) {
    for (int c = 0; c < kColumns; ++c) out[i].raw[c] = out[i].smooth[c] = kNaN;
    out[i].last = kNaN;
  }
  for (size_t i = lo; i < hi;) {
    const int64_t label = ceilLabel(rows[i].epoch, kStep3);
    size_t j = i + 1;
    while (j < hi && ceilLabel(rows[j].epoch, kStep3) == label) ++j;
    if (label >= first) {
      Bucket3 &b = out[(label - first) / kStep3];
      for (int c = 0; c < kColumns; ++c) b.raw[c] = bucketMedian(rows, i, j, kField[c]);
      b.last = rows[j - 1].epoch;
    }
    i = j;
  }
  for (int i = 2; i < count; ++i)
    for (int c = 0; c < kColumns; ++c)
      out[i].smooth[c] = middleOfThree(out[i - 2].raw[c], out[i - 1].raw[c], out[i].raw[c]);
  return count;
}

inline int fillFifteen(const Sample *rows, size_t lo, size_t hi, Bucket15 *out, int capacity, int64_t &first) {
  const int64_t end = ceilLabel(rows[hi - 1].epoch, kStep15);
  first = ceilLabel(rows[lo].epoch, kStep15);
  if ((end - first) / kStep15 + 1 > capacity) first = end - int64_t(capacity - 1) * kStep15;
  const int count = int((end - first) / kStep15) + 1;
  for (int i = 0; i < count; ++i) out[i] = Bucket15{kNaN, kNaN, kNaN, kNaN, kNaN, kNaN, kNaN};
  for (size_t i = lo; i < hi;) {
    const int64_t label = ceilLabel(rows[i].epoch, kStep15);
    size_t j = i + 1;
    while (j < hi && ceilLabel(rows[j].epoch, kStep15) == label) ++j;
    if (label >= first) {
      Bucket15 &b = out[(label - first) / kStep15];
      b.pm = bucketMedian(rows, i, j, &Sample::pm02);
      b.pm10 = bucketMedian(rows, i, j, &Sample::pm10);
      b.last = rows[j - 1].epoch;
    }
    i = j;
  }
  return count;
}

struct FormingSignal {
  bool present = false, moist = false;
  double detected = kNaN;
};

// dry_clearing_feature_frame and fast_dry_clearing_signal.
inline FormingSignal dryClearing(const Bucket3 *b, int n, int64_t first, double latest) {
  FormingSignal out;
  const auto S = [&](int column, int k) { return k >= 0 ? b[k].smooth[column] : kNaN; };
  const auto R = [&](int column, int k) { return k >= 0 ? b[k].raw[column] : kNaN; };
  const auto smoothFell = [&](int k) { return k >= 1 && b[k].smooth[kPm] - b[k - 1].smooth[kPm] < 0; };
  const auto rawFell = [&](int column, int k) { return k >= 1 && b[k].raw[column] - b[k - 1].raw[column] < 0; };
  const auto countFalls = [&](auto fell, int k, int window) {
    if (k - window + 1 < 0) return -1;
    int falls = 0;
    for (int at = k - window + 1; at <= k; ++at) falls += fell(at) ? 1 : 0;
    return falls;
  };
  int runBase = 0, runCount = 0, runGas = 0, runMoist = 0, runSustained = 0;
  int clusterFirst = -1, clusterLast = -1;
  bool clusterMoist = false;
  int64_t previousCandidate = 0;
  for (int i = 0; i < n; ++i) {
    const double *raw = b[i].raw;
    const bool base = !missing(raw[kPm]) && !missing(raw[kPm10]) && !missing(raw[kHumidity]);
    const bool hasCount = !missing(raw[kCount]);
    runBase = base ? runBase + 1 : 0;
    runCount = hasCount ? runCount + 1 : 0;
    runGas = (!missing(raw[kTvoc]) && !missing(raw[kCo2])) ? runGas + 1 : 0;
    runMoist = (base && !missing(raw[kTemperature]) && !missing(raw[kHeat])) ? runMoist + 1 : 0;
    runSustained = (base && hasCount) ? runSustained + 1 : 0;
    if (runBase < 4) continue;  // Every branch below needs at least four complete buckets.

    // earlyJointShock: raw 3-minute values over nine minutes.
    bool shock = false;
    {
      const double pm9 = raw[kPm] - R(kPm, i - 3);
      if (pm9 <= -10.0 && pct(raw[kPm], R(kPm, i - 3)) <= -0.12 && pct(raw[kPm10], R(kPm10, i - 3)) <= -0.10 &&
          countFalls([&](int k) { return rawFell(kPm, k); }, i, 3) == 3 &&
          countFalls([&](int k) { return rawFell(kPm10, k); }, i, 3) >= 2) {
        const bool particle = runCount >= 4 && pct(raw[kCount], R(kCount, i - 3)) <= -0.12 &&
                              raw[kHumidity] - R(kHumidity, i - 3) <= -1.0;
        const bool gas = runGas >= 4 && pct(raw[kTvoc], R(kTvoc, i - 3)) <= -0.30 &&
                         pct(raw[kCo2], R(kCo2, i - 3)) <= -0.01;
        shock = particle || gas;
      }
    }
    // rapidForming: smoothed values over fifteen minutes.
    bool rapid = false;
    if (runBase >= 8) {
      const double sPm = S(kPm, i), pm15 = sPm - S(kPm, i - 5);
      if (pm15 <= -10.0 && pct(sPm, S(kPm, i - 5)) <= -0.12 && pct(S(kPm10, i), S(kPm10, i - 5)) <= -0.10 &&
          countFalls(smoothFell, i, 3) >= 2) {
        const bool particle = runCount >= 8 && pct(S(kCount, i), S(kCount, i - 5)) <= -0.15 &&
                              S(kHumidity, i) - S(kHumidity, i - 5) <= -1.0;
        const bool gas = runGas >= 8 && pct(S(kTvoc, i), S(kTvoc, i - 5)) <= -0.30 &&
                         pct(S(kCo2, i), S(kCo2, i - 5)) <= -0.01;
        rapid = particle || gas;
      }
    }
    // rapidMoistCoolingSupport: daytime cooling with a joint 30-minute fall.
    bool moist = false;
    if (runMoist >= 13) {
      const int64_t label = first + int64_t(i) * kStep3;
      const int hour = int(((label + 28800) % 86400) / 3600);  // Malaysia has no daylight saving.
      if (hour >= 6 && hour < 19 && S(kTemperature, i) - S(kTemperature, i - 10) <= -1.5 &&
          S(kHumidity, i) - S(kHumidity, i - 10) >= 2.0 && pct(S(kPm, i), S(kPm, i - 10)) <= -0.20 &&
          pct(S(kPm10, i), S(kPm10, i - 10)) <= -0.20) {
        int observed = 0;
        for (int k = i - 12; k <= i; ++k) observed += !missing(b[k].raw[kCount]) ? 1 : 0;
        moist = observed == 0 || (observed == 13 && pct(S(kCount, i), S(kCount, i - 10)) <= -0.20);
      }
    }
    // sustainedForming: a complete joint 60-minute decline with no rebound from the low.
    bool sustained = false;
    if (runSustained >= 23) {
      const double sPm = S(kPm, i);
      const double low = (i >= 4) ? std::min({S(kPm, i - 4), S(kPm, i - 3), S(kPm, i - 2), S(kPm, i - 1), sPm}) : kNaN;
      const bool lowKnown = i >= 4 && finite(S(kPm, i - 4)) && finite(S(kPm, i - 3)) && finite(S(kPm, i - 2)) &&
                            finite(S(kPm, i - 1)) && finite(sPm);
      sustained = sPm - S(kPm, i - 15) <= -10.0 && pct(sPm, S(kPm, i - 15)) <= -0.08 &&
                  sPm - S(kPm, i - 20) <= -10.0 && pct(sPm, S(kPm, i - 20)) <= -0.08 &&
                  pct(S(kPm10, i), S(kPm10, i - 15)) <= -0.06 && pct(S(kCount, i), S(kCount, i - 15)) <= -0.06 &&
                  S(kHumidity, i) - S(kHumidity, i - 15) <= -1.0 && countFalls(smoothFell, i, 15) >= 9 &&
                  countFalls(smoothFell, i, 5) >= 3 && lowKnown && sPm - low <= 3.0;
    }
    if (!(shock || rapid || moist || sustained)) continue;
    const int64_t label = first + int64_t(i) * kStep3;
    if (clusterFirst < 0 || label - previousCandidate > 3600) { clusterFirst = i; clusterMoist = false; }
    clusterLast = i;
    clusterMoist = clusterMoist || moist;
    previousCandidate = label;
  }
  if (clusterFirst < 0) return out;
  const double firstLabel = double(first + int64_t(clusterFirst) * kStep3);
  const double lastLabel = double(first + int64_t(clusterLast) * kStep3);
  const double detected = std::min(latest, finite(b[clusterFirst].last) ? b[clusterFirst].last : firstLabel);
  const double evidence = std::min(latest, finite(b[clusterLast].last) ? b[clusterLast].last : lastLabel);
  if (latest - evidence > 36 * 60) return out;
  // Stop once particles have materially rebounded from the event's low.
  double minimum = kNaN, newest = kNaN;
  for (int i = clusterFirst; i < n; ++i) {
    const double v = b[i].smooth[kPm];
    if (missing(v)) continue;
    minimum = missing(minimum) ? v : std::min(minimum, v);
    newest = v;
  }
  if (finite(minimum) && newest >= minimum + std::max(5.0, minimum * 0.08)) return out;
  out.present = true;
  out.moist = clusterMoist;
  out.detected = detected;
  return out;
}

struct WashoutSignal {
  Washout state = Washout::None;
  double detected = kNaN;
  bool moist = false;
};

inline bool jointWashout(const Bucket15 *b, int k) {
  if (k < 2) return false;
  return b[k].pm - b[k - 2].pm <= -20.0 && pct(b[k].pm, b[k - 2].pm) <= -0.25 && pct(b[k].pm10, b[k - 2].pm10) <= -0.20;
}

// The first reading inside the confirmation bucket that made it qualify.
inline double firstQualifyingEpoch(const Sample *rows, size_t lo, size_t hi, double bucketEnd, const Bucket15 &before, double fallback) {
  double pm[kMaxPerBucket], pm10[kMaxPerBucket];
  int usedPm = 0, usedPm10 = 0;
  for (size_t i = lo; i < hi && rows[i].epoch <= bucketEnd; ++i) {
    if (rows[i].epoch <= bucketEnd - kStep15) continue;
    if (usedPm < kMaxPerBucket && finite(rows[i].pm02)) pm[usedPm++] = rows[i].pm02;
    if (usedPm10 < kMaxPerBucket && finite(rows[i].pm10)) pm10[usedPm10++] = rows[i].pm10;
    double a[kMaxPerBucket], c[kMaxPerBucket];
    std::copy(pm, pm + usedPm, a);
    std::copy(pm10, pm10 + usedPm10, c);
    const double medianPm = medianOf(a, usedPm), medianPm10 = medianOf(c, usedPm10);
    if (medianPm - before.pm <= -20.0 && pct(medianPm, before.pm) <= -0.25 && pct(medianPm10, before.pm10) <= -0.20)
      return rows[i].epoch;
  }
  return fallback;
}

// passing_shower_signal, restricted to what reaches the label.
inline WashoutSignal washoutSignal(const Sample *rows, size_t lo, size_t hi, const Bucket15 *b, int n, int64_t first,
                                   int complete, const FormingSignal &forming, double latest) {
  WashoutSignal out;
  const auto useForming = [&] {
    if (!forming.present) return;
    out.state = Washout::Forming;
    out.detected = forming.detected;
    out.moist = forming.moist;
  };
  if (complete < 3) { useForming(); return out; }
  int clusterFirst = -1, clusterLast = -1;
  int64_t previous = 0;
  for (int k = 0; k < n; ++k) {
    const int64_t label = first + int64_t(k) * kStep15;
    if (double(label) < latest - 3 * 3600 || !jointWashout(b, k)) continue;
    if (clusterFirst < 0 || label - previous > 3600) clusterFirst = k;
    clusterLast = k;
    previous = label;
  }
  if (clusterFirst < 0) { useForming(); return out; }
  const double lastLabel = double(first + int64_t(clusterLast) * kStep15);
  const double confirmedLast = std::min(latest, finite(b[clusterLast].last) ? b[clusterLast].last : lastLabel);
  if (forming.present && forming.detected > confirmedLast + 3600) { useForming(); return out; }
  const double confirmation = double(first + int64_t(clusterFirst) * kStep15);
  const double fallback = finite(b[clusterFirst].last) ? b[clusterFirst].last : confirmation;
  const double raw = firstQualifyingEpoch(rows, lo, hi, confirmation, b[clusterFirst - 2], fallback);
  out.detected = std::min(latest, raw);
  // Python's round(seconds / 60) <= 30, with ties to even: 1830 s rounds to 30.
  out.state = latest - out.detected <= 1830 ? Washout::Active : Washout::Recent;
  return out;
}

inline bool forecastEligible(const Sample *rows, size_t lo, size_t hi, double end) {
  int count = 0;
  double first = 0, last = 0, largest = 0;
  for (size_t i = lo; i < hi && rows[i].epoch <= end; ++i) {
    if (rows[i].epoch <= end - kStep15 || !finite(rows[i].pm02)) continue;
    if (count) largest = std::max(largest, rows[i].epoch - last); else first = rows[i].epoch;
    last = rows[i].epoch;
    ++count;
  }
  return count >= 3 && first - (end - kStep15) <= 240 && end - last <= 240 && largest <= 480;
}

inline size_t firstAtOrAfter(const Sample *rows, size_t count, double epoch) {
  size_t lo = 0, hi = count;
  while (lo < hi) { const size_t mid = (lo + hi) / 2; if (rows[mid].epoch < epoch) lo = mid + 1; else hi = mid; }
  return lo;
}

}  // namespace detail

// The label at `issue` from the readings in rows. `lookback` limits how far back
// is read; `now` is the clock the display age is judged against (default: issue).
inline Result observe(const Sample *rows, size_t count, double issue, double lookback, const Scratch &work, double now = kNaN) {
  using namespace detail;
  Result r;
  if (missing(now)) now = issue;
  const size_t lo = firstAtOrAfter(rows, count, std::floor(issue - lookback));
  const size_t hi = firstAtOrAfter(rows, count, issue + 0.5);
  if (lo >= hi) return r;
  const double latest = rows[hi - 1].epoch;

  FormingSignal forming;
  if (hi - lo >= 4) {
    int64_t first3 = 0;
    const int n3 = fillThree(rows, lo, hi, work.three, work.threeCapacity, first3);
    forming = dryClearing(work.three, n3, first3, latest);
  }

  Bucket15 *const b = work.fifteen;
  int64_t first15 = 0;
  const int n15 = fillFifteen(rows, lo, hi, b, work.fifteenCapacity, first15);
  int complete = 0, end = -1;
  for (int k = 0; k < n15; ++k)
    if (!missing(b[k].pm) && !missing(b[k].pm10)) { ++complete; end = k; }

  const WashoutSignal washout = washoutSignal(rows, lo, hi, b, n15, first15, complete, forming, latest);
  r.washout = washout.state;
  r.detectedEpoch = washout.detected;

  if (complete < 5) return r;  // "Collecting particle history…"

  const double anchor = std::floor(issue / kStep15) * kStep15;
  const double age = issue - latest;
  const bool freshReading = age >= 0 && age <= 420;  // SENSOR_DEGRADED_SECONDS
  const double lastLabel = double(first15 + int64_t(end) * kStep15);
  r.lastBucket = lastLabel;
  const int anchorIndex = int((int64_t(anchor) - first15) / kStep15);
  const bool anchorClosed = anchor <= lastLabel && anchor >= double(first15) && !missing(b[anchorIndex].pm) &&
                            forecastEligible(rows, lo, hi, anchor);
  if (!anchorClosed || !freshReading) {
    r.state = freshReading ? State::Collecting : State::Stale;
    r.label = freshReading ? "Collecting a complete current forecast bucket…" : "Sensor reading is stale";
    return r;
  }

  Ewm fast(3), slow(8), coarse(3), fine(3);
  for (int k = 0; k <= end; ++k) {
    const double pm = b[k].pm, pm10 = b[k].pm10;
    b[k].fast = fast.update(pm);
    b[k].slow = slow.update(pm);
    const double coarseNow = pm10 - pm;  // NaN when either side is missing.
    b[k].coarse = coarse.update(missing(coarseNow) ? kNaN : std::max(coarseNow, 0.0));
    double share = 100.0 * pm / pm10;
    if (!missing(share)) share = std::min(std::max(share, 0.0), 100.0);
    b[k].fine = fine.update(share);
  }
  const auto change = [&](int k, int steps) { return k >= steps ? b[k].pm - b[k - steps].pm : kNaN; };
  const auto pm10Fraction = [&](int k, int steps) { return k >= steps ? pct(b[k].pm10, b[k - steps].pm10) : kNaN; };
  const auto mixRebound = [&](int k) {
    if (k < 2) return false;
    return b[k].fine - b[k - 2].fine >= 3.0 && b[k].coarse - b[k - 2].coarse <= -1.5 && b[k].fine >= 85.0;
  };
  const auto fastRiseFlag = [&](int k) {
    if (k < 0) return false;
    const double c30 = change(k, 2), f30 = pm10Fraction(k, 2);
    return (c30 >= 15.0 && f30 >= 0.15 && b[k].fast > b[k].slow) ||
           (c30 >= 10.0 && f30 >= 0.08 && b[k].fast > b[k].slow && mixRebound(k));
  };

  const double current = b[end].pm;
  const double change30 = change(end, 2), change60 = change(end, 4);
  const double f30 = pm10Fraction(end, 2), f60 = pm10Fraction(end, 4);
  const double gap = b[end].fast - b[end].slow;
  const bool momentum = finite(change30) && !missing(f30);
  const bool fastRise = momentum && (fastRiseFlag(end) ||
                                     ((fastRiseFlag(end) || fastRiseFlag(end - 1) || fastRiseFlag(end - 2)) &&
                                      change30 >= 5.0 && gap > 0));
  const bool eventRecent = washout.state != Washout::None;
  double minimumSince = current;
  if (eventRecent && washout.detected) {
    const double from = washout.detected - kStep15;
    bool any = false;
    double low = kNaN;
    for (int k = end; k >= 0 && double(first15 + int64_t(k) * kStep15) >= from; --k) {
      any = true;
      if (!missing(b[k].pm)) low = missing(low) ? b[k].pm : std::min(low, b[k].pm);
    }
    if (any) minimumSince = low;
  }
  const bool rebound = eventRecent && (change30 >= 5.0 || (current - minimumSince >= 10.0 && current >= minimumSince * 1.20));
  const bool holdingLower = end >= 1 && b[end].pm <= 35.0 && b[end - 1].pm <= 35.0;
  const bool rapidImprovement = eventRecent || (change30 <= -20.0 && gap <= -8.0);
  const bool complete60 = end >= 4 && [&] {
    for (int k = end - 4; k <= end; ++k) if (missing(b[k].pm) || missing(b[k].pm10)) return false;
    return true;
  }();
  const bool sustainedImprovement = complete60 && finite(change60) && change60 <= -8.0 && !missing(f60) && f60 <= -0.06 && !missing(gap) && gap <= -2.0;

  r.available = true;
  r.current = current;
  r.change30 = change30;
  r.gap = gap;
  r.pm10Fraction30 = f30;
  r.fineShare = b[end].fine;
  r.coarse = b[end].coarse;
  if (eventRecent) r.minimumSinceEvent = minimumSince;
  r.throughEpoch = std::min(latest, lastLabel);

  static constexpr const char *kFastRebound = "Fast PM2.5 rebound detected", *kFastRise = "Fast PM2.5 rise detected",
                              *kRebound = "Particle rebound may be starting", *kSustained = "Sustained PM2.5 reduction detected",
                              *kDry = "Dry clearing forming", *kMoist = "Moist-cooling particle clearing forming",
                              *kReduction = "PM2.5 reduction forming", *kRapid = "Rapid PM2.5 reduction detected",
                              *kBelow35 = "Recent PM2.5 medians at or below 35 µg/m³";
  if (!momentum) { r.state = State::Collecting; r.label = "Recent particle movement unavailable"; }
  else if (fastRise) { r.state = State::FastRise; r.label = rebound ? kFastRebound : kFastRise; }
  else if (rebound) { r.state = State::Rebound; r.label = kRebound; }
  else if (holdingLower) { r.state = State::Confirmed; r.label = kSustained; }
  else if (washout.state == Washout::Forming) { r.state = State::Forming; r.label = washout.moist ? kMoist : kDry; }
  else if (rapidImprovement && current <= 55.0) { r.state = State::Forming; r.label = kReduction; }
  else if (rapidImprovement) { r.state = State::RapidImprovement; r.label = kRapid; }
  else if (change30 >= 8.0) { r.state = State::Rising; r.label = "PM2.5 rising"; }
  else if (change30 <= -8.0 || sustainedImprovement) { r.state = State::Improving; r.label = "PM2.5 falling"; }
  else if (current > 55.0) { r.state = State::StableHigh; r.label = "PM2.5 holding near its recent level"; }
  else { r.state = State::Steady; r.label = "No strong particle shift"; }

  if (r.state == State::FastRise || r.state == State::Rebound || r.state == State::Confirmed ||
      r.state == State::Forming || r.state == State::RapidImprovement)
    r.eventLabel = r.state == State::Confirmed ? kBelow35 : r.label;

  // device_display.current_display_text, with the reading and the analysis from one clock.
  if (r.eventLabel) {
    const double sensorAge = now - latest;
    if (!(r.throughEpoch > latest || latest - r.throughEpoch > 600 || !(sensorAge >= 0 && sensorAge <= 720))) {
      static constexpr const char *kTexts[][2] = {
          {kFastRebound, "Observed: Fast PM2.5 rebound detected"},
          {kFastRise, "Observed: Fast PM2.5 rise detected"},
          {kRebound, "Observed: Particle rebound may be starting"},
          {kBelow35, "Observed: Recent PM2.5 medians at or below 35 µg/m³"},
          {kDry, "Observed: Dry clearing forming"},
          {kMoist, "Observed: Moist-cooling particle clearing forming"},
          {kReduction, "Observed: PM2.5 reduction forming"},
          {kRapid, "Observed: Rapid PM2.5 reduction detected"},
      };
      for (const auto &entry : kTexts)
        if (entry[0] == r.eventLabel) r.displayText = entry[1];
    }
  }
  return r;
}

}  // namespace obs
