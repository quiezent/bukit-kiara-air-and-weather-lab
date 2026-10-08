#include "HistoryGraphs.h"
#include "DashboardTypography.h"
#include <time.h>

namespace {
using DashboardTypography::text;
using DashboardTypography::width;
constexpr uint32_t VALID_EPOCH = 1700000000;
constexpr int PLOT_TOP = 70;
constexpr int PLOT_BOTTOM = 170;
constexpr int AXIS_BASELINE = 186;

struct Plot {
  int left;
  int right;
  int labelLeft;
};

struct Limits {
  float low = INFINITY;
  float high = -INFINITY;
  bool valid = false;
};

struct Latest {
  uint32_t epoch = 0;
  float value = NAN;
};

enum class Field { Pm25, Temperature, HeatIndex };

float pointValue(const HistoryPoint &point, Field field) {
  float value = field == Field::Pm25 ? point.pm25
      : field == Field::Temperature ? point.temperatureC : point.heatIndexC;
  return isfinite(value) && (field != Field::Pm25 || value >= 0) ? value : NAN;
}

uint16_t pointCount(const HistorySeries &history) {
  return history.points ? min(history.count, HISTORY_GRAPH_POINT_LIMIT) : 0;
}

bool inWindow(const HistorySeries &history, uint32_t epoch) {
  return epoch >= VALID_EPOCH && epoch >= history.startEpoch && epoch <= history.endEpoch;
}

bool crossesGap(const HistorySeries &history, uint32_t leftEpoch, uint32_t rightEpoch) {
  uint16_t count = history.gaps ? min(history.gapCount, HISTORY_GRAPH_GAP_LIMIT) : 0;
  for (uint16_t i = 0; i < count; ++i) {
    const HistoryGap &gap = history.gaps[i];
    if (gap.rightEpoch <= gap.leftEpoch) continue;
    // Open intervals overlap: a point exactly on either edge remains valid,
    // but a segment from one edge to the other cannot fill the missing period.
    if (leftEpoch < gap.rightEpoch && rightEpoch > gap.leftEpoch) return true;
  }
  return false;
}

void include(Limits &limits, float value) {
  if (!isfinite(value)) return;
  limits.low = min(limits.low, value);
  limits.high = max(limits.high, value);
  limits.valid = true;
}

float niceStep(float raw) {
  if (!isfinite(raw) || raw <= 0) return 1;
  float magnitude = powf(10, floorf(log10f(raw)));
  float unit = raw / magnitude;
  float rounded = unit <= 1 ? 1 : unit <= 2 ? 2 : unit <= 2.5f ? 2.5f : unit <= 5 ? 5 : 10;
  return rounded * magnitude;
}

Limits plotLimits(const HistorySeries &history, bool temperature) {
  Limits limits;
  for (uint16_t i = 0; i < pointCount(history); ++i) {
    const HistoryPoint &point = history.points[i];
    if (!inWindow(history, point.epoch)) continue;
    include(limits, pointValue(point, temperature ? Field::Temperature : Field::Pm25));
    if (temperature) include(limits, pointValue(point, Field::HeatIndex));
  }
  if (!limits.valid) return limits;
  float span = limits.high - limits.low;
  // Constant and single-point series need a real, nonzero plotting range.
  float padding = max(temperature ? 0.5f : 1.0f, span * 0.08f);
  float low = limits.low - padding;
  float high = limits.high + padding;
  if (!temperature) low = max(0.0f, low);
  float step = niceStep((high - low) / 4.0f);
  limits.low = floorf(low / step) * step;
  limits.high = ceilf(high / step) * step;
  if (limits.high <= limits.low) limits.high = limits.low + step;
  return limits;
}

Latest latestPoint(const HistorySeries &history, Field field) {
  Latest latest;
  for (uint16_t i = 0; i < pointCount(history); ++i) {
    const HistoryPoint &point = history.points[i];
    float value = pointValue(point, field);
    if (inWindow(history, point.epoch) && isfinite(value) && point.epoch >= latest.epoch) {
      latest.epoch = point.epoch;
      latest.value = value;
    }
  }
  return latest;
}

bool old(const HistorySeries &history, const Latest &latest) {
  if (!latest.epoch) return false;
  return !history.fresh || !history.available
      || (history.nowEpoch >= latest.epoch && history.nowEpoch - latest.epoch > history.freshAfterSeconds);
}

String numeric(float value, uint8_t decimals = 1) {
  return isfinite(value) ? String(value, static_cast<unsigned int>(decimals)) : "--";
}

String clockText(uint32_t epoch, const char *format = "%H:%M") {
  if (epoch < VALID_EPOCH) return "--:--";
  // Convert UTC explicitly: the graph does not depend on the caller's TZ setup.
  time_t malaysia = static_cast<time_t>(epoch) + 8 * 3600;
  struct tm local;
  gmtime_r(&malaysia, &local);
  char buffer[24];
  strftime(buffer, sizeof(buffer), format, &local);
  return String(buffer);
}

bool summaryAvailable(const HistoryPmSummary &summary) {
  return summary.available && summary.sampleCount > 0
      && isfinite(summary.average) && isfinite(summary.lowest) && isfinite(summary.highest)
      && summary.lowest >= 0 && summary.average >= summary.lowest && summary.highest >= summary.average;
}

void summaryValue(U8G2 &gfx, int center, const char *label, float value) {
  gfx.setFont(u8g2_font_helvB08_tf);
  text(gfx, center - width(gfx, label) / 2, 199, label, 60);
  gfx.setFont(u8g2_font_helvB08_tf);
  String number = numeric(value);
  text(gfx, center - width(gfx, number.c_str()) / 2, 215, number, 60);
}

int pointX(const HistorySeries &history, const Plot &plot, uint32_t epoch) {
  double fraction = static_cast<double>(epoch - history.startEpoch) / (history.endEpoch - history.startEpoch);
  return plot.left + lround(fraction * (plot.right - plot.left));
}

int pointY(const Limits &limits, float value) {
  float fraction = (value - limits.low) / (limits.high - limits.low);
  return PLOT_BOTTOM - lroundf(fraction * (PLOT_BOTTOM - PLOT_TOP));
}

void marker(U8G2 &gfx, const Plot &plot, int x, int y) {
  gfx.drawPixel(x, y);
  if (x > plot.left) gfx.drawPixel(x - 1, y);
  if (x < plot.right) gfx.drawPixel(x + 1, y);
  if (y > PLOT_TOP) gfx.drawPixel(x, y - 1);
  if (y < PLOT_BOTTOM) gfx.drawPixel(x, y + 1);
}

void dashedLine(U8G2 &gfx, int x0, int y0, int x1, int y1, uint32_t &phase) {
  int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
  int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
  int error = dx + dy;
  // Carry the pattern across segments, so dense histories remain visibly dashed.
  for (;;) {
    if ((phase++ % 7) < 4) gfx.drawPixel(x0, y0);
    if (x0 == x1 && y0 == y1) break;
    int doubled = 2 * error;
    if (doubled >= dy) { error += dy; x0 += sx; }
    if (doubled <= dx) { error += dx; y0 += sy; }
  }
}

void trace(U8G2 &gfx, const HistorySeries &history, const Plot &plot,
    const Limits &limits, Field field, bool dashed) {
  if (!limits.valid) return;
  bool previousValid = false;
  uint32_t previousEpoch = 0, phase = 0;
  int previousX = 0, previousY = 0;
  for (uint16_t i = 0; i < pointCount(history); ++i) {
    const HistoryPoint &point = history.points[i];
    float value = pointValue(point, field);
    if (!inWindow(history, point.epoch) || !isfinite(value)) {
      previousValid = false;
      phase = 0;
      continue;
    }
    int x = pointX(history, plot, point.epoch), y = pointY(limits, value);
    bool connect = previousValid && point.epoch > previousEpoch
        && (history.sourceGapsKnown || point.epoch - previousEpoch <= history.gapThresholdSeconds)
        && !crossesGap(history, previousEpoch, point.epoch);
    if (connect) {
      if (dashed) dashedLine(gfx, previousX, previousY, x, y, phase);
      else gfx.drawLine(previousX, previousY, x, y);
    } else {
      marker(gfx, plot, x, y);
      phase = 0;
    }
    previousEpoch = point.epoch;
    previousX = x;
    previousY = y;
    previousValid = true;
  }
}

void axes(U8G2 &gfx, const HistorySeries &history, const Plot &plot, const Limits &limits) {
  gfx.setFont(u8g2_font_helvB08_tf);
  gfx.drawVLine(plot.left - 1, PLOT_TOP, PLOT_BOTTOM - PLOT_TOP + 1);
  gfx.drawHLine(plot.left - 1, PLOT_BOTTOM + 1, plot.right - plot.left + 2);
  for (int tick = 0; tick < 3; ++tick) {
    int y = PLOT_BOTTOM - tick * ((PLOT_BOTTOM - PLOT_TOP) / 2);
    for (int x = plot.left + 2; x <= plot.right; x += 6) gfx.drawPixel(x, y);
    if (!limits.valid) continue;
    float value = limits.low + tick * (limits.high - limits.low) / 2;
    String label = numeric(value, limits.high - limits.low >= 20 ? 0 : 1);
    int rightAligned = plot.left - 4 - width(gfx, label.c_str());
    text(gfx, max(plot.labelLeft, rightAligned), y + 3, label, plot.left - plot.labelLeft - 4);
  }
  if (history.endEpoch <= history.startEpoch || history.startEpoch < VALID_EPOCH) return;
  String start = clockText(history.startEpoch, "%d/%m %H:%M");
  String end = clockText(history.endEpoch, "%d/%m %H:%M");
  text(gfx, plot.left - 3, AXIS_BASELINE, start, 70);
  text(gfx, plot.right - width(gfx, end.c_str()), AXIS_BASELINE, end, 70);
}

void unavailable(U8G2 &gfx, const Plot &plot, const char *field) {
  gfx.setFont(u8g2_font_helvB08_tf);
  String first = "No " + String(field);
  text(gfx, plot.left + 5, 111, first, plot.right - plot.left - 10);
  text(gfx, plot.left + 5, 128, "observations", plot.right - plot.left - 10);
}
}

void drawHistoryGraphs(U8G2 &gfx, const HistorySeries &history) {
  const Plot pm = {42, 197, 8};
  const Plot outdoor = {245, 391, 212};
  gfx.setDrawColor(1);
  gfx.drawVLine(204, 34, 184);
  Latest latestPm = latestPoint(history, Field::Pm25);
  Latest latestTemperature = latestPoint(history, Field::Temperature);
  Latest latestHeatIndex = latestPoint(history, Field::HeatIndex);
  bool validWindow = history.startEpoch >= VALID_EPOCH && history.endEpoch > history.startEpoch;
  Limits pmLimits = validWindow ? plotLimits(history, false) : Limits();
  Limits temperatureLimits = validWindow ? plotLimits(history, true) : Limits();

  gfx.setFont(u8g2_font_helvB08_tf);
  text(gfx, 8, 43, String("PM2.5 µg/m³") + (old(history, latestPm) ? " | OLD" : ""), 190);
  bool temperatureOld = old(history, latestTemperature) || old(history, latestHeatIndex);
  text(gfx, 212, 43, String("OUTDOOR °C") + (temperatureOld ? " | OLD" : ""), 180);
  text(gfx, 8, 59, "Latest " + numeric(latestPm.value), 190);
  uint32_t temperatureEpoch = max(latestTemperature.epoch, latestHeatIndex.epoch);
  text(gfx, 212, 59, temperatureEpoch ? "Observed " + clockText(temperatureEpoch) : "History unavailable", 180);

  axes(gfx, history, pm, pmLimits);
  axes(gfx, history, outdoor, temperatureLimits);
  if (!pmLimits.valid) unavailable(gfx, pm, "PM2.5");
  else trace(gfx, history, pm, pmLimits, Field::Pm25, false);
  if (!temperatureLimits.valid) unavailable(gfx, outdoor, "outdoor");
  else {
    trace(gfx, history, outdoor, temperatureLimits, Field::Temperature, false);
    trace(gfx, history, outdoor, temperatureLimits, Field::HeatIndex, true);
  }

  bool summaryOk = validWindow && summaryAvailable(history.pmSummary);
  summaryValue(gfx, 39, "Average", summaryOk ? history.pmSummary.average : NAN);
  summaryValue(gfx, 103, "Lowest", summaryOk ? history.pmSummary.lowest : NAN);
  summaryValue(gfx, 167, "Highest", summaryOk ? history.pmSummary.highest : NAN);

  gfx.setFont(u8g2_font_helvB08_tf);
  gfx.drawHLine(212, 195, 14);
  text(gfx, 234, 199, "Temperature " + numeric(latestTemperature.value) + "°C", 157);
  uint32_t legendPhase = 0;
  dashedLine(gfx, 212, 212, 229, 212, legendPhase);
  text(gfx, 234, 216, "Heat index " + numeric(latestHeatIndex.value) + "°C", 157);
  gfx.setDrawColor(1);
}
