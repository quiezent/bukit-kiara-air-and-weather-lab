#pragma once
#include <Arduino.h>
#include <U8g2lib.h>
#include <math.h>

// Observations only. Use NAN for an unavailable field; it breaks that trace.
// The caller supplies at most 128 points, preferably in increasing UTC order.
constexpr uint16_t HISTORY_GRAPH_POINT_LIMIT = 128;
constexpr uint16_t HISTORY_GRAPH_GAP_LIMIT = 128;

struct HistoryPoint {
  uint32_t epoch = 0;
  float pm25 = NAN;
  float temperatureC = NAN;
  float heatIndexC = NAN;
};

// Actual collection gaps, retained even if the server downsamples the points.
// The endpoints can be observations; their open interior must not be joined.
struct HistoryGap {
  uint32_t leftEpoch = 0;
  uint32_t rightEpoch = 0;
};

// Supplied by the server from accepted PM source observations in this exact
// history window, before downsampling. Never populate from retained points.
struct HistoryPmSummary {
  bool available = false;
  float average = NAN;
  float lowest = NAN;
  float highest = NAN;
  uint32_t sampleCount = 0;
};

struct HistorySeries {
  const HistoryPoint *points = nullptr;
  uint16_t count = 0;
  const HistoryGap *gaps = nullptr;
  uint16_t gapCount = 0;
  // True only when gaps came from original collection, before downsampling.
  // A known empty gap list means continuous collection; retained-point spacing
  // then does not represent a sensor outage.
  bool sourceGapsKnown = false;
  uint32_t startEpoch = 0;
  uint32_t endEpoch = 0;
  uint32_t nowEpoch = 0;
  uint32_t gapThresholdSeconds = 900;
  uint32_t freshAfterSeconds = 420;
  bool available = true;
  bool fresh = true;
  HistoryPmSummary pmSummary;
};

// Draws only x=8..392, y=32..218, leaving the page header and device rows free.
// Both temperature traces use the same degrees-Celsius axis. All times are MYT.
// This does not clear the display or send its buffer.
void drawHistoryGraphs(U8G2 &gfx, const HistorySeries &history);
