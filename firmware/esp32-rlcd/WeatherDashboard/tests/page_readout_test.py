"""Compile actual PageReadout against real ArduinoJson and an offline clock.

Only the graphics class and Arduino String storage are stubbed. The String
formatter is the exact installed Arduino-ESP32 dtostrf implementation, and
Dashboard availability/summary helpers are extracted verbatim from production.
No PCM, network, serial port, speaker, or ESP32 build is used.
"""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


ARDUINO = r"""
#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <ctime>
#include <string>
#include <math.h>
// MinGW lacks the POSIX API. These single-threaded tests use the same UTC result.
#if defined(_WIN32)
inline std::tm *gmtime_r(const std::time_t *value, std::tm *result) {
  const std::tm *converted = std::gmtime(value);
  if (converted) *result = *converted;
  return converted ? result : nullptr;
}
#endif
@@DTOSTRF@@
class String {
 public:
  String() = default;
  String(const char *value) : storage_(value ? value : "") {}
  String(float value, unsigned decimals) {
    char buffer[128];
    storage_ = dtostrf(value, decimals + 2, decimals, buffer);
  }
  String &operator+=(const char *value) { storage_ += value; return *this; }
  String &operator+=(char value) { storage_ += value; return *this; }
  String &operator+=(const String &value) { storage_ += value.storage_; return *this; }
  const char *c_str() const { return storage_.c_str(); }
  float toFloat() const { return std::strtof(c_str(), nullptr); }
  void trim() {
    while (!storage_.empty() && std::isspace(static_cast<unsigned char>(storage_.back()))) storage_.pop_back();
    const size_t start = storage_.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) storage_.clear();
    else storage_.erase(0, start);
  }
  const std::string &str() const { return storage_; }
 private:
  std::string storage_;
};
"""


HARNESS = r"""
#include "PageReadout.h"
#include "BatteryEstimate.h"
#include "ForecastSemantics.h"
#include "ArrivalChange.h"
#include <cassert>
#include <cstdio>
#include <initializer_list>
#include <string>
using C = SpeechClip;
constexpr uint32_t now = 1791428400; // 8 Oct 2026 11:00 Malaysia, UTC+8.
constexpr uint32_t midnight = (now + 28800) / 86400 * 86400 - 28800;
constexpr uint32_t clipSamples[] = {@@CLIP_SAMPLES@@};
uint32_t dashboardNow(const DashboardContext &context) {
  return (*context.weather)["test_now"].as<uint32_t>();
}
static void contains(const String &text, const char *part) {
  if (text.str().find(part) == std::string::npos) {
    fprintf(stderr, "Missing: %s\nActual: %s\n", part, text.c_str());
    assert(false);
  }
}
static void lacks(const String &text, const char *part) { assert(text.str().find(part) == std::string::npos); }
static size_t count(const SpeechPlaylist &clips, C clip) {
  size_t result = 0;
  for (size_t i = 0; i < clips.count; ++i) result += clips.clips[i] == clip;
  return result;
}
static void sequence(const SpeechPlaylist &clips, std::initializer_list<C> wanted) {
  assert(wanted.size());
  for (size_t start = 0; start + wanted.size() <= clips.count; ++start) {
    size_t offset = start;
    bool match = true;
    for (C clip : wanted) match &= clips.clips[offset++] == clip;
    if (match) return;
  }
  fprintf(stderr, "Missing clip sequence starting ID %u\n", unsigned(*wanted.begin()));
  assert(false);
}
static float duration(const SpeechPlaylist &clips) {
  uint32_t samples = 0;
  for (size_t i = 0; i < clips.count; ++i) samples += clipSamples[size_t(clips.clips[i])];
  return samples / 16000.0f;
}
static void build(DashboardContext &context, SpeechPlaylist &clips, String &text) {
  assert(buildCurrentPageReadout(context, clips, text));
  assert(clips.count && clips.count <= SpeechPlaylist::kCapacity);
  assert(!text.str().empty() && text.str().back() == '.');
  for (size_t i = 0; i < clips.count; ++i) assert(clips.clips[i] < C::Count);
  assert(!count(clips, C::Micrograms));
  assert(!count(clips, C::Observed)); // Retained recording is omitted from narration.
  lacks(text, "Observed:");
  // Crossing events remain diagnostic data. Retired recordings are absent
  // from the enum, and their obsolete meaning must never enter narration.
  for (const char *legacy : {"First rise", "First drop", "Rise of twenty or more",
                            "Drop of twenty or more", "No change of twenty or more"})
    lacks(text, legacy);
}
static void sessionFixture(JsonVariant session, uint32_t start, float pm, float rain = 12) {
  session["issued_epoch"] = now;
  session["start_epoch"] = start;
  session["end_epoch"] = start + 7200;
  session["pm"]["available"] = true;
  session["pm"]["role"] = "experimental_window_mean";
  session["pm"]["pm25_ugm3"] = pm;
  session["pm"]["range_low_ugm3"] = 999; // A range endpoint must not drive ranking.
  session["weather"]["available"] = true;
  session["weather"]["fresh"] = true;
  session["weather"]["fetched_epoch"] = now;
  session["weather"]["rain_chance_max_pct"] = rain;
}
static void fixture(JsonDocument &d) {
  d.clear(); d["test_now"] = now;
  auto current = d["current"];
  current["available"] = true; current["fresh"] = true;
  current["observed_epoch"] = now - 60;
  current["pm25_ugm3"] = 162.2; current["temperature_c"] = 27.6;
  current["display_text"] = "Latest sensor reading";
  auto weather = d["weather"];
  weather["available"] = true; weather["fresh"] = true; weather["fetched_epoch"] = now;
  // current_hour's rain covers the PREVIOUS hour. It must not be spoken as now.
  weather["current_hour"]["rain_start_epoch"] = now - 3600;
  weather["current_hour"]["rain_end_epoch"] = now;
  weather["current_hour"]["rain_chance_pct"] = 99;
  for (unsigned i = 0; i < 3; ++i) {
    weather["next_hours"][i]["rain_start_epoch"] = now + i * 3600;
    weather["next_hours"][i]["rain_end_epoch"] = now + (i + 1) * 3600;
    weather["next_hours"][i]["rain_chance_pct"] = i + 1;
  }
  auto forecast = d["forecast"];
  forecast["available"] = true; forecast["fresh"] = true; forecast["issued_epoch"] = now;
  auto near = forecast["near90"];
  near["available"] = true; near["target_epoch"] = now + 5400; near["pm25_ugm3"] = 888.8;
  near["role"] = "experimental_model_output";
  auto event = near["first20"];
  event["available"] = true; event["direction"] = "rise";
  event["rise_probability"] = .937; event["drop_probability"] = .063;
  event["none_probability"] = 0; event["reference_ugm3"] = 157.7;
  auto arrival = near["arrival_change"];
  arrival["available"] = true;
  arrival["issued_epoch"] = now; arrival["arrival_epoch"] = now + 5400;
  arrival["fresh_reference_epoch"] = now - 60;
  arrival["reference_ugm3"] = 157.7;
  arrival["fall20"] = .106; arrival["fall40"] = .03;
  arrival["rise20"] = .106; arrival["rise40"] = .02; arrival["within20"] = .788;
  arrival["outcome"] = "within20"; arrival["outcome_probability"] = .788;
  arrival["display_text"] = "No change on arrival";
  auto ride = forecast["ride90_210"].to<JsonObject>();
  sessionFixture(ride, now + 5400, 777.7);
  ride["available"] = true; ride["range_kind"] = "empirical_q10_q90";
  ride["pm25_ugm3"] = 777.7; ride["role"] = "experimental_window_mean";
  ride["range_low_ugm3"] = 126.4; ride["range_high_ugm3"] = 184.7;
  ride["mean_le70"]["available"] = true; ride["mean_le70"]["probability"] = .048;
  sessionFixture(forecast["sessions"]["morning"].to<JsonObject>(), midnight + 86400 + 9 * 3600, 111.1);
  sessionFixture(forecast["sessions"]["afternoon"].to<JsonObject>(), midnight + 14 * 3600, 222.2);
  auto history = d["history"];
  history["available"] = true; history["fresh"] = true;
  history["start_epoch"] = now - 21600; history["end_epoch"] = now;
  history["observed_epoch"] = now - 10;
  history["columns"][0] = "epoch"; history["columns"][1] = "pm25_ugm3";
  history["columns"][2] = "temperature_c"; history["columns"][3] = "heat_index_c";
  for (unsigned i = 0; i < 2; ++i) {
    history["points"][i][0] = now - (i ? 10 : 100);
    history["points"][i][1] = i ? 135.6 : 149.5;
    history["points"][i][2] = 29.1; history["points"][i][3] = 36.1;
  }
  auto summary = history["pm25_summary"];
  summary["available"] = true; summary["sample_count"] = 1000;
  summary["average_ugm3"] = 154.6; summary["lowest_ugm3"] = 120.1; summary["highest_ugm3"] = 190.9;
}
static IndoorReading indoorFixture() {
  IndoorReading x; x.valid = true; x.temperatureC = 29.6f; x.humidityPct = 64.4f;
  x.minimumC = 28.1f; x.maximumC = 31.2f; return x;
}
struct Test {
  JsonDocument d; IndoorReading indoor = indoorFixture(); DashboardContext c;
  SpeechPlaylist clips; String text;
  Test() { fixture(d); c.weather = &d; c.indoor = &indoor; c.batteryV = 4.126f; }
  void page(unsigned p) { c.page = p; build(c, clips, text); }
  void reset() { fixture(d); }
};
static void routingAndSnapshot() {
  Test t;
  t.page(0);
  assert(t.text.str() == "Current outdoor PM2.5 162.2. Latest sensor reading. Outdoor temperature 27.6 degrees Celsius. Rain chance 1 percent. Ninety minute forecast PM2.5 888.8. No change on arrival, 78.8 percent. Rain chance in an hour 2 percent.");
  lacks(t.text, "Indoor"); lacks(t.text, "126.4");
  sequence(t.clips, {C::NoChangeOnArrival, C::N70, C::N8, C::Point, C::N8, C::Percent, C::RainInAnHour});
  t.page(1);
  assert(t.text.str() == "PM2.5 forecast range 126.4 to 184.7. Chance the ride window average seventy or less is 4.8 percent. Rain chance 12 percent. Tomorrow morning has lower forecast PM2.5 than This afternoon.");
  lacks(t.text, "888.8"); lacks(t.text, "777.7"); lacks(t.text, "Feels like");
  t.page(2);
  assert(t.text.str() == "Average PM2.5 154.6, lowest 120.1, highest 190.9. Indoor temperature 29.6 degrees Celsius, lowest 28.1 degrees Celsius. Indoor humidity 64.4 percent. Battery 4.13 volts, 95 percent capacity.");
  assert(count(t.clips, C::DegreesCelsius) == 2 && count(t.clips, C::Lowest) == 2);
  lacks(t.text, "Latest"); lacks(t.text, "heat index");
  // Only clip IDs/transcript survive; later data changes do not alter a queued readout.
  SpeechPlaylist snapshot = t.clips; const std::string transcript = t.text.str();
  t.d.clear(); t.indoor.temperatureC = 99; t.c.batteryV = 3;
  assert(t.text.str() == transcript && t.clips.count == snapshot.count);
  for (size_t i = 0; i < snapshot.count; ++i) assert(t.clips.clips[i] == snapshot.clips[i]);
  t.c.weather = nullptr; t.page(2);
  assert(t.clips.count == 1 && t.text.str() == "unavailable.");
}
static void demoReadoutDisclosure() {
  Test t;
  for (unsigned page = 0; page < 3; ++page) {
    t.reset(); t.page(page);
    const std::string liveText = t.text.str();
    const SpeechPlaylist liveClips = t.clips;
    assert(!count(liveClips, C::DemoData));
    t.d["demo"] = true; t.page(page);
    assert(t.text.str() == "Demonstration data. " + liveText);
    assert(t.clips.count == liveClips.count + 1);
    assert(t.clips.clips[0] == C::DemoData && count(t.clips, C::DemoData) == 1);
    // The disclosure must leave every numeric value and following audio clip
    // unchanged: synthetic mode is provenance, not a different forecast path.
    for (size_t i = 0; i < liveClips.count; ++i)
      assert(t.clips.clips[i + 1] == liveClips.clips[i]);
    for (unsigned nonDemo = 0; nonDemo < 5; ++nonDemo) {
      if (nonDemo == 0) t.d["demo"] = false;
      if (nonDemo == 1) t.d["demo"] = nullptr;
      if (nonDemo == 2) t.d["demo"] = "true";
      if (nonDemo == 3) t.d["demo"] = 1;
      if (nonDemo == 4) t.d.remove("demo");
      t.page(page);
      assert(t.text.str() == liveText && t.clips.count == liveClips.count);
      assert(!count(t.clips, C::DemoData));
      for (size_t i = 0; i < liveClips.count; ++i)
        assert(t.clips.clips[i] == liveClips.clips[i]);
    }
  }
}
static void rainIntervalsAndValues() {
  Test t;
  // Refresh clocks advance but cached hourly positions do not: search intervals.
  t.d["test_now"] = now + 3600; t.page(0);
  contains(t.text, "Rain chance 2 percent."); contains(t.text, "Rain chance in an hour 3 percent.");
  t.reset(); t.d["weather"]["next_hours"][0]["rain_start_epoch"] = now + 100;
  t.page(0); contains(t.text, "Rain chance unavailable.");
  t.reset(); t.d["weather"]["next_hours"][1]["rain_end_epoch"] = now + 3600;
  t.page(0); contains(t.text, "Rain chance in an hour unavailable.");
  t.reset(); t.d["weather"]["current_hour"]["rain_start_epoch"] = now;
  t.d["weather"]["current_hour"]["rain_end_epoch"] = now + 3600;
  t.page(0); contains(t.text, "Rain chance 99 percent.");
  t.reset(); t.d["weather"]["next_hours"][0]["rain_chance_pct"] = nullptr;
  t.page(0); sequence(t.clips, {C::RainChance, C::UnavailableValue});
  t.d["weather"]["next_hours"][0]["rain_chance_pct"] = 0; t.page(0);
  sequence(t.clips, {C::RainChance, C::N0, C::Percent});
  t.d["weather"]["next_hours"][0]["rain_chance_pct"] = 101; t.page(0);
  contains(t.text, "Rain chance unavailable.");
  t.reset(); t.d["weather"]["fresh"] = false; t.page(0);
  assert(count(t.clips, C::ForecastOld) == 1);
  t.d["weather"]["fetched_epoch"] = now - 7201; t.page(0);
  assert(!count(t.clips, C::ForecastOld)); contains(t.text, "Rain chance unavailable.");
  t.d["weather"]["fetched_epoch"] = now + 1; t.page(0);
  contains(t.text, "Rain chance in an hour unavailable.");
  t.reset(); t.d["current"]["pm25_ugm3"] = 0; t.d["current"]["temperature_c"] = -0.04;
  t.page(0); sequence(t.clips, {C::CurrentPM25, C::N0, C::Point, C::N0});
  sequence(t.clips, {C::OutdoorTemperature, C::N0, C::Point, C::N0, C::DegreesCelsius});
  t.d["current"]["temperature_c"] = 26.25; t.page(0);
  contains(t.text, "Outdoor temperature 26.2 degrees Celsius.");
  t.d["current"]["fresh"] = false; t.page(0); assert(count(t.clips, C::CurrentDataOld) == 1);
  t.d["current"]["observed_epoch"] = now - 901; t.page(0);
  assert(!count(t.clips, C::CurrentDataOld)); contains(t.text, "Current outdoor PM2.5 unavailable.");
}
static void observedStatusReadout() {
  struct Expected { const char *body; C clip; };
  const Expected labels[] = {
    {"Fast PM2.5 rise detected", C::ObservedFastRise},
    {"Fast PM2.5 rebound detected", C::ObservedFastRebound},
    {"Particle rebound may be starting", C::ObservedReboundStarting},
    {"Recent PM2.5 medians at or below 35 µg/m³", C::ObservedRecentMedians35},
    {"Moist-cooling particle clearing forming", C::ObservedMoistCoolingClearing},
    {"Dry clearing forming", C::ObservedDryClearing},
    {"PM2.5 reduction forming", C::ObservedReductionForming},
    {"Rapid PM2.5 reduction detected", C::ObservedRapidReduction},
  };
  Test t;
  const auto statusCount = [&]() {
    size_t result = count(t.clips, C::Observed) + count(t.clips, C::LatestSensorReading)
        + count(t.clips, C::ObservedStatusUnavailable);
    for (const Expected &label : labels) result += count(t.clips, label.clip);
    return result;
  };
  float longest = 0;
  size_t longestCount = 0;
  float longestComplex = 0;
  size_t longestComplexCount = 0;
  for (const Expected &label : labels) {
    t.reset();
    const std::string source = std::string("Observed: ") + label.body;
    t.d["current"]["display_text"] = source;
    t.page(0);
    const std::string expected = std::string("Current outdoor PM2.5 162.2. ")
        + label.body + ". Outdoor temperature 27.6 degrees Celsius.";
    contains(t.text, expected.c_str());
    assert(count(t.clips, C::Observed) == 0 && count(t.clips, label.clip) == 1 && statusCount() == 1);
    sequence(t.clips, {C::CurrentPM25, C::N1, C::Hundred, C::N60, C::N2,
        C::Point, C::N2, label.clip, C::OutdoorTemperature});
    contains(t.text, "No change on arrival, 78.8 percent.");
    assert(duration(t.clips) < 60 && t.clips.count < SpeechPlaylist::kCapacity);
    if (duration(t.clips) > longest) { longest = duration(t.clips); longestCount = t.clips.count; }
    t.d["current"]["pm25_ugm3"] = 9999.9;
    t.d["current"]["temperature_c"] = -99.9;
    t.d["current"]["fresh"] = false;
    t.d["weather"]["fresh"] = false;
    t.d["forecast"]["fresh"] = false;
    t.d["forecast"]["near90"]["pm25_ugm3"] = 9999.9;
    t.page(0);
    assert(statusCount() == 1 && count(t.clips, C::CurrentDataOld) == 1);
    assert(duration(t.clips) < 60 && t.clips.count < SpeechPlaylist::kCapacity);
    if (duration(t.clips) > longestComplex) {
      longestComplex = duration(t.clips); longestComplexCount = t.clips.count;
    }
    for (unsigned page : {1u, 2u}) {
      t.page(page); assert(statusCount() == 0); lacks(t.text, label.body);
    }
  }
  printf("Longest observed Page 1: %.2fs, %zu/%zu clip IDs\n", longest, longestCount, SpeechPlaylist::kCapacity);
  printf("Longest complex observed Page 1: %.2fs, %zu/%zu clip IDs\n",
      longestComplex, longestComplexCount, SpeechPlaylist::kCapacity);
  // The API-backed text and clip IDs are captured together before later refreshes.
  t.reset(); t.d["current"]["display_text"] = "Observed: Particle rebound may be starting";
  t.page(0); const SpeechPlaylist captured = t.clips; const std::string text = t.text.str();
  t.d["current"]["display_text"] = "Observed: Rapid PM2.5 reduction detected";
  t.d["current"]["pm25_ugm3"] = 42; t.d.clear();
  assert(t.text.str() == text && t.clips.count == captured.count);
  for (size_t i = 0; i < captured.count; ++i) assert(t.clips.clips[i] == captured.clips[i]);
  t.reset(); t.page(0);
  contains(t.text, "Current outdoor PM2.5 162.2. Latest sensor reading. Outdoor temperature");
  sequence(t.clips, {C::Point, C::N2, C::LatestSensorReading, C::OutdoorTemperature});
  assert(statusCount() == 1 && !count(t.clips, C::Observed));
  // Exact matching avoids speaking a known fragment while logging unsupported text.
  for (const char *unknown : {"", "Observed: New server status", "Particle rebound may be starting",
      "Observed: Particle rebound may be starting extra", "Observed: Particle rebound may be starting ",
      "observed: Particle rebound may be starting", " Latest sensor reading"}) {
    t.reset(); t.d["current"]["display_text"] = unknown; t.page(0);
    contains(t.text, "Current outdoor PM2.5 162.2. Observed status unavailable. Outdoor temperature");
    assert(statusCount() == 1 && count(t.clips, C::ObservedStatusUnavailable) == 1);
    sequence(t.clips, {C::Point, C::N2, C::ObservedStatusUnavailable, C::OutdoorTemperature});
    if (unknown[0]) lacks(t.text, unknown);
  }
  for (unsigned malformed = 0; malformed < 7; ++malformed) {
    t.reset();
    if (malformed == 0) t.d["current"].remove("display_text");
    if (malformed == 1) t.d["current"]["display_text"] = nullptr;
    if (malformed == 2) t.d["current"]["display_text"] = true;
    if (malformed == 3) t.d["current"]["display_text"] = 42;
    if (malformed == 4) t.d["current"]["display_text"] = 1.5;
    if (malformed == 5) t.d["current"]["display_text"].to<JsonObject>()["text"] = "Latest sensor reading";
    if (malformed == 6) t.d["current"]["display_text"].to<JsonArray>().add("Latest sensor reading");
    t.page(0); assert(statusCount() == 1 && count(t.clips, C::ObservedStatusUnavailable) == 1);
  }
  for (uint32_t sourceAge : {uint32_t(0), uint32_t(420), uint32_t(421), uint32_t(900), uint32_t(901)}) {
    t.reset(); t.d["current"]["display_text"] = "Observed: Particle rebound may be starting";
    t.d["current"]["observed_epoch"] = now - sourceAge;
    t.page(0);
    assert(statusCount() == (sourceAge <= 900 ? 1 : 0));
    assert(count(t.clips, C::CurrentDataOld) == (sourceAge > 420 && sourceAge <= 900 ? 1 : 0));
    if (sourceAge > 420 && sourceAge <= 900)
      contains(t.text, "Outdoor data is old. Current outdoor PM2.5 162.2. Particle rebound may be starting.");
  }
  // Missing/nonboolean freshness is old, matching the screen. It does not erase
  // a still-available cached observation or duplicate the old-current warning.
  for (unsigned freshState = 0; freshState < 5; ++freshState) {
    t.reset(); t.d["current"]["display_text"] = "Observed: Particle rebound may be starting";
    if (freshState == 0) t.d["current"]["fresh"] = false;
    if (freshState == 1) t.d["current"].remove("fresh");
    if (freshState == 2) t.d["current"]["fresh"] = nullptr;
    if (freshState == 3) t.d["current"]["fresh"] = 1;
    if (freshState == 4) t.d["current"]["fresh"] = "true";
    t.page(0); assert(statusCount() == 1 && count(t.clips, C::CurrentDataOld) == 1);
  }
  // The status never bypasses clock typing, source expiry or numerical validity.
  for (unsigned invalid = 0; invalid < 12; ++invalid) {
    t.reset(); t.d["current"]["display_text"] = "Observed: Particle rebound may be starting";
    if (invalid == 0) t.d["current"]["available"] = false;
    if (invalid == 1) t.d["current"].remove("observed_epoch");
    if (invalid == 2) t.d["current"]["observed_epoch"] = nullptr;
    if (invalid == 3) t.d["current"]["observed_epoch"] = now + 1;
    if (invalid == 4) t.d["current"]["observed_epoch"] = 0;
    if (invalid == 5) t.d["current"]["observed_epoch"] = std::to_string(now - 60);
    if (invalid == 6) t.d["current"]["observed_epoch"] = -1;
    if (invalid == 7) t.d["current"]["observed_epoch"] = true;
    if (invalid == 8) t.d["current"]["observed_epoch"] = 1791428340.5;
    if (invalid == 9) t.d["current"]["observed_epoch"] = uint64_t(UINT32_MAX) + 1;
    if (invalid == 10) t.d["current"]["pm25_ugm3"] = "162.2";
    if (invalid == 11) t.d["current"]["pm25_ugm3"] = true;
    t.page(0); assert(statusCount() == 0); lacks(t.text, "Particle rebound");
  }
  for (float invalid : {-1.0f, 10000.0f, float(NAN), float(INFINITY), float(-INFINITY)}) {
    t.reset(); t.d["current"]["display_text"] = "Observed: Particle rebound may be starting";
    t.d["current"]["pm25_ugm3"] = invalid; t.page(0); assert(statusCount() == 0);
  }
  t.reset(); t.d["current"]["display_text"] = "Observed: Particle rebound may be starting";
  t.d["current"]["pm25_ugm3"] = nullptr; t.page(0); assert(statusCount() == 0);
  for (float endpoint : {0.0f, 9999.9f}) {
    t.reset(); t.d["current"]["display_text"] = "Observed: Particle rebound may be starting";
    t.d["current"]["pm25_ugm3"] = endpoint; t.page(0); assert(statusCount() == 1);
  }
  t.reset(); t.d["current"]["display_text"] = "Observed: Particle rebound may be starting";
  t.d["weather"]["available"] = false; t.d["forecast"]["available"] = false;
  t.page(0); assert(statusCount() == 1); // Forecast availability does not gate observations.
}
static void first20Semantics() {
  Test t; JsonObject event = t.d["forecast"]["near90"]["first20"].as<JsonObject>();
  assert(first20ModelCall(event) == First20Call::Rise);
  // Argmax can indicate a rise below50%; do not invent a >=50% threshold.
  event["rise_probability"] = .4; event["drop_probability"] = .3; event["none_probability"] = .3;
  assert(first20ModelCall(event) == First20Call::Rise);
  event["rise_probability"] = .2; event["drop_probability"] = .7; event["none_probability"] = .1;
  event["direction"] = "drop"; assert(first20ModelCall(event) == First20Call::Drop);
  event["rise_probability"] = .4; event["drop_probability"] = .4; event["none_probability"] = .2;
  event["direction"] = "unresolved"; assert(first20ModelCall(event) == First20Call::Unresolved);
  event["direction"] = "rise"; assert(first20ModelCall(event) == First20Call::Invalid);
  t.reset(); event = t.d["forecast"]["near90"]["first20"].as<JsonObject>();
  event["rise_probability"] = nullptr; assert(first20ModelCall(event) == First20Call::Invalid);
  t.reset(); event = t.d["forecast"]["near90"]["first20"].as<JsonObject>();
  event["direction"] = "drop"; event["rise_probability"] = .1;
  event["drop_probability"] = .8; event["none_probability"] = .1; event["reference_ugm3"] = 19;
  assert(first20ModelCall(event) == First20Call::Invalid);
  t.reset(); t.d["forecast"]["fresh"] = false; t.page(0);
  assert(count(t.clips, C::Old) == 1 && count(t.clips, C::NoChangeOnArrival) == 1);
  t.d["forecast"]["issued_epoch"] = now - 601; t.page(0);
  assert(!count(t.clips, C::NoChangeOnArrival)); contains(t.text, "Ninety minute forecast PM2.5 unavailable.");
  t.reset(); t.d["forecast"]["near90"]["target_epoch"] = now; t.page(0);
  assert(!count(t.clips, C::NoChangeOnArrival));
}
static void sessionComparisonAndWarnings() {
  Test t;
  auto morning = t.d["forecast"]["sessions"]["morning"];
  auto afternoon = t.d["forecast"]["sessions"]["afternoon"];
  morning["weather"]["rain_chance_max_pct"] = 0;
  afternoon["weather"]["rain_chance_max_pct"] = 97; t.page(1);
  contains(t.text, "Tomorrow morning has lower forecast PM2.5 than This afternoon.");
  contains(t.text, "Warning This afternoon has a high rain chance of 97 percent.");
  assert(count(t.clips, C::Warning) == 1);
  morning["weather"]["rain_chance_max_pct"] = 75; afternoon["weather"]["rain_chance_max_pct"] = 75;
  t.d["forecast"]["ride90_210"]["weather"]["rain_chance_max_pct"] = 75; t.page(1);
  assert(!count(t.clips, C::Warning) && !count(t.clips, C::RideRainWarning));
  morning["weather"]["rain_chance_max_pct"] = 75.01;
  afternoon["weather"]["rain_chance_max_pct"] = 100;
  t.d["forecast"]["ride90_210"]["weather"]["rain_chance_max_pct"] = 75.01; t.page(1);
  assert(count(t.clips, C::Warning) == 2 && count(t.clips, C::RideRainWarning) == 1);
  morning["weather"]["rain_chance_max_pct"] = 75.1; t.page(1);
  contains(t.text, "high rain chance of 75.1 percent.");
  morning["pm"]["pm25_ugm3"] = 333; t.page(1);
  contains(t.text, "This afternoon has lower forecast PM2.5 than Tomorrow morning.");
  afternoon["pm"]["pm25_ugm3"] = 111.12; morning["pm"]["pm25_ugm3"] = 111.14;
  t.page(1); assert(count(t.clips, C::SimilarPM) == 1);
  morning["pm"]["role"] = "persistence_anchor"; t.page(1);
  assert(count(t.clips, C::ComparisonUnavailable) == 1 && count(t.clips, C::Warning) == 2);
  t.reset(); t.d["forecast"]["sessions"]["morning"]["pm"]["pm25_ugm3"] = nullptr;
  t.page(1); assert(count(t.clips, C::ComparisonUnavailable) == 1);
  t.reset(); t.d["forecast"]["sessions"]["morning"]["issued_epoch"] = now - 1;
  t.page(1); assert(count(t.clips, C::ComparisonUnavailable) == 1);
  t.reset(); t.d["forecast"]["sessions"]["morning"]["start_epoch"] = "tomorrow";
  t.page(1); assert(count(t.clips, C::ComparisonUnavailable) == 1);
  t.reset(); t.d["forecast"]["sessions"]["morning"]["end_epoch"] = midnight + 86400 + 12 * 3600;
  t.page(1); assert(count(t.clips, C::ComparisonUnavailable) == 1);
  t.reset(); t.d["forecast"]["sessions"]["morning"]["end_epoch"] = now;
  t.d["forecast"]["sessions"]["morning"]["weather"]["rain_chance_max_pct"] = 99;
  t.page(1); assert(count(t.clips, C::ComparisonUnavailable) == 1 && !count(t.clips, C::Warning));
  t.reset(); t.d["forecast"]["sessions"]["morning"]["start_epoch"] = midnight + 2 * 86400 + 9 * 3600;
  t.d["forecast"]["sessions"]["morning"]["end_epoch"] = midnight + 2 * 86400 + 11 * 3600;
  t.page(1); assert(count(t.clips, C::ComparisonUnavailable) == 1);
  t.reset(); t.d["forecast"]["fresh"] = false; t.page(1);
  assert(count(t.clips, C::ComparisonUnavailable) == 1 && count(t.clips, C::Old) == 1);
  t.reset(); t.d["test_now"] = now + 121; t.page(1);
  assert(count(t.clips, C::ComparisonUnavailable) == 1);
  t.reset(); auto bad = t.d["forecast"]["sessions"]["afternoon"]["weather"];
  bad["rain_chance_max_pct"] = 100; bad["fetched_epoch"] = now - 7201; t.page(1);
  assert(!count(t.clips, C::Warning));
  bad["fetched_epoch"] = now + 1; t.page(1); assert(!count(t.clips, C::Warning));
  bad["fetched_epoch"] = now; bad["rain_chance_max_pct"] = 101; t.page(1);
  assert(!count(t.clips, C::Warning));
  bad["rain_chance_max_pct"] = nullptr; t.page(1); assert(!count(t.clips, C::Warning));
  bad["rain_chance_max_pct"] = 97; bad["fresh"] = false; t.page(1);
  assert(count(t.clips, C::Old) == 1 && count(t.clips, C::Warning) == 1);
  // After Malaysia midnight labels follow UTC+8 dates, regardless of host TZ.
  t.reset(); t.d["test_now"] = midnight + 86400 + 1;
  t.d["forecast"]["issued_epoch"] = midnight + 86400;
  t.d["forecast"]["sessions"]["morning"]["issued_epoch"] = midnight + 86400;
  auto a = t.d["forecast"]["sessions"]["afternoon"];
  a["issued_epoch"] = midnight + 86400;
  a["start_epoch"] = midnight + 86400 + 14 * 3600; a["end_epoch"] = midnight + 86400 + 16 * 3600;
  t.page(1); contains(t.text, "This morning has lower forecast PM2.5 than This afternoon.");
}
static void rangeHistoryIndoorBattery() {
  Test t; auto ride = t.d["forecast"]["ride90_210"];
  ride["range_low_ugm3"] = 0; ride["range_high_ugm3"] = 0; ride["mean_le70"]["probability"] = 0;
  t.page(1); contains(t.text, "PM2.5 forecast range 0.0 to 0.0.");
  sequence(t.clips, {C::ChanceBelow70Is, C::N0, C::Point, C::N0, C::Percent});
  ride["range_low_ugm3"] = 200; ride["range_high_ugm3"] = 100;
  ride["mean_le70"]["probability"] = 1.01; t.page(1);
  contains(t.text, "PM2.5 forecast 777.7.");
  assert(count(t.clips, C::PMForecast) == 1 && !count(t.clips, C::PMForecastRange));
  sequence(t.clips, {C::ChanceBelow70Is, C::UnavailableValue});
  t.reset(); t.d["forecast"]["ride90_210"]["range_kind"] = "unknown"; t.page(1);
  contains(t.text, "PM2.5 forecast 777.7.");
  t.reset(); t.d["forecast"]["issued_epoch"] = now - 601; t.page(1);
  sequence(t.clips, {C::PMForecastRange, C::UnavailableValue});
  sequence(t.clips, {C::ChanceBelow70Is, C::UnavailableValue});
  t.reset(); t.d["history"]["pm25_summary"]["average_ugm3"] = nullptr; t.page(2);
  contains(t.text, "Average PM2.5 unavailable, lowest unavailable, highest unavailable.");
  contains(t.text, "Indoor temperature 29.6 degrees Celsius");
  t.reset(); t.d["history"]["pm25_summary"]["sample_count"] = 0; t.page(2);
  sequence(t.clips, {C::AveragePM, C::UnavailableValue});
  t.reset(); t.d["history"]["pm25_summary"]["lowest_ugm3"] = 200; t.page(2);
  sequence(t.clips, {C::AveragePM, C::UnavailableValue});
  t.reset(); t.d["history"]["end_epoch"] = now + 1; t.page(2);
  sequence(t.clips, {C::AveragePM, C::UnavailableValue});
  t.reset(); t.d["history"]["fresh"] = false; t.page(2); assert(count(t.clips, C::Old) == 1);
  t.reset(); auto s = t.d["history"]["pm25_summary"];
  s["average_ugm3"] = 0; s["lowest_ugm3"] = 0; s["highest_ugm3"] = 0;
  t.page(2); contains(t.text, "Average PM2.5 0.0, lowest 0.0, highest 0.0.");
  t.indoor.valid = false; t.page(2);
  contains(t.text, "Indoor temperature unavailable, lowest 28.1 degrees Celsius.");
  contains(t.text, "Indoor humidity unavailable.");
  t.indoor.valid = true; t.indoor.minimumC = NAN; t.page(2);
  contains(t.text, "Indoor temperature 29.6 degrees Celsius, lowest unavailable.");
  t.indoor.temperatureC = NAN; t.indoor.minimumC = -99.9; t.indoor.humidityPct = 101; t.page(2);
  contains(t.text, "Indoor temperature unavailable, lowest -99.9 degrees Celsius.");
  contains(t.text, "Indoor humidity unavailable.");
  t.c.indoor = nullptr; t.page(2); contains(t.text, "lowest unavailable.");
  for (float battery : {NAN, 0.0f, 4.6f}) {
    t.c.batteryV = battery; t.page(2); contains(t.text, "Battery unavailable, unavailable percent capacity.");
    assert(!count(t.clips, C::Volts));
  }
  t.c.batteryV = 3; t.page(2); contains(t.text, "Battery 3.00 volts, 0 percent capacity.");
  t.c.batteryV = 4.2f; t.page(2); contains(t.text, "Battery 4.20 volts, 100 percent capacity.");
}
static void capacityDurationAndReset() {
  Test t;
  for (unsigned p = 0; p < 3; ++p) {
    t.page(p); assert(duration(t.clips) < 60);
    printf("Page %u: %.2fs, %zu/%zu clip IDs\n%s\n", p + 1, duration(t.clips), t.clips.count, SpeechPlaylist::kCapacity, t.text.c_str());
  }
  t.d["current"]["pm25_ugm3"] = 9999.9; t.d["current"]["temperature_c"] = -99.9;
  t.d["current"]["fresh"] = false; t.d["weather"]["fresh"] = false;
  t.d["forecast"]["near90"]["pm25_ugm3"] = 9999.9;
  auto ride = t.d["forecast"]["ride90_210"];
  ride["range_low_ugm3"] = 8888.8; ride["range_high_ugm3"] = 9999.9;
  ride["mean_le70"]["probability"] = .999; ride["weather"]["rain_chance_max_pct"] = 100;
  ride["weather"]["fresh"] = false;
  for (const char *name : {"morning", "afternoon"})
    t.d["forecast"]["sessions"][name]["weather"]["rain_chance_max_pct"] = 100;
  auto s = t.d["history"]["pm25_summary"];
  s["average_ugm3"] = 8888.8; s["lowest_ugm3"] = 7777.7; s["highest_ugm3"] = 9999.9;
  t.d["history"]["fresh"] = false; t.indoor.temperatureC = -99.9; t.indoor.minimumC = -99.9;
  for (unsigned p = 0; p < 3; ++p) {
    t.clips.count = SpeechPlaylist::kCapacity; t.text = "reset required"; t.page(p);
    lacks(t.text, "reset required"); assert(duration(t.clips) < 60);
    printf("Complex Page %u: %.2fs, %zu clips\n", p + 1, duration(t.clips), t.clips.count);
  }
  t.d.clear(); t.d["test_now"] = now; t.c.indoor = nullptr;
  for (unsigned p = 0; p < 3; ++p) { t.page(p); contains(t.text, "unavailable"); }
  SpeechPlaylist clips;
  assert(speechNumberPrecision(clips, -1.05f, 2));
  sequence(clips, {C::Minus, C::N1, C::Point, C::N0, C::N5});
  clips = SpeechPlaylist{};
  for (size_t i = 0; i < SpeechPlaylist::kCapacity - 3; ++i) assert(clips.add(C::N9));
  const size_t original = clips.count;
  assert(!speechNumberPrecision(clips, 4.13f, 2)); assert(clips.count == original);
  for (size_t i = 0; i < original; ++i) assert(clips.clips[i] == C::N9);
}
static void modernForecastSemantics() {
  Test t;
  auto ride = t.d["forecast"]["ride90_210"].as<JsonObject>();
  assert(ridePmRange(ride).available && !ridePmRange(ride).minimumMaximum);
  auto extrema = ride["minimum_maximum"].to<JsonObject>();
  extrema["available"] = true; extrema["minimum_ugm3"] = 101.2;
  extrema["maximum_ugm3"] = 203.4; extrema["resolution_minutes"] = 15;
  extrema["kind"] = "predicted_window_minimum_maximum";
  ride["range_low_ugm3"] = 1; ride["range_high_ugm3"] = 2;
  ride["role"] = "experimental_model_output";
  ride["pm25_ugm3"] = 150.7; ride["reference_pm25_ugm3"] = 999;
  RidePmRange range = ridePmRange(ride);
  assert(range.available && range.minimumMaximum && range.low == float(101.2) && range.high == float(203.4));
  t.page(1); contains(t.text, "PM2.5 forecast range 101.2 to 203.4.");
  lacks(t.text, "999"); lacks(t.text, "Experimental");
  // Extrema/probability heads remain available while the mean model warms up.
  ride["available"] = false; ride["pm25_ugm3"] = nullptr;
  assert(ridePmRange(ride).available); t.page(1);
  contains(t.text, "PM2.5 forecast range 101.2 to 203.4.");
  contains(t.text, "seventy or less is 4.8 percent.");
  ride["available"] = true; ride["pm25_ugm3"] = 150.7;
  for (unsigned failure = 0; failure < 7; ++failure) {
    extrema["available"] = true; extrema["minimum_ugm3"] = 101.2;
    extrema["maximum_ugm3"] = 203.4; extrema["resolution_minutes"] = 15;
    extrema["kind"] = "predicted_window_minimum_maximum";
    if (failure == 0) extrema["available"] = false;
    if (failure == 1) extrema["minimum_ugm3"] = nullptr;
    if (failure == 2) extrema["minimum_ugm3"] = 300;
    if (failure == 3) extrema["maximum_ugm3"] = NAN;
    if (failure == 4) extrema["resolution_minutes"] = 30;
    if (failure == 5) extrema["kind"] = "empirical_q10_q90";
    if (failure == 6) extrema["maximum_ugm3"] = 10000;
    assert(!ridePmRange(ride).available); t.page(1);
    contains(t.text, "PM2.5 forecast 150.7.");
    assert(!count(t.clips, C::PMForecastRange));
  }
  ride["minimum_maximum"] = nullptr;
  assert(!ridePmRange(ride).available); t.page(1); contains(t.text, "PM2.5 forecast 150.7.");
  ride["pm25_ugm3"] = nullptr; t.page(1);
  sequence(t.clips, {C::PMForecastRange, C::UnavailableValue}); lacks(t.text, "999");
  ride.remove("minimum_maximum");
  range = ridePmRange(ride);
  assert(range.available && !range.minimumMaximum && range.low == 1 && range.high == 2);

  t.reset(); t.d["forecast"]["near90"]["reference_pm25_ugm3"] = 42.3;
  t.d["forecast"]["near90"]["arrival_change"]["reference_ugm3"] = 42.3;
  t.page(0); contains(t.text, "Ninety minute forecast PM2.5 888.8."); lacks(t.text, "42.3");
  t.d["forecast"]["near90"]["role"] = "persistence_anchor";
  t.page(0); contains(t.text, "Ninety minute forecast PM2.5 unavailable.");
  assert(count(t.clips, C::NoChangeOnArrival) == 1);
  t.d["forecast"]["near90"]["role"] = "experimental_model_output";
  t.d["forecast"]["near90"]["available"] = false; t.d["forecast"]["near90"]["pm25_ugm3"] = nullptr;
  t.page(0); contains(t.text, "Ninety minute forecast PM2.5 unavailable.");
  assert(count(t.clips, C::NoChangeOnArrival) == 1); lacks(t.text, "42.3");
  for (const char *role : {"raw_model_output", "experimental_model_output", "experimental_window_mean"}) {
    for (const char *name : {"morning", "afternoon"}) {
      auto pm = t.d["forecast"]["sessions"][name]["pm"];
      pm["role"] = role; pm["qualified"] = false; pm["used_for_decision"] = false;
      pm["reference_pm25_ugm3"] = name[0] == 'm' ? 500 : 1;
      assert(forecastModelPointAvailable(pm));
    }
    t.page(1); contains(t.text, "Tomorrow morning has lower forecast PM2.5 than This afternoon.");
  }
  t.d["forecast"]["sessions"]["morning"]["pm"]["role"] = "persistence_anchor";
  t.page(1); assert(count(t.clips, C::ComparisonUnavailable) == 1);
  t.d["forecast"]["sessions"]["morning"]["pm"]["role"] = "experimental_model_output";
  t.d["forecast"]["sessions"]["morning"]["pm"]["pm25_ugm3"] = -1;
  t.page(1); assert(count(t.clips, C::ComparisonUnavailable) == 1);

  t.reset(); auto event = t.d["forecast"]["near90"]["first20"];
  event["direction"] = "unresolved"; event["diagnostic_direction"] = "rise";
  event["qualification"]["operational_use_eligible"] = false;
  t.page(0); assert(count(t.clips, C::NoChangeOnArrival) == 1);
  assert(first20ModelCall(event) == First20Call::Rise);
  event["rise_probability"] = .1; event["drop_probability"] = .8; event["none_probability"] = .1;
  event["diagnostic_direction"] = "drop";
  t.page(0); assert(count(t.clips, C::NoChangeOnArrival) == 1);
  assert(first20ModelCall(event) == First20Call::Drop);
  event["rise_probability"] = .1; event["drop_probability"] = .2; event["none_probability"] = .7;
  event["diagnostic_direction"] = "unresolved";
  t.page(0); assert(count(t.clips, C::NoChangeOnArrival) == 1);
  assert(first20ModelNoChangeMostLikely(event));
  event["rise_probability"] = .4; event["drop_probability"] = .4; event["none_probability"] = .2;
  t.page(0); assert(count(t.clips, C::NoChangeOnArrival) == 1);
  assert(!first20ModelNoChangeMostLikely(event));
  event["diagnostic_direction"] = "rise";
  assert(first20ModelCall(event) == First20Call::Invalid);
  event["diagnostic_direction"] = nullptr;
  assert(first20ModelCall(event) == First20Call::Invalid);
  event["diagnostic_direction"] = "unresolved"; event["none_probability"] = .3;
  assert(first20ModelCall(event) == First20Call::Invalid);

  t.reset(); auto weather = t.d["forecast"]["ride90_210"]["weather"].as<JsonObject>();
  assert(windowWeatherCoverageAvailable(weather)); // Genuine legacy payload.
  weather["coverage_verified"] = true;
  assert(!windowWeatherCoverageAvailable(weather)); t.page(1); contains(t.text, "Rain chance unavailable.");
  weather["coverage"]["complete"] = true;
  assert(windowWeatherCoverageAvailable(weather)); t.page(1); contains(t.text, "Rain chance 12 percent.");
  weather["coverage"]["complete"] = false;
  assert(!windowWeatherCoverageAvailable(weather));
  weather["coverage"]["complete"] = true; weather["coverage_verified"] = false;
  assert(!windowWeatherCoverageAvailable(weather));
  weather.remove("coverage_verified"); assert(!windowWeatherCoverageAvailable(weather));
  weather["coverage_verified"] = "true"; assert(!windowWeatherCoverageAvailable(weather));
  weather["coverage_verified"] = true; weather["coverage"] = nullptr;
  assert(!windowWeatherCoverageAvailable(weather));
  for (const char *name : {"morning", "afternoon"}) {
    auto sessionWeather = t.d["forecast"]["sessions"][name]["weather"];
    sessionWeather["rain_chance_max_pct"] = 100;
    sessionWeather["coverage_verified"] = true;
    sessionWeather["coverage"]["complete"] = false;
  }
  t.page(1); assert(!count(t.clips, C::Warning));
  t.d["forecast"]["sessions"]["afternoon"]["weather"]["coverage"]["complete"] = true;
  t.page(1); assert(count(t.clips, C::Warning) == 1);
  contains(t.text, "Warning This afternoon has a high rain chance of 100 percent.");
}
static void arrivalProbabilityReadout() {
  Test t;
  const C headlines[] = {C::RiseOnArrival, C::FallOnArrival, C::NoChangeOnArrival};
  const char *outcomes[] = {"rise20", "fall20", "within20"};
  const char *labels[] = {"Rise on arrival", "Fall on arrival", "No change on arrival"};
  const char *sentences[] = {
      "Rise on arrival, 78.8 percent.", "Fall on arrival, 78.8 percent.",
      "No change on arrival, 78.8 percent."};
  for (unsigned winner = 0; winner < 3; ++winner) {
    t.reset(); JsonObject a = t.d["forecast"]["near90"]["arrival_change"].as<JsonObject>();
    a["rise20"] = winner == 0 ? .788 : .106;
    a["fall20"] = winner == 1 ? .788 : .106;
    a["within20"] = winner == 2 ? .788 : .106;
    a["outcome"] = outcomes[winner]; a["outcome_probability"] = .788;
    a["display_text"] = labels[winner];
    t.page(0); contains(t.text, sentences[winner]);
    sequence(t.clips, {headlines[winner], C::N70, C::N8, C::Point, C::N8, C::Percent, C::RainInAnHour});
    assert(count(t.clips, C::Percent) == 3); // Two hourly rain values plus endpoint winner.
    for (unsigned other = 0; other < 3; ++other) assert(count(t.clips, headlines[other]) == (other == winner ? 1 : 0));
  }
  {
    t.reset(); JsonObject a = t.d["forecast"]["near90"]["arrival_change"].as<JsonObject>();
    a["rise20"] = .7884; a["fall20"] = .1; a["within20"] = .1116;
    a["outcome"] = "rise20"; a["outcome_probability"] = .7884; a["display_text"] = "Rise on arrival";
    t.page(0); contains(t.text, "Rise on arrival, 78.8 percent.");
    sequence(t.clips, {C::RiseOnArrival, C::N70, C::N8, C::Point, C::N8, C::Percent});
    a["rise20"] = .7886; a["within20"] = .1114; a["outcome_probability"] = .7886;
    t.page(0); contains(t.text, "Rise on arrival, 78.9 percent.");
    sequence(t.clips, {C::RiseOnArrival, C::N70, C::N8, C::Point, C::N9, C::Percent});
  }
  // Existing stale behavior retains an explicit notice and the same model score.
  t.d["forecast"]["fresh"] = false; t.page(0);
  assert(count(t.clips, C::Old) == 1); contains(t.text, "Rise on arrival, 78.9 percent.");
  for (unsigned failure = 0; failure < 22; ++failure) {
    t.reset(); JsonObject near = t.d["forecast"]["near90"].as<JsonObject>();
    JsonObject invalid = near["arrival_change"].as<JsonObject>();
    if (failure == 0) near["arrival_change"] = nullptr;
    if (failure == 1) near.remove("arrival_change");
    if (failure == 2) invalid["available"] = false;
    if (failure == 3) invalid["available"] = "true";
    if (failure == 4) invalid.remove("within20");
    if (failure == 5) invalid["within20"] = .5; // Native distribution is incoherent.
    if (failure == 6) invalid["outcome"] = "rise20"; // Winner disagrees with the native heads.
    if (failure == 7) invalid["outcome_probability"] = nullptr;
    if (failure == 8) invalid["outcome_probability"] = NAN;
    if (failure == 9) invalid["outcome_probability"] = 1.2;
    if (failure == 10) invalid["reference_ugm3"] = NAN;
    if (failure == 11) invalid["issued_epoch"] = now - 1;
    if (failure == 12) invalid["arrival_epoch"] = now + 5399;
    if (failure == 13) invalid["fresh_reference_epoch"] = now - 241;
    if (failure == 14) invalid["fresh_reference_epoch"] = now + 1;
    if (failure == 15) invalid["display_text"] = "";
    if (failure == 16) near["target_epoch"] = now;
    if (failure == 17) t.d["forecast"]["issued_epoch"] = now - 601;
    if (failure == 18) invalid["fall40"] = .2; // A >=40 tail cannot exceed its >=20 tail.
    if (failure == 19) near["reference_pm25_ugm3"] = 42.3;
    if (failure == 20) invalid["rise20"] = -1;
    if (failure == 21) invalid["fall20"] = INFINITY;
    t.page(0);
    for (C headline : headlines) assert(!count(t.clips, headline));
    assert(!count(t.clips, C::ArrivalOutcomeUncertain));
    assert(count(t.clips, C::Percent) == 2); lacks(t.text, "on arrival,");
    sequence(t.clips, {C::UnavailableValue, C::RainInAnHour});
    if (failure != 16 && failure != 17) contains(t.text, "Ninety minute forecast PM2.5 888.8.");
  }
  {
    t.reset(); JsonObject a = t.d["forecast"]["near90"]["arrival_change"].as<JsonObject>();
    a["fall20"] = .4; a["rise20"] = .4; a["within20"] = .2;
    a["outcome"] = nullptr; a["outcome_probability"] = nullptr; a["display_text"] = "Arrival outcome uncertain";
    t.page(0); contains(t.text, "Arrival outcome uncertain. Rain chance in an hour 2 percent.");
    sequence(t.clips, {C::ArrivalOutcomeUncertain, C::RainInAnHour});
    assert(count(t.clips, C::Percent) == 2);
    // Exact raw equality defines a tie, not an epsilon around a unique winner.
    a["fall20"] = .4000000000001; a["within20"] = .1999999999999;
    a["outcome"] = "fall20"; a["outcome_probability"] = .4000000000001; a["display_text"] = "Fall on arrival";
    t.page(0); contains(t.text, "Fall on arrival, 40.0 percent.");
    assert(!count(t.clips, C::ArrivalOutcomeUncertain));
    sequence(t.clips, {C::FallOnArrival, C::N40, C::Point, C::N0, C::Percent});
    a["fall20"] = 0; a["fall40"] = 0; a["rise20"] = 0; a["rise40"] = 0; a["within20"] = 1;
    a["outcome"] = "within20"; a["outcome_probability"] = 1; a["display_text"] = "No change on arrival";
    t.page(0); contains(t.text, "No change on arrival, 100.0 percent.");
    sequence(t.clips, {C::NoChangeOnArrival, C::N1, C::Hundred, C::Point, C::N0, C::Percent});
  }
  // A changing actual observation never rewrites the fixed issued reference.
  t.reset(); t.d["current"]["pm25_ugm3"] = 333.3; t.page(0);
  contains(t.text, "Current outdoor PM2.5 333.3."); contains(t.text, "No change on arrival, 78.8 percent.");
  t.d["forecast"]["near90"]["first20"] = nullptr; t.page(0);
  contains(t.text, "No change on arrival, 78.8 percent.");
  // A null legacy point reference can accompany an independent native head
  // while the point model warms up. Only the native reference is mandatory.
  {
    t.reset(); JsonObject near = t.d["forecast"]["near90"].as<JsonObject>();
    near["available"] = false; near["pm25_ugm3"] = nullptr;
    near["reference_pm25_ugm3"] = nullptr;
    assert(decodeArrivalChange(near, now, now).available);
    t.page(0); contains(t.text, "Ninety minute forecast PM2.5 unavailable.");
    contains(t.text, "No change on arrival, 78.8 percent.");
    sequence(t.clips, {C::NoChangeOnArrival, C::N70, C::N8, C::Point, C::N8, C::Percent});
    near["arrival_change"]["reference_ugm3"] = nullptr;
    t.page(0); assert(!count(t.clips, C::NoChangeOnArrival));
    sequence(t.clips, {C::UnavailableValue, C::RainInAnHour});
  }
  // Parent's 600-second expiry owns availability, even with a valid native head.
  for (uint32_t parentAge : {uint32_t(600), uint32_t(601)}) {
    t.reset(); const uint32_t issued = now - parentAge;
    t.d["forecast"]["issued_epoch"] = issued;
    JsonObject near = t.d["forecast"]["near90"].as<JsonObject>();
    near["target_epoch"] = issued + 5400;
    JsonObject a = near["arrival_change"].as<JsonObject>();
    a["issued_epoch"] = issued; a["arrival_epoch"] = issued + 5400;
    a["fresh_reference_epoch"] = issued - 240;
    assert(decodeArrivalChange(near, issued, now).available);
    t.page(0);
    assert(count(t.clips, C::NoChangeOnArrival) == (parentAge == 600 ? 1 : 0));
    assert(count(t.clips, C::Old) == (parentAge == 600 ? 1 : 0));
  }
}
int main() {
  routingAndSnapshot(); demoReadoutDisclosure(); rainIntervalsAndValues(); observedStatusReadout(); first20Semantics();
  sessionComparisonAndWarnings(); rangeHistoryIndoorBattery(); capacityDurationAndReset(); modernForecastSemantics(); arrivalProbabilityReadout();
  puts("PASS: strict demo provenance, nullable point reference independence, observed status body without narration prefix, freshness/expiry/type guards and snapshot; native arrival winner probability, independent legacy crossing decoder, ties, rounding, malformed/stale/expired suppression, modern min/max and coverage");
}
"""


OVERFLOW_HARNESS = r"""
#include "PageReadout.h"
#include <cassert>
#include <cstdio>
uint32_t dashboardNow(const DashboardContext &) { return 1800000000; }
int main() {
  static_assert(SpeechPlaylist::kCapacity == 4, "Controlled playlist-capacity fault");
  JsonDocument document;
  DashboardContext context;
  context.weather = &document;
  SpeechPlaylist clips;
  String text;
  for (unsigned page = 0; page < 3; ++page) {
    context.page = page;
    clips.count = SpeechPlaylist::kCapacity;
    text = "previous transcript";
    assert(!buildCurrentPageReadout(context, clips, text));
    assert(clips.count == 0 && text.str().empty());
  }
  // A valid 0.0 PM2.5 uses all four slots. The subsequent status must fail
  // transactionally instead of retaining the preceding value or old text.
  context.page = 0;
  document["current"]["available"] = true;
  document["current"]["fresh"] = true;
  document["current"]["observed_epoch"] = uint32_t(1800000000);
  document["current"]["pm25_ugm3"] = 0;
  for (const char *status : {"Latest sensor reading", "Observed: Particle rebound may be starting", "unknown"}) {
    document["current"]["display_text"] = status;
    text = "previous transcript";
    assert(!buildCurrentPageReadout(context, clips, text));
    assert(clips.count == 0 && text.str().empty());
  }
  context.weather = nullptr;
  assert(buildCurrentPageReadout(context, clips, text));
  assert(clips.count == 1 && text.str() == "unavailable.");
  puts("PASS: forced capacity overflow clears both transcript and playlist on all pages");
}
"""


def extract_function(source, signature):
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 1
    end = brace + 1
    while depth:
        if source[end] == "{":
            depth += 1
        elif source[end] == "}":
            depth -= 1
        end += 1
    return source[start:end]


def main():
    firmware = Path(__file__).resolve().parents[1]
    package = firmware.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", required=True)
    parser.add_argument("--cxx-arg", action="append", default=[])
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--arduino-json-src", type=Path,
                        default=package / ".state/arduino-user/libraries/ArduinoJson/src",
                        help="ArduinoJson 7.4.3 src directory")
    parser.add_argument("--esp32-core-dir", type=Path,
                        default=package / ".state/arduino-data/packages/esp32/hardware/esp32/3.3.12/cores/esp32",
                        help="Arduino-ESP32 3.3.12 cores/esp32 directory")
    parser.add_argument("--speech-manifest", type=Path,
                        default=package / "build/speech/manifest.json",
                        help="Generated clip manifest with ID and sample_count for each clip")
    args = parser.parse_args()
    arduino_json = args.arduino_json_src.resolve()
    core = args.esp32_core_dir.resolve()
    manifest_path = args.speech_manifest.resolve()
    for required in (arduino_json / "ArduinoJson.h", core / "stdlib_noniso.c", manifest_path):
        if not required.is_file():
            parser.error(f"Required test input not found: {required}")
    build = args.build_dir.resolve()
    build.mkdir(parents=True, exist_ok=True)
    dtostrf = extract_function((core / "stdlib_noniso.c").read_text(), "char *dtostrf(")
    (build / "Arduino.h").write_text(ARDUINO.replace("@@DTOSTRF@@", "inline " + dtostrf))
    (build / "U8g2lib.h").write_text("#pragma once\nclass U8G2 {};\n")
    production = (firmware / "Dashboard.cpp").read_text(encoding="utf-8")
    helpers = ['#include "Dashboard.h"']
    for signature in ("uint32_t ageSeconds(", "bool dashboardCurrentAvailable(",
                      "bool dashboardForecastAvailable(", "HistoryPmSummary dashboardHistoryPmSummary("):
        helpers.append(extract_function(production, signature))
    helper_source = build / "dashboard_helpers.cpp"
    helper_source.write_text("\n\n".join(helpers), encoding="utf-8")
    test_source = build / "page_readout_test.cpp"
    # Derive duration from the actual immutable clip manifest, not word count.
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    enum = (firmware / "SpeechClips.h").read_text(encoding="utf-8").split("enum class SpeechClip", 1)[1]
    ids = re.findall(r"^\s+(\w+),", enum.split("Count", 1)[0], re.MULTILINE)
    clips = {clip["id"]: clip for clip in manifest["clips"]}
    assert len(ids) == manifest["clip_count"] and all(identifier in clips for identifier in ids)
    # Audio's finite status wording must describe the same observation as the
    # transcript, including the confirmed median threshold and concentration unit.
    status_phrases = {
        "Observed": "Observed",
        "ObservedFastRise": "Fast P M two point five rise detected",
        "ObservedFastRebound": "Fast P M two point five rebound detected",
        "ObservedReboundStarting": "Particle rebound may be starting",
        "ObservedRecentMedians35": "Recent P M two point five medians at or below thirty five micrograms per cubic metre",
        "ObservedMoistCoolingClearing": "Moist cooling particle clearing forming",
        "ObservedDryClearing": "Dry clearing forming",
        "ObservedReductionForming": "P M two point five reduction forming",
        "ObservedRapidReduction": "Rapid P M two point five reduction detected",
        "LatestSensorReading": "Latest sensor reading",
        "ObservedStatusUnavailable": "Observed status unavailable",
    }
    for identifier, phrase in status_phrases.items():
        assert clips[identifier]["phrase"] == phrase, (identifier, clips[identifier]["phrase"], phrase)
        assert clips[identifier]["sample_count"] > 0
    sample_counts = ", ".join(str(clips[identifier]["sample_count"]) for identifier in ids)
    test_source.write_text(HARNESS.replace("@@CLIP_SAMPLES@@", sample_counts), encoding="utf-8")
    suffix = ".exe" if sys.platform == "win32" else ""
    executable = build / ("page_readout_test" + suffix)
    command = [args.cxx, *args.cxx_arg, "-std=c++17", "-Wall", "-Wextra", "-Werror",
               "-I", str(build), "-I", str(firmware), "-I", str(arduino_json),
               str(test_source), str(firmware / "PageReadout.cpp"), str(helper_source),
               "-o", str(executable)]
    subprocess.run(command, check=True)
    subprocess.run([str(executable)], check=True)

    # Fault injection changes only a generated header's capacity. PageReadout's
    # production body remains verbatim and must return neither partial output.
    overflow = build / "overflow"
    overflow.mkdir(exist_ok=True)
    header = (firmware / "WeatherReadout.h").read_text(encoding="utf-8")
    header, changes = re.subn(r"kCapacity = \d+;", "kCapacity = 4;", header)
    assert changes == 1
    (overflow / "WeatherReadout.h").write_text(header, encoding="utf-8")
    for filename in ("PageReadout.h", "PageReadout.cpp"):
        (overflow / filename).write_text((firmware / filename).read_text(encoding="utf-8"), encoding="utf-8")
    overflow_source = overflow / "page_readout_overflow_test.cpp"
    overflow_source.write_text(OVERFLOW_HARNESS, encoding="utf-8")
    overflow_executable = overflow / ("page_readout_overflow_test" + suffix)
    command = [args.cxx, *args.cxx_arg, "-std=c++17", "-Wall", "-Wextra", "-Werror",
               "-I", str(overflow), "-I", str(build), "-I", str(firmware), "-I", str(arduino_json),
               str(overflow_source), str(overflow / "PageReadout.cpp"), str(helper_source),
               "-o", str(overflow_executable)]
    subprocess.run(command, check=True)
    subprocess.run([str(overflow_executable)], check=True)


if __name__ == "__main__":
    main()
