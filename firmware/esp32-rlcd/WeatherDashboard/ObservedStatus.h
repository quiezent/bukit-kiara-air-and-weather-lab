#pragma once

#include <ArduinoJson.h>
#include <cstring>
#include "SpeechClips.h"

struct ObservedStatusPhrase {
  SpeechClip clip = SpeechClip::ObservedStatusUnavailable;
  const char *words = "Observed status unavailable";
  bool observed = false;
};

// The API owns the observation label; this finite vocabulary supplies matching
// recorded speech. Unknown future wording is never added to the transcript
// without audio. Availability and source-clock guards belong to the caller.
inline ObservedStatusPhrase observedStatusPhrase(JsonVariantConst displayText) {
  if (!displayText.is<const char *>()) return {};
  const char *text = displayText.as<const char *>();
  if (!text) return {};
  if (std::strcmp(text, "Latest sensor reading") == 0)
    return {SpeechClip::LatestSensorReading, "Latest sensor reading", false};
  struct Entry {
    const char *text;
    SpeechClip clip;
    const char *body;
  };
  static const Entry phrases[] = {
    {"Observed: Fast PM2.5 rise detected", SpeechClip::ObservedFastRise,
        "Fast PM2.5 rise detected"},
    {"Observed: Fast PM2.5 rebound detected", SpeechClip::ObservedFastRebound,
        "Fast PM2.5 rebound detected"},
    {"Observed: Particle rebound may be starting", SpeechClip::ObservedReboundStarting,
        "Particle rebound may be starting"},
    {"Observed: Recent PM2.5 medians at or below 35 µg/m³", SpeechClip::ObservedRecentMedians35,
        "Recent PM2.5 medians at or below 35 µg/m³"},
    {"Observed: Moist-cooling particle clearing forming", SpeechClip::ObservedMoistCoolingClearing,
        "Moist-cooling particle clearing forming"},
    {"Observed: Dry clearing forming", SpeechClip::ObservedDryClearing,
        "Dry clearing forming"},
    {"Observed: PM2.5 reduction forming", SpeechClip::ObservedReductionForming,
        "PM2.5 reduction forming"},
    {"Observed: Rapid PM2.5 reduction detected", SpeechClip::ObservedRapidReduction,
        "Rapid PM2.5 reduction detected"},
  };
  for (const Entry &phrase : phrases)
    if (std::strcmp(text, phrase.text) == 0) return {phrase.clip, phrase.body, true};
  return {};
}
