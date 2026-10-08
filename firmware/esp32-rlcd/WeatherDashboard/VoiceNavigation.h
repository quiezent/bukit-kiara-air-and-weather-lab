#pragma once
#include <stdint.h>

enum class VoiceCommand : uint8_t {
  Next = 1,
  Overview = 2,
  Sports = 3,
  History = 4,
  Back = 5,
  ReadInfo = 6
};

constexpr uint8_t kDashboardPageCount = 3;

constexpr uint8_t nextDashboardPage(uint8_t page) {
  return (page + 1) % kDashboardPageCount;
}

constexpr uint8_t previousDashboardPage(uint8_t page) {
  return (page + kDashboardPageCount - 1) % kDashboardPageCount;
}

// Recognition accepts both navigation and actions such as speaker readout.
// pageForVoiceCommand deliberately handles only commands that change pages.
inline bool knownVoiceCommand(int commandId) {
  switch (commandId) {
    case int(VoiceCommand::Next):
    case int(VoiceCommand::Overview):
    case int(VoiceCommand::Sports):
    case int(VoiceCommand::History):
    case int(VoiceCommand::Back):
    case int(VoiceCommand::ReadInfo): return true;
    default: return false;
  }
}

inline bool pageForVoiceCommand(int commandId, uint8_t currentPage, uint8_t &page) {
  switch (commandId) {
    case int(VoiceCommand::Next): page = nextDashboardPage(currentPage); return true;
    case int(VoiceCommand::Overview): page = 0; return true;
    case int(VoiceCommand::Sports): page = 1; return true;
    case int(VoiceCommand::History): page = 2; return true;
    case int(VoiceCommand::Back): page = previousDashboardPage(currentPage); return true;
    default: return false;
  }
}

inline const char *voiceCommandName(int commandId) {
  switch (commandId) {
    case int(VoiceCommand::Next): return "next";
    case int(VoiceCommand::Overview): return "overview";
    case int(VoiceCommand::Sports): return "forecast";
    case int(VoiceCommand::History): return "graph";
    case int(VoiceCommand::Back): return "back";
    case int(VoiceCommand::ReadInfo): return "read info";
    default: return "unknown";
  }
}

// Both directions wrap through the same three pages, including at the edges.
static_assert(previousDashboardPage(0) == 2, "Back from Overview opens History");
static_assert(previousDashboardPage(1) == 0, "Back from Sports opens Overview");
static_assert(previousDashboardPage(2) == 1, "Back from History opens Sports");
static_assert(nextDashboardPage(previousDashboardPage(0)) == 0, "Back/next restores Overview");
static_assert(nextDashboardPage(previousDashboardPage(1)) == 1, "Back/next restores Sports");
static_assert(nextDashboardPage(previousDashboardPage(2)) == 2, "Back/next restores History");

inline bool voiceEventIsFresh(uint32_t nowMs, uint32_t recognizedMs, uint32_t maximumAgeMs) {
  return uint32_t(nowMs - recognizedMs) <= maximumAgeMs;
}

// One detection per utterance. Audio still drains while blocked. Durations are
// measured in audio samples, so task scheduling or a millis() wrap cannot shorten
// the required pause. VAD is used only to rearm, never to cut the start of speech.
class VoiceUtteranceGate {
 public:
  VoiceUtteranceGate(uint32_t minimumGapSamples, uint32_t releaseSilenceSamples)
      : gapTarget_(minimumGapSamples), silenceTarget_(releaseSilenceSamples) {}

  bool ready() const { return !blocked_; }

  void block() {
    blocked_ = true;
    gapSamples_ = 0;
    silenceSamples_ = 0;
  }

  void observe(bool speech, uint32_t samples) {
    if (!blocked_) return;
    gapSamples_ = advance(gapSamples_, samples, gapTarget_);
    silenceSamples_ = speech ? 0 : advance(silenceSamples_, samples, silenceTarget_);
    if (gapSamples_ >= gapTarget_ && silenceSamples_ >= silenceTarget_) blocked_ = false;
  }

 private:
  static uint32_t advance(uint32_t value, uint32_t count, uint32_t target) {
    return count >= target - value ? target : value + count;
  }
  uint32_t gapTarget_;
  uint32_t silenceTarget_;
  uint32_t gapSamples_ = 0;
  uint32_t silenceSamples_ = 0;
  bool blocked_ = false;
};
