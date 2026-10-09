#pragma once

#include "BoardSpeaker.h"
#include "WeatherReadout.h"
#include "VoiceControl.h"
#include <atomic>

class ReadoutPlayer {
 public:
  // start/update run on the loop task; only PCM writes run on the worker.
  bool start(const SpeechPlaylist &playlist, VoiceControl &voice);
  // Request cooperative cancellation and mute immediately. Hardware cleanup
  // still waits for the worker's acknowledgement in update().
  bool stop();
  void update(VoiceControl &voice);
  bool busy() const { return busy_; }
  bool stopRequested() const { return busy_ && stopRequested_.load(); }
  uint32_t completed() const { return completed_; }
  uint32_t stopped() const { return stopped_; }
  uint32_t samplesWritten() const { return samplesWritten_.load(); }
  uint32_t plannedSamples() const { return plannedSamples_; }
  const char *lastError() const { return error_; }

 private:
  static void play(void *argument);
  BoardSpeaker speaker_;
  SpeechPlaylist playlist_;
  bool busy_ = false;
  uint32_t completed_ = 0;
  uint32_t stopped_ = 0;
  uint32_t plannedSamples_ = 0;
  std::atomic<bool> stopRequested_{false};
  std::atomic<bool> done_{false};
  std::atomic<bool> failed_{false};
  std::atomic<uint32_t> samplesWritten_{0};
  char error_[160] = "";
};
