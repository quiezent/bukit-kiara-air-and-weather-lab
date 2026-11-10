#pragma once

#include "BoardSpeaker.h"
#include "WeatherReadout.h"
#include <atomic>

// KEY's Read Info: plays a page summary from the flash clips. No microphone,
// wake word or laptop voice server is involved.
class ReadoutPlayer {
 public:
  // start/update run on the loop task; only PCM writes run on the worker.
  bool start(const SpeechPlaylist &playlist);
  // Request cooperative cancellation and mute immediately. Hardware cleanup
  // still waits for the worker's acknowledgement in update().
  bool stop();
  void update();
  bool busy() const { return busy_; }
  bool stopRequested() const { return busy_ && stopRequested_.load(); }
  uint32_t completed() const { return completed_; }
  uint32_t stopped() const { return stopped_; }
  uint32_t plannedSamples() const { return plannedSamples_; }
  uint32_t samplesWritten() const { return samplesWritten_.load(); }
  // Smallest free stack seen at the end of any playback task, in bytes.
  uint32_t stackFreeMin() const { return stackFree_.load(); }
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
  std::atomic<uint32_t> stackFree_{UINT32_MAX};
  char error_[160] = "";
};
