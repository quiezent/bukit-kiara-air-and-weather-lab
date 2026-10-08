#pragma once
#include <stdint.h>

// ESP-SR returns zero when its feed ring is full. That frame was not queued,
// although BSS has already processed it, so drop it and keep real-time capture.
// Repeated zero returns must still become a fault if the consumer stays stuck.
class VoiceFeedHealth {
 public:
  enum class Result { Accepted, Backpressure, Fault };
  static constexpr uint32_t kFailureMs = 3000;

  explicit VoiceFeedHealth(uint32_t startedMs) : lastSuccessMs_(startedMs) {}

  Result observe(int feedResult, uint32_t nowMs) {
    if (feedResult > 0) {
      lastSuccessMs_ = nowMs;
      return Result::Accepted;
    }
    if (feedResult < 0 || uint32_t(nowMs - lastSuccessMs_) >= kFailureMs) {
      return Result::Fault;
    }
    return Result::Backpressure;
  }

 private:
  uint32_t lastSuccessMs_;
};
