#pragma once
#include <stddef.h>
#include <stdint.h>
#include <string.h>

// Idle audio must not consume MultiNet's duration-limited recognition window.
// Start at VAD onset, replaying the AFE cache before that frame, and retain a
// short silence tail so the recognizer can finish the last word.
class VoiceRecognitionSession {
 public:
  enum class FrameAction { Ignore, Start, Continue };

  explicit VoiceRecognitionSession(uint32_t tailSilenceSamples)
      : tailSilenceSamples_(tailSilenceSamples) {}

  FrameAction observe(bool speech, uint32_t samples) {
    if (!active_) {
      if (!speech) return FrameAction::Ignore;
      active_ = true;
      silenceSamples_ = 0;
      return FrameAction::Start;
    }
    if (speech) {
      silenceSamples_ = 0;
    } else {
      silenceSamples_ += samples >= tailSilenceSamples_ - silenceSamples_
          ? tailSilenceSamples_ - silenceSamples_ : samples;
    }
    return FrameAction::Continue;
  }

  bool finished() const { return active_ && silenceSamples_ >= tailSilenceSamples_; }
  void stop() { active_ = false; silenceSamples_ = 0; }

 private:
  uint32_t tailSilenceSamples_;
  uint32_t silenceSamples_ = 0;
  bool active_ = false;
};

// AFE's VAD cache is a byte-sized prefix, not necessarily a whole MultiNet
// frame. Assemble that prefix and fetched audio without padding or truncation.
// Storage is one caller-owned recognition frame, regardless of cache length.
class VoiceFrameAssembler {
 public:
  VoiceFrameAssembler(int16_t *buffer, size_t frameSamples)
      : buffer_(buffer), frameSamples_(frameSamples) {}

  void reset() { filled_ = 0; }

  template <typename Consume>
  bool append(const int16_t *samples, size_t count, Consume consume) {
    while (count) {
      size_t copied = frameSamples_ - filled_;
      if (copied > count) copied = count;
      memcpy(buffer_ + filled_, samples, copied * sizeof(int16_t));
      filled_ += copied;
      samples += copied;
      count -= copied;
      if (filled_ == frameSamples_) {
        filled_ = 0;
        if (!consume(buffer_)) return false;
      }
    }
    return true;
  }

 private:
  int16_t *buffer_;
  size_t frameSamples_;
  size_t filled_ = 0;
};
