#include "ReadoutPlayer.h"
#include <Arduino.h>
#include <algorithm>
#include <cstdio>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

bool ReadoutPlayer::start(const SpeechPlaylist &playlist, VoiceControl &voice) {
  if (busy_) return false;
  error_[0] = '\0';
  if (!playlist.count || playlist.count > SpeechPlaylist::kCapacity) {
    snprintf(error_, sizeof(error_), "Empty or invalid weather readout");
    return false;
  }
  plannedSamples_ = 0;
  for (size_t i = 0; i < playlist.count; ++i) {
    const SpeechClipData clip = speechClipData(playlist.clips[i]);
    if (!clip.samples || !clip.count) {
      snprintf(error_, sizeof(error_), "Invalid speech clip in page readout");
      return false;
    }
    plannedSamples_ += clip.count;
  }
  if (!voice.pauseForPlayback()) {
    snprintf(error_, sizeof(error_), "Microphone did not pause for speaker playback");
    return false;
  }
  if (!speaker_.begin()) {
    snprintf(error_, sizeof(error_), "%s", speaker_.lastError());
    speaker_.end();
    voice.resumeAfterPlayback();
    return false;
  }
  playlist_ = playlist;
  samplesWritten_.store(0);
  done_.store(false);
  failed_.store(false);
  stopRequested_.store(false);
  busy_ = true;
  if (xTaskCreatePinnedToCore(play, "weather-speech", 4096, this, 2, nullptr, 1) != pdPASS) {
    snprintf(error_, sizeof(error_), "Could not start weather readout task");
    speaker_.end();
    voice.resumeAfterPlayback();
    busy_ = false;
    return false;
  }
  return true;
}

bool ReadoutPlayer::stop() {
  if (!busy_) return false;
  if (stopRequested_.load()) return true;
  if (done_.load()) return false;  // Already finished naturally; await cleanup.
  stopRequested_.store(true);
  speaker_.mute();
  return true;
}

void ReadoutPlayer::play(void *argument) {
  auto *self = static_cast<ReadoutPlayer *>(argument);
  for (size_t i = 0; i < self->playlist_.count && !self->failed_.load()
      && !self->stopRequested_.load(); ++i) {
    const SpeechClipData clip = speechClipData(self->playlist_.clips[i]);
    if (!clip.samples || !clip.count) { self->failed_.store(true); break; }
    size_t offset = 0;
    while (offset < clip.count && !self->stopRequested_.load()) {
      size_t got = self->speaker_.write(clip.samples + offset,
          std::min<size_t>(1600, clip.count - offset), 250);
      if (!got || self->speaker_.lastError()[0]) { self->failed_.store(true); break; }
      offset += got;
      self->samplesWritten_.fetch_add(got);
    }
  }
  // Push silence through the DMA queue, then allow its final buffers to drain.
  if (!self->failed_.load() && !self->stopRequested_.load()) {
    const int16_t silence[256] = {};
    for (int i = 0; i < 8 && !self->stopRequested_.load(); ++i) {
      if (self->speaker_.write(silence, 256, 250) != 256 || self->speaker_.lastError()[0]) {
        self->failed_.store(true);
        break;
      }
    }
  }
  if (!self->stopRequested_.load()) vTaskDelay(pdMS_TO_TICKS(120));
  self->done_.store(true);
  vTaskDelete(nullptr);
}

void ReadoutPlayer::update(VoiceControl &voice) {
  if (!busy_ || !done_.load()) return;
  if (failed_.load()) snprintf(error_, sizeof(error_), "%s",
      speaker_.lastError()[0] ? speaker_.lastError() : "Speech clip playback failed");
  speaker_.end();
  if (!voice.resumeAfterPlayback()) {
    snprintf(error_, sizeof(error_), "Microphone did not restart after speaker playback");
  }
  if (stopRequested_.load()) ++stopped_;
  else if (!error_[0]) ++completed_;
  busy_ = false;
}
