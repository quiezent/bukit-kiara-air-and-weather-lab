#pragma once

#include <stddef.h>
#include <stdint.h>
#include <driver/i2s_common.h>

// ESP32-S3-RLCD-4.2 ES8311 speaker output, Arduino-ESP32 3.3 / IDF 5.5+.
// begin()/end() run on the Arduino loop task after shared Wire is initialized.
// The microphone must be stopped and released first: it shares the clock pins.
// There is exactly one write() caller, on the playback task. Never call end()
// until that task has completed. This class never starts or stops shared Wire.
class BoardSpeaker {
 public:
  static constexpr uint32_t kSampleRate = 16000;

  BoardSpeaker() = default;
  ~BoardSpeaker();
  BoardSpeaker(const BoardSpeaker&) = delete;
  BoardSpeaker& operator=(const BoardSpeaker&) = delete;

  bool begin();

  // Returns input mono samples accepted, duplicating each into both I2S slots.
  // timeoutMs is a total deadline; zero polls. On a partial stereo frame, its
  // remaining bytes are retained and flushed before accepting any next sample.
  // Retry at source + returned count. An ordinary timeout is not an error.
  // Before end(), write >=100 ms of silence to finish queued audio and the
  // possible partial frame; write() reports DMA acceptance, not acoustic end.
  size_t write(const int16_t* mono, size_t samples, uint32_t timeoutMs);

  void end();
  const char* lastError() const { return error_; }

 private:
  static constexpr size_t kDmaFrames = 128;
  static constexpr size_t kStereoFrameBytes = 2 * sizeof(int16_t);
  static constexpr size_t kDmaBytes = kDmaFrames * kStereoFrameBytes;

  i2s_chan_handle_t tx_ = nullptr;
  // Reserve the unconfigured RX partner too, so an existing I2S0 microphone
  // prevents allocation instead of silently sharing and replacing its clocks.
  i2s_chan_handle_t reservedRx_ = nullptr;
  int16_t stereo_[2 * kDmaFrames] = {};
  uint8_t pending_[kStereoFrameBytes] = {};
  size_t pendingBytes_ = 0;
  size_t dmaWriteOffset_ = 0;
  bool enabled_ = false;
  bool ready_ = false;
  bool codecTouched_ = false;
  char error_[160] = {};

  bool configureCodec();
  bool writeRegister(uint8_t reg, uint8_t value, bool reportFailure = true);
  bool readRegister(uint8_t reg, uint8_t& value);
  bool verifyRegister(uint8_t reg, uint8_t mask, uint8_t expected);
  void powerDownCodec();
  void driverError(const char* operation, int code);
};
