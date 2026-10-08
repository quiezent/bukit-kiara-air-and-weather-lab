#pragma once

#include <stddef.h>
#include <stdint.h>
#include <driver/i2s_common.h>

// ESP32-S3-RLCD-4.2 microphone input, Arduino-ESP32 3.3 / ESP-IDF 5.5+.
//
// Call begin() after IndoorSensor.begin(): that component owns Wire and starts
// the shared SDA=13 / SCL=14 bus. This class never calls Wire.begin()/end().
// begin() and end() must run while no read() is in progress. Use one reader task.
// Only begin()/end() touch I2C; capture cannot interrupt a sensor transaction.
//
// The ADC transports four interleaved 16-bit TDM slots. The board schematic
// connects physical microphones to ADC MIC1 and MIC2; ADC MIC3 is the analog
// speaker reference and MIC4 is unused. ES7210 Philips TDM serial order is
// ADC1, ADC3, ADC2, ADC4, so physical mics occupy slots0/2 and reference slot1.
// readInterleaved() returns both physical microphones, never the reference.
// read() remains available for MIC1-only capture. Signed 16 kHz PCM in both.
// See BoardMicrophone.cpp for the exact hardware routing and sources.
class BoardMicrophone {
 public:
  static constexpr uint32_t kSampleRate = 16000;
  static constexpr size_t kMicrophoneChannels = 2;
  static constexpr size_t kMicrophone1Slot = 0;
  static constexpr size_t kMicrophone2Slot = 2;

  BoardMicrophone() = default;
  ~BoardMicrophone();
  BoardMicrophone(const BoardMicrophone&) = delete;
  BoardMicrophone& operator=(const BoardMicrophone&) = delete;

  bool begin();

  // Returns the number of mono samples copied, which may be less than samples.
  // timeout_ms is the total wait budget for this call, not a per-chunk timeout;
  // zero polls without waiting. Incomplete TDM frames survive across calls.
  // An ordinary timeout is not an error: lastError() remains an empty string.
  // On a driver error, any already-copied samples are returned and lastError()
  // is populated. The caller should stop its reader before calling end().
  size_t read(int16_t* destination, size_t samples, uint32_t timeout_ms);

  // Returns the number of frames copied. Each frame occupies two int16_t
  // samples in destination: [physical MIC1, physical MIC2], in that order.
  // Caller allocates frames * kMicrophoneChannels samples. Deadline, partial
  // frames, and error behavior match read(). Switching between APIs consumes
  // one shared stream, without duplicating or losing already staged frames.
  size_t readInterleaved(int16_t* destination, size_t frames, uint32_t timeout_ms);

  void end();
  const char* lastError() const { return error_; }

 private:
  static constexpr size_t kSlots = 4;
  static constexpr size_t kFrameBytes = kSlots * sizeof(int16_t);
  static constexpr size_t kDmaFrames = 128;
  static constexpr size_t kDmaBytes = kDmaFrames * kFrameBytes;
  static constexpr size_t kBufferFrames = 2 * kDmaFrames;
  static constexpr size_t kBufferBytes = kBufferFrames * kFrameBytes;

  i2s_chan_handle_t rx_ = nullptr;
  uint8_t* buffer_ = nullptr;
  size_t buffered_ = 0;
  size_t consumed_ = 0;
  size_t dmaReadOffset_ = 0;
  bool enabled_ = false;
  bool ready_ = false;
  bool codecTouched_ = false;
  char error_[144] = {};

  bool configureCodec();
  size_t readChannels(int16_t* destination, size_t frames, size_t channels,
                      uint32_t timeout_ms);
  bool writeRegister(uint8_t reg, uint8_t value, bool reportFailure = true);
  bool readRegister(uint8_t reg, uint8_t& value);
  bool updateRegister(uint8_t reg, uint8_t mask, uint8_t value);
  bool verifyRegister(uint8_t reg, uint8_t mask, uint8_t expected);
  void powerDownCodec();
  void driverError(const char* operation, int code);
};
