/*
 * ES7210 register initialization is adapted from esp_codec_dev's es7210.c:
 * SPDX-FileCopyrightText: 2023 Espressif Systems (Shanghai) CO LTD
 * SPDX-License-Identifier: Apache-2.0
 *
 * Modifications: reduced to this board's 16 kHz slave / four-slot capture;
 * Wire control, checked I/O, cleanup, and mono / dual-microphone buffering added.
 * Licensed under the Apache License, Version 2.0. You may obtain a copy at
 * https://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software is
 * distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for permissions and
 * limitations. Preserve the upstream license with redistributed source.
 *
 * Sources (Waveshare's official ESP32-S3-RLCD-4.2 repository):
 * https://github.com/waveshareteam/ESP32-S3-RLCD-4.2
 * 02_Example/Arduino/07_Audio_Test/src/ExternLib/esp_codec_dev/device/
 *   es7210/es7210.c (blob 023efae211d19b1e999e7cc6f84712a5904bda48)
 * 02_Example/ESP-IDF/07_Audio_Test/components/ExternLib/codec_board/board_cfg.txt
 * 02_Example/XiaoZhi/XiaoZhiCode_V2.1.0/main/audio/codecs/box_audio_codec.cc
 * 02_Example/XiaoZhi/XiaoZhiCode_V2.1.0/main/boards/waveshare-s3-rlcd-4.2/config.h
 * https://files.waveshare.com/wiki/ESP32-S3-RLCD-4.2/
 *   ESP32-S3-RLCD-4.2-schematic.pdf
 * https://files.waveshare.com/wiki/common/ES7210_DS.pdf
 *   Revision21.0 page8 Figure2e: Philips TDM ADC order 1,3,2,4.
 * https://github.com/espressif/esp-audio-dev/blob/main/esp_codec_dev/device/
 *   es7210/es7210.c (order_info: TDM_PHILIPS 4 -> CHANNEL_MAP_4CH(1,3,2,4)).
 *
 * Physical routing follows this board's schematic: MIC1_P/N -> ES7210
 * MIC1P/N pins16/15; physical
 * MIC2_P/N -> MIC2P/N pins19/20. Speaker OUTP/OUTN -> the AEC attenuation network
 * -> ADC_MIC3_P/N -> MIC3P/N pins31/32. MIC4 is unused. Espressif's ES7210
 * channel-gain function maps ADC channel0/1/2 to MIC1/2/3 respectively, but
 * Philips TDM wire order is MIC1,MIC3,MIC2,MIC4 (datasheet Figure2e). Therefore
 * slots0+2 are the physical microphones; slot1 is the speaker reference.
 * Keep all four transport slots active and extract only the requested mics.
 * No channel XOR, guessed L/R swap, averaging, or byte reversal is performed.
 */

#include "BoardMicrophone.h"

#include <Arduino.h>
#include <Wire.h>
#include <driver/i2s_tdm.h>
#include <esp_err.h>
#include <esp_heap_caps.h>
#include <esp_idf_version.h>
#include <esp_timer.h>
#include <stdio.h>
#include <string.h>

#if !defined(CONFIG_IDF_TARGET_ESP32S3) || !CONFIG_IDF_TARGET_ESP32S3
#error "BoardMicrophone requires an ESP32-S3 target."
#endif
#if ESP_IDF_VERSION < ESP_IDF_VERSION_VAL(5, 5, 0)
#error "BoardMicrophone requires Arduino-ESP32 3.3 / ESP-IDF 5.5 or newer."
#endif

namespace {
constexpr uint8_t kAdcAddress = 0x40;  // Wire uses 7-bit, not codec_dev's 0x80.
constexpr gpio_num_t kMclk = GPIO_NUM_16;
constexpr gpio_num_t kBclk = GPIO_NUM_9;
constexpr gpio_num_t kWordSelect = GPIO_NUM_45;
constexpr gpio_num_t kDataIn = GPIO_NUM_10;
constexpr uint8_t kAmplifierEnable = 46;
constexpr uint8_t kGain30dB = 0x0a;

struct RegisterValue {
  uint8_t reg;
  uint8_t value;
};
}  // namespace

BoardMicrophone::~BoardMicrophone() { end(); }

void BoardMicrophone::driverError(const char* operation, int code) {
  snprintf(error_, sizeof(error_), "%s: %s (0x%x)", operation,
           esp_err_to_name(static_cast<esp_err_t>(code)), code);
}

bool BoardMicrophone::writeRegister(uint8_t reg, uint8_t value,
                                    bool reportFailure) {
  Wire.beginTransmission(kAdcAddress);
  const bool queued = Wire.write(reg) == 1 && Wire.write(value) == 1;
  const uint8_t result = Wire.endTransmission(true);
  if (queued && result == 0) return true;
  if (reportFailure) {
    snprintf(error_, sizeof(error_), "ES7210 write reg 0x%02x failed (I2C %u)",
             reg, unsigned(result));
  }
  return false;
}

bool BoardMicrophone::readRegister(uint8_t reg, uint8_t& value) {
  Wire.beginTransmission(kAdcAddress);
  const bool queued = Wire.write(reg) == 1;
  const uint8_t result = Wire.endTransmission(false);
  if (!queued || result != 0) {
    snprintf(error_, sizeof(error_), "ES7210 select reg 0x%02x failed (I2C %u)",
             reg, unsigned(result));
    return false;
  }
  if (Wire.requestFrom(kAdcAddress, size_t(1), true) != 1 || !Wire.available()) {
    snprintf(error_, sizeof(error_), "ES7210 read reg 0x%02x failed", reg);
    return false;
  }
  value = static_cast<uint8_t>(Wire.read());
  return true;
}

bool BoardMicrophone::updateRegister(uint8_t reg, uint8_t mask, uint8_t value) {
  uint8_t current;
  return readRegister(reg, current) &&
         writeRegister(reg, (current & uint8_t(~mask)) | (value & mask));
}

bool BoardMicrophone::verifyRegister(uint8_t reg, uint8_t mask,
                                     uint8_t expected) {
  uint8_t actual;
  if (!readRegister(reg, actual)) return false;
  if ((actual & mask) == (expected & mask)) return true;
  snprintf(error_, sizeof(error_),
           "ES7210 reg 0x%02x readback 0x%02x, expected 0x%02x (mask 0x%02x)",
           reg, actual, expected, mask);
  return false;
}

bool BoardMicrophone::configureCodec() {
  codecTouched_ = true;
  // Reset, then Espressif's slave-mode analog / clock / high-pass setup.
  // In slave mode the host supplies MCLK=256*16000=4.096 MHz; the vendor
  // driver uses OSR 0x20 and MAINCLK 0xc1 without its master divider table.
  const RegisterValue setup[] = {
      {0x00, 0xff}, {0x00, 0x41}, {0x01, 0x3f},
      {0x09, 0x30}, {0x0a, 0x30},
      {0x23, 0x2a}, {0x22, 0x0a}, {0x20, 0x0a}, {0x21, 0x2a},
      {0x40, 0x43}, {0x41, 0x70}, {0x42, 0x70},
      {0x07, 0x20}, {0x02, 0xc1},
  };
  for (const auto& setting : setup) {
    if (!writeRegister(setting.reg, setting.value)) return false;
  }
  if (!updateRegister(0x08, 0x01, 0x00)) return false;  // ES7210 is slave.

  // Preserve all four ADC positions in the TDM stream. Enable ADC clocks and
  // both analog pairs, with the vendor's 30 dB starting microphone gain.
  for (uint8_t reg = 0x43; reg <= 0x46; ++reg) {
    if (!updateRegister(reg, 0x10, 0x00)) return false;
  }
  if (!writeRegister(0x4b, 0xff) || !writeRegister(0x4c, 0xff) ||
      !updateRegister(0x01, 0x1f, 0x00) ||
      !writeRegister(0x4b, 0x00) || !writeRegister(0x4c, 0x00)) {
    return false;
  }
  for (uint8_t reg = 0x43; reg <= 0x46; ++reg) {
    if (!updateRegister(reg, 0x1f, 0x10 | kGain30dB)) return false;
  }
  if (!writeRegister(0x12, 0x02) ||             // Four-channel TDM on SDOUT1.
      !updateRegister(0x11, 0xe3, 0x60)) {     // 16-bit, Philips I2S format.
    return false;
  }

  // Espressif es7210_start(): power up ADCs, then synchronize the state machine.
  const RegisterValue start[] = {
      {0x01, 0x20}, {0x06, 0x00}, {0x40, 0x43},
      {0x47, 0x08}, {0x48, 0x08}, {0x49, 0x08}, {0x4a, 0x08},
      {0x4b, 0x00}, {0x4c, 0x00}, {0x40, 0x43},
      {0x00, 0x71}, {0x00, 0x41},
  };
  for (const auto& setting : start) {
    if (!writeRegister(setting.reg, setting.value)) return false;
  }
  if (!updateRegister(0x14, 0x03, 0x00) ||
      !updateRegister(0x15, 0x03, 0x00)) return false;  // Unmute both ADC pairs.

  return verifyRegister(0x08, 0x01, 0x00) &&
         verifyRegister(0x11, 0xe3, 0x60) &&
         verifyRegister(0x12, 0x03, 0x02) &&
         verifyRegister(0x43, 0x1f, 0x10 | kGain30dB);
}

bool BoardMicrophone::begin() {
  end();
  error_[0] = '\0';
  if (rx_) {
    snprintf(error_, sizeof(error_), "Previous I2S channel could not be released");
    return false;
  }

  // This feature captures only. Keep the physical speaker amplifier quiet.
  digitalWrite(kAmplifierEnable, LOW);
  pinMode(kAmplifierEnable, OUTPUT);

  Wire.beginTransmission(kAdcAddress);
  const uint8_t probe = Wire.endTransmission(true);
  if (probe != 0) {
    snprintf(error_, sizeof(error_),
             "ES7210 at 0x40 unavailable (I2C %u); initialize IndoorSensor first",
             unsigned(probe));
    return false;
  }

  // This is a copy/staging buffer, not a DMA descriptor. Keep it in internal
  // byte-addressable RAM. The I2S driver allocates its own DMA-capable buffers.
  buffer_ = static_cast<uint8_t*>(
      heap_caps_malloc(kBufferBytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
  if (!buffer_) {
    snprintf(error_, sizeof(error_), "Not enough internal RAM for microphone buffer");
    return false;
  }

  i2s_chan_config_t channel = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
  channel.dma_desc_num = 8;
  channel.dma_frame_num = kDmaFrames;  // 8*128*4*2 = 8192 bytes, 64 ms of ADC frames.
  esp_err_t result = i2s_new_channel(&channel, nullptr, &rx_);
  if (result != ESP_OK) {
    driverError("Create microphone I2S", result);
    end();
    return false;
  }

  i2s_tdm_config_t tdm = {};
  const i2s_tdm_clk_config_t clock = I2S_TDM_CLK_DEFAULT_CONFIG(kSampleRate);
  const i2s_tdm_slot_config_t slots = I2S_TDM_PHILIPS_SLOT_DEFAULT_CONFIG(
      I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_STEREO,
      static_cast<i2s_tdm_slot_mask_t>(I2S_TDM_SLOT0 | I2S_TDM_SLOT1 |
                                       I2S_TDM_SLOT2 | I2S_TDM_SLOT3));
  tdm.clk_cfg = clock;
  tdm.slot_cfg = slots;
  tdm.slot_cfg.total_slot = kSlots;
  tdm.gpio_cfg.mclk = kMclk;
  tdm.gpio_cfg.bclk = kBclk;
  tdm.gpio_cfg.ws = kWordSelect;
  tdm.gpio_cfg.din = kDataIn;
  tdm.gpio_cfg.dout = I2S_GPIO_UNUSED;
  result = i2s_channel_init_tdm_mode(rx_, &tdm);
  if (result != ESP_OK) {
    driverError("Configure microphone TDM", result);
    end();
    return false;
  }
  // The ADC needs its clocks while registers are being configured.
  result = i2s_channel_enable(rx_);
  if (result != ESP_OK) {
    driverError("Start microphone I2S", result);
    end();
    return false;
  }
  enabled_ = true;
  if (!configureCodec()) {
    end();
    return false;
  }
  ready_ = true;

  // Drain startup/reset transients while the ADC settles. A total 400 ms
  // deadline prevents a bad capture path from indefinitely blocking startup.
  int16_t discarded[128];
  size_t remaining = kSampleRate / 10;
  const int64_t deadline = esp_timer_get_time() + 400000;
  while (remaining) {
    const int64_t leftUs = deadline - esp_timer_get_time();
    if (leftUs <= 0) break;
    const size_t count = remaining < 128 ? remaining : 128;
    const size_t got = read(discarded, count, uint32_t((leftUs + 999) / 1000));
    remaining -= got;
    if (error_[0]) break;
  }
  if (remaining || error_[0]) {
    if (!error_[0]) snprintf(error_, sizeof(error_), "Timed out draining microphone startup audio");
    end();
    return false;
  }
  return true;
}

size_t BoardMicrophone::read(int16_t* destination, size_t samples,
                             uint32_t timeout_ms) {
  return readChannels(destination, samples, 1, timeout_ms);
}

size_t BoardMicrophone::readInterleaved(int16_t* destination, size_t frames,
                                       uint32_t timeout_ms) {
  return readChannels(destination, frames, kMicrophoneChannels, timeout_ms);
}

size_t BoardMicrophone::readChannels(int16_t* destination, size_t samples,
                                     size_t channels, uint32_t timeout_ms) {
  error_[0] = '\0';
  if (!samples) return 0;
  if (!ready_ || !rx_ || !buffer_) {
    snprintf(error_, sizeof(error_), "Microphone is not running");
    return 0;
  }
  if (!destination) {
    snprintf(error_, sizeof(error_), "Null microphone destination");
    return 0;
  }

  const int64_t deadline = esp_timer_get_time() + int64_t(timeout_ms) * 1000;
  size_t produced = 0;
  bool finishAfterBuffer = false;
  while (produced < samples) {
    // Consume complete frames before considering the timeout. Do not discard
    // already buffered samples, and never reinterpret a partial frame as mono.
    while (produced < samples && buffered_ - consumed_ >= kFrameBytes) {
      // The ES7210's Philips TDM order is ADC1,ADC3,ADC2,ADC4. ADC3 / slot1
      // is the analog playback reference, not the second physical microphone.
      // Copy individual MIC samples; contiguous first-four-byte extraction
      // would mistakenly feed microphone + quiet reference into dual-mic BSS.
      memcpy(destination + produced * channels,
             buffer_ + consumed_ + kMicrophone1Slot * sizeof(int16_t),
             sizeof(int16_t));
      if (channels == kMicrophoneChannels) {
        memcpy(destination + produced * channels + 1,
               buffer_ + consumed_ + kMicrophone2Slot * sizeof(int16_t),
               sizeof(int16_t));
      }
      consumed_ += kFrameBytes;
      ++produced;
    }
    if (produced == samples || finishAfterBuffer) break;

    uint32_t waitMs = 0;
    if (timeout_ms) {
      const int64_t leftUs = deadline - esp_timer_get_time();
      if (leftUs <= 0) break;
      waitMs = uint32_t(leftUs / 1000);
    }

    // Only a trailing partial TDM frame remains. Preserve it across both
    // chunk boundaries and timeout returns, including odd-byte fragments.
    const size_t tail = buffered_ - consumed_;
    if (tail && consumed_) memmove(buffer_, buffer_ + consumed_, tail);
    buffered_ = tail;
    consumed_ = 0;
    // ESP-IDF applies timeout_ms again for every DMA buffer it dequeues. Read
    // exactly to one DMA boundary, then recalculate the deadline ourselves.
    // Staging surplus samples also keeps the driver's cursor aligned when a
    // caller asks for fewer than 128 samples. On this single-reader path the
    // driver normally returns one whole DMA buffer or zero; retain the offset
    // as well as the PCM tail if a partial return occurs.
    const size_t request = kDmaBytes - dmaReadOffset_;
    size_t received = 0;
    const esp_err_t result =
        i2s_channel_read(rx_, buffer_ + tail, request, &received, waitMs);
    buffered_ += received;
    dmaReadOffset_ = (dmaReadOffset_ + received) % kDmaBytes;

    if (result != ESP_OK && result != ESP_ERR_TIMEOUT) {
      driverError("Read microphone I2S", result);
      finishAfterBuffer = true;
    } else if (result == ESP_ERR_TIMEOUT || received == 0) {
      finishAfterBuffer = true;
    }
  }
  return produced;
}

void BoardMicrophone::powerDownCodec() {
  // Best-effort es7210_stop(), preserving the original error if setup failed.
  const RegisterValue stop[] = {
      {0x47, 0xff}, {0x48, 0xff}, {0x49, 0xff}, {0x4a, 0xff},
      {0x4b, 0xff}, {0x4c, 0xff}, {0x40, 0xc0}, {0x01, 0x7f}, {0x06, 0x07},
  };
  for (const auto& setting : stop) {
    if (!writeRegister(setting.reg, setting.value, false) && !error_[0]) {
      snprintf(error_, sizeof(error_), "ES7210 power-down write failed at 0x%02x",
               setting.reg);
    }
  }
}

void BoardMicrophone::end() {
  ready_ = false;
  if (codecTouched_) {
    powerDownCodec();
    codecTouched_ = false;
  }
  if (rx_ && enabled_) {
    const esp_err_t result = i2s_channel_disable(rx_);
    if (result == ESP_OK) enabled_ = false;
    else if (!error_[0]) driverError("Stop microphone I2S", result);
  }
  if (rx_) {
    const esp_err_t result = i2s_del_channel(rx_);
    if (result == ESP_OK) {
      rx_ = nullptr;
      enabled_ = false;
    } else if (!error_[0]) {
      driverError("Release microphone I2S", result);
    }
  }
  if (buffer_) {
    heap_caps_free(buffer_);
    buffer_ = nullptr;
  }
  buffered_ = consumed_ = 0;
  dmaReadOffset_ = 0;
}
