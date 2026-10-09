/*
 * ES8311 register initialization adapted from Espressif esp_codec_dev:
 * SPDX-FileCopyrightText: 2023 Espressif Systems (Shanghai) CO LTD
 * SPDX-License-Identifier: Apache-2.0
 *
 * Modifications: fixed 16 kHz DAC-only slave setup, checked Arduino Wire I/O,
 * exclusive I2S clock ownership, bounded mono-to-stereo writes, quiet startup.
 * Licensed under the Apache License, Version 2.0. You may obtain a copy at
 * https://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software is
 * distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. Preserve the upstream license with source.
 *
 * Sources in Waveshare's official ESP32-S3-RLCD-4.2 repository:
 * https://github.com/waveshareteam/ESP32-S3-RLCD-4.2
 * 02_Example/Arduino/07_Audio_Test/src/ExternLib/esp_codec_dev/device/
 *   es8311/es8311.c (open, set_fs, 4096000/16000 coefficients, start, suspend)
 * 02_Example/ESP-IDF/07_Audio_Test/components/ExternLib/codec_board/board_cfg.txt
 * 02_Example/XiaoZhi/XiaoZhiCode_V2.1.0/main/boards/waveshare-s3-rlcd-4.2/config.h
 * The latter two agree on amplifier GPIO46 and speaker data GPIO8.
 */

#include "BoardSpeaker.h"

#include <Arduino.h>
#include <Wire.h>
#include <driver/i2s_std.h>
#include <esp_err.h>
#include <esp_idf_version.h>
#include <esp_timer.h>
#include <stdio.h>
#include <string.h>

#if !defined(CONFIG_IDF_TARGET_ESP32S3) || !CONFIG_IDF_TARGET_ESP32S3
#error "BoardSpeaker requires an ESP32-S3 target."
#endif
#if ESP_IDF_VERSION < ESP_IDF_VERSION_VAL(5, 5, 0)
#error "BoardSpeaker requires Arduino-ESP32 3.3 / ESP-IDF 5.5 or newer."
#endif

namespace {
constexpr uint8_t kDacAddress = 0x18;  // Wire is 7-bit; codec_dev uses 0x30.
constexpr gpio_num_t kMclk = GPIO_NUM_16;
constexpr gpio_num_t kBclk = GPIO_NUM_9;
constexpr gpio_num_t kWordSelect = GPIO_NUM_45;
constexpr gpio_num_t kDataOut = GPIO_NUM_8;
constexpr uint8_t kAmplifierEnable = 46;
constexpr uint8_t kDacVolume = 0xbf;  // 0 dB: 7.5 dB louder after the user's test.
struct RegisterValue { uint8_t reg; uint8_t value; };
}  // namespace

BoardSpeaker::~BoardSpeaker() { end(); }

void BoardSpeaker::driverError(const char* operation, int code) {
  snprintf(error_, sizeof(error_), "%s: %s (0x%x)", operation,
           esp_err_to_name(static_cast<esp_err_t>(code)), code);
}

bool BoardSpeaker::writeRegister(uint8_t reg, uint8_t value, bool reportFailure) {
  Wire.beginTransmission(kDacAddress);
  const bool queued = Wire.write(reg) == 1 && Wire.write(value) == 1;
  const uint8_t result = Wire.endTransmission(true);
  if (queued && result == 0) return true;
  if (reportFailure) {
    snprintf(error_, sizeof(error_), "ES8311 write reg 0x%02x failed (I2C %u)",
             reg, unsigned(result));
  }
  return false;
}

bool BoardSpeaker::readRegister(uint8_t reg, uint8_t& value) {
  Wire.beginTransmission(kDacAddress);
  const bool queued = Wire.write(reg) == 1;
  const uint8_t result = Wire.endTransmission(false);
  if (!queued || result != 0) {
    snprintf(error_, sizeof(error_), "ES8311 select reg 0x%02x failed (I2C %u)",
             reg, unsigned(result));
    return false;
  }
  if (Wire.requestFrom(kDacAddress, size_t(1), true) != 1 || !Wire.available()) {
    snprintf(error_, sizeof(error_), "ES8311 read reg 0x%02x failed", reg);
    return false;
  }
  value = static_cast<uint8_t>(Wire.read());
  return true;
}

bool BoardSpeaker::verifyRegister(uint8_t reg, uint8_t mask, uint8_t expected) {
  uint8_t actual;
  if (!readRegister(reg, actual)) return false;
  if ((actual & mask) == (expected & mask)) return true;
  snprintf(error_, sizeof(error_),
           "ES8311 reg 0x%02x readback 0x%02x, expected 0x%02x (mask 0x%02x)",
           reg, actual, expected, mask);
  return false;
}

bool BoardSpeaker::configureCodec() {
  codecTouched_ = true;
  // Espressif es8311_open(): external non-inverted MCLK, slave, analog DAC.
  // The first I2C write can fail on this chip; upstream deliberately repeats it.
  writeRegister(0x44, 0x08, false);
  const RegisterValue open[] = {
      {0x44, 0x08}, {0x01, 0x30}, {0x02, 0x00}, {0x03, 0x10},
      {0x16, 0x24}, {0x04, 0x10}, {0x05, 0x00}, {0x0b, 0x00},
      {0x0c, 0x00}, {0x10, 0x1f}, {0x11, 0x7f}, {0x00, 0x80},
      {0x01, 0x3f}, {0x06, 0x00}, {0x13, 0x10}, {0x1b, 0x0a},
      {0x1c, 0x6a}, {0x44, 0x08},
  };
  for (const auto& setting : open) {
    if (!writeRegister(setting.reg, setting.value)) return false;
  }
  // Upstream 4096000 Hz / 16000 Hz coefficient: pre_div=1, pre_multi=1,
  // ADC/DAC div=1, LRCK divider=255, BCLK divider=4, ADC OSR16/DAC OSR32.
  const RegisterValue format[] = {
      {0x09, 0x0c}, {0x0a, 0x0c},  // 16-bit, Philips I2S.
      {0x02, 0x00}, {0x05, 0x00}, {0x03, 0x10}, {0x04, 0x20},
      {0x07, 0x00}, {0x08, 0xff}, {0x06, 0x03},
  };
  for (const auto& setting : format) {
    if (!writeRegister(setting.reg, setting.value)) return false;
  }
  const RegisterValue start[] = {
      {0x00, 0x80}, {0x01, 0x3f}, {0x09, 0x0c}, {0x0a, 0x4c},
      {0x17, 0xbf}, {0x0e, 0x02}, {0x12, 0x00}, {0x14, 0x1a},
      {0x0d, 0x01}, {0x15, 0x40}, {0x37, 0x08}, {0x45, 0x00},
      {0x32, kDacVolume},
  };
  for (const auto& setting : start) {
    if (!writeRegister(setting.reg, setting.value)) return false;
  }
  uint8_t mute;
  if (!readRegister(0x31, mute) || !writeRegister(0x31, mute & 0x9f)) return false;
  return verifyRegister(0x00, 0x40, 0x00) &&  // ES8311 never drives host clocks.
         verifyRegister(0x01, 0xff, 0x3f) &&
         verifyRegister(0x04, 0x7f, 0x20) &&
         verifyRegister(0x09, 0x5f, 0x0c) &&
         verifyRegister(0x0a, 0x40, 0x40) &&
         verifyRegister(0x12, 0xff, 0x00) &&
         verifyRegister(0x31, 0x60, 0x00) &&
         verifyRegister(0x32, 0xff, kDacVolume);
}

bool BoardSpeaker::begin() {
  end();
  error_[0] = '\0';
  if (tx_ || reservedRx_) {
    snprintf(error_, sizeof(error_), "Previous speaker I2S channel could not be released");
    return false;
  }
  digitalWrite(kAmplifierEnable, LOW);
  pinMode(kAmplifierEnable, OUTPUT);
  Wire.beginTransmission(kDacAddress);
  const uint8_t probe = Wire.endTransmission(true);
  if (probe != 0) {
    snprintf(error_, sizeof(error_), "ES8311 at 0x18 unavailable (I2C %u)", unsigned(probe));
    return false;
  }

  i2s_chan_config_t channel = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
  channel.dma_desc_num = 6;
  channel.dma_frame_num = kDmaFrames;  // 6 * 128 / 16000 = 48 ms of stereo.
  channel.auto_clear = true;
  esp_err_t result = i2s_new_channel(&channel, &tx_, &reservedRx_);
  if (result != ESP_OK) {
    driverError("Reserve speaker I2S; release microphone first", result);
    end();
    return false;
  }
  i2s_std_config_t config = {};
  config.clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(kSampleRate);
  config.clk_cfg.mclk_multiple = I2S_MCLK_MULTIPLE_256;
  config.slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
      I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_STEREO);
  config.gpio_cfg.mclk = kMclk;
  config.gpio_cfg.bclk = kBclk;
  config.gpio_cfg.ws = kWordSelect;
  config.gpio_cfg.dout = kDataOut;
  config.gpio_cfg.din = I2S_GPIO_UNUSED;
  result = i2s_channel_init_std_mode(tx_, &config);
  if (result != ESP_OK) {
    driverError("Configure speaker I2S", result);
    end();
    return false;
  }
  memset(stereo_, 0, sizeof(stereo_));
  size_t loaded = 0;
  result = i2s_channel_preload_data(tx_, stereo_, sizeof(stereo_), &loaded);
  if (result != ESP_OK || loaded != sizeof(stereo_)) {
    if (result != ESP_OK) driverError("Preload speaker silence", result);
    else snprintf(error_, sizeof(error_), "Incomplete speaker silence preload");
    end();
    return false;
  }
  result = i2s_channel_enable(tx_);
  if (result != ESP_OK) {
    driverError("Start speaker I2S", result);
    end();
    return false;
  }
  enabled_ = true;
  if (!configureCodec()) {
    end();
    return false;
  }
  ready_ = true;
  const int16_t silence[kDmaFrames] = {};
  for (uint8_t i = 0; i < 3; ++i) {
    if (write(silence, kDmaFrames, 250) != kDmaFrames || error_[0]) {
      if (!error_[0]) snprintf(error_, sizeof(error_), "Timed out priming speaker silence");
      end();
      return false;
    }
  }
  delay(25);  // Let clocks and DAC ramp settle while the physical amp is quiet.
  digitalWrite(kAmplifierEnable, HIGH);
  return true;
}

size_t BoardSpeaker::write(const int16_t* mono, size_t samples, uint32_t timeoutMs) {
  error_[0] = '\0';
  if (!ready_ || !tx_ || !enabled_) {
    snprintf(error_, sizeof(error_), "Speaker is not running");
    return 0;
  }
  if (samples && !mono) {
    snprintf(error_, sizeof(error_), "Null speaker source");
    return 0;
  }
  const int64_t deadline = esp_timer_get_time() + int64_t(timeoutMs) * 1000;
  size_t accepted = 0;
  while (pendingBytes_ || accepted < samples) {
    uint32_t waitMs = 0;
    if (timeoutMs) {
      const int64_t leftUs = deadline - esp_timer_get_time();
      if (leftUs <= 0) break;
      waitMs = uint32_t(leftUs / 1000);
    }
    const size_t boundary = kDmaBytes - dmaWriteOffset_;
    if (pendingBytes_) {
      const size_t request = pendingBytes_ < boundary ? pendingBytes_ : boundary;
      size_t written = 0;
      const esp_err_t result = i2s_channel_write(tx_, pending_, request, &written, waitMs);
      if (written > request) {
        snprintf(error_, sizeof(error_), "Invalid speaker I2S byte count");
        break;
      }
      pendingBytes_ -= written;
      if (pendingBytes_) memmove(pending_, pending_ + written, pendingBytes_);
      dmaWriteOffset_ = (dmaWriteOffset_ + written) % kDmaBytes;
      if (result != ESP_OK && result != ESP_ERR_TIMEOUT) driverError("Write speaker I2S", result);
      if (result != ESP_OK || !written) break;
      continue;
    }

    // Limit a driver call to one actual DMA boundary. ESP-IDF repeats its
    // timeout for every dequeued buffer; this caller owns the total deadline.
    size_t count = samples - accepted;
    const size_t boundarySamples = boundary / kStereoFrameBytes;
    if (count > boundarySamples) count = boundarySamples;
    for (size_t i = 0; i < count; ++i) {
      stereo_[2 * i] = mono[accepted + i];
      stereo_[2 * i + 1] = mono[accepted + i];
    }
    const size_t request = count * kStereoFrameBytes;
    size_t written = 0;
    const esp_err_t result = i2s_channel_write(tx_, stereo_, request, &written, waitMs);
    if (written > request) {
      snprintf(error_, sizeof(error_), "Invalid speaker I2S byte count");
      break;
    }
    const size_t complete = written / kStereoFrameBytes;
    const size_t partial = written % kStereoFrameBytes;
    accepted += complete;
    if (partial) {
      pendingBytes_ = kStereoFrameBytes - partial;
      memcpy(pending_, reinterpret_cast<const uint8_t*>(stereo_) + written, pendingBytes_);
      ++accepted;  // This input sample is owned by the retained frame remainder.
    }
    dmaWriteOffset_ = (dmaWriteOffset_ + written) % kDmaBytes;
    if (result != ESP_OK && result != ESP_ERR_TIMEOUT) driverError("Write speaker I2S", result);
    if (result != ESP_OK || !written) break;
  }
  return accepted;
}

void BoardSpeaker::powerDownCodec() {
  const RegisterValue suspend[] = {
      {0x32, 0x00}, {0x17, 0x00}, {0x0e, 0xff}, {0x12, 0x02},
      {0x14, 0x00}, {0x0d, 0xfa}, {0x15, 0x00}, {0x02, 0x10},
      {0x00, 0x00}, {0x00, 0x1f}, {0x01, 0x30}, {0x01, 0x00},
      {0x45, 0x00}, {0x0d, 0xfc}, {0x02, 0x00},
  };
  for (const auto& setting : suspend) {
    if (!writeRegister(setting.reg, setting.value, false) && !error_[0]) {
      snprintf(error_, sizeof(error_), "ES8311 standby write failed at 0x%02x", setting.reg);
    }
  }
}

void BoardSpeaker::mute() {
  digitalWrite(kAmplifierEnable, LOW);
}

void BoardSpeaker::end() {
  // Must run after the playback task has finished, before microphone restart.
  mute();
  ready_ = false;
  if (codecTouched_) {
    powerDownCodec();
    codecTouched_ = false;
  }
  if (tx_ && enabled_) {
    const esp_err_t result = i2s_channel_disable(tx_);
    if (result == ESP_OK) enabled_ = false;
    else if (!error_[0]) driverError("Stop speaker I2S", result);
  }
  if (tx_) {
    const esp_err_t result = i2s_del_channel(tx_);
    if (result == ESP_OK) {
      tx_ = nullptr;
      enabled_ = false;
    } else if (!error_[0]) driverError("Release speaker I2S", result);
  }
  if (reservedRx_) {
    const esp_err_t result = i2s_del_channel(reservedRx_);
    if (result == ESP_OK) reservedRx_ = nullptr;
    else if (!error_[0]) driverError("Release speaker RX reservation", result);
  }
  pendingBytes_ = dmaWriteOffset_ = 0;
}
