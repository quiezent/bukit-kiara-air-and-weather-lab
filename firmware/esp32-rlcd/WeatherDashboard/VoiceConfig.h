#pragma once
#include <stdint.h>

// Set to 0 to build the dashboard without starting microphone/recognition tasks.
#ifndef WEATHER_VOICE_ENABLED
#define WEATHER_VOICE_ENABLED 1
#endif

namespace VoiceConfig {
constexpr uint32_t kSampleRate = 16000;
// Neural VAD is included in the pinned model image. BSS processing is enabled,
// but this SDK's WakeNet-off branch sends original MIC0 to MultiNet. NSNET is
// disabled for the current BSS configuration and is excluded by user choice.
constexpr const char *kVadModelName = "vadnet1_medium";
constexpr float kVadEnergyThresholdDbfs = -60.0f;
// User-selected confidence restored after the v20 sensitivity trial.
// This same threshold gates both MultiNet detection and accepted actions.
constexpr float kMinimumConfidence = 0.80f;
constexpr uint32_t kMinimumCommandGapMs = 900;
constexpr uint32_t kReleaseSilenceMs = 300;
constexpr uint32_t kMaximumEventAgeMs = 2500;
constexpr uint32_t kAcknowledgementMs = 2500;
// Count this window from each speech onset, including the AFE pre-roll, rather
// than from arbitrary idle silence. Keep a tail for delayed final-word results.
constexpr uint32_t kRecognitionTimeoutMs = 6000;
constexpr uint32_t kSessionTailSilenceMs = 600;
constexpr uint32_t kVadMinimumSpeechMs = 128;
// Cover the speech qualification interval and VAD's initial frame delay.
constexpr uint32_t kVadPreRollMs = 256;
}
