#pragma once
#include <Arduino.h>
#include "VoiceConfig.h"
#include "VoiceNavigation.h"
#include "VoiceDiagnostic.h"

struct VoiceEvent {
  int commandId = 0;
  int phraseId = -1;
  float confidence = 0;
  uint32_t recognizedMs = 0;
};

struct VoiceStatus {
  bool enabled = WEATHER_VOICE_ENABLED != 0;
  bool ready = false;
  bool playbackSuspended = false;
  const char *model = "mn7_en";
  const char *frontEnd = "Espressif AFE SR";
  const char *inputFormat = "MM";
  uint8_t microphoneChannels = 2;
  const char *microphoneSlots = "0,2";
  const char *vadModel = VoiceConfig::kVadModelName;
  const char *noiseSuppression = "Disabled; dual-microphone BSS";
  bool echoCancellation = false;
  bool blindSourceSeparation = true;
  bool fixedFirstChannel = false;
  bool fixedOutputChannel = false;
  const char *error = "Not started";
  uint32_t audioFrames = 0;
  uint32_t feedBackpressureFrames = 0;
  uint32_t recognitionFrames = 0;
  uint32_t microphonePeak = 0; // Absolute raw PCM peak in the most recent frame.
  uint32_t microphoneRms = 0;
  uint32_t mic1Peak = 0, mic1Rms = 0;
  uint32_t mic2Peak = 0, mic2Rms = 0;
  uint32_t clippedSamples = 0;
  uint32_t accepted = 0;
  uint32_t rejected = 0;
  uint32_t droppedEvents = 0;
  uint32_t timeouts = 0;
  uint32_t sessionsStarted = 0;
  uint32_t sessionsWithoutCommand = 0;
  uint32_t modelFrames = 0;
  uint8_t loadedPhrases = 0;
  uint32_t diagnosticsDropped = 0;
};

class VoiceControl {
 public:
  // Call once from setup(), after IndoorSensor.begin() has started shared Wire.
  bool begin();
  // Call only on the Arduino loop task. Recognition never draws or changes pages.
  bool take(VoiceEvent &event);
  // Main-loop-only. Copied phoneme diagnostics are never used to execute commands.
  bool takeDiagnostic(VoiceDiagnostic &diagnostic);
  // Main-loop-only: release shared audio pins before speaker.begin(). The
  // workers keep draining AFE silence and do not recognize playback speech.
  bool pauseForPlayback();
  // Main-loop-only, after speaker.end(): restart capture and allow commands
  // again once a quiet post-playback tail has cleared the audio front end.
  bool resumeAfterPlayback();
  VoiceStatus status() const;

 private:
  struct State;
  State *state_ = nullptr;
  bool attempted_ = false;
  char startupError_[160] = "Not started";
};
