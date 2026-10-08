#include "VoiceControl.h"
#include "VoicePhrases.h"

#if WEATHER_VOICE_ENABLED

#include "sdkconfig.h"
#if !CONFIG_IDF_TARGET_ESP32S3 || !CONFIG_MODEL_IN_FLASH
#error "Voice navigation needs the ESP32-S3 target and the ESP-SR flash-model SDK; use Arduino-ESP32 3.3.12."
#endif

#include "BoardMicrophone.h"
#include "SpeechModelInfo.h"
#include "VoiceSession.h"
#include "VoiceFeedHealth.h"
#include <atomic>
#include <cmath>
#include <new>
#include "esp_heap_caps.h"
#include "esp_partition.h"
#include "esp_timer.h"
#include "esp_afe_sr_models.h"
#include "esp_mn_models.h"
#include "esp_mn_speech_commands.h"
#include "model_path.h"
#include "mbedtls/sha256.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
extern "C" {
#include "flite_g2p.h"
}

namespace VoiceRuntime {
enum class Fault : uint8_t {
  None, Microphone, MicrophoneSuspend, MicrophoneResume, AudioFeed, AudioFetch, FrameSize
};
constexpr uint32_t kPlaybackQuietMs = 600;

// AFE may have buffered frames, so counting silence samples alone could clear
// suppression too quickly. Require both elapsed time and consecutive silence.
class PlaybackQuietTail {
 public:
  void reset() { silenceSamples_ = 0; }
  bool observe(bool speech, uint32_t samples, uint32_t nowMs, uint32_t resumedMs) {
    constexpr uint32_t target = kPlaybackQuietMs * VoiceConfig::kSampleRate / 1000;
    if (speech) silenceSamples_ = 0;
    else silenceSamples_ += samples >= target - silenceSamples_
        ? target - silenceSamples_ : samples;
    return silenceSamples_ >= target && uint32_t(nowMs - resumedMs) >= kPlaybackQuietMs;
  }
 private:
  uint32_t silenceSamples_ = 0;
};
}

namespace {
using RuntimeFault = VoiceRuntime::Fault;
using PlaybackQuietTail = VoiceRuntime::PlaybackQuietTail;
constexpr uint32_t kPlaybackPauseTimeoutMs = 500;

using WeatherVoicePhrases::kPhrases;

const char *faultText(RuntimeFault fault) {
  switch (fault) {
    case RuntimeFault::Microphone: return "Microphone stream stopped; restart the board";
    case RuntimeFault::MicrophoneSuspend: return "Microphone could not release audio pins for playback; restart the board";
    case RuntimeFault::MicrophoneResume: return "Microphone could not restart after playback; restart the board";
    case RuntimeFault::AudioFeed: return "Audio front-end feed failed; restart the board";
    case RuntimeFault::AudioFetch: return "Audio front-end output stopped; restart the board";
    case RuntimeFault::FrameSize: return "Unexpected recognition frame size; restart the board";
    default: return "";
  }
}

// ESP-SR's binary model parser trusts its file table. Verify the exact supplied
// image first, including on an erased or interrupted model flash, so ordinary
// model-installation errors do not enter that parser.
bool modelImageIsValid(char *error, size_t errorSize) {
  const esp_partition_t *partition = esp_partition_find_first(
      ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, "model");
  if (!partition || partition->size < weather_speech_model::kImageBytes) {
    snprintf(error, errorSize, "Missing model partition; flash the supplied partition table and model.bin");
    return false;
  }
  uint8_t buffer[2048];
  uint8_t digest[32];
  mbedtls_sha256_context sha;
  mbedtls_sha256_init(&sha);
  bool ok = mbedtls_sha256_starts(&sha, 0) == 0;
  for (size_t offset = 0; ok && offset < weather_speech_model::kImageBytes;) {
    size_t count = weather_speech_model::kImageBytes - offset;
    if (count > sizeof(buffer)) count = sizeof(buffer);
    ok = esp_partition_read(partition, offset, buffer, count) == ESP_OK
        && mbedtls_sha256_update(&sha, buffer, count) == 0;
    offset += count;
    if ((offset % 65536) == 0) delay(1);
  }
  if (ok) ok = mbedtls_sha256_finish(&sha, digest) == 0;
  mbedtls_sha256_free(&sha);
  if (!ok || memcmp(digest, weather_speech_model::kSha256, sizeof(digest)) != 0) {
    snprintf(error, errorSize, "Model image missing, incomplete, or different; flash the supplied model.bin");
    return false;
  }
  return true;
}
}

struct VoiceControl::State {
  BoardMicrophone microphone;
  srmodel_list_t *models = nullptr;
  const esp_afe_sr_iface_t *afe = nullptr;
  esp_afe_sr_data_t *afeData = nullptr;
  const esp_mn_iface_t *multiNet = nullptr;
  model_iface_data_t *multiNetData = nullptr;
  bool commandsAllocated = false;
  QueueHandle_t events = nullptr;
  QueueHandle_t diagnostics = nullptr;
  TaskHandle_t feedTask = nullptr;
  TaskHandle_t detectTask = nullptr;
  int16_t *input = nullptr;
  int16_t *recognitionInput = nullptr;
  int feedSamples = 0;
  int fetchSamples = 0;
  std::atomic<bool> running{false};
  // Zero means capture; each pause has a unique token. A matching worker token
  // proves it is no longer inside microphone.read(), including repeated reads.
  std::atomic<uint32_t> capturePauseRequested{0}, capturePaused{0};
  // Generation-tagged suppression lets the detector clear its own quiet tail
  // without overwriting a new playback request arriving on the main loop.
  std::atomic<uint32_t> playbackSuppressed{0};
  std::atomic<uint32_t> playbackEpoch{0}, playbackResumedMs{0};
  uint32_t nextPauseToken = 0;  // Main loop owns token allocation and codec I/O.
  bool microphoneReleased = false;
  std::atomic<RuntimeFault> fault{RuntimeFault::None};
  std::atomic<uint32_t> audioFrames{0}, recognitionFrames{0};
  std::atomic<uint32_t> feedBackpressureFrames{0};
  // A dropped feed frame invalidates partial recognition and queued commands.
  // The detector acknowledges each generation only after clearing both.
  std::atomic<uint32_t> audioDiscontinuityEpoch{0}, recognitionDiscontinuityEpoch{0};
  std::atomic<uint32_t> peak{0}, rms{0}, clipped{0};
  std::atomic<uint32_t> mic1Peak{0}, mic1Rms{0}, mic2Peak{0}, mic2Rms{0};
  std::atomic<uint32_t> accepted{0}, rejected{0}, dropped{0}, timeouts{0};
  std::atomic<uint32_t> sessionsStarted{0}, sessionsWithoutCommand{0}, modelFrames{0};
  std::atomic<uint32_t> diagnosticsDropped{0};
  uint32_t nextDiagnosticSequence = 0; // Detector is the sole producer.
  uint8_t loadedPhrases = 0; // Set before the workers start; immutable afterward.

  // Used only if initialization fails, while any created tasks still wait for
  // their initial notification. No task can be using these resources then.
  void releaseBeforeStart() {
    if (feedTask) { vTaskDelete(feedTask); feedTask = nullptr; }
    if (detectTask) { vTaskDelete(detectTask); detectTask = nullptr; }
    microphone.end();
    // The pinned MN7 destructor frees its command table as well as model state.
    if (multiNetData) {
      multiNet->destroy(multiNetData);
      multiNetData = nullptr;
      commandsAllocated = false;
    } else if (commandsAllocated) {
      esp_mn_commands_free();
      commandsAllocated = false;
    }
    if (afeData) { afe->destroy(afeData); afeData = nullptr; }
    if (models) { esp_srmodel_deinit(models); models = nullptr; }
    if (input) { heap_caps_free(input); input = nullptr; }
    if (recognitionInput) { heap_caps_free(recognitionInput); recognitionInput = nullptr; }
    if (events) { vQueueDelete(events); events = nullptr; }
    if (diagnostics) { vQueueDelete(diagnostics); diagnostics = nullptr; }
  }

  void stopWithFault(RuntimeFault reason) {
    RuntimeFault expected = RuntimeFault::None;
    fault.compare_exchange_strong(expected, reason);
    running.store(false);
  }

  void discardEvents() {
    VoiceEvent unused;
    while (xQueueReceive(events, &unused, 0) == pdTRUE) dropped.fetch_add(1);
  }

  static void feedAudio(void *argument) {
    auto *self = static_cast<State *>(argument);
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    size_t filled = 0;
    uint32_t lastAudioMs = millis();
    bool silenceMode = false;
    int64_t nextSilentUs = 0;
    const int64_t silentFrameUs = int64_t(self->feedSamples) * 1000000 / VoiceConfig::kSampleRate;
    constexpr size_t channels = BoardMicrophone::kMicrophoneChannels;
    const size_t inputElements = size_t(self->feedSamples) * channels;
    VoiceFeedHealth feedHealth(millis()); // Start only after the worker is released.
    auto feedFrame = [&]() {
      const int fed = self->afe->feed(self->afeData, self->input);
      if (fed == 0) {
        self->feedBackpressureFrames.fetch_add(1);
        self->audioDiscontinuityEpoch.fetch_add(1);
      }
      const auto result = feedHealth.observe(fed, millis());
      if (result == VoiceFeedHealth::Result::Fault) {
        self->stopWithFault(RuntimeFault::AudioFeed);
        return false;
      }
      if (result == VoiceFeedHealth::Result::Accepted) self->audioFrames.fetch_add(1);
      // A zero return drops this input, rather than processing it twice in BSS.
      return true;
    };
    while (self->running.load()) {
      const uint32_t pauseToken = self->capturePauseRequested.load();
      if (pauseToken) {
        // Only the main loop touches Wire or stops/restarts the codec. Once
        // acknowledged it can release I2S safely while this worker feeds zero
        // PCM at the original rate, keeping AFE and its liveness timers alive.
        if (!silenceMode) {
          filled = 0;
          memset(self->input, 0, inputElements * sizeof(int16_t));
          self->peak.store(0);
          self->rms.store(0);
          self->mic1Peak.store(0);
          self->mic1Rms.store(0);
          self->mic2Peak.store(0);
          self->mic2Rms.store(0);
          nextSilentUs = esp_timer_get_time();
          silenceMode = true;
        }
        self->capturePaused.store(pauseToken);
        lastAudioMs = millis();
        if (!feedFrame()) break;
        nextSilentUs += silentFrameUs;
        int64_t leftUs = nextSilentUs - esp_timer_get_time();
        if (leftUs > 0) {
          const TickType_t ticks = pdMS_TO_TICKS(uint32_t((leftUs + 999) / 1000));
          vTaskDelay(ticks ? ticks : 1);
        } else {
          // Never catch up with a burst of unpaced zero frames after a delay.
          nextSilentUs = esp_timer_get_time();
          vTaskDelay(1);
        }
        continue;
      }
      if (silenceMode) {
        filled = 0;
        lastAudioMs = millis();
        silenceMode = false;
      }
      self->capturePaused.store(0);
      // readInterleaved returns ADC time frames, not the number of int16_t
      // elements. Both microphone samples from each frame remain together.
      const size_t got = self->microphone.readInterleaved(
          self->input + filled * channels, self->feedSamples - filled, 100);
      if (self->microphone.lastError()[0]) {
        self->stopWithFault(RuntimeFault::Microphone);
        break;
      }
      if (!got) {
        if (uint32_t(millis() - lastAudioMs) >= 2000) {
          self->stopWithFault(RuntimeFault::Microphone);
          break;
        }
        vTaskDelay(pdMS_TO_TICKS(1));
        continue;
      }
      lastAudioMs = millis();
      filled += got;
      // A request arriving during the bounded read must discard that partial
      // microphone frame before acknowledgement and before feeding AFE.
      if (self->capturePauseRequested.load()) {
        filled = 0;
        continue;
      }
      if (filled < size_t(self->feedSamples)) continue;
      uint32_t peak = 0, clipped = 0;
      uint32_t channelPeak[channels] = {};
      uint64_t channelSquares[channels] = {};
      uint64_t sumSquares = 0;
      for (size_t i = 0; i < inputElements; ++i) {
        int32_t sample = self->input[i];
        uint32_t magnitude = sample < 0 ? -sample : sample;
        if (magnitude > peak) peak = magnitude;
        const size_t channel = i % channels;
        if (magnitude > channelPeak[channel]) channelPeak[channel] = magnitude;
        if (magnitude >= 32760) ++clipped;
        sumSquares += int64_t(sample) * sample;
        channelSquares[channel] += int64_t(sample) * sample;
      }
      self->peak.store(peak);
      self->rms.store(uint32_t(sqrt(double(sumSquares) / inputElements)));
      self->mic1Peak.store(channelPeak[0]);
      self->mic1Rms.store(uint32_t(sqrt(double(channelSquares[0]) / self->feedSamples)));
      self->mic2Peak.store(channelPeak[1]);
      self->mic2Rms.store(uint32_t(sqrt(double(channelSquares[1]) / self->feedSamples)));
      self->clipped.fetch_add(clipped);
      if (!self->running.load()) break;
      if (!feedFrame()) break;
      filled = 0;
    }
    // The lifetime is the board's uptime. On a runtime fault, stop the workers
    // without destroying objects possibly in use by ESP-SR's internal tasks.
    // The main dashboard remains usable and reports the reason in /status.
    vTaskDelete(nullptr);
  }

  static void detectCommands(void *argument) {
    auto *self = static_cast<State *>(argument);
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    VoiceUtteranceGate gate(
        VoiceConfig::kMinimumCommandGapMs * VoiceConfig::kSampleRate / 1000,
        VoiceConfig::kReleaseSilenceMs * VoiceConfig::kSampleRate / 1000);
    VoiceRecognitionSession session(
        VoiceConfig::kSessionTailSilenceMs * VoiceConfig::kSampleRate / 1000);
    VoiceFrameAssembler assembler(self->recognitionInput, self->fetchSamples);
    PlaybackQuietTail playbackTail;
    uint32_t playbackEpoch = self->playbackEpoch.load();
    uint32_t audioEpoch = self->recognitionDiscontinuityEpoch.load();
    VoiceDiagnostic trace;
    bool traceActive = false;
    uint32_t traceStartedMs = 0;
    // The exact MN7 SDK owns one stable results object. On detection/timeout detect()
    // snapshots strings internally before resetting paths; another getter
    // would overwrite them from the now-empty paths. Keep this borrowed pointer
    // only on the detector task and copy it before explicit clean().
    const esp_mn_results_t *sessionResults = nullptr;
    auto finishTrace = [&](VoiceDiagnosticReason reason) {
      if (!traceActive) return;
      const esp_mn_results_t *results = reason == VoiceDiagnosticReason::SilenceTail
          ? self->multiNet->get_results(self->multiNetData) : sessionResults;
      trace.reason = reason;
      trace.endedMs = millis();
      trace.sessionDurationMs = uint32_t(trace.endedMs - traceStartedMs);
      trace.sequence = ++self->nextDiagnosticSequence;
      copyVoiceDiagnosticResults(trace, results);
      if (xQueueSend(self->diagnostics, &trace, 0) != pdTRUE)
        self->diagnosticsDropped.fetch_add(1);
      traceActive = false;
      sessionResults = nullptr;
    };
    auto clearRecognition = [&]() {
      traceActive = false; // Playback/feed gaps are intentionally not utterance traces.
      sessionResults = nullptr;
      self->multiNet->clean(self->multiNetData);
      session.stop();
      assembler.reset();
      gate.block();
      playbackTail.reset();
      self->discardEvents();
    };
    auto clearDiscontinuity = [&]() {
      const uint32_t current = self->audioDiscontinuityEpoch.load();
      if (current == audioEpoch) return false;
      clearRecognition();
      audioEpoch = current;
      self->recognitionDiscontinuityEpoch.store(current);
      return true;
    };
    auto consume = [&](int16_t *samples) {
      if (clearDiscontinuity()) return false;
      const uint32_t suppressionToken = self->playbackSuppressed.load();
      if (suppressionToken) {
        clearRecognition();
        return false;
      }
      self->modelFrames.fetch_add(1);
      if (traceActive) ++trace.modelFrames;
      const esp_mn_state_t result = self->multiNet->detect(self->multiNetData, samples);
      if (traceActive) trace.detectState = int(result);
      // Feed can lose a frame while MultiNet evaluates the previous one.
      if (clearDiscontinuity()) return false;
      // Playback can be requested while detect() is evaluating this frame.
      if (self->playbackSuppressed.load()) {
        clearRecognition();
        return false;
      }
      if (result == ESP_MN_STATE_DETECTING) return true;

      if (result == ESP_MN_STATE_TIMEOUT) {
        self->timeouts.fetch_add(1);
        self->sessionsWithoutCommand.fetch_add(1);
        finishTrace(VoiceDiagnosticReason::Timeout);
      } else {
        const esp_mn_results_t *matches = sessionResults;
        VoiceEvent event;
        bool valid = result == ESP_MN_STATE_DETECTED && matches && matches->num > 0;
        if (valid) {
          // Result memory belongs to MultiNet; copy before clean()/next detect().
          event.commandId = matches->command_id[0];
          event.phraseId = matches->phrase_id[0];
          event.confidence = matches->prob[0];
          event.recognizedMs = millis();
          valid = event.confidence >= VoiceConfig::kMinimumConfidence
              && event.confidence <= 1.0f
              && knownVoiceCommand(event.commandId);
        }
        if (valid) {
          self->accepted.fetch_add(1);
          if (xQueueSend(self->events, &event, 0) != pdTRUE) self->dropped.fetch_add(1);
        } else {
          self->rejected.fetch_add(1);
        }
        finishTrace(valid ? VoiceDiagnosticReason::Detected : VoiceDiagnosticReason::Rejected);
      }
      self->multiNet->clean(self->multiNetData);
      session.stop();
      gate.block();
      return false;
    };
    uint32_t lastFrameMs = millis();
    while (self->running.load()) {
      afe_fetch_result_t *frame = self->afe->fetch_with_delay(self->afeData, pdMS_TO_TICKS(100));
      if (!self->running.load()) break;
      if (!frame || frame->ret_value != ESP_OK || !frame->data) {
        if (uint32_t(millis() - lastFrameMs) >= 3000) {
          self->stopWithFault(RuntimeFault::AudioFetch);
          break;
        }
        vTaskDelay(1);
        continue;
      }
      lastFrameMs = millis();
      if (frame->data_size != int(self->fetchSamples * sizeof(int16_t))) {
        self->stopWithFault(RuntimeFault::FrameSize);
        break;
      }
      self->recognitionFrames.fetch_add(1);
      if (clearDiscontinuity()) continue; // Discard the frame crossing the gap.
      const bool speech = frame->vad_state == VAD_SPEECH;
      const uint32_t currentPlaybackEpoch = self->playbackEpoch.load();
      if (currentPlaybackEpoch != playbackEpoch) {
        playbackEpoch = currentPlaybackEpoch;
        clearRecognition();
      }
      const uint32_t suppressionToken = self->playbackSuppressed.load();
      if (suppressionToken) {
        self->discardEvents();
        if (self->capturePauseRequested.load()) {
          playbackTail.reset();
        } else {
          gate.observe(speech, self->fetchSamples);
          if (playbackTail.observe(speech, self->fetchSamples,
              millis(), self->playbackResumedMs.load())) {
            // The fetched frame and VAD cache still belong to the suppressed
            // interval. Resume only on a subsequent frame after this quiet tail.
            uint32_t expectedToken = suppressionToken;
            self->playbackSuppressed.compare_exchange_strong(expectedToken, 0);
          }
        }
        continue;
      }
      gate.observe(speech, self->fetchSamples);
      if (!gate.ready()) continue;
      const auto action = session.observe(speech, self->fetchSamples);
      if (action == VoiceRecognitionSession::FrameAction::Ignore) continue;
      if (action == VoiceRecognitionSession::FrameAction::Start) {
        const uint32_t sessionId = self->sessionsStarted.fetch_add(1) + 1;
        self->multiNet->clean(self->multiNetData);
        trace = VoiceDiagnostic{};
        trace.sessionId = sessionId;
        trace.vadCacheSamples = frame->vad_cache_size > 0
            ? uint32_t(frame->vad_cache_size / sizeof(int16_t)) : 0;
        traceStartedMs = millis();
        traceActive = true;
        sessionResults = self->multiNet->get_results(self->multiNetData);
        assembler.reset();
      }
      if (traceActive) {
        if (trace.afeTriggerChannel >= 0 && trace.afeTriggerChannel != frame->trigger_channel_id)
          ++trace.afeChannelChanges;
        trace.afeTriggerChannel = frame->trigger_channel_id;
        trace.afeRawChannels = frame->raw_data_channels;
        if (speech) {
          const uint32_t samples = uint32_t(self->fetchSamples);
          trace.speechSamples += samples >= UINT32_MAX - trace.speechSamples
              ? UINT32_MAX - trace.speechSamples : samples;
          if (std::isfinite(frame->data_volume)) {
            if (!std::isfinite(trace.minimumSpeechDbfs) || frame->data_volume < trace.minimumSpeechDbfs)
              trace.minimumSpeechDbfs = frame->data_volume;
            if (!std::isfinite(trace.maximumSpeechDbfs) || frame->data_volume > trace.maximumSpeechDbfs)
              trace.maximumSpeechDbfs = frame->data_volume;
          }
        }
      }
      if (action == VoiceRecognitionSession::FrameAction::Start) {
        // AFE owns this cache until the next fetch. Consume it now, before the
        // current speech frame, to retain audio preceding VAD's delayed onset.
        if (frame->vad_cache_size < 0 || frame->vad_cache_size % sizeof(int16_t)
            || (frame->vad_cache_size && !frame->vad_cache)) {
          self->stopWithFault(RuntimeFault::FrameSize);
          break;
        }
        if (!assembler.append(frame->vad_cache,
            frame->vad_cache_size / sizeof(int16_t), consume)) continue;
      }
      if (!assembler.append(frame->data, self->fetchSamples, consume)) continue;
      if (session.finished()) {
        self->sessionsWithoutCommand.fetch_add(1);
        finishTrace(VoiceDiagnosticReason::SilenceTail);
        // Keep the real-time fetch task independent of USB console readers.
        // HTTP /status retains the unmatched-session and model-frame counters.
        self->multiNet->clean(self->multiNetData);
        session.stop();
        assembler.reset();
      }
    }
    vTaskDelete(nullptr);
  }
};

bool VoiceControl::begin() {
  if (attempted_) return state_ && state_->running.load();
  attempted_ = true;
  if (!psramFound() || ESP.getPsramSize() < 4 * 1024 * 1024) {
    snprintf(startupError_, sizeof(startupError_), "PSRAM unavailable; select OPI PSRAM for this 8 MB PSRAM board");
    return false;
  }
  if (!modelImageIsValid(startupError_, sizeof(startupError_))) return false;
  State *state = new (std::nothrow) State;
  if (!state) {
    snprintf(startupError_, sizeof(startupError_), "No memory for voice state");
    return false;
  }
  auto fail = [&](const char *reason) {
    snprintf(startupError_, sizeof(startupError_), "%s", reason);
    state->releaseBeforeStart();
    delete state;
    return false;
  };
  state->events = xQueueCreate(4, sizeof(VoiceEvent));
  if (!state->events) return fail("No memory for voice event queue");
  state->diagnostics = xQueueCreate(4, sizeof(VoiceDiagnostic));
  if (!state->diagnostics) return fail("No memory for voice diagnostic queue");
  state->models = esp_srmodel_init("model");
  if (!state->models) return fail("ESP-SR could not load the model partition");
  char *modelName = esp_srmodel_filter(state->models, ESP_MN_PREFIX, ESP_MN_ENGLISH);
  if (!modelName || strcmp(modelName, weather_speech_model::kModelName) != 0)
    return fail("The supplied mn7_en model is required");
  char *vadName = esp_srmodel_filter(state->models, ESP_VADN_PREFIX, nullptr);
  if (!vadName || strcmp(vadName, VoiceConfig::kVadModelName) != 0)
    return fail("The supplied vadnet1_medium model is required");

  // Physical MIC1/MIC2 use ADC1/ADC2. ES7210 TDM Philips serial order is
  // ADC1,ADC3,ADC2,ADC4, so their transport slots are 0/2; the ADC3 analog
  // speaker reference is slot 1 (ES7210 datasheet Fig 2e, page 8):
  // https://files.waveshare.com/wiki/common/ES7210_DS.pdf
  // Exclusive playback currently suspends capture, so reference is not
  // passed to AEC as a fake concurrent reference.
  afe_config_t *config = afe_config_init("MM", state->models, AFE_TYPE_SR, AFE_MODE_LOW_COST);
  if (!config) return fail("No memory for audio front-end configuration");
  config->wakenet_init = false;
  config->wakenet_model_name = nullptr;
  config->wakenet_model_name_2 = nullptr;
  config->aec_init = false; // Concurrent reference capture is not active yet.
  config->se_init = true;
  // ESP-SR prioritizes dual-channel BSS over NS. Do not report an inactive
  // single-channel NS stage as enabled. NSNET remains excluded by user choice.
  config->ns_init = false;
  config->afe_ns_mode = AFE_NS_MODE_WEBRTC;
  config->ns_model_name = nullptr;
  config->vad_init = true;
  config->vad_model_name = vadName;
  config->vad_mode = VAD_MODE_1;
  config->vad_energy_threshold = VoiceConfig::kVadEnergyThresholdDbfs;
  config->vad_min_speech_ms = VoiceConfig::kVadMinimumSpeechMs;
  config->vad_min_noise_ms = 128;
  config->vad_delay_ms = VoiceConfig::kVadPreRollMs;
  config->vad_enable_channel_trigger = true;
  config->agc_init = false;
  config->fixed_first_channel = false;
  config->fixed_output_channel = false;
  config->memory_alloc_mode = AFE_MEMORY_ALLOC_MORE_PSRAM;
  config->afe_perferred_core = 1;
  config->afe_perferred_priority = 5;
  config->afe_ringbuf_size = 50;
  // Check the normalized SDK configuration before advertising active stages.
  // A disabled/changed dual-mic pipeline must fail visibly instead of silently
  // dropping a microphone while /status still describes BSS.
  afe_config_check(config);
  if (config->pcm_config.mic_num != int(BoardMicrophone::kMicrophoneChannels)
      || config->pcm_config.total_ch_num != int(BoardMicrophone::kMicrophoneChannels)
      || !config->se_init || config->ns_init || config->aec_init
      || config->fixed_first_channel || config->fixed_output_channel
      || !config->vad_init || !config->vad_model_name
      || strcmp(config->vad_model_name, VoiceConfig::kVadModelName) != 0) {
    afe_config_free(config);
    return fail("Audio front-end rejected the dual-microphone BSS/VAD configuration");
  }
  state->afe = esp_afe_handle_from_config(config);
  if (state->afe) state->afeData = state->afe->create_from_config(config);
  afe_config_free(config);
  if (!state->afeData) return fail("Could not create the audio front-end; check free PSRAM");
  state->afe->print_pipeline(state->afeData);

  state->multiNet = esp_mn_handle_from_name(modelName);
  if (!state->multiNet) return fail("MultiNet 7 is unavailable in this Arduino SDK");
  state->multiNetData = state->multiNet->create(modelName, VoiceConfig::kRecognitionTimeoutMs);
  if (!state->multiNetData) return fail("Could not create MultiNet; check free PSRAM");
  state->multiNet->set_det_threshold(state->multiNetData, VoiceConfig::kMinimumConfidence);
  state->feedSamples = state->afe->get_feed_chunksize(state->afeData);
  state->fetchSamples = state->afe->get_fetch_chunksize(state->afeData);
  if (state->feedSamples <= 0 || state->feedSamples > 2048 || state->fetchSamples <= 0
      || state->fetchSamples > 2048
      || state->fetchSamples != state->multiNet->get_samp_chunksize(state->multiNetData)
      || state->afe->get_feed_channel_num(state->afeData) != int(BoardMicrophone::kMicrophoneChannels)
      || state->afe->get_samp_rate(state->afeData) != int(VoiceConfig::kSampleRate)
      || state->multiNet->get_samp_rate(state->multiNetData) != int(VoiceConfig::kSampleRate))
    return fail("Audio and MultiNet frame formats do not match");

  if (esp_mn_commands_alloc(state->multiNet, state->multiNetData) != ESP_OK)
    return fail("Could not allocate the speech command table");
  state->commandsAllocated = true;
  for (const auto &entry : kPhrases) {
    // Runtime G2P is logged for comparison only. Recognition uses the reviewed
    // offline table, with the imperative REED pronunciation of "read info".
    char *legacy = flite_g2p(entry.text, 1);
    Serial.printf("{\"event\":\"voice_phrase\",\"command_id\":%u,\"text\":\"%s\",\"phonemes\":\"%s\",\"runtime_phonemes\":\"%s\"}\n",
        unsigned(entry.command), entry.text, entry.phonemes, legacy ? legacy : "");
    free(legacy);
    esp_err_t result = esp_mn_commands_phoneme_add(int(entry.command), entry.text, entry.phonemes);
    if (result != ESP_OK) return fail("Could not add a speech command");
  }
  const esp_mn_error_t *commandErrors = esp_mn_commands_update();
  if (commandErrors) {
    for (int i = 0; i < commandErrors->num; ++i)
      Serial.printf("Voice command rejected by model: %s\n", commandErrors->phrases[i]->string);
    return fail("The speech model rejected a command; check the serial log");
  }
  // Validate the SDK's registered table after FST update, before capture starts.
  // Missing or altered phrases must be a visible startup error.
  for (size_t i = 0; i < WeatherVoicePhrases::kCount; ++i) {
    const esp_mn_phrase_t *stored = esp_mn_commands_get_from_index(int(i));
    const auto &expected = kPhrases[i];
    if (!stored || !stored->string || !stored->phonemes
        || stored->command_id != int(expected.command)
        || strcmp(stored->string, expected.text) != 0
        || strcmp(stored->phonemes, expected.phonemes) != 0)
      return fail("Registered speech commands differ from the pronunciation table");
  }
  if (esp_mn_commands_get_from_index(int(WeatherVoicePhrases::kCount)))
    return fail("Unexpected extra speech commands in the registered table");
  state->loadedPhrases = WeatherVoicePhrases::kCount;
  Serial.printf("{\"event\":\"voice_phrase_table\",\"loaded\":%u}\n", unsigned(state->loadedPhrases));
  esp_mn_active_commands_print();
  state->multiNet->clean(state->multiNetData);
  state->input = static_cast<int16_t *>(heap_caps_malloc(
      state->feedSamples * BoardMicrophone::kMicrophoneChannels * sizeof(int16_t),
      MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
  if (!state->input) return fail("No internal memory for microphone frames");
  state->recognitionInput = static_cast<int16_t *>(heap_caps_malloc(
      state->fetchSamples * sizeof(int16_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
  if (!state->recognitionInput) return fail("No internal memory for recognition frames");
  if (!state->microphone.begin()) return fail(state->microphone.lastError());

  // Tasks wait for notification until both have been created successfully.
  if (xTaskCreatePinnedToCore(State::feedAudio, "voice-feed", 4096, state, 4,
      &state->feedTask, 0) != pdPASS) return fail("Could not start the microphone task");
  if (xTaskCreatePinnedToCore(State::detectCommands, "voice-detect", 8192, state, 4,
      &state->detectTask, 1) != pdPASS) return fail("Could not start the recognition task");
  state_ = state;
  startupError_[0] = '\0';
  state_->running.store(true);
  xTaskNotifyGive(state_->detectTask);
  xTaskNotifyGive(state_->feedTask);
  return true;
}

bool VoiceControl::take(VoiceEvent &event) {
  if (!state_ || !state_->running.load()) return false;
  if (state_->audioDiscontinuityEpoch.load() != state_->recognitionDiscontinuityEpoch.load()) {
    state_->discardEvents();
    return false;
  }
  if (state_->playbackSuppressed.load()) {
    state_->discardEvents();
    return false;
  }
  return xQueueReceive(state_->events, &event, 0) == pdTRUE;
}

bool VoiceControl::takeDiagnostic(VoiceDiagnostic &diagnostic) {
  if (!state_ || !state_->diagnostics) return false;
  return xQueueReceive(state_->diagnostics, &diagnostic, 0) == pdTRUE;
}

bool VoiceControl::pauseForPlayback() {
  if (!state_) return true;  // Disabled or startup failed; no capture to release.
  State *self = state_;
  if (!self->running.load()) return false;
  if (self->microphoneReleased) return true;
  uint32_t token = ++self->nextPauseToken;
  if (!token) token = ++self->nextPauseToken;
  self->playbackSuppressed.store(self->playbackEpoch.fetch_add(1) + 1);
  self->capturePauseRequested.store(token);
  self->discardEvents();
  const uint32_t started = millis();
  while (self->running.load() && self->capturePaused.load() != token
      && uint32_t(millis() - started) < kPlaybackPauseTimeoutMs) delay(1);
  if (!self->running.load()) return false;
  if (self->capturePaused.load() != token) {
    // Leave the capture hardware alone on timeout. Recognition clears itself
    // after a quiet tail; neither worker is killed or its resources destroyed.
    self->playbackResumedMs.store(millis());
    self->playbackSuppressed.store(self->playbackEpoch.fetch_add(1) + 1);
    self->capturePauseRequested.store(0);
    return false;
  }
  self->microphone.end();
  self->microphoneReleased = true;
  if (self->microphone.lastError()[0]) {
    self->stopWithFault(RuntimeFault::MicrophoneSuspend);
    return false;
  }
  return true;
}

bool VoiceControl::resumeAfterPlayback() {
  if (!state_) return true;
  State *self = state_;
  if (!self->running.load()) return false;
  if (!self->microphoneReleased) return self->capturePauseRequested.load() == 0;
  // Feed remains paused during shared-Wire codec setup and startup draining.
  if (!self->microphone.begin()) {
    self->stopWithFault(RuntimeFault::MicrophoneResume);
    return false;
  }
  self->microphoneReleased = false;
  self->discardEvents();
  self->playbackResumedMs.store(millis());
  self->playbackSuppressed.store(self->playbackEpoch.fetch_add(1) + 1);
  self->capturePauseRequested.store(0);
  return true;
}

VoiceStatus VoiceControl::status() const {
  VoiceStatus result;
  result.error = startupError_;
  if (!state_) return result;
  result.ready = state_->running.load();
  result.playbackSuspended = state_->playbackSuppressed.load();
  result.error = faultText(state_->fault.load());
  result.audioFrames = state_->audioFrames.load();
  result.feedBackpressureFrames = state_->feedBackpressureFrames.load();
  result.recognitionFrames = state_->recognitionFrames.load();
  result.microphonePeak = state_->peak.load();
  result.microphoneRms = state_->rms.load();
  result.mic1Peak = state_->mic1Peak.load();
  result.mic1Rms = state_->mic1Rms.load();
  result.mic2Peak = state_->mic2Peak.load();
  result.mic2Rms = state_->mic2Rms.load();
  result.clippedSamples = state_->clipped.load();
  result.accepted = state_->accepted.load();
  result.rejected = state_->rejected.load();
  result.droppedEvents = state_->dropped.load();
  result.timeouts = state_->timeouts.load();
  result.sessionsStarted = state_->sessionsStarted.load();
  result.sessionsWithoutCommand = state_->sessionsWithoutCommand.load();
  result.modelFrames = state_->modelFrames.load();
  result.loadedPhrases = state_->loadedPhrases;
  result.diagnosticsDropped = state_->diagnosticsDropped.load();
  return result;
}

#else

bool VoiceControl::begin() {
  attempted_ = true;
  snprintf(startupError_, sizeof(startupError_), "Disabled in VoiceConfig.h");
  return false;
}
bool VoiceControl::take(VoiceEvent &) { return false; }
bool VoiceControl::takeDiagnostic(VoiceDiagnostic &) { return false; }
bool VoiceControl::pauseForPlayback() { return true; }
bool VoiceControl::resumeAfterPlayback() { return true; }
VoiceStatus VoiceControl::status() const {
  VoiceStatus result;
  result.error = startupError_;
  return result;
}

#endif
