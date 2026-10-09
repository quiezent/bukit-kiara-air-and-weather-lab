"""Exercise production voice lifecycle, parked workers and ReadoutPlayer offline.

The actual State declarations, feed/detect workers, main-loop lifecycle methods,
quiet-tail implementation and complete ReadoutPlayer.cpp are compiled verbatim.
Only SDK/clock/task/codec boundaries are deterministic substitutes; no second
implementation of recovery policy is used. No ESP32 SDK, USB or device is used.
"""

import argparse
from pathlib import Path
import re
import subprocess
import sys


HARNESS = r"""
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <functional>
#include <initializer_list>
#include <string>
#include <vector>
using i2s_chan_handle_t = void *;
using TaskHandle_t = void *;
using TickType_t = uint32_t;
constexpr int pdTRUE = 1, pdPASS = 1, ESP_OK = 0;
constexpr uint32_t portMAX_DELAY = UINT32_MAX;
#define pdMS_TO_TICKS(value) (value)
using std::sqrt;
static uint32_t clockMs = 1000;
static size_t checks = 0;
static void check(bool valid, const char *why) {
  ++checks;
  if (!valid) { std::fprintf(stderr, "FAILED at %u ms: %s\n", clockMs, why); std::abort(); }
}
uint32_t millis() { return clockMs; }
int64_t esp_timer_get_time() { return int64_t(clockMs) * 1000; }
static std::function<void()> delayHook;
void delay(uint32_t amount) { clockMs += amount; if (delayHook) delayHook(); }
void vTaskDelay(uint32_t amount) { delay(amount); }
struct ParkReached {};
static unsigned notificationTakes = 0, notifications = 0, deletedTasks = 0;
static std::function<bool(unsigned)> notificationHook;
uint32_t ulTaskNotifyTake(int, uint32_t) {
  const unsigned take = ++notificationTakes;
  if (take == 1 || (notificationHook && notificationHook(take))) return 1;
  throw ParkReached{}; // Yield a genuinely parked persistent task to the host.
}
void xTaskNotifyGive(TaskHandle_t) { ++notifications; }
void vTaskDelete(TaskHandle_t) { ++deletedTasks; }
static void (*speakerTask)(void *) = nullptr;
static void *speakerArgument = nullptr;
static bool createTaskFails = false;
int xTaskCreatePinnedToCore(void (*task)(void *), const char *, unsigned, void *argument,
                          unsigned, TaskHandle_t *, unsigned) {
  if (createTaskFails) return 0;
  speakerTask = task; speakerArgument = argument; return pdPASS;
}
struct FakeQueue { std::deque<std::vector<uint8_t>> values; unsigned resets = 0; };
using QueueHandle_t = FakeQueue *;
int xQueueReceive(QueueHandle_t queue, void *out, uint32_t) {
  if (queue->values.empty()) return 0;
  std::memcpy(out, queue->values.front().data(), queue->values.front().size());
  queue->values.pop_front(); return pdTRUE;
}
int xQueueSend(QueueHandle_t, const void *, uint32_t) { return pdTRUE; }
int xQueueReset(QueueHandle_t queue) { queue->values.clear(); ++queue->resets; return pdTRUE; }
struct srmodel_list_t {};
struct esp_afe_sr_data_t {};
struct model_iface_data_t {};
enum esp_mn_state_t { ESP_MN_STATE_DETECTING, ESP_MN_STATE_DETECTED, ESP_MN_STATE_TIMEOUT };
constexpr int VAD_SPEECH = 1;
struct esp_mn_results_t {
  int state = 0, num = 0, command_id[5]{}, phrase_id[5]{};
  float prob[5]{};
  char string[256]{}, raw_string[256]{};
};
struct afe_fetch_result_t {
  int ret_value = ESP_OK, data_size = 512, vad_state = 0, vad_cache_size = 0;
  int trigger_channel_id = 0, raw_data_channels = 3;
  float data_volume = -70;
  int16_t *data = nullptr, *vad_cache = nullptr;
};
struct esp_afe_sr_iface_t {
  int (*feed)(esp_afe_sr_data_t *, const int16_t *);
  afe_fetch_result_t *(*fetch_with_delay)(esp_afe_sr_data_t *, uint32_t);
  int (*reset_buffer)(esp_afe_sr_data_t *);
  int (*reset_vad)(esp_afe_sr_data_t *);
};
struct esp_mn_iface_t {
  void (*clean)(model_iface_data_t *);
  esp_mn_state_t (*detect)(model_iface_data_t *, int16_t *);
  const esp_mn_results_t *(*get_results)(model_iface_data_t *);
};
@@HEADERS@@
@@RUNTIME@@
using RuntimeFault = VoiceRuntime::Fault;
using PlaybackQuietTail = VoiceRuntime::PlaybackQuietTail;
@@CONSTANTS@@
@@FAULT_TEXT@@
@@STATE@@
@@METHODS@@
@@READOUT@@

static unsigned microphoneBegins = 0, microphoneEnds = 0, microphoneReads = 0;
static unsigned speakerBegins = 0, speakerEnds = 0, speakerWrites = 0, speakerMutes = 0;
static unsigned resetBuffers = 0, resetVads = 0, modelCleans = 0;
static int resetBufferResult = 1, resetVadResult = 1, feedResult = 1;
static bool microphoneBeginFails = false, microphoneReleaseFails = false, readInFlight = false;
static bool speakerOwned = false, speakerBeginFails = false, speakerWriteFails = false;
static bool speakerWriteActive = false;
static size_t clipSampleCount = 64;
static std::function<void()> readHook, feedHook, fetchHook, writeHook;
static bool missingFetch = false;
static unsigned feedCalls = 0, fetchCalls = 0;
static int16_t frameSamples[256]{};
static afe_fetch_result_t fetchFrame;
static esp_mn_results_t results;
BoardMicrophone::~BoardMicrophone() = default;
bool BoardMicrophone::begin() {
  ++microphoneBegins; check(!speakerOwned, "microphone never begins while speaker owns pins");
  if (microphoneBeginFails) return false;
  rx_ = reinterpret_cast<void *>(1); ready_ = enabled_ = true; error_[0] = 0; return true;
}
void BoardMicrophone::end() {
  ++microphoneEnds; check(!readInFlight, "microphone is never released during a capture read");
  if (!microphoneReleaseFails) { rx_ = nullptr; ready_ = enabled_ = false; }
}
size_t BoardMicrophone::readInterleaved(int16_t *destination, size_t frames, uint32_t) {
  ++microphoneReads; readInFlight = true;
  if (readHook) readHook();
  std::fill(destination, destination + frames * kMicrophoneChannels, int16_t(0));
  clockMs += uint32_t(frames * 1000 / kSampleRate);
  readInFlight = false; return frames;
}
BoardSpeaker::~BoardSpeaker() = default;
bool BoardSpeaker::begin() { ++speakerBegins; if (speakerBeginFails) return false; speakerOwned = true; return true; }
void BoardSpeaker::mute() { ++speakerMutes; }
void BoardSpeaker::end() {
  check(!speakerWriteActive, "speaker cleanup never overlaps an active PCM write");
  ++speakerEnds; speakerOwned = false;
}
size_t BoardSpeaker::write(const int16_t *, size_t samples, uint32_t) {
  speakerWriteActive = true;
  ++speakerWrites; if (writeHook) writeHook();
  speakerWriteActive = false;
  if (speakerWriteFails) { std::strcpy(error_, "scripted speaker failure"); return 0; }
  return samples;
}
SpeechClipData speechClipData(SpeechClip) { static const int16_t pcm[4096]{}; return {pcm, clipSampleCount}; }
static int mockFeed(esp_afe_sr_data_t *, const int16_t *) { ++feedCalls; if (feedHook) feedHook(); return feedResult; }
static afe_fetch_result_t *mockFetch(esp_afe_sr_data_t *, uint32_t wait) {
  ++fetchCalls; delay(missingFetch ? wait : 16); if (fetchHook) fetchHook();
  return missingFetch ? nullptr : &fetchFrame;
}
static int mockResetBuffer(esp_afe_sr_data_t *) { ++resetBuffers; check(!readInFlight && !speakerOwned, "AFE reset occurs only outside capture/playback"); return resetBufferResult; }
static int mockResetVad(esp_afe_sr_data_t *) { ++resetVads; return resetVadResult; }
static void mockClean(model_iface_data_t *) { ++modelCleans; }
static esp_mn_state_t mockDetect(model_iface_data_t *, int16_t *) { return ESP_MN_STATE_DETECTING; }
static const esp_mn_results_t *mockResults(model_iface_data_t *) { return &results; }
static const esp_afe_sr_iface_t afe{mockFeed, mockFetch, mockResetBuffer, mockResetVad};
static const esp_mn_iface_t mn{mockClean, mockDetect, mockResults};
static void resetBoundary() {
  clockMs = 1000; delayHook = {}; readHook = {}; feedHook = {}; fetchHook = {}; writeHook = {};
  notificationHook = {}; notificationTakes = notifications = deletedTasks = 0;
  microphoneBegins = microphoneEnds = microphoneReads = 0;
  speakerBegins = speakerEnds = speakerWrites = speakerMutes = 0;
  resetBuffers = resetVads = modelCleans = feedCalls = fetchCalls = 0;
  resetBufferResult = resetVadResult = feedResult = 1;
  microphoneBeginFails = microphoneReleaseFails = readInFlight = false;
  speakerOwned = speakerBeginFails = speakerWriteFails = createTaskFails = missingFetch = false;
  speakerWriteActive = false; clipSampleCount = 64;
  speakerTask = nullptr; speakerArgument = nullptr; fetchFrame = {}; fetchFrame.data = frameSamples;
}
struct Fixture {
  VoiceControl voice;
  VoiceControl::State state;
  FakeQueue events, diagnostics;
  esp_afe_sr_data_t afeData;
  model_iface_data_t model;
  int16_t input[1024]{}, recognition[256]{};
  Fixture() {
    resetBoundary(); voice.state_ = &state; state.afe = &afe; state.afeData = &afeData;
    state.multiNet = &mn; state.multiNetData = &model; state.input = input; state.recognitionInput = recognition;
    state.events = &events; state.diagnostics = &diagnostics;
    state.feedSamples = 512; state.fetchSamples = 256;
    state.feedTask = reinterpret_cast<void *>(1); state.detectTask = reinterpret_cast<void *>(2);
    state.microphone.rx_ = reinterpret_cast<void *>(3);
    state.running.store(true); state.feedParked.store(false); state.detectParked.store(false);
  }
  void fault(RuntimeFault reason = RuntimeFault::AudioFeed) {
    state.lastFeedResult.store(0); state.lastFeedMs.store(clockMs - 10); state.lastFetchMs.store(clockMs - 30);
    state.stopWithFault(reason);
  }
  void parked() { state.feedParked.store(true); state.detectParked.store(true); }
};
static void worker(void (*run)(void *), VoiceControl::State &state) {
  notificationTakes = 0;
  try { run(&state); check(false, "worker must park rather than return/delete itself"); }
  catch (const ParkReached &) {}
  check(deletedTasks == 0, "persistent workers are never deleted on runtime faults");
}
static void gatesAndSnapshot() {
  Fixture f; f.fault();
  const auto first = f.voice.status();
  check(!first.ready && first.recoveryPending && first.lastFaultFeedResult == 0, "fault metadata is visible before stop");
  const uint32_t faultTime = first.lastFaultMs;
  clockMs += 25; f.state.lastFeedResult.store(1); f.state.stopWithFault(RuntimeFault::AudioFetch);
  check(f.state.lastFault.load() == RuntimeFault::AudioFeed && f.state.lastFaultMs.load() == faultTime
      && f.state.lastFaultFeedResult.load() == 0, "competing fault cannot overwrite the first snapshot");
  f.voice.update(false); check(microphoneEnds == 0 && resetBuffers == 0, "neither worker parked blocks release/reset");
  f.state.feedParked.store(true); f.voice.update(false);
  check(microphoneEnds == 0 && f.voice.status().recoveryPending, "missing detector acknowledgement keeps repair pending");
  f.state.detectParked.store(true); f.voice.update(true);
  check(microphoneEnds == 0 && f.state.recoveryAttempts == 0, "speaker busy defers all repair work");
  f.voice.update(false); const uint32_t due = f.state.recoveryDueMs;
  clockMs = due - 1; f.voice.update(false); check(resetBuffers == 0, "automatic repair observes initial backoff");
  clockMs = due; f.voice.update(false);
  check(f.state.running.load() && f.state.recoveryCount == 1 && f.state.recoveryAttempts == 1
      && !f.state.recoveryPending.load(), "both parked permit one successful in-place repair");
  check(resetBuffers == 1 && resetVads == 1 && modelCleans == 1 && microphoneBegins == 1
      && notifications == 2, "exact existing objects reset and both existing tasks are notified");
  check(f.state.feedTask == reinterpret_cast<void *>(1) && f.state.detectTask == reinterpret_cast<void *>(2), "worker handles remain intact");
  check(f.state.afeData == &f.afeData && f.state.multiNetData == &f.model, "model allocations remain intact");
  check(f.state.playbackSuppressed.load() && !f.state.feedParked.load() && !f.state.detectParked.load(), "restart needs fresh worker/quiet acknowledgements");
  check(f.voice.status().lastFaultFeedResult == 0 && f.voice.status().lastFaultFeedMs == first.lastFaultFeedMs,
      "original fault telemetry survives repair and later feed progress");
  // A fault claimed but not yet published must never initiate a reset.
  Fixture recording; recording.state.running.store(false); recording.state.faultClaimed.store(true);
  recording.state.recoveryPending.store(true); recording.parked(); recording.voice.update(false);
  check(resetBuffers == 0 && recording.state.recoveryAttempts == 0, "half-published fault cannot trigger recovery");
}
static void handoffAndSpeakerOnly() {
  Fixture f; f.fault();
  const uint32_t started = clockMs;
  delayHook = [&]() { if (uint32_t(clockMs - started) >= 220) f.state.feedParked.store(true); };
  check(f.voice.pauseForPlayback(), "faulted capture hands off after feed parks");
  check(clockMs == started + 220 && microphoneEnds == 1 && f.state.microphone.released(), "release waits for exact parked acknowledgement");
  check(f.voice.resumeAfterPlayback() && microphoneBegins == 0 && !f.voice.status().ready, "speaker-only resume preserves the recognition fault");
  delayHook = {}; f.parked();
  ReadoutPlayer player; SpeechPlaylist playlist; check(playlist.add(SpeechClip::N1), "valid playlist");
  check(player.start(playlist, f.voice) && player.busy(), "actual ReadoutPlayer starts while recognition is faulted");
  f.voice.update(player.busy()); check(f.state.recoveryAttempts == 0, "readout defers recovery");
  speakerTask(speakerArgument); player.update(f.voice);
  check(!player.busy() && player.completed() == 1 && !player.lastError()[0]
      && player.samplesWritten() == 64, "actual speaker-only playback completes successfully");
  check(!f.voice.status().ready && f.voice.status().recoveryPending, "speaker completion does not claim recognition recovery");
  check(f.voice.requestRecovery(), "manual recovery request accepted"); f.voice.update(false);
  check(f.voice.status().ready && f.voice.status().recoveryCount == 1, "repair starts once speaker is idle");
  Fixture timeout; timeout.fault(); const uint32_t before = clockMs;
  check(!timeout.voice.pauseForPlayback() && uint32_t(clockMs - before) == 500
      && microphoneEnds == 0, "unacknowledged faulted capture times out without release");
  Fixture healthy; const uint32_t healthyStart = clockMs;
  delayHook = [&]() { if (uint32_t(clockMs - healthyStart) >= 100) healthy.state.capturePaused.store(healthy.state.capturePauseRequested.load()); };
  check(healthy.voice.pauseForPlayback() && microphoneEnds == 1, "running capture uses matching pause generation");
  check(healthy.voice.resumeAfterPlayback() && microphoneBegins == 1
      && !healthy.state.capturePauseRequested.load() && healthy.state.playbackSuppressed.load(), "healthy capture resumes with a quiet tail");
  Fixture release; release.fault(); release.parked(); microphoneReleaseFails = true;
  check(!release.voice.pauseForPlayback() && !release.state.microphone.released(), "retained I2S channel prevents speaker handoff");
  Fixture oldError; oldError.fault(); oldError.parked(); std::strcpy(oldError.state.microphone.error_, "prior capture error");
  check(oldError.voice.pauseForPlayback(), "old capture error does not reject verified I2S release");
}
static void retriesAndManualDeferral() {
  for (unsigned failure = 0; failure < 5; ++failure) {
    Fixture f; f.fault(); f.parked(); f.voice.requestRecovery();
    if (failure == 0) resetBufferResult = 0;
    if (failure == 1) resetBufferResult = -1;
    if (failure == 2) resetVadResult = 0;
    if (failure == 3) resetVadResult = -1;
    if (failure == 4) microphoneBeginFails = true;
    f.voice.update(false);
    check(f.state.recoveryAttempts == 1 && f.state.recoveryFailures == 1 && !f.state.recoveryCount
        && f.state.recoveryPending.load() && !f.state.running.load(), "invalid reset result/startup remains faulted and pending");
    check(notifications == 0 && f.state.feedParked.load() && f.state.detectParked.load(), "failed repair never wakes workers");
  }
  Fixture budget; budget.fault(); budget.parked(); resetBufferResult = 0;
  budget.voice.update(false); clockMs = budget.state.recoveryDueMs; const uint32_t firstAttempt = clockMs;
  budget.voice.update(false);
  check(budget.state.recoveryDueMs == clockMs + 5000, "first failed repair backs off five seconds");
  clockMs = budget.state.recoveryDueMs; budget.voice.update(false);
  check(budget.state.recoveryDueMs == clockMs + 15000, "second failed repair backs off fifteen seconds");
  clockMs = budget.state.recoveryDueMs; budget.voice.update(false);
  check(budget.state.recoveryAttempts == 3, "three actual attempts counted");
  clockMs = firstAttempt + 59999; budget.voice.requestRecovery(); budget.voice.update(false);
  check(budget.state.recoveryAttempts == 3, "manual requests cannot bypass rolling sixty-second budget");
  clockMs = firstAttempt + 60000; resetBufferResult = 1; budget.voice.update(false);
  check(budget.state.recoveryAttempts == 4 && budget.state.recoveryCount == 1, "retry becomes eligible exactly when oldest attempt expires");
  Fixture manual; check(manual.voice.requestRecovery(), "healthy manual reset schedules a real stop"); manual.parked();
  manual.voice.update(true); check(!resetBuffers && manual.state.recoveryPending.load(), "manual reset is deferred during playback");
  manual.voice.update(false); check(manual.state.recoveryCount == 1, "deferred manual reset succeeds after playback");
  Fixture wrap; clockMs = UINT32_MAX - 500; wrap.fault(); wrap.parked(); wrap.voice.update(false);
  clockMs = wrap.state.recoveryDueMs; wrap.voice.update(false);
  check(wrap.state.recoveryCount == 1, "automatic delay survives millis rollover");
}
static void actualWorkersAndQuietTail() {
  Fixture f; feedResult = 0;
  worker(VoiceControl::State::feedAudio, f.state);
  check(f.state.feedParked.load() && !f.state.running.load() && f.state.lastFault.load() == RuntimeFault::AudioFeed,
      "actual feed worker parks on sustained real-boundary backpressure");
  check(feedCalls >= 90 && f.state.lastFaultFeedResult.load() == 0 && !f.state.diagnosticStallUntilMs.load(), "feed health publishes original zero result and cancels stall");
  worker(VoiceControl::State::detectCommands, f.state);
  check(f.state.detectParked.load() && fetchCalls == 0, "actual detector parks before calling SDK after peer fault");
  Fixture inRead;
  readHook = [&]() {
    inRead.state.stopWithFault(RuntimeFault::Microphone);
    check(!inRead.voice.pauseForPlayback() && microphoneEnds == 0, "fault during actual capture cannot release the in-flight read");
    readHook = {};
  };
  worker(VoiceControl::State::feedAudio, inRead.state);
  check(inRead.state.feedParked.load() && !readInFlight && feedCalls == 0, "read returns before actual worker publishes parking");
  check(inRead.voice.pauseForPlayback(), "handoff becomes safe after actual reader parks");
  Fixture stalled; check(!stalled.voice.diagnosticStall(999) && !stalled.voice.diagnosticStall(10001), "stall diagnostic enforces bounded interval");
  check(stalled.voice.diagnosticStall(10000), "bounded USB stall starts"); const uint32_t begun = clockMs;
  delayHook = [&]() { if (uint32_t(clockMs - begun) >= 200) stalled.state.stopWithFault(RuntimeFault::AudioFeed); };
  worker(VoiceControl::State::detectCommands, stalled.state);
  check(stalled.state.detectParked.load() && uint32_t(clockMs - begun) == 200 && fetchCalls == 0,
      "actual diagnostic hold ends promptly on fault and parks without fetch");
  Fixture fetch; missingFetch = true; worker(VoiceControl::State::detectCommands, fetch.state);
  check(fetch.state.detectParked.load() && fetch.state.lastFault.load() == RuntimeFault::AudioFetch,
      "actual fetch liveness fault parks detector");
  Fixture tail; tail.state.playbackSuppressed.store(1); tail.state.playbackEpoch.store(1);
  tail.state.playbackResumedMs.store(clockMs); const uint32_t resumed = clockMs;
  bool clearedAfterQuiet = false;
  fetchHook = [&]() {
    if (!tail.state.playbackSuppressed.load()) { clearedAfterQuiet = true; tail.state.stopWithFault(RuntimeFault::RecoveryRequested); }
  };
  worker(VoiceControl::State::detectCommands, tail.state);
  check(clearedAfterQuiet && uint32_t(clockMs - resumed) >= 600 && tail.state.modelFrames.load() == 0,
      "actual detector suppresses model execution until elapsed and accumulated silence pass");
  PlaybackQuietTail quiet; quiet.reset();
  check(!quiet.observe(false, 9600, 1100, 1000), "buffered silence alone cannot clear tail early");
  check(quiet.observe(false, 1, 1600, 1000), "both elapsed and silence permit release");
  quiet.reset(); quiet.observe(false, 9000, 2000, 1000); quiet.observe(true, 1000, 2000, 1000);
  check(!quiet.observe(false, 1000, 2000, 1000), "speech resets consecutive quiet samples");
}
static void taskAndSpeakerFailures() {
  Fixture f; f.fault(); f.parked();
  ReadoutPlayer player; SpeechPlaylist playlist; playlist.add(SpeechClip::N0);
  createTaskFails = true;
  check(!player.start(playlist, f.voice) && !player.busy() && !speakerOwned && f.state.recoveryPending.load(),
      "actual task creation failure releases speaker without losing recognition fault");
  createTaskFails = false; speakerWriteFails = true;
  check(player.start(playlist, f.voice), "faulted voice does not block another readout attempt");
  speakerTask(speakerArgument); player.update(f.voice);
  check(!player.completed() && !player.busy() && std::strstr(player.lastError(), "scripted speaker"), "speaker error still fails actual readout");
  Fixture during; const uint32_t started = clockMs;
  delayHook = [&]() { if (uint32_t(clockMs - started) >= 1) during.state.capturePaused.store(during.state.capturePauseRequested.load()); };
  ReadoutPlayer active; check(active.start(playlist, during.voice), "healthy readout starts"); delayHook = {};
  writeHook = [&]() { during.fault(); during.parked(); during.voice.update(active.busy()); };
  speakerTask(speakerArgument); active.update(during.voice);
  check(active.completed() == 1 && !active.lastError()[0] && during.state.recoveryAttempts == 0,
      "fault during speaker playback leaves readout successful and defers repair");
}
static void cancelledReadout() {
  Fixture before; before.fault(); before.parked();
  ReadoutPlayer queued; SpeechPlaylist playlist;
  playlist.add(SpeechClip::N0); playlist.add(SpeechClip::N1); playlist.add(SpeechClip::N2);
  check(!queued.stop() && !queued.stopRequested(), "stopping an idle player is a no-op");
  check(queued.start(playlist, before.voice), "cancel-before-start readout begins");
  const uint32_t stoppedAt = clockMs;
  check(queued.stop() && queued.stopRequested() && speakerMutes == 1,
      "stop immediately mutes while publishing cancellation");
  check(queued.stop() && speakerMutes == 1 && !queued.stopped(),
      "repeated cancellation is idempotent before cleanup");
  queued.update(before.voice); before.voice.update(queued.busy());
  check(queued.busy() && !speakerEnds && !microphoneBegins && !before.state.recoveryAttempts,
      "stop does not release I2S or resume/recover while the worker is pending");
  speakerTask(speakerArgument);
  check(!speakerWrites && clockMs == stoppedAt,
      "cancel before worker start skips all clips, silence and drain delay");
  queued.update(before.voice);
  check(!queued.busy() && !queued.stopRequested() && queued.stopped() == 1 && !queued.completed() && !queued.lastError()[0]
      && speakerEnds == 1 && !microphoneBegins,
      "intentional stop cleans up exactly once without a completion or error");
  queued.update(before.voice);
  check(queued.stopped() == 1 && speakerEnds == 1 && !queued.stop(), "idle cleanup and stop cannot double count");
  before.voice.update(false); delay(1000); before.voice.update(false);
  check(before.voice.status().ready && before.voice.status().recoveryCount == 1,
      "faulted recognition recovers only after cancelled playback becomes idle");
  delayHook = [&]() { before.state.capturePaused.store(before.state.capturePauseRequested.load()); };
  check(queued.start(playlist, before.voice) && !queued.stopRequested(), "next readout clears cancellation");
  delayHook = {}; speakerTask(speakerArgument); queued.update(before.voice);
  check(!queued.busy() && queued.completed() == 1 && queued.stopped() == 1
      && queued.samplesWritten() == 192 && !queued.lastError()[0], "full playback succeeds after an intentional stop");

  Fixture within; within.fault(); within.parked(); clipSampleCount = 3200;
  ReadoutPlayer writing;
  check(writing.start(playlist, within.voice), "mid-write cancellation readout begins");
  const uint32_t writeStarted = clockMs;
  writeHook = [&]() {
    check(writing.stop() && speakerMutes == 1 && speakerWriteActive, "stop can mute during an in-flight write");
    writing.update(within.voice); within.voice.update(writing.busy());
    check(writing.busy() && !speakerEnds && speakerOwned && !within.state.recoveryAttempts,
        "in-flight cancellation waits for the writer acknowledgement");
  };
  speakerTask(speakerArgument); writeHook = {};
  check(speakerWrites == 1 && writing.samplesWritten() == 1600 && clockMs == writeStarted,
      "mid-write stop keeps accepted sample count and skips remaining chunks/clips/tail");
  writing.update(within.voice);
  check(!writing.busy() && writing.stopped() == 1 && !writing.completed() && !writing.lastError()[0],
      "mid-write stop is a successful cancellation");

  Fixture tail; tail.fault(); tail.parked();
  ReadoutPlayer flushing; SpeechPlaylist shortPlaylist; shortPlaylist.add(SpeechClip::N0);
  check(flushing.start(shortPlaylist, tail.voice), "tail cancellation readout begins");
  const uint32_t tailStarted = clockMs;
  writeHook = [&]() { if (speakerWrites == 2) check(flushing.stop(), "stop accepted during first silence write"); };
  speakerTask(speakerArgument); writeHook = {}; flushing.update(tail.voice);
  check(speakerWrites == 2 && flushing.samplesWritten() == 64 && clockMs == tailStarted
      && flushing.stopped() == 1 && !flushing.completed() && !flushing.lastError()[0],
      "tail stop skips remaining silence and normal drain without changing PCM count");

  Fixture failure; failure.fault(); failure.parked();
  ReadoutPlayer broken; check(broken.start(shortPlaylist, failure.voice), "concurrent-error cancellation begins");
  writeHook = [&]() { check(broken.stop(), "cancellation can accompany a write error"); speakerWriteFails = true; };
  speakerTask(speakerArgument); writeHook = {}; broken.update(failure.voice);
  check(!broken.busy() && !broken.completed() && broken.stopped() == 1
      && std::strstr(broken.lastError(), "scripted speaker"), "a real write failure is retained despite user cancellation");

  Fixture late; late.fault(); late.parked();
  ReadoutPlayer finished; check(finished.start(shortPlaylist, late.voice), "natural completion test begins");
  speakerTask(speakerArgument);
  check(!finished.stop() && !finished.stopRequested() && !speakerMutes,
      "a worker already done naturally is not reclassified as stopped");
  finished.update(late.voice);
  check(finished.completed() == 1 && !finished.stopped(), "late click preserves natural completion count");

  Fixture during; const uint32_t started = clockMs;
  delayHook = [&]() { if (uint32_t(clockMs - started) >= 1) during.state.capturePaused.store(during.state.capturePauseRequested.load()); };
  ReadoutPlayer active; check(active.start(shortPlaylist, during.voice), "healthy playback for fault-during-stop begins"); delayHook = {};
  writeHook = [&]() {
    during.fault(); during.parked(); check(active.stop(), "stop accepted when recognition faults during playback");
    active.update(during.voice); during.voice.update(active.busy());
    check(!during.state.recoveryAttempts && speakerOwned && !speakerEnds,
        "fault-during-stop cannot reset/release the active writer");
  };
  speakerTask(speakerArgument); writeHook = {}; active.update(during.voice);
  check(!active.busy() && active.stopped() == 1 && !active.completed() && !active.lastError()[0]
      && !during.state.recoveryAttempts && during.state.recoveryPending.load(),
      "cancelled speaker completion preserves pending voice fault");
  during.voice.update(active.busy()); delay(1000); during.voice.update(active.busy());
  check(during.voice.status().ready && during.voice.status().recoveryCount == 1,
      "deferred voice recovery resumes after fault-during-stop cleanup");
}
int main() {
  gatesAndSnapshot(); handoffAndSpeakerOnly(); retriesAndManualDeferral();
  actualWorkersAndQuietTail(); taskAndSpeakerFailures(); cancelledReadout();
  std::printf("PASS: %zu checks on production lifecycle methods, actual persistent feed/detect workers, quiet suppression and ReadoutPlayer; no device/SDK/model recreation\n", checks);
}
"""


def extract(source, signature):
    start = source.index(signature)
    opening = source.index("{", start)
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def header(path):
    value = path.read_text(encoding="utf-8")
    value = re.sub(r"^#(?:include|pragma).*$", "", value, flags=re.MULTILINE)
    return value.replace(" private:", " public:")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", required=True)
    parser.add_argument("--cxx-arg", action="append", default=[])
    parser.add_argument("--build-dir", required=True, type=Path)
    args = parser.parse_args()
    firmware = Path(__file__).resolve().parents[1]
    build = args.build_dir.resolve()
    build.mkdir(parents=True, exist_ok=True)
    voice = (firmware / "VoiceControl.cpp").read_text(encoding="utf-8")
    headers = "\n".join((f'#include "{(firmware / name).as_posix()}"' if name == "VoiceConfig.h"
                         else header(firmware / name)) for name in (
        "VoiceConfig.h", "VoiceNavigation.h", "VoiceDiagnostic.h", "VoiceControl.h",
        "BoardMicrophone.h", "BoardSpeaker.h", "SpeechClips.h", "WeatherReadout.h", "ReadoutPlayer.h",
        "VoiceSession.h", "VoiceFeedHealth.h"))
    declarations = voice[voice.index("struct VoiceControl::State {"):voice.index("  // Used only if initialization fails")]
    state_methods = "\n".join(extract(voice, signature) for signature in (
        "  void stopWithFault(", "  void discardEvents(", "  static void feedAudio(", "  static void detectCommands("))
    methods = "\n".join(extract(voice, signature) for signature in (
        "bool VoiceControl::pauseForPlayback(", "bool VoiceControl::resumeAfterPlayback(",
        "bool VoiceControl::requestRecovery(", "bool VoiceControl::diagnosticStall(",
        "void VoiceControl::update(", "VoiceStatus VoiceControl::status("))
    readout = (firmware / "ReadoutPlayer.cpp").read_text(encoding="utf-8")
    readout = re.sub(r"^#include.*$", "", readout, flags=re.MULTILINE)
    constants = "\n".join(re.findall(r"constexpr uint32_t k(?:PlaybackPauseTimeoutMs|RecoveryWindowMs|RecoveryBackoffMs)[^;]*;", voice))
    generated = (HARNESS.replace("@@HEADERS@@", headers)
                 .replace("@@RUNTIME@@", extract(voice, "namespace VoiceRuntime {"))
                 .replace("@@CONSTANTS@@", constants)
                 .replace("@@FAULT_TEXT@@", extract(voice, "const char *faultText("))
                 .replace("@@STATE@@", declarations + state_methods + "\n};")
                 .replace("@@METHODS@@", methods).replace("@@READOUT@@", readout))
    source = build / "voice_lifecycle_test.cpp"
    source.write_text(generated, encoding="utf-8")
    executable = build / ("voice_lifecycle_test.exe" if sys.platform == "win32" else "voice_lifecycle_test")
    subprocess.run([args.cxx, *args.cxx_arg, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    str(source), "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    main()
