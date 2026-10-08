"""Exercise the actual detector finishTrace lambda with a model and queue stub.

Verify zero-wait queue transport, full-queue drop accounting, snapshot ownership,
and the exact-SDK timeout case where a second getter would erase saved phonemes.
"""
import argparse
from pathlib import Path
import subprocess
import sys


HARNESS = r"""
#include "VoiceDiagnostic.h"
#include "esp_mn_iface.h"
#include <atomic>
#include <cassert>
#include <cstring>
#include <cstdio>
#include <deque>

constexpr int pdTRUE = 1;
uint32_t fakeNow = 0;
uint32_t millis() { return fakeNow; }
struct Queue { std::deque<VoiceDiagnostic> records; };
int xQueueSend(Queue *queue, const VoiceDiagnostic *trace, unsigned wait) {
  assert(wait == 0);
  if (queue->records.size() == 4) return 0;
  queue->records.push_back(*trace);
  return pdTRUE;
}
struct Model {
  esp_mn_results_t result{};
  unsigned getterCalls = 0;
  bool pathsReset = false;
};
esp_mn_results_t *getResults(model_iface_data_t *opaque) {
  auto *model = reinterpret_cast<Model *>(opaque);
  ++model->getterCalls;
  if (model->pathsReset) {
    // Pinned MN7 resets paths internally on TIMEOUT. Another getter sees empty
    // paths and overwrites the snapshot made by its internal model_reset().
    model->result.string[0] = model->result.raw_string[0] = '\0';
  }
  return &model->result;
}
struct State {
  esp_mn_iface_t *multiNet;
  model_iface_data_t *multiNetData;
  Queue *diagnostics;
  uint32_t nextDiagnosticSequence = 0;
  std::atomic<uint32_t> diagnosticsDropped{0};
};

int main() {
  esp_mn_iface_t iface{};
  iface.get_results = getResults;
  Model model;
  Queue queue;
  State state{&iface, reinterpret_cast<model_iface_data_t *>(&model), &queue};
  State *self = &state;
  VoiceDiagnostic trace;
  bool traceActive = false;
  uint32_t traceStartedMs = 0;
  const esp_mn_results_t *sessionResults = nullptr;
  @@FINISH_TRACE@@

  fakeNow = 300;
  traceStartedMs = 100;
  traceActive = true;
  trace.sessionId = 1;
  trace.modelFrames = 12;
  trace.detectState = ESP_MN_STATE_DETECTED;
  model.result.state = ESP_MN_STATE_DETECTED;
  model.result.num = 1;
  model.result.command_id[0] = 6;
  model.result.phrase_id[0] = 21;
  model.result.prob[0] = 0.87f;
  model.pathsReset = true; // Detection also resets internal paths before returning.
  std::strcpy(model.result.string, "RmD gNFb");
  std::strcpy(model.result.raw_string, "RmD gNFb");
  sessionResults = &model.result;
  finishTrace(VoiceDiagnosticReason::Detected);
  assert(model.getterCalls == 0 && !traceActive && !sessionResults);
  assert(queue.records.size() == 1);
  assert(queue.records.back().sessionDurationMs == 200 && queue.records.back().sequence == 1);
  assert(queue.records.back().candidateCount == 1 && queue.records.back().commandIds[0] == 6);
  std::memset(&model.result, 0, sizeof(model.result)); // Actual caller now cleans.
  assert(!std::strcmp(queue.records.back().rawString, "RmD gNFb"));

  trace = VoiceDiagnostic{};
  trace.detectState = ESP_MN_STATE_TIMEOUT;
  traceActive = true;
  traceStartedMs = UINT32_MAX - 99;
  fakeNow = 20;
  model.getterCalls = 0;
  model.pathsReset = true;
  model.result.state = ESP_MN_STATE_TIMEOUT;
  model.result.num = 5; // Stale arrays cannot become unmatched candidates.
  std::strcpy(model.result.string, "pre reset constrained");
  std::strcpy(model.result.raw_string, "pre reset raw");
  sessionResults = &model.result;
  finishTrace(VoiceDiagnosticReason::Timeout);
  assert(model.getterCalls == 0 && queue.records.size() == 2);
  assert(queue.records.back().sessionDurationMs == 120);
  assert(!std::strcmp(queue.records.back().rawString, "pre reset raw"));
  assert(!queue.records.back().candidatesValid && !queue.records.back().candidateCount);
  assert(queue.records.back().modelState == ESP_MN_STATE_TIMEOUT);

  model.pathsReset = false;
  model.result.state = ESP_MN_STATE_DETECTING;
  std::strcpy(model.result.raw_string, "partial phonemes");
  traceActive = true;
  sessionResults = &model.result;
  trace.detectState = ESP_MN_STATE_DETECTING;
  finishTrace(VoiceDiagnosticReason::SilenceTail);
  assert(model.getterCalls == 1 && queue.records.size() == 3);
  assert(queue.records.back().reason == VoiceDiagnosticReason::SilenceTail);
  assert(!queue.records.back().candidateCount);
  assert(!std::strcmp(queue.records.back().rawString, "partial phonemes"));

  traceActive = true;
  finishTrace(VoiceDiagnosticReason::Rejected);
  assert(queue.records.size() == 4 && state.diagnosticsDropped.load() == 0);
  traceActive = true;
  finishTrace(VoiceDiagnosticReason::SilenceTail);
  assert(queue.records.size() == 4 && state.diagnosticsDropped.load() == 1);
  assert(!traceActive && state.nextDiagnosticSequence == 5);
  const unsigned calls = model.getterCalls;
  finishTrace(VoiceDiagnosticReason::SilenceTail); // No active session, no getter/queue work.
  assert(model.getterCalls == calls && state.nextDiagnosticSequence == 5);
  puts("PASS: production diagnostic publish, zero-wait queue/drop accounting, capture before clean, timeout snapshot retention and millis rollover");
}
"""


def main():
    firmware = Path(__file__).resolve().parents[1]
    package = firmware.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cxx', required=True)
    parser.add_argument('--cxx-arg', action='append', default=[])
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--esp-sr-include', type=Path,
                        default=package / '.state/arduino-data/packages/esp32/tools/esp32s3-libs/3.3.12/include/espressif__esp-sr/include/esp32s3',
                        help='Pinned ESP-SR target include directory containing esp_mn_iface.h')
    args = parser.parse_args()
    sdk = args.esp_sr_include.resolve()
    if not (sdk / 'esp_mn_iface.h').is_file():
        parser.error(f'Required SDK header not found: {sdk / "esp_mn_iface.h"}')
    production = (firmware / 'VoiceControl.cpp').read_text(encoding='utf-8')
    start = production.index('auto finishTrace =')
    brace = production.index('{', start)
    depth, end = 1, brace + 1
    while depth:
        if production[end] == '{':
            depth += 1
        elif production[end] == '}':
            depth -= 1
        end += 1
    assert production[end] == ';'
    implementation = production[start:end + 1]
    build = args.build_dir.resolve()
    build.mkdir(parents=True, exist_ok=True)
    source = build / 'voice_diagnostic_publish_test.cpp'
    source.write_text(HARNESS.replace('@@FINISH_TRACE@@', implementation), encoding='utf-8')
    suffix = '.exe' if sys.platform == 'win32' else ''
    executable = build / ('voice_diagnostic_publish_test' + suffix)
    subprocess.run([args.cxx, *args.cxx_arg, '-std=c++17', '-Wall', '-Wextra', '-Werror',
                    '-I', str(firmware), '-I', str(sdk), str(source), '-o', str(executable)], check=True)
    subprocess.run([str(executable)], check=True)


if __name__ == '__main__':
    main()
