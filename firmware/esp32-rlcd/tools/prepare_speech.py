#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the public dashboard's 117 local speech clips with eSpeak NG 1.52.0.

Requires Python 3.10+ and an independently installed native eSpeak NG executable.
No Microsoft voice recordings, previous cache, cloud service or audio playback
is used. Only phrase text is input. Generated C++ and build/speech files are
build artifacts, and must stay outside version control.

The synthesizer is a separate GPL-3.0-or-later build tool; it is not linked into
the ESP32 firmware. See the package's third-party notices for output provenance.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import wave


PACKAGE = Path(__file__).resolve().parents[1]
VERSION = "1.52.0"
VOICE = "en-us"
RATE = 16000
SPEED = 165
PEAK = 31000
TRIM_THRESHOLD = 80
PADDING = RATE * 40 // 1000
FADE = RATE * 4 // 1000
RESAMPLER = "windowed-sinc-32-tap-q30-v1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_phrases(path: Path) -> list[dict[str, str]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    phrases = document["phrases"]
    if not isinstance(phrases, list) or len(phrases) != 117:
        raise ValueError("phrases.json must contain all 117 ordered public dashboard clips")
    names = []
    for phrase in phrases:
        if not isinstance(phrase, dict) or not isinstance(phrase.get("id"), str):
            raise ValueError("Every phrase requires an id and text")
        name, text = phrase["id"], phrase.get("text")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
            raise ValueError(f"Invalid clip ID: {name!r}")
        if not isinstance(text, str) or not text.strip() or "\x00" in text:
            raise ValueError(f"Invalid text for {name}")
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError("Clip IDs must be unique")
    header = PACKAGE / "WeatherDashboard" / "SpeechClips.h"
    if not header.is_file():
        raise ValueError(f"Missing {header}; use the complete firmware source package")
    body = header.read_text(encoding="utf-8").split("enum class SpeechClip", 1)[1]
    enum_ids = re.findall(r"^\s+(\w+),", body.split("Count", 1)[0], re.MULTILINE)
    if names != enum_ids:
        raise ValueError("Phrase IDs/order differ from WeatherDashboard/SpeechClips.h")
    return phrases


def find_engine(argument: str | None) -> Path:
    candidate = argument or shutil.which("espeak-ng")
    if not candidate:
        raise ValueError(
            "eSpeak NG 1.52.0 was not found. Install the native engine from "
            "https://github.com/espeak-ng/espeak-ng/releases/tag/1.52.0 "
            "and pass --espeak-ng PATH (or put espeak-ng on PATH)."
        )
    path = Path(candidate).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"eSpeak NG executable does not exist: {path}")
    return path


def engine_environment(engine: Path, data_argument: str | None) -> dict[str, str]:
    environment = dict(os.environ)
    if data_argument:
        data = Path(data_argument).expanduser().resolve()
        if not data.is_dir():
            raise ValueError(f"eSpeak NG data directory does not exist: {data}")
        environment["ESPEAK_DATA_PATH"] = str(data)
    elif (engine.parent / "espeak-ng-data").is_dir():
        # The Windows MSI can be extracted without installation. Its executable
        # otherwise looks for installed data and may fail before printing version.
        environment["ESPEAK_DATA_PATH"] = str(engine.parent / "espeak-ng-data")
    return environment


def engine_provenance(engine: Path, environment: dict[str, str]) -> dict:
    result = subprocess.run(
        [str(engine), "--version"], capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=environment, timeout=30,
    )
    output = result.stdout.strip()
    matched = re.search(r"text-to-speech:\s+(\S+)\s+Data at:\s+(.+)", output)
    if result.returncode or not matched:
        raise ValueError(f"eSpeak NG version check failed ({result.returncode}): {output} {result.stderr.strip()}")
    if matched.group(1) != VERSION:
        raise ValueError(f"Expected eSpeak NG {VERSION}; found {matched.group(1)}")
    data = Path(matched.group(2).strip()).resolve()
    if not data.is_dir():
        raise ValueError(f"eSpeak NG reported a missing data directory: {data}")
    data_hash = hashlib.sha256()
    data_files = []
    for file in sorted((f for f in data.rglob("*") if f.is_file()),
                       key=lambda f: f.relative_to(data).as_posix()):
        relative = file.relative_to(data).as_posix()
        value = digest(file)
        data_hash.update(relative.encode("utf-8") + b"\x00" + bytes.fromhex(value))
        data_files.append({"path": relative, "sha256": value})
    if not data_files:
        raise ValueError("eSpeak NG voice data directory is empty")
    binaries = {engine.name: digest(engine)}
    adjacent_library = engine.parent / "libespeak-ng.dll"
    if adjacent_library.is_file():
        binaries[adjacent_library.name] = digest(adjacent_library)
    return {
        "name": "eSpeak NG", "version": VERSION, "voice": VOICE,
        "release_url": "https://github.com/espeak-ng/espeak-ng/releases/tag/1.52.0",
        "license": "GPL-3.0-or-later", "binaries_sha256": binaries,
        "voice_data_tree_sha256": data_hash.hexdigest(),
        "voice_data_files": data_files,
    }


def read_pcm(path: Path) -> tuple[list[int], int]:
    with wave.open(str(path), "rb") as audio:
        if audio.getnchannels() != 1 or audio.getsampwidth() != 2 or audio.getcomptype() != "NONE":
            raise ValueError(f"Expected mono signed 16-bit PCM WAV: {path.name}")
        source_rate = audio.getframerate()
        if not 8000 <= source_rate <= 96000:
            raise ValueError(f"Unexpected source sample rate: {source_rate}")
        raw = audio.readframes(audio.getnframes())
    samples = array.array("h", raw)
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        raise ValueError(f"Empty synthesized WAV: {path.name}")
    return list(samples), source_rate


def resample(samples: list[int], source_rate: int) -> list[int]:
    if source_rate == RATE:
        return samples
    # Anti-aliased rational resampling using fixed Q30 kernels. All arithmetic
    # for each output sample is integral, and no external DSP package is needed.
    radius, scale = 16, 1 << 30
    cutoff = min(1.0, RATE / source_rate) * 0.94
    offsets = range(-radius + 1, radius + 1)
    kernels = {}
    for remainder in range(0, RATE, math.gcd(source_rate, RATE)):
        fraction = remainder / RATE
        values = []
        for offset in offsets:
            distance = offset - fraction
            sinc = cutoff if distance == 0 else math.sin(math.pi * cutoff * distance) / (math.pi * distance)
            window = 0.5 * (1 + math.cos(math.pi * distance / radius)) if abs(distance) < radius else 0.0
            values.append(sinc * window)
        total = sum(values)
        coefficients = [round(value / total * scale) for value in values]
        coefficients[radius - 1] += scale - sum(coefficients)
        kernels[remainder] = coefficients
    output = []
    count = (len(samples) * RATE + source_rate // 2) // source_rate
    for index in range(count):
        position, remainder = divmod(index * source_rate, RATE)
        accumulator = 0
        for offset, coefficient in zip(offsets, kernels[remainder]):
            source_index = position + offset
            if 0 <= source_index < len(samples):
                accumulator += samples[source_index] * coefficient
        value = (abs(accumulator) + scale // 2) // scale
        output.append(-value if accumulator < 0 else value)
    return output


def process(samples: list[int]) -> tuple[array.array, dict]:
    active = [i for i, value in enumerate(samples) if abs(value) >= TRIM_THRESHOLD]
    if not active:
        raise ValueError("Synthesized speech is silent")
    samples = samples[max(0, active[0] - PADDING):min(len(samples), active[-1] + PADDING + 1)]
    for index in range(min(FADE, len(samples) // 2)):
        gain = index / FADE
        samples[index] = round(samples[index] * gain)
        samples[-1 - index] = round(samples[-1 - index] * gain)
    original_peak = max(abs(value) for value in samples)
    if original_peak < 1000:
        raise ValueError(f"Synthesized speech peak is too low: {original_peak}")
    gain = PEAK / original_peak
    processed = array.array("h", (round(value * gain) for value in samples))
    peak = max(abs(value) for value in processed)
    clipped = sum(abs(value) >= 32767 for value in processed)
    duration = len(processed) / RATE
    if clipped or peak != PEAK or not 0.15 <= duration <= 8:
        raise ValueError(f"Unacceptable speech waveform: peak={peak}, clipped={clipped}, seconds={duration}")
    return processed, {
        "sample_count": len(processed), "pcm_bytes": len(processed) * 2,
        "duration_seconds": round(duration, 4), "peak_absolute": peak,
        "rms": round(math.sqrt(sum(value * value for value in processed) / len(processed)), 2),
        "clipped_samples": clipped, "normalization_gain": round(gain, 6),
    }


def pcm_bytes(samples: array.array) -> bytes:
    little = array.array("h", samples)
    if sys.byteorder != "little":
        little.byteswap()
    return little.tobytes()


def generate(engine: Path, environment: dict[str, str], phrases_path: Path,
             phrases: list[dict[str, str]], output: Path) -> dict:
    provenance = engine_provenance(engine, environment)
    source = output / "build" / "speech" / "source"
    processed_directory = output / "build" / "speech" / "wav"
    firmware = output / "WeatherDashboard"
    for directory in (source, processed_directory, firmware):
        directory.mkdir(parents=True, exist_ok=True)
    clips = []
    lines = ["// Generated by tools/prepare_speech.py; do not edit.",
             "// Native eSpeak NG build output: 16 kHz signed 16-bit mono PCM.",
             '#include "SpeechClips.h"', "", "namespace {", ""]
    for phrase in phrases:
        name, text = phrase["id"], phrase["text"]
        native = source / f"{name}.wav"
        command = [str(engine), "-D", "-v", VOICE, "-s", str(SPEED), "-a", "100",
                   "-p", "50", "-g", "0", "-k", "0", "-z", "-w", str(native), "--", text]
        result = subprocess.run(command, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", env=environment, timeout=30)
        if result.returncode:
            raise ValueError(f"{name}: synthesis failed ({result.returncode}): {result.stderr.strip()}")
        samples, native_rate = read_pcm(native)
        samples, metadata = process(resample(samples, native_rate))
        payload = pcm_bytes(samples)
        destination = processed_directory / f"{name}.wav"
        with wave.open(str(destination), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(RATE)
            audio.writeframes(payload)
        clips.append({
            "id": name, "phrase": text, "source_sample_rate_hz": native_rate,
            "source_wav": native.relative_to(output).as_posix(), "source_sha256": digest(native),
            "wav": destination.relative_to(output).as_posix(), "wav_sha256": digest(destination),
            "pcm_sha256": hashlib.sha256(payload).hexdigest(), **metadata,
        })
        lines.append(f"alignas(4) const int16_t k{name}[] = {{")
        for offset in range(0, len(samples), 16):
            lines.append("  " + ", ".join(str(value) for value in samples[offset:offset + 16]) + ",")
        lines.extend(["};", ""])
        print(f"{name}: {metadata['sample_count']} samples", flush=True)
    lines.append("const SpeechClipData kClips[] = {")
    for clip in clips:
        name = clip["id"]
        lines.append(f"  {{k{name}, sizeof(k{name}) / sizeof(k{name}[0])}},")
    lines.extend(["};", 'static_assert(sizeof(kClips) / sizeof(kClips[0]) == static_cast<size_t>(SpeechClip::Count), "Speech clip table mismatch");',
                  "}  // namespace", "", "SpeechClipData speechClipData(SpeechClip clip) {",
                  "  const size_t index = static_cast<size_t>(clip);",
                  "  if (index >= static_cast<size_t>(SpeechClip::Count)) return {nullptr, 0};",
                  "  return kClips[index];", "}", ""])
    cpp = firmware / "SpeechClips.cpp"
    cpp.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    manifest = {
        "schema_version": 1, "generation_script": "tools/prepare_speech.py",
        "generation_script_sha256": digest(Path(__file__)),
        "phrases": "assets/speech/phrases.json", "phrases_sha256": digest(phrases_path),
        "engine": "eSpeak NG native formant synthesis", "engine_provenance": provenance,
        "voice_name": VOICE, "synthesis_words_per_minute": SPEED,
        "deterministic_random_mode": True, "synthesis_volume": 100,
        "sample_rate_hz": RATE, "bits_per_sample": 16, "channels": 1,
        "encoding": "signed little-endian PCM", "resampler": RESAMPLER,
        "trim_threshold_absolute": TRIM_THRESHOLD, "boundary_padding_ms": 40,
        "boundary_fade_ms": 4, "normalization_target_peak": PEAK,
        "clip_count": len(clips), "total_pcm_bytes": sum(c["pcm_bytes"] for c in clips),
        "total_duration_seconds": round(sum(c["sample_count"] for c in clips) / RATE, 4),
        "source_cpp": "WeatherDashboard/SpeechClips.cpp", "source_cpp_sha256": digest(cpp),
        "source_header": "WeatherDashboard/SpeechClips.h",
        "source_header_sha256": digest(PACKAGE / "WeatherDashboard" / "SpeechClips.h"),
        "clips": clips,
        "validation": {"all_117_phrase_ids_match_header_order": True,
                       "no_silent_clips": True, "no_clipped_samples": True,
                       "all_processed_clips_16khz_16bit_mono": True},
    }
    path = output / "build" / "speech" / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--espeak-ng", help="Path to the native eSpeak NG 1.52.0 executable")
    parser.add_argument("--espeak-data", help="Optional extracted espeak-ng-data directory")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE,
                        help="Artifact root (default: this firmware package); useful for clean reproducibility checks")
    arguments = parser.parse_args()
    try:
        phrases_path = PACKAGE / "assets" / "speech" / "phrases.json"
        phrases = read_phrases(phrases_path)
        engine = find_engine(arguments.espeak_ng)
        environment = engine_environment(engine, arguments.espeak_data)
        manifest = generate(engine, environment, phrases_path, phrases, arguments.output_dir.resolve())
        print(f"Generated {manifest['clip_count']} clips: {manifest['total_pcm_bytes']} PCM bytes; "
              f"manifest {arguments.output_dir.resolve() / 'build/speech/manifest.json'}")
    except (OSError, ValueError, KeyError, IndexError, subprocess.SubprocessError, wave.Error) as error:
        parser.exit(1, f"Speech preparation failed: {error}\n")


if __name__ == "__main__":
    main()
