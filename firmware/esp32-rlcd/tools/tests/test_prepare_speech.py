# SPDX-License-Identifier: Apache-2.0
"""Speech-preparation checks with a stand-in synthesizer; eSpeak NG is not needed or run."""

import contextlib
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave


TOOLS = Path(__file__).resolve().parents[1]
PACKAGE = TOOLS.parent
SPEC = importlib.util.spec_from_file_location("rlcd_prepare_speech", TOOLS / "prepare_speech.py")
speech = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(speech)

PHRASES = json.loads((PACKAGE / "assets/speech/phrases.json").read_text(encoding="utf-8"))["phrases"]
ENUM = re.findall(r"^\s+(\w+),", (PACKAGE / "WeatherDashboard/SpeechClips.h").read_text(encoding="utf-8")
                  .split("enum class SpeechClip", 1)[1].split("Count", 1)[0], re.MULTILINE)


class FakeEngine:
    """Answers --version like eSpeak NG 1.52.0 and writes a short tone WAV for each phrase."""

    def __init__(self, data_dir, version="1.52.0"):
        self.data_dir = data_dir
        self.version = version
        self.synthesized = []

    def __call__(self, command, **unused):
        if command[1:] == ["--version"]:
            return subprocess.CompletedProcess(
                command, 0, stdout=f"eSpeak NG text-to-speech: {self.version}  Data at: {self.data_dir}\n", stderr="")
        path, text = Path(command[command.index("-w") + 1]), command[-1]
        self.synthesized.append((command, text))
        # Most clips at the target rate; every 40th at eSpeak's native 22050 Hz exercises the resampler.
        rate = 22050 if len(self.synthesized) % 40 == 1 else 16000
        silence = [0] * (rate // 20)
        tone = [round(12000 * math.sin(2 * math.pi * 220 * i / rate))
                for i in range(int(rate * (0.2 + 0.004 * len(text))))]
        with wave.open(str(path), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(rate)
            audio.writeframes(b"".join(v.to_bytes(2, "little", signed=True) for v in silence + tone + silence))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")


class PrepareSpeechTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="rlcd-speech-test-")
        self.root = Path(self.directory.name)
        self.engine = self.root / "espeak-ng"
        self.engine.write_bytes(b"stand-in executable; never run")
        self.data = self.root / "espeak-ng-data"
        (self.data / "voices").mkdir(parents=True)
        (self.data / "voices" / "en-us").write_text("stand-in voice data\n", encoding="utf-8")
        self.output = self.root / "out"

    def tearDown(self):
        self.directory.cleanup()

    def run_main(self, fake):
        argv = ["prepare_speech.py", "--espeak-ng", str(self.engine), "--espeak-data", str(self.data),
                "--output-dir", str(self.output)]
        stdout = io.StringIO()
        with patch.object(speech.subprocess, "run", fake), patch.object(sys, "argv", argv), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
            speech.main()
        return stdout.getvalue()

    def test_generates_every_v40_clip_in_enum_order_with_the_fields_the_host_tests_read(self):
        package_clips = PACKAGE / "WeatherDashboard/SpeechClips.cpp"
        existed = package_clips.is_file() and package_clips.read_bytes()
        fake = FakeEngine(self.data)
        output = self.run_main(fake)
        self.assertEqual(len(ENUM), speech.CLIP_COUNT)
        self.assertEqual([phrase["id"] for phrase in PHRASES], ENUM)
        # One synthesis per phrase, in enum order, text passed after "--" so it is never an option.
        self.assertEqual([text for _, text in fake.synthesized], [phrase["text"] for phrase in PHRASES])
        for command, text in fake.synthesized:
            self.assertEqual(command[:4], [str(self.engine.resolve()), "-D", "-v", "en-us"])
            self.assertEqual(command[-2:], ["--", text])

        manifest = json.loads((self.output / "build/speech/manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["clip_count"], 120)
        self.assertEqual([clip["id"] for clip in manifest["clips"]], ENUM)
        self.assertEqual([clip["phrase"] for clip in manifest["clips"]], [phrase["text"] for phrase in PHRASES])
        self.assertTrue(all(isinstance(clip["sample_count"], int) and clip["sample_count"] > 0
                            for clip in manifest["clips"]))
        self.assertEqual({clip["source_sample_rate_hz"] for clip in manifest["clips"]}, {16000, 22050})
        self.assertTrue(manifest["validation"]["all_phrase_ids_match_header_order"])
        self.assertIn("Generated 120 clips", output)

        cpp = self.output / "WeatherDashboard/SpeechClips.cpp"
        text = cpp.read_text(encoding="utf-8")
        table = text.split("const SpeechClipData kClips[] = {", 1)[1].split("};", 1)[0]
        self.assertEqual(re.findall(r"\{k(\w+), sizeof", table), ENUM)
        self.assertIn("== static_cast<size_t>(SpeechClip::Count)", text)
        self.assertNotIn("\r", text)
        # build.py accepts the clips only when these hashes match the files it compiles.
        self.assertEqual(manifest["source_cpp_sha256"], hashlib.sha256(cpp.read_bytes()).hexdigest())
        self.assertEqual(manifest["source_header_sha256"], hashlib.sha256(
            (PACKAGE / "WeatherDashboard/SpeechClips.h").read_bytes()).hexdigest())
        for clip in manifest["clips"]:
            with wave.open(str(self.output / clip["wav"]), "rb") as audio:
                self.assertEqual((audio.getnchannels(), audio.getsampwidth(), audio.getframerate()), (1, 2, 16000))
                self.assertEqual(audio.getnframes(), clip["sample_count"])
            self.assertEqual(clip["peak_absolute"], speech.PEAK)
            self.assertEqual(clip["clipped_samples"], 0)
        # The package's own tree is untouched when --output-dir points elsewhere.
        self.assertEqual(package_clips.is_file() and package_clips.read_bytes(), existed)

    def test_phrase_list_must_name_all_clips_once_in_enum_order(self):
        cases = {"count": PHRASES[:-1], "order": [PHRASES[1], PHRASES[0], *PHRASES[2:]],
                 "unique": [*PHRASES[:-1], PHRASES[0]]}
        expected = {"count": "all 120 ordered", "order": "order differ", "unique": "unique"}
        for name, phrases in cases.items():
            with self.subTest(case=name):
                path = self.root / f"{name}.json"
                path.write_text(json.dumps({"schema_version": 1, "phrases": phrases}), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, expected[name]):
                    speech.read_phrases(path)
        self.assertEqual(len(speech.read_phrases(PACKAGE / "assets/speech/phrases.json")), 120)

    def test_wrong_engine_version_or_missing_engine_fails_without_writing_clips(self):
        argv = ["prepare_speech.py", "--espeak-ng", str(self.engine), "--espeak-data", str(self.data),
                "--output-dir", str(self.output)]
        stderr = io.StringIO()
        with patch.object(speech.subprocess, "run", FakeEngine(self.data, version="1.51.0")), \
                patch.object(sys, "argv", argv), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as failure:
                speech.main()
        self.assertEqual(failure.exception.code, 1)
        self.assertIn("Expected eSpeak NG 1.52.0", stderr.getvalue())
        self.assertFalse((self.output / "WeatherDashboard/SpeechClips.cpp").exists())
        with patch.object(speech.shutil, "which", return_value=None):
            with self.assertRaisesRegex(ValueError, "1.52.0 was not found"):
                speech.find_engine(None)


if __name__ == "__main__":
    unittest.main()
