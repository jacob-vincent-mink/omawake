import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("cloud_qualify", Path(__file__).with_name("qualify.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class Qualification(unittest.TestCase):
    def test_records_tts_and_asr_runs_without_running_actions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / "fixture"
            binary.write_text('#!/usr/bin/env python3\nimport sys,json\nif "--version" in sys.argv: print("fixture 1");sys.exit(0)\nassert "cloud" in sys.argv and "smoke" in sys.argv\nprint(json.dumps({"provider":"fixture","actions_executed":False,"first_audio_ms":1,"total_ms":2}))\n')
            binary.chmod(0o700)
            config = root / "config"
            config.write_text("fixture")
            for mode, clips, count in [("tts", [], 3), ("asr", [root / "wake.wav", root / "silence.wav"], 2)]:
                out = root / mode
                self.assertTrue(module.qualify(binary, config, out, clips))
                report = json.loads((out / "report.json").read_text())
                self.assertEqual(len(report["cases"]), count)
                self.assertEqual(report["mode"], mode)
                self.assertTrue((out / "REVIEW.md").exists())
    def test_failures_are_recorded_and_existing_reports_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / "fixture"
            binary.write_text('#!/bin/sh\nif [ "$1" = "--version" ]; then echo fixture; exit 0; fi\necho "provider transport failed" >&2\nexit 1\n')
            binary.chmod(0o700)
            out = root / "out"
            self.assertFalse(module.qualify(binary, root / "config", out, []))
            before = (out / "report.json").read_bytes()
            with self.assertRaises(FileExistsError):
                module.qualify(binary, root / "config", out, [])
            self.assertEqual((out / "report.json").read_bytes(), before)

if __name__ == "__main__":
    unittest.main()
