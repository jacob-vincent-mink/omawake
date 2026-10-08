#!/usr/bin/env python3
"""Opt-in cloud qualification driver; only stdlib and the packaged native CLI."""
import argparse
import json
import subprocess
from pathlib import Path

TEXTS = [
    "Hello. Your cloud voice is ready.",
    "Dr. Smith paid twelve dollars and fifty cents at three thirty. Is everything ready?",
    "The next release brings cloud voices, streaming playback, and local speech options. " * 4,
]


def qualify(binary, config, output, recordings):
    output.mkdir(parents=True, exist_ok=False)
    prefix = [str(binary), "--config", str(config)]
    version = subprocess.run([str(binary), "--version"], capture_output=True, text=True, check=True).stdout.strip()
    cases = recordings or TEXTS
    records = []
    for index, case in enumerate(cases, 1):
        if recordings:
            command = prefix + ["cloud", "smoke", "--audio", str(case)]
        else:
            command = prefix + ["cloud", "smoke", "--text", case, "--no-play", "--out", str(output / f"case-{index}.wav")]
        completed = subprocess.run(command, capture_output=True, text=True)
        record = {"case": index, "ok": completed.returncode == 0}
        if record["ok"]:
            try:
                record["measurement"] = json.loads(completed.stdout)
            except json.JSONDecodeError:
                record["ok"] = False
                record["error"] = "CLI did not return JSON"
        else:
            # Native CLI redacts provider bodies and credentials. Don't include config or env.
            record["error"] = completed.stderr.strip()[-2000:]
        records.append(record)
    report = {"version": version, "mode": "asr" if recordings else "tts", "live_qualification": "pending human review", "cases": records}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "REVIEW.md").write_text(
        "# Cloud qualification review\n\n"
        "Record provider/account access, model/voice, date, architecture and audio device.\n"
        "For TTS, listen to every WAV: pronunciation, truncation, pauses and chunk continuity.\n"
        "For ASR, record the expected phrase and compare detections for wake, unrelated speech and silence.\n"
        "Separately test live pause/cancel/resume and verify no stale audio/actions replay.\n"
        "Record pass/fail and unresolved issues; successful requests alone do not qualify quality.\n"
    )
    return all(record["ok"] for record in records)


def main():
    parser = argparse.ArgumentParser(description="Make explicit cloud smoke requests and save reproducible results.")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--audio", type=Path, action="append", default=[], help="ASR recording; repeat for wake, negative and silence clips")
    parser.add_argument("--accept-charges", action="store_true", required=True, help="authorize these provider requests; TTS makes three requests")
    args = parser.parse_args()
    for path in [args.binary, args.config, *args.audio]:
        if not path.is_file():
            parser.error(f"file unavailable: {path}")
    if args.out_dir.exists():
        parser.error("output directory already exists; choose a new directory")
    raise SystemExit(0 if qualify(args.binary.resolve(), args.config.resolve(), args.out_dir.resolve(), args.audio) else 1)


if __name__ == "__main__":
    main()
