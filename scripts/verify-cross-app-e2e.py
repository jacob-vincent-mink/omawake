#!/usr/bin/env python3
"""Run actual speech and wake inference with isolated CPU or GPU profiles.

Example:
  python3 scripts/verify-cross-app-e2e.py \
    --omawake target/debug/omawake --omaspeak ../omaspeak/target/debug/omaspeak \
    --wake-model ~/.local/share/omawake/models/moonshine-streaming-tiny-q8_0-silero-v6.2.1 \
    --speech-model ~/.local/share/omaspeak/models/kokoro-82m-gguf \
    --cpu-provider /usr/lib/omaspeak/libaudiocpp.so \
    --vulkan-provider /path/to/vulkan/build/bin \
    --runtime vulkan --artifacts /tmp/oma-cross-app-vulkan

This Python file drives the Rust applications; it is never used by either app
to download a model or perform inference.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import wave


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("omawake", "omaspeak", "wake-model", "speech-model", "cpu-provider", "artifacts"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--vulkan-provider", type=Path)
    parser.add_argument("--cuda-provider", type=Path)
    parser.add_argument("--wake-cpu-provider", type=Path)
    parser.add_argument("--runtime", choices=("default", "vulkan", "cuda"), required=True)
    args = parser.parse_args()
    wake = args.omawake.resolve(strict=True)
    speech = args.omaspeak.resolve(strict=True)
    wake_model = args.wake_model.expanduser().resolve(strict=True)
    speech_model = args.speech_model.expanduser().resolve(strict=True)
    cpu_provider = args.cpu_provider.resolve(strict=True)
    root = args.artifacts.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.runtime == "vulkan" and args.vulkan_provider is None:
        parser.error("--vulkan-provider is required for Vulkan")
    if args.runtime == "cuda" and args.cuda_provider is None:
        parser.error("--cuda-provider is required for CUDA")
    selected_provider = args.vulkan_provider if args.runtime == "vulkan" else args.cuda_provider
    provider = selected_provider.resolve(strict=True) if selected_provider else None
    steps = []

    def environment(app, model):
        home = root / app
        for name in ("config", "data", "cache", "state", "run"):
            (home / name).mkdir(parents=True, exist_ok=True)
        models = home / "data" / app / "models"
        models.mkdir(parents=True, exist_ok=True)
        target = models / model.name
        if not target.exists():
            target.symlink_to(model, target_is_directory=True)
        return dict(os.environ, XDG_CONFIG_HOME=str(home / "config"),
                    XDG_DATA_HOME=str(home / "data"), XDG_CACHE_HOME=str(home / "cache"),
                    XDG_STATE_HOME=str(home / "state"), XDG_RUNTIME_DIR=str(home / "run"))

    wake_env = environment("omawake", wake_model)
    speech_env = environment("omaspeak", speech_model)

    def run(label, binary, env, *arguments, json_output=False, expect_failure=False):
        process = subprocess.run([str(binary), *arguments], env=env, capture_output=True,
                                 text=True, timeout=300)
        entry = {"label": label, "command": [binary.name, *arguments],
                 "exit_code": process.returncode, "stdout": process.stdout,
                 "stderr": process.stderr}
        steps.append(entry)
        if (process.returncode == 0) == expect_failure:
            raise AssertionError(f"{label}: unexpected exit {process.returncode}: {process.stderr}")
        if json_output:
            return json.loads(process.stdout)
        return process.stdout

    def wav(label):
        path = root / f"{label}.wav"
        with wave.open(str(path)) as audio:
            assert audio.getnchannels() == 1 and audio.getnframes() > 1000, label
            assert any(audio.readframes(audio.getnframes())), label
        return path

    try:
        run("pin-cpu-provider", speech, speech_env, "config", "set", "backend.library", str(cpu_provider))
        run("speech-model", speech, speech_env, "setup", "model", "--set", speech_model.name)
        wake_setup_env = wake_env.copy()
        if args.wake_cpu_provider:
            wake_setup_env["OMAWAKE_AUDIOCPP_LIBRARY"] = str(args.wake_cpu_provider.resolve(strict=True))
        run("wake-model", wake, wake_setup_env, "setup", "model", "--set", wake_model.name)
        if provider:
            for app, binary, env in (("speech", speech, speech_env), ("wake", wake, wake_env)):
                proof = run(f"{app}-{args.runtime}-apply", binary, env, "setup", "runtime",
                            "--runtime", args.runtime, "--device", "gpu", "--device-id", "0",
                            "--dir", str(provider), "--apply", json_output=True)
                assert proof["applied"] and proof["probe"]["evidence"]["model_inference_verified"]
        for label, phrase in (("positive", "Computer"), ("negative", "Good morning")):
            run(f"say-{label}", speech, speech_env, "say", phrase, "--no-play", "--out",
                str(root / f"{label}.wav"), json_output=True)
            wav(label)

        def detect(label, audio, expected):
            result = run(label, wake, wake_env, "test", "--audio", str(audio), "--json",
                         json_output=True)
            assert result["backend"]["effective_runtime"] == args.runtime, result
            ids = [item["id"] for item in result["detections"]]
            assert ids == expected, (label, ids, expected)

        positive = wav("positive")
        detect("positive-first", positive, ["computer"])
        detect("positive-repeat", positive, ["computer"])
        detect("negative", wav("negative"), [])
        run("add-other", wake, wake_env, "wake-word", "add", "--id", "hello",
            "--phrase", "Hello", "--", "/usr/bin/true")
        run("remove-computer", wake, wake_env, "wake-word", "remove", "computer")
        detect("after-remove", positive, [])
        run("restore-computer", wake, wake_env, "wake-word", "add", "--id", "computer",
            "--phrase", "Computer", "--", "/usr/bin/true")
        detect("after-restore", positive, ["computer"])
        run("remove-other", wake, wake_env, "wake-word", "remove", "hello")
        run("reject-last-remove", wake, wake_env, "wake-word", "remove", "computer",
            expect_failure=True)
        detect("after-rejected-remove", positive, ["computer"])
        (root / "result.json").write_text(json.dumps({"status": "pass", "steps": steps}, indent=2) + "\n")
        print(f"PASS {args.runtime}: speech, repeat wake detection, negative, remove, restore, guard")
    except Exception as error:
        (root / "result.json").write_text(json.dumps({"status": "fail", "error": str(error),
                                                        "steps": steps}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
