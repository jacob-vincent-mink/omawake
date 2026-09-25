#!/usr/bin/env python3
"""Run the complete setup TUI against an isolated, already verified model cache.

Example: python3 scripts/verify-setup-tui.py --binary ~/.local/bin/omawake \
    --model-cache ~/.local/share/omawake/models/moonshine-streaming-tiny-q8_0-silero-v6.2.1
"""

import argparse
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
import time


APP = "omawake"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    binary = args.binary.expanduser().resolve(strict=True)
    source = args.model_cache.expanduser().resolve(strict=True)
    if not source.is_dir():
        parser.error("--model-cache must be an installed catalog model directory")
    if shutil.which("script") is None:
        parser.error("util-linux script(1) is required")

    with tempfile.TemporaryDirectory(prefix=f"{APP}-setup-e2e-") as scratch:
        root = Path(scratch)
        model = root / "data" / APP / "models" / source.name
        shutil.copytree(source, model)
        (root / "run").mkdir()
        env = dict(
            os.environ,
            XDG_CONFIG_HOME=str(root / "config"),
            XDG_DATA_HOME=str(root / "data"),
            XDG_STATE_HOME=str(root / "state"),
            XDG_CACHE_HOME=str(root / "cache"),
            XDG_RUNTIME_DIR=str(root / "run"),
            TERM="xterm-256color",
        )
        verified = subprocess.run(
            [str(binary), "setup", "model", "--verify", source.name],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if verified.returncode:
            raise SystemExit(f"model verification failed before Apply:\n{verified.stdout}{verified.stderr}")

        transcript = root / "terminal.log"
        command = f"stty rows 30 cols 100 && {shlex.quote(str(binary))} setup"
        child = subprocess.Popen(
            ["script", "-fqec", command, str(transcript)],
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            deadline = time.monotonic() + 15
            model_line = ""
            while time.monotonic() < deadline:
                if transcript.exists():
                    screen = re.sub(rb"\x1b\[[0-9;?]*[A-Za-z]", b"", transcript.read_bytes()).decode(errors="replace")
                    model_line = next((line for line in screen.splitlines() if line.startswith("Model:")), "")
                if model_line or child.poll() is not None:
                    break
                time.sleep(0.1)
            expected = [source.name.split("-")[0]]
            if source.name.endswith(("-openvino", "-gguf")):
                expected.append(source.name.rsplit("-", 1)[1])
            if not model_line or not all(word in model_line.lower() for word in expected):
                child.kill()
                child.communicate()
                raise SystemExit(
                    f"recommended model does not match cached {source.name}; "
                    f"refusing Apply. First page: {model_line or 'missing'}"
                )
            for key in (b"\x1b[D", b"\r", b"\x1b[D", b"\r"):
                if child.poll() is not None:
                    break
                child.stdin.write(key)
                child.stdin.flush()
                time.sleep(0.3)
            _, stderr = child.communicate(timeout=args.timeout)
        except (subprocess.TimeoutExpired, BrokenPipeError) as error:
            child.kill()
            child.communicate()
            raise SystemExit(f"setup did not complete: {error}") from error
        text = re.sub(rb"\x1b\[[0-9;?]*[A-Za-z]", b"", transcript.read_bytes()).decode(errors="replace")
        required = ("Select setup", "Accept setup", "Setup complete.")
        missing = [marker for marker in required if marker not in text]
        config = root / "config" / APP / "config.toml"
        launcher = root / "data" / "applications" / f"{APP}-settings.desktop"
        if child.returncode or missing or not config.is_file() or not launcher.is_file():
            raise SystemExit(
                f"setup failed (exit={child.returncode}, missing={missing}, "
                f"config={config.is_file()}, launcher={launcher.is_file()}):\n"
                f"{text[-6000:]}\n{stderr.decode(errors='replace')}"
            )
        print(f"{APP}: Select → Accept → Finish passed; model verified; config and launcher installed in isolation")


if __name__ == "__main__":
    main()
