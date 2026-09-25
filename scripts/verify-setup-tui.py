#!/usr/bin/env python3
"""Run setup TUI apply or pre-Apply cancellation in isolated XDG directories.

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
    parser.add_argument("--model-cache", type=Path)
    parser.add_argument("--cancel-before-accept", action="store_true")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    binary = args.binary.expanduser().resolve(strict=True)
    if not args.cancel_before_accept and args.model_cache is None:
        parser.error("--model-cache is required for a full Apply")
    source = args.model_cache.expanduser().resolve(strict=True) if args.model_cache else None
    if source is not None and not source.is_dir():
        parser.error("--model-cache must be an installed catalog model directory")
    if shutil.which("script") is None:
        parser.error("util-linux script(1) is required")

    with tempfile.TemporaryDirectory(prefix=f"{APP}-setup-e2e-") as scratch:
        root = Path(scratch)
        if not args.cancel_before_accept:
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
        if not args.cancel_before_accept:
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
            expected = [source.name.split("-")[0]] if source else []
            if source and source.name.endswith(("-openvino", "-gguf")):
                expected.append(source.name.rsplit("-", 1)[1])
            if not model_line or not all(word in model_line.lower() for word in expected):
                child.kill()
                child.communicate()
                raise SystemExit(
                    f"recommended model does not match cached {source.name if source else 'model'}; "
                    f"refusing Apply. First page: {model_line or 'missing'}"
                )
            keys = (b"\x1b[D", b"\r", b"q", b"q") if args.cancel_before_accept else (b"\x1b[D", b"\r", b"\x1b[D", b"\r")
            for key in keys:
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
        required = ("Select setup", "Accept setup", "Setup cancelled.") if args.cancel_before_accept else ("Select setup", "Accept setup", "Setup complete.")
        missing = [marker for marker in required if marker not in text]
        config = root / "config" / APP / "config.toml"
        launcher = root / "data" / "applications" / f"{APP}-settings.desktop"
        cache_has_files = any(path.is_file() for path in (root / "cache" / APP).rglob("*"))
        if args.cancel_before_accept:
            valid_state = not config.exists() and not launcher.exists() and not (root / "data" / APP).exists() and not cache_has_files
        else:
            valid_state = config.is_file() and launcher.is_file()
        if child.returncode or missing or not valid_state:
            raise SystemExit(
                f"setup failed (exit={child.returncode}, missing={missing}, "
                f"config={config.is_file()}, launcher={launcher.is_file()}, cache_files={cache_has_files}):\n"
                f"{text[-6000:]}\n{stderr.decode(errors='replace')}"
            )
        if args.cancel_before_accept:
            print(f"{APP}: final Accept page cancelled; no config, model, launcher, or compiled cache")
        else:
            print(f"{APP}: Select → Accept → Finish passed; model verified; config and launcher installed in isolation")


if __name__ == "__main__":
    main()
