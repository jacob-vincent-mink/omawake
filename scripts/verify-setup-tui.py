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
    parser.add_argument("--customize", action="store_true", help="navigate two Customize pages, then cancel")
    parser.add_argument("--customize-review", action="store_true", help="navigate Customize to final review, then cancel")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    binary = args.binary.expanduser().resolve(strict=True)
    if sum((args.cancel_before_accept, args.customize, args.customize_review)) > 1:
        parser.error("choose one verification mode")
    no_apply = args.cancel_before_accept or args.customize or args.customize_review
    copy_model = not no_apply or args.customize_review
    if copy_model and args.model_cache is None:
        parser.error("--model-cache is required for Apply or --customize-review")
    source = args.model_cache.expanduser().resolve(strict=True) if args.model_cache else None
    if source is not None and not source.is_dir():
        parser.error("--model-cache must be an installed catalog model directory")
    if shutil.which("script") is None:
        parser.error("util-linux script(1) is required")

    with tempfile.TemporaryDirectory(prefix=f"{APP}-setup-e2e-") as scratch:
        root = Path(scratch)
        if copy_model:
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
        if copy_model:
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
            if args.customize_review:
                def page_title():
                    latest = transcript.read_bytes().split(b"\x1b[2J")[-1]
                    plain = re.sub(rb"\x1b\[[0-9;?]*[A-Za-z]", b"", latest).decode(errors="replace")
                    lines = [line.strip() for line in plain.splitlines() if line.strip()]
                    return lines[2] if len(lines) >= 3 else ""

                def advance(previous):
                    for key in (b"\x1b[D", b"\r"):
                        child.stdin.write(key)
                        child.stdin.flush()
                        time.sleep(0.2)
                    until = time.monotonic() + 15
                    while time.monotonic() < until:
                        title = page_title()
                        if title and title != previous:
                            return title
                        if child.poll() is not None:
                            break
                        time.sleep(0.1)
                    raise SystemExit(f"Customize did not advance from {previous}: {page_title()}")

                child.stdin.write(b"\x1b[C")
                child.stdin.flush()
                time.sleep(0.2)
                child.stdin.write(b"\r")
                child.stdin.flush()
                title = "Select setup"
                until = time.monotonic() + 15
                while time.monotonic() < until and page_title() == title:
                    time.sleep(0.1)
                title = page_title()
                seen = []
                for _ in range(12):
                    if title == "Accept setup":
                        break
                    seen.append(title)
                    title = advance(title)
                if title != "Accept setup":
                    raise SystemExit(f"Customize did not reach Accept setup: {seen}, {title}")
                required_pages = {"Inference runtime", "Inference device", "Wake-word model", "Audio device"}
                if not required_pages.issubset(seen):
                    raise SystemExit(f"Customize skipped a page: {seen}")
                child.stdin.write(b"q")
                child.stdin.flush()
                keys = ()
            elif args.customize:
                keys = (b"\x1b[C", b"\r", b"\r", b"\x1b[D", b"\r", b"q")
            elif args.cancel_before_accept:
                keys = (b"\x1b[D", b"\r", b"q", b"q")
            else:
                keys = (b"\x1b[D", b"\r", b"\x1b[D", b"\r")
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
        except BaseException:
            if child.poll() is None:
                child.kill()
                child.communicate()
            raise
        text = re.sub(rb"\x1b\[[0-9;?]*[A-Za-z]", b"", transcript.read_bytes()).decode(errors="replace")
        if args.customize_review:
            required = ("Select setup", "Wake-word model", "Audio device", "Accept setup", "Setup cancelled.")
        elif args.customize:
            required = ("Select setup", "Inference runtime", "Inference device", "Setup cancelled.")
        elif args.cancel_before_accept:
            required = ("Select setup", "Accept setup", "Setup cancelled.")
        else:
            required = ("Select setup", "Accept setup", "Setup complete.")
        missing = [marker for marker in required if marker not in text]
        config = root / "config" / APP / "config.toml"
        launcher = root / "data" / "applications" / f"{APP}-settings.desktop"
        cache_has_files = any(path.is_file() for path in (root / "cache" / APP).rglob("*"))
        if no_apply:
            expected_models = {source.name} if args.customize_review else set()
            models = root / "data" / APP / "models"
            installed_models = {path.name for path in models.iterdir()} if models.is_dir() else set()
            valid_state = (
                not config.exists() and not launcher.exists() and not cache_has_files
                and installed_models == expected_models
                and not (root / "data" / APP / "downloads").exists()
            )
        else:
            valid_state = config.is_file() and launcher.is_file()
        if child.returncode or missing or not valid_state:
            raise SystemExit(
                f"setup failed (exit={child.returncode}, missing={missing}, "
                f"config={config.is_file()}, launcher={launcher.is_file()}, cache_files={cache_has_files}):\n"
                f"{text[-6000:]}\n{stderr.decode(errors='replace')}"
            )
        if args.customize_review:
            print(f"{APP}: Customize reached final Accept; cancelled with no download, cache, config, or launcher")
        elif args.customize:
            print(f"{APP}: Customize runtime and device pages navigated; no files changed")
        elif args.cancel_before_accept:
            print(f"{APP}: final Accept page cancelled; no config, model, launcher, or compiled cache")
        else:
            print(f"{APP}: Select → Accept → Finish passed; model verified; config and launcher installed in isolation")


if __name__ == "__main__":
    main()
