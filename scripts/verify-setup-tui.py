#!/usr/bin/env python3
"""Exercise setup in an isolated XDG home through a real tmux terminal.

Examples:
  python3 scripts/verify-setup-tui.py --binary target/debug/omawake --customize-review
  python3 scripts/verify-setup-tui.py --binary target/debug/omawake --model-cache ~/.local/share/omawake/models/MODEL
"""

import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import time
import uuid

APP = "omawake"
CUSTOM_TABS = ("Choose runtime", "Choose device", "Choose model", "Choose microphone", "Review and apply")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path)
    parser.add_argument("--provider-library", type=Path)
    parser.add_argument("--model-down", type=int, default=0, help="move this many model rows before Apply")
    parser.add_argument("--cancel-before-accept", action="store_true")
    parser.add_argument("--customize", action="store_true")
    parser.add_argument("--customize-review", action="store_true")
    parser.add_argument("--customize-apply", action="store_true", help="apply the CPU runtime and selected model")
    parser.add_argument("--shortcut", action="store_true", help="press r to review recommended defaults")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--page-timeout", type=int, default=60)
    args = parser.parse_args()
    if sum((args.cancel_before_accept, args.customize, args.customize_review, args.customize_apply)) > 1:
        parser.error("choose one verification mode")
    if args.shortcut and (args.customize or args.customize_review or args.customize_apply):
        parser.error("--shortcut cannot be combined with custom selection modes")
    binary = args.binary.expanduser().resolve(strict=True)
    source = args.model_cache.expanduser().resolve(strict=True) if args.model_cache else None
    no_apply = args.cancel_before_accept or args.customize or args.customize_review
    if not no_apply and source is None:
        parser.error("--model-cache is required for Apply")
    if source is not None and not source.is_dir():
        parser.error("--model-cache must be an installed catalog model directory")
    if shutil.which("tmux") is None:
        parser.error("tmux is required")

    with tempfile.TemporaryDirectory(prefix=f"{APP}-setup-e2e-") as scratch:
        root = Path(scratch)
        copied = source is not None and (not no_apply or args.customize_review)
        if copied:
            model = root / "data" / APP / "models" / source.name
            shutil.copytree(source, model)
        (root / "run").mkdir()
        env = dict(os.environ, XDG_CONFIG_HOME=str(root / "config"), XDG_DATA_HOME=str(root / "data"),
                   XDG_STATE_HOME=str(root / "state"), XDG_CACHE_HOME=str(root / "cache"),
                   XDG_RUNTIME_DIR=str(root / "run"), TERM="xterm-256color")
        provider = args.provider_library or Path(f"/usr/lib/{APP}/libaudiocpp.so")
        needs_provider_override = args.provider_library is not None or binary.parent.name in {"debug", "release"}
        if APP == "omawake" and needs_provider_override and provider.is_file():
            env["OMAWAKE_AUDIOCPP_LIBRARY"] = str(provider.resolve())
        if copied:
            verified = subprocess.run([str(binary), "setup", "model", "--verify", source.name],
                                      env=env, capture_output=True, text=True, timeout=30)
            if verified.returncode:
                raise SystemExit(f"cached model verification failed: {verified.stdout}{verified.stderr}")

        socket = f"{APP}-e2e-{uuid.uuid4().hex[:12]}"
        base = ["tmux", "-L", socket, "-f", "/dev/null"]
        command = f"{shlex.quote(str(binary))} setup; code=$?; printf '\\nE2E_EXIT=%s\\n' \"$code\"; sleep 600"

        def tmux(*parts):
            return subprocess.run([*base, *parts], env=env, capture_output=True, text=True, check=True).stdout

        def screen():
            return tmux("capture-pane", "-p", "-J")

        def await_text(needle, timeout=None):
            timeout = args.page_timeout if timeout is None else timeout
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                shown = screen()
                if needle in shown:
                    return shown
                time.sleep(0.1)
            raise AssertionError(f"screen did not show {needle!r}:\n{screen()}")

        def key(*names):
            tmux("send-keys", *names)

        try:
            tmux("new-session", "-d", "-x", "100", "-y", "30", command)
            first = await_text(CUSTOM_TABS[0])
            if source and not no_apply and not args.customize_apply:
                expected = source.name.split("-")[0]
                if expected not in first.lower():
                    raise AssertionError(f"recommended model did not match cached {source.name}:\n{first}")
            if args.shortcut:
                key("r")
                await_text(CUSTOM_TABS[-1])
            else:
                if args.customize_apply:
                    key("Home", "Space")
                end = 2 if args.customize else len(CUSTOM_TABS) - 1
                for index in range(end):
                    if args.customize_apply and index == 2:
                        for _ in range(args.model_down):
                            key("Down")
                        key("Space")
                    key("Right")
                    await_text(CUSTOM_TABS[index + 1])
                if args.customize_review:
                    key("Left")
                    await_text(CUSTOM_TABS[-2])
                    key("Right")
                    await_text(CUSTOM_TABS[-1])
            if no_apply:
                key("q")
            else:
                config_before_apply = root / "config" / APP / "config.toml"
                assert not config_before_apply.exists(), "configuration changed before Apply"
                key("Enter")
            final = await_text("E2E_EXIT=", args.timeout)
            if "E2E_EXIT=0" not in final:
                raise AssertionError(f"setup returned an error:\n{final}")
            config = root / "config" / APP / "config.toml"
            launcher = root / "data" / "applications" / f"{APP}-settings.desktop"
            cache_has_files = any(path.is_file() for path in (root / "cache" / APP).rglob("*"))
            models = root / "data" / APP / "models"
            installed_models = {path.name for path in models.iterdir()} if models.is_dir() else set()
            if no_apply:
                expected_models = {source.name} if copied else set()
                assert not config.exists() and not launcher.exists() and not cache_has_files
                assert installed_models == expected_models
                assert not (root / "data" / APP / "downloads").exists()
                assert "Setup cancelled" in final
            else:
                assert config.is_file() and launcher.is_file(), final
                assert "Setup complete" in final, final
            print(f"{APP}: single-menu setup passed")
        finally:
            subprocess.run([*base, "kill-server"], env=env, capture_output=True)


if __name__ == "__main__":
    main()
