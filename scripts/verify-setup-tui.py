#!/usr/bin/env python3
"""Repeatable real-terminal setup E2E suite with isolated homes and saved frames.

Examples:
  python3 scripts/verify-setup-tui.py --binary target/debug/omawake --suite smoke
  python3 scripts/verify-setup-tui.py --binary ~/.local/bin/omawake --suite full \
    --model-cache ~/.local/share/omawake/models/moonshine-streaming-tiny-q8_0-silero-v6.2.1
"""

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
import time
import traceback
import uuid

APP = "omawake"
TABS = ("Choose runtime", "Choose device", "Choose model", "Choose microphone", "Review and apply")
SMOKE = ("cancel-start", "cancel-review", "revise-reset", "shortcut-cancel", "missing-provider")
FULL = (*SMOKE, "recommended-apply", "shortcut-apply", "custom-apply")


class BlankStartupTimeout(AssertionError):
    """The terminal remained blank before the first setup page appeared."""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--suite", choices=("smoke", "full"))
    parser.add_argument("--scenario", choices=FULL)
    parser.add_argument("--artifacts", type=Path, help="directory for frames and JSON results")
    parser.add_argument("--model-cache", type=Path, help="verified installed recommended model")
    parser.add_argument("--custom-model-cache", type=Path, help="verified installed CPU model")
    parser.add_argument("--provider-library", type=Path)
    parser.add_argument("--model-down", type=int, default=0, help="rows below preferred CPU model")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--page-timeout", type=int, default=60)
    # Keep the earlier one-case invocations working.
    parser.add_argument("--cancel-before-accept", action="store_true")
    parser.add_argument("--customize", action="store_true")
    parser.add_argument("--customize-review", action="store_true")
    parser.add_argument("--customize-apply", action="store_true")
    parser.add_argument("--shortcut", action="store_true")
    args = parser.parse_args()
    legacy = sum((args.cancel_before_accept, args.customize, args.customize_review, args.customize_apply))
    if legacy > 1 or (args.suite or args.scenario) and legacy:
        parser.error("choose one suite, scenario, or legacy verification mode")
    if args.suite and args.scenario:
        parser.error("--suite and --scenario are mutually exclusive")
    if args.suite:
        scenarios = SMOKE if args.suite == "smoke" else FULL
    elif args.scenario:
        scenarios = (args.scenario,)
    elif args.customize_apply:
        scenarios = ("custom-apply",)
    elif args.customize_review:
        scenarios = ("revise-reset",)
    elif args.customize:
        scenarios = ("cancel-review",)
    elif args.cancel_before_accept:
        scenarios = ("shortcut-cancel" if args.shortcut else "cancel-review",)
    else:
        scenarios = ("shortcut-apply" if args.shortcut else "recommended-apply",)
    if args.shortcut and (args.suite or args.scenario or args.customize or args.customize_review or args.customize_apply):
        parser.error("--shortcut only modifies the legacy recommended or cancel mode")
    binary = args.binary.expanduser().resolve(strict=True)
    if shutil.which("tmux") is None:
        parser.error("tmux is required")
    recommended = args.model_cache.expanduser().resolve(strict=True) if args.model_cache else None
    custom = args.custom_model_cache.expanduser().resolve(strict=True) if args.custom_model_cache else recommended
    for source in (recommended, custom):
        if source is not None and not source.is_dir():
            parser.error("model caches must be installed catalog model directories")
    if any(case in scenarios for case in ("recommended-apply", "shortcut-apply")) and recommended is None:
        parser.error("--model-cache is required for recommended Apply")
    if "custom-apply" in scenarios and custom is None:
        parser.error("--custom-model-cache or --model-cache is required for custom Apply")
    if args.artifacts:
        artifacts = args.artifacts.expanduser().resolve()
        artifacts.mkdir(parents=True, exist_ok=True)
    else:
        artifacts = Path(tempfile.mkdtemp(prefix=f"{APP}-setup-e2e-results-"))
    return args, binary, scenarios, recommended, custom, artifacts


def run_case(args, binary, case, recommended, custom, artifacts, attempt):
    case_dir = artifacts / case / f"attempt-{attempt}"
    case_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = case_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    started = time.monotonic()
    source = custom if case == "custom-apply" else recommended if case in ("recommended-apply", "shortcut-apply") else None
    with tempfile.TemporaryDirectory(prefix=f"{APP}-{case}-") as scratch:
        root = Path(scratch)
        (root / "run").mkdir()
        env = dict(os.environ, XDG_CONFIG_HOME=str(root / "config"), XDG_DATA_HOME=str(root / "data"),
                   XDG_STATE_HOME=str(root / "state"), XDG_CACHE_HOME=str(root / "cache"),
                   XDG_RUNTIME_DIR=str(root / "run"), TERM="xterm-256color")
        provider = args.provider_library or Path(f"/usr/lib/{APP}/libaudiocpp.so")
        if (args.provider_library is not None or binary.parent.name in {"debug", "release"}) and provider.is_file():
            env["OMAWAKE_AUDIOCPP_LIBRARY"] = str(provider.resolve())
        if case == "missing-provider":
            env["OMAWAKE_AUDIOCPP_LIBRARY"] = str(root / "missing-provider.so")
        if source is not None:
            model = root / "data" / APP / "models" / source.name
            shutil.copytree(source, model)
            verified = subprocess.run([str(binary), "setup", "model", "--verify", source.name],
                                      env=env, capture_output=True, text=True, timeout=30)
            if verified.returncode:
                raise AssertionError(f"cached model verification failed: {verified.stdout}{verified.stderr}")
        socket = f"{APP}-{uuid.uuid4().hex[:12]}"
        base = ["tmux", "-L", socket, "-f", "/dev/null"]
        command = f"{shlex.quote(str(binary))} setup; code=$?; printf '\\nE2E_EXIT=%s\\n' \"$code\"; sleep 600"
        frame_number = 0

        def tmux(*parts):
            return subprocess.run([*base, *parts], env=env, capture_output=True, text=True, check=True).stdout

        def screen():
            return tmux("capture-pane", "-p", "-J")

        def frame(label, shown=None):
            nonlocal frame_number
            shown = screen() if shown is None else shown
            (frames_dir / f"{frame_number:02d}-{label}.txt").write_text(shown)
            frame_number += 1
            return shown

        def await_screen(predicate, label, timeout=None):
            began = time.monotonic()
            deadline = began + (args.page_timeout if timeout is None else timeout)
            checkpoints = iter((5, 20, 60, 180)) if label == "final" else iter(())
            next_checkpoint = next(checkpoints, None)
            while time.monotonic() < deadline:
                shown = screen()
                if predicate(shown):
                    return frame(label, shown)
                if next_checkpoint is not None and time.monotonic() - began >= next_checkpoint:
                    frame(f"progress-{label}-{next_checkpoint}", shown)
                    next_checkpoint = next(checkpoints, None)
                time.sleep(0.1)
            shown = frame(f"timeout-{label}")
            if label == "initial" and not shown.strip():
                raise BlankStartupTimeout("terminal remained blank before the first setup page")
            raise AssertionError(f"screen did not reach {label}:\n{shown}")

        def await_text(needle, label=None, timeout=None):
            return await_screen(lambda shown: needle in shown, label or needle.lower().replace(" ", "-"), timeout)

        def key(*names):
            tmux("send-keys", *names)

        def next_tab(index):
            key("Right")
            await_text(TABS[index + 1], f"tab-{index + 1}")

        config = root / "config" / APP / "config.toml"
        launcher = root / "data" / "applications" / f"{APP}-settings.desktop"
        downloads = root / "data" / APP / "downloads"
        try:
            tmux("new-session", "-d", "-x", "100", "-y", "30", command)
            first = await_text(TABS[0], "initial")
            match = re.search(r"Runtime:\s*(\w+)", first)
            if not match:
                raise AssertionError(f"recommendation is missing from first page:\n{first}")
            original_runtime = match.group(1)
            if case in ("recommended-apply", "shortcut-apply"):
                if source.name.split("-")[0] not in first.lower():
                    raise AssertionError(f"recommended model differs from cache {source.name}:\n{first}")
            if case == "cancel-start":
                key("q")
            elif case in ("shortcut-cancel", "shortcut-apply"):
                key("r")
                await_text(TABS[-1], "recommended-review")
                if case == "shortcut-cancel":
                    key("q")
            else:
                if case in ("custom-apply", "missing-provider"):
                    key("Home", "Space")
                elif case == "revise-reset":
                    key("Home")
                    if original_runtime == "default":
                        key("Down")
                    key("Space")
                for index in range(len(TABS) - 1):
                    if case == "custom-apply" and index == 2:
                        for _ in range(args.model_down):
                            key("Down")
                        key("Space")
                    next_tab(index)
                review = frame("review")
                if case == "revise-reset":
                    selected_runtime = "openvino" if original_runtime == "default" else "default"
                    if f"Runtime: {selected_runtime}" not in review:
                        raise AssertionError(f"changed runtime absent from review:\n{review}")
                    key("r")
                    await_screen(lambda shown: f"Runtime: {original_runtime}" in shown and TABS[-1] in shown,
                                 "restored-defaults")
                    key("Left")
                    await_text(TABS[-2], "backward-review")
                    key("Right")
                    await_text(TABS[-1], "forward-review")
                    key("q")
                elif case in ("cancel-review",):
                    key("q")
            if case in ("recommended-apply", "shortcut-apply", "custom-apply", "missing-provider"):
                if config.exists() or launcher.exists() or downloads.exists():
                    raise AssertionError("setup wrote files before final Apply")
                frame("before-apply")
                key("Enter")
            final = await_text("E2E_EXIT=", "final", args.timeout)
            (case_dir / "terminal-final.txt").write_text(final)
            exit_match = re.search(r"E2E_EXIT=(\d+)", final)
            if not exit_match:
                raise AssertionError(f"setup did not exit:\n{final}")
            exit_code = int(exit_match.group(1))
            applies = case in ("recommended-apply", "shortcut-apply", "custom-apply")
            if applies and exit_code != 0:
                raise AssertionError(f"setup failed:\n{final}")
            if case == "missing-provider" and exit_code == 0:
                raise AssertionError("setup accepted a missing provider")
            if not applies and case != "missing-provider" and exit_code != 0:
                raise AssertionError(f"cancel returned {exit_code}:\n{final}")
            if applies:
                if not config.is_file() or not launcher.is_file() or "Setup complete" not in final:
                    raise AssertionError(f"Apply did not finish transaction:\n{final}")
                config_text = config.read_text()
                (case_dir / "config.toml").write_text(config_text)
                if f'name = "{source.name}"' not in config_text:
                    raise AssertionError("saved model differs from the verified cached model")
                if case == "custom-apply" and 'runtime = "default"' not in config_text:
                    raise AssertionError("custom CPU runtime was not saved")
            else:
                if config.exists() or launcher.exists() or downloads.exists():
                    raise AssertionError("cancel or failed Apply changed setup files")
                if case != "missing-provider" and "Setup cancelled" not in final:
                    raise AssertionError("cancel confirmation is missing")
            result = {"scenario": case, "status": "pass", "attempt": attempt, "exit_code": exit_code,
                      "duration_seconds": round(time.monotonic() - started, 2),
                      "frames": frame_number, "applied": applies}
            (case_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            return result
        except Exception:
            (case_dir / "error.txt").write_text(traceback.format_exc())
            try:
                frame("failure")
            except Exception:
                pass
            raise
        finally:
            subprocess.run([*base, "kill-server"], env=env, capture_output=True)


def main():
    args, binary, scenarios, recommended, custom, artifacts = parse_args()
    results = []
    for case in scenarios:
        for attempt in (1, 2):
            try:
                result = run_case(args, binary, case, recommended, custom, artifacts, attempt)
                print(f"PASS {APP} {case} ({result['duration_seconds']}s, attempt {attempt})", flush=True)
                results.append(result)
                break
            except BlankStartupTimeout as error:
                if attempt == 1:
                    print(f"RETRY {APP} {case}: {error}", flush=True)
                    continue
                print(f"FAIL {APP} {case}: {error}", flush=True)
                results.append({"scenario": case, "status": "fail", "error": str(error)})
            except Exception as error:
                print(f"FAIL {APP} {case}: {error}", flush=True)
                results.append({"scenario": case, "status": "fail", "error": str(error)})
                break
    summary = {"app": APP, "binary": str(binary), "artifacts": str(artifacts),
               "passed": sum(result["status"] == "pass" for result in results),
               "total": len(results), "results": results}
    (artifacts / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Artifacts: {artifacts}", flush=True)
    if summary["passed"] != summary["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
