#!/usr/bin/env python3
"""Exercise every writable config key against isolated real CLI processes.

The app under test remains Rust. This driver saves JSON evidence for each
precondition, first change, second change, repeated change, and unset.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    output = args.artifacts.resolve()
    output.mkdir(parents=True, exist_ok=True)
    app = binary.name
    results = []

    def profile(name):
        root = output / "profiles" / name.replace(".", "_")
        for directory in ("config", "data", "state", "cache", "run"):
            (root / directory).mkdir(parents=True, exist_ok=True)
        return dict(os.environ, XDG_CONFIG_HOME=str(root / "config"),
                    XDG_DATA_HOME=str(root / "data"), XDG_STATE_HOME=str(root / "state"),
                    XDG_CACHE_HOME=str(root / "cache"), XDG_RUNTIME_DIR=str(root / "run"))

    def command(env, *parts):
        process = subprocess.run([str(binary), "config", *parts], env=env,
                                 capture_output=True, text=True, timeout=60)
        if process.returncode:
            raise AssertionError(f"config {' '.join(parts)}: {process.stderr.strip()}")
        return process.stdout

    schema = json.loads(command(profile("schema"), "schema", "--json"))
    keys = [entry["key"] for entry in schema["keys"]]
    (output / "schema.json").write_text(json.dumps(schema, indent=2) + "\n")

    def values(key, entry):
        base = entry["value"]
        choices = [v.get("value") if isinstance(v, dict) else v
                   for v in entry.get("choices", [])]
        if key == "audio.device":
            return ["default", "default"]
        if key == "backend.kind":
            return (["audiocpp", "audiocpp"] if app == "omawake"
                    else ["audiocpp", "audiocpp"])
        if key == "backend.runtime":
            return ["default", "default"]
        if key == "backend.device":
            return ["auto", "cpu"]
        if key == "backend.fallback":
            return ["cpu", "error"]
        if key == "model.family":
            return [base, base]
        if key == "model.voice":
            return ["1", "2"]
        if key == "model.language":
            return (["en", "es"] if app == "omawake" else ["es", "fr"])
        if key == "backend.library_dirs":
            return ["/usr/lib", "/usr/lib:/usr/lib/openvino"]
        if key.endswith("library"):
            return [f"/usr/lib/{app}/libaudiocpp.so", f"/home/jacob/.local/lib/{app}/libaudiocpp.so"]
        if key == "backend.openvino_plugins":
            return ["/usr/lib/openvino/plugins.xml", "/usr/lib/openvino/plugins.xml"]
        if key.endswith("directory"):
            return [str(output / "model-one"), str(output / "model-two")]
        if entry["type"] == "integer":
            start = max(int(entry.get("min", 0)), int(base or 0))
            return [str(start + 1), str(start + 2)]
        if choices:
            alternatives = [str(v) for v in choices if str(v) != str(base)]
            return (alternatives + [str(base), str(base)])[:2]
        return [(str(base) or "fixture") + "-e2e-one", (str(base) or "fixture") + "-e2e-two"]

    for entry in schema["keys"]:
        key = entry["key"]
        env = profile(key)
        record = {"key": key, "steps": []}
        try:
            if key == "backend.device_id":
                command(env, "set", "backend.runtime", "cuda")
            before = json.loads(command(env, "get", key, "--json"))
            record["before"] = before
            first, second = values(key, entry)
            for label, value in (("first", first), ("second", second), ("repeat", second)):
                command(env, "set", key, value)
                actual = json.loads(command(env, "get", key, "--json"))
                expected = (value.split(":") if entry["type"] == "path-list" else
                            int(value) if entry["type"] == "integer" or key == "model.voice" else value)
                if actual != expected:
                    raise AssertionError(f"{label}: expected {value!r}, got {actual!r}")
                record["steps"].append({"operation": label, "input": value, "actual": actual})
            command(env, "unset", key)
            after = json.loads(command(env, "get", key, "--json"))
            if after != before:
                raise AssertionError(f"unset: expected {before!r}, got {after!r}")
            record["after"] = after
            record["status"] = "pass"
        except Exception as error:
            record["status"] = "fail"
            record["error"] = str(error)
        results.append(record)
        print(f"{record['status']}: {key}", flush=True)

    for prefix in (entry["prefix"] for entry in schema.get("collections", []) if "prefix" in entry):
        key = prefix + "e2e_probe"
        env = profile(key)
        record = {"key": key, "steps": []}
        try:
            for value in ("first", "second", "second"):
                command(env, "set", key, value)
                actual = json.loads(command(env, "get", key, "--json"))
                assert actual == value, (key, value, actual)
                record["steps"].append({"input": value, "actual": actual})
            command(env, "unset", key)
            whole = json.loads(command(env, "get", "--json"))
            section, _, option = key.partition(".options.")
            assert option not in whole[section]["options"], key
            record["status"] = "pass"
        except Exception as error:
            record["status"] = "fail"
            record["error"] = str(error)
        results.append(record)
        print(f"{record['status']}: {key}", flush=True)

    for entry in schema["keys"]:
        if entry["type"] != "enum":
            continue
        key = entry["key"]
        choices = [choice.get("value") if isinstance(choice, dict) else choice
                   for choice in entry.get("choices", [])]
        if not choices:
            continue
        env = profile("choices_" + key)
        record = {"key": key, "choices": choices, "tested": []}
        try:
            before = json.loads(command(env, "get", key, "--json"))
            for choice in choices:
                command(env, "set", key, str(choice))
                actual = json.loads(command(env, "get", key, "--json"))
                if actual != choice:
                    raise AssertionError(f"choice {choice!r}: got {actual!r}")
                record["tested"].append(choice)
            command(env, "unset", key)
            after = json.loads(command(env, "get", key, "--json"))
            if after != before:
                raise AssertionError(f"choice reset: expected {before!r}, got {after!r}")
            record["status"] = "pass"
        except Exception as error:
            record["status"] = "fail"
            record["error"] = str(error)
        results.append(record)
        print(f"{record['status']}: {key} choices ({len(record['tested'])}/{len(choices)})", flush=True)

    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"{sum(r['status'] == 'pass' for r in results)}/{len(results)} passed; evidence: {output}")
    raise SystemExit(any(r["status"] != "pass" for r in results))


if __name__ == "__main__":
    main()
