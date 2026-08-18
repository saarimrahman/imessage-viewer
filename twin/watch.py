#!/usr/bin/env python3
"""Report whether the queued Twin jobs are still moving.

Prints one block for a person to read. Exits 0 when something advanced since
the last call, 1 when nothing did and work is still running, which is the
signal that a job has hung.

State lives in `.cache/twin/experiments/watch_state.json`, so each call
compares against the previous one.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from twin.export import TWIN_DIR

RESULTS = os.path.join(TWIN_DIR, "experiments", "results.jsonl")
MULTITURN = os.path.join(TWIN_DIR, "experiments", "multiturn.jsonl")
STATE = os.path.join(TWIN_DIR, "experiments", "watch_state.json")
# CAUTION: keep this list out of any shell command that the queue scripts
# themselves pgrep for. Those scripts waited on each other by filename, and this
# watchdog's own pgrep command line matched those names, so the wait never
# cleared. `finish.sh` replaced the chain with one sequential script.
JOBS = ("finish.sh",)


def running():
    try:
        out = subprocess.run(
            ["pgrep", "-fl", "|".join(JOBS) + "|mlx_lora|bench.py|multiturn.py"],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    names = []
    for line in out.splitlines():
        for job in JOBS:
            if job in line:
                names.append(job)
        if "mlx_lora" in line:
            names.append("training")
        elif "bench.py" in line:
            names.append("bench")
        elif "multiturn.py" in line:
            names.append("multiturn")
    return sorted(set(names))


def count(path):
    try:
        with open(path, encoding="utf-8") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def latest_label(path):
    last = ""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                last = line
    except OSError:
        return ""
    try:
        return json.loads(last).get("label", "")
    except (json.JSONDecodeError, AttributeError):
        return ""


def newest_checkpoint():
    """Modification time of the most recent adapter weight file."""
    newest = 0.0
    for root, _, files in os.walk(os.path.join(TWIN_DIR, "adapters")):
        for name in files:
            if name.endswith("_adapters.safetensors"):
                try:
                    newest = max(newest, os.path.getmtime(os.path.join(root, name)))
                except OSError:
                    pass
    for root, _, files in os.walk(os.path.join(TWIN_DIR, "people")):
        for name in files:
            if name.endswith("_adapters.safetensors"):
                try:
                    newest = max(newest, os.path.getmtime(os.path.join(root, name)))
                except OSError:
                    pass
    return round(newest, 1)


def main():
    now = time.time()
    state = {}
    if os.path.isfile(STATE):
        try:
            with open(STATE, encoding="utf-8") as f:
                state = json.load(f)
        except (OSError, json.JSONDecodeError):
            state = {}

    current = {
        "results": count(RESULTS),
        "multiturn": count(MULTITURN),
        "label": latest_label(RESULTS),
    }
    jobs = running()
    # A training pass writes no benchmark row for 15 to 80 minutes, so counting
    # results alone reports a healthy run as stalled. A checkpoint file written
    # recently is the proof that training is moving.
    current["checkpoint"] = newest_checkpoint()
    moved = any(
        current[k] != state.get(k) for k in ("results", "multiturn", "checkpoint")
    )
    since = now - state.get("changed_at", now)

    print(f"[twin watch] {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  running   : {', '.join(jobs) if jobs else 'nothing'}")
    print(f"  benchmarks: {current['results']} (was {state.get('results', '?')})")
    print(f"  multiturn : {current['multiturn']} (was {state.get('multiturn', '?')})")
    print(f"  last run  : {current['label'] or '-'}")
    if current["checkpoint"]:
        age = (now - current["checkpoint"]) / 60
        print(f"  checkpoint: written {age:.0f} min ago")

    if moved:
        print("  status    : advancing")
    elif not jobs:
        print("  status    : idle, queue is empty")
    else:
        print(f"  STALLED   : no new result for {since / 60:.0f} min while {jobs} run")

    state.update(current)
    if moved or not state.get("changed_at"):
        state["changed_at"] = now
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    with open(STATE, "w", encoding="utf-8") as f:
        json.dump(state, f)
    return 0 if (moved or not jobs) else 1


if __name__ == "__main__":
    sys.exit(main())
