"""Background dataset, training metrics, and on-demand Twin chat."""

import importlib.util
import json
import os
import re
import signal
import threading
import time

from twin.export import (
    ME,
    dataset_profile,
    export_dataset,
    list_subjects,
    parse_person_arg,
    resolve_subject,
    system_for,
)
from twin.train import (
    DEFAULT_MODEL,
    MODELS,
    TWIN_DIR,
    adapter_dir,
    has_adapter,
    model_config,
    resolved_adapter_dir,
    run_train,
    steps_for_examples,
    write_run_metadata,
)

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
ITER_RE = re.compile(r"Iter\s+(\d+):")
TRAIN_RE = re.compile(
    r"Iter\s+(\d+):\s+Train loss\s+([0-9.eE+-]+).*?Tokens/sec\s+([0-9.eE+-]+)"
    r"(?:.*?Peak mem\s+([0-9.eE+-]+)\s+GB)?"
)
VAL_RE = re.compile(r"Iter\s+(\d+):\s+Val loss\s+([0-9.eE+-]+)")
DOWNLOAD_RE = re.compile(r"(?i)(downloading|fetching\s+\d+\s+files)")
HISTORY_TURNS = 10
SMOKE_LIMIT = 160
SMOKE_ITERS = 30
BUSY_PHASES = (
    "inspecting",
    "exporting",
    "downloading",
    "loading",
    "training",
    "cancelling",
)
MAX_RUNS = 8
RUNS_PATH = os.path.join(TWIN_DIR, "runs.json")
STEP_PHASES = (
    ("inspecting",),
    ("exporting",),
    ("downloading", "loading", "training", "cancelling"),
)

_lock = threading.Lock()
_gen_lock = threading.Lock()
_cancel = threading.Event()
_proc = None
_runs = None
_model = None
_tokenizer = None
_loaded_key = None
_state = {
    "phase": "idle",
    "detail": "",
    "iter": 0,
    "iters": 0,
    "examples": 0,
    "sent_texts": 0,
    "chats": 0,
    "augmented": 0,
    "model": DEFAULT_MODEL,
    "person": ME,
    "person_name": "You",
    "run": "complete",
    "metrics": [],
    "started_at": None,
    "phase_started_at": None,
    "train_started_at": None,
    "ended_at": None,
    "phase_seconds": {},
}


class TwinError(Exception):
    pass


class TwinCancelled(Exception):
    pass


def format_duration(seconds):
    if seconds is None or seconds < 0 or not isinstance(seconds, (int, float)):
        return ""
    total = int(round(seconds))
    if total < 60:
        return f"{total}s"
    minutes, sec = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m {sec:02d}s" if sec else f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m" if minutes else f"{hours}h"


def estimate_eta(phase, iteration, iters, train_started_at, now):
    if phase != "training" or not train_started_at:
        return None
    if iteration < 1 or not iters or iteration >= iters:
        return None
    elapsed = now - train_started_at
    if elapsed <= 0:
        return None
    return elapsed / iteration * (iters - iteration)


def activity_label(s):
    if not s.get("busy"):
        return ""
    phase = s.get("phase")
    eta = format_duration(s.get("eta_seconds"))
    if phase == "training":
        n = int(s.get("iter") or 0)
        total = int(s.get("iters") or 0)
        core = f"Twin · {n:,}/{total:,}" if total else "Twin · training"
        return f"{core} · {eta} left" if eta else core
    labels = {
        "inspecting": "Twin · auditing",
        "exporting": "Twin · building pairs",
        "downloading": "Twin · downloading",
        "loading": "Twin · loading",
        "cancelling": "Twin · stopping",
    }
    return labels.get(phase) or (s.get("detail") or "Twin · training")


def mlx_installed():
    return importlib.util.find_spec("mlx_lm") is not None


def cached_model_repos():
    """Return repos with complete model weights in the Hugging Face cache."""
    try:
        from huggingface_hub import scan_cache_dir

        cached = set()
        for repo in scan_cache_dir().repos:
            if repo.repo_type != "model":
                continue
            if any(
                file.file_name.endswith(".safetensors")
                for revision in repo.revisions
                for file in revision.files
            ):
                cached.add(repo.repo_id)
        return cached
    except Exception:
        return set()


def public_models(person_id=ME):
    cached = cached_model_repos()
    return [
        {
            key: value
            for key, value in config.items()
            if key not in ("repo", "batch_size", "layers", "chat_template_args")
        }
        | {
            "has_adapter": has_adapter(model_key, person_id=person_id),
            "cached": config["repo"] in cached,
        }
        for model_key, config in MODELS.items()
    ]


def parse_iter(line):
    m = ITER_RE.search(ANSI_RE.sub("", line or ""))
    return int(m.group(1)) if m else None


def parse_metric(line):
    clean = ANSI_RE.sub("", line or "")
    match = TRAIN_RE.search(clean)
    if match:
        metric = {
            "iter": int(match.group(1)),
            "train_loss": float(match.group(2)),
            "tokens_sec": float(match.group(3)),
        }
        if match.group(4):
            metric["memory_gb"] = float(match.group(4))
        return metric
    match = VAL_RE.search(clean)
    if match:
        return {"iter": int(match.group(1)), "reference_loss": float(match.group(2))}
    return None


def snapshot(brief=False):
    now = time.time()
    with _lock:
        phase = _state["phase"]
        model_key = _state["model"]
        person_id = _state.get("person") or ME
        person_name = _state.get("person_name") or "You"
        started_at = _state.get("started_at")
        phase_seconds = dict(_state.get("phase_seconds") or {})
        phase_started = _state.get("phase_started_at")
        if phase not in BUSY_PHASES:
            _abandon_stale_runs_locked()
        if phase in BUSY_PHASES and phase_started:
            phase_seconds[phase] = now - phase_started
        eta = estimate_eta(
            phase,
            _state.get("iter") or 0,
            _state.get("iters") or 0,
            _state.get("train_started_at"),
            now,
        )
        out = {
            "phase": phase,
            "detail": _state["detail"],
            "iter": _state["iter"],
            "iters": _state["iters"],
            "examples": _state["examples"],
            "sent_texts": _state["sent_texts"],
            "chats": _state["chats"],
            "augmented": _state["augmented"],
            "model": model_key,
            "person": person_id,
            "person_name": person_name,
            "run": _state["run"],
            "busy": phase in BUSY_PHASES,
            "has_adapter": has_adapter(model_key, person_id=person_id),
            "started_at": started_at,
            "phase_started_at": phase_started,
            "train_started_at": _state.get("train_started_at"),
            "phase_seconds": phase_seconds,
            "step_seconds": [
                sum(phase_seconds.get(name, 0) for name in group)
                for group in STEP_PHASES
            ],
            "elapsed_seconds": ((_state.get("ended_at") or now) - started_at) if started_at else None,
            "eta_seconds": eta,
            "mlx": mlx_installed(),
        }
        if brief:
            return out
        return out | {
            "metrics": [dict(point) for point in _state["metrics"]],
            "models": public_models(person_id),
            "runs": [dict(row) for row in _runs_locked()[:MAX_RUNS]],
        }


def inspect_data(person_id=ME):
    try:
        person_id = parse_person_arg(person_id)
        subject = resolve_subject(person_id)
    except ValueError as e:
        raise TwinError(str(e)) from e
    return dataset_profile(target_key=subject["key"], handles=subject.get("handles"))


def list_people():
    from contacts import avatar_html

    return [
        {
            "id": subject["id"],
            "name": subject["name"],
            "handle": subject["handle"],
            "texts": subject["texts"],
            "trained": [
                key for key in MODELS if has_adapter(key, person_id=subject["id"])
            ],
            "avatar": avatar_html(subject["name"], subject["handle"]),
        }
        for subject in list_subjects()
    ]


def start_train(run="complete", model_key=DEFAULT_MODEL, person_id=ME):
    if run not in ("quick", "complete"):
        return False, "run must be quick or complete"
    try:
        config = model_config(model_key)
        person_id = parse_person_arg(person_id)
        subject = resolve_subject(person_id)
    except ValueError as e:
        return False, str(e)
    now = time.time()
    with _lock:
        if _state["phase"] in BUSY_PHASES:
            return False, "already training"
        _cancel.clear()
        _state.update(
            phase="inspecting",
            detail="Auditing the complete message archive…",
            iter=0,
            iters=SMOKE_ITERS if run == "quick" else 0,
            examples=0,
            sent_texts=0,
            chats=0,
            augmented=0,
            model=model_key,
            person=subject["id"],
            person_name=subject["name"],
            run=run,
            metrics=[],
            started_at=now,
            phase_started_at=now,
            train_started_at=None,
            ended_at=None,
            phase_seconds={},
        )
        _drop_model_locked()
        _begin_run_locked(config, run, now, subject)
    threading.Thread(
        target=_train_worker, args=(run, model_key, subject), daemon=True
    ).start()
    return True, ""


def stop_train():
    with _lock:
        if _state["phase"] not in BUSY_PHASES:
            if _abandon_stale_runs_locked():
                return True, ""
            return False, "not training"
        proc = _proc
        _cancel.set()
    _set(phase="cancelling", detail="Stopping training…")
    _kill_proc(proc)
    return True, ""


def _drop_model_locked():
    global _model, _tokenizer, _loaded_key
    _model = None
    _tokenizer = None
    _loaded_key = None


def _set(**kwargs):
    now = time.time()
    with _lock:
        if "phase" in kwargs and kwargs["phase"] != _state["phase"]:
            prev = _state["phase"]
            started = _state.get("phase_started_at")
            extra = {"phase_started_at": now}
            if started and prev in BUSY_PHASES:
                times = dict(_state.get("phase_seconds") or {})
                times[prev] = times.get(prev, 0) + (now - started)
                extra["phase_seconds"] = times
            if kwargs["phase"] == "training" and not _state.get("train_started_at"):
                extra["train_started_at"] = now
            if kwargs["phase"] in ("ready", "error", "cancelled") and _state.get("started_at"):
                extra["ended_at"] = now
            kwargs = {**kwargs, **extra}
        _state.update(kwargs)


def _append_metric(metric):
    with _lock:
        points = _state["metrics"]
        existing = next((p for p in points if p["iter"] == metric["iter"]), None)
        if existing:
            existing.update(metric)
        else:
            points.append(metric)


def _remember_proc(proc):
    global _proc
    _proc = proc


def _kill_proc(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.terminate()
        except OSError:
            pass


def _abandon_stale_runs_locked():
    """A reload or crash can leave a running row after the trainer is idle."""
    if _state["phase"] in BUSY_PHASES:
        return False
    runs = _runs_locked()
    now = time.time()
    changed = False
    for row in runs:
        if row.get("status") != "running":
            continue
        started = row.get("started_at") or now
        row.update(
            {
                "status": "cancelled",
                "ended_at": now,
                "elapsed_seconds": now - started,
                "detail": "Training was interrupted.",
            }
        )
        changed = True
    if changed:
        _save_runs_locked(runs)
    return changed


def _runs_locked():
    global _runs
    if _runs is None:
        try:
            with open(RUNS_PATH, encoding="utf-8") as f:
                data = json.load(f)
            _runs = data if isinstance(data, list) else []
        except (OSError, json.JSONDecodeError, TypeError):
            _runs = []
    return _runs


def _save_runs_locked(runs):
    global _runs
    _runs = runs[:MAX_RUNS]
    os.makedirs(os.path.dirname(RUNS_PATH), exist_ok=True)
    with open(RUNS_PATH, "w", encoding="utf-8") as f:
        json.dump(_runs, f)


def _begin_run_locked(config, run, now, subject):
    row = {
        "id": f"{now:.3f}",
        "model": config["key"],
        "name": config["name"],
        "params": config["params"],
        "run": run,
        "person": subject["id"],
        "person_name": subject["name"],
        "status": "running",
        "started_at": now,
    }
    runs = [row, *[r for r in _runs_locked() if r.get("status") != "running"]]
    _save_runs_locked(runs)


def _complete_run(status):
    now = time.time()
    with _lock:
        runs = _runs_locked()
        current = next((row for row in runs if row.get("status") == "running"), None)
        if not current:
            return
        train_loss = None
        reference_loss = None
        for point in _state.get("metrics") or []:
            if "train_loss" in point:
                train_loss = point["train_loss"]
            if "reference_loss" in point:
                reference_loss = point["reference_loss"]
        current.update(
            {
                "status": status,
                "ended_at": now,
                "elapsed_seconds": now - current.get("started_at", now),
                "phase_seconds": dict(_state.get("phase_seconds") or {}),
                "iter": _state.get("iter") or 0,
                "iters": _state.get("iters") or 0,
                "examples": _state.get("examples") or 0,
                "sent_texts": _state.get("sent_texts") or 0,
                "chats": _state.get("chats") or 0,
                "augmented": _state.get("augmented") or 0,
                "train_loss": train_loss,
                "reference_loss": reference_loss,
                "detail": _state.get("detail") or "",
            }
        )
        _save_runs_locked(runs)


def _train_worker(run, model_key, subject):
    # MLX generation and training share the same GPU memory pool. Let an active
    # reply finish before training starts, then block new replies until it ends.
    with _gen_lock:
        _train_worker_exclusive(run, model_key, subject)


def _train_worker_exclusive(run, model_key, subject):
    try:
        if _cancel.is_set():
            raise TwinCancelled()
        who = "sent text" if subject["id"] == ME else f"{subject['name']}'s texts"
        _set(phase="exporting", detail=f"Building conversation pairs from {who}…")
        is_quick = run == "quick"
        stats = export_dataset(
            limit=SMOKE_LIMIT if is_quick else 0,
            per_chat=240 if is_quick else 0,
            augment=not is_quick,
            target_key=subject["key"],
            name=subject["name"],
        )
        if _cancel.is_set():
            raise TwinCancelled()
        config = model_config(model_key)
        iters = SMOKE_ITERS if is_quick else steps_for_examples(
            stats["train"], config["batch_size"]
        )
        cached = config["repo"] in cached_model_repos()
        if cached:
            load_phase = "loading"
            load_detail = f"Loading {config['name']} {config['params']} from disk…"
        else:
            load_phase = "downloading"
            load_detail = (
                f"Downloading {config['name']} {config['params']} · {config['download']}…"
            )
        _set(
            examples=stats["train"],
            sent_texts=stats["sent_texts"],
            chats=stats["chats"],
            augmented=stats["augmented"],
            iters=iters,
            phase=load_phase,
            detail=load_detail,
        )

        saw_iter = False

        def on_line(line):
            nonlocal saw_iter
            if _cancel.is_set():
                return
            metric = parse_metric(line)
            if metric:
                _append_metric(metric)
            n = parse_iter(line)
            if n is not None:
                saw_iter = True
                _set(
                    phase="training",
                    iter=n,
                    detail=f"Training {n:,}/{iters:,} steps",
                )
            elif saw_iter:
                return
            elif DOWNLOAD_RE.search(line):
                _set(phase="downloading", detail=load_detail)
            elif "Loading pretrained model" in line:
                _set(
                    phase="loading",
                    detail=f"Loading {config['name']} {config['params']} from disk…",
                )
            elif "Training" in line:
                _set(phase="training", detail=f"Starting {iters:,} training steps…")

        if _cancel.is_set():
            raise TwinCancelled()
        target_adapter = adapter_dir(model_key, subject["id"])
        try:
            run_train(
                iters=iters,
                model=config["repo"],
                adapter=target_adapter,
                batch_size=config["batch_size"],
                num_layers=config["layers"],
                on_line=on_line,
                on_proc=_remember_proc,
            )
        except RuntimeError:
            if _cancel.is_set():
                raise TwinCancelled() from None
            raise
        if _cancel.is_set():
            raise TwinCancelled()
        write_run_metadata(
            target_adapter,
            {
                "model": model_key,
                "repo": config["repo"],
                "person": subject["id"],
                "person_name": subject["name"],
                "examples": stats["train"],
                "sent_texts": stats["sent_texts"],
                "chats": stats["chats"],
                "augmented": stats["augmented"],
                "iters": iters,
                "run": run,
            },
        )
        _set(phase="ready", detail="Adapter ready.", iter=iters)
        _complete_run("ready")
    except TwinCancelled:
        _set(phase="cancelled", detail="Training stopped.")
        _complete_run("cancelled")
    except Exception as e:
        _set(phase="error", detail=str(e))
        _complete_run("error")
    finally:
        _remember_proc(None)


def _load_model(model_key, person_id=ME):
    global _model, _tokenizer, _loaded_key
    config = model_config(model_key)
    loaded = (model_key, person_id)
    with _lock:
        if _model is not None and _loaded_key == loaded:
            return _model, _tokenizer
        if _state["phase"] in BUSY_PHASES:
            raise TwinError("Training is running. Wait for it to finish.")
        if not has_adapter(model_key, person_id=person_id):
            raise TwinError(f"{config['name']} has not been trained yet.")
        if not mlx_installed():
            raise TwinError(
                "mlx-lm is not installed. Run: "
                "./.venv/bin/python -m pip install -r twin/requirements.txt"
            )
    try:
        from mlx_lm import load

        model, tokenizer = load(
            config["repo"], adapter_path=resolved_adapter_dir(model_key, person_id)
        )
    except Exception as e:
        raise TwinError(f"Could not load {config['name']}: {e}") from e
    with _lock:
        _model, _tokenizer, _loaded_key = model, tokenizer, loaded
        return _model, _tokenizer


def chat(text, history=None, model_key=DEFAULT_MODEL, person_id=ME):
    text = (text or "").strip()
    if not text:
        raise TwinError("Empty message.")
    try:
        config = model_config(model_key)
        person_id = parse_person_arg(person_id)
        subject = resolve_subject(person_id)
    except ValueError as e:
        raise TwinError(str(e)) from e
    with _gen_lock:
        model, tokenizer = _load_model(model_key, person_id)
        from mlx_lm import generate

        messages = [{"role": "system", "content": system_for(subject["name"])}]
        if history:
            for turn in history[-HISTORY_TURNS:]:
                role = turn.get("role")
                content = (turn.get("content") or "").strip()
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": text})
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **config.get("chat_template_args", {}),
        )
        try:
            return generate(model, tokenizer, prompt=prompt, max_tokens=96).strip()
        except Exception as e:
            with _lock:
                _drop_model_locked()
            raise TwinError(f"The local model could not generate a reply: {e}") from e
