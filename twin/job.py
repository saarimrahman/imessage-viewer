"""Background dataset, training metrics, and on-demand Twin chat."""

import importlib.util
import json
import os
import re
import shutil
import signal
import threading
import time

from twin.export import (
    CHAT_TEMP,
    CHAT_TOP_P,
    CONTEXT_TURNS,
    ME,
    clip_bubbles,
    coerce_chat,
    dataset_profile,
    export_dataset,
    list_subjects,
    parse_person_arg,
    resolve_subject,
    system_for,
)
from twin.train import (
    COMPLETE_LR,
    DEFAULT_MODEL,
    MAX_ITERS,
    MAX_SEQ_LENGTH,
    MODELS,
    QUICK_LR,
    TWIN_DIR,
    adapter_dir,
    epochs_for,
    grad_accumulation_for,
    hash_train_file,
    has_adapter,
    has_weights,
    list_adapter_runs,
    make_run_id,
    model_config,
    note_reference_eval,
    parse_checkpoint_id,
    read_run_metadata,
    resolve_checkpoint,
    resolved_adapter_dir,
    restore_best_checkpoint,
    run_train,
    steps_for_examples,
    write_run_metadata,
)

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
ITER_RE = re.compile(r"Iter\s+(\d+):")
TRAIN_RE = re.compile(
    r"Iter\s+(?P<iter>\d+):\s+Train loss\s+(?P<loss>[0-9.eE+-]+)"
    r"(?:.*?Learning Rate\s+(?P<lr>[0-9.eE+-]+))?"
    r"(?:.*?It/sec\s+(?P<it>[0-9.eE+-]+))?"
    r".*?Tokens/sec\s+(?P<tok>[0-9.eE+-]+)"
    r"(?:.*?Trained Tokens\s+(?P<n_tok>[0-9.eE+-]+))?"
    r"(?:.*?Peak mem\s+(?P<mem>[0-9.eE+-]+)\s+GB)?"
)
VAL_RE = re.compile(r"Iter\s+(\d+):\s+Val loss\s+([0-9.eE+-]+)")
DOWNLOAD_RE = re.compile(r"(?i)(downloading|fetching\s+\d+\s+files)")
HISTORY_TURNS = CONTEXT_TURNS
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
_early_stop = threading.Event()
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
    "run_id": "",
    "data_hash": "",
    "resume_from": "",
    "early_stopped": False,
    "best_iter": 0,
    "best_reference_loss": None,
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


class TwinEarlyStop(Exception):
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
            if key not in (
                "repo",
                "batch_size",
                "layers",
                "chat_template_args",
                "epochs",
                "grad_accumulation",
                "learning_rate",
            )
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
            "iter": int(match.group("iter")),
            "train_loss": float(match.group("loss")),
            "tokens_sec": float(match.group("tok")),
        }
        if match.group("lr"):
            metric["learning_rate"] = float(match.group("lr"))
        if match.group("it"):
            metric["it_sec"] = float(match.group("it"))
        if match.group("n_tok"):
            metric["trained_tokens"] = float(match.group("n_tok"))
        if match.group("mem"):
            metric["memory_gb"] = float(match.group("mem"))
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
            "run_id": _state.get("run_id") or "",
            "data_hash": _state.get("data_hash") or "",
            "resume_from": _state.get("resume_from") or "",
            "early_stopped": bool(_state.get("early_stopped")),
            "best_iter": _state.get("best_iter") or 0,
            "best_reference_loss": _state.get("best_reference_loss"),
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
            "adapter_runs": list_adapter_runs(person_id),
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
            "adapters": list_adapter_runs(subject["id"]),
            "avatar": avatar_html(subject["name"], subject["handle"]),
        }
        for subject in list_subjects()
    ]


def start_train(
    run="complete",
    model_key=DEFAULT_MODEL,
    person_id=ME,
    iters=None,
    resume_from=None,
):
    if run not in ("quick", "complete"):
        return False, "run must be quick or complete"
    if iters is not None:
        try:
            iters = int(iters)
        except (TypeError, ValueError):
            return False, "steps must be a number"
        if iters < 1 or iters > MAX_ITERS:
            return False, f"steps must be between 1 and {MAX_ITERS:,}"
    resume = None
    try:
        config = model_config(model_key)
        person_id = parse_person_arg(person_id)
        subject = resolve_subject(person_id)
        if resume_from:
            resume = resolve_checkpoint(resume_from, subject["id"])
            if resume["model"] != model_key:
                return False, "Resume from a checkpoint of the same model."
    except ValueError as e:
        return False, str(e)
    now = time.time()
    with _lock:
        if _state["phase"] in BUSY_PHASES:
            return False, "already training"
        _cancel.clear()
        _early_stop.clear()
        _state.update(
            phase="inspecting",
            detail="Auditing the complete message archive…",
            iter=0,
            iters=iters if iters is not None else (SMOKE_ITERS if run == "quick" else 0),
            examples=0,
            sent_texts=0,
            chats=0,
            augmented=0,
            model=model_key,
            person=subject["id"],
            person_name=subject["name"],
            run=run,
            run_id="",
            data_hash="",
            resume_from=resume["id"] if resume else "",
            early_stopped=False,
            best_iter=0,
            best_reference_loss=None,
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
        target=_train_worker, args=(run, model_key, subject, iters, resume), daemon=True
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
        "run_id": "",
        "data_hash": "",
        "resume_from": _state.get("resume_from") or "",
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
                "run_id": _state.get("run_id") or "",
                "data_hash": _state.get("data_hash") or "",
                "resume_from": _state.get("resume_from") or "",
                "train_loss": train_loss,
                "reference_loss": (
                    _state.get("best_reference_loss")
                    if _state.get("best_reference_loss") is not None
                    else reference_loss
                ),
                "early_stopped": bool(_state.get("early_stopped")),
                "best_iter": _state.get("best_iter") or 0,
                "detail": _state.get("detail") or "",
            }
        )
        _save_runs_locked(runs)


def _train_worker(run, model_key, subject, custom_iters=None, resume=None):
    # MLX generation and training share the same GPU memory pool. Let an active
    # reply finish before training starts, then block new replies until it ends.
    with _gen_lock:
        _train_worker_exclusive(run, model_key, subject, custom_iters, resume)


def _train_worker_exclusive(run, model_key, subject, custom_iters=None, resume=None):
    target_adapter = None
    metadata = {}
    tracker = {}
    iters = 0
    try:
        if _cancel.is_set():
            raise TwinCancelled()
        who = "sent text" if subject["id"] == ME else f"{subject['name']}'s texts"
        _set(phase="exporting", detail=f"Building conversation pairs from {who}…")
        is_quick = run == "quick"
        config = model_config(model_key)
        stats = export_dataset(
            limit=SMOKE_LIMIT if is_quick else 0,
            per_chat=240 if is_quick else 0,
            target_key=subject["key"],
            name=subject["name"],
            model_key=model_key,
            max_seq_length=MAX_SEQ_LENGTH,
            chat_template_args=config.get("chat_template_args"),
        )
        if _cancel.is_set():
            raise TwinCancelled()
        iters = custom_iters if custom_iters is not None else (
            SMOKE_ITERS if is_quick else steps_for_examples(
                stats["train"],
                config["batch_size"],
                epochs=epochs_for(config, complete=True),
            )
        )
        data_hash = hash_train_file(TWIN_DIR)
        created = time.time()
        root = adapter_dir(model_key, subject["id"])
        os.makedirs(root, exist_ok=True)
        run_id = make_run_id(created, data_hash, iters, root)
        target_adapter = os.path.join(root, run_id)
        os.makedirs(target_adapter, exist_ok=True)
        metadata = {
            "model": model_key,
            "repo": config["repo"],
            "person": subject["id"],
            "person_name": subject["name"],
            "run_id": run_id,
            "created_at": created,
            "data_hash": data_hash,
            "examples": stats["train"],
            "sent_texts": stats["sent_texts"],
            "chats": stats["chats"],
            "augmented": stats["augmented"],
            "valid": stats.get("valid") or 0,
            "test": stats.get("test") or 0,
            "sessions": stats.get("sessions") or 0,
            "iters": iters,
            "run": run,
            "batch_size": config["batch_size"],
            "layers": config["layers"],
            "learning_rate": QUICK_LR if is_quick else COMPLETE_LR,
            "epochs": 1 if is_quick else epochs_for(config, complete=True),
            "grad_accumulation": 1 if is_quick else grad_accumulation_for(config),
            "resume_from": resume["id"] if resume else "",
            "status": "running",
        }
        write_run_metadata(target_adapter, metadata)
        _set(
            examples=stats["train"],
            sent_texts=stats["sent_texts"],
            chats=stats["chats"],
            augmented=stats["augmented"],
            iters=iters,
            run_id=run_id,
            data_hash=data_hash,
        )
        with _lock:
            current = next((row for row in _runs_locked() if row.get("status") == "running"), None)
            if current:
                current.update(run_id=run_id, data_hash=data_hash, iters=iters)
                _save_runs_locked(_runs_locked())
        cached = config["repo"] in cached_model_repos()
        if cached:
            load_phase = "loading"
            load_detail = f"Loading {config['name']} {config['params']} from disk…"
        else:
            load_phase = "downloading"
            load_detail = (
                f"Downloading {config['name']} {config['params']} · {config['download']}…"
            )
        _set(phase=load_phase, detail=load_detail)

        saw_iter = False
        tracker = {}
        steps_per_eval = max(1, iters // 12)

        def on_line(line):
            nonlocal saw_iter
            if _cancel.is_set():
                return
            metric = parse_metric(line)
            if metric:
                _append_metric(metric)
                if (
                    not is_quick
                    and "reference_loss" in metric
                    and note_reference_eval(tracker, metric["reference_loss"], metric["iter"])
                    and not _early_stop.is_set()
                ):
                    _early_stop.set()
                    _set(
                        best_iter=tracker.get("best_iter") or 0,
                        best_reference_loss=tracker.get("best"),
                        detail="Holdout loss plateaued. Keeping the best checkpoint…",
                    )
                    _kill_proc(_proc)
            n = parse_iter(line)
            if n is not None:
                saw_iter = True
                _set(
                    phase="training",
                    iter=n,
                    detail=(
                        "Holdout loss plateaued. Keeping the best checkpoint…"
                        if _early_stop.is_set()
                        else f"Training {n:,}/{iters:,} steps"
                    ),
                    best_iter=tracker.get("best_iter") or 0,
                    best_reference_loss=tracker.get("best"),
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
        try:
            run_train(
                iters=iters,
                model=config["repo"],
                adapter=target_adapter,
                batch_size=config["batch_size"],
                num_layers=config["layers"],
                on_line=on_line,
                on_proc=_remember_proc,
                save_every=None if is_quick else steps_per_eval,
                resume_adapter_file=resume["weights"] if resume else None,
                learning_rate=QUICK_LR if is_quick else COMPLETE_LR,
                grad_accumulation=1 if is_quick else grad_accumulation_for(config),
                schedule=not is_quick,
                run_test=not is_quick,
                val_batches=25 if is_quick else -1,
                chat_template_args=config.get("chat_template_args"),
            )
        except RuntimeError:
            if _cancel.is_set():
                raise TwinCancelled() from None
            if _early_stop.is_set():
                raise TwinEarlyStop() from None
            raise
        if _cancel.is_set():
            raise TwinCancelled()
        if _early_stop.is_set():
            raise TwinEarlyStop()
        _finish_ready(target_adapter, metadata, tracker, early=False, iters=iters)
    except TwinEarlyStop:
        if target_adapter and has_weights(target_adapter):
            _finish_ready(target_adapter, metadata, tracker, early=True, iters=iters)
        else:
            if target_adapter:
                shutil.rmtree(target_adapter, ignore_errors=True)
            _set(phase="cancelled", detail="Stopped before a checkpoint was saved.")
            _complete_run("cancelled")
    except TwinCancelled:
        if target_adapter and has_weights(target_adapter):
            meta = {
                **(read_run_metadata(target_adapter)),
                "status": "cancelled",
            }
            write_run_metadata(target_adapter, meta)
        elif target_adapter:
            shutil.rmtree(target_adapter, ignore_errors=True)
        _set(phase="cancelled", detail="Training stopped.")
        _complete_run("cancelled")
    except Exception as e:
        if target_adapter and not has_weights(target_adapter):
            shutil.rmtree(target_adapter, ignore_errors=True)
        elif target_adapter:
            write_run_metadata(
                target_adapter,
                {**(read_run_metadata(target_adapter)), "status": "error"},
            )
        _set(phase="error", detail=str(e))
        _complete_run("error")
    finally:
        _remember_proc(None)


def _finish_ready(target_adapter, metadata, tracker, early, iters):
    best_iter = int(tracker.get("best_iter") or 0)
    best_loss = tracker.get("best")
    if target_adapter and best_iter:
        restore_best_checkpoint(target_adapter, best_iter)
    scored = None
    if target_adapter and (metadata or {}).get("run") == "complete" and not _cancel.is_set():
        try:
            from twin.eval import score_checkpoints

            _set(detail="Scoring holdout replies…")
            scored = score_checkpoints(
                target_adapter,
                TWIN_DIR,
                metadata.get("model") or DEFAULT_MODEL,
                split="valid",
                max_examples=24,
                person_name=metadata.get("person_name") or "You",
            )
        except Exception:
            scored = None
    meta = {**(read_run_metadata(target_adapter) if target_adapter else {}), **(metadata or {})}
    meta.update(
        status="ready",
        early_stopped=bool(early),
        best_iter=best_iter,
        best_reference_loss=best_loss,
        holdout_score=scored,
    )
    if target_adapter:
        write_run_metadata(target_adapter, meta)
    ready = {
        "phase": "ready",
        "detail": (
            "Holdout loss plateaued. Kept the best checkpoint."
            if early
            else "Adapter ready."
        ),
        "early_stopped": bool(early),
        "best_iter": best_iter,
        "best_reference_loss": best_loss,
    }
    if not early:
        ready["iter"] = iters
    _set(**ready)
    _complete_run("ready")


def _load_model(model_key, person_id=ME, run_id=None, step="latest"):
    global _model, _tokenizer, _loaded_key
    config = model_config(model_key)
    loaded = (model_key, person_id, run_id or "", str(step or "latest"))
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
    adapter_path = resolved_adapter_dir(
        model_key, person_id, run_id=run_id, step=step
    )
    if not os.path.isfile(os.path.join(adapter_path, "adapters.safetensors")):
        raise TwinError(f"{config['name']} has not been trained yet.")
    try:
        from mlx_lm import load

        model, tokenizer = load(config["repo"], adapter_path=adapter_path)
    except Exception as e:
        raise TwinError(f"Could not load {config['name']}: {e}") from e
    with _lock:
        _model, _tokenizer, _loaded_key = model, tokenizer, loaded
        return _model, _tokenizer


def chat(
    text,
    history=None,
    model_key=DEFAULT_MODEL,
    person_id=ME,
    adapter=None,
    retrieve_pairs=True,
    temp=CHAT_TEMP,
    top_p=CHAT_TOP_P,
    to=None,
):
    text = (text or "").strip()
    if not text:
        raise TwinError("Empty message.")
    run_id = None
    step = "latest"
    try:
        person_id = parse_person_arg(person_id)
        subject = resolve_subject(person_id)
        if adapter:
            model_key, run_id, step = parse_checkpoint_id(adapter)
        config = model_config(model_key)
    except ValueError as e:
        raise TwinError(str(e)) from e
    with _gen_lock:
        model, tokenizer = _load_model(model_key, person_id, run_id, step)
        from mlx_lm import generate
        from twin.chat import generate_kwargs

        messages = [{"role": "system", "content": system_for(subject["name"], peer=to)}]
        shots = []
        if retrieve_pairs:
            from twin.retrieve import RETRIEVE_K, few_shot_messages, load_index, retrieve

            pairs = retrieve(
                text, load_index(TWIN_DIR), k=RETRIEVE_K, exclude=[text], peer=to
            )
            shots = few_shot_messages(pairs)
        live = []
        if history:
            for turn in history:
                role = turn.get("role")
                content = (turn.get("content") or "").strip()
                if role in ("user", "assistant") and content:
                    live.append({"role": role, "content": content})
        live.append({"role": "user", "content": text})
        budget = max(2, HISTORY_TURNS - len(shots))
        live = live[-budget:]
        messages = coerce_chat(messages + shots + live)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **config.get("chat_template_args", {}),
        )
        try:
            kwargs = generate_kwargs(96, temp=temp, top_p=top_p, seed=0)
            return clip_bubbles(generate(model, tokenizer, prompt=prompt, **kwargs).strip())
        except Exception as e:
            with _lock:
                _drop_model_locked()
            raise TwinError(f"The local model could not generate a reply: {e}") from e
