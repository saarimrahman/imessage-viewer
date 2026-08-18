#!/usr/bin/env python3
"""Train model-specific QLoRA adapters with MLX LM."""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CACHE_DIR

TWIN_DIR = os.path.join(CACHE_DIR, "twin")

# Model keys are permanent adapter identities. Do not reuse an existing key for
# a newer base model: MLX adapters only work with the exact base that trained
# them.
MODELS = {
    "qwen3-capable": {
        "key": "qwen3-capable",
        "name": "Qwen 3 4B Instruct 2507",
        "params": "4B",
        "repo": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
        "download": "~2.3 GB",
        "memory": "Comfortable on 16 GB",
        "description": "Stable default from general Tiny benchmarks, not Twin holdout scores. Use Complete, not Quick, before judging reply quality.",
        "publisher": "Alibaba",
        "category": "Featured benchmark picks",
        "batch_size": 1,
        "layers": 8,
        "recommended": True,
    },
    "qwen35-4b": {
        "key": "qwen35-4b",
        "name": "Qwen 3.5 4B",
        "params": "4B",
        "repo": "mlx-community/Qwen3.5-4B-4bit",
        "download": "~2.3 GB",
        "memory": "Comfortable on 16 GB",
        "description": "A Twin candidate. Compare it on holdout replies before replacing Qwen 3 4B Instruct.",
        "publisher": "Alibaba",
        "category": "Featured benchmark picks",
        "batch_size": 1,
        "layers": 8,
        "chat_template_args": {"enable_thinking": False},
    },
    "g9v3-3b": {
        "key": "g9v3-3b",
        "name": "G9v3 3B",
        "params": "3B",
        "repo": "cof139/G9v3-3B-mlx-4Bit",
        "download": "~1.7 GB",
        "memory": "Comfortable on 16 GB",
        "description": "The current Tiny intelligence leader, using a community MLX conversion.",
        "publisher": "AI9Stars",
        "category": "Featured benchmark picks",
        "batch_size": 1,
        "layers": 8,
        "chat_template_args": {"enable_thinking": False},
    },
    "minicpm5-1b": {
        "key": "minicpm5-1b",
        "name": "MiniCPM5 1B",
        "params": "1B",
        "repo": "mlx-community/MiniCPM5-1B-4bit",
        "download": "~620 MB",
        "memory": "Very comfortable on 16 GB",
        "description": "A benchmark-leading small model for fast iteration and inexpensive full runs.",
        "publisher": "OpenBMB",
        "category": "Featured benchmark picks",
        "batch_size": 2,
        "layers": 8,
        "chat_template_args": {"enable_thinking": False},
    },
    "nanbeige41-3b": {
        "key": "nanbeige41-3b",
        "name": "Nanbeige 4.1 3B",
        "params": "3.9B",
        "repo": "mlx-community/Nanbeige4.1-3B-4bit",
        "download": "~2.2 GB",
        "memory": "Comfortable on 16 GB",
        "description": "A strong recent Tiny benchmark entry with a standard non-reasoning chat format.",
        "publisher": "Nanbeige",
        "category": "Featured benchmark picks",
        "batch_size": 1,
        "layers": 8,
    },
    "nemotron3-nano-4b": {
        "key": "nemotron3-nano-4b",
        "name": "Nemotron 3 Nano 4B",
        "params": "4B",
        "repo": "mlx-community/NVIDIA-Nemotron-3-Nano-4B-4bit",
        "download": "~2.3 GB",
        "memory": "Comfortable on 16 GB",
        "description": "NVIDIA's compact hybrid model, tuned for strong capability per active parameter.",
        "publisher": "NVIDIA",
        "category": "Featured benchmark picks",
        "batch_size": 1,
        "layers": 8,
        "chat_template_args": {"enable_thinking": False},
    },
    "ministral3-3b": {
        "key": "ministral3-3b",
        "name": "Ministral 3 3B Instruct",
        "params": "3B",
        "repo": "mlx-community/Ministral-3-3B-Instruct-2512-4bit",
        "download": "~2.8 GB",
        "memory": "Comfortable on 16 GB",
        "description": "Mistral's recent edge model with a verified system-message chat template.",
        "publisher": "Mistral",
        "category": "Featured benchmark picks",
        "batch_size": 1,
        "layers": 8,
    },
    "phi4-mini": {
        "key": "phi4-mini",
        "name": "Phi-4 Mini Instruct",
        "params": "3.8B",
        "repo": "mlx-community/Phi-4-mini-instruct-4bit",
        "download": "~2.2 GB",
        "memory": "Comfortable on 16 GB",
        "description": "Microsoft's compact instruction model and a useful alternative writing voice.",
        "publisher": "Microsoft",
        "category": "Featured benchmark picks",
        "batch_size": 1,
        "layers": 8,
    },
    "granite41-3b": {
        "key": "granite41-3b",
        "name": "Granite 4.1 3B",
        "params": "3B",
        "repo": "mlx-community/granite-4.1-3b-4bit",
        "download": "~2.1 GB",
        "memory": "Comfortable on 16 GB",
        "description": "IBM's efficient current-generation model with a clean conversational template.",
        "publisher": "IBM",
        "category": "Featured benchmark picks",
        "batch_size": 1,
        "layers": 8,
    },
    "gemma3-4b": {
        "key": "gemma3-4b",
        "name": "Gemma 3 4B",
        "params": "4B",
        "repo": "mlx-community/gemma-3-4b-it-4bit",
        "download": "~3.4 GB",
        "memory": "Comfortable on 16 GB",
        "description": "Google's larger compact model for more capacity than the 1B Gemma option.",
        "publisher": "Google",
        "category": "Featured benchmark picks",
        "batch_size": 1,
        "layers": 8,
    },
    "qwen3-8b": {
        "key": "qwen3-8b",
        "name": "Qwen 3 8B",
        "params": "8B",
        "repo": "mlx-community/Qwen3-8B-4bit",
        "download": "~4.6 GB",
        "memory": "16 GB limit · close heavy apps",
        "description": "The highest-capacity Qwen that Twin offers on a 16 GB Mac.",
        "publisher": "Alibaba",
        "category": "Maximum quality on 16 GB",
        "batch_size": 1,
        "layers": 4,
        "chat_template_args": {"enable_thinking": False},
    },
    "llama31-8b": {
        "key": "llama31-8b",
        "name": "Llama 3.1 8B Instruct",
        "params": "8B",
        "repo": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        "download": "~4.5 GB",
        "memory": "16 GB limit · close heavy apps",
        "description": "A mature 8B baseline with broad language capability and a stable MLX path.",
        "publisher": "Meta",
        "category": "Maximum quality on 16 GB",
        "batch_size": 1,
        "layers": 4,
    },
    "granite41-8b": {
        "key": "granite41-8b",
        "name": "Granite 4.1 8B",
        "params": "8B",
        "repo": "mlx-community/granite-4.1-8b-4bit",
        "download": "~5.3 GB",
        "memory": "16 GB limit · close heavy apps",
        "description": "A newer 8B alternative at the upper edge of practical local QLoRA training.",
        "publisher": "IBM",
        "category": "Maximum quality on 16 GB",
        "batch_size": 1,
        "layers": 4,
    },
    "qwen3-compact": {
        "key": "qwen3-compact",
        "name": "Qwen 3 0.6B",
        "params": "0.6B",
        "repo": "mlx-community/Qwen3-0.6B-4bit",
        "download": "~350 MB",
        "memory": "Very comfortable on 16 GB",
        "description": "The quickest current Qwen for experiments and short replies.",
        "publisher": "Alibaba",
        "category": "Efficient alternatives",
        "batch_size": 4,
        "layers": 8,
        "chat_template_args": {"enable_thinking": False},
    },
    "qwen3-balanced": {
        "key": "qwen3-balanced",
        "name": "Qwen 3 1.7B",
        "params": "1.7B",
        "repo": "mlx-community/Qwen3-1.7B-4bit",
        "download": "~1 GB",
        "memory": "Very comfortable on 16 GB",
        "description": "Natural replies and fast training with more capacity than the smallest models.",
        "publisher": "Alibaba",
        "category": "Efficient alternatives",
        "batch_size": 2,
        "layers": 8,
        "chat_template_args": {"enable_thinking": False},
    },
    "gemma3-compact": {
        "key": "gemma3-compact",
        "name": "Gemma 3 Compact",
        "params": "1B",
        "repo": "mlx-community/gemma-3-1b-it-4bit",
        "download": "~770 MB",
        "memory": "Very comfortable on 16 GB",
        "description": "A compact, current model with a different voice and tokenizer from Qwen.",
        "publisher": "Google",
        "category": "Efficient alternatives",
        "batch_size": 2,
        "layers": 8,
    },
    "smollm3-capable": {
        "key": "smollm3-capable",
        "name": "SmolLM3 Capable",
        "params": "3B",
        "repo": "mlx-community/SmolLM3-3B-4bit",
        "download": "~1.8 GB",
        "memory": "Comfortable on 16 GB",
        "description": "A recent open model from Hugging Face with more room to learn your style.",
        "publisher": "Hugging Face",
        "category": "Efficient alternatives",
        "batch_size": 1,
        "layers": 8,
        "chat_template_args": {"enable_thinking": False},
    },
    "compact": {
        "key": "compact",
        "name": "Qwen 2.5 Compact",
        "params": "0.5B",
        "repo": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        "download": "~290 MB",
        "memory": "Lowest memory · previous generation",
        "description": "Keep using an existing 0.5B adapter, or run a lightweight baseline.",
        "publisher": "Alibaba",
        "category": "Existing Qwen 2.5 adapters",
        "batch_size": 4,
        "layers": 8,
    },
    "balanced": {
        "key": "balanced",
        "name": "Qwen 2.5 Balanced",
        "params": "1.5B",
        "repo": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        "download": "~1 GB",
        "memory": "Good on 16 GB · previous generation",
        "description": "Keep using an existing 1.5B adapter with its original base model.",
        "publisher": "Alibaba",
        "category": "Existing Qwen 2.5 adapters",
        "batch_size": 2,
        "layers": 8,
    },
    "capable": {
        "key": "capable",
        "name": "Qwen 2.5 Capable",
        "params": "3B",
        "repo": "mlx-community/Qwen2.5-3B-Instruct-4bit",
        "download": "~1.8 GB",
        "memory": "Slower on 16 GB · previous generation",
        "description": "Keep using an existing 3B adapter with its original base model.",
        "publisher": "Alibaba",
        "category": "Existing Qwen 2.5 adapters",
        "batch_size": 1,
        "layers": 8,
    },
}
DEFAULT_MODEL = "qwen3-capable"
MODEL = MODELS["compact"]["repo"]  # Backwards-compatible public constant.
ADAPTER_WEIGHTS = "adapters.safetensors"
LEGACY_RUN = "legacy"
MAX_ITERS = 1_000_000
COMPLETE_EPOCHS = 3
COMPLETE_LR = 1e-5
QUICK_LR = 1e-4
EFFECTIVE_BATCH = 8
MAX_SEQ_LENGTH = 768
LORA_ENTRY = str(Path(__file__).resolve().parent / "mlx_lora.py")
EARLY_STOP_PATIENCE = 6
EARLY_STOP_MIN_DELTA = 0.01
EARLY_STOP_MIN_EVALS = 3
CHECKPOINT_RE = re.compile(r"^(\d{7})_adapters\.safetensors$")


def model_config(model_key):
    try:
        return MODELS[model_key]
    except KeyError as e:
        raise ValueError(f"Unknown model: {model_key}") from e


def data_dir(person_id="me"):
    """Where one person's exported dataset lives.

    Adapters were already stored for each person, but every export wrote to
    TWIN_DIR. Training a second person therefore overwrote the first person's
    data, and chat retrieved few-shot pairs written by whoever was exported
    last.
    """
    if not person_id or person_id == "me":
        return TWIN_DIR
    return os.path.join(TWIN_DIR, "people", person_id, "data")


def adapter_dir(model_key, person_id="me"):
    if not person_id or person_id == "me":
        return os.path.join(TWIN_DIR, "adapters", model_key)
    return os.path.join(TWIN_DIR, "people", person_id, "adapters", model_key)


def run_dir(model_key, run_id, person_id="me"):
    if run_id == LEGACY_RUN:
        return adapter_dir(model_key, person_id)
    return os.path.join(adapter_dir(model_key, person_id), run_id)


# Existing callers/tests imported ADAPTER before model selection existed.
ADAPTER = adapter_dir("compact")


def has_weights(path):
    return os.path.isfile(os.path.join(path, ADAPTER_WEIGHTS)) or bool(
        checkpoint_steps(path)
    )


def checkpoint_steps(path):
    if not os.path.isdir(path):
        return []
    steps = []
    for name in os.listdir(path):
        match = CHECKPOINT_RE.match(name)
        if match:
            steps.append(int(match.group(1)))
    return sorted(steps)


def has_adapter(model_key="compact", adapter=None, person_id="me"):
    if adapter is not None:
        return has_weights(adapter)
    return any(iter_run_paths(model_key, person_id))


def iter_run_paths(model_key, person_id="me"):
    """Yield (run_id, path) for adapter folders that have trainable weights."""
    root = adapter_dir(model_key, person_id)
    if has_weights(root):
        yield LEGACY_RUN, root
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            if name in (LEGACY_RUN, ".load"):
                continue
            path = os.path.join(root, name)
            if os.path.isdir(path) and has_weights(path):
                yield name, path
    if (
        (not person_id or person_id == "me")
        and model_key == "compact"
        and not has_weights(root)
    ):
        legacy = os.path.join(TWIN_DIR, "adapters")
        if has_weights(legacy) and legacy != root:
            yield LEGACY_RUN, legacy


def read_run_metadata(path):
    try:
        with open(os.path.join(path, "twin_run.json"), encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def hash_train_file(data_dir):
    path = os.path.join(data_dir, "train.jsonl")
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def make_run_id(created_at, data_hash, iters, root):
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(created_at))
    base = f"{stamp}-{data_hash}-{int(iters)}"
    candidate = base
    extra = 2
    while os.path.exists(os.path.join(root, candidate)):
        candidate = f"{base}-{extra}"
        extra += 1
    return candidate


def parse_checkpoint_id(value):
    text = (value or "").strip()
    parts = text.split("/")
    if len(parts) < 3:
        raise ValueError("Checkpoint id must be model/run/step.")
    model_key, step = parts[0], parts[-1]
    run_id = "/".join(parts[1:-1])
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}")
    if not run_id:
        raise ValueError("Checkpoint id is missing a run.")
    if step != "latest" and not step.isdigit():
        raise ValueError("Checkpoint step must be latest or a number.")
    return model_key, run_id, step


def checkpoint_id(model_key, run_id, step="latest"):
    return f"{model_key}/{run_id}/{step}"


def checkpoint_weight_file(path, step="latest"):
    if step == "latest" or step is None:
        latest = os.path.join(path, ADAPTER_WEIGHTS)
        if os.path.isfile(latest):
            return latest
        steps = checkpoint_steps(path)
        if not steps:
            return None
        step = steps[-1]
    numbered = os.path.join(path, f"{int(step):07d}_{ADAPTER_WEIGHTS}")
    return numbered if os.path.isfile(numbered) else None


def resolve_checkpoint(value, person_id="me"):
    model_key, run_id, step = parse_checkpoint_id(value)
    path = None
    for found_id, found_path in iter_run_paths(model_key, person_id):
        if found_id == run_id:
            path = found_path
            break
    if path is None:
        raise ValueError("That checkpoint does not exist.")
    weights = checkpoint_weight_file(path, step)
    if not weights:
        raise ValueError("That checkpoint has no saved weights.")
    return {
        "id": checkpoint_id(model_key, run_id, step),
        "model": model_key,
        "run_id": run_id,
        "step": step,
        "path": path,
        "weights": weights,
    }


def _replace_link(dest, src):
    if os.path.lexists(dest):
        os.remove(dest)
    os.symlink(os.path.abspath(src), dest)


def load_dir_for(path, step="latest"):
    """Directory MLX can load: adapter_config.json + adapters.safetensors."""
    weights = checkpoint_weight_file(path, step)
    if not weights:
        # A named step that was never written must not fall back to the final
        # weights. Doing so scored four "checkpoints" of one run as identical
        # and wasted the run. `save_every` floors at 50 and is `iters // 20`,
        # so a step has to be a multiple of that.
        if str(step).isdigit():
            saved = checkpoint_steps(path)
            raise FileNotFoundError(
                f"No checkpoint at step {step} in {path}. Saved steps: "
                f"{saved if saved else 'none'}"
            )
        return path
    latest = os.path.join(path, ADAPTER_WEIGHTS)
    if os.path.abspath(weights) == os.path.abspath(latest):
        return path
    step_n = int(step) if str(step).isdigit() else checkpoint_steps(path)[-1]
    dest_dir = os.path.join(path, ".load", str(step_n))
    os.makedirs(dest_dir, exist_ok=True)
    _replace_link(os.path.join(dest_dir, ADAPTER_WEIGHTS), weights)
    config = os.path.join(path, "adapter_config.json")
    if os.path.isfile(config):
        _replace_link(os.path.join(dest_dir, "adapter_config.json"), config)
    return dest_dir


def resolved_adapter_dir(model_key, person_id="me", run_id=None, step="latest"):
    if run_id:
        path = run_dir(model_key, run_id, person_id)
        if run_id == LEGACY_RUN and not has_weights(path):
            fallback = os.path.join(TWIN_DIR, "adapters")
            if (
                (not person_id or person_id == "me")
                and model_key == "compact"
                and has_weights(fallback)
            ):
                path = fallback
        return load_dir_for(path, step)
    runs = list(iter_run_paths(model_key, person_id))
    if not runs:
        return adapter_dir(model_key, person_id)
    latest_id, latest_path = _newest_run(runs)
    return load_dir_for(latest_path, "latest")


def _path_created_at(path, meta=None):
    meta = meta if meta is not None else read_run_metadata(path)
    created = meta.get("created_at")
    if isinstance(created, (int, float)) and created > 0:
        return created
    try:
        return os.path.getmtime(os.path.join(path, ADAPTER_WEIGHTS))
    except OSError:
        try:
            return os.path.getmtime(path)
        except OSError:
            return 0


def _newest_run(runs):
    return max(runs, key=lambda item: _path_created_at(item[1]))


def public_checkpoints(path, model_key, run_id, iters=0, status="", recommended=None):
    """List the checkpoints for a run, best first.

    The last checkpoint is the worst one measured. Training to the end of the
    schedule cost 95% on style distance for the 4B, 32% for the 8B and 129% for
    a per-contact run. When a holdout sweep has named a step, that step leads the
    list and the chat page selects it.
    """
    numbered = checkpoint_steps(path)
    if status == "ready" and iters:
        latest_n = iters
    elif numbered:
        latest_n = numbered[-1]
    else:
        latest_n = iters or 0
    out = [
        {
            "id": checkpoint_id(model_key, run_id, "latest"),
            "step": "latest",
            "step_n": latest_n,
        }
    ]
    for step in reversed(numbered):
        if latest_n and step == latest_n:
            continue
        out.append(
            {
                "id": checkpoint_id(model_key, run_id, step),
                "step": step,
                "step_n": step,
            }
        )
    if recommended:
        # The best step is often the last one, and that entry is the "latest"
        # row rather than a numbered one. Marking only numbered rows left the
        # right weights selected with nothing saying why.
        for i, ckpt in enumerate(out):
            if ckpt["step_n"] == int(recommended):
                ckpt["recommended"] = True
                out.insert(0, out.pop(i))
                break
    return out


def list_adapter_runs(person_id="me", model_key=None):
    keys = [model_key] if model_key else list(MODELS)
    runs = []
    for key in keys:
        config = MODELS[key]
        for run_id, path in iter_run_paths(key, person_id):
            meta = read_run_metadata(path)
            iters = int(meta["iters"]) if meta.get("iters") else 0
            created = _path_created_at(path, meta)
            runs.append(
                {
                    "id": f"{key}/{run_id}",
                    "model": key,
                    "run_id": run_id,
                    "name": config["name"],
                    "params": config["params"],
                    "created_at": created,
                    "data_hash": meta.get("data_hash") or "",
                    "iters": iters,
                    "examples": int(meta["examples"]) if meta.get("examples") else 0,
                    "run": meta.get("run") or ("legacy" if run_id == LEGACY_RUN else ""),
                    "resume_from": meta.get("resume_from") or "",
                    "recommended_score": meta.get("recommended_score"),
                    "checkpoints": public_checkpoints(
                        path,
                        key,
                        run_id,
                        iters,
                        meta.get("status") or "",
                        meta.get("recommended_step"),
                    ),
                }
            )
    runs.sort(key=lambda row: row["created_at"], reverse=True)
    return runs


def save_every_for(iters):
    if iters <= 50:
        return max(iters, 1)
    return max(50, iters // 20)


def grad_accumulation_for(config, batch_size=None):
    """Reach an effective batch near 8 without growing the MLX microbatch."""
    batch = max(1, int(batch_size or config.get("batch_size") or 1))
    target = 4 if str(config.get("params") or "").startswith("8") else EFFECTIVE_BATCH
    return max(1, target // batch)


def epochs_for(config, complete=True):
    if not complete:
        return 1
    return int(config.get("epochs") or COMPLETE_EPOCHS)


def last_train_error(lines, limit=240):
    """Last useful MLX log line, for the UI when the trainer process dies."""
    for line in reversed(lines or ()):
        text = str(line).strip()
        if not text:
            continue
        if text.startswith(("File ", "~", "^", "The above")):
            continue
        return text[:limit]
    return ""


def write_train_config(path, iters, learning_rate, lora_rank=None, lora_scale=None, lora_dropout=None):
    """Warmup plus cosine decay. MLX reads this as --config."""
    import yaml

    warmup = max(1, min(100, max(1, iters) // 20))
    payload = {
        "lr_schedule": {
            "name": "cosine_decay",
            "arguments": [float(learning_rate), int(iters), float(learning_rate) * 0.1],
            "warmup": warmup,
            "warmup_init": 0.0,
        }
    }
    if lora_rank or lora_scale or lora_dropout:
        payload["lora_parameters"] = {
            "rank": int(lora_rank or 8),
            "scale": float(lora_scale if lora_scale is not None else 20.0),
            "dropout": float(lora_dropout or 0.0),
        }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return path


def note_reference_eval(
    tracker,
    loss,
    iteration,
    min_delta=EARLY_STOP_MIN_DELTA,
    patience=EARLY_STOP_PATIENCE,
    min_evals=EARLY_STOP_MIN_EVALS,
):
    """Update reference-loss early-stop state. Lower loss is better."""
    tracker["evals"] = int(tracker.get("evals") or 0) + 1
    best = tracker.get("best")
    if best is None or loss <= best - min_delta:
        tracker["best"] = loss
        tracker["best_iter"] = iteration
        tracker["stale"] = 0
        return False
    tracker["stale"] = int(tracker.get("stale") or 0) + 1
    return tracker["evals"] >= min_evals and tracker["stale"] >= patience


def restore_best_checkpoint(path, best_iter):
    """Copy the best numbered save over adapters.safetensors. Return the source path."""
    latest = os.path.join(path, ADAPTER_WEIGHTS)
    if not best_iter:
        return latest if os.path.isfile(latest) else None
    numbered = os.path.join(path, f"{int(best_iter):07d}_{ADAPTER_WEIGHTS}")
    src = numbered if os.path.isfile(numbered) else None
    if src is None:
        earlier = [step for step in checkpoint_steps(path) if step <= int(best_iter)]
        if earlier:
            src = os.path.join(path, f"{earlier[-1]:07d}_{ADAPTER_WEIGHTS}")
    if src is None or not os.path.isfile(src):
        return latest if os.path.isfile(latest) else None
    if os.path.abspath(src) != os.path.abspath(latest):
        shutil.copy2(src, latest)
    return src


def steps_for_examples(examples, batch_size, epochs=1):
    """Microbatches to cover ``epochs`` passes through the shuffled training rows."""
    return max(1, math.ceil(examples / max(1, batch_size)) * max(1, int(epochs)))


def train_command(
    model,
    data,
    adapter,
    iters,
    batch_size=1,
    num_layers=8,
    max_seq_length=MAX_SEQ_LENGTH,
    steps_per_report=10,
    steps_per_eval=100,
    save_every=None,
    resume_adapter_file=None,
    learning_rate=COMPLETE_LR,
    grad_accumulation=1,
    seed=0,
    val_batches=25,
    config_path=None,
    run_test=False,
):
    cmd = [
        sys.executable,
        "-u",
        LORA_ENTRY,
        "--model",
        model,
        "--train",
        "--data",
        data,
        "--adapter-path",
        adapter,
        "--iters",
        str(iters),
        "--batch-size",
        str(batch_size),
        "--learning-rate",
        str(learning_rate),
        "--max-seq-length",
        str(max_seq_length),
        "--num-layers",
        str(num_layers),
        "--mask-prompt",
        "--steps-per-report",
        str(steps_per_report),
        "--steps-per-eval",
        str(steps_per_eval),
        "--val-batches",
        str(val_batches),
        "--grad-accumulation-steps",
        str(max(1, int(grad_accumulation))),
        "--save-every",
        str(save_every if save_every is not None else save_every_for(iters)),
        "--seed",
        str(seed),
    ]
    if config_path:
        cmd.extend(["--config", config_path])
    if run_test:
        cmd.extend(["--test", "--test-batches", "-1"])
    if resume_adapter_file:
        cmd.extend(["--resume-adapter-file", resume_adapter_file])
    return cmd


def write_run_metadata(adapter, metadata):
    os.makedirs(adapter, exist_ok=True)
    with open(os.path.join(adapter, "twin_run.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def run_train(
    iters=10,
    model=MODEL,
    data=TWIN_DIR,
    adapter=ADAPTER,
    batch_size=1,
    num_layers=8,
    max_seq_length=MAX_SEQ_LENGTH,
    on_line=None,
    on_proc=None,
    save_every=None,
    resume_adapter_file=None,
    learning_rate=COMPLETE_LR,
    grad_accumulation=1,
    seed=0,
    val_batches=25,
    schedule=False,
    run_test=False,
    chat_template_args=None,
    multiturn_loss=False,
    lora_rank=None,
    lora_scale=None,
    lora_dropout=None,
):
    train_path = os.path.join(data, "train.jsonl")
    if not os.path.exists(train_path):
        raise FileNotFoundError("No training data. Run export first.")
    steps_per_report = max(1, iters // 120)
    steps_per_eval = max(1, iters // 12)
    os.makedirs(adapter, exist_ok=True)
    config_path = None
    if schedule:
        config_path = os.path.join(adapter, "twin_train.yaml")
        write_train_config(
            config_path, iters, learning_rate,
            lora_rank=lora_rank, lora_scale=lora_scale, lora_dropout=lora_dropout,
        )
    cmd = train_command(
        model,
        data,
        adapter,
        iters,
        batch_size=batch_size,
        num_layers=num_layers,
        max_seq_length=max_seq_length,
        steps_per_report=steps_per_report,
        steps_per_eval=steps_per_eval,
        save_every=save_every if save_every is not None else save_every_for(iters),
        resume_adapter_file=resume_adapter_file,
        learning_rate=learning_rate,
        grad_accumulation=grad_accumulation,
        seed=seed,
        val_batches=val_batches,
        config_path=config_path,
        run_test=run_test,
    )
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if chat_template_args:
        env["TWIN_CHAT_TEMPLATE_ARGS"] = json.dumps(chat_template_args)
    if multiturn_loss:
        env["TWIN_MULTITURN_LOSS"] = "1"
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        start_new_session=True,
    )
    if on_proc:
        on_proc(proc)
    tail = []
    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line.strip():
                tail.append(line)
                if len(tail) > 80:
                    del tail[: len(tail) - 80]
            if on_line:
                on_line(line)
            else:
                print(line, flush=True)
        code = proc.wait()
    finally:
        if on_proc:
            on_proc(None)
    if code != 0:
        hint = last_train_error(tail)
        raise RuntimeError(f"training exited {code}" + (f": {hint}" if hint else ""))


def main():
    parser = argparse.ArgumentParser(description="LoRA-tune a local texting twin")
    parser.add_argument("--model-key", choices=MODELS, default="compact")
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument(
        "--complete",
        action="store_true",
        help="Train for the default epoch count on train.jsonl",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--data", default=TWIN_DIR)
    parser.add_argument("--adapter")
    parser.add_argument(
        "--resume",
        help="Checkpoint id (model/run/step) or path to adapters.safetensors",
    )
    parser.add_argument(
        "--person",
        default="me",
        help="me, or a phone number / email / person id from the Twin page",
    )
    parser.add_argument(
        "--val-batches",
        type=int,
        default=None,
        help="Holdout batches for each in-training eval. -1 uses the whole file.",
    )
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="Skip the final test-set pass",
    )
    parser.add_argument(
        "--multiturn-loss",
        action="store_true",
        help="Put loss on every assistant turn in the window, not only the last",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-scale", type=float, default=None)
    parser.add_argument("--lora-dropout", type=float, default=None)
    parser.add_argument("--layers", type=int, default=None, help="LoRA layers to adapt")
    args = parser.parse_args()
    from twin.export import parse_person_arg

    try:
        person_id = parse_person_arg(args.person)
    except ValueError as e:
        sys.exit(str(e))
    config = model_config(args.model_key)
    batch_size = args.batch_size if args.batch_size is not None else config["batch_size"]
    if args.iters is not None:
        if args.iters < 1 or args.iters > MAX_ITERS:
            sys.exit(f"--iters must be between 1 and {MAX_ITERS:,}")
        iters = args.iters
    elif args.complete:
        train_path = os.path.join(args.data, "train.jsonl")
        try:
            with open(train_path, encoding="utf-8") as f:
                examples = sum(1 for line in f if line.strip())
        except OSError:
            sys.exit("No training data. Run: ./.venv/bin/python twin/export.py")
        epochs = args.epochs if args.epochs is not None else epochs_for(config, complete=True)
        iters = steps_for_examples(examples, batch_size, epochs=epochs)
    else:
        iters = 10

    resume_file = None
    if args.resume:
        if os.path.isfile(args.resume):
            resume_file = args.resume
        else:
            try:
                resume_file = resolve_checkpoint(args.resume, person_id)["weights"]
            except ValueError as e:
                sys.exit(str(e))

    created = time.time()
    adapter = args.adapter
    data_hash = ""
    run_id = ""
    if not adapter:
        try:
            data_hash = hash_train_file(args.data)
        except OSError:
            sys.exit("No training data. Run: ./.venv/bin/python twin/export.py")
        root = adapter_dir(args.model_key, person_id)
        os.makedirs(root, exist_ok=True)
        run_id = make_run_id(created, data_hash, iters, root)
        adapter = os.path.join(root, run_id)
    meta = {
        "model": args.model_key,
        "repo": config["repo"],
        "person": person_id,
        "run_id": run_id or os.path.basename(adapter),
        "created_at": created,
        "data_hash": data_hash,
        "iters": iters,
        "run": "complete" if args.complete else "custom",
        "resume_from": args.resume or "",
        "status": "running",
        "learning_rate": args.learning_rate or (COMPLETE_LR if args.complete else QUICK_LR),
        "epochs": args.epochs if args.epochs is not None else epochs_for(config, complete=args.complete),
        "layers": args.layers if args.layers is not None else config["layers"],
        "batch_size": batch_size,
        "seed": args.seed,
    }
    write_run_metadata(adapter, meta)

    try:
        run_train(
            iters=iters,
            model=config["repo"],
            data=args.data,
            adapter=adapter,
            batch_size=batch_size,
            num_layers=args.layers if args.layers is not None else config["layers"],
            resume_adapter_file=resume_file,
            learning_rate=meta["learning_rate"],
            grad_accumulation=grad_accumulation_for(config, batch_size) if args.complete else 1,
            schedule=bool(args.complete),
            run_test=bool(args.complete) and not args.no_test,
            val_batches=(
                args.val_batches
                if args.val_batches is not None
                else (-1 if args.complete else 25)
            ),
            seed=args.seed,
            chat_template_args=config.get("chat_template_args"),
            multiturn_loss=args.multiturn_loss,
        )
    except FileNotFoundError:
        sys.exit("No training data. Run: ./.venv/bin/python twin/export.py")
    except RuntimeError:
        meta["status"] = "error"
        write_run_metadata(adapter, meta)
        sys.exit(1)
    meta["status"] = "ready"
    write_run_metadata(adapter, meta)


if __name__ == "__main__":
    main()
