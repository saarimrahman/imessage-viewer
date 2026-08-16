#!/usr/bin/env python3
"""Train model-specific QLoRA adapters with MLX LM."""

import argparse
import json
import math
import os
import subprocess
import sys
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
        "description": "A top Tiny benchmark pick and the best default for quality, speed, and training headroom.",
        "publisher": "Alibaba",
        "category": "Featured benchmark picks",
        "batch_size": 1,
        "layers": 8,
        "recommended": True,
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


def model_config(model_key):
    try:
        return MODELS[model_key]
    except KeyError as e:
        raise ValueError(f"Unknown model: {model_key}") from e


def adapter_dir(model_key):
    return os.path.join(TWIN_DIR, "adapters", model_key)


# Existing callers/tests imported ADAPTER before model selection existed.
ADAPTER = adapter_dir("compact")


def has_adapter(model_key="compact", adapter=None):
    path = adapter or adapter_dir(model_key)
    current = os.path.isfile(os.path.join(path, "adapters.safetensors"))
    if adapter is not None:
        return current
    # Recognize the adapter produced by the first Twin implementation.
    if model_key == "compact" and not current:
        legacy = os.path.join(TWIN_DIR, "adapters", "adapters.safetensors")
        return os.path.isfile(legacy)
    return current


def resolved_adapter_dir(model_key):
    path = adapter_dir(model_key)
    if has_adapter(model_key, path):
        return path
    if model_key == "compact":
        legacy = os.path.join(TWIN_DIR, "adapters")
        if os.path.isfile(os.path.join(legacy, "adapters.safetensors")):
            return legacy
    return path


def steps_for_examples(examples, batch_size):
    """One complete pass through the shuffled training rows."""
    return max(1, math.ceil(examples / max(1, batch_size)))


def train_command(
    model,
    data,
    adapter,
    iters,
    batch_size=1,
    num_layers=8,
    max_seq_length=768,
    steps_per_report=10,
    steps_per_eval=100,
):
    return [
        sys.executable,
        "-u",
        "-m",
        "mlx_lm",
        "lora",
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
        "1e-4",
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
        "25",
        "--save-every",
        str(max(iters, 1)),
        "--seed",
        "0",
    ]


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
    max_seq_length=768,
    on_line=None,
    on_proc=None,
):
    train_path = os.path.join(data, "train.jsonl")
    if not os.path.exists(train_path):
        raise FileNotFoundError("No training data. Run export first.")
    steps_per_report = max(1, iters // 120)
    steps_per_eval = max(1, iters // 12)
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
    )
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
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
    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            if on_line:
                on_line(line)
            else:
                print(line, flush=True)
        code = proc.wait()
    finally:
        if on_proc:
            on_proc(None)
    if code != 0:
        raise RuntimeError(f"training exited {code}")


def main():
    parser = argparse.ArgumentParser(description="LoRA-tune a local texting twin")
    parser.add_argument("--model-key", choices=MODELS, default="compact")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--complete",
        action="store_true",
        help="Use enough steps for one pass through train.jsonl",
    )
    parser.add_argument("--data", default=TWIN_DIR)
    parser.add_argument("--adapter")
    args = parser.parse_args()
    config = model_config(args.model_key)
    iters = args.iters
    if args.complete:
        train_path = os.path.join(args.data, "train.jsonl")
        try:
            with open(train_path, encoding="utf-8") as f:
                examples = sum(1 for line in f if line.strip())
        except OSError:
            sys.exit("No training data. Run: ./.venv/bin/python twin/export.py")
        iters = steps_for_examples(examples, config["batch_size"])

    try:
        run_train(
            iters=iters,
            model=config["repo"],
            data=args.data,
            adapter=args.adapter or adapter_dir(args.model_key),
            batch_size=config["batch_size"],
            num_layers=config["layers"],
        )
    except FileNotFoundError:
        sys.exit("No training data. Run: ./.venv/bin/python twin/export.py")
    except RuntimeError:
        sys.exit(1)


if __name__ == "__main__":
    main()
