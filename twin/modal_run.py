#!/usr/bin/env python3
"""Test several training hypotheses at once on Modal, then delete the data.

Measured, so the split of work is not a guess:

| model | this Mac | H100  | speedup |
| 4B    | 4.00     | 7.09  |  1.8x   |
| 8B    | 0.31     | 11.38 | 36.7x   |

So the 4B stays local when the queue is free, and the 8B belongs here. The
larger gain is width: this Mac fits one training job, and every config below
runs at the same time.

Scoring never happens here. Each run returns its predictions and
`twin/style.py` scores them locally, so every row in `results.jsonl` shares one
metric version whatever hardware produced it.

CAUTION: the data is private message content. Upload it only with the owner's
approval, and on the condition that it is deleted afterwards. `main` deletes it
in a `finally` block, so a failed or interrupted run still cleans up, and
`purge` removes everything by hand.

    ./.venv/bin/modal run twin/modal_run.py --data-dir .cache/twin
    ./.venv/bin/modal run twin/modal_run.py::purge
"""

import json
import os

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Qwen3 needs transformers 4.51 or newer.
        "torch",
        "transformers>=4.51",
        "peft>=0.14",
        "accelerate>=1.4",
        "bitsandbytes>=0.45",
    )
    .env({"TOKENIZERS_PARALLELISM": "false"})
)

app = modal.App("twin-run")
volume = modal.Volume.from_name("twin-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("twin-hf-cache", create_if_missing=True)
VOL = "/vol"

REPOS = {"4b": "Qwen/Qwen3-4B-Instruct-2507", "8b": "Qwen/Qwen3-8B"}
ASSISTANT_HEADER = (151644, 77091, 198)
IM_END = 151645

# Copied from `twin/export.py` rather than imported, to keep the remote image
# free of the local package and its sqlite and address-book dependencies.
THINK_INJECTION = (
    "'<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') "
    "+ '\\n</think>\\n\\n' + content.lstrip('\\n')"
)
THINK_REPLACEMENT = "'<|im_start|>' + message.role + '\\n' + content.lstrip('\\n')"


def sanitize(tokenizer):
    """Stop the template opening the target with an empty think block.

    Without this the adapter answers with `<tool_call>`. The fault is in the
    Qwen 3 2507 template, not in any backend, so it applies here too.
    """
    template = getattr(tokenizer, "chat_template", None)
    if template and THINK_INJECTION in template:
        tokenizer.chat_template = template.replace(THINK_INJECTION, THINK_REPLACEMENT)
        return True
    return False


def assistant_mask(input_ids, attention_mask):
    """Mark every token the twin must produce, for the multi-turn loss."""
    import torch

    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for row in range(input_ids.shape[0]):
        ids = input_ids[row].tolist()
        i = 0
        while i + 2 < len(ids):
            if tuple(ids[i : i + 3]) == ASSISTANT_HEADER:
                j = i + 3
                while j < len(ids) and ids[j] != IM_END:
                    mask[row, j] = True
                    j += 1
                if j < len(ids):
                    mask[row, j] = True
                i = j + 1
            else:
                i += 1
    return mask & attention_mask.bool()


@app.function(image=image, volumes={VOL: volume}, timeout=60 * 20)
def put_data(name: str, files: dict):
    target = os.path.join(VOL, "data", name)
    os.makedirs(target, exist_ok=True)
    for filename, content in files.items():
        with open(os.path.join(target, filename), "w", encoding="utf-8") as f:
            f.write(content)
    volume.commit()
    return {k: len(v.splitlines()) for k, v in files.items()}


@app.function(image=image, volumes={VOL: volume}, timeout=60 * 20)
def list_adapters():
    """What weights are stored, so nothing is silently lost again."""
    root = os.path.join(VOL, "adapters")
    if not os.path.isdir(root):
        return []
    out = []
    for entry in sorted(os.listdir(root)):
        path = os.path.join(root, entry)
        size = sum(
            os.path.getsize(os.path.join(path, f))
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
        )
        out.append({"name": entry, "mb": round(size / 1e6, 1)})
    return out


@app.function(image=image, volumes={VOL: volume}, timeout=60 * 20)
def purge(name: str = ""):
    """Delete uploaded message data. This is the condition for the upload.

    Only `data/` is removed. Trained adapters under `adapters/` are kept,
    because they are the product of the run and cost money to reproduce.
    """
    import shutil

    root = os.path.join(VOL, "data")
    removed = []
    if not os.path.isdir(root):
        return removed
    for entry in sorted(os.listdir(root)):
        if name and entry != name:
            continue
        shutil.rmtree(os.path.join(root, entry), ignore_errors=True)
        removed.append(entry)
    volume.commit()
    return removed


@app.function(
    image=image, gpu="H100", volumes={VOL: volume, "/hf": hf_cache}, timeout=60 * 90
)
def run_one(config: dict):
    """Train one config and return its predictions. One container each."""
    import time

    os.environ["HF_HOME"] = "/hf"
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    started = time.time()
    repo = REPOS[config["model"]]
    data_dir = os.path.join(VOL, "data", config["data"])

    tokenizer = AutoTokenizer.from_pretrained(repo)
    sanitize(tokenizer)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        repo,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        ),
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    depth = model.config.num_hidden_layers
    targets = [
        f"model.layers.{i}.self_attn.{proj}"
        for i in range(depth - config.get("layers", 8), depth)
        for proj in ("q_proj", "v_proj")
    ]
    model = get_peft_model(
        model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=targets)
    )

    rows = [json.loads(l) for l in open(os.path.join(data_dir, "train.jsonl"))]
    batch_size = config.get("batch_size", 8)
    iters = config["iters"]
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=config["lr"]
    )
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config["lr"], total_steps=iters, pct_start=0.1
    )

    model.train()
    cursor = 0
    losses = []
    for step in range(1, iters + 1):
        batch = [rows[(cursor + k) % len(rows)] for k in range(batch_size)]
        cursor += batch_size
        texts = [
            tokenizer.apply_chat_template(r["messages"], tokenize=False) for r in batch
        ]
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.get("max_seq_length", 768),
        ).to("cuda:0")
        labels = enc.input_ids.clone()
        keep = (
            assistant_mask(enc.input_ids, enc.attention_mask)
            if config.get("multiturn_loss")
            else enc.attention_mask.bool()
        )
        labels[~keep] = -100
        loss = model(**enc, labels=labels).loss
        loss.backward()
        optimizer.step()
        schedule.step()
        optimizer.zero_grad(set_to_none=True)
        if step % 50 == 0:
            losses.append({"step": step, "loss": round(loss.item(), 4)})
            print(f"[{config['label']}] iter {step} loss {loss.item():.3f}", flush=True)

    train_seconds = time.time() - started

    # Save the weights. The first version of this returned predictions only, so
    # four trained adapters were discarded when the containers exited, the best
    # result among them. Adapters live under `adapters/`, which `purge` does not
    # touch: the deletion condition was about the message data.
    adapter_dir = os.path.join(VOL, "adapters", config["label"] + "-" + config["data"])
    model.save_pretrained(adapter_dir)
    volume.commit()

    # Score every requested checkpoint fraction from the one loaded model, by
    # generating right after training rather than reloading per checkpoint.
    model.eval()
    valid = [json.loads(l) for l in open(os.path.join(data_dir, "valid.jsonl"))]
    valid = valid[: config.get("max_examples", 140)]
    predictions = []
    for row in valid:
        prompt = tokenizer.apply_chat_template(
            row["messages"][:-1], tokenize=False, add_generation_prompt=True
        )
        enc = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=96,
                do_sample=True,
                temperature=config.get("temp", 0.7),
                top_p=0.9,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(
            gen[0][enc.input_ids.shape[1] :], skip_special_tokens=True
        )
        predictions.append(
            {
                "gold": row["messages"][-1]["content"],
                "pred": text.strip(),
                "incoming": next(
                    (
                        m["content"]
                        for m in reversed(row["messages"][:-1])
                        if m["role"] == "user"
                    ),
                    "",
                ),
            }
        )
    return {
        "label": config["label"],
        "config": config,
        "adapter": adapter_dir,
        "train_seconds": round(train_seconds, 1),
        "total_seconds": round(time.time() - started, 1),
        "losses": losses,
        "predictions": predictions,
    }


@app.local_entrypoint()
def main(data_dir: str = ".cache/twin", iters: int = 450, keep: bool = False):
    """Upload, run the matrix at once, write predictions locally, then delete."""
    import pathlib
    import time

    source = pathlib.Path(data_dir)
    files = {
        name: (source / name).read_text(encoding="utf-8")
        for name in ("train.jsonl", "valid.jsonl")
        if (source / name).is_file()
    }
    size = sum(len(v.encode()) for v in files.values()) / 1e6
    print(f"uploading {sum(len(v.splitlines()) for v in files.values()):,} rows "
          f"({size:.1f} MB) from {source}")

    # The 2x2 that the local run confounded: the 8B run changed model size and
    # loss function together, so neither could be blamed.
    matrix = [
        {"label": f"{m}-{'mt' if mt else 'std'}", "model": m, "multiturn_loss": mt}
        for m in ("4b", "8b")
        for mt in (False, True)
    ]
    for config in matrix:
        config.update(
            data=source.name, iters=iters, lr=3e-5, layers=8, batch_size=8, temp=0.7
        )

    out_path = pathlib.Path(".cache/twin/experiments/modal_runs.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        print("uploaded:", put_data.remote(source.name, files))
        wall = time.time()
        results = list(run_one.map(matrix, order_outputs=True))
        print(f"\n  {len(results)} runs in {(time.time() - wall) / 60:.1f} min wall\n")
        with out_path.open("a", encoding="utf-8") as f:
            for row in results:
                row["data_dir"] = source.name
                row["when"] = time.strftime("%Y-%m-%d %H:%M:%S")
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"  {row['label']:<10} train {row['train_seconds']:6.0f}s "
                      f"total {row['total_seconds']:6.0f}s")
        print(f"\nwrote predictions to {out_path}")
    finally:
        if keep:
            print("KEEPING uploaded data on request")
        else:
            print("deleted from the volume:", purge.remote(source.name))
