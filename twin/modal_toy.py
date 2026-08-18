#!/usr/bin/env python3
"""Measure Modal against this Mac on the one number that matters.

Locally, measured: the 8B trains at 0.31 sequences each second and the 4B at
4.00. This runs the same shape of work on an A100 and reports sequences each
second, so the comparison is like for like.

It uploads no message data. The batch is synthetic text of the same length as a
real training row, 768 tokens.

    ./.venv/bin/modal run twin/modal_toy.py
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Qwen3 needs transformers 4.51 or newer. 4.46 raises
        # "does not recognize this architecture".
        "torch",
        "transformers>=4.51",
        "peft>=0.14",
        "accelerate>=1.4",
        "bitsandbytes>=0.45",
    )
    .env({"TOKENIZERS_PARALLELISM": "false"})
)

app = modal.App("twin-toy")
hf_cache = modal.Volume.from_name("twin-hf-cache", create_if_missing=True)


@app.function(image=image, gpu="A100-40GB", volumes={"/hf": hf_cache}, timeout=60 * 45)
def bench(repo: str = "Qwen/Qwen3-8B", steps: int = 30, batch_size: int = 8,
          seq_len: int = 768, layers: int = 8):
    """Time a LoRA training loop and report sequences each second."""
    import os
    import time

    os.environ["HF_HOME"] = "/hf"

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    report = {"repo": repo, "batch_size": batch_size, "seq_len": seq_len}

    load_started = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        repo,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        ),
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    report["gpu"] = torch.cuda.get_device_name(0)
    report["load_seconds"] = round(time.time() - load_started, 1)

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    total = model.config.num_hidden_layers
    targets = [
        f"model.layers.{i}.self_attn.{proj}"
        for i in range(total - layers, total)
        for proj in ("q_proj", "v_proj")
    ]
    model = get_peft_model(
        model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=targets)
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    report["trainable_params"] = trainable

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=3e-5
    )
    vocab = model.config.vocab_size
    batch = torch.randint(0, vocab - 1, (batch_size, seq_len), device="cuda:0")
    labels = batch.clone()

    model.train()
    for _ in range(3):  # warm up, so compilation is not counted
        loss = model(input_ids=batch, labels=labels).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    started = time.time()
    for _ in range(steps):
        loss = model(input_ids=batch, labels=labels).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    elapsed = time.time() - started

    report["steps"] = steps
    report["seconds"] = round(elapsed, 1)
    report["seq_per_sec"] = round(steps * batch_size / elapsed, 2)
    report["tokens_per_sec"] = round(steps * batch_size * seq_len / elapsed, 0)
    report["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    return report


# Modal's published hourly rates, for cost per unit of work.
PRICE = {
    "T4": 0.59, "L4": 0.80, "A10": 1.10, "L40S": 1.95,
    "A100-40GB": 2.10, "A100-80GB": 2.50, "H100": 3.95,
    "H200": 4.54, "B200": 6.25,
}


@app.function(image=image, volumes={"/hf": hf_cache}, timeout=60 * 90)
def compare(repo: str, gpus: str, steps: int = 30, batch_size: int = 8):
    """Time the same loop on several GPUs at once, one container for each.

    This is the shape that matters. The Mac runs one job; here the whole
    comparison costs the wall clock of its slowest member.
    """
    jobs = [g.strip() for g in gpus.split(",") if g.strip()]
    handles = [
        bench.with_options(gpu=g).spawn(repo=repo, steps=steps, batch_size=batch_size)
        for g in jobs
    ]
    out = []
    for gpu, handle in zip(jobs, handles):
        try:
            row = handle.get()
            row["gpu_requested"] = gpu
            out.append(row)
        except Exception as e:  # one GPU failing must not lose the rest
            out.append({"gpu_requested": gpu, "error": str(e)[:200]})
    return out


@app.local_entrypoint()
def main(repo: str = "Qwen/Qwen3-8B", steps: int = 30):
    import time

    # Measured on this Mac: 8B 2,200 iters in 7h50m at grad accumulation 4,
    # 4B 2,400 iters in 80 min at grad accumulation 8.
    local = {"Qwen/Qwen3-8B": 0.31, "Qwen/Qwen3-4B-Instruct-2507": 4.00}

    wall = time.time()
    out = bench.remote(repo=repo, steps=steps)
    out["wall_seconds_including_cold_start"] = round(time.time() - wall, 1)

    print()
    for key, value in out.items():
        print(f"  {key:38} {value}")

    here = local.get(repo)
    if here:
        print()
        print(f"  local seq/s on the M1 Pro                {here}")
        print(f"  speedup                                  {out['seq_per_sec'] / here:.1f}x")
        for iters, accum, label in ((2200, 4, "8B 2,200 iters"), (2400, 8, "4B 2,400 iters")):
            seqs = iters * accum
            print(
                f"  {label:38} "
                f"{seqs / out['seq_per_sec'] / 60:6.1f} min there, "
                f"{seqs / here / 60:6.1f} min here"
            )


@app.local_entrypoint()
def gpus(repo: str = "Qwen/Qwen3-8B", gpus: str = "A100-40GB,A100-80GB,H100,L40S",
         steps: int = 30):
    """Throughput and cost for each GPU, measured rather than guessed."""
    rows = compare.remote(repo=repo, gpus=gpus, steps=steps)
    local = {"Qwen/Qwen3-8B": 0.31, "Qwen/Qwen3-4B-Instruct-2507": 4.00}.get(repo)

    print(f"\n  {repo}   local M1 Pro: {local} seq/s\n")
    print(f"  {'gpu':<14}{'seq/s':>8}{'$/hour':>9}{'$/1k seq':>10}{'peak GB':>9}{'vs local':>10}")
    ok = [r for r in rows if "seq_per_sec" in r]
    for r in sorted(ok, key=lambda r: -r["seq_per_sec"]):
        g = r["gpu_requested"]
        price = PRICE.get(g, 0.0)
        per_1k = 1000 / r["seq_per_sec"] / 3600 * price
        speed = r["seq_per_sec"] / local if local else 0
        print(f"  {g:<14}{r['seq_per_sec']:>8.2f}{price:>9.2f}{per_1k:>10.3f}"
              f"{r['peak_gb']:>9.1f}{speed:>9.1f}x")
    for r in rows:
        if "error" in r:
            print(f"  {r['gpu_requested']:<14} FAILED: {r['error'][:90]}")

    if ok:
        best = min(ok, key=lambda r: 1000 / r["seq_per_sec"] / 3600 * PRICE.get(r["gpu_requested"], 0))
        g = best["gpu_requested"]
        cost = 8800 / best["seq_per_sec"] / 3600 * PRICE.get(g, 0)
        mins = 8800 / best["seq_per_sec"] / 60
        print(f"\n  cheapest per unit of work: {g}")
        print(f"  an 8B run of 2,200 iters costs ${cost:.2f} and takes {mins:.0f} min there")
        if local:
            print(f"  the same run here takes {8800 / local / 60:.0f} min")
