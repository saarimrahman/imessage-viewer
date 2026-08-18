#!/usr/bin/env python3
"""Run Twin training on Modal instead of this Mac.

CAUTION: this uploads exported message content. `twin/export.py` reads `chat.db`
and the address book locally, so those never leave, but the JSONL it writes is
the text of every exported message, from the account owner and from everyone who
texted them. Uploading it is a decision to make on purpose. Nothing here runs
until `upload-data` is called explicitly.

Why this exists. On the M1 Pro the 8B trains at 0.31 sequences each second,
against 4.00 for the 4B. A 2x parameter difference costs 13x, because 7.7 GB of
weights on a 16 GB shared-memory machine is bandwidth-starved. An A100 should
reach 3 to 6 sequences each second, so 7h50m becomes 35 to 45 minutes.

The larger win is width, not speed. This Mac fits one training job: two do not
(Finding 5). Training runs one container for each variant, and `score_all` uses
vLLM to hold several LoRA adapters against one loaded base, so a checkpoint
sweep pays the 8B load once instead of once for each step.

vLLM does not fine-tune. It is an inference engine and is used here for scoring
only. Training stays on PEFT.

CAUTION: none of this is verified. MLX does not run on NVIDIA, so this is a
port and not a move, and an adapter trained here **cannot** be scored by
`twin/bench.py`, which loads MLX. Treat any number it produces as a separate
track until the two are reconciled.

Usage:
    pip install modal && python3 -m modal setup
    modal run twin/modal_app.py::upload_data --data-dir .cache/twin-rich5
    modal run twin/modal_app.py::train --iters 2400
    modal run twin/modal_app.py::score_all --steps 400,800,1200,1600,2000,2400
"""

import os

import modal

# Qwen 3 on NVIDIA, not the `mlx-community` 4-bit conversions, which are Apple
# only. Quantization here comes from bitsandbytes instead.
MODELS = {
    "4b": "Qwen/Qwen3-4B-Instruct-2507",
    "8b": "Qwen/Qwen3-8B",
}
BUBBLE = "※"
# `<|im_start|>assistant\n` and `<|im_end|>` for the Qwen tokenizer, the same
# ids `twin/mlx_lora.py` uses.
ASSISTANT_HEADER = (151644, 77091, 198)
IM_END = 151645

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Qwen3 needs transformers 4.51 or newer.
        "torch",
        "transformers>=4.51",
        "peft>=0.14",
        "accelerate>=1.4",
        "bitsandbytes>=0.45",
        "datasets",
        "sentencepiece",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "0", "TOKENIZERS_PARALLELISM": "false"})
)

app = modal.App("twin")
# Data and adapters persist between runs. Model weights get their own cache so
# an 8B download is paid once rather than on every container.
volume = modal.Volume.from_name("twin-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("twin-hf-cache", create_if_missing=True)
VOL = "/vol"
HF = "/hf"


def assistant_mask(input_ids):
    """Mark every token the twin itself must produce.

    The port of `assistant_spans` in `twin/mlx_lora.py`. Training on the last
    assistant turn only supervised 9.0% of each sequence, and every turn
    supervises 33.3%. The closing `<|im_end|>` stays inside its turn so the
    model still learns to stop.
    """
    import torch

    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    header = torch.tensor(ASSISTANT_HEADER, device=input_ids.device)
    for row in range(input_ids.shape[0]):
        ids = input_ids[row]
        i = 0
        while i + 2 < ids.shape[0]:
            if torch.equal(ids[i : i + 3], header):
                j = i + 3
                while j < ids.shape[0] and ids[j] != IM_END:
                    mask[row, j] = True
                    j += 1
                if j < ids.shape[0]:
                    mask[row, j] = True
                i = j + 1
            else:
                i += 1
    return mask


@app.function(image=image, volumes={VOL: volume}, timeout=60 * 60)
def upload_data(payload: dict):
    """Write the exported JSONL into the volume. Called by the local entrypoint."""
    target = os.path.join(VOL, "data", payload["name"])
    os.makedirs(target, exist_ok=True)
    for filename, content in payload["files"].items():
        with open(os.path.join(target, filename), "w", encoding="utf-8") as f:
            f.write(content)
    volume.commit()
    return {name: len(text.splitlines()) for name, text in payload["files"].items()}


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={VOL: volume, HF: hf_cache},
    timeout=60 * 60 * 4,
)
def train(
    model_key: str = "8b",
    data_name: str = "twin-rich5",
    iters: int = 2400,
    learning_rate: float = 3e-5,
    layers: int = 8,
    batch_size: int = 8,
    max_seq_length: int = 768,
    save_every: int = 200,
    multiturn_loss: bool = True,
):
    """One LoRA run, same recipe as the local trainer."""
    import json

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    os.environ["HF_HOME"] = HF
    repo = MODELS[model_key]
    data_dir = os.path.join(VOL, "data", data_name)

    tokenizer = AutoTokenizer.from_pretrained(repo)
    # The same template fix as `sanitize_chat_template`: without it the target
    # opens with an empty think block and the adapter emits `<tool_call>`.
    from twin_template import sanitize  # packaged below

    sanitize(tokenizer)

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
    model.gradient_checkpointing_enable()

    # Match the local recipe: adapters on the last N blocks only.
    total = model.config.num_hidden_layers
    targets = [
        f"model.layers.{i}.{proj}"
        for i in range(total - layers, total)
        for proj in ("self_attn.q_proj", "self_attn.v_proj")
    ]
    model = get_peft_model(
        model,
        LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=targets),
    )
    model.print_trainable_parameters()

    rows = [json.loads(line) for line in open(os.path.join(data_dir, "train.jsonl"))]
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=learning_rate
    )
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate, total_steps=iters, pct_start=0.1
    )

    out_dir = os.path.join(VOL, "adapters", f"{model_key}-{data_name}-{iters}")
    os.makedirs(out_dir, exist_ok=True)

    model.train()
    cursor = 0
    for step in range(1, iters + 1):
        batch = []
        for _ in range(batch_size):
            batch.append(rows[cursor % len(rows)])
            cursor += 1
        texts = [
            tokenizer.apply_chat_template(r["messages"], tokenize=False) for r in batch
        ]
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
        ).to("cuda:0")

        labels = enc.input_ids.clone()
        if multiturn_loss:
            keep = assistant_mask(enc.input_ids) & enc.attention_mask.bool()
        else:
            keep = enc.attention_mask.bool()
        labels[~keep] = -100

        loss = model(**enc, labels=labels).loss
        loss.backward()
        optimizer.step()
        schedule.step()
        optimizer.zero_grad(set_to_none=True)

        if step % 20 == 0:
            print(f"Iter {step}: Train loss {loss.item():.3f}", flush=True)
        if step % save_every == 0 or step == iters:
            model.save_pretrained(os.path.join(out_dir, f"{step:07d}"))
            volume.commit()
    return out_dir


vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.6.6", "huggingface_hub==0.26.2")
    .env({"VLLM_ALLOW_RUNTIME_LORA_UPDATING": "1"})
)


@app.function(
    image=vllm_image,
    gpu="A100-40GB",
    volumes={VOL: volume, HF: hf_cache},
    timeout=60 * 60 * 2,
)
def score_all(adapter_dir: str, model_key: str, data_name: str, steps: str,
              temp: float = 0.7, max_examples: int = 200):
    """Score every checkpoint from one loaded base model.

    vLLM does not train. It serves, and it serves several LoRA adapters against
    one base, which is what a checkpoint sweep needs. The earlier design gave
    each checkpoint its own container and paid the 8B load every time, then
    generated one prompt at a time. Here the base loads once and all 200 prompts
    for all checkpoints run through continuous batching.
    """
    import json
    import os

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    os.environ["HF_HOME"] = HF
    wanted = [int(s) for s in steps.split(",") if s.strip()]
    paths = [(n, os.path.join(adapter_dir, f"{n:07d}")) for n in wanted]
    paths = [(n, p) for n, p in paths if os.path.isdir(p)]
    if not paths:
        raise FileNotFoundError(f"No checkpoints among {wanted} in {adapter_dir}")

    llm = LLM(
        model=MODELS[model_key],
        enable_lora=True,
        max_lora_rank=8,
        max_loras=2,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        gpu_memory_utilization=0.90,
        max_model_len=1024,
    )
    tokenizer = llm.get_tokenizer()
    from twin_template import sanitize

    sanitize(tokenizer)

    rows = [
        json.loads(line)
        for line in open(os.path.join(VOL, "data", data_name, "valid.jsonl"))
    ][:max_examples]
    prompts = [
        tokenizer.apply_chat_template(
            r["messages"][:-1], tokenize=False, add_generation_prompt=True
        )
        for r in rows
    ]
    sampling = SamplingParams(
        temperature=temp, top_p=0.9, max_tokens=96, repetition_penalty=1.15
    )

    results = []
    for step, path in paths:
        # One call for all 200 prompts. vLLM batches them internally.
        outs = llm.generate(
            prompts, sampling, lora_request=LoRARequest(str(step), step, path)
        )
        results.append(
            {
                "step": step,
                "predictions": [
                    {
                        "gold": r["messages"][-1]["content"],
                        "pred": o.outputs[0].text.strip(),
                    }
                    for r, o in zip(rows, outs)
                ],
            }
        )
        print(f"scored step {step}", flush=True)

    payload = os.path.join(VOL, "results", f"{os.path.basename(adapter_dir)}.json")
    os.makedirs(os.path.dirname(payload), exist_ok=True)
    with open(payload, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    volume.commit()
    return payload


@app.local_entrypoint()
def main(data_dir: str = ".cache/twin-rich5", model_key: str = "8b", iters: int = 2400):
    """Upload, train, then sweep. Prints what it is about to send first."""
    import pathlib

    source = pathlib.Path(data_dir)
    files = {}
    for name in ("train.jsonl", "valid.jsonl", "multiturn.jsonl"):
        path = source / name
        if path.is_file():
            files[name] = path.read_text(encoding="utf-8")
    total = sum(len(text.splitlines()) for text in files.values())
    size = sum(len(text.encode()) for text in files.values()) / 1e6
    print(f"about to upload {total:,} rows, {size:.1f} MB of message text from {source}")

    counts = upload_data.remote({"name": source.name, "files": files})
    print("uploaded:", counts)

    adapter_dir = train.remote(model_key=model_key, data_name=source.name, iters=iters)
    print("adapter:", adapter_dir)

    every = ",".join(str(s) for s in range(200, iters + 1, 200))
    print("results:", score_all.remote(adapter_dir, model_key, source.name, every))
