#!/usr/bin/env python3
"""Holdout evaluation for Twin adapters.

Scores replies with surface overlap, length, and style stats. Optionally
compares greedy decoding to modest sampling, and LoRA to LoRA plus retrieval.
Does not download extra metric packages.
"""

import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from twin.export import (
    BUBBLE,
    CHAT_TEMP,
    CHAT_TOP_P,
    CONTEXT_TURNS,
    TWIN_DIR,
    clip_bubbles,
    peer_from_system,
    split_bubbles,
    system_for,
)
from twin.retrieve import RETRIEVE_K, few_shot_messages, load_index, retrieve
from twin.train import (
    DEFAULT_MODEL,
    MODELS,
    checkpoint_steps,
    load_dir_for,
    model_config,
    restore_best_checkpoint,
)

EMOJI_RE = re.compile(
    "["
    "\U0001f300-\U0001faff"
    "\U00002700-\U000027bf"
    "\U0001f1e0-\U0001f1ff"
    "]+",
    flags=re.UNICODE,
)
SEARCH_GRID = {
    "learning_rate": [1e-5, 3e-5, 1e-4],
    "epochs": [1, 2, 3, 5],
    "effective_batch": [8],
    "layers": [8, 16],
    "seeds": [0, 1],
}


def _ngrams(text, n):
    chars = list((text or "").lower())
    if len(chars) < n:
        return Counter(["".join(chars)] if chars else [])
    return Counter("".join(chars[i : i + n]) for i in range(len(chars) - n + 1))


def chr_f(pred, gold, max_n=6, beta=2.0):
    """Character n-gram F-score. Higher is better."""
    pred = pred or ""
    gold = gold or ""
    if not pred or not gold:
        return 0.0
    prec = rec = 0.0
    used = 0
    for n in range(1, max_n + 1):
        p = _ngrams(pred, n)
        g = _ngrams(gold, n)
        if not p or not g:
            continue
        overlap = sum((p & g).values())
        prec += overlap / max(1, sum(p.values()))
        rec += overlap / max(1, sum(g.values()))
        used += 1
    if not used:
        return 0.0
    prec /= used
    rec /= used
    if prec == 0 or rec == 0:
        return 0.0
    beta2 = beta * beta
    return (1 + beta2) * prec * rec / (beta2 * prec + rec)


def _lcs(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for ch in a:
        cur = [0]
        for j, other in enumerate(b):
            cur.append(prev[j] + 1 if ch == other else max(cur[-1], prev[j + 1]))
        prev = cur
    return prev[-1]


def rouge_l(pred, gold):
    pred_t = (pred or "").split()
    gold_t = (gold or "").split()
    if not pred_t or not gold_t:
        return 0.0
    lcs = _lcs(pred_t, gold_t)
    prec = lcs / len(pred_t)
    rec = lcs / len(gold_t)
    if prec == 0 or rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def style_stats(text):
    text = text or ""
    letters = [ch for ch in text if ch.isalpha()]
    upper = sum(1 for ch in letters if ch.isupper())
    return {
        "chars": len(text),
        "words": len(text.split()),
        "bubbles": max(1, len(split_bubbles(text))),
        "emoji": len(EMOJI_RE.findall(text)),
        "upper_frac": (upper / len(letters)) if letters else 0.0,
        "ends_punct": bool(text) and text[-1] in ".!?",
    }


def score_reply(pred, gold, prompt="", hit_limit=False):
    pred = (pred or "").strip()
    gold = (gold or "").strip()
    gold_stats = style_stats(gold)
    pred_stats = style_stats(pred)
    gold_len = max(1, gold_stats["chars"])
    return {
        "chrf": chr_f(pred, gold),
        "rouge_l": rouge_l(pred, gold),
        "len_ratio": pred_stats["chars"] / gold_len,
        "bubble_delta": pred_stats["bubbles"] - gold_stats["bubbles"],
        "emoji_delta": pred_stats["emoji"] - gold_stats["emoji"],
        "copied_history": bool(pred) and pred in (prompt or ""),
        "exact": pred == gold,
        "hit_limit": bool(hit_limit),
        "text": pred,
        "pred": pred_stats,
        "gold": gold_stats,
    }


def ranking_score(row):
    """Scalar used to pick a checkpoint. Higher is better."""
    length_pen = abs(math.log(max(0.05, min(20.0, row["len_ratio"]))))
    copied = 1.0 if row["copied_history"] else 0.0
    return row["chrf"] + 0.3 * row["rouge_l"] - 0.15 * length_pen - copied


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def last_user(messages):
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content") or ""
    return ""


def gold_reply(messages):
    if messages and messages[-1].get("role") == "assistant":
        return messages[-1].get("content") or ""
    return ""


def prompt_messages(example, retrieve_index=None, retrieve_k=0):
    body = [dict(m) for m in example.get("messages") or []]
    if body and body[-1].get("role") == "assistant":
        body = body[:-1]
    if retrieve_index and retrieve_k:
        query = last_user(body)
        system_text = body[0].get("content") or "" if body and body[0].get("role") == "system" else ""
        shots = retrieve(
            query,
            retrieve_index,
            k=retrieve_k,
            exclude=[query],
            peer=peer_from_system(system_text),
        )
        if shots:
            system = body[:1] if body and body[0].get("role") == "system" else []
            rest = body[len(system) :]
            body = system + few_shot_messages(shots) + rest
    # Keep the live thread inside the shared context budget.
    system = body[:1] if body and body[0].get("role") == "system" else []
    rest = body[len(system) :]
    rest = rest[-CONTEXT_TURNS:]
    while rest and rest[0].get("role") != "user":
        rest = rest[1:]
    return system + rest


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def summarize(rows):
    if not rows:
        return {}
    out = {
        "n": len(rows),
        "chrf": mean(r["chrf"] for r in rows),
        "rouge_l": mean(r["rouge_l"] for r in rows),
        "ranking": mean(ranking_score(r) for r in rows),
        "exact": mean(1.0 if r["exact"] else 0.0 for r in rows),
        "copied_history": mean(1.0 if r["copied_history"] else 0.0 for r in rows),
        "len_ratio": mean(r["len_ratio"] for r in rows),
        "bubble_delta": mean(r["bubble_delta"] for r in rows),
        "eos_rate": mean(0.0 if r.get("hit_limit") else 1.0 for r in rows),
    }
    out.update(recipient_summary(rows))
    return out


def recipient_summary(rows):
    """How much reply quality and modal text vary by recipient."""
    by = {}
    for row in rows:
        peer = row.get("peer") or ""
        if not peer:
            continue
        by.setdefault(peer, []).append(row)
    if len(by) < 2:
        return {}
    means = {
        peer: mean(ranking_score(row) for row in group)
        for peer, group in by.items()
        if len(group) >= 2
    }
    if len(means) < 2:
        return {"peers": len(by)}
    modal = {}
    for peer, group in by.items():
        texts = [row.get("text") or "" for row in group]
        modal[peer] = Counter(texts).most_common(1)[0][0] if texts else ""
    global_mode = Counter(row.get("text") or "" for row in rows).most_common(1)
    mode_text = global_mode[0][0] if global_mode else ""
    collapsed = mean(1.0 if text == mode_text and mode_text else 0.0 for text in modal.values())
    return {
        "peers": len(by),
        "peer_score_range": max(means.values()) - min(means.values()),
        "peer_mode_collapse": collapsed,
    }


def smoke_template(tokenizer, chat_template_args=None, name="You"):
    """Render one legal Twin example. Raise if the chat template rejects it."""
    messages = [
        {"role": "system", "content": system_for(name)},
        {"role": "user", "content": "you free"},
        {"role": "assistant", "content": f"yeah{BUBBLE}give me 10"},
    ]
    kwargs = dict(chat_template_args or {})
    tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )
    tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
    return True


def generate_reply(model, tokenizer, messages, chat_template_args=None, max_tokens=96, temp=0.0, top_p=CHAT_TOP_P, seed=0):
    from mlx_lm import generate
    from twin.chat import generate_kwargs

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **(chat_template_args or {}),
    )
    kwargs = generate_kwargs(max_tokens, temp=temp, top_p=top_p, seed=seed)
    raw = generate(model, tokenizer, prompt=prompt, **kwargs)
    text = (raw or "").strip()
    n_tok = 0
    if hasattr(tokenizer, "encode"):
        try:
            n_tok = len(tokenizer.encode(text))
        except Exception:
            n_tok = 0
    return clip_bubbles(text), n_tok >= max(1, max_tokens) - 1


def evaluate_split(
    examples,
    model,
    tokenizer,
    chat_template_args=None,
    retrieve_index=None,
    retrieve_k=0,
    temp=0.0,
    top_p=CHAT_TOP_P,
    seed=0,
    max_examples=0,
    max_tokens=96,
):
    rows = examples[:max_examples] if max_examples else examples
    scored = []
    for i, example in enumerate(rows):
        messages = prompt_messages(example, retrieve_index, retrieve_k)
        gold = gold_reply(example.get("messages") or [])
        prompt = " ".join(m.get("content") or "" for m in messages)
        pred, hit_limit = generate_reply(
            model,
            tokenizer,
            messages,
            chat_template_args=chat_template_args,
            max_tokens=max_tokens,
            temp=temp,
            top_p=top_p,
            seed=seed + i,
        )
        row = score_reply(pred, gold, prompt, hit_limit=hit_limit)
        system = (example.get("messages") or [{}])[0].get("content") or ""
        row["peer"] = peer_from_system(system)
        scored.append(row)
    return summarize(scored), scored


def _load_mlx(model_key, adapter_path=None):
    from mlx_lm import load

    config = model_config(model_key)
    kwargs = {}
    if adapter_path:
        kwargs["adapter_path"] = adapter_path
    return load(config["repo"], **kwargs), config


def score_checkpoints(
    adapter_dir,
    data_dir,
    model_key,
    split="valid",
    max_examples=24,
    person_name="You",
):
    """Score the latest and numbered checkpoints; restore the generation winner."""
    del person_name
    path = os.path.join(data_dir, f"{split}.jsonl")
    if not os.path.isfile(path):
        return None
    examples = load_jsonl(path)
    if not examples:
        return None
    steps = checkpoint_steps(adapter_dir)
    latest = os.path.join(adapter_dir, "adapters.safetensors")
    candidates = []
    if os.path.isfile(latest):
        candidates.append(("latest", adapter_dir))
    if steps:
        pick = {steps[-1], steps[len(steps) // 2]}
        if len(steps) > 2:
            pick.add(steps[0])
        for step in sorted(pick):
            candidates.append((step, load_dir_for(adapter_dir, step)))
    seen = set()
    unique = []
    for label, dest in candidates:
        if dest in seen:
            continue
        seen.add(dest)
        unique.append((label, dest))
    best = None
    reports = []
    for label, dest in unique:
        (model, tokenizer), config = _load_mlx(model_key, dest)
        summary, _ = evaluate_split(
            examples,
            model,
            tokenizer,
            chat_template_args=config.get("chat_template_args"),
            max_examples=max_examples,
        )
        summary["checkpoint"] = label
        reports.append(summary)
        score = summary.get("ranking") or 0
        if best is None or score > best[0]:
            best = (score, label, dest)
    if best and best[1] != "latest" and isinstance(best[1], int):
        restore_best_checkpoint(adapter_dir, best[1])
    return {"best": best[1] if best else None, "reports": reports}


def print_grid():
    print("Holdout search grid (run configs separately; do not assume one winner):")
    for key, values in SEARCH_GRID.items():
        print(f"  {key}: {values}")
    print("Complete default: lr=1e-5, epochs=3, effective batch 8, layers=8, seed=0.")
    print("Select by twin/eval.py holdout ranking, not training loss.")


def main():
    parser = argparse.ArgumentParser(description="Score Twin holdout replies")
    parser.add_argument("--model-key", choices=MODELS, default=DEFAULT_MODEL)
    parser.add_argument("--data", default=TWIN_DIR)
    parser.add_argument("--adapter", help="Adapter directory to load")
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--max-examples", type=int, default=64)
    parser.add_argument("--retrieve", action="store_true")
    parser.add_argument("--base", action="store_true", help="Score the base model without an adapter")
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=CHAT_TOP_P)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid", action="store_true", help="Print the hyperparameter grid and exit")
    parser.add_argument("--smoke", action="store_true", help="Render one example with the tokenizer")
    parser.add_argument("--compare-decoding", action="store_true")
    parser.add_argument("--person-name", default="You")
    args = parser.parse_args()

    if args.grid:
        print_grid()
        return

    config = model_config(args.model_key)
    path = os.path.join(args.data, f"{args.split}.jsonl")
    if args.smoke:
        (model, tokenizer), _ = _load_mlx(args.model_key, None if args.base else args.adapter)
        del model
        smoke_template(tokenizer, config.get("chat_template_args"), args.person_name)
        print(f"{args.model_key} chat template accepted a Twin example.")
        return
    if not os.path.isfile(path):
        sys.exit(f"No {args.split}.jsonl in {args.data}. Export first.")
    examples = load_jsonl(path)
    rng = random.Random(args.seed)
    if args.max_examples and len(examples) > args.max_examples:
        examples = rng.sample(examples, args.max_examples)

    adapter = None if args.base else args.adapter
    (model, tokenizer), _ = _load_mlx(args.model_key, adapter)
    index = load_index(args.data) if args.retrieve else None
    temps = [0.0, CHAT_TEMP] if args.compare_decoding else [args.temp]
    retrieve_ks = [0, RETRIEVE_K] if args.retrieve else [0]
    for temp in temps:
        for k in retrieve_ks:
            summary, _ = evaluate_split(
                examples,
                model,
                tokenizer,
                chat_template_args=config.get("chat_template_args"),
                retrieve_index=index,
                retrieve_k=k,
                temp=temp,
                top_p=args.top_p,
                seed=args.seed,
            )
            tag = f"temp={temp} retrieve={k} {'base' if args.base else 'lora'}"
            print(json.dumps({"tag": tag, **summary}))


if __name__ == "__main__":
    main()
