#!/usr/bin/env python3
"""Run one Twin holdout benchmark and append the result to a log.

Every run writes one JSON line to ``.cache/twin/experiments/results.jsonl``
with the config, the dataset hash, the metrics, and sample generations.
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from twin.eval import (
    _load_mlx,
    bootstrap_ci,
    chr_f,
    evaluate_split,
    load_jsonl,
    ranking_score,
)
from twin.export import CHAT_TOP_P, TWIN_DIR
from twin.retrieve import load_index
from twin.style import style_report
from twin.train import MODELS, load_dir_for

LOG_DIR = os.path.join(TWIN_DIR, "experiments")
LOG_PATH = os.path.join(LOG_DIR, "results.jsonl")


def last_incoming(example):
    for message in reversed(example["messages"][:-1]):
        if message.get("role") == "user":
            return message.get("content") or ""
    return ""


def sanity_stats(rows, examples):
    """Catch a model that scores well by saying the same safe thing every time.

    ``top1_share`` is modal collapse. ``distinct_2`` is word-bigram variety over
    all replies. ``relevance`` compares each reply against its own incoming
    message and against a shifted one, so a reply that ignores the question
    scores near zero.
    """
    texts = [row["text"] for row in rows]
    counts = Counter(texts)
    bigrams = set()
    total_bigrams = 0
    for text in texts:
        words = text.split()
        for i in range(len(words) - 1):
            bigrams.add((words[i], words[i + 1]))
            total_bigrams += 1
    incoming = [last_incoming(ex) for ex in examples[: len(rows)]]
    matched = [chr_f(t, i) for t, i in zip(texts, incoming)]
    shifted = [chr_f(t, incoming[(j + 1) % len(incoming)]) for j, t in enumerate(texts)]
    return {
        "top1_share": counts.most_common(1)[0][1] / max(1, len(texts)),
        "top1_text": counts.most_common(1)[0][0][:40],
        "unique_share": len(counts) / max(1, len(texts)),
        "distinct_2": len(bigrams) / max(1, total_bigrams),
        "relevance": (sum(matched) - sum(shifted)) / max(1, len(texts)),
    }


def promote_best_step(adapter, data_dir):
    """Record the best-scoring numbered checkpoint for this adapter.

    The chat page opens on `recommended_step` when the run metadata carries one,
    because the last checkpoint is the worst one measured. Only runs scored on
    the same dataset and the same decoding are compared, so a sweep at one
    temperature cannot be beaten by a single run at another.
    """
    want = data_hash(data_dir)
    try:
        with open(os.path.join(adapter, "twin_run.json"), encoding="utf-8") as f:
            meta_iters = json.load(f).get("iters") or 0
    except (OSError, json.JSONDecodeError):
        meta_iters = 0
    # `ls -dt` yields a trailing slash and a direct call does not, so the same
    # adapter appears under two spellings in the log.
    adapter = os.path.normpath(adapter)
    sweeps = {}
    with open(LOG_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if os.path.normpath(row.get("adapter") or "") != adapter:
                continue
            step = row.get("step")
            if not step:
                # A run with no `--step` scored the final weights. Record it
                # under its iteration count so it can win like any other.
                step = meta_iters
                if not step:
                    continue
            if row.get("data_hash") != want or row.get("split") != "valid":
                continue
            # Re-score from the stored predictions. A `style_distance` written
            # into the log came from whatever version of `style.py` existed
            # then, and comparing across versions is meaningless: the 8B was
            # logged at 1.175 before `caps_gap` existed and is really 1.705.
            stored = row.get("predictions") or []
            if not stored:
                continue
            # Initiative is the term that reversed the 4B/8B call, so the
            # promotion must see the incoming messages too (Finding 20).
            score = style_report(
                [x["pred"] for x in stored],
                [x["gold"] for x in stored],
                [x.get("incoming", "") for x in stored],
            )["style_distance"]
            key = (row.get("temp"), row.get("repetition_penalty"), row.get("retrieve_k"))
            sweeps.setdefault(key, {})[int(step)] = score
    if not sweeps:
        return 0
    # Take the sweep holding the best score, not the widest one. Choosing by
    # width preferred a 5-point sweep at temp 0.5 over a better 4-point sweep
    # at 0.7, which is the recommended setting.
    # A single scored checkpoint still gets its score recorded, so runs stay
    # comparable. Only a real sweep may move `recommended_step`.
    usable = [group for group in sweeps.values() if len(group) >= 2]
    scored = (
        min(usable, key=lambda group: min(group.values()))
        if usable
        else min(sweeps.values(), key=lambda group: min(group.values()))
    )
    best = (None, None, min(scored, key=scored.get))
    meta_path = os.path.join(adapter, "twin_run.json")
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return 0
    score = round(scored[best[2]], 4)
    if meta.get("recommended_step") == best[2] and meta.get("recommended_score") == score:
        return best[2]
    meta["recommended_step"] = best[2]
    # The chat page opens on the best-scoring run, not the newest one. The 8B
    # finished last and lost to the 4B on every metric.
    meta["recommended_score"] = score
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return best[2]


def data_hash(data_dir):
    h = hashlib.sha256()
    for name in ("train.jsonl", "valid.jsonl", "test.jsonl"):
        path = os.path.join(data_dir, name)
        if os.path.isfile(path):
            with open(path, "rb") as f:
                h.update(f.read())
    return h.hexdigest()[:12]


def main():
    parser = argparse.ArgumentParser(description="Benchmark one Twin config")
    parser.add_argument("--label", required=True, help="Name for this run in the log")
    parser.add_argument("--model-key", choices=MODELS, default="qwen3-compact")
    parser.add_argument("--data", default=TWIN_DIR)
    parser.add_argument("--adapter", default="")
    parser.add_argument("--step", default="", help="Numbered checkpoint inside --adapter")
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument("--retrieve-k", type=int, default=0)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=CHAT_TOP_P)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    path = os.path.join(args.data, f"{args.split}.jsonl")
    examples = load_jsonl(path)
    # Shuffle once, then slice. A smaller n is then a prefix of a larger n, so
    # runs scored at different sizes stay comparable.
    random.Random(0).shuffle(examples)
    if args.max_examples:
        examples = examples[: args.max_examples]

    adapter = args.adapter or None
    if adapter and args.step:
        adapter = load_dir_for(adapter, args.step)
    (model, tokenizer), config = _load_mlx(args.model_key, adapter)
    index = load_index(args.data) if args.retrieve_k else None
    started = time.time()
    summary, rows = evaluate_split(
        examples,
        model,
        tokenizer,
        chat_template_args=config.get("chat_template_args"),
        retrieve_index=index,
        retrieve_k=args.retrieve_k,
        temp=args.temp,
        top_p=args.top_p,
        seed=args.seed,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
    )
    ranks = [ranking_score(row) for row in rows]
    avg, lo, hi = bootstrap_ci(ranks, rng=random.Random(0))
    summary["ranking"] = avg
    summary["ranking_lo"] = lo
    summary["ranking_hi"] = hi
    summary["placeholder_rate"] = sum(
        1.0 for row in rows if "[" in row["text"] and "]" in row["text"]
    ) / max(1, len(rows))
    summary["blank_rate"] = sum(1.0 for row in rows if not row["text"]) / max(1, len(rows))
    ratios = sorted(row["len_ratio"] for row in rows)
    # The mean length ratio hides a bimodal model. One run showed mean 1.59
    # while the median was 0.44 and half the replies were under half length.
    summary["len_ratio_median"] = ratios[len(ratios) // 2] if ratios else 0.0
    summary["too_short_rate"] = sum(1.0 for r in ratios if r < 0.5) / max(1, len(ratios))
    summary["too_long_rate"] = sum(1.0 for r in ratios if r > 2.0) / max(1, len(ratios))
    summary.update(sanity_stats(rows, examples))
    style = style_report(
        [row["text"] for row in rows],
        [ex["messages"][-1]["content"] for ex in examples[: len(rows)]],
        [last_incoming(ex) for ex in examples[: len(rows)]],
    )
    summary["style_distance"] = style.get("style_distance", 0.0)
    summary["question_rate"] = style.get("initiative", {}).get("question", 0.0)
    summary["ack_rate"] = style.get("initiative", {}).get("ack", 0.0)
    summary["gold_question_rate"] = style.get("gold_initiative", {}).get("question", 0.0)
    summary["vocab_overlap"] = style.get("vocab_overlap", 0.0)
    summary["median_chars"] = style.get("pred", {}).get("median_chars", 0)
    summary["gold_median_chars"] = style.get("gold", {}).get("median_chars", 0)
    summary["bubbles"] = style.get("pred", {}).get("bubbles", 0.0)
    summary["seconds"] = round(time.time() - started, 1)

    record = {
        "label": args.label,
        "model_key": args.model_key,
        "adapter": args.adapter,
        "step": args.step,
        "split": args.split,
        "n": len(examples),
        "retrieve_k": args.retrieve_k,
        "temp": args.temp,
        "top_p": args.top_p,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "repetition_penalty": args.repetition_penalty,
        "data_hash": data_hash(args.data),
        "note": args.note,
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": summary,
    }
    sample_rng = random.Random(7)
    picks = sample_rng.sample(range(len(rows)), min(args.samples, len(rows)))
    record["samples"] = [
        {"peer": rows[i].get("peer", ""), "gold": rows[i]["gold"] and None, "pred": rows[i]["text"]}
        for i in picks
    ]
    for slot, i in zip(record["samples"], picks):
        slot["gold"] = examples[i]["messages"][-1]["content"]
        slot["last_in"] = next(
            (m["content"] for m in reversed(examples[i]["messages"][:-1]) if m["role"] == "user"),
            "",
        )

    # Store every prediction so a later metric change can re-score offline
    # without paying for generation again.
    record["predictions"] = [
        {
            "gold": ex["messages"][-1]["content"],
            "pred": row["text"],
            "peer": row.get("peer", ""),
            # Stored so a later metric can score initiative offline.
            "incoming": last_incoming(ex),
        }
        for ex, row in zip(examples, rows)
    ]
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    if args.adapter and args.step:
        promoted = promote_best_step(args.adapter, args.data)
        if promoted:
            print(f"[bench] recommended_step={promoted} for {os.path.basename(args.adapter)}")
    print(json.dumps({"label": args.label, **summary}, ensure_ascii=False))
    for slot in record["samples"][:4]:
        print(f"  in : {slot['last_in'][:110]}")
        print(f"  me : {slot['gold'][:110]}")
        print(f"  bot: {slot['pred'][:110]}")
        print()


if __name__ == "__main__":
    main()
