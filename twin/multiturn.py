#!/usr/bin/env python3
"""Self-conditioned multi-turn replay for Twin adapters.

Single-reply evaluation always gives the model the real prior replies. This
script replaces them with the model's own replies, so drift, self-repetition,
and collapse over a conversation become visible.

For each holdout window with several of my turns, the script walks the turns in
order. At turn i, the context holds the real incoming messages and the model's
own generations for turns 1..i-1. Each generation is scored against the real
reply for that turn.
"""

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from twin.eval import (
    _load_mlx,
    chr_f,
    generate_reply,
    load_jsonl,
    ranking_score,
    score_reply,
)
from twin.export import CHAT_TOP_P, TWIN_DIR, peer_from_system
from twin.style import is_degenerate
from twin.train import MODELS, load_dir_for

LOG_DIR = os.path.join(TWIN_DIR, "experiments")
LOG_PATH = os.path.join(LOG_DIR, "multiturn.jsonl")
SELF_REPEAT_CHRF = 0.75


def assistant_positions(messages):
    return [i for i, m in enumerate(messages) if m.get("role") == "assistant"]


def replay(model, tokenizer, example, chat_template_args=None, temp=0.0,
           top_p=CHAT_TOP_P, seed=0, max_tokens=96, min_turns=3,
           repetition_penalty=None):
    """Generate every assistant turn in one window, conditioned on own output."""
    messages = [dict(m) for m in example.get("messages") or []]
    slots = assistant_positions(messages)
    if len(slots) < min_turns:
        return None
    working = [dict(m) for m in messages]
    turns = []
    for turn_index, pos in enumerate(slots):
        context = working[:pos]
        gold = messages[pos].get("content") or ""
        pred, hit_limit = generate_reply(
            model,
            tokenizer,
            context,
            chat_template_args=chat_template_args,
            max_tokens=max_tokens,
            temp=temp,
            top_p=top_p,
            seed=seed + turn_index,
            repetition_penalty=repetition_penalty,
        )
        prompt = " ".join(m.get("content") or "" for m in context)
        row = score_reply(pred, gold, prompt, hit_limit=hit_limit)
        row["turn"] = turn_index
        row["self_repeat"] = any(
            chr_f(pred, prior["text"]) >= SELF_REPEAT_CHRF and pred
            for prior in turns
        )
        turns.append(row)
        working[pos] = {"role": "assistant", "content": pred}
    return turns


def summarize_turns(conversations, depths=None):
    flat = [row for convo in conversations for row in convo]
    if not flat:
        return {}
    by_turn = {}
    for row in flat:
        by_turn.setdefault(min(row["turn"], 5), []).append(row)
    out = {
        "conversations": len(conversations),
        "turns": len(flat),
        "ranking": statistics.fmean(ranking_score(r) for r in flat),
        "chrf": statistics.fmean(r["chrf"] for r in flat),
        "len_ratio": statistics.fmean(r["len_ratio"] for r in flat),
        "self_repeat": statistics.fmean(1.0 if r["self_repeat"] else 0.0 for r in flat),
        "copied_history": statistics.fmean(1.0 if r["copied_history"] else 0.0 for r in flat),
        "eos_rate": statistics.fmean(0.0 if r.get("hit_limit") else 1.0 for r in flat),
        "blank_rate": statistics.fmean(1.0 if not r["text"] else 0.0 for r in flat),
    }
    out["by_turn"] = {
        str(k): {
            "n": len(v),
            "ranking": round(statistics.fmean(ranking_score(r) for r in v), 4),
            "chrf": round(statistics.fmean(r["chrf"] for r in v), 4),
            "len_ratio": round(statistics.fmean(r["len_ratio"] for r in v), 3),
            "self_repeat": round(statistics.fmean(1.0 if r["self_repeat"] else 0.0 for r in v), 3),
        }
        for k, v in sorted(by_turn.items())
    }
    if depths:
        # Report by window depth as well as by turn index. A model can hold two
        # turns and fall apart over six, and averaging over a mixed set of
        # window lengths hides that.
        buckets = {}
        for convo, depth in zip(conversations, depths):
            buckets.setdefault(depth, []).extend(convo)
        out["by_depth"] = {
            str(k): {
                "n": len(v),
                "windows": sum(1 for c, d in zip(conversations, depths) if d == k),
                "ranking": round(statistics.fmean(ranking_score(r) for r in v), 4),
                "self_repeat": round(
                    statistics.fmean(1.0 if r["self_repeat"] else 0.0 for r in v), 3
                ),
                "degenerate": round(
                    statistics.fmean(1.0 if is_degenerate(r["text"]) else 0.0 for r in v), 3
                ),
                "len_ratio": round(statistics.fmean(r["len_ratio"] for r in v), 3),
            }
            for k, v in sorted(buckets.items())
        }
    out["degenerate"] = statistics.fmean(
        1.0 if is_degenerate(r["text"]) else 0.0 for r in flat
    )
    early = [r for r in flat if r["turn"] <= 1]
    late = [r for r in flat if r["turn"] >= 3]
    if early and late:
        out["late_minus_early"] = statistics.fmean(
            ranking_score(r) for r in late
        ) - statistics.fmean(ranking_score(r) for r in early)
    return out


def main():
    parser = argparse.ArgumentParser(description="Multi-turn replay for a Twin adapter")
    parser.add_argument("--label", required=True)
    parser.add_argument("--model-key", choices=MODELS, default="qwen3-compact")
    parser.add_argument("--data", default=TWIN_DIR)
    parser.add_argument("--adapter", default="")
    parser.add_argument("--step", default="")
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--conversations", type=int, default=40)
    parser.add_argument("--min-turns", type=int, default=3)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=CHAT_TOP_P)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--show", type=int, default=2)
    parser.add_argument("--note", default="")
    parser.add_argument(
        "--split-only",
        action="store_true",
        help="Ignore multiturn.jsonl and pick windows from the split instead",
    )
    args = parser.parse_args()

    # Prefer the fixed depth-stratified holdout so every run scores the same
    # windows and a change at 6 turns is not hidden by a change at 2.
    fixed = os.path.join(args.data, "multiturn.jsonl")
    if os.path.isfile(fixed) and not args.split_only:
        eligible = load_jsonl(fixed)
    else:
        examples = load_jsonl(os.path.join(args.data, f"{args.split}.jsonl"))
        eligible = [
            ex for ex in examples
            if len(assistant_positions(ex.get("messages") or [])) >= args.min_turns
        ]
        random.Random(0).shuffle(eligible)
    if args.conversations:
        eligible = eligible[: args.conversations]

    adapter = args.adapter or None
    if adapter and args.step:
        adapter = load_dir_for(adapter, args.step)
    (model, tokenizer), config = _load_mlx(args.model_key, adapter)
    started = time.time()
    conversations = []
    transcripts = []
    depths = []
    for example in eligible:
        turns = replay(
            model,
            tokenizer,
            example,
            chat_template_args=config.get("chat_template_args"),
            temp=args.temp,
            top_p=args.top_p,
            seed=args.seed,
            max_tokens=args.max_tokens,
            min_turns=args.min_turns,
            repetition_penalty=args.repetition_penalty,
        )
        if not turns:
            continue
        conversations.append(turns)
        depths.append(
            example.get("depth")
            or len(assistant_positions(example.get("messages") or []))
        )
        transcripts.append(
            {
                "peer": peer_from_system(
                    (example["messages"][0].get("content") or "")
                    if example["messages"][0].get("role") == "system"
                    else ""
                ),
                "turns": [
                    {"gold": g, "pred": t["text"]}
                    for g, t in zip(
                        [
                            example["messages"][p]["content"]
                            for p in assistant_positions(example["messages"])
                        ],
                        turns,
                    )
                ],
            }
        )

    summary = summarize_turns(conversations, depths)
    summary["seconds"] = round(time.time() - started, 1)
    record = {
        "label": args.label,
        "model_key": args.model_key,
        "adapter": args.adapter,
        "step": args.step,
        "split": args.split,
        "temp": args.temp,
        "seed": args.seed,
        "min_turns": args.min_turns,
        "repetition_penalty": args.repetition_penalty,
        "note": args.note,
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": summary,
        "transcripts": transcripts[:6],
    }
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"label": args.label, **summary}, ensure_ascii=False))
    for convo in transcripts[: args.show]:
        print(f"--- {convo['peer']}")
        for i, turn in enumerate(convo["turns"]):
            print(f"  t{i} me : {turn['gold'][:100]}")
            print(f"  t{i} bot: {turn['pred'][:100]}")


if __name__ == "__main__":
    main()
