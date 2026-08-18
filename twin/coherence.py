#!/usr/bin/env python3
"""Score whether a reply is a plausible continuation, not whether it sounds right.

Every other metric here scores voice. None of them notice that a fluent reply
can still be nonsense, and `relevance` actively points the wrong way: the gold
replies score lower on it than either model, because real texting answers a
message without reusing its words.

This scores the average negative log-likelihood of a reply under the
**untrained** base model, given the same context. The base model is a competent
language model, so an incoherent continuation costs it probability. The adapter
is never used for scoring, which keeps the judge independent of what is judged.

A raw NLL means nothing on its own. Read every number against the value for my
real replies on the same examples, which this always reports.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from twin.eval import _load_mlx, load_jsonl
from twin.export import TWIN_DIR
from twin.train import MODELS

LOG_PATH = os.path.join(TWIN_DIR, "experiments", "coherence.jsonl")


def reply_nll(model, tokenizer, messages, reply, chat_template_args=None):
    """Average NLL over the reply tokens, conditioned on the context."""
    import mlx.core as mx
    import mlx.nn as nn

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **(chat_template_args or {}),
    )
    prompt_ids = tokenizer.encode(prompt)
    reply_ids = tokenizer.encode(reply)
    if not reply_ids:
        return None
    ids = mx.array([prompt_ids + reply_ids])
    logits = model(ids[:, :-1]).astype(mx.float32)
    targets = ids[:, 1:]
    start = len(prompt_ids) - 1
    span_logits = logits[:, start:, :]
    span_targets = targets[:, start:]
    losses = nn.losses.cross_entropy(
        span_logits.reshape(-1, span_logits.shape[-1]), span_targets.reshape(-1)
    )
    return float(mx.mean(losses).item())


def main():
    parser = argparse.ArgumentParser(description="Score reply coherence under the base model")
    parser.add_argument("--label", required=True)
    parser.add_argument("--model-key", choices=MODELS, default="qwen3-capable")
    parser.add_argument("--data", default=TWIN_DIR)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument(
        "--from-run",
        default="",
        help="Label in results.jsonl whose stored predictions to score",
    )
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    examples = load_jsonl(os.path.join(args.data, f"{args.split}.jsonl"))
    random.Random(0).shuffle(examples)
    if args.max_examples:
        examples = examples[: args.max_examples]

    preds = None
    if args.from_run:
        results = os.path.join(TWIN_DIR, "experiments", "results.jsonl")
        with open(results, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("label") == args.from_run and row.get("predictions"):
                    preds = [p["pred"] for p in row["predictions"]]
        if preds is None:
            sys.exit(f"No stored predictions for label {args.from_run}")
        examples = examples[: len(preds)]

    # The judge is always the untrained model, never the adapter under test.
    (model, tokenizer), config = _load_mlx(args.model_key, None)
    template_args = config.get("chat_template_args")

    started = time.time()
    gold_scores, pred_scores = [], []
    for i, example in enumerate(examples):
        context = example["messages"][:-1]
        gold = example["messages"][-1]["content"]
        score = reply_nll(model, tokenizer, context, gold, template_args)
        if score is not None:
            gold_scores.append(score)
        if preds is not None:
            score = reply_nll(model, tokenizer, context, preds[i], template_args)
            if score is not None:
                pred_scores.append(score)

    def median(values):
        ordered = sorted(values)
        return ordered[len(ordered) // 2] if ordered else 0.0

    summary = {
        "label": args.label,
        "n": len(examples),
        "gold_nll_median": round(median(gold_scores), 4),
        "gold_nll_mean": round(sum(gold_scores) / max(1, len(gold_scores)), 4),
        "seconds": round(time.time() - started, 1),
        "note": args.note,
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if pred_scores:
        summary["pred_nll_median"] = round(median(pred_scores), 4)
        summary["pred_nll_mean"] = round(sum(pred_scores) / max(1, len(pred_scores)), 4)
        # Above 1.0 means the model's replies are less plausible continuations
        # than my real ones. This ratio is the number to read.
        summary["nll_ratio"] = round(
            summary["pred_nll_median"] / max(1e-6, summary["gold_nll_median"]), 3
        )
        summary["from_run"] = args.from_run

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
