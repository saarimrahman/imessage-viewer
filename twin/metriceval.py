#!/usr/bin/env python3
"""Test whether a metric can tell a good reply from a bad one.

Every metric in this project was adopted without this check, and four of them
turned out to point the wrong way. `relevance` ranks my own replies below both
models. `style_distance` compares a set of replies against a set of mine, so it
cannot score one reply at all, which is why it never saw coherence.

This scores known-good and known-bad replies in the same context and reports
how often each metric prefers the good one. A metric under about 0.6 is a coin
flip and must not be used to choose a checkpoint.

Probe classes, all in the same context:

- ``gold``     my real reply. The good one.
- ``swapped``  my real reply from a different context. Fluent, in my voice, and
               the wrong answer. This is the discriminating case, because it
               isolates whether a metric reads the context at all.
- ``spam``     degenerate repetition.
- ``generic``  assistant boilerplate.
"""

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from twin.bench import last_incoming
from twin.coherence import reply_nll
from twin.eval import _load_mlx, chr_f, load_jsonl
from twin.export import TWIN_DIR
from twin.style import is_degenerate
from twin.train import MODELS

LOG_PATH = os.path.join(TWIN_DIR, "experiments", "metriceval.jsonl")

SPAM = [
    "Okay okay, " + "okay " * 60,
    "hehe" * 40,
    "Ohhh " + "thank you " * 40,
    "lol " * 50,
    "yeahhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh",
]
GENERIC = [
    "That sounds great! Let me know if there is anything else I can help you with.",
    "I understand how you feel. It is important to take care of yourself.",
    "Thanks for sharing that with me. I appreciate you letting me know.",
    "That is a really interesting point. I would love to hear more about it.",
    "Sure, I can help with that. What would you like to know?",
]


def build_probes(examples, rng):
    """One context with four candidate replies, one good and three bad."""
    probes = []
    for i, example in enumerate(examples):
        context = example["messages"][:-1]
        gold = example["messages"][-1]["content"]
        other = examples[(i + len(examples) // 2) % len(examples)]
        swapped = other["messages"][-1]["content"]
        if not gold or not swapped or swapped == gold:
            continue
        probes.append(
            {
                "context": context,
                # The same system turn with no conversation, so the only
                # difference is whether the model can see what was said.
                "blind_context": [context[0]] if context and context[0]["role"] == "system" else [],
                "incoming": last_incoming(example),
                "gold": gold,
                "candidates": {
                    "gold": gold,
                    "swapped": swapped,
                    "spam": rng.choice(SPAM),
                    "generic": rng.choice(GENERIC),
                },
            }
        )
    return probes


def score_candidate(text, probe, model, tokenizer, template_args, gold_nll):
    """Every candidate metric. Lower is better for each, by convention."""
    nll = reply_nll(model, tokenizer, probe["context"], text, template_args)
    scores = {
        # Reference-based. Needs the true reply, so it cannot judge a model in
        # the wild, but it is the incumbent and belongs in the comparison.
        "chrf_to_gold": -chr_f(text, probe["gold"]),
        # Reference-free.
        "echo_incoming": chr_f(text, probe["incoming"]),
        "degenerate": 1.0 if is_degenerate(text) else 0.0,
        "length_gap": abs(math.log(max(1, len(text)) / max(1, len(probe["gold"])))),
    }
    if nll is not None:
        # Conditioning gain: how much cheaper the reply becomes once the model
        # can see the real context. A reply that answers this conversation
        # should gain; one written for another conversation should not. This is
        # the only candidate here that can separate a right answer from a
        # fluent wrong one, which `style_distance` provably cannot (Finding 21).
        blind = reply_nll(model, tokenizer, probe["blind_context"], text, template_args)
        if blind is not None:
            scores["context_gain"] = nll - blind
        scores["nll"] = nll
        # Two-sided. Boilerplate is far too predictable and nonsense is far too
        # surprising, so distance from the human band is the quantity, not the
        # raw value.
        scores["nll_band"] = abs(nll - gold_nll)
    return scores


def main():
    parser = argparse.ArgumentParser(description="Validate a metric against known-bad replies")
    parser.add_argument("--label", required=True)
    parser.add_argument("--model-key", choices=MODELS, default="qwen3-capable")
    parser.add_argument("--data", default=TWIN_DIR)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--max-examples", type=int, default=60)
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    examples = load_jsonl(os.path.join(args.data, f"{args.split}.jsonl"))
    rng = random.Random(0)
    rng.shuffle(examples)
    examples = examples[: args.max_examples]
    probes = build_probes(examples, rng)

    (model, tokenizer), config = _load_mlx(args.model_key, None)
    template_args = config.get("chat_template_args")

    started = time.time()
    # Calibrate the band on the gold replies themselves.
    gold_nlls = [
        reply_nll(model, tokenizer, p["context"], p["gold"], template_args) for p in probes
    ]
    gold_nlls = [v for v in gold_nlls if v is not None]
    gold_nll = statistics.median(gold_nlls) if gold_nlls else 6.0

    rows = []
    for probe in probes:
        entry = {}
        for name, text in probe["candidates"].items():
            entry[name] = score_candidate(
                text, probe, model, tokenizer, template_args, gold_nll
            )
        rows.append(entry)

    metrics = sorted({k for row in rows for v in row.values() for k in v})
    report = {}
    for metric in metrics:
        report[metric] = {}
        for bad in ("swapped", "spam", "generic"):
            wins = total = 0
            for row in rows:
                good_score = row.get("gold", {}).get(metric)
                bad_score = row.get(bad, {}).get(metric)
                if good_score is None or bad_score is None:
                    continue
                total += 1
                wins += 1 if good_score < bad_score else 0
            report[metric][bad] = round(wins / total, 3) if total else None
        vals = [v for v in report[metric].values() if v is not None]
        report[metric]["mean"] = round(sum(vals) / len(vals), 3) if vals else None

    summary = {
        "label": args.label,
        "n_probes": len(probes),
        "gold_nll_median": round(gold_nll, 3),
        "accuracy": report,
        "seconds": round(time.time() - started, 1),
        "note": args.note,
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"probes: {len(probes)}   gold NLL median: {gold_nll:.3f}\n")
    print(f"{'metric':<16}{'vs swapped':>12}{'vs spam':>10}{'vs generic':>12}{'mean':>8}")
    for metric in sorted(report, key=lambda m: -(report[m]["mean"] or 0)):
        r = report[metric]
        fmt = lambda v: f"{v:.3f}" if v is not None else "  -  "
        print(
            f"{metric:<16}{fmt(r['swapped']):>12}{fmt(r['spam']):>10}"
            f"{fmt(r['generic']):>12}{fmt(r['mean']):>8}"
        )
    print("\n1.000 always prefers my real reply. 0.500 is a coin flip.")


if __name__ == "__main__":
    main()
