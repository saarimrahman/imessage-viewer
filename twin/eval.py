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
    TWIN_DIR,
    clip_bubbles,
    peer_from_system,
    strip_scaffold,
    split_bubbles,
    system_for,
)
from twin.retrieve import RETRIEVE_K, load_index, with_retrieved_shots
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
RANK_EXAMPLES = 100
RANK_BOOTSTRAP = 1000
RANK_OVERRIDE_P = 0.95
RANK_MIN_DELTA = 0.02


def _ngrams(text, n):
    chars = list((text or "").lower())
    if len(chars) < n:
        return Counter(["".join(chars)] if chars else [])
    return Counter("".join(chars[i : i + n]) for i in range(len(chars) - n + 1))


def chr_f(pred, gold, max_n=6, beta=1.0):
    """Character n-gram F-score. Higher is better.

    Beta is 1.0, so precision and recall weigh the same. The earlier 2.0
    weighted recall four times, which paid a model for writing long. Under
    that setting "repeat the incoming message twice" scored −0.037 against
    −0.011 for the untrained 4B, so a degenerate strategy tied a real model.
    """
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


COPY_MIN_CHARS = 20


def copied_from_prompt(pred, prompt):
    """True when the reply repeats a substantial span of the prompt.

    A bare substring test punished the style we are training for. Replies such
    as "ok", "kk" and "LMAO" appear inside any long multi-turn prompt by
    chance, and the flagged predictions had a median length of 4 characters.
    The real fault is parroting a whole incoming message, so the test needs a
    minimum length. Gold replies trip this at 0%, against 1% for the bare test.
    """
    pred = (pred or "").strip()
    return len(pred) >= COPY_MIN_CHARS and pred in (prompt or "")


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
        "copied_history": copied_from_prompt(pred, prompt),
        "exact": pred == gold,
        "hit_limit": bool(hit_limit),
        "text": pred,
        "pred": pred_stats,
        "gold": gold_stats,
    }


LENGTH_PENALTY = 0.5


def ranking_score(row):
    """Scalar used to pick a checkpoint. Higher is better.

    The length weight is 0.5. At the earlier 0.15 the penalty could not offset
    the recall bonus that a long reply collects, so padding was profitable.
    Checked against fixed strategies on 200 valid examples: gold +1.290,
    echo the incoming message −0.365, echo it twice −0.496, echo it four times
    −0.746, constant "kk" −1.241, empty −1.498.
    """
    length_pen = abs(math.log(max(0.05, min(20.0, row["len_ratio"]))))
    copied = 1.0 if row["copied_history"] else 0.0
    return row["chrf"] + 0.3 * row["rouge_l"] - LENGTH_PENALTY * length_pen - copied


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def gold_reply(messages):
    if messages and messages[-1].get("role") == "assistant":
        return messages[-1].get("content") or ""
    return ""


def prompt_messages(example, retrieve_index=None, retrieve_k=0):
    body = [dict(m) for m in example.get("messages") or []]
    if body and body[-1].get("role") == "assistant":
        body = body[:-1]
    system_text = body[0].get("content") or "" if body and body[0].get("role") == "system" else ""
    return with_retrieved_shots(
        body,
        retrieve_index or [],
        k=retrieve_k,
        peer=peer_from_system(system_text),
    )


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def bootstrap_ci(values, n_boot=RANK_BOOTSTRAP, alpha=0.05, rng=None):
    """Mean and percentile interval. Returns (mean, lo, hi)."""
    values = list(values)
    n = len(values)
    avg = mean(values)
    if n < 2:
        return avg, avg, avg
    rng = rng or random.Random(0)
    samples = []
    for _ in range(n_boot):
        total = 0.0
        for _i in range(n):
            total += values[rng.randrange(n)]
        samples.append(total / n)
    samples.sort()
    lo = samples[int((alpha / 2) * n_boot)]
    hi = samples[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return avg, lo, hi


def paired_win_rate(challenger, baseline, n_boot=RANK_BOOTSTRAP, rng=None):
    """Paired bootstrap P(mean(challenger) > mean(baseline))."""
    n = min(len(challenger), len(baseline))
    if n == 0:
        return 0.0
    rng = rng or random.Random(0)
    wins = 0
    ties = 0
    for _ in range(n_boot):
        total_a = 0.0
        total_b = 0.0
        for _i in range(n):
            j = rng.randrange(n)
            total_a += challenger[j]
            total_b += baseline[j]
        if total_a > total_b:
            wins += 1
        elif total_a == total_b:
            ties += 1
    return (wins + 0.5 * ties) / n_boot


def pick_ranked_checkpoint(
    scores_by_label,
    nll_label="latest",
    min_p=RANK_OVERRIDE_P,
    min_delta=RANK_MIN_DELTA,
    rng=None,
):
    """Keep the NLL checkpoint unless another is significantly better on ranking."""
    if not scores_by_label:
        return nll_label
    if nll_label not in scores_by_label:
        return max(scores_by_label.items(), key=lambda item: mean(item[1]))[0]
    baseline = scores_by_label[nll_label]
    base_mean = mean(baseline)
    rng = rng or random.Random(0)
    winners = []
    for label, scores in scores_by_label.items():
        if label == nll_label:
            continue
        if mean(scores) < base_mean + min_delta:
            continue
        if paired_win_rate(scores, baseline, rng=rng) >= min_p:
            winners.append((mean(scores), label))
    if not winners:
        return nll_label
    winners.sort(key=lambda item: (-item[0], str(item[1])))
    return winners[0][1]


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
    from twin.export import bubble_token_count

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
    pieces = bubble_token_count(tokenizer)
    if pieces is not None and pieces >= 5:
        raise RuntimeError(
            f"{BUBBLE} tokenizes into {pieces} pieces; pick a shorter delimiter"
        )
    return True


def generate_reply(model, tokenizer, messages, chat_template_args=None, max_tokens=96, temp=0.0, top_p=CHAT_TOP_P, seed=0, repetition_penalty=None):
    from mlx_lm import generate
    from twin.chat import generate_kwargs

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **(chat_template_args or {}),
    )
    kwargs = generate_kwargs(
        max_tokens, temp=temp, top_p=top_p, seed=seed,
        repetition_penalty=repetition_penalty,
    )
    raw = generate(model, tokenizer, prompt=prompt, **kwargs)
    text = (raw or "").strip()
    n_tok = 0
    if hasattr(tokenizer, "encode"):
        try:
            n_tok = len(tokenizer.encode(text))
        except Exception:
            n_tok = 0
    return clip_bubbles(strip_scaffold(text)), n_tok >= max(1, max_tokens) - 1


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
    repetition_penalty=None,
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
            repetition_penalty=repetition_penalty,
        )
        row = score_reply(pred, gold, prompt, hit_limit=hit_limit)
        system = (example.get("messages") or [{}])[0].get("content") or ""
        row["peer"] = peer_from_system(system)
        scored.append(row)
    return summarize(scored), scored


def _load_mlx(model_key, adapter_path=None, backend=None):
    """Load through `twin.backends`. The name is kept for its six call sites.

    Only loading and generation are backend-specific. Scoring always runs where
    the results log is, so every row shares one metric version.
    """
    from twin.backends import load

    return load(model_key, adapter_path, backend=backend)


def score_checkpoints(
    adapter_dir,
    data_dir,
    model_key,
    split="valid",
    max_examples=RANK_EXAMPLES,
    person_name="You",
    temp=0.0,
):
    """Score numbered checkpoints against the NLL-best latest; restore only a clear win."""
    del person_name
    path = os.path.join(data_dir, f"{split}.jsonl")
    if not os.path.isfile(path):
        return None
    examples = load_jsonl(path)
    if not examples:
        return None
    rng = random.Random(0)
    if max_examples and len(examples) > max_examples:
        examples = rng.sample(examples, max_examples)
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
    reports = []
    scores_by_label = {}
    for label, dest in unique:
        (model, tokenizer), config = _load_mlx(model_key, dest)
        summary, rows = evaluate_split(
            examples,
            model,
            tokenizer,
            chat_template_args=config.get("chat_template_args"),
            temp=temp,
        )
        ranks = [ranking_score(row) for row in rows]
        avg, lo, hi = bootstrap_ci(ranks, rng=random.Random(0))
        summary["checkpoint"] = label
        summary["ranking"] = avg
        summary["ranking_lo"] = lo
        summary["ranking_hi"] = hi
        reports.append(summary)
        scores_by_label[label] = ranks
    nll_label = "latest" if "latest" in scores_by_label else None
    if nll_label:
        nll_scores = scores_by_label[nll_label]
        for summary, label in ((row, row["checkpoint"]) for row in reports):
            if label == nll_label:
                summary["p_better_than_nll"] = 0.5
            else:
                summary["p_better_than_nll"] = paired_win_rate(
                    scores_by_label[label], nll_scores, rng=random.Random(0)
                )
    choice = pick_ranked_checkpoint(scores_by_label, nll_label=nll_label or "latest")
    if choice != "latest" and isinstance(choice, int):
        restore_best_checkpoint(adapter_dir, choice)
    return {
        "best": choice,
        "overrode_nll": bool(nll_label) and choice != nll_label,
        "n": len(examples),
        "reports": reports,
    }


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
    parser.add_argument("--max-examples", type=int, default=RANK_EXAMPLES)
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
