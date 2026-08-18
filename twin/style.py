#!/usr/bin/env python3
"""Content-free style distance between a model's replies and mine.

The ranking score is dominated by character overlap with one gold reply, and
what a person says next is largely unpredictable. A real reply taken from
another conversation scores −0.495 on that scale, so overlap alone cannot say
whether a model matches the target voice.

This module compares the two sets of replies as distributions instead. It never
pairs a prediction with its own gold, so it measures voice, not content.
"""

import math
import re
import statistics
import zlib
from collections import Counter

from twin.eval import EMOJI_RE
from twin.export import split_bubbles

WORD_RE = re.compile(r"[a-z0-9']+", re.I)
VOCAB_TOP = 300


DEGEN_MIN_CHARS = 25
DEGEN_RATIO = 0.35


def is_degenerate(text):
    """True for `okay okay okay ...` and `hehehehehe...`.

    Degenerate text compresses almost to nothing. The reported spam samples
    compress to 0.05 and 0.07, while real replies sit near 0.93 and a short one
    near 1.89. Anything under 0.35 is spam, and no gold reply in the 200-row
    holdout falls below 0.575.
    """
    text = text or ""
    if len(text) < DEGEN_MIN_CHARS:
        return False
    raw = text.encode()
    return len(zlib.compress(raw, 9)) / max(1, len(raw)) < DEGEN_RATIO


def trim_degenerate(text, keep=2):
    """Last resort when every retry came back as spam.

    Collapses a repeated block of one to four words, so ninety copies of
    `thank you` become two. Then collapses a repeated character cycle such as
    `hehehehe`, which carries no space to split on.
    """
    words = (text or "").split()
    for size in range(1, 5):
        i = 0
        out = []
        while i < len(words):
            block = words[i : i + size]
            if len(block) < size:
                out.extend(words[i:])
                break
            runs = 0
            while words[i + runs * size : i + (runs + 1) * size] == block:
                runs += 1
            for _ in range(min(runs, keep)):
                out.extend(block)
            i += runs * size
        words = out
    text = " ".join(words)
    for size in (1, 2, 3, 4):
        unit = text[:size]
        if unit and len(text) >= size * 6 and set(re.findall(f"(?s).{{{size}}}", text)) == {unit}:
            return unit * keep
    return text


ACK_RE = re.compile(r"^[\W_]*(.*?)[\W_]*$", re.S)


def is_ack(text):
    """True when a reply only acknowledges and adds nothing."""
    from twin.export import LOW_SIGNAL

    core = (text or "").strip().lower().strip(".!,")
    return core in LOW_SIGNAL


def initiative(texts, incomings):
    """Does the reply move the conversation, or only agree with it?

    Replaying a real conversation showed the fault that no surface metric sees.
    A gold reply proposes the next step. The 4B agrees and stops. Both match the
    target register, and only one carries the conversation.

    The question rate separates the two models where `style_distance` could not,
    which is why the model a reader prefers can rank worse on surface features.
    """
    texts = list(texts)
    if not texts:
        return {"question": 0.0, "ack": 0.0, "new_words": 0.0}
    fresh = []
    for text, incoming in zip(texts, list(incomings) + [""] * len(texts)):
        words = set(WORD_RE.findall((text or "").lower()))
        prior = set(WORD_RE.findall((incoming or "").lower()))
        fresh.append(len(words - prior) / len(words) if words else 0.0)
    return {
        "question": statistics.fmean(1.0 if "?" in t else 0.0 for t in texts),
        "ack": statistics.fmean(1.0 if is_ack(t) else 0.0 for t in texts),
        "new_words": statistics.fmean(fresh),
    }


def features(texts):
    texts = [t for t in texts if t is not None]
    if not texts:
        return {}
    lengths = sorted(len(t) for t in texts)
    letters = "".join(texts)
    alpha = [c for c in letters if c.isalpha()]
    words = [w.lower() for t in texts for w in WORD_RE.findall(t)]
    return {
        "median_chars": lengths[len(lengths) // 2],
        "bubbles": statistics.fmean(len(split_bubbles(t)) for t in texts),
        "lower_frac": (sum(1 for c in alpha if c.islower()) / len(alpha)) if alpha else 0.0,
        # `lower_frac` averages over every character and hides this. The best 4B
        # opened replies with a capital about five times as often as the gold
        # replies, while the two lower_frac values were 0.964 and 0.984.
        "caps_start": statistics.fmean(1.0 if t and t[0].isupper() else 0.0 for t in texts),
        "ends_punct": statistics.fmean(1.0 if t and t[-1] in ".!?" else 0.0 for t in texts),
        "emoji_per_reply": statistics.fmean(len(EMOJI_RE.findall(t)) for t in texts),
        "short_frac": statistics.fmean(1.0 if len(t) <= 12 else 0.0 for t in texts),
        "degenerate": statistics.fmean(1.0 if is_degenerate(t) else 0.0 for t in texts),
        "vocab": Counter(words),
    }


def vocab_overlap(pred_vocab, gold_vocab, top=VOCAB_TOP):
    """Share of my most common words that the model also reaches for."""
    if not pred_vocab or not gold_vocab:
        return 0.0
    gold_top = [w for w, _ in gold_vocab.most_common(top)]
    if not gold_top:
        return 0.0
    pred_total = sum(pred_vocab.values()) or 1
    gold_total = sum(gold_vocab.values()) or 1
    # Compare frequency profiles, so using a word at the right rate scores best.
    score = 0.0
    for word in gold_top:
        p = pred_vocab.get(word, 0) / pred_total
        g = gold_vocab[word] / gold_total
        score += min(p, g)
    return score / sum(gold_vocab[w] / gold_total for w in gold_top)


def style_report(preds, golds, incomings=None):
    """Per-feature deltas plus one distance. Lower distance is closer to me.

    Pass ``incomings`` to score initiative, which is the only part of this that
    asks whether a reply carries the conversation rather than how it looks.
    """
    p = features(preds)
    g = features(golds)
    if not p or not g:
        return {}
    length_gap = abs(math.log(max(1, p["median_chars"]) / max(1, g["median_chars"])))
    overlap = vocab_overlap(p["vocab"], g["vocab"])
    parts = {
        "length_gap": length_gap,
        "bubble_gap": abs(p["bubbles"] - g["bubbles"]),
        "lower_gap": abs(p["lower_frac"] - g["lower_frac"]),
        "punct_gap": abs(p["ends_punct"] - g["ends_punct"]),
        "emoji_gap": abs(p["emoji_per_reply"] - g["emoji_per_reply"]),
        "short_gap": abs(p["short_frac"] - g["short_frac"]),
        "caps_gap": abs(p["caps_start"] - g["caps_start"]),
        "degenerate": p["degenerate"],
        "vocab_overlap": overlap,
    }
    # Length is a weak signal: a longer reply is acceptable. Repetition
    # spam is not, so it carries the largest weight in this sum.
    parts["style_distance"] = (
        0.25 * length_gap
        + parts["bubble_gap"]
        + 2.0 * parts["lower_gap"]
        + parts["punct_gap"]
        + 0.5 * parts["emoji_gap"]
        + parts["short_gap"]
        # `caps_gap` stays reported but out of the sum. It was 84% of the gap
        # between the 8B and the 4B, and it is useless signal: a capital letter
        # is cosmetic and a post-process can fix it.
        + 5.0 * parts["degenerate"]
        + 2.0 * (1.0 - overlap)
    )
    if incomings is not None:
        pi = initiative(preds, incomings)
        gi = initiative(golds, incomings)
        parts["initiative"] = pi
        parts["gold_initiative"] = gi
        parts["question_gap"] = abs(pi["question"] - gi["question"])
        parts["ack_gap"] = abs(pi["ack"] - gi["ack"])
        parts["new_word_gap"] = abs(pi["new_words"] - gi["new_words"])
        # Weighted to matter. A reply that only agrees is the fault a reader
        # reports first, and no surface feature above can see it.
        parts["style_distance"] += (
            3.0 * parts["question_gap"]
            + 3.0 * parts["ack_gap"]
            + 2.0 * parts["new_word_gap"]
        )
    parts["pred"] = {k: v for k, v in p.items() if k != "vocab"}
    parts["gold"] = {k: v for k, v in g.items() if k != "vocab"}
    return parts
