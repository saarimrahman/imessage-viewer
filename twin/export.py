#!/usr/bin/env python3
"""Build an all-history iMessage dataset for local MLX fine-tuning.

Every non-empty text sent by the user is present in the training file. Messages
are grouped into natural turns, so a run of consecutive bubbles becomes one
target without losing any of the original text. Real short-context variants
add useful examples without asking a cloud model to invent private data.
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CACHE_DIR
from db import MSG_SELECT, REACTION_EXCLUDE_SQL, get_conn, message_text

SYSTEM = (
    "You are my texting twin. Reply as I would reply in iMessage: match my "
    "word choice, casing, punctuation, rhythm, emoji, and message length. "
    "Respond only with the text I would send. Never use an assistant voice."
)
OPENER_PROMPT = "Send a natural text in my voice to continue or open this conversation."
TWIN_DIR = os.path.join(CACHE_DIR, "twin")
CONTEXT_TURNS = 6
REFERENCE_FRACTION = 0.05
REFERENCE_MAX = 500
TURN_GAP_NS = 30 * 60 * 1_000_000_000


def collapse_turns(pairs):
    """Merge consecutive bubbles from the same side into one chat turn."""
    turns = []
    for pair in pairs:
        is_from_me, text = pair[:2]
        date = pair[2] if len(pair) > 2 else None
        text = (text or "").strip()
        if not text:
            continue
        role = "assistant" if is_from_me else "user"
        close_in_time = (
            date is None
            or turns[-1].get("_date") is None
            or date - turns[-1]["_date"] <= TURN_GAP_NS
        ) if turns else False
        if turns and turns[-1]["role"] == role and close_in_time:
            turns[-1]["content"] += "\n" + text
            turns[-1]["_date"] = date
        else:
            turns.append({"role": role, "content": text, "_date": date})
    for turn in turns:
        turn.pop("_date", None)
    return turns


def _clean_turn(turn):
    return {"role": turn["role"], "content": turn["content"]}


def _context_window(turns, end, max_turns=CONTEXT_TURNS):
    """Return an alternating window that ends on the sent turn at ``end``."""
    start = max(0, end + 1 - max_turns)
    window = turns[start : end + 1]
    while window and window[0]["role"] != "user":
        window = window[1:]
    if len(window) < 2:
        return [
            {"role": "user", "content": OPENER_PROMPT},
            _clean_turn(turns[end]),
        ]
    return [_clean_turn(turn) for turn in window]


def examples_from_turns(turns, system=SYSTEM, augment=False):
    """Build a target for every sent turn and optional real-context variants."""
    examples = []
    for i, turn in enumerate(turns):
        if turn["role"] != "assistant":
            continue
        window = _context_window(turns, i)
        examples.append({"messages": [{"role": "system", "content": system}] + window})

        # A shorter view of the same real exchange teaches direct reply style and
        # gives older conversations more chances to be sampled. It is only added
        # when it differs from the primary context window.
        if augment and len(window) > 2:
            short_window = _context_window(turns, i, max_turns=2)
            if short_window != window:
                examples.append(
                    {"messages": [{"role": "system", "content": system}] + short_window}
                )
    return examples


def conversation_ids(conn, newest_first=False):
    """Every chat with a real message, ordered for complete or quick export."""
    return [
        r["chat_id"]
        for r in conn.execute(
            f"""SELECT cmj.chat_id, min(m.date) AS first_date
                       , max(m.date) AS last_date
                FROM chat_message_join cmj
                JOIN message m ON m.ROWID = cmj.message_id
                WHERE {REACTION_EXCLUDE_SQL}
                GROUP BY cmj.chat_id
                ORDER BY {"last_date DESC" if newest_first else "first_date, cmj.chat_id"}"""
        )
    ]


def fetch_pairs(conn, chat_id, per_chat=0):
    """Fetch text from one chat; ``per_chat=0`` reads its complete history."""
    params = [chat_id]
    limit_sql = ""
    if per_chat:
        limit_sql = " LIMIT ?"
        params.append(per_chat)
    rows = conn.execute(
        f"""{MSG_SELECT}
            WHERE cmj.chat_id = ? AND {REACTION_EXCLUDE_SQL}
            ORDER BY m.date DESC, m.ROWID DESC{limit_sql}""",
        params,
    ).fetchall()
    pairs = []
    for row in reversed(rows):
        text = message_text(row)
        if text:
            pairs.append((bool(row["is_from_me"]), text, row["date"], row["id"]))
    return pairs


def collect_examples(conn, limit=0, per_chat=0, augment=True):
    """Collect examples across 1:1 and group chats.

    The optional limit is only for a fast smoke run. Complete runs use zero for
    both limits and therefore scan every chat and every year.
    """
    examples = []
    chats = 0
    sent_turns = 0
    sent_texts = 0
    contextual_turns = 0
    opener_turns = 0
    augmented = 0
    seen_message_ids = set()
    for chat_id in conversation_ids(conn, newest_first=bool(limit)):
        pairs = [
            pair
            for pair in fetch_pairs(conn, chat_id, per_chat)
            if pair[3] not in seen_message_ids
        ]
        seen_message_ids.update(pair[3] for pair in pairs)
        turns = collapse_turns(pairs)
        sent = [turn for turn in turns if turn["role"] == "assistant"]
        if not sent:
            continue
        chats += 1
        sent_turns += len(sent)
        sent_texts += sum(1 for pair in pairs if pair[0])
        for i, turn in enumerate(turns):
            if turn["role"] != "assistant":
                continue
            if i and any(t["role"] == "user" for t in turns[max(0, i - CONTEXT_TURNS + 1) : i]):
                contextual_turns += 1
            else:
                opener_turns += 1
        rows = examples_from_turns(turns, augment=augment)
        augmented += max(0, len(rows) - len(sent))
        examples.extend(rows)
        if limit and len(examples) >= limit:
            examples = examples[:limit]
            break
    return examples, {
        "chats": chats,
        "sent_turns": sent_turns,
        "sent_texts": sent_texts,
        "contextual_turns": contextual_turns,
        "opener_turns": opener_turns,
        "augmented": augmented,
        "examples": len(examples),
    }


def _reference_rows(examples):
    """A stable fit-reference sample while every row remains in training.

    This is intentionally not called a holdout: removing it would violate the
    product promise that every sent text contributes to the final adapter.
    """
    if len(examples) < 10:
        return []
    rng = random.Random(0)
    sample = list(examples)
    rng.shuffle(sample)
    n_reference = min(REFERENCE_MAX, max(1, math.ceil(len(sample) * REFERENCE_FRACTION)))
    return sample[:n_reference]


def write_dataset(examples, out_dir):
    """Write all examples to train.jsonl and a sampled fit-reference file."""
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(0)
    train = list(examples)
    rng.shuffle(train)
    reference = _reference_rows(examples)
    for name, rows in (("train", train), ("valid", reference)):
        path = os.path.join(out_dir, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(train), len(reference)


def dataset_profile():
    """Return a text-only privacy-safe audit for the Twin page."""
    conn = get_conn()
    try:
        sent_total = sent_texts = received_texts = attachments_only = 0
        seen = set()
        for row in conn.execute(
            f"""{MSG_SELECT}
                WHERE {REACTION_EXCLUDE_SQL}
                ORDER BY m.ROWID"""
        ):
            if row["id"] in seen:
                continue
            seen.add(row["id"])
            text = message_text(row)
            if row["is_from_me"]:
                sent_total += 1
                if text:
                    sent_texts += 1
                else:
                    attachments_only += 1
            elif text:
                received_texts += 1
        chat_counts = conn.execute(
            """SELECT sum(n = 1) AS direct, sum(n > 1) AS groups FROM (
                   SELECT chat_id, count(*) AS n
                   FROM chat_handle_join GROUP BY chat_id
               )"""
        ).fetchone()
        first_last = conn.execute(
            f"""SELECT min(m.date) AS first_date, max(m.date) AS last_date
                FROM message m WHERE {REACTION_EXCLUDE_SQL}"""
        ).fetchone()
        return {
            "sent_texts": sent_texts,
            "sent_messages": sent_total,
            "received_texts": received_texts,
            "attachments_only": attachments_only,
            "direct_chats": int(chat_counts["direct"] or 0),
            "group_chats": int(chat_counts["groups"] or 0),
            "first_date": first_last["first_date"] or 0,
            "last_date": first_last["last_date"] or 0,
        }
    finally:
        conn.close()


def export_dataset(limit=0, per_chat=0, out_dir=TWIN_DIR, augment=True):
    """Write MLX JSONL and return detailed coverage counts."""
    conn = get_conn()
    try:
        examples, stats = collect_examples(conn, limit, per_chat, augment)
    finally:
        conn.close()
    if not examples:
        raise RuntimeError(
            "No sent text to export. Grant Full Disk Access if chat.db is blocked."
        )
    n_train, n_reference = write_dataset(examples, out_dir)
    return {**stats, "train": n_train, "reference": n_reference}


def main():
    parser = argparse.ArgumentParser(description="Export all sent iMessage text for LoRA")
    parser.add_argument("--limit", type=int, default=0, help="Max examples (0 = all)")
    parser.add_argument(
        "--per-chat", type=int, default=0, help="Recent messages per chat (0 = all)"
    )
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--out", default=TWIN_DIR)
    args = parser.parse_args()

    try:
        stats = export_dataset(
            args.limit, args.per_chat, args.out, augment=not args.no_augment
        )
    except RuntimeError as e:
        sys.exit(str(e))
    print(
        f"Wrote {stats['train']} train + {stats['reference']} reference examples "
        f"covering {stats['sent_texts']} sent texts from {stats['chats']} chats to {args.out}"
    )


if __name__ == "__main__":
    main()
