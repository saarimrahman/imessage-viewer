#!/usr/bin/env python3
"""Build an iMessage dataset for local MLX fine-tuning.

Direct 1:1 chats are split into sessions, then each authentic reply becomes one
training example. Consecutive bubbles stay separate with a delimiter. Later
sessions are held out so validation measures generalization, not memorization.
Replies that still contain a redaction placeholder are dropped so the twin
does not learn to emit tokens such as ``[phone]``. Training prompts include
retrieved incoming/reply pairs so they match chat.
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CACHE_DIR
from contacts import person_key, resolve_contact
from db import MSG_SELECT, REACTION_EXCLUDE_SQL, get_conn, message_text

ME = "me"
# One token on the offered bases. `<|bubble|>` split into 5–6 pieces.
BUBBLE = "※"
BUBBLE_ALIASES = (BUBBLE, "<|bubble|>")
CONTEXT_TURNS = 10
MAX_SEQ_LENGTH = 768
SESSION_GAP_NS = 12 * 60 * 60 * 1_000_000_000
TURN_GAP_NS = 30 * 60 * 1_000_000_000
TRAIN_FRACTION = 0.80
VALID_FRACTION = 0.10
SYSTEM = (
    "You are my texting twin. Continue the conversation exactly as I would. "
    "Match my wording, capitalization, punctuation, emoji, cadence, and typical "
    "length. Output only the message text. Separate consecutive iMessage bubbles "
    f"with {BUBBLE}."
)
OPENER_PROMPT = "Send a natural text in my voice to continue or open this conversation."
MEDIA_MARK = "[attachment]"
MAX_EXACT_TARGET = 4
MAX_SHORT_TARGET = 2
RECENCY_HALF_LIFE_NS = 365 * 24 * 60 * 60 * 1_000_000_000
RECENCY_FLOOR = 0.15
# 1.0 measured best on the holdout once the metric scored initiative: style
# distance 1.138 against 1.282 at 0.7, with the question rate and the median
# reply length both closer to the gold replies. The setting climbed 0.5 to 0.7
# to 1.0 as the metric improved, because a surface metric rewards a model for
# writing the average sentence and a cooler sampler does exactly that.
# Held at 0.7. Temperature 1.0 wins on single replies, 1.138 against 1.282, but
# every benchmark here scores one reply and chat runs many. At 1.0 with its own
# replies in the context the 8B degraded into incoherent continuations. The
# multi-turn comparison decides this.
CHAT_TEMP = 0.7
CHAT_TOP_P = 0.9
MAX_BUBBLES = 4
REPETITION_PENALTY = 1.0
CHAT_REPETITION_PENALTY = 1.15
TWIN_DIR = os.path.join(CACHE_DIR, "twin")
LOW_SIGNAL = {
    "ok", "k", "kk", "okay", "ok.", "okay.",
    "lol", "lmao", "haha", "ha", "hahaha",
    "yeah", "yea", "yep", "yup", "yes", "ya",
    "no", "nah", "nope",
    "np", "ty", "thanks", "thank you",
    "sure", "bet", "word", "nice", "cool", "true", "facts", "same",
    "sounds good", "ok sounds good", "okay sounds good",
    "omg", "wow",
    "👍", "👌", "😂", "❤️", "💕", "🔥",
}
ARTIFACT_QUOTE_RE = re.compile(
    r"(?is)^(Loved|Liked|Disliked|Laughed at|Emphasized|Questioned|"
    r"Reacted(?: with \S+)?|Removed (?:a )?(?:like|love|laugh|emphasis|"
    r"question mark|reaction))\s+[“\"'].+[”\"']$"
)
ARTIFACT_BARE_RE = re.compile(
    r"(?is)^(Loved|Liked|Disliked|Laughed at|Emphasized|Questioned)\s+"
    r"(an?|your|my)\s+\S+"
)
UNSENT_TEXT_RE = re.compile(
    r"(?is)^(you )?unsent a message\.?$|^this message was (deleted|unsent)\.?$"
)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(
    r"(?<!\d)(?:\+1[\s.\-]?)?(?:\(?\d{3}\)?[\s.\-]?)\d{3}[\s.\-]\d{4}(?!\d)"
)
CARD_RE = re.compile(r"(?<!\d)(?:\d[ \-]*){13,19}(?!\d)")
SECRET_RE = re.compile(
    r"(?i)\b(?:password|passwd|passcode|pin|otp|2fa code|verification code|"
    r"login code|auth code)\b[:\s#]*\S+"
)
ADDR_RE = re.compile(
    r"\b\d{1,5}\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?\s+"
    r"(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane|Dr|Drive|"
    r"Ct|Court|Way|Hwy|Highway)\.?\b",
    re.I,
)
CODE_MSG_RE = re.compile(r"^\d{4,8}$")
URL_ONLY_RE = re.compile(r"(?i)^https?://\S+$")
REDACTION_TOKEN_RE = re.compile(r"\[(?:phone|email|code|password|card|address)\]")
REDACTION_ONLY_RE = re.compile(
    r"^(?:\[(?:phone|email|code|password|card|address)\](?:\s*"
    + re.escape(BUBBLE)
    + r"\s*)?)+$"
)
_TWIN_SELECT_SQL = None


def person_id_for(handle):
    """Stable filesystem id for a contact. Phone and email on the same card merge."""
    if not handle or handle == ME:
        return ME
    key = person_key(handle)
    if not key:
        return None
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def parse_person_arg(value):
    """Accept ``me``, a 16-character person id, or a phone/email handle."""
    if not value or value == ME:
        return ME
    text = str(value).strip()
    if len(text) == 16 and all(c in "0123456789abcdef" for c in text.lower()):
        return text.lower()
    pid = person_id_for(text)
    if not pid:
        raise ValueError("Could not resolve that contact.")
    return pid


def system_for(name=None, peer=None):
    if not name or name == "You":
        base = SYSTEM
    else:
        base = (
            f"You are {name}'s texting twin. Continue the conversation exactly as "
            f"{name} would. Match their wording, capitalization, punctuation, emoji, "
            "cadence, and typical length. Output only the message text. Separate "
            f"consecutive iMessage bubbles with {BUBBLE}."
        )
    if peer:
        peer = " ".join(str(peer).split())[:80]
        if peer:
            base = f"{base} You are texting {peer}."
    return base


def opener_for(name=None):
    if not name or name == "You":
        return OPENER_PROMPT
    return f"Send a natural text as {name} to continue or open this conversation."


def join_bubbles(parts):
    return BUBBLE.join(part for part in parts if part)


def split_bubbles(text):
    text = str(text or "")
    for alias in BUBBLE_ALIASES:
        if alias != BUBBLE:
            text = text.replace(alias, BUBBLE)
    return [part.strip() for part in text.split(BUBBLE) if part.strip()]


SCAFFOLD_RE = re.compile(
    r"<think>.*?</think>|</?think>|<tool_call>.*?</tool_call>|</?tool_call>"
    r"|<\|im_start\|>|<\|im_end\|>",
    flags=re.DOTALL,
)


THINK_INJECTION = (
    "'<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') "
    "+ '\\n</think>\\n\\n' + content.lstrip('\\n')"
)
THINK_REPLACEMENT = "'<|im_start|>' + message.role + '\\n' + content.lstrip('\\n')"


def sanitize_chat_template(tokenizer):
    """Stop the template from opening the target reply with an empty think block.

    The Qwen 3 2507 template renders only the **last** assistant turn as
    `<think>\\n\\n</think>\\n\\n` plus the content. Training then teaches the model
    to emit that scaffolding first, while the base model goes straight to the
    content after the generation prompt. The two disagree on the very first
    token, and the adapter answers with stray special tokens such as
    `<tool_call>`. Removing the injection makes the target match generation.

    Returns True when the template changed.
    """
    template = getattr(tokenizer, "chat_template", None)
    if not template or THINK_INJECTION not in template:
        return False
    patched = template.replace(THINK_INJECTION, THINK_REPLACEMENT)
    try:
        tokenizer.chat_template = patched
    except Exception:
        return False
    return True


def strip_scaffold(text):
    """Remove chat-template scaffolding that the model emits before the reply.

    The Qwen 3 2507 template renders the last assistant turn with an empty
    `<think>` block. Training therefore teaches the model to open with
    scaffolding, and it sometimes reaches for a neighbouring special token such
    as `<tool_call>`. None of that is part of my message, so it must not be
    scored or shown.
    """
    if not text:
        return ""
    return SCAFFOLD_RE.sub("", text).strip()


def clip_bubbles(text, max_bubbles=MAX_BUBBLES):
    """Keep the first N bubbles so greedy decoding cannot loop on the delimiter."""
    text = (text or "").strip()
    # Always rejoin. `split_bubbles` drops empty parts, so this also removes a
    # trailing delimiter, which the model emits and the chat page displayed.
    return join_bubbles(split_bubbles(text)[:max_bubbles])


def is_artifact(text):
    """True for tapback summaries, unsent notices, and similar non-reply text."""
    stripped = (text or "").strip()
    if not stripped:
        return False
    return bool(
        ARTIFACT_QUOTE_RE.match(stripped)
        or ARTIFACT_BARE_RE.match(stripped)
        or UNSENT_TEXT_RE.match(stripped)
    )


def is_media_only(text):
    parts = split_bubbles(text)
    return bool(parts) and all(part == MEDIA_MARK for part in parts)


def is_link_only(text):
    body = (text or "").replace(BUBBLE, " ").strip()
    return bool(URL_ONLY_RE.match(body))


def is_low_signal(text):
    stripped = (text or "").replace(BUBBLE, " ").strip()
    if not stripped:
        return True
    if stripped in LOW_SIGNAL:
        return True
    norm = re.sub(r"[^\w\s]+", "", stripped.lower()).strip()
    if norm in LOW_SIGNAL:
        return True
    words = stripped.split()
    return len(words) <= 2 and len(stripped) <= 12


def redact_text(text):
    """Replace secrets and contact details. Keep ordinary chat wording."""
    if not text:
        return text
    out = []
    for part in str(text).split(BUBBLE):
        piece = part
        if CODE_MSG_RE.match(piece.strip()):
            out.append("[code]")
            continue
        piece = EMAIL_RE.sub("[email]", piece)
        piece = SECRET_RE.sub("[password]", piece)
        piece = CARD_RE.sub("[card]", piece)
        piece = PHONE_RE.sub("[phone]", piece)
        piece = ADDR_RE.sub("[address]", piece)
        out.append(piece)
    return BUBBLE.join(out)


def redact_messages(messages):
    out = []
    for message in messages:
        if message.get("role") == "system":
            out.append(message)
            continue
        out.append({**message, "content": redact_text(message.get("content") or "")})
    return out


def contains_redaction(text):
    """True when any redaction placeholder remains in the text."""
    return bool(REDACTION_TOKEN_RE.search(text or ""))


def is_redaction_only(text):
    compact = (text or "").strip()
    return (not compact.replace(BUBBLE, "").strip()) or bool(REDACTION_ONLY_RE.match(compact))


def bubble_token_count(tokenizer):
    """How many tokenizer pieces ``BUBBLE`` becomes. None if it cannot be counted."""
    if tokenizer is None or not hasattr(tokenizer, "encode"):
        return None
    try:
        ids = tokenizer.encode(BUBBLE, add_special_tokens=False)
    except TypeError:
        ids = tokenizer.encode(BUBBLE)
    if isinstance(ids, dict):
        ids = ids.get("input_ids") or []
    try:
        return len(ids)
    except TypeError:
        return None


def recency_weight(date_ns, now_ns, half_life=RECENCY_HALF_LIFE_NS):
    age = max(0, (now_ns or 0) - (date_ns or 0))
    if half_life <= 0:
        return 1.0
    return 0.5 ** (age / half_life)


def cap_duplicate_targets(examples, max_exact=MAX_EXACT_TARGET, max_short=MAX_SHORT_TARGET):
    """Keep at most a few copies of any exact reply, fewer if it is an ack."""
    groups = {}
    order = []
    for example in examples:
        key = example.get("_reply") or ""
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(example)
    kept = []
    dropped = 0
    for key in order:
        rows = groups[key]
        rows.sort(key=lambda row: row.get("_date") or 0, reverse=True)
        limit = max_short if is_low_signal(key) else max_exact
        kept.extend(rows[:limit])
        dropped += max(0, len(rows) - limit)
    kept.sort(key=lambda row: (row.get("_date") or 0, str(row.get("_id") or "")))
    return kept, dropped


def cap_low_signal_share(examples, max_share=1.0, rng=None):
    """Hold ack replies to at most ``max_share`` of the training rows.

    A model trained by maximum likelihood then decoded greedily amplifies its
    safest mode. Measured here: the adapter produces acks at about twice the
    rate of the gold replies. Training below the true rate cancels that
    amplification, so the output rate lands near the real one.

    Keeps the most recent acks, because those carry current habits.
    """
    if max_share >= 1.0 or not examples:
        return examples, 0
    low = [row for row in examples if is_low_signal(row.get("_reply") or "")]
    rich = [row for row in examples if not is_low_signal(row.get("_reply") or "")]
    if not low:
        return examples, 0
    # Solve keep / (keep + len(rich)) = max_share for keep.
    allowed = int(max_share * len(rich) / max(1e-9, 1.0 - max_share))
    if allowed >= len(low):
        return examples, 0
    low.sort(key=lambda row: row.get("_date") or 0, reverse=True)
    kept = low[:allowed]
    dropped = len(low) - len(kept)
    out = rich + kept
    out.sort(key=lambda row: (row.get("_date") or 0, str(row.get("_id") or "")))
    return out, dropped


def downsample_recency(examples, rng=None, floor=RECENCY_FLOOR):
    """Keep recent train rows; keep older rows with decaying probability."""
    if not examples:
        return examples, 0
    rng = rng or random.Random(0)
    now = max(example.get("_date") or 0 for example in examples)
    kept = []
    dropped = 0
    for example in examples:
        keep_p = max(floor, recency_weight(example.get("_date") or 0, now))
        if keep_p >= 1.0 or rng.random() <= keep_p:
            kept.append(example)
        else:
            dropped += 1
    if not kept or len(kept) < max(8, len(examples) // 5):
        return examples, 0
    return kept, dropped


def target_concentration(examples, top_n=20):
    replies = [example.get("_reply") or "" for example in examples]
    n = len(replies)
    if not n:
        return {"unique_targets": 0, "top20_share": 0.0}
    counts = Counter(replies)
    top = sum(count for _, count in counts.most_common(top_n))
    return {"unique_targets": len(counts), "top20_share": top / n}


def peer_from_system(text):
    marker = "You are texting "
    if marker not in (text or ""):
        return ""
    return (text or "").split(marker, 1)[1].split(".", 1)[0].strip()


def twin_msg_select(conn):
    """Message select plus unsent and attachment flags when the columns exist."""
    global _TWIN_SELECT_SQL
    if _TWIN_SELECT_SQL is None:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(message)")}
        unsent = "COALESCE(m.is_unsent, 0)" if "is_unsent" in cols else "0"
        attach = (
            "COALESCE(m.cache_has_attachments, 0)"
            if "cache_has_attachments" in cols
            else "0"
        )
        _TWIN_SELECT_SQL = (
            "SELECT m.ROWID as id, m.guid, m.text, m.attributedBody, m.date, "
            f"m.is_from_me, h.id as handle, {unsent} AS is_unsent, "
            f"{attach} AS has_attachments "
            "FROM message m "
            "JOIN chat_message_join cmj ON cmj.message_id = m.ROWID "
            "LEFT JOIN handle h ON h.ROWID = m.handle_id"
        )
    return _TWIN_SELECT_SQL


def chat_peer_label(conn, chat_id, target_key=None):
    """Contact card name for a 1:1 chat, or a generic label that is not a handle."""
    if target_key is not None:
        return "You"
    row = conn.execute(
        """SELECT h.id AS handle FROM chat_handle_join chj
           JOIN handle h ON h.ROWID = chj.handle_id
           WHERE chj.chat_id = ? LIMIT 1""",
        [chat_id],
    ).fetchone()
    if not row:
        return None
    return resolve_contact(row["handle"]) or "this contact"


def is_assistant_row(row, target_key=None):
    """True when this message is the voice we are training."""
    if target_key is None:
        return bool(row["is_from_me"])
    return (not row["is_from_me"]) and person_key(row["handle"]) == target_key


def resolve_subject(person_id=ME):
    """Return the twin subject for ``person_id``, or raise ValueError."""
    pid = person_id or ME
    if pid == ME:
        return {
            "id": ME,
            "name": "You",
            "handle": None,
            "handles": None,
            "key": None,
            "texts": 0,
        }
    for subject in list_subjects():
        if subject["id"] == pid:
            return subject
    raise ValueError("Unknown contact.")


def list_subjects():
    """You first, then contacts merged by card, most messages first."""
    conn = get_conn()
    try:
        sent = conn.execute(
            f"""SELECT count(*) AS n FROM message m
                WHERE m.is_from_me = 1 AND {REACTION_EXCLUDE_SQL}"""
        ).fetchone()["n"]
        people = [
            {
                "id": ME,
                "name": "You",
                "handle": None,
                "handles": None,
                "key": None,
                "texts": int(sent or 0),
            }
        ]
        grouped = {}
        for row in conn.execute(
            f"""SELECT h.id AS handle, count(*) AS c
                FROM message m JOIN handle h ON h.ROWID = m.handle_id
                WHERE m.is_from_me = 0 AND {REACTION_EXCLUDE_SQL}
                GROUP BY h.id"""
        ):
            pid = person_id_for(row["handle"])
            if not pid:
                continue
            resolved = resolve_contact(row["handle"])
            name = resolved or row["handle"]
            entry = grouped.get(pid)
            if entry is None:
                grouped[pid] = {
                    "id": pid,
                    "name": name,
                    "handle": row["handle"],
                    "handles": [row["handle"]],
                    "key": person_key(row["handle"]),
                    "texts": row["c"],
                    "_best": row["c"],
                }
                continue
            entry["texts"] += row["c"]
            entry["handles"].append(row["handle"])
            if resolved:
                entry["name"] = resolved
            if row["c"] > entry["_best"]:
                entry["_best"] = row["c"]
                entry["handle"] = row["handle"]
                entry["key"] = person_key(row["handle"])
                if resolved:
                    entry["name"] = resolved
        people.extend(
            sorted(
                ({k: v for k, v in p.items() if k != "_best"} for p in grouped.values()),
                key=lambda p: -p["texts"],
            )
        )
        return people
    finally:
        conn.close()


def chat_sender_names(conn, chat_id):
    """Display name for each handle in a group chat, empty for a direct chat.

    A direct chat needs no labels: one other person owns every incoming turn.
    """
    rows = conn.execute(
        """SELECT h.id FROM chat_handle_join chj
           JOIN handle h ON h.ROWID = chj.handle_id
           WHERE chj.chat_id = ?""",
        (chat_id,),
    ).fetchall()
    handles = [row[0] for row in rows if row[0]]
    if len(handles) < 2:
        return {}
    names = {}
    for handle in handles:
        label = (resolve_contact(handle) or "").strip() or handle
        names[handle] = label.split()[0][:24]
    return names


def collapse_turns(pairs):
    """Merge consecutive bubbles from the same sender when they are close in time."""
    turns = []
    for pair in pairs:
        is_from_me, text = pair[:2]
        date = pair[2] if len(pair) > 2 else None
        msg_id = pair[3] if len(pair) > 3 else None
        sender = pair[4] if len(pair) > 4 else ("assistant" if is_from_me else "user")
        text = (text or "").strip()
        if not text:
            continue
        role = "assistant" if is_from_me else "user"
        close_in_time = (
            date is None
            or turns[-1].get("_date") is None
            or date - turns[-1]["_date"] <= TURN_GAP_NS
        ) if turns else False
        same_sender = turns and turns[-1].get("_sender") == sender
        if turns and turns[-1]["role"] == role and same_sender and close_in_time:
            turns[-1]["content"] = join_bubbles([turns[-1]["content"], text])
            turns[-1]["_date"] = date
            turns[-1]["_id"] = msg_id
        else:
            turns.append(
                {
                    "role": role,
                    "content": text,
                    "_date": date,
                    "_id": msg_id,
                    "_sender": sender,
                }
            )
    return turns


def sessionize(turns, gap_ns=SESSION_GAP_NS):
    """Split a chat wherever the gap is longer than a sitting."""
    sessions = []
    current = []
    for turn in turns:
        if current:
            prev = current[-1].get("_date")
            cur = turn.get("_date")
            if prev is not None and cur is not None and cur - prev > gap_ns:
                sessions.append(current)
                current = []
        current.append(turn)
    if current:
        sessions.append(current)
    return sessions


def _clean_turn(turn):
    return {"role": turn["role"], "content": turn["content"]}


def _merge_same_role(turns):
    """Collapse consecutive same-role turns so chat templates stay alternating."""
    merged = []
    for turn in turns:
        if merged and merged[-1]["role"] == turn["role"]:
            merged[-1] = {
                **merged[-1],
                "content": join_bubbles([merged[-1]["content"], turn["content"]]),
                "_date": turn.get("_date"),
                "_id": turn.get("_id"),
            }
        else:
            merged.append(dict(turn))
    return merged


def coerce_chat(messages):
    """Keep a legal user/assistant thread for strict chat templates."""
    if not messages:
        return []
    system = [messages[0]] if messages[0].get("role") == "system" else []
    rest = messages[len(system) :]
    merged = _merge_same_role(rest)
    while merged and merged[0]["role"] != "user":
        merged = merged[1:]
    return system + [_clean_turn(turn) for turn in merged]


def is_alternating(messages):
    """True when roles are user/assistant/user/assistant, ending on assistant."""
    roles = [m["role"] for m in messages if m["role"] != "system"]
    if len(roles) < 2 or roles[0] != "user" or roles[-1] != "assistant":
        return False
    return all(roles[i] != roles[i + 1] for i in range(len(roles) - 1))


def _context_window(turns, end, max_turns=CONTEXT_TURNS):
    """Return an alternating window that ends on the sent turn at ``end``.

    Conversation starters with no incoming message in this session are skipped.
    """
    start = max(0, end + 1 - max_turns)
    window = turns[start : end + 1]
    context = _merge_same_role(window[:-1])
    while context and context[0]["role"] != "user":
        context = context[1:]
    if not context or context[-1]["role"] != "user":
        return None
    merged = [_clean_turn(turn) for turn in context] + [_clean_turn(window[-1])]
    if not is_alternating(merged):
        return None
    return merged


SENDER_RE = re.compile(r"^[^\n:]{1,24}: ")


def label_senders(turns, names):
    """Prefix each incoming turn with who sent it.

    A group chat puts several people in the `user` role, so without this the
    model cannot tell who is speaking and cannot hold a thread with more than
    one person. Group chats hold about a third of the sent messages in a typical
    database, so this is the largest untapped source.

    The trained voice is never labelled. The twin must emit the message text
    alone.
    """
    out = []
    for turn in turns:
        item = dict(turn)
        if item["role"] == "user":
            who = names.get(item.get("_sender") or "")
            if who and not SENDER_RE.match(item["content"] or ""):
                item["content"] = f"{who}: {item['content']}"
        out.append(item)
    return out


def examples_from_turns(
    turns, system=SYSTEM, augment=False, opener=OPENER_PROMPT, depth_ladder=()
):
    """Build one target for every reply that has incoming context in this session.

    ``augment`` is ignored: each target is written once. ``opener`` is unused;
    conversation starters are a separate mode and are not mixed into reply training.
    """
    del augment, opener
    examples = []
    for i, turn in enumerate(turns):
        if turn["role"] != "assistant":
            continue
        window = _context_window(turns, i)
        if window is None:
            continue
        depths = [CONTEXT_TURNS]
        if depth_ladder:
            # A shallow copy of the same target. Almost every row carries a full
            # context window, yet the first message of every real chat has
            # exactly one turn. Rotating the extra depth by
            # position keeps each target to two copies instead of four.
            # The ladder is written in context turns, which is what the data
            # audit reports. `_context_window` counts the target too, so a
            # ladder value of 1 means `max_turns=2`. Passing 1 straight through
            # asked for a window holding only the target and produced nothing.
            shallow = depth_ladder[i % len(depth_ladder)] + 1
            if shallow < CONTEXT_TURNS:
                depths.append(shallow)
        seen = set()
        for depth in depths:
            view = _context_window(turns, i, max_turns=depth)
            if view is None:
                continue
            key = len(view)
            if key in seen:
                continue
            seen.add(key)
            examples.append(
                {
                    "messages": [{"role": "system", "content": system}] + view,
                    "_sid": None,
                    "_date": turn.get("_date") or 0,
                    "_id": turn.get("_id"),
                    "_query": view[-2]["content"],
                    "_reply": view[-1]["content"],
                }
            )
    return examples


def opener_examples(turns, system=SYSTEM, name=None):
    """Targets for the messages that open a session.

    Thousands of sent turns start a conversation with nothing before them, so
    `_context_window` drops every one. The twin therefore never learns to text
    first, which roleplay needs. The prompt stands in for the missing incoming
    message.
    """
    examples = []
    for i, turn in enumerate(turns):
        if turn["role"] != "assistant":
            continue
        if _context_window(turns, i) is not None:
            continue
        reply = (turn.get("content") or "").strip()
        if not reply:
            continue
        prompt = opener_for(name)
        examples.append(
            {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": reply},
                ],
                "_sid": None,
                "_date": turn.get("_date") or 0,
                "_id": turn.get("_id"),
                "_query": prompt,
                "_reply": reply,
                "_opener": True,
            }
        )
    return examples


def rendered_len(messages, tokenizer, chat_template_args=None):
    """Token count of a chat example, matching MLX ChatDataset rendering."""
    kwargs = dict(chat_template_args or {})
    if tokenizer is None:
        text = " ".join(m.get("content") or "" for m in messages)
        return (len(text) + 3) // 4
    try:
        out = tokenizer.apply_chat_template(messages, return_dict=False, **kwargs)
    except TypeError:
        out = tokenizer.apply_chat_template(messages, **kwargs)
    if isinstance(out, str):
        if hasattr(tokenizer, "encode"):
            return len(tokenizer.encode(out))
        return len(out)
    return len(out)


def fit_messages(messages, tokenizer, max_seq_length=MAX_SEQ_LENGTH, chat_template_args=None):
    """Drop oldest context from the left. Keep the full target. Reject if it cannot fit."""
    if not messages or messages[-1].get("role") != "assistant":
        return None
    system = [messages[0]] if messages[0].get("role") == "system" else []
    body = messages[len(system) :]
    target = body[-1]
    context = body[:-1]
    while True:
        candidate = system + context + [target]
        if len(candidate) < 2:
            return None
        if rendered_len(candidate, tokenizer, chat_template_args) <= max_seq_length:
            return candidate if is_alternating(candidate) else None
        if not context:
            return None
        context = context[1:]
        while context and context[0]["role"] != "user":
            context = context[1:]
        if not context:
            return None


CHUNK_ROWS = 80


def chunk_sessions(groups, order):
    """Cut a long session into chunks so the split has a fine enough unit.

    A session is the only leak-free unit, but sessions are very unequal. One
    measured contact had 41 sessions whose 5 largest held 88% of the rows, so no
    session-level split can hit 80/10/10 and stay representative. Chunking gives
    units of a similar size instead.

    Also returns the previous chunk of the same session for each chunk, so the
    caller can drop the rows whose context reaches back over a cut. Trimming
    after assignment costs nothing when both sides of a cut share a split.
    """
    chunked, new_order, previous = {}, [], {}
    for sid in order:
        rows = sorted(groups[sid], key=lambda ex: ex.get("_date") or 0)
        last = None
        for start in range(0, len(rows), CHUNK_ROWS):
            key = (sid, start)
            chunked[key] = rows[start : start + CHUNK_ROWS]
            previous[key] = last
            new_order.append(key)
            last = key
    return chunked, new_order, previous


def stratified_ids(order, sizes, train_frac, valid_frac):
    """Spread the holdout over the whole history, and size it by rows.

    Walks the units oldest first and sends each one to whichever split is
    furthest below its row quota, so both the size and the era mix come out
    right.
    """
    total = sum(sizes)
    quota = {
        "train": train_frac * total,
        "valid": valid_frac * total,
        "test": max(0.0, 1.0 - train_frac - valid_frac) * total,
    }
    got = {"train": 0, "valid": 0, "test": 0}
    assigned = {"train": set(), "valid": set(), "test": set()}
    for sid, size in zip(order, sizes):
        name = max(quota, key=lambda k: (quota[k] - got[k]) / max(1.0, quota[k]))
        assigned[name].add(sid)
        got[name] += size
    if not assigned["train"]:
        return None
    return assigned


def partition_examples(
    examples,
    train_frac=TRAIN_FRACTION,
    valid_frac=VALID_FRACTION,
    strategy="chrono",
):
    """Session-level split. Augmented copies of a target stay together.

    ``chrono`` holds out the newest sessions. ``stratified`` spreads the
    holdout over the whole history and sizes it by rows.
    """
    groups = {}
    order = []
    for i, example in enumerate(examples):
        sid = example.get("_sid")
        if sid is None:
            sid = ("row", i)
        if sid not in groups:
            groups[sid] = []
            order.append(sid)
        groups[sid].append(example)
    order.sort(
        key=lambda sid: (
            min(ex.get("_date") or 0 for ex in groups[sid]),
            str(sid),
        )
    )
    n = len(order)
    if strategy == "stratified" and n >= 10:
        pieces, piece_order, previous = chunk_sessions(groups, order)
        assigned = stratified_ids(
            piece_order,
            [len(pieces[key]) for key in piece_order],
            train_frac,
            valid_frac,
        )
        if assigned:
            where = {key: name for name, ids in assigned.items() for key in ids}
            for key in piece_order:
                before = previous.get(key)
                if before is not None and where[before] != where[key]:
                    pieces[key] = pieces[key][CONTEXT_TURNS:]
            return {
                name: [row for key in piece_order if key in ids for row in pieces[key]]
                for name, ids in assigned.items()
            }
    if n < 10:
        n_test = 1 if n >= 3 else 0
        n_valid = 1 if n >= 2 else 0
        n_train = n - n_valid - n_test
    else:
        n_test = max(1, round(n * (1 - train_frac - valid_frac)))
        n_valid = max(1, round(n * valid_frac))
        n_train = n - n_valid - n_test
        if n_train < 1:
            n_train = 1
            n_valid = max(0, n - n_train - n_test)
    train_ids = set(order[:n_train])
    valid_ids = set(order[n_train : n_train + n_valid])
    test_ids = set(order[n_train + n_valid :])
    split = {"train": [], "valid": [], "test": []}
    for sid, rows in groups.items():
        if sid in train_ids:
            split["train"].extend(rows)
        elif sid in valid_ids:
            split["valid"].extend(rows)
        elif sid in test_ids:
            split["test"].extend(rows)
    return split


def inject_train_shots(
    examples,
    tokenizer=None,
    max_seq_length=MAX_SEQ_LENGTH,
    chat_template_args=None,
):
    """Prepend retrieved train pairs so the adapter learns to use them.

    ``retrieve.jsonl`` stays the original pairs. If shots cannot fit with the
    full target, the example is left unchanged.
    """
    from twin.retrieve import RETRIEVE_K, index_from_examples, retrieve, with_retrieved_shots

    if not examples:
        return examples, 0
    index = index_from_examples(examples)
    out = []
    injected = 0
    for example in examples:
        query = example.get("_query") or ""
        reply = example.get("_reply") or ""
        pairs = retrieve(
            query,
            index,
            k=RETRIEVE_K,
            exclude=[query],
            exclude_replies=[reply],
            peer=example.get("_peer") or None,
        )
        if not pairs:
            out.append(example)
            continue
        candidate = with_retrieved_shots(example["messages"], pairs=pairs)
        fitted = fit_messages(
            candidate,
            tokenizer,
            max_seq_length=max_seq_length,
            chat_template_args=chat_template_args,
        )
        if fitted is None or not is_alternating(fitted):
            out.append(example)
            continue
        shot_query = pairs[0]["query"]
        if not any(m.get("content") == shot_query for m in fitted):
            out.append(example)
            continue
        row = dict(example)
        row["messages"] = fitted
        out.append(row)
        injected += 1
    return out, injected


def public_example(example):
    return {"messages": example["messages"]}


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_dataset(examples, out_dir, train=None, valid=None, test=None):
    """Write disjoint train/valid/test JSONL files.

    ``examples`` may be a flat list (then it is partitioned) or omitted when
    the three splits are passed explicitly.
    """
    os.makedirs(out_dir, exist_ok=True)
    if train is None and valid is None and test is None:
        split = partition_examples(examples or [])
        train, valid, test = split["train"], split["valid"], split["test"]
    rng = random.Random(0)
    train_rows = [public_example(row) for row in (train or [])]
    rng.shuffle(train_rows)
    write_jsonl(os.path.join(out_dir, "train.jsonl"), train_rows)
    write_jsonl(os.path.join(out_dir, "valid.jsonl"), [public_example(row) for row in (valid or [])])
    write_jsonl(os.path.join(out_dir, "test.jsonl"), [public_example(row) for row in (test or [])])
    retrieve_rows = [
        {
            "query": row["_query"],
            "reply": row["_reply"],
            "peer": row.get("_peer") or "",
            "date": row.get("_date") or 0,
        }
        for row in (train or [])
        if row.get("_query") and row.get("_reply")
    ]
    write_jsonl(os.path.join(out_dir, "retrieve.jsonl"), retrieve_rows)
    write_multiturn(out_dir, valid or [])
    return len(train_rows), len(valid or []), len(test or [])


MULTITURN_DEPTHS = (2, 3, 4, 6)
MULTITURN_PER_DEPTH = 40


def write_multiturn(out_dir, valid):
    """Persist a depth-stratified multi-turn holdout.

    Single-reply scoring hides the failure that matters, and picking windows
    opportunistically means each run scores a different mix of conversation
    lengths. This fixes the set: the same windows every time, balanced over
    depth, so a change at 6 turns is not hidden by a change at 2.

    Depth is the number of my turns the model must generate in one window. The
    rows come from `valid` only, so they carry the split already in force.
    """
    by_depth = {}
    for row in valid:
        messages = row.get("messages") or []
        depth = sum(1 for m in messages if m.get("role") == "assistant")
        for bucket in MULTITURN_DEPTHS:
            if depth >= bucket:
                by_depth.setdefault(bucket, []).append(row)
    rows = []
    seen = set()
    rng = random.Random(0)
    for bucket in MULTITURN_DEPTHS:
        pool = by_depth.get(bucket) or []
        rng.shuffle(pool)
        taken = 0
        for row in pool:
            key = id(row)
            if key in seen:
                continue
            seen.add(key)
            item = public_example(row)
            item["depth"] = bucket
            rows.append(item)
            taken += 1
            if taken >= MULTITURN_PER_DEPTH:
                break
    write_jsonl(os.path.join(out_dir, "multiturn.jsonl"), rows)
    return len(rows)


def conversation_ids(conn, newest_first=False, chat_ids=None):
    """Every chat with a real message, ordered for complete or quick export."""
    extra = ""
    params = []
    if chat_ids is not None:
        if not chat_ids:
            return []
        extra = f" AND cmj.chat_id IN ({','.join('?' * len(chat_ids))})"
        params = list(chat_ids)
    return [
        r["chat_id"]
        for r in conn.execute(
            f"""SELECT cmj.chat_id, min(m.date) AS first_date
                       , max(m.date) AS last_date
                FROM chat_message_join cmj
                JOIN message m ON m.ROWID = cmj.message_id
                WHERE {REACTION_EXCLUDE_SQL}{extra}
                GROUP BY cmj.chat_id
                ORDER BY {"last_date DESC" if newest_first else "first_date, cmj.chat_id"}""",
            params,
        )
    ]


def direct_chat_ids(conn):
    """Chats with exactly one other participant."""
    return {
        r["chat_id"]
        for r in conn.execute(
            """SELECT chat_id FROM chat_handle_join
               GROUP BY chat_id HAVING count(*) = 1"""
        )
    }


def fetch_pairs(conn, chat_id, per_chat=0, target_key=None):
    """Fetch text from one chat; ``per_chat=0`` reads its complete history."""
    params = [chat_id]
    limit_sql = ""
    if per_chat:
        limit_sql = " LIMIT ?"
        params.append(per_chat)
    rows = conn.execute(
        f"""{twin_msg_select(conn)}
            WHERE cmj.chat_id = ? AND {REACTION_EXCLUDE_SQL}
            ORDER BY m.date DESC, m.ROWID DESC{limit_sql}""",
        params,
    ).fetchall()
    pairs = []
    for row in reversed(rows):
        if row["is_unsent"]:
            continue
        text = message_text(row)
        if is_artifact(text or ""):
            continue
        if not text:
            if row["has_attachments"]:
                text = MEDIA_MARK
            else:
                continue
        sender = "me" if row["is_from_me"] else (row["handle"] or "user")
        pairs.append(
            (is_assistant_row(row, target_key), text, row["date"], row["id"], sender)
        )
    return pairs


def load_export_tokenizer(model_key=None, chat_template_args=None):
    """Load the selected model's tokenizer, or None when it is not available."""
    if not model_key:
        return None
    try:
        from transformers import AutoTokenizer
        from twin.train import model_config

        config = model_config(model_key)
        tok = AutoTokenizer.from_pretrained(config["repo"], trust_remote_code=True)
        extra = chat_template_args if chat_template_args is not None else config.get("chat_template_args") or {}
        if extra:
            original = tok.apply_chat_template

            def apply_chat_template(*args, **kwargs):
                return original(*args, **{**extra, **kwargs})

            tok.apply_chat_template = apply_chat_template
        return tok
    except Exception:
        return None


def collect_examples(
    conn,
    limit=0,
    per_chat=0,
    augment=True,
    target_key=None,
    name=None,
    tokenizer=None,
    max_seq_length=MAX_SEQ_LENGTH,
    chat_template_args=None,
    include_groups=False,
    sender_labels=True,
    depth_ladder=(),
    openers=False,
):
    """Collect one reply example per authentic sent turn in 1:1 sessions."""
    del augment
    examples = []
    chats = 0
    sent_turns = 0
    sent_texts = 0
    contextual_turns = 0
    opener_turns = 0
    skipped_overflow = 0
    skipped_illegal = 0
    skipped_groups = 0
    skipped_media = 0
    skipped_link = 0
    skipped_pii = 0
    sessions_n = 0
    seen_message_ids = set()
    allowed = None if include_groups else direct_chat_ids(conn)
    for chat_id in conversation_ids(conn, newest_first=bool(limit), chat_ids=allowed):
        pairs = [
            pair
            for pair in fetch_pairs(conn, chat_id, per_chat, target_key)
            if pair[3] not in seen_message_ids
        ]
        seen_message_ids.update(pair[3] for pair in pairs)
        turns = collapse_turns(pairs)
        sent = [turn for turn in turns if turn["role"] == "assistant"]
        if not sent:
            continue
        chats += 1
        sent_turns += len(sent)
        sent_texts += sum(1 for pair in pairs if pair[0] and pair[1] != MEDIA_MARK)
        peer = None if include_groups else chat_peer_label(conn, chat_id, target_key)
        group_names = (
            chat_sender_names(conn, chat_id)
            if include_groups and sender_labels
            else {}
        )
        if group_names:
            turns = label_senders(turns, group_names)
        for session_i, session in enumerate(sessionize(turns)):
            sessions_n += 1
            sid = (chat_id, session_i)
            for i, turn in enumerate(session):
                if turn["role"] == "assistant" and _context_window(session, i) is None:
                    opener_turns += 1
            rows = examples_from_turns(
                session,
                system=system_for(name, peer=peer),
                depth_ladder=depth_ladder,
            )
            rows += opener_examples(
                session, system=system_for(name, peer=peer), name=name
            ) if openers else []
            for row in rows:
                query = row.get("_query") or ""
                reply = row.get("_reply") or ""
                if is_media_only(query) or is_media_only(reply):
                    skipped_media += 1
                    continue
                if is_link_only(reply):
                    skipped_link += 1
                    continue
                row["messages"] = redact_messages(row["messages"])
                row["_query"] = row["messages"][-2]["content"]
                row["_reply"] = row["messages"][-1]["content"]
                if contains_redaction(row["_reply"]):
                    skipped_pii += 1
                    continue
                row["_peer"] = peer or ""
                fitted = fit_messages(
                    row["messages"],
                    tokenizer,
                    max_seq_length=max_seq_length,
                    chat_template_args=chat_template_args,
                )
                if fitted is None:
                    skipped_overflow += 1
                    continue
                if not is_alternating(fitted):
                    skipped_illegal += 1
                    continue
                try:
                    if tokenizer is not None:
                        tokenizer.apply_chat_template(fitted, return_dict=False)
                except Exception:
                    skipped_illegal += 1
                    continue
                row["messages"] = fitted
                row["_sid"] = sid
                row["_query"] = fitted[-2]["content"]
                row["_reply"] = fitted[-1]["content"]
                examples.append(row)
                contextual_turns += 1
        if limit and len(examples) >= limit:
            examples = examples[:limit]
            break
    if not include_groups:
        skipped_groups = conn.execute(
            """SELECT count(*) AS n FROM (
                   SELECT chat_id FROM chat_handle_join
                   GROUP BY chat_id HAVING count(*) > 1
               )"""
        ).fetchone()["n"] or 0
    return examples, {
        "chats": chats,
        "sessions": sessions_n,
        "sent_turns": sent_turns,
        "sent_texts": sent_texts,
        "contextual_turns": contextual_turns,
        "opener_turns": opener_turns,
        "augmented": 0,
        "skipped_overflow": skipped_overflow,
        "skipped_illegal": skipped_illegal,
        "skipped_groups": skipped_groups,
        "skipped_media": skipped_media,
        "skipped_link": skipped_link,
        "skipped_pii": skipped_pii,
        "examples": len(examples),
    }


def dataset_profile(target_key=None, handles=None):
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
            if is_assistant_row(row, target_key):
                sent_total += 1
                if text:
                    sent_texts += 1
                else:
                    attachments_only += 1
            elif text:
                received_texts += 1
        if handles:
            qmarks = ",".join("?" * len(handles))
            chat_counts = conn.execute(
                f"""SELECT sum(n = 1) AS direct, sum(n > 1) AS groups FROM (
                       SELECT chj.chat_id, count(*) AS n
                       FROM chat_handle_join chj
                       JOIN handle h ON h.ROWID = chj.handle_id
                       WHERE chj.chat_id IN (
                           SELECT chj2.chat_id FROM chat_handle_join chj2
                           JOIN handle h2 ON h2.ROWID = chj2.handle_id
                           WHERE h2.id IN ({qmarks})
                       )
                       GROUP BY chj.chat_id
                   )""",
                handles,
            ).fetchone()
        else:
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


def export_dataset(
    limit=0,
    per_chat=0,
    out_dir=TWIN_DIR,
    augment=True,
    target_key=None,
    name=None,
    model_key=None,
    tokenizer=None,
    max_seq_length=MAX_SEQ_LENGTH,
    chat_template_args=None,
    include_groups=False,
    sender_labels=True,
    depth_ladder=(),
    openers=False,
    train_shots=True,
    recency_floor=RECENCY_FLOOR,
    low_signal_share=1.0,
    split_strategy="chrono",
):
    """Write MLX JSONL splits and return detailed coverage counts."""
    del augment
    if tokenizer is None and model_key:
        tokenizer = load_export_tokenizer(model_key, chat_template_args)
    conn = get_conn()
    try:
        examples, stats = collect_examples(
            conn,
            limit,
            per_chat,
            target_key=target_key,
            name=name,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            chat_template_args=chat_template_args,
            include_groups=include_groups,
            sender_labels=sender_labels,
            depth_ladder=depth_ladder,
            openers=openers,
        )
    finally:
        conn.close()
    if not examples:
        who = name if name and name != "You" else "you"
        raise RuntimeError(
            f"No text from {who} to export. Grant Full Disk Access if chat.db is blocked."
        )
    split = partition_examples(examples, strategy=split_strategy)
    raw_train = list(split["train"])
    concentration = target_concentration(raw_train)
    split["train"], capped = cap_duplicate_targets(split["train"])
    split["train"], recency_dropped = downsample_recency(
        split["train"], floor=recency_floor
    )
    split["train"], low_signal_dropped = cap_low_signal_share(
        split["train"], max_share=low_signal_share
    )
    # Shots are prepended to training prompts only. That erases the short
    # contexts the depth ladder creates, leaving train with almost no one-turn
    # rows while valid keeps thousands, and it disagrees with chat, where
    # retrieval measured harmful and is off.
    if train_shots:
        split["train"], train_with_shots = inject_train_shots(
            split["train"],
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            chat_template_args=chat_template_args,
        )
    else:
        train_with_shots = 0
    n_train, n_valid, n_test = write_dataset(
        examples, out_dir, train=split["train"], valid=split["valid"], test=split["test"]
    )
    bubble_pieces = bubble_token_count(tokenizer)
    meta = {
        **stats,
        "train": n_train,
        "valid": n_valid,
        "test": n_test,
        "reference": n_valid,
        "model_key": model_key or "",
        "max_seq_length": max_seq_length,
        "context_turns": CONTEXT_TURNS,
        "session_gap_hours": SESSION_GAP_NS / (60 * 60 * 1_000_000_000),
        "unique_targets": concentration["unique_targets"],
        "top20_share": concentration["top20_share"],
        "capped_targets": capped,
        "recency_dropped": recency_dropped,
        "recency_floor": recency_floor,
        "split_strategy": split_strategy,
        "low_signal_share": low_signal_share,
        "low_signal_dropped": low_signal_dropped,
        "train_with_shots": train_with_shots,
        "bubble_tokens": bubble_pieces,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "split.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def main():
    parser = argparse.ArgumentParser(description="Export sent iMessage text for LoRA")
    parser.add_argument("--limit", type=int, default=0, help="Max examples (0 = all)")
    parser.add_argument(
        "--per-chat", type=int, default=0, help="Recent messages per chat (0 = all)"
    )
    parser.add_argument("--no-augment", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--groups", action="store_true", help="Include group chats")
    parser.add_argument("--out", default=TWIN_DIR)
    parser.add_argument(
        "--person",
        default=ME,
        help="me, or a phone number / email to train as that contact",
    )
    parser.add_argument("--model-key", default="", help="Tokenizer to fit examples to")
    parser.add_argument(
        "--recency-floor",
        type=float,
        default=RECENCY_FLOOR,
        help="Lowest keep probability for an old train session. 1.0 keeps every session.",
    )
    parser.add_argument(
        "--depth-ladder",
        default="",
        help="Extra shallow context depths, e.g. 1,3,5. Empty keeps one copy per reply.",
    )
    parser.add_argument(
        "--no-train-shots",
        action="store_true",
        help="Do not prepend retrieved pairs to training prompts",
    )
    parser.add_argument(
        "--openers",
        action="store_true",
        help="Include conversation starters, which have no incoming message",
    )
    parser.add_argument(
        "--no-sender-labels",
        action="store_true",
        help="Do not prefix group turns with the sender name",
    )
    parser.add_argument(
        "--split",
        choices=("chrono", "stratified"),
        default="chrono",
        help="chrono holds out the newest sessions. stratified spreads the "
        "holdout over the whole history and sizes it by rows.",
    )
    parser.add_argument(
        "--low-signal-share",
        type=float,
        default=1.0,
        help="Cap ack replies at this share of the training rows. 1.0 keeps all.",
    )
    args = parser.parse_args()

    try:
        person_id = parse_person_arg(args.person)
        subject = resolve_subject(person_id)
    except ValueError as e:
        sys.exit(str(e))
    try:
        stats = export_dataset(
            args.limit,
            args.per_chat,
            args.out,
            target_key=subject["key"],
            name=subject["name"],
            model_key=args.model_key or None,
            include_groups=args.groups,
            sender_labels=not args.no_sender_labels,
            depth_ladder=tuple(int(x) for x in args.depth_ladder.split(",") if x.strip()),
            openers=args.openers,
            train_shots=not args.no_train_shots,
            recency_floor=args.recency_floor,
            low_signal_share=args.low_signal_share,
            split_strategy=args.split,
        )
    except RuntimeError as e:
        sys.exit(str(e))
    who = "you" if subject["id"] == ME else subject["name"]
    print(
        f"Wrote {stats['train']} train + {stats['valid']} valid + {stats['test']} test "
        f"covering {stats['sent_texts']} texts from {who} across {stats['chats']} "
        f"direct chats to {args.out}"
    )


if __name__ == "__main__":
    main()
