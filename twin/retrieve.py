"""Retrieve similar historical incoming-message/reply pairs from the train split."""

import json
import os
import re

from twin.export import CONTEXT_TURNS, TWIN_DIR, peer_from_system, recency_weight

TOKEN_RE = re.compile(r"[a-z0-9']+", re.I)
RETRIEVE_K = 2


def tokenize(text):
    return TOKEN_RE.findall((text or "").lower())


def _overlap(query_tokens, other_tokens):
    if not query_tokens or not other_tokens:
        return 0.0
    q = set(query_tokens)
    o = set(other_tokens)
    inter = len(q & o)
    if not inter:
        return 0.0
    return inter / ((len(q) ** 0.5) * (len(o) ** 0.5))


def index_row(query, reply, peer="", date=0):
    query = (query or "").strip()
    reply = (reply or "").strip()
    if not query or not reply:
        return None
    return {
        "query": query,
        "reply": reply,
        "peer": (peer or "").strip(),
        "date": date or 0,
        "tokens": tokenize(query),
    }


def index_from_examples(examples):
    """Build a retrieve index from export examples (``_query`` / ``_reply``)."""
    rows = []
    for example in examples or []:
        row = index_row(
            example.get("_query") or example.get("query"),
            example.get("_reply") or example.get("reply"),
            peer=example.get("_peer") or example.get("peer") or "",
            date=example.get("_date") or example.get("date") or 0,
        )
        if row:
            rows.append(row)
    return rows


def load_index(data_dir=TWIN_DIR):
    path = os.path.join(data_dir, "retrieve.jsonl")
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                item = index_row(
                    row.get("query"),
                    row.get("reply"),
                    peer=row.get("peer") or "",
                    date=row.get("date") or 0,
                )
                if item:
                    rows.append(item)
    except OSError:
        return []
    return rows


def last_user(messages):
    for message in reversed(messages or []):
        if message.get("role") == "user":
            return message.get("content") or ""
    return ""


def retrieve(query, index, k=RETRIEVE_K, exclude=None, peer=None, exclude_replies=None):
    """Return up to ``k`` train pairs whose incoming text is closest to ``query``.

    Recency and a matching recipient raise the score so old or off-thread
    pairs do not crowd out current style.
    """
    tokens = tokenize(query)
    exclude = {item.strip() for item in (exclude or []) if item}
    exclude_replies = {item.strip() for item in (exclude_replies or []) if item}
    now = max((row.get("date") or 0) for row in index) if index else 0
    want = (peer or "").strip()
    scored = []
    for row in index:
        if row["query"] in exclude or row["query"] == (query or "").strip():
            continue
        if row["reply"] in exclude_replies:
            continue
        overlap = _overlap(tokens, row["tokens"])
        if overlap <= 0:
            continue
        recency = recency_weight(row.get("date") or 0, now)
        boost = 1.2 if want and row.get("peer") == want else 1.0
        score = overlap * (0.5 + 0.5 * recency) * boost
        scored.append((score, row))
    scored.sort(key=lambda item: (-item[0], item[1]["query"]))
    return [
        {"query": row["query"], "reply": row["reply"]}
        for _, row in scored[: max(0, k)]
    ]


def few_shot_messages(pairs):
    """Turn retrieved pairs into alternating user/assistant turns."""
    messages = []
    for pair in pairs:
        messages.append({"role": "user", "content": pair["query"]})
        messages.append({"role": "assistant", "content": pair["reply"]})
    return messages


def with_retrieved_shots(
    messages,
    index=None,
    k=RETRIEVE_K,
    exclude=None,
    exclude_replies=None,
    peer=None,
    pairs=None,
    context_turns=None,
):
    """Prepend retrieved pairs, then keep live turns inside the remaining budget.

    Training, eval, and chat share this shape so the adapter sees retrieval
    at train time the same way chat injects it.
    """
    messages = [dict(m) for m in (messages or [])]
    system = [messages[0]] if messages and messages[0].get("role") == "system" else []
    rest = messages[len(system) :]
    turns = CONTEXT_TURNS if context_turns is None else context_turns
    if peer is None and system:
        peer = peer_from_system(system[0].get("content") or "")
    if pairs is None:
        query = last_user(rest)
        pairs = (
            retrieve(
                query,
                index or [],
                k=k,
                exclude=exclude or [query],
                exclude_replies=exclude_replies,
                peer=peer,
            )
            if index and k
            else []
        )
    shots = few_shot_messages(pairs or [])
    budget = max(2, turns - len(shots)) if shots else turns
    rest = rest[-budget:]
    while rest and rest[0].get("role") != "user":
        rest = rest[1:]
    return system + shots + rest
