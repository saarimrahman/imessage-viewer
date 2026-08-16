"""Retrieve similar historical incoming-message/reply pairs from the train split."""

import json
import os
import re

from twin.export import TWIN_DIR, recency_weight

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


def load_index(data_dir=TWIN_DIR):
    path = os.path.join(data_dir, "retrieve.jsonl")
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                query = (row.get("query") or "").strip()
                reply = (row.get("reply") or "").strip()
                if query and reply:
                    rows.append(
                        {
                            "query": query,
                            "reply": reply,
                            "peer": (row.get("peer") or "").strip(),
                            "date": row.get("date") or 0,
                            "tokens": tokenize(query),
                        }
                    )
    except OSError:
        return []
    return rows


def retrieve(query, index, k=RETRIEVE_K, exclude=None, peer=None):
    """Return up to ``k`` train pairs whose incoming text is closest to ``query``.

    Recency and a matching recipient raise the score so old or off-thread
    pairs do not crowd out current style.
    """
    tokens = tokenize(query)
    exclude = {item.strip() for item in (exclude or []) if item}
    now = max((row.get("date") or 0) for row in index) if index else 0
    want = (peer or "").strip()
    scored = []
    for row in index:
        if row["query"] in exclude or row["query"] == (query or "").strip():
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
