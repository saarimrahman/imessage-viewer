"""Word stats from your sent messages. Cached in .cache/voice.json and rebuilt
only when the search index has new messages."""

import json
import math
import os
import re
import sqlite3
import time
from collections import Counter, defaultdict

from config import NLTK_DATA, SEARCH_DB, VOICE_CACHE, VOICE_SCHEMA
from contacts import resolve_contact

TOKEN = re.compile(r"[a-z][a-z']{1,}")
URL = re.compile(r"https?://\S+|www\.\S+")
PHRASE_SKIP = frozenset("to of a an the and is it its we as in on at".split())
# Unpunctuated forms NLTK's list does not cover.
SMS_STOP = frozenset(
    "im ive id ill youre thats dont doesnt didnt cant wont theyre weve youve "
    "gonna wanna gotta ur".split()
)
MIN_PERSON_MSGS = 80
TOP_WORDS = 80
TOP_PHRASES = 20
MAX_PHRASE_N = 6
MIN_PHRASE_COUNT = 3
TOP_PEOPLE = 12
DISTINCTIVE_PER_PERSON = 12


def _stopwords():
    os.makedirs(NLTK_DATA, exist_ok=True)
    try:
        import nltk
    except ImportError as e:
        raise RuntimeError(
            "nltk is required for word stats. Run: pip3 install -r requirements.txt"
        ) from e
    if NLTK_DATA not in nltk.data.path:
        nltk.data.path.insert(0, NLTK_DATA)
    try:
        words = set(nltk.corpus.stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", download_dir=NLTK_DATA, quiet=True)
        words = set(nltk.corpus.stopwords.words("english"))
    words.update(SMS_STOP)
    return words


def _tokenize(text):
    text = URL.sub(" ", (text or "").lower().replace("\u2019", "'"))
    return [t.replace("'", "") for t in TOKEN.findall(text) if t.replace("'", "")]


def phrase_counts(tokens, min_n=2, max_n=MAX_PHRASE_N):
    """Count n-grams from min_n to max_n. Skip phrases that start or end on filler."""
    counts = Counter()
    n = len(tokens)
    hi = min(max_n, n)
    for size in range(min_n, hi + 1):
        for start in range(n - size + 1):
            phrase = tokens[start : start + size]
            if phrase[0] in PHRASE_SKIP or phrase[-1] in PHRASE_SKIP:
                continue
            if any(len(token) < 2 for token in phrase):
                continue
            counts[" ".join(phrase)] += 1
    return counts


def _token_subspan(short, long_tokens):
    n = len(short)
    if n >= len(long_tokens):
        return False
    for i in range(len(long_tokens) - n + 1):
        if long_tokens[i : i + n] == short:
            return True
    return False


def frequent_phrases(counts, limit=TOP_PHRASES, min_count=MIN_PHRASE_COUNT):
    """Rank frequent phrases of any length. Drop a short phrase when longer ones cover it."""
    items = []
    for phrase, n in counts.items():
        if n < min_count:
            continue
        items.append((phrase.split(), phrase, n))
    items.sort(key=lambda x: (-len(x[0]), -x[2]))
    kept = []
    for toks, phrase, n in items:
        explained = sum(kn for kt, _, kn in kept if _token_subspan(toks, kt))
        if n - explained < min_count:
            continue
        kept.append((toks, phrase, n))
    kept.sort(key=lambda x: (-x[2] * len(x[0]), -x[2], x[1]))
    return [{"phrase": p, "n": n} for _, p, n in kept[:limit]]


def _name_tokens(name):
    return {t for t in re.findall(r"[a-z]+", (name or "").lower()) if len(t) > 2}


def _search_last_msg_id():
    if not os.path.exists(SEARCH_DB):
        return 0
    conn = sqlite3.connect(f"file:{SEARCH_DB}?mode=ro", uri=True)
    try:
        row = conn.execute("SELECT v FROM meta WHERE k='last_msg_id'").fetchone()
        return int(row[0]) if row else 0
    except sqlite3.Error:
        return 0
    finally:
        conn.close()


def load_voice():
    """Return cached word stats, or None if they have not been built yet."""
    try:
        with open(VOICE_CACHE) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("schema") != VOICE_SCHEMA:
        return None
    return data


def _cache_is_current(last_msg_id):
    data = load_voice()
    return bool(data) and data.get("last_msg_id") == last_msg_id


def _distinctive(person_counts, person_total, overall, n_tokens, skip):
    scored = []
    for w, n in person_counts.items():
        if n < 6 or w in skip:
            continue
        g = overall[w]
        if not g:
            continue
        lift = (n / person_total) / (g / n_tokens)
        if lift < 2.0:
            continue
        scored.append((n * math.log(lift), lift, n, w))
    scored.sort(reverse=True)
    return [{"word": w, "n": n, "lift": round(lift, 1)} for _, lift, n, w in scored[:DISTINCTIVE_PER_PERSON]]


def build_voice_stats(force=False, verbose=False):
    """Walk sent messages in search.db. No-op when the cache matches last_msg_id."""
    last_msg_id = _search_last_msg_id()
    if not last_msg_id:
        if verbose:
            print("Word stats: search index is empty")
        return None
    if not force and _cache_is_current(last_msg_id):
        if verbose:
            print("Word stats: up to date")
        return load_voice()

    stop = _stopwords()
    t0 = time.time()
    conn = sqlite3.connect(f"file:{SEARCH_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    overall = Counter()
    ngrams = Counter()
    per = defaultdict(Counter)
    per_n = Counter()
    n_tokens = 0
    n_msgs = 0
    len_sum = 0
    try:
        for row in conn.execute(
            "SELECT handle, body FROM docs WHERE is_from_me=1"
        ):
            body = row["body"] or ""
            toks = _tokenize(body)
            n_msgs += 1
            len_sum += len(body)
            content = [t for t in toks if t not in stop and len(t) >= 2]
            overall.update(content)
            n_tokens += len(content)
            ngrams.update(phrase_counts(toks))
            handle = row["handle"] or ""
            if handle:
                per[handle].update(content)
                per_n[handle] += 1
    finally:
        conn.close()

    if not n_msgs:
        if verbose:
            print("Word stats: no sent messages")
        return None

    words = [{"word": w, "n": n} for w, n in overall.most_common(TOP_WORDS)]
    phrases = frequent_phrases(ngrams)

    people = []
    for handle, msgs in per_n.most_common():
        if msgs < MIN_PERSON_MSGS:
            continue
        name = resolve_contact(handle) or handle
        tot = sum(per[handle].values()) or 1
        people.append(
            {
                "handle": handle,
                "name": name,
                "msgs": msgs,
                "words": _distinctive(per[handle], tot, overall, n_tokens, _name_tokens(name)),
            }
        )
        if len(people) >= TOP_PEOPLE:
            break

    data = {
        "schema": VOICE_SCHEMA,
        "last_msg_id": last_msg_id,
        "msgs": n_msgs,
        "avg_len": round(len_sum / n_msgs) if n_msgs else 0,
        "words": words,
        "phrases": phrases,
        "people": people,
    }
    tmp = VOICE_CACHE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, VOICE_CACHE)
    print(
        f"Word stats: {n_msgs} sent texts, {len(words)} words, "
        f"{len(people)} people in {time.time() - t0:.1f}s"
    )
    return data
