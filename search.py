"""FTS5 search index over message text, stored in .cache/search.db."""

import html
import os
import re
import sqlite3
import threading
import time

from config import SEARCH_DB, SEARCH_LIMIT, SEARCH_SCHEMA, VOICE_CACHE
from contacts import chat_label, load_participants, resolve_contact
from db import REACTION_EXCLUDE_SQL, DbUnavailable, get_conn, message_text

_index_lock = threading.Lock()
_index_ready = False
_index_building = False
_index_error = None


def parse_search_query(raw):
    """Split a query into exact phrases (quoted) and loose terms (everything else)."""
    phrases = []
    terms = []
    for m in re.finditer(r'"([^"]*)"|(\S+)', (raw or "").strip()):
        if m.group(1) is not None:
            phrase = m.group(1).strip()
            if phrase:
                phrases.append(phrase)
        else:
            term = m.group(2).strip()
            if term:
                terms.append(term)
    return phrases, terms


def _fts_term(term):
    cleaned = re.sub(r"[^\w'-]+", "", term, flags=re.UNICODE).strip("-'")
    if len(cleaned) < 2:
        return None
    if cleaned.lower() in ("and", "or", "not", "near"):
        return f'"{cleaned}"'
    return f"{cleaned}*"


def to_fts_query(phrases, terms, combiner="AND"):
    parts = [f'"{p.replace(chr(34), chr(34)+chr(34))}"' for p in phrases]
    parts.extend(tok for tok in (_fts_term(t) for t in terms) if tok)
    return f" {combiner} ".join(parts)


def highlight_match(text, phrases, terms):
    needles = [p for p in phrases if p] + [t for t in terms if len(t) > 1]
    needles.sort(key=len, reverse=True)
    if not needles:
        return html.escape(text)
    pattern = "|".join(re.escape(n) for n in needles)
    out = []
    last = 0
    for m in re.finditer(pattern, text, re.I):
        out.append(html.escape(text[last : m.start()]))
        out.append("<mark>" + html.escape(m.group(0)) + "</mark>")
        last = m.end()
    out.append(html.escape(text[last:]))
    return "".join(out)


def snippet_html(text, phrases, terms, limit=220):
    needles = [p for p in phrases if p] + [t for t in terms if t]
    lower = text.lower()
    pos = -1
    for n in needles:
        i = lower.find(n.lower())
        if i != -1 and (pos == -1 or i < pos):
            pos = i
    if pos == -1:
        cut = text[:limit]
        return highlight_match(cut, phrases, terms) + ("…" if len(text) > limit else "")
    start = max(0, pos - 50)
    end = min(len(text), start + limit)
    prefix = "…" if start else ""
    suffix = "…" if end < len(text) else ""
    return prefix + highlight_match(text[start:end], phrases, terms) + suffix


def _search_meta(conn, key, default=None):
    row = conn.execute("SELECT v FROM meta WHERE k=?", (key,)).fetchone()
    return row["v"] if row else default


def _search_set_meta(conn, key, value):
    conn.execute("INSERT OR REPLACE INTO meta(k, v) VALUES (?, ?)", (key, str(value)))


def _init_search_schema(conn):
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
        CREATE TABLE IF NOT EXISTS docs (
          id INTEGER PRIMARY KEY,
          msg_id INTEGER NOT NULL,
          chat_id INTEGER NOT NULL,
          date INTEGER,
          is_from_me INTEGER,
          handle TEXT,
          body TEXT NOT NULL,
          sender TEXT,
          UNIQUE(msg_id, chat_id)
        );
        CREATE INDEX IF NOT EXISTS docs_chat ON docs(chat_id);
        CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
          body, sender,
          content='docs',
          content_rowid='id',
          tokenize='porter unicode61 remove_diacritics 2'
        );
        """
    )


def _rebuild_search_schema(conn):
    conn.executescript("DROP TABLE IF EXISTS docs_fts; DROP TABLE IF EXISTS docs; DROP TABLE IF EXISTS meta;")
    _init_search_schema(conn)


def _open_search_rw():
    conn = sqlite3.connect(SEARCH_DB)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


def _build_search_index(verbose=False):
    """Append new messages into .cache/search.db. First run walks the whole
    history once; later runs only index ROWIDs we have not seen."""
    try:
        chat_conn = get_conn()
    except DbUnavailable:
        if os.path.exists(SEARCH_DB):
            if verbose:
                print("Search index: using cached search.db (live chat.db not readable)")
            return
        raise
    search_conn = _open_search_rw()
    try:
        _init_search_schema(search_conn)
        if _search_meta(search_conn, "schema") != SEARCH_SCHEMA:
            _rebuild_search_schema(search_conn)
            _search_set_meta(search_conn, "schema", SEARCH_SCHEMA)

        src_max = chat_conn.execute("SELECT max(ROWID) FROM message").fetchone()[0] or 0
        last_id = int(_search_meta(search_conn, "last_msg_id", "0"))
        if src_max < last_id:
            _rebuild_search_schema(search_conn)
            _search_set_meta(search_conn, "schema", SEARCH_SCHEMA)
            last_id = 0
        if src_max <= last_id:
            if verbose:
                print("Search index: up to date")
            return

        t0 = time.time()
        indexed = 0
        batch = []
        cur = chat_conn.execute(
            f"""SELECT m.ROWID as id, m.text,
                       CASE WHEN m.text IS NULL OR trim(m.text) = '' THEN m.attributedBody ELSE NULL END
                         as attributedBody,
                       m.date, m.is_from_me, h.id as handle, cmj.chat_id as chat_id
                FROM message m
                JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
                LEFT JOIN handle h ON h.ROWID = m.handle_id
                WHERE m.ROWID > ? AND {REACTION_EXCLUDE_SQL}
                ORDER BY m.ROWID""",
            (last_id,),
        )

        def flush(rows):
            if not rows:
                return
            before = search_conn.execute("SELECT ifnull(max(id), 0) FROM docs").fetchone()[0]
            search_conn.executemany(
                "INSERT OR IGNORE INTO docs(msg_id, chat_id, date, is_from_me, handle, body, sender) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            search_conn.execute(
                "INSERT INTO docs_fts(rowid, body, sender) "
                "SELECT id, body, sender FROM docs WHERE id > ?",
                (before,),
            )

        max_seen = last_id
        for row in cur:
            max_seen = row["id"]
            body = message_text(row)
            if not body:
                continue
            sender = ""
            if not row["is_from_me"]:
                sender = resolve_contact(row["handle"]) or row["handle"] or ""
            batch.append(
                (row["id"], row["chat_id"], row["date"], row["is_from_me"], row["handle"], body, sender)
            )
            if len(batch) >= 2000:
                flush(batch)
                indexed += len(batch)
                batch = []
                search_conn.commit()
        flush(batch)
        indexed += len(batch)
        _search_set_meta(search_conn, "last_msg_id", max_seen)
        search_conn.commit()
        print(f"Search index: +{indexed} messages in {time.time() - t0:.1f}s (through ROWID {max_seen})")
    finally:
        search_conn.close()
        chat_conn.close()


def _wipe_indexes():
    for path in (SEARCH_DB, SEARCH_DB + "-wal", SEARCH_DB + "-shm", VOICE_CACHE):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def _index_worker(force=False):
    global _index_ready, _index_building, _index_error
    with _index_lock:
        try:
            if force:
                _wipe_indexes()
            _build_search_index(verbose=True)
            _index_ready = True
            _index_error = None
            from voice import build_voice_stats

            build_voice_stats(force=force, verbose=True)
        except Exception as e:
            _index_error = str(e)
            print(f"Index failed: {e}")
        finally:
            _index_building = False


def is_index_ready():
    return _index_ready


def index_error():
    return _index_error


def kick_search_index(force=False):
    """Build search + word stats in the background if they are not already current."""
    global _index_building
    if _index_building:
        return
    if not force and _index_ready:
        return
    _index_building = True
    threading.Thread(target=_index_worker, args=(force,), daemon=True).start()


def ensure_search_index():
    """Block until the search index exists. Cheap no-op when already up to date."""
    global _index_ready, _index_error
    with _index_lock:
        _build_search_index()
        _index_ready = True
        _index_error = None


def ensure_indexes(force=False):
    """Build search index and word stats. No-op when both are current unless force."""
    global _index_ready, _index_building, _index_error
    with _index_lock:
        _index_building = True
        try:
            if force:
                _wipe_indexes()
            _build_search_index(verbose=True)
            _index_ready = True
            _index_error = None
            from voice import build_voice_stats

            build_voice_stats(force=force, verbose=True)
        except Exception as e:
            _index_error = str(e)
            print(f"Index failed: {e}")
            raise
        finally:
            _index_building = False


def _search_match(conn, match, chat_id, limit):
    sql = """SELECT d.msg_id, d.chat_id, d.date, d.is_from_me, d.handle, d.body
             FROM docs_fts f JOIN docs d ON d.id = f.rowid
             WHERE f MATCH ?"""
    params = [match]
    if chat_id is not None:
        sql += " AND d.chat_id = ?"
        params.append(chat_id)
    sql += " ORDER BY rank LIMIT ?"
    params.append(limit)
    return conn.execute(sql, params).fetchall()


def search_messages(query, chat_id=None, limit=SEARCH_LIMIT):
    phrases, terms = parse_search_query(query)
    match_and = to_fts_query(phrases, terms, "AND")
    if not match_and:
        return [], phrases, terms, False
    ensure_search_index()
    conn = sqlite3.connect(f"file:{SEARCH_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    broadened = False
    try:
        rows = _search_match(conn, match_and, chat_id, limit + 1)
        if not rows:
            match_or = to_fts_query(phrases, terms, "OR")
            if match_or and match_or != match_and:
                rows = _search_match(conn, match_or, chat_id, limit + 1)
                broadened = bool(rows)
    except sqlite3.OperationalError:
        rows = []
        broadened = False
    finally:
        conn.close()
    return rows, phrases, terms, broadened


def matching_conversations(query, limit=8):
    needle = re.sub(r'"', "", query or "").strip().lower()
    if len(needle) < 2:
        return []
    conn = get_conn()
    rows = conn.execute(
        "SELECT ROWID as id, chat_identifier, display_name FROM chat"
    ).fetchall()
    need = [r["id"] for r in rows if not r["display_name"] and not resolve_contact(r["chat_identifier"])]
    pmap = load_participants(conn, need)
    conn.close()
    hits = []
    for r in rows:
        name = chat_label(r["display_name"], r["chat_identifier"], pmap.get(r["id"]))
        if needle in name.lower():
            hits.append((r["id"], name, r["chat_identifier"]))
    hits.sort(key=lambda x: x[1].lower())
    return hits[:limit]
