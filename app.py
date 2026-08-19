#!/usr/bin/env python3
"""Local read-only viewer for macOS Messages. First time:

  python3 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  python3 app.py --index
  python3 app.py

Then open http://127.0.0.1:8765/

  python3 app.py              start the server (builds indexes if needed)
  python3 app.py --index      build search + word stats and exit
  python3 app.py --rebuild    wipe indexes, rebuild, then start the server
  python3 app.py --rebuild --index
                              wipe, rebuild, and exit
"""

import argparse
import hashlib
import json
import mimetypes
import os
import shutil
import socket
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from config import (
    CACHE_CONTROL,
    CACHE_DIR,
    DB_PATH,
    PAGE_SIZE,
    PORT,
    SCRIPT_DIR,
    START_NEWEST,
    START_OLDEST,
    THUMB_SIZE,
    save_pref,
)
from contacts import AVATAR_INDEX, CONTACTS
from db import (
    DbUnavailable,
    apple_date,
    fetch_messages,
    get_conn,
    load_attachments,
    load_reactions,
    sender_context,
)
from render import (
    render_all_media,
    render_chat,
    render_chat_heatmap,
    render_chat_list,
    render_db_error,
    render_media,
    render_media_more,
    render_message_blocks,
    render_search,
    render_stats,
    render_twin,
    render_circles,
    is_time_break,
)
from twin.job import (
    TwinError,
    chat as twin_chat,
    inspect_data as twin_inspect_data,
    list_people as twin_list_people,
    snapshot as twin_snapshot,
    start_train,
    stop_train,
)
from search import ensure_indexes, kick_search_index

STATIC_TYPES = {
    "app.css": "text/css; charset=utf-8",
    "app.js": "application/javascript; charset=utf-8",
}


def convert_heic(path):
    """HEIC/HEIF only render natively in Safari. Convert to JPEG on demand
    and cache the result so other browsers can display it."""
    digest = hashlib.sha1(path.encode()).hexdigest()
    out_path = os.path.join(CACHE_DIR, digest + ".jpg")
    if not os.path.exists(out_path):
        result = subprocess.run(
            ["sips", "-s", "format", "jpeg", path, "--out", out_path],
            capture_output=True,
        )
        if result.returncode != 0 or not os.path.exists(out_path):
            return None
    return out_path


def make_thumb(path, dest=None):
    """Max-edge 512px JPEG for chat bubbles and the media grid. Lightbox still
    uses the original via /attachment."""
    out_path = dest or os.path.join(
        CACHE_DIR, hashlib.sha1(f"thumb{THUMB_SIZE}:{path}".encode()).hexdigest() + ".jpg"
    )
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path
    result = subprocess.run(
        ["sips", "-Z", str(THUMB_SIZE), "-s", "format", "jpeg", path, "--out", out_path],
        capture_output=True,
    )
    if result.returncode != 0 or not os.path.exists(out_path):
        return None
    return out_path


def is_heic(path, mime):
    return mime in ("image/heic", "image/heif") or path.lower().endswith((".heic", ".heif"))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        try:
            self._dispatch()
        except DbUnavailable as e:
            self._send_html(render_db_error(str(e)), status=503)

    def do_POST(self):
        try:
            self._dispatch_post()
        except DbUnavailable as e:
            self._send_json({"error": str(e)}, status=503)
        except json.JSONDecodeError:
            self._send_json({"error": "bad json"}, status=400)

    def _read_json(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        if n < 0 or n > 100_000:
            raise json.JSONDecodeError("too large", "", 0)
        raw = self.rfile.read(n) if n else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _dispatch_post(self):
        parsed = urlparse(self.path)
        parts = [p for p in parsed.path.split("/") if p]
        body = self._read_json()
        if not isinstance(body, dict):
            self._send_json({"error": "bad json"}, status=400)
            return
        if parts == ["twin", "train"]:
            run = body.get("run") or "complete"
            model = body.get("model") or "balanced"
            iters = body.get("iters")
            if iters in ("", None):
                iters = None
            ok, err = start_train(
                run=run,
                model_key=model,
                person_id=body.get("person") or "me",
                iters=iters,
                resume_from=body.get("resume") or None,
            )
            if not ok:
                self._send_json({"error": err, **twin_snapshot()}, status=409)
                return
            self._send_json(twin_snapshot())
        elif parts == ["twin", "stop"]:
            ok, err = stop_train()
            if not ok:
                self._send_json({"error": err, **twin_snapshot()}, status=409)
                return
            self._send_json(twin_snapshot())
        elif parts == ["twin", "chat"]:
            history = body.get("history") or []
            if not isinstance(history, list):
                history = []
            try:
                reply = twin_chat(
                    body.get("text") or "",
                    history,
                    model_key=body.get("model") or "balanced",
                    person_id=body.get("person") or "me",
                    adapter=body.get("adapter") or None,
                    to=body.get("to") or None,
                )
            except TwinError as e:
                self._send_json({"error": str(e)}, status=409)
                return
            self._send_json({"reply": reply})
        else:
            self._send_error(404)

    def _dispatch(self):
        parsed = urlparse(self.path)
        parts = [p for p in parsed.path.split("/") if p]
        qs = parse_qs(parsed.query)

        if not parts:
            start = qs.get("start", [None])[0]
            if start in (START_OLDEST, START_NEWEST):
                save_pref("start", start)
            sort = qs.get("sort", ["recent"])[0]
            self._send_html(render_chat_list(sort))
        elif parts[0] == "static" and len(parts) == 2:
            self._serve_static(parts[1])
        elif parts[0] == "stats" and len(parts) == 1:
            self._send_html(render_stats())
        elif parts[0] == "circles" and len(parts) == 1:
            self._send_html(render_circles())
        elif parts[0] == "twin" and len(parts) == 1:
            self._send_html(render_twin())
        elif parts[0] == "twin" and len(parts) == 2 and parts[1] == "status":
            brief = qs.get("brief", [""])[0] in ("1", "true")
            self._send_json(twin_snapshot(brief=brief))
        elif parts[0] == "twin" and len(parts) == 2 and parts[1] == "data":
            try:
                self._send_json(twin_inspect_data(qs.get("person", ["me"])[0]))
            except TwinError as e:
                self._send_json({"error": str(e)}, status=400)
        elif parts[0] == "twin" and len(parts) == 2 and parts[1] == "people":
            self._send_json({"people": twin_list_people()})
        elif parts[0] == "search" and len(parts) == 1:
            query = qs.get("q", [None])[0]
            self._send_html(render_search(query))
        elif parts[0] == "media" and len(parts) == 1:
            self._send_html(render_all_media(qs.get("date", [None])[0]))
        elif parts[0] == "media" and len(parts) == 2 and parts[1] == "more":
            self._serve_media_more(None, qs)
        elif parts[0] == "chat" and len(parts) == 2:
            try:
                chat_id = int(parts[1])
            except ValueError:
                self._send_error(404)
                return
            date_str = qs.get("date", [None])[0]
            around = qs.get("around", [None])[0]
            around_id = int(around) if around and around.isdigit() else None
            out = render_chat(chat_id, date_str, around_id)
            self._send_html(out) if out is not None else self._send_error(404)
        elif parts[0] == "chat" and len(parts) == 3 and parts[2] == "heatmap":
            try:
                chat_id = int(parts[1])
            except ValueError:
                self._send_error(404)
                return
            out = render_chat_heatmap(chat_id)
            if out is None:
                self._send_error(404)
                return
            self._send_html(out)
        elif parts[0] == "chat" and len(parts) == 3 and parts[2] == "more":
            self._serve_more(parts[1], qs)
        elif parts[0] == "chat" and len(parts) == 3 and parts[2] == "media":
            try:
                chat_id = int(parts[1])
            except ValueError:
                self._send_error(404)
                return
            out = render_media(chat_id, qs.get("date", [None])[0])
            self._send_html(out) if out is not None else self._send_error(404)
        elif parts[0] == "chat" and len(parts) == 4 and parts[2] == "media" and parts[3] == "more":
            try:
                chat_id = int(parts[1])
            except ValueError:
                self._send_error(404)
                return
            self._serve_media_more(chat_id, qs)
        elif parts[0] == "chat" and len(parts) == 3 and parts[2] == "search":
            try:
                chat_id = int(parts[1])
            except ValueError:
                self._send_error(404)
                return
            query = qs.get("q", [None])[0]
            out = render_search(query, chat_id)
            self._send_html(out) if out is not None else self._send_error(404)
        elif parts[0] == "attachment" and len(parts) == 2:
            self._serve_attachment(parts[1])
        elif parts[0] == "thumb" and len(parts) == 2:
            self._serve_thumb(parts[1])
        elif parts[0] == "avatar" and len(parts) == 2:
            self._serve_avatar(parts[1])
        else:
            self._send_error(404)

    def _serve_more(self, chat_id, qs):
        try:
            chat_id = int(chat_id)
        except ValueError:
            self._send_error(404)
            return

        def take(key):
            raw = qs.get(key, [None])[0]
            if raw in (None, ""):
                return None
            try:
                return int(raw)
            except ValueError:
                return None

        after_date, after_id = take("after_date"), take("after_id")
        before_date, before_id = take("before_date"), take("before_id")
        conn = get_conn()
        prev_day = prev_sender = prev_date = next_day = next_sender = next_date = None
        strip_group_start = False

        if before_date is not None and before_id is not None:
            rows = fetch_messages(conn, chat_id, before=(before_date, before_id), limit=PAGE_SIZE)
            next_day, next_sender, next_date = sender_context(conn, before_id)
            has_more_older = len(rows) == PAGE_SIZE
            has_more_newer = True
            if rows and next_sender is not None:
                last = rows[-1]
                strip_group_start = (
                    apple_date(last["date"])[:10] == next_day
                    and (last["is_from_me"], last["handle"]) == next_sender
                    and not is_time_break(last["date"], next_date)
                )
        elif after_date is not None and after_id is not None:
            rows = fetch_messages(conn, chat_id, after=(after_date, after_id), limit=PAGE_SIZE)
            prev_day, prev_sender, prev_date = sender_context(conn, after_id)
            has_more_newer = len(rows) == PAGE_SIZE
            has_more_older = True
        else:
            conn.close()
            self._send_error(404)
            return

        ids = [r["id"] for r in rows]
        att_by_msg = load_attachments(conn, ids)
        reactions_by_guid = load_reactions(conn, chat_id, [r["guid"] for r in rows])
        blocks_html = render_message_blocks(
            rows,
            att_by_msg,
            reactions_by_guid,
            prev_day=prev_day,
            prev_sender=prev_sender,
            prev_date=prev_date,
            next_day=next_day,
            next_sender=next_sender,
            next_date=next_date,
        )
        conn.close()
        self._send_json(
            {
                "html": blocks_html,
                "first_id": rows[0]["id"] if rows else 0,
                "first_date": str(rows[0]["date"]) if rows else "0",
                "last_id": rows[-1]["id"] if rows else 0,
                "last_date": str(rows[-1]["date"]) if rows else "0",
                "has_more_older": has_more_older,
                "has_more_newer": has_more_newer,
                "strip_group_start": strip_group_start,
            }
        )

    def _serve_media_more(self, chat_id, qs):
        def take(key):
            raw = qs.get(key, [None])[0]
            if raw in (None, ""):
                return None
            try:
                return int(raw)
            except ValueError:
                return None

        after = before = None
        after_date, after_id = take("after_date"), take("after_id")
        before_date, before_id = take("before_date"), take("before_id")
        if before_date is not None and before_id is not None:
            before = (before_date, before_id)
        elif after_date is not None and after_id is not None:
            after = (after_date, after_id)
        else:
            self._send_error(404)
            return
        out = render_media_more(chat_id, after=after, before=before)
        if out is None:
            self._send_error(404)
            return
        self._send_json(out)

    def _send_html(self, body, status=200):
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, obj, status=200):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, code):
        self.send_response(code)
        self.end_headers()

    def _serve_static(self, name):
        mime = STATIC_TYPES.get(name)
        if not mime:
            self._send_error(404)
            return
        self._send_file(os.path.join(SCRIPT_DIR, "static", name), mime, cache="no-cache")

    def _send_file(self, path, mime, cache=CACHE_CONTROL):
        try:
            size = os.path.getsize(path)
            handle = open(path, "rb")
        except OSError:
            self._send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        try:
            shutil.copyfileobj(handle, self.wfile, 65536)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        finally:
            handle.close()

    def _attachment_path(self, att_id):
        try:
            att_id = int(att_id)
        except ValueError:
            return None
        conn = get_conn()
        row = conn.execute(
            "SELECT filename, mime_type FROM attachment WHERE ROWID=?", (att_id,)
        ).fetchone()
        conn.close()
        if not row or not row["filename"]:
            return None
        path = os.path.expanduser(row["filename"])
        mime = row["mime_type"] or mimetypes.guess_type(path)[0] or "application/octet-stream"
        return path, mime

    def _serve_avatar(self, avatar_id):
        entry = AVATAR_INDEX.get(avatar_id)
        if not entry:
            self._send_error(404)
            return
        mime, data = entry
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", CACHE_CONTROL)
        self.end_headers()
        self.wfile.write(data)

    def _serve_thumb(self, att_id):
        try:
            att_id_int = int(att_id)
        except ValueError:
            self._send_error(404)
            return
        cached = os.path.join(CACHE_DIR, f"thumb{THUMB_SIZE}-{att_id_int}.jpg")
        if os.path.exists(cached) and os.path.getsize(cached) > 0:
            self._send_file(cached, "image/jpeg")
            return
        info = self._attachment_path(att_id)
        if not info:
            self._send_error(404)
            return
        path, mime = info
        if not os.path.exists(path):
            self._send_error(404)
            return
        thumb = (
            make_thumb(path, cached)
            if mime.startswith("image/") or is_heic(path, mime)
            else None
        )
        if thumb:
            self._send_file(thumb, "image/jpeg")
            return
        if is_heic(path, mime):
            converted = convert_heic(path)
            if converted:
                self._send_file(converted, "image/jpeg")
                return
        self._send_file(path, mime)

    def _serve_attachment(self, att_id):
        info = self._attachment_path(att_id)
        if not info:
            self._send_error(404)
            return
        path, mime = info
        if is_heic(path, mime):
            converted = convert_heic(path)
            if converted:
                path, mime = converted, "image/jpeg"
        if not os.path.exists(path):
            self._send_error(404)
            return
        self._send_file(path, mime)


def serve(rebuild=False):
    print(f"Loaded {len(CONTACTS)} contact lookup entries")
    try:
        conn = get_conn()
        n = conn.execute("SELECT count(*) FROM chat").fetchone()[0]
        conn.close()
        print(f"Opened {DB_PATH} ({n} conversations)")
    except DbUnavailable as e:
        print(e)
    kick_search_index(force=rebuild)
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    lan_ip = socket.gethostbyname(socket.gethostname())
    print(f"Serving on http://127.0.0.1:{PORT}/ and http://{lan_ip}:{PORT}/")
    server.serve_forever()


def _file_mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def _stop(proc):
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _watch_mtime():
    here = os.path.dirname(os.path.abspath(__file__))
    roots = [here, os.path.join(here, "twin")]
    files = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        files.extend(
            os.path.join(root, name)
            for name in sorted(os.listdir(root))
            if name.endswith(".py")
        )
    return tuple(_file_mtime(path) for path in files)


def reloader():
    """Restart the server process when any .py file in this folder changes."""
    here = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.copy()
    env["_IMESSAGE_RELOADER"] = "1"
    cmd = [sys.executable, os.path.abspath(__file__), *sys.argv[1:]]
    proc = None
    print(f"Watching {here}/*.py — save to reload")
    try:
        while True:
            proc = subprocess.Popen(cmd, env=env)
            mtime = _watch_mtime()
            while proc.poll() is None:
                time.sleep(0.4)
                now = _watch_mtime()
                if now != mtime:
                    print("Reloading...")
                    _stop(proc)
                    break
            else:
                if proc.returncode == 0:
                    return
                print(f"Server exited ({proc.returncode}). Waiting for a file change...")
                mtime = _watch_mtime()
                while _watch_mtime() == mtime:
                    time.sleep(0.4)
    except KeyboardInterrupt:
        print()
        _stop(proc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local iMessage viewer")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Wipe and rebuild the search index and word stats",
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="Build indexes and exit without starting the server",
    )
    args = parser.parse_args()
    if args.index:
        ensure_indexes(force=args.rebuild)
    elif os.environ.get("_IMESSAGE_RELOADER") == "1":
        serve(rebuild=args.rebuild)
    else:
        reloader()
