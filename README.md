# iMessage Viewer

Local viewer for macOS Messages. Reads `~/Library/Messages/chat.db` and Contacts on this Mac. Writes indexes to `.cache/` (not in git). Binds to `127.0.0.1` only.

Grant Full Disk Access to Terminal or Cursor: System Settings → Privacy & Security → Full Disk Access.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 app.py
```

Open http://127.0.0.1:8765/

The first start builds the search index in the background.

`python3 app.py --rebuild` wipes and rebuilds indexes.
