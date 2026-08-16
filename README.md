# iMessage Viewer

Fast local viewer for every iMessage and SMS on this Mac. Scroll chats, search them, look at photos, and read stats on how you text.

Nothing ever leaves this Mac. There is no cloud, no account, and no telemetry. All messages stay on your machine. Completely private.

Reads `~/Library/Messages/chat.db` and Contacts. Writes indexes to `.cache/` (not in git).

## Full Disk Access

macOS blocks Messages until you grant Full Disk Access to the app that runs this server.

1. Open System Settings → Privacy & Security → Full Disk Access.
2. Turn on Terminal, iTerm, or Cursor (the app you use to run `python3 app.py`).
3. Quit that app with Command-Q, then open it again.

Without this, every page shows old data or no data.

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 app.py
```

Open http://127.0.0.1:8765/

The first start builds the search index in the background.

`python3 app.py --rebuild` wipes and rebuilds indexes.

## Stats

- Volume: daily heatmap, monthly sent vs received, most contacted
- People: eras, who held each year, who is fading in or out, long-term friends, people you fell out of touch with
- How you talk: reply speed, who starts conversations, tapbacks, words and phrases
