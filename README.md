# iMessage Viewer

A fast, private way to explore your messages on your Mac. Browse conversations, search chats, view photos, and discover insights about how you text.

Nothing ever leaves your Mac. There is no cloud, no account, and no telemetry. All messages stay on your machine. Completely private.

The app reads your Messages database (`~/Library/Messages/chat.db`) and Contacts, and stores local indexes in `.cache/` (excluded from Git).

### Allow Access to Messages

macOS requires **Full Disk Access** to read Messages.

1. Open the setting directly:
   ```bash
   open "x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles"
   ```
2. Enable the app running `python3 app.py`—such as **Terminal**, **iTerm**, or **Cursor**.

Without this permission, pages may show outdated or missing data.

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
