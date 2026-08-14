"""Paths and constants shared by every module."""

import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, ".cache")
DB_PATH = os.path.expanduser("~/Library/Messages/chat.db")
CONTACTS_GLOBS = [
    os.path.expanduser("~/Library/Application Support/AddressBook/AddressBook-v22.abcddb"),
    os.path.expanduser("~/Library/Application Support/AddressBook/Sources/*/AddressBook-v22.abcddb"),
]
PORT = 8765
PAGE_SIZE = 150
THUMB_SIZE = 512
APPLE_EPOCH = 978307200  # 2001-01-01 relative to Unix epoch
CACHE_CONTROL = "public, max-age=31536000, immutable"
SNAPSHOT_DB = os.path.join(CACHE_DIR, "chatdb", "chat.db")
SEARCH_DB = os.path.join(CACHE_DIR, "search.db")
SEARCH_SCHEMA = "1"
SEARCH_LIMIT = 300
VOICE_CACHE = os.path.join(CACHE_DIR, "voice.json")
VOICE_SCHEMA = "5"
NLTK_DATA = os.path.join(CACHE_DIR, "nltk_data")
PREFS_PATH = os.path.join(CACHE_DIR, "prefs.json")
START_OLDEST = "oldest"
START_NEWEST = "newest"

os.makedirs(CACHE_DIR, exist_ok=True)


def load_prefs():
    try:
        with open(PREFS_PATH) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        data = {}
    start = data.get("start", START_OLDEST)
    if start not in (START_OLDEST, START_NEWEST):
        start = START_OLDEST
    return {"start": start}


def save_pref(key, value):
    prefs = load_prefs()
    prefs[key] = value
    with open(PREFS_PATH, "w") as f:
        json.dump(prefs, f)
