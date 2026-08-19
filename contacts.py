"""macOS AddressBook lookup: names, avatars, chat titles."""

import glob
import hashlib
import html
import re
import sqlite3

from config import CONTACTS_GLOBS


def normalize_phone(s):
    digits = re.sub(r"\D", "", s or "")
    return digits[-10:] if len(digits) >= 10 else digits


def normalize_email(s):
    return (s or "").strip().lower()


def load_contacts():
    lookup = {}
    photos = {}
    owners = {}
    paths = []
    for pattern in CONTACTS_GLOBS:
        paths.extend(glob.glob(pattern))

    for path in paths:
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            names = {}
            record_photos = {}
            for r in conn.execute(
                "SELECT Z_PK, ZFIRSTNAME, ZLASTNAME, ZORGANIZATION, ZNICKNAME, "
                "ZTHUMBNAILIMAGEDATA, ZIMAGEDATA FROM ZABCDRECORD"
            ):
                full = " ".join(p for p in [r["ZFIRSTNAME"], r["ZLASTNAME"]] if p).strip()
                name = full or r["ZNICKNAME"] or r["ZORGANIZATION"]
                if name:
                    names[r["Z_PK"]] = name
                raw = r["ZTHUMBNAILIMAGEDATA"] or r["ZIMAGEDATA"]
                if raw and len(raw) > 100:
                    image = raw[1:]  # AddressBook prefixes a version byte
                    if image.startswith(b"\x89PNG"):
                        record_photos[r["Z_PK"]] = ("image/png", image)
                    elif image.startswith(b"\xff\xd8\xff"):
                        record_photos[r["Z_PK"]] = ("image/jpeg", image)
            for r in conn.execute("SELECT ZOWNER, ZFULLNUMBER FROM ZABCDPHONENUMBER"):
                name = names.get(r["ZOWNER"])
                key = normalize_phone(r["ZFULLNUMBER"])
                if name and key:
                    lookup["phone:" + key] = name
                    owners["phone:" + key] = f"{path}:{r['ZOWNER']}"
                    if r["ZOWNER"] in record_photos:
                        photos["phone:" + key] = record_photos[r["ZOWNER"]]
            for r in conn.execute("SELECT ZOWNER, ZADDRESS FROM ZABCDEMAILADDRESS"):
                name = names.get(r["ZOWNER"])
                if name and r["ZADDRESS"]:
                    email_key = "email:" + normalize_email(r["ZADDRESS"])
                    lookup[email_key] = name
                    owners[email_key] = f"{path}:{r['ZOWNER']}"
                    if r["ZOWNER"] in record_photos:
                        photos[email_key] = record_photos[r["ZOWNER"]]
            conn.close()
        except Exception:
            continue

    # A photo reused byte-for-byte under two different names is a generic
    # placeholder (e.g. macOS's own "photo sync failed" icon), not a real
    # picture of either person, so drop it everywhere it appears.
    names_by_photo = {}
    for key, (_, image) in photos.items():
        names_by_photo.setdefault(image, set()).add(lookup[key])
    photos = {k: v for k, v in photos.items() if len(names_by_photo[v[1]]) == 1}

    return lookup, photos, owners


CONTACTS, CONTACT_PHOTOS, CONTACT_OWNERS = load_contacts()
AVATAR_INDEX = {hashlib.sha1(k.encode()).hexdigest()[:16]: mime_bytes for k, mime_bytes in CONTACT_PHOTOS.items()}


def handle_key(identifier):
    if not identifier:
        return None
    if "@" in identifier:
        return "email:" + normalize_email(identifier)
    key = normalize_phone(identifier)
    return "phone:" + key if key else None


def resolve_contact(identifier):
    key = handle_key(identifier)
    return CONTACTS.get(key) if key else None


def person_key(identifier):
    """One key for every handle that belongs to the same person. A contact card
    that lists a number and an address collapses both onto its record. A handle
    with no card falls back to its own normalized form, so the same number in
    two formats still merges."""
    key = handle_key(identifier)
    if not key:
        return None
    return CONTACT_OWNERS.get(key, key)


def avatar_id(identifier):
    key = handle_key(identifier)
    if not key or key not in CONTACT_PHOTOS:
        return None
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def load_participants(conn, chat_ids):
    if not chat_ids:
        return {}
    qmarks = ",".join("?" * len(chat_ids))
    out = {}
    for row in conn.execute(
        f"""SELECT chj.chat_id, h.id as handle
            FROM chat_handle_join chj JOIN handle h ON h.ROWID = chj.handle_id
            WHERE chj.chat_id IN ({qmarks})""",
        chat_ids,
    ):
        out.setdefault(row["chat_id"], []).append(row["handle"])
    return out


AVATAR_COLORS = ["#ff9500", "#ff3b30", "#af52de", "#5856d6", "#007aff", "#34c759", "#ff2d55", "#5ac8fa"]


def avatar_html(name, identifier=None):
    aid = avatar_id(identifier)
    if aid:
        return f'<img class="avatar" src="/avatar/{aid}" loading="lazy" decoding="async">'
    initial = (name or "?").strip()[:1].upper() or "?"
    color = AVATAR_COLORS[sum(ord(c) for c in (name or "")) % len(AVATAR_COLORS)]
    return f'<span class="avatar" style="background:{color}">{html.escape(initial)}</span>'


def chat_label(display_name, identifier, participants):
    if display_name:
        return display_name
    resolved_self = resolve_contact(identifier)
    if resolved_self:
        return resolved_self
    if participants:
        return ", ".join(resolve_contact(p) or p for p in participants)
    return identifier or "Unknown"


def load_chat_labels(conn, chat_ids):
    ids = list(dict.fromkeys(i for i in chat_ids if i is not None))
    if not ids:
        return {}
    qmarks = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT ROWID as id, chat_identifier, display_name FROM chat WHERE ROWID IN ({qmarks})",
        ids,
    ).fetchall()
    need = [r["id"] for r in rows if not r["display_name"] and not resolve_contact(r["chat_identifier"])]
    pmap = load_participants(conn, need)
    return {
        r["id"]: chat_label(r["display_name"], r["chat_identifier"], pmap.get(r["id"]))
        for r in rows
    }
