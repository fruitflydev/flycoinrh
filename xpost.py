"""
Post-only X (Twitter) client for the fly's journal.

Pure Python, stdlib plus requests. A port of the working Node client: OAuth
1.0a user context signed with HMAC-SHA1, text through v2 POST /2/tweets,
images through v1.1 media/upload.

Why post-only: the fly never reads its mentions. Nothing that comes back
from X reaches the narrator, so there is no route for replies to steer the
words.

Why a ledger: the free tier allows 500 posts a month. A loop that wakes
every few minutes and posts unconditionally would burn that in a day, so
every successful post is recorded in a small JSON file and publish()
refuses once the rolling 24-hour count reaches MAX_PER_DAY, or when the
text repeats one of the last 50 posts.

  py xpost.py --check                        which X_* vars are set (never values)
  py xpost.py --post "text" [--image path]   respects X_ENABLED
"""
import argparse
import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import sys
import time
import unicodedata
from pathlib import Path
from urllib.parse import quote

import requests

try:
    from envcfg import load_env
except ImportError:
    from launch import load_env

ROOT = Path(__file__).parent

UPLOAD_URL = "https://upload.twitter.com/1.1/media/upload.json"
TWEET_URL = "https://api.twitter.com/2/tweets"
LIMIT = 280
URL_WEIGHT = 23          # every link becomes a t.co link of this length
LEDGER_KEEP = 500        # entries retained; enough for the cap and the dedupe window
DEDUPE_WINDOW = 50
DAY = 86400


def _get(key):
    # .env first, then the process environment. Railway injects secrets as
    # env vars, local runs keep them in .env, and load_env only merges FLY_*
    # from the environment, so the X_* and OPENROUTER keys need this second
    # look.
    return load_env().get(key) or os.environ.get(key)


def _int_env(key, default):
    try:
        return int(_get(key) or default)
    except ValueError:
        return default


# 12 a day is 360 a month against a 500-a-month free tier, which leaves room
# for the odd manual post without ever reaching the ceiling.
MAX_PER_DAY = _int_env("X_MAX_POSTS_PER_DAY", 12)


def _log(msg):
    # stderr, so `--post` can print its JSON result on stdout alone.
    print(f"[xpost] {msg}", file=sys.stderr, flush=True)


# -- OAuth 1.0a ---------------------------------------------------------------

def enc(s):
    """
    RFC 3986 percent-encoding, identical to the JS enc() it replaces:
    everything except A-Za-z0-9-_.~ is escaped as uppercase hex. quote()
    would otherwise leave "/" alone, and X's signature base must not.
    """
    return quote(str(s), safe="-_.~")


def oauth_header(method, url, creds, extra_params=None, nonce=None, timestamp=None):
    """
    Build the Authorization header for one request.

    creds: dict with key, secret, token, tsecret.
    extra_params: query or form parameters that belong in the signature
    base. There are none for a JSON body or a multipart upload, which is
    every request this module makes. nonce and timestamp are injectable so
    the signature can be checked against a fixed vector.
    """
    oauth = {
        "oauth_consumer_key": creds["key"],
        "oauth_nonce": nonce or secrets.token_hex(16),
        "oauth_signature_method": "HMAC-SHA1",
        "oauth_timestamp": str(timestamp if timestamp is not None else int(time.time())),
        "oauth_token": creds["token"],
        "oauth_version": "1.0",
    }
    every = {**oauth, **(extra_params or {})}
    param_string = "&".join(f"{enc(k)}={enc(every[k])}" for k in sorted(every))
    base = "&".join([method.upper(), enc(url), enc(param_string)])
    signing_key = f"{enc(creds['secret'])}&{enc(creds['tsecret'])}"
    digest = hmac.new(signing_key.encode(), base.encode(), hashlib.sha1).digest()
    oauth["oauth_signature"] = base64.b64encode(digest).decode()
    return "OAuth " + ", ".join(f'{enc(k)}="{enc(oauth[k])}"' for k in sorted(oauth))


# -- length as X counts it ----------------------------------------------------

# twitter-text v3 weights: code points in these ranges count 1, everything
# else counts 2. X normalises to NFC before counting, so we do too.
_LIGHT = ((0, 4351), (8192, 8205), (8208, 8223), (8242, 8247))

# What X auto-links, and therefore bills as one 23-character t.co link:
# anything with a scheme, plus bare domains on a generic TLD the narrator is
# likely to write (flybrain.online, ponsfamily.com). X does not link a bare
# ccTLD domain without a path - roam.py, mushroom.py - so those stay text.
_GTLDS = "com|net|org|info|biz|xyz|online|app|dev|fun|live|site|tech|finance|money|cloud"
_URL = re.compile(
    r"(?<![\w@$#/.-])(?:https?://[^\s<>\"']+"
    r"|(?:[a-z0-9-]+\.)+(?:" + _GTLDS + r")(?:/[^\s<>\"']*)?(?![\w-]))",
    re.IGNORECASE,
)
_TRAIL = ".,:;!?)]}\"'"


def _weight(text):
    return sum(1 if any(a <= ord(ch) <= b for a, b in _LIGHT) else 2 for ch in text)


def x_length(text):
    """
    Length as X counts it: NFC, most characters weigh 1, CJK and emoji
    weigh 2, and every URL weighs 23 whatever its real length. A
    multi-code-point emoji is over-counted (X bills the whole sequence as
    2); over-counting can only reject a post that would have fit, which is
    the safe direction for a bot that cannot read the error page.
    """
    text = unicodedata.normalize("NFC", text)
    total, pos = 0, 0
    for m in _URL.finditer(text):
        url = m.group(0)
        # Closing punctuation is not part of the link on X, so it is text.
        trailing = url[len(url.rstrip(_TRAIL)):]
        total += _weight(text[pos:m.start()]) + URL_WEIGHT + _weight(trailing)
        pos = m.end()
    return total + _weight(text[pos:])


# -- ledger -------------------------------------------------------------------

def ledger_path():
    # Same rule as mushroom.py: on a host with a persistent volume,
    # FLY_STATE_DIR keeps the ledger across redeploys, otherwise ./build.
    # A lost ledger would reset the daily cap, which is why it must persist.
    return Path(_get("FLY_STATE_DIR") or ROOT / "build") / "x_ledger.json"


def read_ledger(path=None):
    """The list of recorded posts, or None when the file exists but cannot be read."""
    path = path or ledger_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    posts = data.get("posts") if isinstance(data, dict) else None
    return posts if isinstance(posts, list) else None


def _write_ledger(path, posts):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"posts": posts[-LEDGER_KEEP:]}, indent=1),
                   encoding="utf-8")
    os.replace(tmp, path)   # a crash mid-write must not leave a half file


def posted_last_24h(posts, now=None):
    # A rolling window rather than a calendar day: it needs no timezone and
    # a burst at 23:50 cannot be followed by another at 00:10.
    now = time.time() if now is None else now
    return sum(1 for p in posts if now - float(p.get("t", 0)) < DAY)


# -- configuration ------------------------------------------------------------

_CRED_NAMES = {"key": "X_API_KEY", "secret": "X_API_SECRET",
               "token": "X_ACCESS_TOKEN", "tsecret": "X_ACCESS_SECRET"}


def enabled():
    return (_get("X_ENABLED") or "").strip().lower() == "true"


def credentials():
    """(creds, []) when all four values are present, else (None, missing names)."""
    creds = {k: _get(v) for k, v in _CRED_NAMES.items()}
    missing = [v for k, v in _CRED_NAMES.items() if not creds[k]]
    return (None, missing) if missing else (creds, [])


# -- the two API calls --------------------------------------------------------

def upload_media(creds, image_bytes, mime="image/jpeg"):
    """
    v1.1 media/upload with a base64 form field. Sent as multipart, the one
    body encoding OAuth 1.0a keeps out of the signature base, so the header
    is signed over the OAuth fields alone - exactly what the working Node
    client does. The simple endpoint sniffs the type from the bytes; mime
    is kept for the log line and for a chunked upload if one is ever needed.
    Returns the media id as a string.
    """
    b64 = base64.b64encode(image_bytes).decode()
    try:
        r = requests.post(UPLOAD_URL,
                          headers={"Authorization": oauth_header("POST", UPLOAD_URL, creds)},
                          files={"media_data": (None, b64)}, timeout=60)
    except requests.RequestException as e:
        raise RuntimeError(f"media/upload failed: {str(e)[:240]}") from e
    if r.status_code // 100 != 2:
        raise RuntimeError(f"media/upload {r.status_code}: {r.text[:240]}")
    body = r.json()
    mid = body.get("media_id_string") or body.get("media_id")
    if not mid:
        raise RuntimeError(f"media/upload: no media id in {r.text[:240]}")
    _log(f"uploaded {len(image_bytes)} bytes of {mime} as media {mid}")
    return str(mid)


def post_tweet(creds, text, media_id=None):
    """v2 POST /2/tweets. Returns the tweet id as a string."""
    payload = {"text": text}
    if media_id:
        payload["media"] = {"media_ids": [media_id]}
    try:
        r = requests.post(TWEET_URL,
                          headers={"Authorization": oauth_header("POST", TWEET_URL, creds)},
                          json=payload, timeout=30)
    except requests.RequestException as e:
        raise RuntimeError(f"POST /2/tweets failed: {str(e)[:240]}") from e
    if r.status_code // 100 != 2:
        raise RuntimeError(f"POST /2/tweets {r.status_code}: {r.text[:240]}")
    data = r.json().get("data") or {}
    tid = data.get("id")
    if not tid:
        raise RuntimeError(f"POST /2/tweets: no id in {r.text[:240]}")
    return str(tid)


# -- the one public entry point ----------------------------------------------

def publish(text, image_bytes=None, mime="image/jpeg"):
    """
    Post one journal entry. Returns {"id": tweet_id} on success, or
    {"skipped": reason} when nothing was sent. Raises ValueError for text X
    would reject on length and RuntimeError when the API refuses the post.
    Being unconfigured never raises: the roamer must keep running with or
    without an X account.
    """
    if not text or not text.strip():
        raise ValueError("empty post")
    n = x_length(text)
    if n > LIMIT:
        raise ValueError(f"post is {n} X-characters, limit is {LIMIT}")
    if not enabled():
        return {"skipped": "X_ENABLED=false"}
    creds, missing = credentials()
    if not creds:
        _log(f"not configured: missing {', '.join(missing)}")
        return {"skipped": "missing " + ", ".join(missing)}

    path = ledger_path()
    posts = read_ledger(path)
    if posts is None:
        # Fail closed: without the ledger the daily cap cannot be enforced,
        # and the cap is what stands between a stuck loop and the quota.
        _log(f"ledger unreadable, refusing to post: {path}")
        return {"skipped": "ledger unreadable"}
    if text in [p.get("text") for p in posts[-DEDUPE_WINDOW:]]:
        return {"skipped": "duplicate"}
    if posted_last_24h(posts) >= MAX_PER_DAY:
        return {"skipped": "daily cap"}

    media_id = None
    if image_bytes:
        try:
            media_id = upload_media(creds, image_bytes, mime)
        except Exception as e:   # the words matter more than the picture
            _log(f"media upload failed, posting text only: {str(e)[:240]}")

    tid = post_tweet(creds, text, media_id)
    posts.append({"t": time.time(), "id": tid, "text": text, "media": bool(media_id)})
    _write_ledger(path, posts)
    _log(f"posted {tid} ({n} chars{', with image' if media_id else ''})")
    return {"id": tid}


# -- CLI ----------------------------------------------------------------------

_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
         ".gif": "image/gif", ".webp": "image/webp"}


def _check():
    # Presence only. The values are secrets and must never reach a terminal
    # or a log that could be pasted somewhere.
    for name in list(_CRED_NAMES.values()) + ["X_ENABLED", "X_MAX_POSTS_PER_DAY"]:
        print(f"{name:20} {'set' if _get(name) else 'missing'}")
    print(f"{'enabled':20} {enabled()}")
    print(f"{'max per day':20} {MAX_PER_DAY}")
    path = ledger_path()
    posts = read_ledger(path)
    state = "unreadable" if posts is None else f"{len(posts)} posts"
    print(f"{'ledger':20} {path} ({state})")
    if posts:
        print(f"{'posted last 24h':20} {posted_last_24h(posts)}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="post-only X client for the fly's journal")
    ap.add_argument("--check", action="store_true", help="show which X_* vars are set")
    ap.add_argument("--post", metavar="TEXT", help="publish TEXT (respects X_ENABLED)")
    ap.add_argument("--image", metavar="PATH", help="attach an image to --post")
    a = ap.parse_args(argv)

    if a.check:
        return _check()
    if a.post is not None:
        image, mime = None, "image/jpeg"
        if a.image:
            p = Path(a.image)
            image = p.read_bytes()
            mime = _MIME.get(p.suffix.lower(), mime)
        print(json.dumps(publish(a.post, image, mime)))
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
