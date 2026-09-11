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
every post is recorded in a small JSON file before it is sent and publish()
refuses once the rolling 24-hour count reaches MAX_PER_DAY, or when the
text repeats or nearly repeats one of the last 50 posts.

Why no --post command: the account is the fly's. The only way text reaches
it is voice.publish -> xpost.publish, after the narrator's draft has passed
the grounding check. A person with a shell here cannot post as the fly.

  py xpost.py --check                        which X_* vars are set (never values)
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
import difflib
import unicodedata
from pathlib import Path
from urllib.parse import parse_qsl, quote, urlsplit

import requests

try:
    from envcfg import load_env
except ImportError:
    from launch import load_env

ROOT = Path(__file__).parent

# api.x.com is the documented host; api.twitter.com still answers but a
# redirect between them would strip the signed header, so redirects are never
# followed (see _post).
UPLOAD_URL = "https://api.x.com/2/media/upload"
UPLOAD_URL_V1 = "https://upload.twitter.com/1.1/media/upload.json"
TWEET_URL = "https://api.x.com/2/tweets"
LIMIT = 280
NEAR_DUPLICATE = 0.8     # SequenceMatcher ratio, digits removed, at or above which a post is a repeat
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
    # RFC 5849 3.4.1: the base URI is scheme://host/path; any query string is
    # decoded into the signed parameter set instead of staying in the URI.
    parts = urlsplit(url)
    base_url = f"{parts.scheme.lower()}://{parts.netloc.lower()}{parts.path}"
    every = {**oauth, **dict(parse_qsl(parts.query, keep_blank_values=True)), **(extra_params or {})}
    param_string = "&".join(f"{enc(k)}={enc(every[k])}" for k in sorted(every))
    base = "&".join([method.upper(), enc(base_url), enc(param_string)])
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
_GTLDS = ("com|net|org|info|biz|xyz|online|app|dev|fun|live|site|tech|finance|money|cloud"
          "|co|tv|io|ai|me|gg|wiki|news|blog|market|world|network|exchange|capital|coin|crypto")
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

_HINTS = {
    "oauth1-permissions": "the app's user authentication is not Read and write",
    "duplicate-content": "X already has this text from this account",
    "usage-capped": "the developer account's monthly cap is reached",
    "client-forbidden": "the app is not attached to a project with API access",
    "not-authorized-for-resource": "the access token is for a different app or user",
}


def _explain(r, what):
    """A one-line reason from X's error body, with a hint for the known ones."""
    detail = ""
    try:
        err = r.json()
        problem = str(err.get("type") or "").rsplit("/", 1)[-1]
        detail = " ".join(x for x in (err.get("title"), err.get("detail")) if x)
        if not detail and err.get("errors"):
            detail = "; ".join(str(e.get("message") or e) for e in err["errors"])[:200]
        if problem in _HINTS:
            detail += f" ({_HINTS[problem]})"
    except ValueError:
        detail = r.text[:240]
    if r.status_code == 429:
        h = getattr(r, "headers", {})
        detail += f" (retry after {h.get('x-rate-limit-reset') or h.get('retry-after') or '?'})"
    return f"{what} {r.status_code}: {detail[:220]}"


def _post(url, creds, **kw):
    """One signed POST. Never follows a redirect: a 3xx would drop the header."""
    try:
        r = requests.post(url, headers={"Authorization": oauth_header("POST", url, creds)},
                          allow_redirects=False, timeout=kw.pop("timeout", 30), **kw)
    except requests.RequestException as e:
        raise RuntimeError(f"{url.split('/', 3)[-1]} failed: {str(e)[:240]}") from e
    if 300 <= r.status_code < 400:
        where = getattr(r, "headers", {}).get("Location")
        raise RuntimeError(f"{url} redirected to {where}; the host has moved")
    return r


def upload_media(creds, image_bytes, mime="image/jpeg"):
    """
    v2 POST /2/media/upload (multipart `media` + media_category), falling
    back to the v1.1 base64 form if v2 refuses the shape. Multipart is the
    one body encoding OAuth 1.0a keeps out of the signature base, so the
    header is signed over the OAuth fields alone. Returns the media id.
    """
    r = _post(UPLOAD_URL, creds, files={"media": ("frame.jpg", image_bytes, mime)},
              data={"media_category": "tweet_image"}, timeout=60)
    if r.status_code in (400, 404, 415):
        b64 = base64.b64encode(image_bytes).decode()
        r = _post(UPLOAD_URL_V1, creds, files={"media_data": (None, b64)}, timeout=60)
    if r.status_code // 100 != 2:
        raise RuntimeError(_explain(r, "media/upload"))
    body = r.json()
    mid = ((body.get("data") or {}).get("id") or body.get("media_id_string") or body.get("media_id"))
    if not mid:
        raise RuntimeError(f"media/upload: no media id in {r.text[:240]}")
    _log(f"uploaded {len(image_bytes)} bytes of {mime} as media {mid}")
    return str(mid)


def post_tweet(creds, text, media_id=None):
    """v2 POST /2/tweets. Returns the tweet id as a string."""
    payload = {"text": text}
    if media_id:
        payload["media"] = {"media_ids": [media_id]}
    r = _post(TWEET_URL, creds, json=payload, timeout=30)
    if r.status_code // 100 != 2:
        raise RuntimeError(_explain(r, "POST /2/tweets"))
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
    recent = posts[-DEDUPE_WINDOW:]
    if text in [p.get("text") for p in recent]:
        return {"skipped": "duplicate"}
    twin = near_duplicate(text, [p.get("text") or "" for p in recent])
    if twin is not None:
        return {"skipped": f"near-duplicate of a post {twin:.2f} alike"}
    if posted_last_24h(posts) >= MAX_PER_DAY:
        return {"skipped": "daily cap"}

    # Write-ahead: the entry is in the ledger before X sees it, so a crash
    # between the post and the write cannot make the cap fail open, and a
    # ledger that cannot be written stops the post rather than the record.
    entry = {"t": time.time(), "id": None, "text": text, "media": False, "pending": True}
    posts.append(entry)
    try:
        _write_ledger(path, posts)
    except OSError as e:
        _log(f"ledger unwritable, refusing to post: {str(e)[:120]}")
        return {"skipped": "ledger unwritable"}

    media_id = None
    if image_bytes:
        try:
            media_id = upload_media(creds, image_bytes, mime)
        except Exception as e:   # the words matter more than the picture
            _log(f"media upload failed, posting text only: {str(e)[:240]}")

    try:
        tid = post_tweet(creds, text, media_id)
    except Exception:
        entry["failed"] = True     # stays in the ledger and still counts
        _write_ledger(path, posts)
        raise
    entry.update({"id": tid, "media": bool(media_id)})
    entry.pop("pending", None)
    _write_ledger(path, posts)
    _log(f"posted {tid} ({n} chars{', with image' if media_id else ''})")
    return {"id": tid}


def _skeleton(text):
    """Text with its numbers, punctuation and case removed: what repeats."""
    t = unicodedata.normalize("NFC", text).lower()
    t = re.sub(r"[\d,.]+", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    return " ".join(t.split())


def near_duplicate(text, earlier):
    """
    The highest similarity ratio between text and any earlier post, with
    numbers stripped, if it reaches NEAR_DUPLICATE; else None. The narrator
    repeating its own sentence with fresh figures is the realistic repeat,
    and an exact-match check never sees it.
    """
    a = _skeleton(text)
    if len(a) < 40:            # too short to judge; exact-match dedupe still applies
        return None
    best = 0.0
    for e in earlier:
        b = _skeleton(e)
        if b:
            best = max(best, difflib.SequenceMatcher(None, a, b).ratio())
    return best if best >= NEAR_DUPLICATE else None


# -- CLI ----------------------------------------------------------------------
# --check only. There is deliberately no way to post from the command line.

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
    ap = argparse.ArgumentParser(description="post-only X client for the fly's journal; "
                                             "posts come from voice.py alone")
    ap.add_argument("--check", action="store_true", help="show which X_* vars are set")
    a = ap.parse_args(argv)
    if a.check:
        return _check()
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
