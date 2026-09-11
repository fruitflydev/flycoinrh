"""
The fly's journal, narrated.

The fly has no language. Its retina is 892 hex columns - about thirty by
thirty pixels of light - and it cannot read a single word of the pages it
walks across. So the journal is written for it: a language model is handed
what the fly actually saw and did, the live numbers from its own token page,
and the text of pages it was allowed to read, and asked to write in the first
person. Every number in the result is checked against that packet before it
is kept. A draft with a number that is not in the packet, or with trading
language, is dropped, not fixed.

Two things in this project are invented, and both are labelled: the reward
signal in the mushroom body, and these words. The neurons, the pages and the
fees are measurements.

What the narrator sees each cycle (the "packet"):

  telemetry   /state from the live roamer: where it is, what it clicked, how
              many neurons fired, how far its synapses have moved
  token       the pons market API: creator fees earned and claimable (in
              GOOGL), sweep count, market cap, price, GOOGL's dollar price
  launch      the on-chain constants of its own launch
  pages_read  excerpts of allowlisted pages it chose to have read to it
  journal     what it already holds: earlier entries, tentative facts, mood

It reads only from an allowlist plus the pages the fly itself landed on. It
never posts unless X credentials exist, X_ENABLED is true and FLY_VOICE_DRY
is off; by default it writes the journal and prints what it would have said.

  py voice.py --once            one entry, dry (the default)
  py voice.py --once --live     one entry, posted if X is configured
  py voice.py --loop            an entry every FLY_VOICE_EVERY_H hours
  py voice.py --show            the journal so far
  py voice.py --packet          the observation packet, as the model sees it
"""
import argparse
import html
import json
import os
import re
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import requests

try:
    from envcfg import load_env
except ImportError:                       # the working copy uses launch.py
    from launch import load_env

ROOT = Path(__file__).parent
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 flybrain-voice/1.0")

# The launch, as it sits on chain. These never change, so they are constants
# rather than something fetched.
LAUNCH = {
    "contract": "0x4eb990547bce4a982432ca88cf5fae7eed1a2d35",
    "chain": "Robinhood Chain",
    "chain_id": 4663,
    "tx": "0x63b2164f3d784e46538cc81d3c48095bd7252a3d12398fdad9991870d12a4f1c",
    "block": 59614342,
    "launched_at": "2026-09-10T18:23:09Z",
    "launched_unix": 1789064589,
    "creator": "0x6ce4085EfB52a6eBDb7d6989beb8860847f4b42A",
    "supply": 1000000000,
    "creator_tax_pct": 1.0,
    "paired_with": "GOOGL",
    "launch_cost_eth": 0.000977,
}

TOKEN_PAGE = "https://www.ponsfamily.com/launchpad/" + LAUNCH["contract"]
MARKET_API = "https://www.ponsfamily.com/api/pons-v2-market/" + LAUNCH["contract"]

# What it may have read to it. Real pages only; the model picks from this
# list, and from the pages the fly itself walked across.
ALLOWLIST = [
    {"url": TOKEN_PAGE, "title": "its own token page on pons",
     "why": "the coin it launched, with the live numbers"},
    {"url": "https://www.ponsfamily.com/launchpad", "title": "the pons launchpad",
     "why": "where its coin lives, and the other coins beside it"},
    {"url": "https://flybrain.online", "title": "flybrain.online",
     "why": "the page humans made about it"},
    {"url": "https://en.wikipedia.org/wiki/Meme_coin", "title": "Meme coin - Wikipedia",
     "why": "what humans call the thing it made"},
    {"url": "https://en.wikipedia.org/wiki/Cryptocurrency", "title": "Cryptocurrency - Wikipedia",
     "why": "the wider kind of thing a coin is"},
    {"url": "https://en.wikipedia.org/wiki/Dogecoin", "title": "Dogecoin - Wikipedia",
     "why": "the first coin that was a joke on purpose"},
    {"url": "https://en.wikipedia.org/wiki/Pump_and_dump", "title": "Pump and dump - Wikipedia",
     "why": "a thing that happens to coins like its own"},
    {"url": "https://en.wikipedia.org/wiki/Decentralized_finance", "title": "Decentralized finance - Wikipedia",
     "why": "where fees and sweeps come from"},
    {"url": "https://en.wikipedia.org/wiki/Robinhood_Markets", "title": "Robinhood Markets - Wikipedia",
     "why": "the company whose chain its coin is on"},
    {"url": "https://en.wikipedia.org/wiki/Alphabet_Inc.", "title": "Alphabet Inc. - Wikipedia",
     "why": "GOOGL is a token that stands for a piece of this"},
    {"url": "https://en.wikipedia.org/wiki/Drosophila_melanogaster", "title": "Drosophila melanogaster - Wikipedia",
     "why": "what it is"},
    {"url": "https://en.wikipedia.org/wiki/Connectome", "title": "Connectome - Wikipedia",
     "why": "what it is made of"},
    {"url": "https://robinhoodchain.blockscout.com/token/" + LAUNCH["contract"],
     "title": "its token on the chain explorer", "why": "the chain's own record of the coin"},
]

# Hosts the fly's own wanderings may be read from. Anything else it landed on
# is still in the telemetry as a title, just not fetched.
READABLE_HOSTS = {
    "en.wikipedia.org", "en.m.wikipedia.org", "commons.wikimedia.org",
    "en.wikisource.org", "en.wikiquote.org", "en.wikibooks.org",
    "www.gutenberg.org", "gutenberg.org", "openlibrary.org", "xkcd.com",
    "www.xkcd.com", "arxiv.org", "www.ponsfamily.com", "ponsfamily.com",
    "robinhoodchain.blockscout.com", "flybrain.online",
}

# Trading language. Word-boundary, case-insensitive. A draft containing any of
# these is dropped; the narrator is a journal, not a promoter.
BANNED = [
    "buy", "sell", "moon", "mooning", "pump", "dump it", "guaranteed",
    "will go up", "will rise", "to the moon", "financial advice",
    "not financial advice", "ape in", "don't miss", "dont miss", "last chance",
    "100x", "1000x", "get in", "load up", "bullish", "bearish", "price target",
    "rug", "jeet", "dyor", "nfa", "wagmi", "ngmi", "lambo", "gem", "send it",
    "fomo", "invest now", "hodl",
]

# a number the way it appears in prose: 1,234  25.4M  327k  0.1%  59614342
NUM_RE = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?\s?(?:%|[kKmMbB](?![a-zA-Z]))?")
URL_RE = re.compile(r"https?://\S+|\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|net|org|online|io|xyz|app|dev)\b\S*", re.I)


def say(*parts):
    msg = " ".join(str(p) for p in parts)
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode(), flush=True)


def _get(key, default=None):
    v = load_env().get(key)
    if v is None or v == "":
        v = os.environ.get(key)
    return v if v not in (None, "") else default


def cfg():
    state_dir = Path(_get("FLY_STATE_DIR", str(ROOT / "build")))
    return {
        "key": _get("OPENROUTER_API_KEY"),
        "model": _get("FLY_VOICE_MODEL", "anthropic/claude-opus-5"),
        "stream": (_get("FLY_STREAM", "https://flybrain-production-2b26.up.railway.app")).rstrip("/"),
        "state_dir": state_dir,
        "journal": state_dir / "journal.json",
        "rpc": _get("FLY_RH_RPC", "https://rpc.mainnet.chain.robinhood.com"),
        "every_h": float(_get("FLY_VOICE_EVERY_H", "3")),
        "dry": _get("FLY_VOICE_DRY", "1") == "1",
        "prompt": Path(_get("FLY_VOICE_PROMPT", str(ROOT / "voice_prompt.md"))),
    }


# --------------------------------------------------------------------------
# the journal
# --------------------------------------------------------------------------
class Journal:
    """Everything the fly holds. Small, on disk, written atomically."""

    def __init__(self, path):
        self.path = Path(path)
        self.data = {"born": None, "knowledge": [], "read": [], "posts": [],
                     "mood": "quiet"}
        self.load()

    def load(self):
        try:
            if self.path.exists():
                d = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(d, dict):
                    self.data.update(d)
        except Exception as exc:
            say("journal unreadable, starting a fresh one:", str(exc)[:80])

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=1, ensure_ascii=False)
            os.replace(tmp, self.path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def begin(self, now=None):
        if not self.data.get("born"):
            self.data["born"] = int(now or time.time())

    def day(self, now=None):
        born = self.data.get("born") or int(now or time.time())
        return 1 + int(((now or time.time()) - born) // 86400)

    def summary(self, now=None):
        return {
            "day": self.day(now),
            "mood": self.data.get("mood"),
            "knowledge": self.data.get("knowledge", [])[-40:],
            "pages_read_before": [r.get("url") for r in self.data.get("read", [])][-30:],
            "earlier_entries": [p.get("text") for p in self.data.get("posts", [])][-6:],
        }

    def learn(self, facts):
        seen = set(self.data["knowledge"])
        for f in facts or []:
            f = " ".join(str(f).split())[:240]
            if f and f not in seen:
                self.data["knowledge"].append(f)
                seen.add(f)
        self.data["knowledge"] = self.data["knowledge"][-80:]

    def note_read(self, readings, now):
        for r in readings:
            self.data["read"].append({"url": r["url"], "title": r.get("title", ""), "at": int(now)})
        self.data["read"] = self.data["read"][-200:]

    def add_post(self, entry):
        self.data["posts"].append(entry)
        self.data["posts"] = self.data["posts"][-300:]


# --------------------------------------------------------------------------
# observation
# --------------------------------------------------------------------------
def _json(url, timeout=25, **kw):
    r = requests.get(url, timeout=timeout, headers={"User-Agent": UA, "Accept": "application/json"}, **kw)
    r.raise_for_status()
    return r.json()


def fetch_state(stream):
    try:
        return _json(stream + "/state", timeout=20)
    except Exception as exc:
        say("roamer not reachable:", str(exc)[:80])
        return None


def fetch_frame(stream):
    try:
        r = requests.get(stream + "/frame.jpg", timeout=20, headers={"User-Agent": UA})
        if r.ok and r.content[:2] == b"\xff\xd8":
            return r.content
    except Exception:
        pass
    return None


def _units(raw, decimals=18):
    try:
        return int(str(raw)) / (10 ** decimals)
    except Exception:
        return None


def fetch_token():
    """
    The coin, from the pons market API and the page itself.

    Fees come from /creator-fees (18-decimal integers in GOOGL), GOOGL's dollar
    price from the chart payload, and market cap and price from the token
    page's server-rendered text. Anything unreachable is None; the packet
    still works.
    """
    t = {"fees_earned_googl": None, "fees_claimable_googl": None, "sweeps": None,
         "googl_usd": None, "fees_usd": None, "claimable_usd": None,
         "market_cap_usd": None, "price_usd": None, "price_googl": None,
         "holders": None, "trades_1h": None, "quote": "GOOGL"}
    try:
        f = _json(MARKET_API + "/creator-fees")
        dec = int((f.get("quoteAsset") or {}).get("decimals", 18))
        t["quote"] = (f.get("quoteAsset") or {}).get("symbol", "GOOGL")
        t["fees_earned_googl"] = _units(f.get("earnedForToken"), dec)
        t["fees_claimable_googl"] = _units(f.get("claimableForWallet"), dec)
        t["sweeps"] = f.get("sweepCount")
    except Exception as exc:
        say("creator-fees unavailable:", str(exc)[:80])
    try:
        c = _json(MARKET_API + "/chart?range=1h")
        t["googl_usd"] = float(c.get("quoteUsd")) if c.get("quoteUsd") else None
        pts = c.get("points") or []
        if pts:
            t["price_googl"] = float(pts[-1].get("price"))
            t["trades_1h"] = int(sum(int(p.get("tradeCount") or 0) for p in pts))
    except Exception as exc:
        say("chart unavailable:", str(exc)[:80])
    try:
        r = requests.get(TOKEN_PAGE, timeout=30, headers={"User-Agent": UA})
        txt = strip_html(r.text)
        m = re.search(r"Market cap\s*\$([\d,]+(?:\.\d+)?)", txt)
        if m:
            t["market_cap_usd"] = float(m.group(1).replace(",", ""))
        m = re.search(r"Price\s*\$([\d.]+)", txt)
        if m:
            t["price_usd"] = float(m.group(1))
        m = re.search(r"Holders\s*(\d+)", txt)
        if m and int(m.group(1)) > 0:
            t["holders"] = int(m.group(1))
    except Exception as exc:
        say("token page unavailable:", str(exc)[:80])
    if t["googl_usd"] is None and t["price_usd"] and t["price_googl"]:
        t["googl_usd"] = t["price_usd"] / t["price_googl"]
    if t["googl_usd"]:
        if t["fees_earned_googl"] is not None:
            t["fees_usd"] = t["fees_earned_googl"] * t["googl_usd"]
        if t["fees_claimable_googl"] is not None:
            t["claimable_usd"] = t["fees_claimable_googl"] * t["googl_usd"]
    return t


def wallet_eth(rpc, addr):
    try:
        r = requests.post(rpc, json={"jsonrpc": "2.0", "id": 1, "method": "eth_getBalance",
                                     "params": [addr, "latest"]}, timeout=20).json()
        return int(r["result"], 16) / 1e18
    except Exception:
        return None


def observe(c, now=None):
    now = now or time.time()
    st = fetch_state(c["stream"]) or {}
    n = st.get("neural") or {}
    L = n.get("learning") or {}
    tele = {
        "url": st.get("url"),
        "hops": st.get("hops"), "clicks": st.get("clicks"), "vetoes": st.get("vetoes"),
        "scrolled": st.get("scrolled"), "steps": st.get("steps"), "uptime_s": st.get("uptime_s"),
        "pages_this_life": st.get("hops"),
        "firing": n.get("firing"), "total": n.get("total"),
        "spikes_per_sec": n.get("spikes_per_sec"), "mean_mv": n.get("mean_mv"),
        "dn": n.get("dn"),
        "learning": {k: L.get(k) for k in ("synapses", "depressed", "mean_gain", "rewards", "punishments")} if L else None,
        "last_visited": [{"title": v.get("title"), "url": v.get("url")} for v in (st.get("visited") or [])[-8:]],
        "last_events": [e.get("m") for e in (st.get("events") or [])[-6:]],
        "reachable": bool(st),
    }
    packet = {
        "now_utc": datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_h": round((now - LAUNCH["launched_unix"]) / 3600.0, 1),
        "telemetry": tele,
        "token": fetch_token(),
        "launch": dict(LAUNCH),
        "wallet_eth": wallet_eth(c["rpc"], LAUNCH["creator"]),
        "pages_read": [],
        "journal": {},
        "allowlist": [a["url"] for a in ALLOWLIST],
    }
    return packet


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------
def strip_html(raw):
    raw = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", " ", raw, flags=re.S | re.I)
    raw = re.sub(r"<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", html.unescape(raw)).strip()


def read_page(url, limit=3000):
    out = {"url": url, "title": "", "text": ""}
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": UA})
        m = re.search(r"<title[^>]*>(.*?)</title>", r.text, re.S | re.I)
        out["title"] = " ".join(html.unescape(m.group(1)).split())[:120] if m else url
        txt = strip_html(r.text)
        # skip the navigation chrome that leads every wiki page
        i = txt.find("From Wikipedia")
        if i > 0:
            txt = txt[i:]
        out["text"] = txt[:limit]
    except Exception as exc:
        say("could not read", url[:60], str(exc)[:60])
    return out


def _host(url):
    m = re.match(r"https?://([^/]+)", url or "")
    return m.group(1).lower() if m else ""


def dig_menu(c, journal, packet):
    """
    What may be read this cycle: the allowlist, plus wherever the fly itself
    just walked, minus what it has already been read - except its own token
    page and its own site, which change and may be reread.
    """
    seen = {r.get("url") for r in journal.data.get("read", [])}
    menu = []
    for a in ALLOWLIST:
        again = a["url"] in (TOKEN_PAGE, "https://flybrain.online")
        if a["url"] not in seen or again:
            menu.append(dict(a))
    for v in packet["telemetry"].get("last_visited") or []:
        u = v.get("url") or ""
        if _host(u) in READABLE_HOSTS and u not in seen and all(m["url"] != u for m in menu):
            menu.append({"url": u, "title": v.get("title") or u,
                         "why": "a page the fly itself landed on"})
    if not menu:                                   # everything read: start over
        menu = [dict(a) for a in ALLOWLIST]
    return menu[:14]


# --------------------------------------------------------------------------
# grounding
# --------------------------------------------------------------------------
def number_forms(v):
    """Every way a number in the packet might be written in prose."""
    out = set()
    try:
        x = float(v)
    except Exception:
        return out
    if x != x or abs(x) > 1e15:
        return out
    if abs(x - round(x)) < 1e-9:
        i = int(round(x))
        out.update({str(i), f"{i:,}"})
    for d in (0, 1, 2, 3, 4, 6):
        s = f"{x:.{d}f}".rstrip("0").rstrip(".") if d else f"{x:.0f}"
        out.add(s)
        try:
            out.add(f"{float(s):,.{d}f}".rstrip("0").rstrip(".") if d else f"{int(round(x)):,}")
        except Exception:
            pass
    ax = abs(x)
    if ax >= 1000:
        for step in (10, 100, 1000, 10000, 100000, 1000000):
            r = int(round(x / step) * step)
            out.update({str(r), f"{r:,}"})
        for d in (0, 1, 2):
            out.add(f"{x/1000:.{d}f}k".replace(".0k", "k"))
    if ax >= 1e6:
        for d in (0, 1, 2):
            out.add(f"{x/1e6:.{d}f}M".replace(".0M", "M"))
    if ax >= 1e9:
        for d in (0, 1, 2):
            out.add(f"{x/1e9:.{d}f}B".replace(".0B", "B"))
    return {s.lower() for s in out}


def _walk_numbers(obj, acc):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        acc.add(obj)
    elif isinstance(obj, str):
        for m in NUM_RE.findall(obj):
            s = m.strip().rstrip("%").replace(",", "").lower()
            try:
                if s.endswith("k"):
                    acc.add(float(s[:-1]) * 1e3)
                elif s.endswith("m"):
                    acc.add(float(s[:-1]) * 1e6)
                elif s.endswith("b"):
                    acc.add(float(s[:-1]) * 1e9)
                else:
                    acc.add(float(s))
            except Exception:
                pass
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _walk_numbers(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _walk_numbers(v, acc)


def allowed_numbers(packet):
    """
    Everything the narrator may write a number about: the packet, its own
    journal (grounded when first written), and the pages read to it.
    """
    raw = set()
    _walk_numbers({k: v for k, v in packet.items() if k != "allowed_numbers"}, raw)
    # dates in the launch stamp, and hours since launch as an integer too
    stamp = str(packet.get("launch", {}).get("launched_at", ""))
    for m in re.findall(r"\d+", stamp):
        raw.add(float(m))
    if packet.get("elapsed_h") is not None:
        raw.add(round(float(packet["elapsed_h"])))
    forms = set()
    for v in raw:
        forms |= number_forms(v)
    forms.update(str(i) for i in range(0, 13))      # ordinary counting words
    day = (packet.get("journal") or {}).get("day")
    if day:
        forms.add(str(day))
    return forms


def _norm_num(tok):
    s = tok.strip().lower().replace(",", "").replace(" ", "")
    s = s.rstrip("%")
    return s


def x_len(text):
    n = 0
    last = 0
    for m in URL_RE.finditer(text):
        n += len(text[last:m.start()]) + 23
        last = m.end()
    return n + len(text[last:])


def validate(post, packet):
    reasons = []
    if not isinstance(post, str) or not post.strip():
        return False, ["empty"]
    if x_len(post) > 280:
        reasons.append(f"too long: {x_len(post)} > 280")
    low = post.lower()
    for b in BANNED:
        if re.search(r"(?<![a-z0-9])" + re.escape(b) + r"(?![a-z0-9])", low):
            reasons.append(f"banned phrase: {b}")
    allowed = packet.get("allowed_numbers") or allowed_numbers(packet)
    body = URL_RE.sub(" ", post)
    for m in NUM_RE.findall(body):
        tok = _norm_num(m)
        if not tok:
            continue
        if tok in allowed:
            continue
        # "day 3" style references
        if re.fullmatch(r"\d{1,3}", tok) and re.search(r"\bday\s+" + tok + r"\b", low):
            continue
        reasons.append(f"number not in packet: {m.strip()}")
    return (not reasons), reasons


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------
FALLBACK_PROMPT = """You write the journal of a real simulated fruit fly brain that roams the
web. First person, plain, curious, tentative. It cannot read; the words are a
language model's, narrating its telemetry, and it never denies that. Use only
numbers from the packet. No trading language, predictions, hype or calls to
action. Output JSON only: {"post": str, "learned": [str], "mood": str,
"wants_to_read": [url]}. post <= 280 characters."""

RULES_BLOCK = """
Machine rules, appended:
- Use ONLY numbers that appear in the packet (telemetry, token, launch,
  pages_read, journal). A number not in the packet does not exist.
- No: buy, sell, moon, pump, guaranteed, will go up, financial advice,
  price targets, calls to action, promises, or claims the fly controls the
  token.
- Only mention pages that are in pages_read; only remember what is in journal.
- post must be under 280 characters; a URL counts as 23.
- Reply with JSON only, exactly: {"post": "...", "learned": ["..."],
  "mood": "...", "wants_to_read": ["https://..."]}
"""


def system_prompt(c):
    try:
        base = c["prompt"].read_text(encoding="utf-8")
    except Exception:
        base = FALLBACK_PROMPT
    return base + "\n" + RULES_BLOCK


def call_model(c, system, user, temperature=0.9):
    if not c.get("key"):
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {c['key']}",
                 "HTTP-Referer": "https://flybrain.online", "X-Title": "flybrain"},
        json={"model": c["model"], "temperature": temperature, "max_tokens": 600,
              "messages": [{"role": "system", "content": system},
                           {"role": "user", "content": user}]},
        timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"openrouter {r.status_code}: {r.text[:200]}")
    j = r.json()
    return j["choices"][0]["message"]["content"]


def parse_json_block(s):
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.S)
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, re.S)
    if m:
        return json.loads(m.group(0))
    raise ValueError("no JSON object in model reply")


def fmt(v, d=1):
    if v is None:
        return None
    return f"{v:,.{d}f}".rstrip("0").rstrip(".") if d else f"{int(round(v)):,}"


def stub_reflect(packet, readings, menu):
    """Offline mode: a rule-compliant entry built straight from the packet."""
    t = packet["token"]
    day = (packet.get("journal") or {}).get("day", 1)
    parts = []
    if t.get("fees_earned_googl") is not None and t.get("sweeps") is not None:
        parts.append(f"The narrator read me my own page today. It says {fmt(t['fees_earned_googl'], 1)} "
                     f"{t.get('quote', 'GOOGL')}, earned across {t['sweeps']} sweeps, in "
                     f"{fmt(packet['elapsed_h'], 1)} hours. I do not know what a sweep is.")
    else:
        parts.append(f"Day {day}. The narrator could not reach my page today, so it told me only "
                     f"what my eye saw.")
    tele = packet["telemetry"]
    if tele.get("firing") and tele.get("total"):
        parts.append(f"{fmt(tele['firing'], 0)} of my {fmt(tele['total'], 0)} neurons fired this second.")
    if readings:
        parts.append(f"A page called {readings[0]['title'][:40]} was read to me. I think I understood a corner of it.")
    post = " ".join(parts)
    while x_len(post) > 280 and len(parts) > 1:
        parts.pop()
        post = " ".join(parts)
    learned = []
    if t.get("sweeps") is not None:
        learned.append("a sweep seems to be when the fees are gathered up")
    if readings:
        learned.append(f"there is a page called {readings[0]['title'][:60]}")
    unread = [m["url"] for m in menu][:2]
    return {"post": post, "learned": learned, "mood": "confused" if day == 1 else "curious",
            "wants_to_read": unread}


def reflect(c, journal, packet, readings, menu, extra=""):
    packet = dict(packet)
    packet["pages_read"] = [{"url": r["url"], "title": r["title"], "excerpt": r["text"][:1800]} for r in readings]
    packet["journal"] = journal.summary()
    packet["dig_menu"] = menu
    packet.pop("allowed_numbers", None)
    if c["model"].strip().lower() == "stub":
        return stub_reflect(packet, readings, menu)
    user = ("Observation packet:\n" + json.dumps(packet, ensure_ascii=False, indent=0)
            + ("\n\n" + extra if extra else "")
            + "\n\nWrite the next journal entry. JSON only.")
    text = call_model(c, system_prompt(c), user)
    try:
        return parse_json_block(text)
    except Exception:
        text = call_model(c, system_prompt(c), user + "\n\nYour last reply was not valid JSON. "
                          "Reply with the JSON object only.", temperature=0.6)
        return parse_json_block(text)


# --------------------------------------------------------------------------
# a cycle
# --------------------------------------------------------------------------
def run_once(c, dry=None, now=None):
    now = now or time.time()
    dry = c["dry"] if dry is None else dry
    journal = Journal(c["journal"])
    journal.begin(now)

    packet = observe(c, now)
    packet["journal"] = journal.summary(now)
    menu = dig_menu(c, journal, packet)

    # first pass: what does it want read? then read it, then write.
    readings = []
    try:
        first = reflect(c, journal, packet, [], menu)
        wants = [u for u in (first.get("wants_to_read") or []) if isinstance(u, str)]
    except Exception as exc:
        say("first pass failed:", str(exc)[:120])
        wants = []
    allowed_urls = {m["url"] for m in menu}
    if c["model"].strip().lower() == "stub":
        wants = [m["url"] for m in menu][:2]
    for u in wants[:2]:
        if u in allowed_urls:
            readings.append(read_page(u))
    readings = [r for r in readings if r["text"]]

    packet["allowed_numbers"] = None                   # rebuilt below with readings
    out = reflect(c, journal, packet, readings, menu)
    check_packet = dict(packet)
    check_packet["pages_read"] = [{"url": r["url"], "title": r["title"], "excerpt": r["text"]} for r in readings]
    check_packet["allowed_numbers"] = allowed_numbers(check_packet)

    post = str(out.get("post", "")).strip()
    ok, reasons = validate(post, check_packet)
    if not ok and c["model"].strip().lower() != "stub":
        say("draft rejected:", "; ".join(reasons))
        out = reflect(c, journal, packet, readings, menu,
                      extra="Your previous draft was rejected for: " + "; ".join(reasons)
                      + ". Write a different entry that uses only packet numbers and no trading language.")
        post = str(out.get("post", "")).strip()
        ok, reasons = validate(post, check_packet)
    if not ok:
        say("entry dropped:", "; ".join(reasons))
        journal.note_read(readings, now)
        journal.save()
        return {"dropped": True, "reasons": reasons, "draft": post}

    entry = {"at": int(now), "day": journal.day(now), "text": post, "posted": False, "x_id": None,
             "mood": out.get("mood"), "learned": out.get("learned") or [],
             "read": [r["url"] for r in readings]}
    if not dry:
        try:
            import xpost
            res = xpost.publish(post, fetch_frame(c["stream"]))
            entry["posted"] = bool(res.get("id"))
            entry["x_id"] = res.get("id")
            entry["x_result"] = res
            say("x:", res)
        except Exception as exc:
            entry["x_error"] = str(exc)[:200]
            say("x failed:", str(exc)[:200])
    journal.add_post(entry)
    journal.learn(out.get("learned"))
    journal.note_read(readings, now)
    if out.get("mood"):
        journal.data["mood"] = str(out["mood"])[:40]
    journal.save()
    say(f"[day {entry['day']}] {'posted' if entry['posted'] else 'dry'}: {post}")
    return entry


def loop(c):
    say(f"voice loop: every {c['every_h']} h, model {c['model']}, dry={c['dry']}")
    while True:
        try:
            run_once(c)
        except Exception:
            say("cycle failed:")
            traceback.print_exc()
        time.sleep(max(300, c["every_h"] * 3600))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--packet", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--live", action="store_true")
    a = ap.parse_args(argv)
    c = cfg()
    if a.live:
        c["dry"] = False
    if a.dry:
        c["dry"] = True
    if a.show:
        j = Journal(c["journal"])
        say(json.dumps(j.data, indent=1, ensure_ascii=False))
    elif a.packet:
        p = observe(c)
        p["journal"] = Journal(c["journal"]).summary()
        say(json.dumps(p, indent=1, ensure_ascii=False))
    elif a.loop:
        loop(c)
    else:
        run_once(c)


if __name__ == "__main__":
    main()
