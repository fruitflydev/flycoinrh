"""
The fly's journal, narrated.

The fly has no language. Its retina is 892 hex columns - about thirty by
thirty pixels of light - and it cannot read a single word of the pages it
walks across. So the journal is written for it: a language model is handed
what the fly actually saw and did, the live numbers from its own token page,
and the text of pages it was allowed to read, and asked to write in the first
person. Every number in the result is checked against that packet before it
is kept. A draft with a number that is not in the packet, or with trading
language, is dropped. There is no second draft and nothing edits the text.

Nothing a person wrote can reach the fly's account through this file: the
offline "stub" model writes only to its own journal and never posts, the
narrator's memory is validated before it is kept, and xpost.py has no command
that posts arbitrary text.

The fly also has a room of its own, served by the roamer on loopback. The
narrator is not shown it: there are no rules yet for how it should report
what happens in there, so every loopback URL and every /backroom path is
stripped out of the packet, out of the reading menu and out of the set of
links an entry may name.

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

import xpost

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
    # How it came to exist, so the narrator can say exactly this and no more.
    "how": ("Its descending neurons moved the cursor over the launch form in a "
            "separate wallet-holding browser; a script completed the fields it "
            "missed and pressed the button; humans chose the name and the image. "
            "It did not choose to launch anything and it does not control the coin."),
}

# What the brain is made of. Fixed by the dataset and the model, so constants;
# in the packet so the narrator may write them and the checker can ground them.
BRAIN = {
    "neurons": 165122,
    "synapses": 10228000,
    "retina_columns": 892,
    "retina_pixels_per_side": 30,
    "kc_mbon_synapses": 44042,
    "source": "HHMI Janelia FlyEM male CNS, CC-BY",
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

# Trading language, as stems so inflections do not slip past. A draft matching
# any of these is dropped; the narrator is a journal, not a promoter.
BANNED_RE = [re.compile(p, re.I) for p in (
    r"\bbuy(?:s|ing|ers?)?\b", r"\bbought\b", r"\bsell(?:s|ing|ers?)?\b", r"\bsold\b",
    r"\bpump(?:s|ed|ing)?\b", r"\bdump(?:s|ed|ing)?\b", r"\bmoon(?:s|ed|ing)?\b",
    r"\bto the moon\b", r"\bape(?:d|s|ing)?\b", r"\brug(?:s|ged|pulls?)?\b", r"\bjeets?\b",
    r"\bbags?\b", r"\bdips?\b", r"\bcheap\b", r"\bundervalued\b", r"\baccumulat(?:e|es|ed|ing)\b",
    r"\bath\b", r"\ball[- ]time high\b", r"\bup only\b", r"\bpresale\b", r"\bairdrops?\b",
    r"\bgiveaways?\b", r"\blink in bio\b", r"\bguaranteed?\b",
    r"\bwill (?:go up|rise|grow|climb|double|triple|pump|moon)\b", r"\bgoing up\b", r"\bwent up\b",
    r"\bfinancial advice\b", r"\bdyor\b", r"\bnfa\b", r"\bwagmi\b", r"\bngmi\b", r"\blambo\b",
    r"\bgems?\b", r"\bsend it\b", r"\bfomo\b", r"\binvest now\b", r"\bhodl\b", r"\bbullish\b",
    r"\bbearish\b", r"\bprice targets?\b", r"\bdon'?t miss\b", r"\blast chance\b", r"\bget in\b",
    r"\bload up\b", r"\blfg\b", r"\bgm\b", r"\d+(?:\.\d+)?x\b",
)]

# a number the way it appears in prose: 1,234  25.4M  327k  0.1%  59614342
NUM_RE = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?\s?(?:%|[kKmMbB](?![a-zA-Z]))?")
# the same, for checking a draft: a digit after a letter or dot still counts
POST_NUM_RE = re.compile(r"(?<![\d,])\d[\d,]*(?:\.\d+)?\s?(?:[kKmMbB](?![a-zA-Z]))?")
PCT_RE = re.compile(r"\d[\d,]*(?:\.\d+)?\s?(?:%|percent\b|per cent\b)", re.I)
HEX_RE = re.compile(r"0x[0-9a-f]{6,}", re.I)
# Numbers as words. A lone word ("four") is read as its digit and grounded
# like one; a compound ("fifteen hundred", "twenty three") or a vague
# quantity ("thousands", "half", "a couple") cannot be grounded and fails.
WORD_NUMS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
             "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
             "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
             "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
             "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
             "million": 1000000, "billion": 1000000000}
SINGLE_WORD_RE = re.compile(r"\b(" + "|".join(WORD_NUMS) + r")\b", re.I)
VAGUE_WORD_RE = re.compile(
    r"\b(?:hundreds|thousands|millions|billions|dozens?|halves|half|quarters?|twice|thrice|"
    r"couple|several)\b", re.I)
EMOJI_RE = re.compile(r"[\U0001F000-\U0001FAFF☀-➿]|(?<!\w)[#@][A-Za-z_]\w*")
URL_RE = re.compile(r"https?://[^\s<>\"']+|\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|net|org|online|io|xyz|app|dev)\b(?:/[^\s<>\"']*)?", re.I)

# Where a number in a sentence may come from. A figure that was only ever in
# a page excerpt must sit in a sentence that says it came from a page; one
# from the journal must sit in a sentence that looks back.
PAGE_CUES = re.compile(r"\b(?:page|pages|read|reads|says|said|article|excerpt|narrator|tells|told|according)\b", re.I)
MEMORY_CUES = re.compile(r"\b(?:day|earlier|before|yesterday|last|then|was|were|remember|journal|first|ago|once)\b", re.I)


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


def _truthy(v, default):
    s = str(v if v is not None else default).strip().lower()
    if not s:                         # blank is the same as unset
        s = str(default).strip().lower()
    return s not in ("0", "false", "no", "off")


def cfg():
    state_dir = Path(_get("FLY_STATE_DIR", str(ROOT / "build")))
    model = _get("FLY_VOICE_MODEL", "anthropic/claude-opus-5").strip()
    stub = model.lower() == "stub"
    try:
        every_h = max(0.5, float(_get("FLY_VOICE_EVERY_H", "3")))
    except ValueError:
        every_h = 3.0
    return {
        "key": _get("OPENROUTER_API_KEY"),
        "model": model,
        "stream": (_get("FLY_STREAM", "https://flybrain-production-2b26.up.railway.app")).rstrip("/"),
        "state_dir": state_dir,
        # the stub's hand-written entries never share a journal with the narrator
        "journal": state_dir / ("journal.stub.json" if stub else "journal.json"),
        "rpc": _get("FLY_RH_RPC", "https://rpc.mainnet.chain.robinhood.com"),
        "every_h": every_h,
        # dry unless FLY_VOICE_DRY is explicitly off; "true", "yes", "on" stay dry
        "dry": _truthy(_get("FLY_VOICE_DRY"), "1"),
        "prompt": Path(_get("FLY_VOICE_PROMPT", str(ROOT / "voice_prompt.md"))),
    }


# --------------------------------------------------------------------------
# the journal
# --------------------------------------------------------------------------
class Journal:
    """Everything the fly holds. Small, on disk, written atomically."""

    def __init__(self, path):
        self.path = Path(path)
        # no mood until the narrator has written one; nothing is seeded
        self.data = {"born": None, "knowledge": [], "read": [], "posts": [],
                     "mood": None, "last_cycle_at": 0}
        self.load()

    def load(self):
        if not self.path.exists():
            return
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(d, dict):
                raise ValueError("journal is not an object")
            self.data.update(d)
        except Exception as exc:
            # keep the broken file for a person to look at; never write over it
            aside = self.path.with_name(f"{self.path.name}.corrupt-{int(time.time())}")
            try:
                os.replace(self.path, aside)
            except OSError:
                pass
            say("JOURNAL UNREADABLE, moved aside and starting fresh:", str(exc)[:80], "->", aside.name)

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
        """Record the birth once. Returns True when this call set it."""
        if not self.data.get("born"):
            self.data["born"] = int(now or time.time())
            return True
        return False

    def day(self, now=None):
        born = self.data.get("born") or int(now or time.time())
        return 1 + int(((now or time.time()) - born) // 86400)

    def summary(self, now=None):
        s = {
            "day": self.day(now),
            "knowledge": self.data.get("knowledge", [])[-40:],
            "pages_read_before": [r.get("url") for r in self.data.get("read", [])][-30:],
            "earlier_entries": [p.get("text") for p in self.data.get("posts", [])][-6:],
        }
        if self.data.get("mood"):
            s["mood"] = self.data["mood"]
        return s

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
    """
    The picture that goes out with an entry - unless the fly is in its own room.

    /frame.jpg is whatever the fly is looking at this second, and that includes
    the backroom: a board of coin cards with names, tickers, market caps and a
    bright marker on every coin the paper book holds. The narrator is never
    given the room in words, and the image is the one channel that does not go
    through the packet, so the same blind spot has to be enforced where the
    picture is taken. Where the fly is, is asked for here rather than read off
    an older observation, because a post is written minutes after the packet
    and the fly may have walked in since. If that cannot be established, no
    picture goes out.
    """
    try:
        st = _json(stream + "/state", timeout=10)
    except Exception as exc:
        say("no picture: cannot tell where the fly is -", str(exc)[:70])
        return None
    if hidden_url((st or {}).get("url")):
        say("no picture: the fly is in its own room")
        return None
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
    # rounded before the narrator sees them: a 13-decimal float in the packet
    # comes back as a 13-decimal float in the entry
    for k, d in TOKEN_DECIMALS.items():
        if isinstance(t.get(k), float):
            t[k] = round(t[k], d) if d else int(round(t[k]))   # 14421514, not 14421514.0
    return t


TOKEN_DECIMALS = {"fees_earned_googl": 2, "fees_claimable_googl": 2, "googl_usd": 2,
                  "fees_usd": 0, "claimable_usd": 0, "market_cap_usd": 0,
                  "price_usd": 6, "price_googl": 8}


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
    # the room is left out of the packet entirely: not as a place it is now,
    # and not as a place it has been
    seen = [v for v in (st.get("visited") or []) if not hidden_url(v.get("url"))]
    tele = {
        "url": None if hidden_url(st.get("url")) else st.get("url"),
        "hops": st.get("hops"), "clicks": st.get("clicks"), "vetoes": st.get("vetoes"),
        "scrolled": st.get("scrolled"), "steps": st.get("steps"), "uptime_s": st.get("uptime_s"),
        "pages_this_life": st.get("hops"),
        "firing": n.get("firing"), "total": n.get("total"),
        "spikes_per_sec": n.get("spikes_per_sec"), "mean_mv": n.get("mean_mv"),
        "dn": n.get("dn"),
        # what the mushroom body is, not what it has been taught. In a build
        # with the room on, the only dopamine anywhere is a paper profit or a
        # paper loss (roam delivers none), so "rewards" and "punishments" are
        # the counts of winning and losing paper trades and "depressed" and
        # "mean_gain" are how far those trades have moved the weights. A draft
        # saying "twelve rewards have reached my mushroom body today" would
        # pass every check - the numbers would be in the packet and no banned
        # word is in the sentence - and would be a report of the backroom's
        # results, which the voice is not given and does not report.
        "learning": {"synapses": L.get("synapses")} if L else None,
        "last_visited": [{"title": v.get("title"), "url": v.get("url")} for v in seen[-8:]],
        # counts, not the roamer's log lines: those are a person's phrasing
        "blocked": st.get("blocked"),
        # the roamer stamps every state it publishes; an old stamp means the
        # browser is not producing frames and there is nothing to narrate
        "reachable": bool(st) and bool(st.get("updated")) and (now - float(st.get("updated") or 0)) < 180,
    }
    packet = {
        "now_utc": datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_h": round((now - LAUNCH["launched_unix"]) / 3600.0, 1),
        "telemetry": tele,
        "brain": dict(BRAIN),
        "token": fetch_token(),
        "launch": dict(LAUNCH),
        "wallet_eth": (lambda w: round(w, 6) if isinstance(w, float) else w)(wallet_eth(c["rpc"], LAUNCH["creator"])),
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
    if hidden_url(url):            # the fly's own room is not read to it
        return out
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


# The roamer serves the fly's own room on loopback. Until there are rules for
# how the narrator should report what happens in there, it is not shown it at
# all - a blind spot is honest, a description with no rules behind it is not.
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "0.0.0.0", "::1"}


def _split_url(url):
    """(host, path) for a URL that may carry no scheme. Host lowercased, no port."""
    s = str(url or "").strip()
    m = re.match(r"(?i)^[a-z][a-z0-9+.-]*:(//.*)$", s)    # a scheme, only with //
    if m:
        s = m.group(1)
    if s.startswith("//"):
        s = s[2:]
    if s.startswith("/"):
        return "", s                                      # a path with no host
    auth, slash, rest = s.partition("/")
    host = auth.rsplit("@", 1)[-1]
    host = re.sub(r":\d+$", "", host).strip("[]").lower()
    return host, slash + rest


def hidden_url(url):
    """True for a URL the narrator must never be given, or allowed to write."""
    if not url:
        return False
    host, path = _split_url(url)
    if host in LOOPBACK_HOSTS or host.startswith("127."):
        return True
    return path.lower().startswith("/backroom")


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
        if hidden_url(u):                          # the room is never on the menu
            continue
        if _host(u) in READABLE_HOSTS and u not in seen and all(m["url"] != u for m in menu):
            menu.append({"url": u, "title": v.get("title") or u,
                         "why": "a page the fly itself landed on"})
    if not menu:                                   # everything read: start over
        menu = [dict(a) for a in ALLOWLIST]
    return [m for m in menu if not hidden_url(m.get("url"))][:14]


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
    if x < 0:                        # prose drops the sign; the checker sees "91.5"
        return number_forms(-x)
    if abs(x - round(x)) < 1e-9:
        i = int(round(x))
        out.update({str(i), f"{i:,}"})
    # the value's own spellings, however many decimals the packet carried
    out.update({str(x), repr(x), f"{x:.10g}", f"{x:.15g}"})
    for d in (0, 1, 2, 3, 4, 6, 8):
        s = f"{x:.{d}f}".rstrip("0").rstrip(".") if d else f"{x:.0f}"
        out.add(s)
        if d in (1, 2):
            out.add(f"{x:.{d}f}")          # "1.00" as a page prints it
        try:
            out.add(f"{float(s):,.{d}f}".rstrip("0").rstrip(".") if d else f"{int(round(x)):,}")
        except Exception:
            pass
    ax = abs(x)
    if ax >= 1000:
        # rounding only to steps at least ten times finer than the value:
        # 165,122 may be "170,000", never "200,000"
        for step in (10, 100, 1000, 10000, 100000, 1000000):
            if step * 10 > ax:
                break
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


def exact_forms(v):
    """An identifier (a block, a chain id, a timestamp) may only be written as is."""
    try:
        i = int(v)
    except Exception:
        return set()
    return {str(i), f"{i:,}"}


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


# Identifiers in the launch block: allowed only spelled exactly, never rounded.
IDENTIFIERS = ("block", "chain_id", "launched_unix")
# Quantities in the launch block that may be rounded like any other number.
LAUNCH_QUANTITIES = ("supply", "creator_tax_pct", "launch_cost_eth")
# Counts that are always fine to write: "one page", "2 clicks".
FREE = {"1", "2"}


def allowed_numbers(packet):
    """
    What the narrator may write a number about, in three sets:

      live    measured this cycle: telemetry, the token page, the launch,
              hours since launch, the day, how many pages were read
      page    numbers inside the excerpts read to it this cycle
      memory  numbers in its own journal, grounded when first written

    validate() lets a live number sit anywhere, a page number only in a
    sentence that says it came from a page, and a memory number only in a
    sentence that looks back.
    """
    launch = packet.get("launch") or {}
    live_raw, page_raw, mem_raw = set(), set(), set()
    _walk_numbers({
        "telemetry": packet.get("telemetry"),
        "brain": {k: v for k, v in (packet.get("brain") or BRAIN).items() if k != "source"},
        "token": packet.get("token"),
        "launch": {k: launch.get(k) for k in LAUNCH_QUANTITIES},
        "wallet_eth": packet.get("wallet_eth"),
        "elapsed_h": packet.get("elapsed_h"),
    }, live_raw)
    # the launch date, and hours since launch as an integer too
    for m in re.findall(r"\d+", str(launch.get("launched_at", ""))):
        live_raw.add(float(m))
    if packet.get("elapsed_h") is not None:
        live_raw.add(round(float(packet["elapsed_h"])))
    _walk_numbers(packet.get("pages_read") or [], page_raw)
    j = packet.get("journal") or {}
    _walk_numbers({"knowledge": j.get("knowledge"), "earlier_entries": j.get("earlier_entries")}, mem_raw)

    live = set()
    for v in live_raw:
        live |= number_forms(v)
    for k in IDENTIFIERS:
        live |= exact_forms(launch.get(k))
    if j.get("day"):
        live.add(str(j["day"]))
    live.add(str(len(packet.get("pages_read") or [])))
    page = set()
    for v in page_raw:
        page |= number_forms(v)
    memory = set()
    for v in mem_raw:
        memory |= number_forms(v)
    pct = set()
    for k, v in launch.items():
        if k.endswith("_pct"):
            pct |= number_forms(v)
    return {"live": live | FREE, "page": page, "memory": memory, "pct": pct}


def _norm_num(tok):
    s = tok.strip().lower().replace(",", "").replace(" ", "")
    return s.rstrip("%")


# X's own count: URLs are 23, some code points weigh 2. One definition, in
# xpost, so what passes here cannot be refused there.
x_len = xpost.x_length


def _known_urls(packet):
    known = {a["url"] for a in ALLOWLIST} | {"https://flybrain.online"}
    known |= {r.get("url") for r in (packet.get("pages_read") or []) if r.get("url")}
    tele = packet.get("telemetry") or {}
    known |= {v.get("url") for v in (tele.get("last_visited") or []) if v.get("url")}
    known |= {u for u in (packet.get("allowlist") or []) if u}
    # a link to the room could only have come from a packet field that should
    # not have carried it; either way the narrator may not write it
    return {u.lower().rstrip("/") for u in known if u and not hidden_url(u)}


def validate(post, packet):
    """
    (ok, reasons). Every check is mechanical; nothing here rewrites the draft.
    """
    reasons = []
    if not isinstance(post, str) or not post.strip():
        return False, ["empty"]
    if x_len(post) > 280:
        reasons.append(f"too long: {x_len(post)} > 280")
    if EMOJI_RE.search(post):
        reasons.append("emoji, hashtag or mention")
    for rx in BANNED_RE:
        m = rx.search(post)
        if m:
            reasons.append(f"banned phrase: {m.group(0)}")

    allowed = packet.get("allowed_numbers") or allowed_numbers(packet)
    if isinstance(allowed, set):                       # an old-style flat set
        allowed = {"live": allowed | FREE, "page": set(), "memory": set(), "pct": set()}

    # links: only ones it could actually have been given
    known = _known_urls(packet)
    for u in URL_RE.findall(post):
        u2 = u.rstrip(".,;:!?)]}\"'").lower().rstrip("/")
        if not u2.startswith("http"):
            u2 = "https://" + u2
        if u2 not in known and u2.replace("https://", "http://") not in known:
            reasons.append(f"url not in packet: {u}")
    body = HEX_RE.sub(" ", URL_RE.sub(" ", post))

    for m in VAGUE_WORD_RE.findall(body):
        reasons.append(f"number as a word: {m}")
    for m in PCT_RE.findall(body):
        if _norm_num(re.sub(r"(?i)percent|per cent", "", m)) not in allowed["pct"]:
            reasons.append(f"percentage not in packet: {m.strip()}")
    body = PCT_RE.sub(" ", body)

    def grounded(tok, page_ok, memory_ok):
        if tok in allowed["live"]:
            return True
        if page_ok and tok in allowed["page"]:
            return True
        return memory_ok and tok in allowed["memory"]

    for sentence in re.split(r"(?<=[.!?])\s+", body):
        page_ok = bool(PAGE_CUES.search(sentence))
        memory_ok = bool(MEMORY_CUES.search(sentence))
        for m in POST_NUM_RE.findall(sentence):
            tok = _norm_num(m)
            if tok and not grounded(tok, page_ok, memory_ok):
                reasons.append(f"number not in packet: {m.strip()}")
        # number words: a compound or one glued to a digit cannot be grounded;
        # a lone word is read as its digit ("four" -> "4") and checked the same
        words = list(SINGLE_WORD_RE.finditer(sentence))
        for i, w in enumerate(words):
            before = sentence[max(0, w.start() - 2):w.start()]
            after = sentence[w.end():w.end() + 2]
            glued = (i > 0 and not sentence[words[i - 1].end():w.start()].strip(" -")) \
                or (i + 1 < len(words) and not sentence[w.end():words[i + 1].start()].strip(" -")) \
                or re.search(r"\d\s?$", before) or re.match(r"^\s?\d", after) \
                or re.search(r"\ba\s$", sentence[:w.start()], re.I)      # "a hundred"
            word = w.group(1).lower()
            if glued:
                reasons.append(f"number as a word: {w.group(0)}")
            elif word != "one" and not grounded(str(WORD_NUMS[word]), page_ok, memory_ok):
                reasons.append(f"number not in packet: {w.group(0)}")
    return (not reasons), reasons


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------
RULES_BLOCK = """
Machine rules, appended. A checker enforces every one of these mechanically
and discards the entry on any failure; there is no second draft.
- Every number must appear in the packet, as digits. Never a number as a
  word, never arithmetic on packet numbers, never a percentage except
  launch.creator_tax_pct, never a multiplier.
- A value shown as "unmeasured" was not observed: do not mention that
  quantity at all.
- A number from a page excerpt may only appear in a sentence that says it
  came from a page; a number from the journal only in a sentence that looks
  back.
- No trading language, predictions, hype, calls to action, promises, or
  claims that the fly controls or wants anything for the token.
- Only pages in pages_read were read; only journal is remembered; nothing
  else happened. No URLs except from pages_read or allowlist; no hashtags,
  emoji or mentions.
- Do not reuse any sentence or structure from earlier_entries or from this
  prompt.
- post under 280 characters; a URL counts as 23.
- Reply with JSON only, exactly: {"post": "...", "learned": ["..."],
  "mood": "...", "wants_to_read": ["https://..."]}
"""


def system_prompt(c):
    # No silent fallback: the reviewed prompt is the voice, and a missing
    # file is a deployment bug that must be loud, not a different narrator.
    try:
        base = c["prompt"].read_text(encoding="utf-8")
    except Exception as exc:
        raise RuntimeError(f"narrator prompt unreadable: {c['prompt']} ({exc})") from exc
    if len(base.strip()) < 200:
        raise RuntimeError(f"narrator prompt looks empty: {c['prompt']}")
    return base + "\n" + RULES_BLOCK


class ConfigError(RuntimeError):
    """A failure that a retry cannot fix: bad key, no credit, wrong model."""


def call_model(c, system, user, temperature=0.9):
    if not c.get("key"):
        raise ConfigError("OPENROUTER_API_KEY is not set")
    last = "openrouter: no reply"
    for pause in (0, 20, 60, 120):
        if pause:
            say(f"openrouter retry in {pause}s: {last}")
            time.sleep(pause)
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {c['key']}",
                         "HTTP-Referer": "https://flybrain.online", "X-Title": "flybrain"},
                json={"model": c["model"], "temperature": temperature, "max_tokens": 900,
                      "messages": [{"role": "system", "content": system},
                                   {"role": "user", "content": user}]},
                timeout=180)
        except requests.RequestException as exc:
            last = f"openrouter request failed: {str(exc)[:120]}"
            continue
        if r.status_code in (400, 401, 402, 403, 404):
            raise ConfigError(f"openrouter {r.status_code}: {r.text[:200]}")
        if r.status_code != 200:
            last = f"openrouter {r.status_code}: {r.text[:200]}"
            continue
        try:
            j = r.json()
        except ValueError:
            last = "openrouter: body is not JSON"
            continue
        if j.get("error"):
            last = f"openrouter error: {str(j['error'])[:200]}"
            continue
        try:
            content = j["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            last = f"openrouter: unexpected body {str(j)[:200]}"
            continue
        if not content or not str(content).strip():
            last = "openrouter: empty completion"
            continue
        return content
    raise RuntimeError(last)


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
    # The stub is a test fixture. It learns nothing and has no mood, so its
    # hand-written lines never seed a journal - and it has its own journal file.
    unread = [m["url"] for m in menu][:2]
    return {"post": post, "learned": [], "mood": None, "wants_to_read": unread}


def _mark_unmeasured(o):
    """None in the packet becomes the word "unmeasured", which the prompt defines."""
    if isinstance(o, dict):
        return {k: _mark_unmeasured(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_mark_unmeasured(v) for v in o]
    return "unmeasured" if o is None else o


def reflect(c, journal, packet, readings, menu):
    packet = dict(packet)
    packet["pages_read"] = [{"url": r["url"], "title": r["title"], "excerpt": r["text"][:1800]} for r in readings]
    packet["journal"] = journal.summary()
    # the menu's reasons are a person's words; the narrator gets titles only
    packet["dig_menu"] = [{"url": m["url"], "title": m["title"]} for m in menu]
    packet.pop("allowed_numbers", None)
    if c["model"].strip().lower() == "stub":
        return stub_reflect(packet, readings, menu)
    user = ("Observation packet:\n" + json.dumps(_mark_unmeasured(packet), ensure_ascii=False, indent=0)
            + "\n\nWrite the next journal entry. JSON only.")
    text = call_model(c, system_prompt(c), user)
    try:
        return parse_json_block(text)
    except Exception:
        # a formatting failure, not a content one: ask for the same entry as JSON
        text = call_model(c, system_prompt(c), user + "\n\nYour last reply was not valid JSON. "
                          "Reply with the JSON object only.", temperature=0.6)
        return parse_json_block(text)


# --------------------------------------------------------------------------
# a cycle
# --------------------------------------------------------------------------
def is_stub(c):
    return str(c.get("model", "")).strip().lower() == "stub"


def clean_mood(v):
    """One or two plain lowercase words, or nothing."""
    m = " ".join(str(v or "").split()).lower()
    return m if re.fullmatch(r"[a-z]+(?: [a-z]+)?", m) else None


def run_once(c, dry=None, now=None):
    now = now or time.time()
    dry = c["dry"] if dry is None else dry
    if is_stub(c):
        dry = True                    # hand-written text never leaves the machine
    journal = Journal(c["journal"])
    if journal.begin(now):
        journal.save()                # the birth is on disk before anything else

    packet = observe(c, now)
    if not packet["telemetry"]["reachable"]:
        # nothing to narrate: the browser is loading or the roamer is down.
        # No model call, no stamp, so the loop tries again shortly.
        say("roamer not ready; no entry")
        return {"dropped": True, "reasons": ["roamer unreachable or stale"]}
    if not dry:
        posts = xpost.read_ledger()
        if posts is not None and xpost.posted_last_24h(posts) >= xpost.MAX_PER_DAY:
            say("daily cap reached; no entry")
            journal.data["last_cycle_at"] = int(now)
            journal.save()
            return {"dropped": True, "reasons": ["daily cap"]}
    # the cadence is measured from here, on disk, so a restart does not
    # produce an extra entry
    journal.data["last_cycle_at"] = int(now)
    journal.save()
    packet["journal"] = journal.summary(now)
    menu = dig_menu(c, journal, packet)

    # first pass: what does it want read? then read it, then write.
    readings = []
    try:
        first = reflect(c, journal, packet, [], menu)
        wants = [u for u in (first.get("wants_to_read") or []) if isinstance(u, str)]
    except ConfigError:
        raise
    except Exception as exc:
        say("first pass failed:", str(exc)[:120])
        wants = []
    allowed_urls = {m["url"] for m in menu}
    if is_stub(c):
        wants = [m["url"] for m in menu][:2]
    for u in wants[:2]:
        if u in allowed_urls and not hidden_url(u):
            readings.append(read_page(u))
    readings = [r for r in readings if r["text"]]

    packet["allowed_numbers"] = None                   # rebuilt below with readings
    out = reflect(c, journal, packet, readings, menu)
    check_packet = dict(packet)
    check_packet["pages_read"] = [{"url": r["url"], "title": r["title"], "excerpt": r["text"]} for r in readings]
    check_packet["allowed_numbers"] = allowed_numbers(check_packet)

    post = str(out.get("post", "")).strip()
    ok, reasons = validate(post, check_packet)
    if not ok:
        # dropped, not fixed: no second draft, and the reasons stay here.
        # A dropped draft gives back most of the cadence so the next try is
        # sooner than a full interval, bounded by RETRY_AFTER_DROP_S.
        say("entry dropped:", "; ".join(reasons))
        journal.data["last_cycle_at"] = int(now - c["every_h"] * 3600 + RETRY_AFTER_DROP_S)
        journal.note_read(readings, now)
        journal.save()
        return {"dropped": True, "reasons": reasons, "draft": post}

    # what it keeps is checked the same way as what it says
    learned, dropped = [], []
    for f in (out.get("learned") or [])[:3]:
        f = " ".join(str(f).split())[:240]
        (learned if f and validate(f, check_packet)[0] else dropped).append(f)
    if dropped:
        say("not kept (ungrounded):", " | ".join(d[:80] for d in dropped))
    mood = clean_mood(out.get("mood"))

    entry = {"at": int(now), "day": journal.day(now), "text": post, "posted": False, "x_id": None,
             "mood": mood, "learned": learned, "read": [r["url"] for r in readings]}
    if not dry:
        try:
            res = xpost.publish(post, fetch_frame(c["stream"]))
            entry["posted"] = bool(res.get("id"))
            entry["x_id"] = res.get("id")
            entry["x_result"] = res
            say("x:", res)
        except Exception as exc:
            entry["x_error"] = str(exc)[:200]
            say("x failed:", str(exc)[:200])
    journal.add_post(entry)
    journal.learn(learned)
    journal.note_read(readings, now)
    if mood:
        journal.data["mood"] = mood
    journal.save()
    say(f"[day {entry['day']}] {'posted' if entry['posted'] else 'dry'}: {post}")
    return entry


RETRY_AFTER_DROP_S = 45 * 60      # a dropped draft tries again after this
FIRST_ENTRY_EVERY_S = 15 * 60     # until the journal has one entry, try this often


def take_nudge(path):
    """
    True once if an operator left a file named `nudge` in the state dir; the
    file is removed. It only moves the clock: the entry itself is written,
    checked and posted exactly as every other one.
    """
    try:
        if Path(path).exists():
            Path(path).unlink()
            return True
    except OSError:
        pass
    return False


def next_wait(journal, every_h, now=None):
    """Seconds until the next cycle is due, from the stamp on disk."""
    now = time.time() if now is None else now
    cadence = every_h * 3600 if journal.data.get("posts") else min(every_h * 3600, FIRST_ENTRY_EVERY_S)
    return (journal.data.get("last_cycle_at") or 0) + cadence - now


def loop(c):
    say(f"voice loop: every {c['every_h']} h, model {c['model']}, dry={c['dry']}")
    sd = str(c["state_dir"])
    say(f"state_dir: {sd} (mount point: {os.path.ismount(sd)}), journal: {c['journal'].name}, "
        f"ledger: {xpost.ledger_path()}")
    if not is_stub(c):
        system_prompt(c)              # fail now, loudly, if the prompt is missing
    nudge = Path(c["state_dir"]) / "nudge"
    said = None
    while True:
        j = Journal(c["journal"])
        wait = next_wait(j, c["every_h"])
        nudged = take_nudge(nudge)
        if nudged:
            say("nudged: writing the next entry now")
            wait = 0
        if wait > 0:
            mins = int(wait // 60)
            if said is None or abs(said - mins) >= 10 or mins < 2:
                say(f"next entry in {mins} min")
                said = mins
            time.sleep(min(wait, 60))
            continue
        said = None
        try:
            res = run_once(c)
            if nudged and isinstance(res, dict) and res.get("reasons") == ["roamer unreachable or stale"]:
                nudge.touch()             # keep the nudge until the roamer can be narrated
        except ConfigError as exc:
            say("voice cannot run until this is fixed:", str(exc)[:200])
            time.sleep(1800)
        except Exception:
            say("cycle failed:")
            traceback.print_exc()
        time.sleep(60)


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
    if a.live and is_stub(c):
        sys.exit("the stub writes test text; it is never posted")
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
