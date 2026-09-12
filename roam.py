"""
The fly, loose on the internet.

A page is screenshotted, sampled through the fly's 892 retinotopic hex columns
into L1 and L2, and 165,122 neurons integrate. DNa02's left-right asymmetry
moves the cursor sideways, DNa01 moves it up the page, MDN reverses, and DNp09
- the stopping neuron - clicks. If the click lands on a link, the fly is
somewhere new. Nothing chooses where it goes. That is the whole point.

Be clear about what this is. A fly brain has no language, no goals and no plan.
It cannot read a page, decide a destination or want anything. What it has is a
real nervous system reacting to light, and what comes out is a cursor. Calling
it browsing is fair; calling it deciding is not.

RAILS, and why each one is here
-------------------------------
* No wallet. This browser never gets a key, a provider or an extension. A
  random clicker with a signing key is how you lose everything, so the roaming
  browser and the launching browser have nothing in common but the brain.
* No typing. The fly has no keyboard at all - it cannot fill a field, write a
  message or answer a prompt.
* No downloads, no popups, no file dialogs.
* A click is checked before it lands: anything that looks like a submit, an
  upload, a payment or a sign-in is vetoed and logged as a veto.
* A URL blocklist, checked on every navigation. This is a public live stream;
  an unfiltered random walk will eventually find something nobody wants
  broadcast. Blocked pages bounce straight back.
* A hop budget. When it runs out the fly is put back on a seed page, so a dead
  end does not become a permanent home.

THE BACKROOM (FLY_BACKROOM=1 only)
----------------------------------
With FLY_BACKROOM=1 one more place exists: a page this project serves itself at
/backroom, holding a grid of coin cards. A share of the restarts that would put
the fly back on a web seed put it in there instead (FLY_BACKROOM_SHARE), and it
walks that page with the same eye, the same descending neurons and the same
stopping rule it uses everywhere else. What changes is what a stop means: in the
room a stop on a card is read as a commit and becomes one HTTP POST to the paper
executor on loopback. This build is paper - nothing here signs anything - and
the fence is tightened rather than loosened for it: the only address on this
machine the fly may ever open is that one page, and nothing else on loopback.
Without FLY_BACKROOM=1 none of it exists and the roam is exactly what it was.

Dopamine now comes from one thing only. It used to be handed out here for
finding somewhere new, hitting a wall or reaching for a vetoed control, which
taught the mushroom body about the shape of the web. Profit and loss are the
only lessons left, and they are delivered in backroom.py where they are
measured.

  py roam.py                 open http://localhost:4660 and press START
  py roam.py --headful       watch the real browser too
"""
import argparse
import asyncio
import base64
import io
import ipaddress
import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

import calibration
from flysim import FlyBrain
from envcfg import load_env
from mushroom import MushroomBody
from flyeye import FlyPilot

ROOT = Path(__file__).parent
OUT = ROOT / "build"


def _atomic_write(path, data: bytes):
    """Write then rename, so /state never reads a half-written file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)

# Link-rich, text-heavy, safe places to be dropped into. The fly leaves them
# on its own within a few clicks; these only decide where a life starts.
SEEDS = [
    # Weighted toward Special:Random on purpose. Every time a hop budget runs
    # out the fly is put back on a seed, so if the seeds are a short fixed
    # list it lands on the same few pages forever - which is exactly what
    # happened. Special:Random is a different article every single time, so a
    # reset becomes somewhere new rather than somewhere familiar.
    "https://en.wikipedia.org/wiki/Special:Random",
    "https://en.wikipedia.org/wiki/Special:Random",
    "https://en.wikipedia.org/wiki/Special:Random",
    "https://en.wikipedia.org/wiki/Special:Random",
    "https://commons.wikimedia.org/wiki/Special:Random",
    "https://en.wikisource.org/wiki/Special:Random",
    "https://en.wikiquote.org/wiki/Special:Random",
    "https://www.gutenberg.org/browse/scores/top",
    "https://openlibrary.org/",
    "https://xkcd.com/",
    # and the chain it launched its own token on
    "https://www.ponsfamily.com/launchpad/explore",
    "https://www.ponsfamily.com/launchpad/0x4eb990547bce4a982432ca88cf5fae7eed1a2d35",
    "https://robinhoodchain.blockscout.com/txs",
]

# Hacker News and arXiv were seeds and had to go. Both are link dead ends
# behind a fence: almost every link on them points at a domain that is not
# allowed, so a click there goes nowhere, the budget expires and the fly is
# bounced back to a seed. They looked like rich pages and were traps.


# Checked against every URL the browser tries to commit to.
BLOCK = re.compile(
    r"(porn|xxx|adult|nsfw|escort|hentai|onlyfans|camsoda|chaturbate"
    r"|casino|bet365|poker|gambl|lottery"
    r"|checkout|/cart|/pay|payment|billing|invoice|subscribe"
    # it roams trading sites now, where every other control is a trade
    r"|/buy|/sell|/swap|/trade|connect-wallet|/deposit|/withdraw"
    r"|signin|sign-in|login|log-in|signup|sign-up|register|/auth"
    r"|password|passwd|account/delete|unsubscribe"
    r"|\.exe$|\.dmg$|\.msi$|\.apk$|\.zip$|\.torrent$|magnet:"
    r"|\.epub|\.mobi|\.pdf$|\.iso$|/download|kindle"
    r"|mailto:|tel:)", re.I)

# A keyword blocklist cannot be made complete - "p0rn" walks straight through
# it - so it is the second line of defence, not the first. The first is this:
# the fly stays inside a set of domains unless someone deliberately opens it.
# Wikipedia alone is millions of pages that link everywhere, so this is still a
# real roam; it is just a roam with a fence.
ALLOW = {
    "en.wikipedia.org", "en.m.wikipedia.org", "commons.wikimedia.org",
    "en.wikisource.org", "en.wikiquote.org", "en.wikibooks.org",
    "www.wikidata.org", "species.wikimedia.org",
    "news.ycombinator.com",
    "www.gutenberg.org", "gutenberg.org",
    "openlibrary.org",
    "xkcd.com", "www.xkcd.com",
    "arxiv.org", "www.arxiv.org",
    "www.ponsfamily.com", "ponsfamily.com",
    "robinhoodchain.blockscout.com",
}
OPEN = load_env().get("FLY_ROAM_OPEN") == "1"

# Everything about the room is read when it is asked for rather than at import,
# because the supervisor hands each child its own environment and a test has to
# be able to change it. load_env merges every FLY_* variable of the process over
# .env, so these work from the environment run_all.py builds.
BACKROOM_SHARE = 0.35      # chance a reset puts the fly in the room (CHOSEN)
BACKROOM_STEPS = 120       # steps of one visit before the reset (CHOSEN)
EXECUTOR_PORT = 4671
# Every coin's smell scaled toward the same total. DoOR has no dose axis, so how
# strong a coin smells is an accident of which odorant its words name; scaling
# each profile toward one total takes most of that accident out - not all of it,
# because no receptor may respond above 1.0, so a coin landing on one glomerulus
# totals 1.0 where one spread over twenty totals 2.0 (olfaction.Nose.smell says
# this in full, and every smell carries its achieved `loudness`).
# MEASURED 2026-09-12: without it
# geosmin alone fires 42% of the Kenyon cells against isopentyl acetate's 4.9%,
# so a lesson about a loud coin lands on five times as many synapses and swamps
# the quiet ones. This is the setting the offline screen passed on
# (backroom_screen.py --norm 2.0, build/backroom_screen.json). CHOSEN, and it
# changes nothing outside the room, which is the only place the nose is used.
EQUAL_SNIFF = 2.0
# Names that mean this machine without being addresses. Everything else is
# decided by classifying the address itself, in this_machine().
LOCAL_NAMES = {"localhost"}


def env(name, default=""):
    v = load_env().get(name)
    return default if v is None or v == "" else str(v)


def backroom_on():
    """The room exists only when someone turned it on."""
    return env("FLY_BACKROOM") == "1"


def backroom_share():
    try:
        return min(1.0, max(0.0, float(env("FLY_BACKROOM_SHARE", BACKROOM_SHARE))))
    except (TypeError, ValueError):
        return BACKROOM_SHARE


def backroom_steps():
    try:
        return max(1, int(float(env("FLY_BACKROOM_STEPS", BACKROOM_STEPS))))
    except (TypeError, ValueError):
        return BACKROOM_STEPS


def roam_port():
    return int(STATE.get("port") or os.environ.get("PORT") or 4660)


def backroom_url():
    """The one address on this machine the fly is allowed to open."""
    return f"http://127.0.0.1:{roam_port()}/backroom"


def executor_url():
    try:
        port = int(env("FLY_EXECUTOR_PORT", EXECUTOR_PORT))
    except (TypeError, ValueError):
        port = EXECUTOR_PORT
    return f"http://127.0.0.1:{port}"


def state_dir():
    return Path(env("FLY_STATE_DIR") or OUT)


def is_backroom(url):
    """
    True only for the room's own page on this process's own port.

    Not a prefix test. /backroom/board.json, /state and /frame.jpg are on the
    same socket and none of them is a place for the fly to be; the page it walks
    is exactly one URL.
    """
    try:
        from urllib.parse import urlparse
        u = urlparse(str(url))
    except Exception:
        return False
    if u.scheme != "http" or (u.hostname or "").lower() != "127.0.0.1":
        return False
    if (u.port or 80) != roam_port() or u.query or u.fragment:
        return False
    return u.path.rstrip("/") == "/backroom"


def this_machine(host):
    """
    True for an address that is this machine, or a network this machine sits
    on, however it is spelled.

    This used to be four literal strings - 127.0.0.1, localhost, ::1, 0.0.0.0 -
    which is not the loopback network. Every other spelling fell through to the
    fence, and with the fence down (FLY_ROAM_OPEN=1) 127.0.0.2, ::ffff:127.0.0.1,
    169.254.169.254 - the cloud metadata service - and anything on 10/172.16/192.168
    were all treated as ordinary web addresses. The test is the address itself,
    so the room stays the single exception to "this machine is not the web".
    """
    h = str(host or "").strip().strip("[]").lower()
    if not h:
        return False
    if h in LOCAL_NAMES or h.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(h)
    except ValueError:
        return False
    if getattr(ip, "ipv4_mapped", None) is not None:
        ip = ip.ipv4_mapped
    return bool(ip.is_loopback or ip.is_link_local or ip.is_private
                or ip.is_reserved or ip.is_unspecified or ip.is_multicast)


def allowed_host(url):
    """Open mode drops the fence and leaves only the blocklist behind it."""
    try:
        from urllib.parse import urlparse
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return False
    if this_machine(host):
        # This machine, and the private networks around it, are not the web.
        # The room is the single exception, and only while it is switched on -
        # open mode does not widen this.
        return backroom_on() and is_backroom(url)
    if OPEN:
        return True
    return host in ALLOW


def next_place(rng, room_open=True):
    """
    Where a restart puts the fly: (url, what to call it).

    With the room off this is what it always was, one draw from SEEDS, and the
    random stream is untouched - nothing else is drawn. With the room on, that
    many restarts in FLY_BACKROOM_SHARE go through the door instead.
    `room_open` is False when the room was switched on but could not be built
    (no mushroom body, say): an unbuilt room is a page with nothing behind it,
    so the fly is not sent there.
    """
    if backroom_on() and room_open and rng.random() < backroom_share():
        return backroom_url(), "the backroom"
    return rng.choice(SEEDS), "a seed"


# Checked against the element under the cursor before a click is allowed.
VETO = re.compile(
    r"(submit|upload|sign in|sign up|log in|log out|subscribe|buy|purchase"
    r"|sell|swap|trade|connect wallet|approve|confirm|bridge|stake"
    r"|checkout|pay |donate|delete|remove|report|flag|send|post|reply"
    r"|comment|password|credit card)", re.I)

BLOB_TOKEN = load_env().get("FLY_BLOB_TOKEN", "")
BLOB_BASE = load_env().get("FLY_BLOB_BASE", "")
BLOB_EVERY = 0.5          # seconds between pushes; the run does not wait on it
_last_push = {"at": 0.0}


def blob_put(path, data, ctype):
    """
    Push one object to the public store the site reads.

    Overwrites the same pathname every time, so the site has a stable URL and
    no cleanup to do. Failures are swallowed on purpose: the fly roaming and
    the world watching are separate concerns, and a flaky upload must not stop
    a run.
    """
    import urllib.request
    req = urllib.request.Request(
        f"https://blob.vercel-storage.com/{path}", method="PUT", data=data,
        headers={"authorization": f"Bearer {BLOB_TOKEN}",
                 "x-content-type": ctype,
                 "x-add-random-suffix": "0",
                 "x-allow-overwrite": "1",
                 "x-cache-control-max-age": "0",
                 "x-api-version": "7"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.status


TUNNEL = {"url": None, "proc": None}
CFD = Path(os.path.expanduser("~/.claude/tools/cloudflared/cloudflared.exe"))


def start_tunnel(port):
    """
    Put the roamer's WebSocket on the public internet.

    Object storage is not a stream - the site could only ever poll it - so the
    live screen needs a socket a browser can open. A quick tunnel gives one
    without an account; the address is random and changes every run, so the fly
    publishes whatever it got alongside its state and the page reads it from
    there rather than having it hardcoded anywhere.
    """
    import re as _re
    import subprocess
    import threading

    if os.environ.get("FLY_TUNNEL") == "0":
        # a local run publishes nothing at all: no tunnel, so no address to
        # write anywhere, so nobody outside this machine can watch it
        say("FLY_TUNNEL=0 - this run stays on this machine")
        return
    if not CFD.exists():
        say("no cloudflared - the public feed stays on the slow path")
        return

    proc = subprocess.Popen(
        [str(CFD), "tunnel", "--no-autoupdate", "--url",
         f"http://127.0.0.1:{port}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace", bufsize=1)
    TUNNEL["proc"] = proc

    def watch():
        for line in proc.stdout:
            m = _re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
            if m and not TUNNEL["url"]:
                TUNNEL["url"] = m.group(0)
                say("tunnel open:", TUNNEL["url"])

    threading.Thread(target=watch, daemon=True).start()


LIVE_REPO = "fruitflydev/flycoinrh"
LIVE_PATH = "site/web/live.json"
_addr = {"url": None, "at": 0.0}


def publish_address():
    """
    Tell the world where the tunnel is.

    The quick tunnel's address is random and changes every run, so the page
    cannot have it hardcoded. This writes it to one small file in the public
    repo, which the page reads. It is written only when the address actually
    changes, so a long run makes no commits at all.
    """
    import base64 as _b64
    import urllib.request

    tok = load_env().get("FLY_GH_TOKEN", "")
    url = TUNNEL["url"]
    if not tok or not url or url == _addr["url"]:
        return
    if time.time() - _addr["at"] < 30:
        return
    _addr["at"] = time.time()

    body = json.dumps({"stream": url, "at": int(time.time())}, indent=1) + chr(10)
    api = f"https://api.github.com/repos/{LIVE_REPO}/contents/{LIVE_PATH}"

    def call(method, payload=None):
        req = urllib.request.Request(
            api, method=method,
            data=json.dumps(payload).encode() if payload else None,
            headers={"Authorization": f"Bearer {tok}",
                     "Accept": "application/vnd.github+json",
                     "User-Agent": "flybrain"})
        with urllib.request.urlopen(req, timeout=25) as r:
            return json.loads(r.read() or b"{}")

    try:
        sha = None
        try:
            sha = call("GET").get("sha")
        except Exception:
            pass
        payload = {"message": "the fly moved house",
                   "content": _b64.b64encode(body.encode()).decode()}
        if sha:
            payload["sha"] = sha
        call("PUT", payload)
        _addr["url"] = url
        say("published tunnel address")
    except Exception as exc:
        say("could not publish address:", str(exc)[:90])


def blob_del(path):
    """Drop one object. Old frames are litter, not history."""
    import urllib.request
    try:
        req = urllib.request.Request(
            "https://blob.vercel-storage.com/delete", method="POST",
            data=json.dumps({"urls": [f"{BLOB_BASE}/{path}"]}).encode(),
            headers={"authorization": f"Bearer {BLOB_TOKEN}",
                     "content-type": "application/json",
                     "x-api-version": "7"})
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass


app = FastAPI()
# the public page reads /state and /frame.jpg from a different origin
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"],
                   allow_headers=["*"])
STATE = {"brain": None, "pilot": None, "running": False, "room": None}
CLIENTS = set()


async def broadcast(msg):
    """
    Tell every watcher at once.

    The fly does not roam because somebody is watching and does not stop when
    they leave, so a viewer is a subscriber and never a controller.
    """
    dead = []
    text = json.dumps(msg)
    for ws in list(CLIENTS):
        try:
            await ws.send_text(text)
        except Exception:
            dead.append(ws)
    for ws in dead:
        CLIENTS.discard(ws)


def say(*parts):
    try:
        print(*parts, flush=True)
    except Exception:
        try:
            print(*[str(p).encode("ascii", "replace").decode() for p in parts],
                  flush=True)
        except Exception:
            pass


def load_brain():
    """
    The brain, once, with the calibration measured in calibration.py.

    The stock simulator ignites: any odour fires almost every Kenyon cell, so
    nothing can be learned about one smell rather than another. The gains below
    are the measured setting, and every run in this process - roaming and room
    alike - is made with them, because a fly that saw the world one way and
    learned in another would be two animals.
    """
    if STATE["brain"] is None:
        say("loading the connectome ...")
        fb = FlyBrain()
        STATE["brain"] = fb
        STATE["pilot"] = FlyPilot(fb, sim_steps=60)
        STATE["xy"] = soma_xy(fb)
        STATE["gains"] = calibration.gains_for(fb, calibration.CHOSEN)
        say(f"brain ready: {len(fb.bodies):,} neurons")
        say(f"calibration: {calibration.CHOSEN}")
        try:
            # The store is named here rather than left to mushroom.py, which
            # resolves FLY_STATE_DIR from the process environment alone. This
            # process resolves it through .env as well (state_dir -> env ->
            # launch.load_env), and a run started by hand - `py executor.py`
            # then `py roam.py`, with FLY_STATE_DIR in .env and not exported -
            # otherwise wrote the ledger, the room and the looks to the
            # configured directory and everything the fly had learned to the
            # repo's build/. On a volume-backed host that is the one file the
            # volume exists for, outside the volume.
            STATE["mb"] = MushroomBody(fb, calibration=calibration.CHOSEN,
                                       store=state_dir() / "mb_gains.v2.npz")
            st = STATE["mb"].stats()
            say(f"mushroom body: {st['synapses']:,} KC->MBON synapses "
                f"({st['reward_side']:,} reward / {st['punish_side']:,} punish), "
                f"{st['depressed']:,} already depressed")
        except FileNotFoundError as exc:
            # build/mb_sides.json is derived and gitignored, so an image can be
            # built without it. That is a missing learning circuit, not a
            # missing fly: roaming needs none of it. It must be loud, and the
            # room refuses to open without one rather than pretending to learn.
            STATE["mb"] = None
            say("NO MUSHROOM BODY:", str(exc)[:200])
            say("the fly roams, but nothing can be learned until mb_sides.py has run")
    return STATE["brain"], STATE["pilot"]


def load_room():
    """
    The room, built the first time it is needed and only when it is on.

    backroom.py is imported here rather than at the top of the file so that a
    roamer without FLY_BACKROOM=1 pulls in nothing of the coin side at all -
    not the nose, not the listing, not a single chain library.
    """
    if STATE.get("room") is None and backroom_on():
        import backroom
        from olfaction import Nose
        fb, pilot = load_brain()
        if STATE.get("mb") is None:
            say("the backroom stays shut: there is no mushroom body for it to teach")
            return None
        nose = Nose(fb, equal_sniff=EQUAL_SNIFF)
        nose.max_hz = calibration.SETTINGS[calibration.CHOSEN]["odour_max_hz"]
        STATE["room"] = backroom.Room(
            fb, pilot, STATE["mb"], nose, STATE["gains"], state_dir(),
            executor_url(), env("FLY_INTENT_TOKEN"))
        say(f"the backroom is open: {backroom_url()} -> paper executor at "
            f"{executor_url()}, {nose.cells:,} receptor neurons")
    return STATE.get("room")


def open_room():
    """
    load_room(), with every way of failing degraded to a shut room.

    load_room already degrades when there is no mushroom body: it says so and
    returns None, and next_place then stops sending the fly to the room. Every
    other failure inside it was unhandled, and the call sits inside roam()
    after Chromium, the context and the page are up and before the fly has
    opened a single page. An image without the DoOR tables raises
    FileNotFoundError there; the exception escapes roam(), begin()'s loop logs
    it and sleeps six seconds, and that repeats for ever - a blank stream and a
    service that answers /status while doing nothing at all. A missing dataset
    costs the room, not the roam.
    """
    try:
        return load_room()
    except Exception as exc:
        say("the backroom stays shut:", f"{type(exc).__name__}: {str(exc)[:160]}")
        return None


def backroom_state():
    """
    What the room and the paper book look like right now.

    The book is the executor's own file, read from disk: this process writes
    nothing of it and could not, since it holds no ledger.
    """
    room = STATE.get("room")
    book = None
    try:
        p = state_dir() / "backroom" / "public" / "public.json"
        if p.exists():
            book = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        book = None
    return {"enabled": backroom_on(), "mode": "paper", "book": book,
            "room": room.state() if room is not None else None,
            "updated": int(time.time())}


def soma_xy(fb):
    """
    Each neuron's measured soma position, flattened to the screen.

    Used only to place a dot. Every coordinate is the real one from the
    annotations table - the scatter is anatomy, not a shape.
    """
    try:
        import pandas as pd
        a = pd.read_feather(ROOT / "data" / "body-annotations.feather")
        a = a.drop_duplicates("bodyId").set_index("bodyId")
        loc = a["somaLocation"].reindex(fb.bodies)
        xy = np.full((fb.n, 2), np.nan, dtype=np.float32)
        for i, v in enumerate(loc.to_numpy()):
            if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 3:
                xy[i] = (float(v[0]), float(v[2]))
        ok = ~np.isnan(xy[:, 0])
        if ok.sum() < 100:
            return None
        lo, hi = np.nanmin(xy[ok], 0), np.nanmax(xy[ok], 0)
        xy = (xy - lo) / np.maximum(hi - lo, 1e-6)
        return xy
    except Exception as exc:
        say("no soma coordinates:", str(exc)[:90])
        return None


# --------------------------------------------------------------------------
# what is under the cursor, and may the fly click it
# --------------------------------------------------------------------------
UNDER_JS = """([x, y]) => {
  const e = document.elementFromPoint(x, y);
  if (!e) return null;
  const a = e.closest('a');
  const btn = e.closest('button,input,textarea,select,[role=button]');
  const txt = ((btn || a || e).innerText || (btn || e).value || '')
    .trim().slice(0, 80);
  return {
    tag: e.tagName,
    href: a ? a.href : null,
    control: !!btn,
    type: (btn && btn.type) || '',
    text: txt,
    label: (e.getAttribute('aria-label') || '') + ' ' + (e.name || ''),
  }; }"""


def may_click(under):
    """A click is allowed unless it looks like it commits something."""
    if not under:
        return False, "nothing there"
    href = under.get("href") or ""
    if href and BLOCK.search(href):
        return False, "blocked destination"
    if href and not allowed_host(href):
        return False, "outside the fence"
    blob = " ".join(str(under.get(k) or "") for k in ("text", "label", "type"))
    if VETO.search(blob):
        return False, f"veto: {blob.strip()[:40]}"
    if under.get("type", "").lower() in ("submit", "file", "password"):
        return False, "veto: form control"
    return True, "ok"


async def screenshot(page):
    raw = await page.screenshot(type="jpeg", quality=62)
    return raw


def to_gray(raw, w=None, h=None):
    from PIL import Image
    im = Image.open(io.BytesIO(raw)).convert("L")
    return np.asarray(im, dtype=np.float32) / 255.0


# --------------------------------------------------------------------------
# the roam
# --------------------------------------------------------------------------
async def roam(steps_per_page=44, headful=False, seed=None):
    from playwright.async_api import async_playwright

    fb, pilot = load_brain()
    rng = random.Random(seed if seed is not None else int(time.time()))

    send = broadcast

    async def log(m):
        say("  " + str(m))
        stats["events"].append({"t": time.strftime("%H:%M:%S"), "m": str(m)[:110]})
        stats["events"] = stats["events"][-40:]
        await send({"type": "log", "msg": str(m)})

    stats = {"steps": 0, "clicks": 0, "vetoes": 0, "hops": 0,
             "blocked": 0, "scrolled": 0, "started": time.time(),
             "visited": [], "events": [], "firing": []}

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=not headful,
            # a container gives /dev/shm 64 MB and Chromium crashes on it
            args=["--disable-dev-shm-usage", "--no-sandbox"])
        # No storage, no wallet, no extension, no downloads. A fresh context
        # with nothing in it: the fly cannot be logged in as anyone.
        ctx = await browser.new_context(
            viewport={"width": 1280, "height": 800},
            accept_downloads=False,
            java_script_enabled=True,
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0 Safari/537.36 flybrain/1.0"),
        )
        page = await ctx.new_page()
        page.on("dialog", lambda d: asyncio.create_task(d.dismiss()))

        # A popup is not somewhere the fly chose to go, so it gets closed - but
        # the handler fires for the main page too, and closing that ends the
        # run before it starts.
        def _popup(p):
            if p is not page:
                asyncio.create_task(p.close())
        ctx.on("page", _popup)

        async def goto(url, why=""):
            if BLOCK.search(url) or not allowed_host(url):
                stats["blocked"] += 1
                await log(f"blocked: {url[:70]}")
                return False
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                await page.wait_for_timeout(1200)
                stats["hops"] += 1
                title = (await page.title())[:70]
                stats["visited"].append({"url": page.url, "title": title,
                                         "at": int(time.time())})
                stats["visited"] = stats["visited"][-40:]
                await send({"type": "place", "url": page.url, "title": title,
                            "why": why})
                await log(f"arrived: {title} - {page.url[:64]}")
                return True
            except Exception as exc:
                await log(f"could not open: {str(exc)[:70]}")
                return False

        async def reset(why):
            """
            Every restart goes through one door.

            A life begins, a hop budget runs out, a click lands somewhere
            blocked: all three end up here, and here is the only place that
            decides between a web seed and the room. Nothing else in this file
            may send the fly anywhere.
            """
            url, where = next_place(rng, room is not None)
            return await goto(url, f"{why}: {where}")

        room = open_room()
        await reset("a new life")

        cx, cy = 640.0, 400.0
        px_, py_ = cx, cy
        on_page = 0

        # Screencast, not screenshots.
        #
        # page.screenshot() in a loop tops out near three frames a second -
        # each call is a fresh round trip and a fresh encode - which looked
        # like a slideshow of stills. Chrome's own screencast pushes a frame
        # whenever the page actually changes, which is both faster and more
        # honest: a still page emits nothing because nothing happened.
        latest = {"jpg": None, "n": 0}
        cdp = await ctx.new_cdp_session(page)
        loop_ = asyncio.get_running_loop()

        def on_cast(params):
            try:
                latest["jpg"] = base64.b64decode(params["data"])
                latest["n"] += 1
                asyncio.run_coroutine_threadsafe(
                    cdp.send("Page.screencastFrameAck",
                             {"sessionId": params["sessionId"]}), loop_)
            except Exception:
                pass

        cdp.on("Page.screencastFrame", on_cast)
        await cdp.send("Page.startScreencast", {
            "format": "jpeg", "quality": 55,
            "maxWidth": 1280, "maxHeight": 800, "everyNthFrame": 1})

        async def pump():
            """Push the newest frame to watchers, at most ten times a second."""
            last = -1
            while STATE["running"]:
                if latest["jpg"] is not None and latest["n"] != last:
                    last = latest["n"]
                    await send({"type": "view",
                                "jpg": base64.b64encode(latest["jpg"]).decode(),
                                "cx": cx, "cy": cy})
                await asyncio.sleep(0.1)

        cap = asyncio.create_task(pump())
        for _ in range(60):
            if latest["jpg"] is not None:
                break
            await asyncio.sleep(0.1)
        if latest["jpg"] is None:            # screencast never started
            latest["jpg"] = await screenshot(page)

        while STATE["running"]:
            raw = latest["jpg"]
            img = to_gray(raw)

            # Is the fly in the room? Nothing about the walk changes if it is -
            # same eye, same descending neurons, same stopping rule - but the
            # room reads the stop, smells the card and keeps the dwell, so it
            # runs the step in place of the pilot.
            in_room = room is not None and is_backroom(page.url)
            if room is not None and in_room != room.in_room:
                if in_room:
                    await room.enter(page)
                    await log("in the backroom")
                else:
                    room.leave()

            seed_ = rng.randrange(1 << 30)
            if in_room:
                dx, dy, click, hz, info = await room.step(page, img, cx, cy, seed_)
            else:
                dx, dy, click, hz, info = pilot.step(
                    img, cx, cy, gains=STATE["gains"], seed=seed_, detail=True)
            cx = float(np.clip(cx + dx, 8, 1272))
            cy = float(np.clip(cy + dy, 8, 792))
            stats["steps"] += 1
            on_page += 1

            # A fly that walks off the bottom of what it can see should get
            # more page, not stick to the edge. DNa01 driving down past the
            # margin scrolls down, MDN driving up scrolls back, and the cursor
            # is recentred so the walk continues instead of pinning. This is
            # what makes the view move: without it the page is a still image
            # with a cursor twitching on it.
            EDGE = 110
            if cy > 800 - EDGE and dy > 0:
                await page.mouse.wheel(0, 300)
                cy = 800 - EDGE - 140
                stats["scrolled"] += 1
            elif cy < EDGE and dy < 0:
                await page.mouse.wheel(0, -300)
                cy = EDGE + 140
                stats["scrolled"] += 1

            # a sample of the neurons that actually fired, at their measured
            # soma positions - the scatter is a readout, not an animation
            scatter = []
            xy = STATE.get("xy")
            fired = info.get("fired")
            if xy is not None and fired is not None and len(fired):
                take = fired if len(fired) <= 420 else rng.sample(
                    list(fired), 420)
                for i in take:
                    x, y = xy[i]
                    if not np.isnan(x):
                        scatter.append([round(float(x), 3), round(float(y), 3)])

            # A Kenyon cell that fired just now becomes eligible, and
            # eligibility fades. No dopamine is delivered anywhere in this
            # file: the only lessons are profit and loss, and backroom.py
            # delivers those where they are measured. In the room the Room
            # does this itself, on the same run it read the mushroom body
            # from, so doing it again here would count the look twice.
            mb = STATE.get("mb")
            if mb is not None and not in_room:
                mb.observe(info.get("fired"))
                mb.forget()

            stats["firing"].append(info["firing"])
            stats["firing"] = stats["firing"][-72:]

            neural = {
                "firing": info["firing"], "total": fb.n,
                "history": stats["firing"],
                "vision": info.get("vision"),
                "spikes_per_sec": round(info["spikes_per_sec"]),
                "mean_mv": round(info["mean_mv"], 1),
                "visual": info["visual"], "motor": info["motor"],
                "dn": {k: round(v, 1) for k, v in hz.items()},
                "out": {k: round(info[k], 3) for k in
                        ("turn_l", "turn_r", "forward", "reverse", "click")},
                "scatter": scatter,
                "learning": (STATE["mb"].stats()
                             if STATE.get("mb") is not None else None),
            }

            # The fly decides twice a second - that is 12 ms of brain time per
            # decision, and cutting it shorter would break the retina-to-DN
            # path rather than speed anything up. What was wrong was the move
            # in between: the cursor teleported to the new position and the
            # page never repainted, so nothing was there to stream. Now it
            # travels there, hover states fire, and the screen actually moves.
            steps_ = 9
            for j in range(1, steps_ + 1):
                t_ = j / steps_
                t_ = t_ * t_ * (3 - 2 * t_)
                await page.mouse.move(px_ + (cx - px_) * t_,
                                      py_ + (cy - py_) * t_)
                await send({"type": "cursor",
                            "cx": px_ + (cx - px_) * t_,
                            "cy": py_ + (cy - py_) * t_})
                await asyncio.sleep(0.028)
            px_, py_ = cx, cy

            bk = backroom_state() if backroom_on() else None
            frame = {"type": "frame", "neural": neural,
                     "events": stats["events"][-18:],
                     "visited": stats["visited"][-8:],
                     "cx": cx, "cy": cy,
                     "hz": {k: round(v, 1) for k, v in hz.items()},
                     "stats": {k: stats[k] for k in
                               ("steps", "clicks", "vetoes", "hops",
                                "blocked", "scrolled")},
                     "url": page.url}
            if bk is not None:
                frame["backroom"] = bk
            await send(frame)

            # A stop in the room is not a click. room.step has already read it
            # as a commit or as nothing, and there is nothing on that page to
            # press: no link to follow, no control to veto, no element to ask
            # about. So the whole clicking apparatus is skipped there.
            if click and not in_room:
                under = await page.evaluate(UNDER_JS, [cx, cy])
                ok, why = may_click(under)
                if ok:
                    stats["clicks"] += 1
                    before = page.url
                    await log(f"click on {(under.get('text') or under['tag'])[:44]}")
                    try:
                        await page.mouse.click(cx, cy)
                        await page.wait_for_timeout(1800)
                    except Exception:
                        pass
                    if page.url != before:
                        if BLOCK.search(page.url) or not allowed_host(page.url):
                            stats["blocked"] += 1
                            await log(f"landed somewhere blocked, going back")
                            try:
                                await page.go_back(timeout=15000)
                            except Exception:
                                await reset("bounced")
                        else:
                            stats["hops"] += 1
                            title = (await page.title())[:70]
                            stats["visited"].append(
                                {"url": page.url, "title": title,
                                 "at": int(time.time())})
                            stats["visited"] = stats["visited"][-40:]
                            await send({"type": "place", "url": page.url,
                                        "title": title, "why": "followed a link"})
                            await log(f"followed a link to {title}")
                        on_page = 0
                        cx, cy = 640.0, 400.0
                else:
                    stats["vetoes"] += 1
                    await log(f"did not click - {why}")

            # A fly that has run out of page gets put somewhere else. A visit to
            # the room has its own budget so that a life is not spent in there.
            if on_page >= (backroom_steps() if in_room else steps_per_page):
                on_page = 0
                cx, cy = 640.0, 400.0
                if in_room:
                    room.leave()
                await reset("hop budget spent")

            # Whatever the executor settled since the last step. It is applied
            # here, on this thread, because applying a settled sell runs the
            # brain - the coin is shown to the fly again - and there is one
            # brain.
            if room is not None:
                room.poll_events()

            publish(stats, raw, page.url, hz, neural, bk)
            if mb is not None and stats["steps"] % 40 == 0:
                mb.save()
            await asyncio.sleep(0.05)

        cap.cancel()
        await ctx.close()
        await browser.close()
    await send({"type": "done", "stats": stats})


def publish(stats, jpg, url, hz, neural=None, backroom=None):
    """
    Leave the latest frame and a summary on disk.

    The site is static and the fly runs on a machine, so something has to carry
    the state across. Writing it here keeps that transport separate from the
    roaming itself - if the publisher is not running, the fly does not care.
    """
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        _atomic_write(OUT / "roam_frame.jpg", jpg)
        state_ = {
            "url": url,
            "steps": stats["steps"], "clicks": stats["clicks"],
            "vetoes": stats["vetoes"], "hops": stats["hops"],
            "blocked": stats["blocked"], "scrolled": stats["scrolled"],
            "uptime_s": int(time.time() - stats["started"]),
            "visited": stats["visited"][-12:],
            "events": stats["events"][-18:],
            "hz": {k: round(v, 1) for k, v in hz.items()},
            "neural": neural,
            "stream": TUNNEL["url"],
            "updated": int(time.time()),
        }
        if backroom is not None:
            state_["backroom"] = backroom
        _atomic_write(OUT / "roam_state.json",
                      json.dumps(state_, indent=1).encode("utf-8"))

        now = time.time()
        if BLOB_TOKEN and now - _last_push["at"] >= BLOB_EVERY:
            _last_push["at"] = now
            import threading

            # Nothing goes to object storage any more.
            #
            # Pushing two frames a second suspended the blob store on
            # operation count - 13 MB held, but the writes and deletes blew
            # the free tier and every read started answering 403, which took
            # the public feed down with it. The tunnel already carries frames,
            # telemetry and events for free, so the only thing that ever
            # needed publishing is where the tunnel is. That is one small file,
            # written when the address changes and not otherwise.
            import threading

            def push():
                publish_address()
            threading.Thread(target=push, daemon=True).start()
    except Exception:
        pass


# --------------------------------------------------------------------------
# the local view
# --------------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(str(ROOT / "web" / "roam.html"))


@app.get("/status")
def status():
    return {"running": STATE["running"],
            "seeds": len(SEEDS),
            "brain": bool(STATE["brain"])}


@app.get("/state")
def state():
    p = OUT / "roam_state.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"updated": 0}


@app.get("/frame.jpg")
def frame():
    p = OUT / "roam_frame.jpg"
    if p.exists():
        return Response(p.read_bytes(), media_type="image/jpeg")
    return Response(b"", media_type="image/jpeg")


@app.get("/backroom")
def backroom_page():
    """
    The room itself, served to the fly's own browser over loopback.

    It is a page like any other page, drawn dark with big bright cards because
    that is what 892 hex columns can resolve. There is nothing on it to press.
    """
    return FileResponse(str(ROOT / "web" / "backroom.html"))


@app.get("/backroom/board.json")
def backroom_board():
    """What the page draws. Empty until the room exists, which needs FLY_BACKROOM=1."""
    room = STATE.get("room")
    if room is None:
        return {"cards": [], "holdings": [], "updated": 0}
    return room.board()


@app.get("/backroom/state")
def backroom_view():
    """The paper book and what the fly is doing in the room, for a watcher."""
    return backroom_state()


@app.websocket("/ws")
async def socket(ws: WebSocket):
    """A watcher. There is nothing to send: the fly is already roaming."""
    await ws.accept()
    CLIENTS.add(ws)
    try:
        while True:
            await ws.receive_text()
    except Exception:
        pass
    finally:
        CLIENTS.discard(ws)


@app.on_event("startup")
async def begin():
    """
    Start roaming as soon as the process is up, and keep roaming.

    There is no start button and no stop button. If a run dies - a page hangs,
    a browser falls over - it waits a few seconds and starts a new life rather
    than sitting there waiting to be told.
    """
    if load_env().get("FLY_ALLOW_BROWSER") != "1":
        say("FLY_ALLOW_BROWSER is not 1 - not opening a browser")
        return

    start_tunnel(STATE.get("port", 4660))

    async def forever():
        while True:
            STATE["running"] = True
            try:
                await roam()
            except Exception as exc:
                import traceback
                say("roam ended:", str(exc)[:160])
                traceback.print_exc()
            STATE["running"] = False
            say("starting another life in 6s")
            await asyncio.sleep(6)

    asyncio.create_task(forever())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # a host that assigns the port says so in PORT
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("PORT", "4660")))
    ap.add_argument("--headful", action="store_true")
    a = ap.parse_args()
    STATE["port"] = a.port
    load_brain()
    say(f"the fly roams - open http://localhost:{a.port}")
    # loopback on a desk, every interface in a container
    uvicorn.run(app, host=os.environ.get("FLY_HOST", "127.0.0.1"),
                port=a.port, log_level="warning")
