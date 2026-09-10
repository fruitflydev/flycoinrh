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

  py roam.py                 open http://localhost:4660 and press START
  py roam.py --headful       watch the real browser too
"""
import argparse
import asyncio
import base64
import io
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

from flysim import FlyBrain
from envcfg import load_env
from flyeye import FlyPilot

ROOT = Path(__file__).parent
OUT = ROOT / "build"

# Link-rich, text-heavy, safe places to be dropped into. The fly leaves them
# on its own within a few clicks; these only decide where a life starts.
SEEDS = [
    "https://en.wikipedia.org/wiki/Special:Random",
    "https://en.wikipedia.org/wiki/Drosophila_melanogaster",
    "https://en.wikipedia.org/wiki/Connectome",
    "https://news.ycombinator.com/",
    "https://www.gutenberg.org/browse/scores/top",
    "https://xkcd.com/",
    "https://arxiv.org/list/q-bio.NC/recent",
]

# Checked against every URL the browser tries to commit to.
BLOCK = re.compile(
    r"(porn|xxx|adult|nsfw|escort|hentai|onlyfans|camsoda|chaturbate"
    r"|casino|bet365|poker|gambl|lottery"
    r"|checkout|/cart|/pay|payment|billing|invoice|subscribe"
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
    "news.ycombinator.com", "www.gutenberg.org", "gutenberg.org",
    "xkcd.com", "www.xkcd.com", "arxiv.org", "www.arxiv.org",
    "openlibrary.org", "archive.org", "web.archive.org",
    "www.nature.com", "www.science.org", "elifesciences.org",
    "www.janelia.org", "neuprint.janelia.org", "flywire.ai",
    "en.wikiquote.org", "en.wikisource.org", "www.ponsfamily.com",
}
OPEN = load_env().get("FLY_ROAM_OPEN") == "1"


def allowed_host(url):
    """Open mode drops the fence and leaves only the blocklist behind it."""
    if OPEN:
        return True
    try:
        from urllib.parse import urlparse
        return (urlparse(url).hostname or "").lower() in ALLOW
    except Exception:
        return False


# Checked against the element under the cursor before a click is allowed.
VETO = re.compile(
    r"(submit|upload|sign in|sign up|log in|log out|subscribe|buy|purchase"
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
STATE = {"brain": None, "pilot": None, "running": False}
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
    if STATE["brain"] is None:
        say("loading the connectome ...")
        fb = FlyBrain()
        STATE["brain"] = fb
        STATE["pilot"] = FlyPilot(fb, sim_steps=60)
        STATE["xy"] = soma_xy(fb)
        say(f"brain ready: {len(fb.bodies):,} neurons")
    return STATE["brain"], STATE["pilot"]


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
async def roam(steps_per_page=26, headful=False, seed=None):
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
        browser = await pw.chromium.launch(headless=not headful)
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

        await goto(rng.choice(SEEDS), "seed")

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

            dx, dy, click, hz, info = pilot.step(
                img, cx, cy, seed=rng.randrange(1 << 30), detail=True)
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

            await send({"type": "frame", "neural": neural,
                        "cx": cx, "cy": cy,
                        "hz": {k: round(v, 1) for k, v in hz.items()},
                        "stats": {k: stats[k] for k in
                                  ("steps", "clicks", "vetoes", "hops",
                                   "blocked", "scrolled")},
                        "url": page.url})

            if click:
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
                                await goto(rng.choice(SEEDS), "bounced")
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

            # a fly that has run out of page gets put back on a seed
            if on_page >= steps_per_page:
                on_page = 0
                cx, cy = 640.0, 400.0
                await goto(rng.choice(SEEDS), "hop budget spent")

            publish(stats, raw, page.url, hz, neural)
            await asyncio.sleep(0.05)

        cap.cancel()
        await ctx.close()
        await browser.close()
    await send({"type": "done", "stats": stats})


def publish(stats, jpg, url, hz, neural=None):
    """
    Leave the latest frame and a summary on disk.

    The site is static and the fly runs on a machine, so something has to carry
    the state across. Writing it here keeps that transport separate from the
    roaming itself - if the publisher is not running, the fly does not care.
    """
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "roam_frame.jpg").write_bytes(jpg)
        payload = json.dumps({
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
        }, indent=1)
        (OUT / "roam_state.json").write_text(payload)

        now = time.time()
        if BLOB_TOKEN and now - _last_push["at"] >= BLOB_EVERY:
            _last_push["at"] = now
            import threading

            # Time-addressed frames.
            #
            # The store's CDN answers X-Vercel-Cache: HIT with an Age of twenty
            # seconds even though every upload sets max-age=0, and a query
            # string does not bust it - public blobs are treated as immutable.
            # So a fixed pathname can never carry a live feed. Each push gets
            # its own pathname keyed to the half-second it happened in, and the
            # page asks for the slot it expects rather than looking anything
            # up. Every URL is fetched once, so nothing is ever stale.
            slot = int(now * 2)

            def push():
                try:
                    blob_put(f"roam/t/{slot}.jpg", jpg, "image/jpeg")
                    blob_put(f"roam/t/{slot}.json", payload.encode(),
                             "application/json")
                    blob_put("roam/state.json", payload.encode(),
                             "application/json")   # slow fallback
                    for old_slot in range(slot - 240, slot - 200):
                        blob_del(f"roam/t/{old_slot}.jpg")
                        blob_del(f"roam/t/{old_slot}.json")
                except Exception:
                    pass
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
    ap.add_argument("--port", type=int, default=4660)
    ap.add_argument("--headful", action="store_true")
    a = ap.parse_args()
    STATE["port"] = a.port
    load_brain()
    say(f"the fly roams - open http://localhost:{a.port}")
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
