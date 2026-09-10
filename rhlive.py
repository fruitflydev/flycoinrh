"""
The live rig, Robinhood Chain edition: the fly on ponsfamily.com/launchpad.

Same brain, same streaming, same viewer as live.py - a different venue and a
different wallet. Runs on its own port so the two can coexist.

Where this differs from the Solana rig:

  * the wallet is EIP-1193 (rhprovider.py), and the launchpad accepts it
    directly - no 401, no bypass, so the site's own Launch button can actually
    complete the transaction
  * the launch fee is 0.0005 ETH and gas is ETH on chain 4663
  * pons requires accepting Terms of Use, a Privacy Policy and a
    not-in-a-restricted-jurisdiction attestation before the form will work

  py rhlive.py                 http://localhost:4651
"""
import argparse
import asyncio
import base64
import io
import json
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response

from flysim import FlyBrain
from flyeye import FlyPilot
from envcfg import load_env
from rhwallet import account, CHAIN_ID, RPC, LAUNCH_FEE_ETH, balance
from rhprovider import attach

ROOT = Path(__file__).parent
URL = "https://www.ponsfamily.com/launchpad/create"
IMAGE = "assets/flycoin_square.png"

app = FastAPI()
STATE = {"brain": None, "pilot": None, "remap": None, "xyz": None,
         "grp": None, "running": False}


def say(*parts):
    try:
        print(" ".join(str(x) for x in parts).encode("ascii", "replace").decode(),
              flush=True)
    except Exception:
        pass


def browser_allowed():
    return load_env().get("FLY_ALLOW_BROWSER", "0") == "1"


def boot(n_keep=11000):
    import pandas as pd
    fb = FlyBrain()
    pilot = FlyPilot(fb, sim_steps=60)
    a = pd.read_feather(ROOT / "data" / "body-annotations.feather")
    a = a[["bodyId", "somaLocation"]].dropna(subset=["somaLocation"])
    a = a.drop_duplicates(subset=["bodyId"])
    pos = pd.Series(list(a.somaLocation), index=a.bodyId.to_numpy())
    have = pos.reindex(fb.bodies)
    ok = have.notna().to_numpy()
    P = np.full((fb.n, 3), np.nan, dtype=np.float32)
    P[ok] = np.stack(have[ok].to_numpy()).astype(np.float32)
    grp = np.zeros(fb.n, dtype=np.uint8)
    for t, g in (("^L1$|^L2$", 1), ("^DNa02$|^DNa01$|^MDN$|^DNp09$", 2), ("^MN9$", 3)):
        grp[fb.where(type_re=t)] = g
    special = np.flatnonzero(ok & (grp > 0))
    rest = np.flatnonzero(ok & (grp == 0))
    rng = np.random.default_rng(0)
    take = rng.choice(rest, size=min(n_keep - len(special), len(rest)), replace=False)
    keep = np.sort(np.concatenate([special, take]))
    remap = np.full(fb.n, -1, dtype=np.int32)
    remap[keep] = np.arange(len(keep))
    STATE.update(brain=fb, pilot=pilot, remap=remap,
                 xyz=P[keep].astype(np.float32), grp=grp[keep])
    print(f"brain ready: {fb.n:,} neurons, {fb.W.nnz:,} edges; "
          f"{len(keep):,} streamed to the viewer")


@app.get("/")
async def index():
    return HTMLResponse((ROOT / "web" / "live.html").read_text(encoding="utf-8"))


@app.get("/neurons")
async def neurons():
    return Response(content=STATE["xyz"].tobytes() + STATE["grp"].tobytes(),
                    media_type="application/octet-stream",
                    headers={"X-Count": str(len(STATE["grp"]))})


@app.get("/status")
async def status():
    env = load_env()
    try:
        acct = account(env)
        eth = balance(env, quiet=True)
        return {"wallet": acct.address, "sol": eth, "unit": "ETH",
                "venue": "ponsfamily.com/launchpad",
                "chain": f"Robinhood Chain {CHAIN_ID}",
                "live": env.get("FLY_RH_LIVE", "0") == "1",
                "armed": browser_allowed()}
    except SystemExit:
        return {"wallet": None, "sol": 0, "unit": "ETH", "live": False,
                "armed": browser_allowed()}


# --------------------------------------------------------------------------
# the pons launchpad
# --------------------------------------------------------------------------
DOM_JS = """() => {
  const out = {};
  const byPh = (k, ph) => {
    const e = [...document.querySelectorAll('input,textarea')]
      .find(x => (x.placeholder||'').toLowerCase() === ph);
    if (!e) return;
    const r = e.getBoundingClientRect();
    out[k] = {x:Math.round(r.x), y:Math.round(r.y), w:Math.round(r.width),
              h:Math.round(r.height), filled:(e.value||'').length>0}; };
  byPh('name','token name');
  byPh('ticker','symbol');
  byPh('desc','a short description of the token');
  // The primary action button relabels itself by state: "Connect wallet" ->
  // "Fill token details" -> "Insufficient ETH" -> the actual launch. Matching
  // on text alone missed it entirely, so find it by shape instead: the
  // full-width button lowest in the form column.
  const cands = [...document.querySelectorAll('button')]
    .map(b => ({b, r: b.getBoundingClientRect()}))
    .filter(o => o.r.width > 380 && o.r.height > 40
                 && !/^(advanced|eth|usdc)$/i.test((o.b.textContent||'').trim()));
  const pick = cands.sort((p, q) => (q.r.y + scrollY) - (p.r.y + scrollY))[0];
  if (pick) { const r = pick.r;
    out.launch = {x:Math.round(r.x), y:Math.round(r.y), w:Math.round(r.width),
                  h:Math.round(r.height), label:(pick.b.textContent||'').trim(),
                  disabled: pick.b.disabled}; }
  return out; }"""


SCROLL_JS = """() => {
  const cands = [...document.querySelectorAll('button')]
    .map(b => ({b, r: b.getBoundingClientRect()}))
    .filter(o => o.r.width > 380 && o.r.height > 40
                 && !/^(advanced|eth|usdc)$/i.test((o.b.textContent||'').trim()));
  const pick = cands.sort((p, q) => (q.r.y + scrollY) - (p.r.y + scrollY))[0];
  if (pick) pick.b.scrollIntoView({block:'center'}); }"""


def inside(f, x, y, pad=10):
    return (f["x"] - pad <= x <= f["x"] + f["w"] + pad
            and f["y"] - pad <= y <= f["y"] + f["h"] + pad)


async def accept_terms(page, send=None):
    """
    pons gates the form behind Terms of Use, a Privacy Policy and a
    not-in-a-restricted-jurisdiction attestation. Ticking these is a decision
    for the person running the rig; it happens here only because that
    permission was given explicitly.
    """
    try:
        present = await page.evaluate(
            """() => /review and accept|terms of use/i.test(document.body.innerText)""")
        if not present:
            return False
        boxes = await page.query_selector_all(
            'input[type="checkbox"], [role="checkbox"]')
        for b in boxes:
            try:
                if not await b.is_checked():
                    await b.click(timeout=2500)
                    await page.wait_for_timeout(300)
            except Exception:
                try:
                    await b.click(timeout=1500, force=True)
                except Exception:
                    pass
        await page.wait_for_timeout(400)
        for label in ("Accept and continue", "Accept", "Continue"):
            try:
                await page.get_by_role("button", name=label).first.click(timeout=3000)
                if send:
                    await send({"type": "log", "msg": f"accepted terms ({label})"})
                await page.wait_for_timeout(1800)
                return True
            except Exception:
                continue
        if send:
            await send({"type": "log", "msg": "terms dialog present but not accepted"})
    except Exception as e:
        if send:
            await send({"type": "log", "msg": f"terms step failed: {str(e)[:110]}"})
    return False


async def dismiss_banners(page):
    """The degraded-performance strip is in normal flow, not fixed, so the
    position test used for pump.fun's banner never matched it."""
    try:
        await page.evaluate("""() => {
            const rx = /degraded performance|we are upgrading/i;
            [...document.querySelectorAll('div,section,aside,header')]
              .filter(e => rx.test(e.innerText||'')
                        && e.offsetHeight > 0 && e.offsetHeight < 90
                        && !e.querySelector('input,textarea,button[type=submit]'))
              .forEach(e => e.style.display='none'); }""")
    except Exception:
        pass


async def go_dark(page, send=None):
    """
    Put the launchpad in dark mode.

    Two reasons. It looks wrong on camera next to a dark interface - but more
    to the point, the fly's retina is a luminance map, and every bit of tuning
    it has was done against dark UI. A white page inverts the contrast
    relationships it learned.
    """
    try:
        was = await page.evaluate(
            """() => getComputedStyle(document.body).backgroundColor""")
        clicked = await page.evaluate(
            """() => { const b=[...document.querySelectorAll('button,[role=button]')]
                 .find(x => /theme|dark|light|appearance/i.test(
                   (x.getAttribute('aria-label')||'') + (x.title||'')));
               if (b) { b.click(); return true; } return false; }""")
        if not clicked:
            # next-themes and friends keep it here; set it and reload
            await page.evaluate("""() => {
                try { localStorage.setItem('theme','dark'); } catch(e){}
                document.documentElement.classList.add('dark');
                document.documentElement.setAttribute('data-theme','dark'); }""")
        await page.wait_for_timeout(1200)
        now = await page.evaluate(
            """() => getComputedStyle(document.body).backgroundColor""")
        if send:
            await send({"type": "log", "msg": f"theme {was} -> {now}"})
        return now != was
    except Exception as e:
        if send:
            await send({"type": "log", "msg": f"theme switch failed: {str(e)[:90]}"})
        return False


async def smooth_scroll(page, target_y, send=None, ms=1300, steps=30):
    """
    Scroll there visibly.

    scroll_into_view_if_needed() and scrollIntoView() both teleport - the page
    is simply somewhere else on the next frame, so the move never appears in
    the recording. This eases the scroll over about a second instead, which
    also gives the streamer time to catch it.
    """
    try:
        start = await page.evaluate("window.scrollY")
        for i in range(1, steps + 1):
            t = i / steps
            t = t * t * (3 - 2 * t)
            await page.evaluate(f"window.scrollTo(0, {start + (target_y - start) * t})")
            await asyncio.sleep(ms / 1000.0 / steps)
    except Exception:
        pass


async def scroll_to_el(page, selector_js, send=None, offset=0.45):
    """Ease the page until the named element sits `offset` down the viewport."""
    try:
        y = await page.evaluate(
            "(js) => { const e = eval(js); if (!e) return null;"
            " return e.getBoundingClientRect().top + window.scrollY; }",
            selector_js)
        if y is None:
            return False
        vh = await page.evaluate("innerHeight")
        await smooth_scroll(page, max(0, y - vh * offset), send)
        return True
    except Exception:
        return False


async def glide(page, x, y, send=None, steps=22, hold=0.0):
    """
    Move the mouse there gradually, telling the viewer as it goes.

    Two problems this fixes. Playwright's mouse.move() jumps instantly, and the
    fly sprite only tracked the fly's own control steps - so during everything
    the rig did, the sprite sat frozen while the page changed underneath it.
    Now the sprite follows every movement, and the movement takes real time.
    """
    try:
        cur = getattr(glide, "_pos", (640.0, 300.0))
        x0, y0 = cur
        for i in range(1, steps + 1):
            t = i / steps
            t = t * t * (3 - 2 * t)          # ease in/out
            nx, ny = x0 + (x - x0) * t, y0 + (y - y0) * t
            await page.mouse.move(nx, ny)
            if send:
                await send({"type": "cursor", "cx": nx, "cy": ny})
            await asyncio.sleep(0.022)
        glide._pos = (x, y)
        if hold:
            await asyncio.sleep(hold)
    except Exception:
        pass


async def set_creator_tax(page, pct, send=None, shot=None):
    """
    Open Advanced and set the creator tax.

    The field is the percent input between "Creator wallet" and the snipe
    exemption list. pons caps it at 10 - the page says traders pay 1.00% in
    total and up to 10% of that is the creator's.
    """
    try:
        adv = page.get_by_role("button", name="Advanced").first
        await scroll_to_el(page,
            "[...document.querySelectorAll('button')]"
            ".find(b => /^advanced$/i.test((b.textContent||'').trim()))", send)
        await asyncio.sleep(0.9)
        box = await adv.bounding_box()
        if box:
            await glide(page, box["x"] + box["width"] / 2,
                        box["y"] + box["height"] / 2, send, hold=0.5)
            if shot:
                await shot("opening advanced")
            await page.mouse.click(box["x"] + box["width"] / 2,
                                   box["y"] + box["height"] / 2)
        await asyncio.sleep(1.6)
        if shot:
            await shot("advanced open")

        # the creator tax box: a percent field, empty, below the creator wallet
        tax = await page.evaluate(
            """() => {
               const ins = [...document.querySelectorAll('input')];
               const wal = ins.find(e => /0x/.test(e.placeholder || ''));
               const cand = ins.filter(e => (e.placeholder || '') === '0');
               const pick = cand.find(e => !wal ||
                 e.getBoundingClientRect().top > wal.getBoundingClientRect().top)
                 || cand[0];
               if (!pick) return null;
               const r = pick.getBoundingClientRect();
               return {x: Math.round(r.x), y: Math.round(r.y),
                       w: Math.round(r.width), h: Math.round(r.height)}; }""")
        if not tax:
            if send:
                await send({"type": "log", "msg": "creator tax field not found"})
            return False
        await scroll_to_el(page,
            "[...document.querySelectorAll('input')]"
            ".filter(e => (e.placeholder||'') === '0').pop()", send)
        await asyncio.sleep(0.7)
        tax = await page.evaluate(
            """() => { const e=[...document.querySelectorAll('input')]
                 .filter(x => (x.placeholder||'')==='0').pop();
               if(!e) return null; const r=e.getBoundingClientRect();
               return {x:Math.round(r.x), y:Math.round(r.y),
                       w:Math.round(r.width), h:Math.round(r.height)}; }""")
        if not tax:
            return False
        await glide(page, tax["x"] + tax["w"] / 2, tax["y"] + tax["h"] / 2,
                    send, hold=0.4)
        await page.mouse.click(tax["x"] + tax["w"] / 2, tax["y"] + tax["h"] / 2)
        await page.keyboard.press("Control+A")
        await page.keyboard.press("Delete")
        await asyncio.sleep(0.3)
        await page.keyboard.type(str(pct), delay=220)
        await asyncio.sleep(1.0)
        if shot:
            await shot(f"creator tax {pct}%")
        val = await page.evaluate(
            """() => { const e=[...document.querySelectorAll('input')]
                 .find(x => (x.placeholder||'')==='0'); return e ? e.value : null; }""")
        if send:
            await send({"type": "log", "msg": f"creator tax set to {val}%"})
        return True
    except Exception as e:
        if send:
            await send({"type": "log", "msg": f"creator tax failed: {str(e)[:110]}"})
        return False


def step_brain(pilot, img, cx, cy, gains, seed):
    drive = pilot.eye.look(img, cx, cy)
    r = pilot.fb.run(drive, steps=pilot.sim_steps, gains=gains,
                     record=pilot.motor, seed=seed, spike_log=True)
    hz = {k: float(r[k].mean()) for k in pilot.motor}
    turn = (hz["steer_R"] - hz["steer_L"]) / 450.0
    fwd = (hz["fwd_L"] + hz["fwd_R"]) / 2.0 / 450.0
    back = hz["back"] / 450.0
    stop = hz["stop"] / 450.0
    speed = np.clip(fwd - back, -1, 1) * (1.0 - np.clip(stop, 0, 1))
    fired = np.concatenate(r["_spikes"]) if r["_spikes"] else np.array([], dtype=np.int32)
    click = hz["stop"] >= pilot.click_hz and speed < 0.25
    return (float(np.clip(turn, -1, 1) * 90.0), float(-speed * 90.0),
            bool(click), hz, fired)


async def run_episode(ws, coin, steps, seed, headful):
    from playwright.async_api import async_playwright
    from PIL import Image

    fb, pilot, remap = STATE["brain"], STATE["pilot"], STATE["remap"]
    env = load_env()
    acct = account(env)
    rpc = env.get("FLY_RH_RPC", RPC)
    live_flag = env.get("FLY_RH_LIVE", "0") == "1"
    eth = balance(env, quiet=True)

    gains = None
    tag = "untrained (anatomy only)"
    p = ROOT / "build" / "gains_ui.npz"
    if p.exists():
        z = np.load(p, allow_pickle=False)
        gains = np.ones(fb.n_types, dtype=np.float32)
        gains[z["codes"]] = np.exp(z["theta"])
        tag = f"trained ({len(z['codes'])} cell types)"

    def _echo(m):
        t = m.get("type")
        if t == "step":
            say(f"  step {m['t']:02d} target={m['target']:<7} "
                f"filled={m['filled']} {m.get('note','')}")
        elif t == "done":
            say(f"DONE {m.get('outcome')} :: {str(m.get('msg') or '')[:250]}")
        elif t != "frame":
            say(f"{t.upper()}: {str(m.get('msg') or m)[:250]}")

    async def send(m):
        _echo(m)
        await ws.send_text(json.dumps(m))

    # These were built for the Solana rig and never ported here: a frame per
    # event is a slideshow, and the brain sat idle through setup because the
    # simulation only ran inside the control loop.
    stream = {"on": False, "note": "", "n": 0}

    async def streamer():
        while stream["on"]:
            try:
                raw = await page.screenshot(type="jpeg", quality=48)
                stream["n"] += 1
                await ws.send_text(json.dumps({
                    "type": "frame", "note": stream["note"],
                    "shot": base64.b64encode(raw).decode()}))
            except Exception:
                pass
            await asyncio.sleep(0.07)

    async def shot(note=""):
        stream["note"] = note
        try:
            raw = await page.screenshot(type="jpeg", quality=48)
            await ws.send_text(json.dumps({"type": "frame", "note": note,
                                           "shot": base64.b64encode(raw).decode()}))
        except Exception:
            pass

    await send({"type": "log", "msg": f"gains: {tag}"})
    await send({"type": "log",
                "msg": f"wallet {acct.address[:10]}... {eth:.6f} ETH on chain {CHAIN_ID}"})

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=not headful)
        page = await browser.new_page(viewport={"width": 1280, "height": 720})
        page.on("pageerror", lambda e: say(f"  [page.error] {str(e)[:150]}"))

        sent = {"n": 0}

        async def on_send(tx):
            sent["n"] += 1
            await send({"type": "log",
                        "msg": f"page asked to send a transaction to "
                               f"{str(tx.get('to'))[:14]}..."})
            if not live_flag:
                raise RuntimeError("FLY_RH_LIVE=0 - transaction refused")
            from rhprovider import send_transaction
            return send_transaction(acct, tx, rpc, CHAIN_ID,
                                    say=lambda m: say("  [tx] " + m))

        await attach(page, acct, rpc, CHAIN_ID, allow_send=live_flag,
                     on_send=on_send, log=lambda m: say("  [wallet] " + m))

        await send({"type": "log", "msg": f"loading {URL}"})
        await page.goto(URL, wait_until="domcontentloaded", timeout=60000)
        stream["on"] = True
        stream_task = asyncio.create_task(streamer())

        watch = {"on": True, "n": 0}

        async def watcher():
            loop_ = asyncio.get_running_loop()
            while watch["on"]:
                try:
                    raw = await page.screenshot(type="jpeg", quality=45)
                    arr = np.asarray(Image.open(io.BytesIO(raw)).convert("L"),
                                     dtype=np.float32) / 255.0
                    _, _, _, hz, fired = await loop_.run_in_executor(
                        None, step_brain, pilot, arr, 640.0, 300.0, gains,
                        9000 + watch["n"])
                    mm = remap[fired]
                    mm = np.unique(mm[mm >= 0]).astype(np.uint16)
                    watch["n"] += 1
                    await ws.send_text(json.dumps({
                        "type": "watch",
                        "hz": {k: round(v) for k, v in hz.items()},
                        "idx": base64.b64encode(mm.tobytes()).decode()}))
                except Exception:
                    pass
                await asyncio.sleep(0.2)

        watch_task = asyncio.create_task(watcher())
        await page.wait_for_timeout(5000)
        await go_dark(page, send)
        await dismiss_banners(page)
        await page.wait_for_timeout(1400)
        await shot("launchpad open")
        await asyncio.sleep(1.2)

        st = await page.evaluate(
            """() => ({connected: !!(window.ethereum && window.ethereum.selectedAddress),
                       addr: window.ethereum && window.ethereum.selectedAddress,
                       chain: window.ethereum && window.ethereum.chainId})""")
        await send({"type": "log", "msg": f"wallet in page: {st}"})
        await send({"type": "wallet", "connected": bool(st.get("connected")),
                    "pubkey": st.get("addr")})

        await accept_terms(page, send)
        await dismiss_banners(page)
        await page.wait_for_timeout(1200)
        await shot("terms accepted")
        await asyncio.sleep(1.3)

        img_p = ROOT / IMAGE
        if img_p.exists():
            try:
                # move to the drop zone and click it on camera, then attach
                try:
                    btn = page.get_by_text("Choose image", exact=False).first
                    await btn.scroll_into_view_if_needed(timeout=5000)
                    await page.wait_for_timeout(800)
                    box = await btn.bounding_box()
                    if box:
                        await page.mouse.move(box["x"] + box["width"] / 2,
                                              box["y"] + box["height"] / 2)
                        await page.wait_for_timeout(700)
                        await shot("choosing the token image")
                        await page.mouse.click(box["x"] + box["width"] / 2,
                                               box["y"] + box["height"] / 2)
                        await page.wait_for_timeout(600)
                except Exception:
                    pass
                inp = page.locator('input[type="file"]').first
                await inp.wait_for(state="attached", timeout=12000)
                await inp.set_input_files(str(img_p.resolve()), timeout=20000)
                await page.wait_for_timeout(3200)
                await send({"type": "log", "msg": f"token image selected: {img_p.name}"})
                await shot("image selected")
                await asyncio.sleep(1.6)
            except Exception as e:
                await send({"type": "log", "msg": f"image failed: {str(e)[:110]}"})

        watch["on"] = False
        try:
            await asyncio.wait_for(watch_task, timeout=3)
        except Exception:
            pass
        await send({"type": "log",
                    "msg": f"fly watched the launchpad for {watch['n']} brain steps "
                           f"- now taking the mouse"})

        cx, cy = 640.0, 250.0
        await page.mouse.move(cx, cy)
        filled, spikes_total = set(), 0
        loop = asyncio.get_running_loop()

        for t in range(steps):
            raw = await page.screenshot(type="jpeg", quality=60)
            arr = np.asarray(Image.open(io.BytesIO(raw)).convert("L"),
                             dtype=np.float32) / 255.0
            f = await page.evaluate(DOM_JS)
            for k in ("name", "ticker", "desc"):
                if f.get(k, {}).get("filled"):
                    filled.add(k)

            open_f = [(k, v) for k, v in f.items()
                      if k in ("name", "ticker", "desc") and k not in filled]
            if open_f:
                tk, tf = min(open_f, key=lambda kv: np.hypot(
                    cx - (kv[1]["x"] + kv[1]["w"] / 2),
                    cy - (kv[1]["y"] + kv[1]["h"] / 2)))
            elif "launch" in f:
                tk, tf = "launch", f["launch"]
                if tf["y"] < 40 or tf["y"] + tf["h"] > 700:
                    await page.evaluate(SCROLL_JS)
                    await page.wait_for_timeout(700)
                    f = await page.evaluate(DOM_JS)
                    if "launch" not in f:
                        break
                    tf = f["launch"]
            else:
                break

            dx, dy, click, hz, fired = await loop.run_in_executor(
                None, step_brain, pilot, arr, cx, cy, gains, seed + t)
            m = remap[fired]
            m = np.unique(m[m >= 0]).astype(np.uint16)
            spikes_total += int(len(fired))

            cx = float(np.clip(cx + dx, 2, 1278))
            cy = float(np.clip(cy + dy, 2, 718))
            await page.mouse.move(cx, cy)

            hit = click and inside(tf, cx, cy)
            note = ""
            if hit and tk != "launch":
                await page.mouse.click(cx, cy)
                await page.keyboard.type(str(coin[tk]), delay=75)
                filled.add(tk)
                note = f"typed {tk}"

            await send({"type": "step", "t": t, "cx": cx, "cy": cy, "target": tk,
                        "hz": {k: round(v) for k, v in hz.items()},
                        "click": bool(hit), "filled": sorted(filled),
                        "clicks": len(filled), "spikes": spikes_total, "note": note,
                        "create_label": f.get("launch", {}).get("label", ""),
                        "shot": base64.b64encode(raw).decode(),
                        "idx": base64.b64encode(m.tobytes()).decode()})
            if hit and tk == "launch":
                break
            await asyncio.sleep(0.05)

        await finish(ws, page, coin, filled, live_flag, send, shot, sent)
        watch["on"] = False
        stream["on"] = False
        try:
            await asyncio.wait_for(stream_task, timeout=3)
        except Exception:
            pass
        say(f"streamed {stream['n']} frames")
        await browser.close()


async def finish(ws, page, coin, filled, live_flag, send, shot, sent):
    SEL = {"name": 'input[placeholder="Token name"]',
           "ticker": 'input[placeholder="symbol"]',
           "desc": 'textarea[placeholder="A short description of the token"]'}
    by_fly = sorted(filled)
    by_rig = []
    before = {}
    for k, sel in SEL.items():
        try:
            before[k] = await page.input_value(sel, timeout=3000)
        except Exception:
            before[k] = "<unreadable>"
    await send({"type": "log", "msg": f"field values before completion: {before}"})

    for k, sel in SEL.items():
        cur = before.get(k, "")
        if cur == "<unreadable>" or cur.strip() == str(coin[k]).strip():
            continue
        try:
            el = page.locator(sel).first
            await scroll_to_el(page, f"document.querySelector({sel!r})", send)
            await page.wait_for_timeout(500)
            box = await el.bounding_box()
            if box:
                await glide(page, box["x"] + box["width"] / 2,
                            box["y"] + box["height"] / 2, send, hold=0.45)
                await page.mouse.click(box["x"] + box["width"] / 2,
                                       box["y"] + box["height"] / 2)
            await page.keyboard.press("Control+A")
            await page.keyboard.press("Delete")
            await asyncio.sleep(0.35)
            await page.keyboard.type(str(coin[k]), delay=105)
            await asyncio.sleep(0.9)
            await shot(f"typed {k}")
        except Exception:
            await page.fill(sel, str(coin[k]))
        by_rig.append(k)

    after = {}
    for k, sel in SEL.items():
        try:
            after[k] = await page.input_value(sel, timeout=3000)
        except Exception:
            after[k] = "<unreadable>"
    await send({"type": "log", "msg": f"FINAL field values: {after}"})
    await send({"type": "fields", "by_fly": by_fly, "by_rig": by_rig})
    await shot("form complete")
    await asyncio.sleep(1.4)

    await set_creator_tax(page, 3, send, shot)
    await asyncio.sleep(1.2)

    await scroll_to_el(page,
        "[...document.querySelectorAll('button')]"
        ".map(b=>({b,r:b.getBoundingClientRect()}))"
        ".filter(o=>o.r.width>380 && o.r.height>40)"
        ".sort((p,q)=>(q.r.y+scrollY)-(p.r.y+scrollY))[0].b", send)
    await page.wait_for_timeout(700)
    await dismiss_banners(page)
    await shot("scrolled to launch")
    f = await page.evaluate(DOM_JS)
    c = f.get("launch")
    if not c:
        await send({"type": "done", "outcome": "error",
                    "msg": "launch button not found"})
        return
    await send({"type": "log",
                "msg": f"launch button '{c['label']}' at ({c['x']},{c['y']}) "
                       f"disabled={c['disabled']}"})
    await glide(page, c["x"] + c["w"] / 2, c["y"] + c["h"] / 2, send,
                steps=30, hold=1.6)
    await shot("fly on the launch button")
    await asyncio.sleep(1.4)

    if not live_flag:
        await send({"type": "done", "outcome": "dry",
                    "filled": by_fly + by_rig,
                    "msg": f"FLY_RH_LIVE=0 - stopped on the '{c['label']}' button. "
                           f"Nothing signed, nothing sent. "
                           f"transaction requests seen: {sent['n']}"})
        return

    await send({"type": "log", "msg": "FLY_RH_LIVE=1 - pressing launch"})
    await page.mouse.click(c["x"] + c["w"] // 2, c["y"] + c["h"] // 2)
    for i in range(30):
        await page.wait_for_timeout(2000)
        await shot("waiting for the launch")
        if sent["n"]:
            break
    await send({"type": "done",
                "outcome": "minted" if sent["n"] else "clicked",
                "result": page.url,
                "msg": f"transaction requests: {sent['n']}"})


@app.websocket("/run")
async def run(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            if msg.get("action") != "start" or STATE["running"]:
                continue
            if not browser_allowed():
                await ws.send_text(json.dumps({
                    "type": "done", "outcome": "blocked",
                    "msg": "FLY_ALLOW_BROWSER is not 1"}))
                continue
            STATE["running"] = True
            coin = {"name": msg.get("name") or "test",
                    "ticker": msg.get("ticker") or "test",
                    "desc": msg.get("desc") or "launched by a fruit fly connectome"}
            try:
                await run_episode(ws, coin, int(msg.get("steps", 18)),
                                  int(msg.get("seed", 350)),
                                  bool(msg.get("headful")))
            except Exception as e:
                import traceback
                say("RUN FAILED:")
                traceback.print_exc()
                await ws.send_text(json.dumps({"type": "done", "outcome": "error",
                                               "msg": repr(e)}))
            finally:
                STATE["running"] = False
    except WebSocketDisconnect:
        STATE["running"] = False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=4651)
    a = ap.parse_args()
    boot()
    print(f"\n  Robinhood Chain rig - open http://localhost:{a.port}\n")
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
