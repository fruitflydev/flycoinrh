"""
The one room in which the fly can trade.

The fly roams the open web. Somewhere in that roam is a door into a room this
project draws itself: a near-black page holding a grid of coin cards. The fly
walks it exactly as it walks any other page - same eye, same descending
neurons, same stopping rule - but here a stop on a card is read as a commit,
and a commit becomes an intent sent to the executor over loopback. This module
is the room: it knows what is on the page, how long the fly has been looking at
one card, what the mushroom body said while it looked, and what came back from
the executor afterwards.

It cannot trade. It holds no key, imports no chain library and no signer, and
its only way out is one HTTP POST to 127.0.0.1 carrying a coin, a side and a
number between -1 and 1. There is no route, file, flag or environment variable
here that places, sizes, cancels or reverses anything, and nothing in the
decision path reads a price, a market cap, a profit, a holding time, a cap or
a limit. Size comes from the brain's drive and the balances, and from nothing
else.

MEASURED
  * the connectome, the retinotopic columns, the olfactory receptor neurons,
    the Kenyon cells and the mushroom body output neurons, and which dopamine
    cluster innervates each MBON (build/mb_sides.json);
  * every spike a run produces, and which Kenyon cells fired in it - the two
    readings below are counted synapses, not scores;
  * the DoOR receptor responses behind each coin's smell (olfaction.py);
  * every number the executor reports back: quotes, amounts and balances.

CHOSEN BY PEOPLE, and said so wherever it shows
  * that a coin has a smell at all, and which odorant a coin's words name;
  * which coins the fly is ever offered: pons_board() asks the launchpad for
    the sixteen coins with the most recent buying flow (sort="recentBuys", any
    age, first page, refetched every 20 s), and _board_worker drops the ones
    that are not quoted in the chain's own coin. The listing offers newest,
    oldest, market cap and volume as well; a person picked this one, and it is
    a momentum screen applied before the fly ever sees a card, so what the
    paper book earns is partly the screen's and not the fly's;
  * the room's layout, its ground grey and its card size;
  * that a look is read off the mushroom body's output synapses - the Kenyon
    cells that fired, times the weights dopamine changes - rather than off the
    MBONs' firing rates. Measured 2026-09-12: with the rate readout both sugar
    and shock lowered the score, because those MBONs also carry recurrent input
    from the rest of the brain, so a rewarded coin came out backwards;
  * that a card is measured against the other cards this fly has looked at
    during this visit, and against no blank frame. There is no control run at
    all. Measured 2026-09-12: a uniform ground-grey frame is itself a strong
    stimulus - it fired more approach MBONs (13,075) than a card did (6,796) -
    so training moved the control more than it moved the cards and shock came
    out inverted; and an absolute reading carries almost none of what was
    learned, since learning here is partly global. A card against the other
    cards is what the offline gate passed on (build/backroom_screen.json: the
    coin paired with profit +0.0359 at 5.6 SE, the coin paired with loss
    -0.0399 at 5.8 SE, an untouched coin +0.0041 at 0.8 SE, leak 0.10);
  * WHAT THE GATE DOES NOT COVER, and the room does anyway. The gate measured a
    mean over 60 paired seeds with one picture on every card and the cursor at
    its centre, so that only smell differed. The room commits on one dwell,
    where three things the gate held fixed are free. The per-look spread is
    larger than the lesson: SE * sqrt(60) is 0.076, 0.053 and 0.068 against a
    lesson of 0.036 to 0.040. The untrained coins already differ from each
    other by more than the lesson: +0.0346, -0.0480 and +0.0133 with identical
    pictures, so a coin whose name hashes to a strong odorant carries a fixed
    offset that a visit's worth of looks will not average away. And the eye
    window is 300x210 while the cards are 280x200 fourteen pixels apart, so a
    stop that is not at a card's centre reads its neighbours too: measured over
    the first paper run's nine commits, 29.5% to 85.4% of the window was the
    card being judged (each look record now carries its own `on_card`). None of
    this is capped or corrected here - nothing may stand between the brain and
    the order - but none of it is what the gate certified either;
  * the formula drive = clip(c(this card) - mean(c of the other cards seen this
    visit), -1, 1), where c = (A-V)/(A+V), averaged over the dwell. It has no
    constant. The leaning is a ratio because a coin that fires five times as
    many Kenyon cells as another would otherwise decide every comparison it
    takes part in - which is the other way sugar came out backwards. Both this
    and the reading above are calibration.leaning and calibration.relative, the
    same two functions the offline gate uses, so the two cannot drift apart;
  * that every coin's smell is scaled toward the same total (olfaction.Nose
    equal_sniff). Measured 2026-09-12: geosmin alone fires 42% of the Kenyon
    cells against isopentyl acetate's 4.9%, so without it the loud coins swamp
    the quiet ones. It does not reach equality: no receptor may respond above
    1.0, so a coin whose smell lands on one glomerulus totals 1.0 where a coin
    spread over twenty totals 2.0. In the first paper run the totals were 1.00,
    1.24, 1.48, 1.49 and 2.00, and the quiet looks read 226-239 approach a step
    against 24,275-29,211 for the loud ones;
  * that a look which fired no Kenyon cell is not a reading at all: it is
    neither a drive nor a reference, and the dwell counts it as blind
    (calibration.has_reading). leaning() scores an empty reading 0.0, which is
    about a quarter of the range above every card measured so far;
  * two steps of dwell as "enough evidence" before a stop counts as a commit;
  * that profit arrives as reward dopamine and loss as punishment dopamine,
    and that a lesson is worth amount = 1 - min(q, 1/q);
  * that the first look at a coin the fly has just bought only sets the
    reference it will be judged against, because a cost and a sale price
    differ by the venue's fees whatever the coin does - and the same for a
    sell that settles before the fly ever looked at what it bought: there is
    no fee-free reference to judge it against, so it teaches nothing;
  * that eligibility is wiped when the fly arrives at a card, and both before
    and after a settled sell is shown to it, so a lesson lands on the coin it
    is about and the coin just shown cannot take the next card's lesson;
  * that a settled sell with no stored picture teaches nothing, rather than
    teaching the coin on the blank ground-grey frame every card shares;
  * that a delivery empties the room behind the fly: readings taken under
    weights that have since moved are not what this visit is comparing;
  * that this build is paper: nothing here signs, and the executor books
    simulated fills against live quotes. Every record written says so.

  py backroom.py replay <look_id>    recompute one stored look on the real brain
"""
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

import calibration

ROOT = Path(__file__).parent

# The page's ground colour, and the grey the eye actually samples for it.
# roam turns each screencast frame into luminance with PIL's 'L', which is
# ITU-R 601, so anything this module draws for itself has to use the same
# arithmetic or it would not be a picture of this room. The one thing it draws
# is the empty room a settled sell's stored eye window is put back into
# (_represent); nothing is ever run against a blank frame as a control.
GROUND_CSS = "#0b0c0e"


def grey601(css):
    """ITU-R 601 luminance, 0..1, of a #rrggbb colour."""
    h = str(css).lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255.0


GROUND = grey601(GROUND_CSS)

TOKEN_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")
BOARD_CARDS = 16           # how many coins the board shows at once (CHOSEN)
DWELL_MIN = 2              # looks of evidence before a stop is a commit (CHOSEN)
MAX_LOOKS = 2000           # look records kept on disk (CHOSEN)
FOV_W, FOV_H = 300, 210    # the eye window FlyEye.look samples, in screen pixels
BOARD_MAX_AGE_S = 20.0     # how stale a board may be before a visit refetches it
EVENTS_EVERY_S = 2.0       # how often the room asks the executor what settled
# An intent and a mark both wait on the executor's chain quotes, which are
# about nineteen sequential eth_calls. MEASURED: the public Robinhood Chain RPC
# answered a mark in 14-15 s in the first paper run, so the room's ordinary
# ten-second wait threw away answers that were on their way. These two calls
# wait longer than the rest so a slow but successful quote is read rather than
# discarded. A keyed RPC answers far quicker; the number is CHOSEN to sit well
# above the slowest round trip seen.
CHAIN_TIMEOUT_S = 45.0

# Card rectangles, read from the page the fly is actually looking at. The
# coordinates are viewport coordinates, which is the same frame as the 1280x800
# screencast the eye samples, so a rectangle here and a cursor there mean the
# same thing.
RECTS_JS = """() => Array.from(document.querySelectorAll('[data-token]')).map(e => {
  const r = e.getBoundingClientRect();
  return {token: e.getAttribute('data-token') || '',
          held: e.getAttribute('data-held') === '1',
          x: r.left, y: r.top, w: r.width, h: r.height};
})"""


def amount_for(q):
    """
    How big a lesson is, from a ratio of two values in ETH.

    amount = 1 - min(q, 1/q): a doubling and a halving weigh the same, a 1%
    move is worth about 0.01, and losing everything is worth 1. There is no
    constant to tune. CHOSEN.
    """
    q = float(q)
    if q <= 0.0:
        return 1.0
    return float(1.0 - min(q, 1.0 / q))


def sign_for(q):
    """+1 reward when the value rose, -1 punishment when it fell, 0 when it did not move."""
    q = float(q)
    if q > 1.0:
        return 1
    if q < 1.0:
        return -1
    return 0


def answered_unbooked(status, http=None):
    """
    True when the executor answered and its answer was not a booking.

    The HTTP code decides this, not the body's own word for itself. The
    executor answers a malformed intent with 400 {"status": "error"} and an
    unauthenticated one with 403 {"status": "forbidden"}, so a rule that
    matched on the body never recognised either: those looks stayed pinned for
    the life of the install, waiting for an event the executor never wrote.
    Any 4xx is the executor saying it did not take the intent, and so are the
    two bodies it sends with 200 ("refused") and 409 ("busy"). "unreachable"
    and a 5xx are not answers at all - the intent may still be settling on the
    far side of a slow chain quote - and what settled is decided by the event
    stream, never by the room giving up on a reply.
    """
    try:
        code = int(http)
    except (TypeError, ValueError):
        code = None
    if code is not None and 400 <= code < 500:
        return True
    if code is not None and code >= 500:
        return False                  # the far side broke; nothing was said about the intent
    s = str(status)
    return s in ("refused", "busy") or s.startswith("http 4")


def _atomic_write(path, data: bytes):
    """Write then rename, so a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _field(item, name, default=None):
    """Read one field from a pons Coin or from a plain dict."""
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _thread(fn, *args):
    """The default worker: anything that waits on the network waits off the brain's thread."""
    threading.Thread(target=fn, args=args, daemon=True).start()


class LoopbackHttp:
    """
    The only way out of this module: JSON over HTTP to 127.0.0.1.

    Kept behind an object so a test can hand the room a fake and no test ever
    opens a socket.
    """

    def __init__(self, timeout=10.0):
        self.timeout = float(timeout)

    def _call(self, url, data=None, headers=None, timeout=None):
        req = urllib.request.Request(
            url, data=data, method="POST" if data is not None else "GET",
            headers={"Accept": "application/json", **({"Content-Type": "application/json"} if data else {}),
                     **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=float(timeout or self.timeout)) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as exc:            # a refusal is an answer, not a failure
            try:
                return exc.code, json.loads(exc.read() or b"{}")
            except Exception:
                return exc.code, {}

    def get_json(self, url, headers=None, timeout=None):
        return self._call(url, None, headers, timeout)

    def post_json(self, url, body, headers=None, timeout=None):
        return self._call(url, json.dumps(body).encode("utf-8"), headers, timeout)


def pons_board(page_size=BOARD_CARDS):
    """
    The live board from the pons listing API.

    Imported here rather than at module load so that importing backroom pulls
    in no chain library at all.
    """
    import pons
    return pons.fetch_board(sort="recentBuys", age="all", page_size=page_size, page=1)


class Room:
    """
    What the fly does while it is in the backroom.

    Every brain operation - the one run a step, observe, forget, dopamine,
    apply, save - happens on the thread that called step() or poll_events().
    Only waiting on the network is handed to a worker, because there is one
    brain and it must not be run from two threads.

    A step runs the brain exactly once. What the fly sees and smells, the stop,
    and the Kenyon cells that fired all come out of that single run, and the
    card is judged against the readings of the other cards taken earlier in the
    same visit rather than against a second run made for the purpose.
    """

    max_looks = MAX_LOOKS
    dwell_min = DWELL_MIN

    def __init__(self, fb, pilot, mb, nose, gains, state_dir, executor_url,
                 intent_token, fetch_board=None, http=None, clock=time.time, spawn=None):
        self.fb = fb
        self.pilot = pilot
        self.mb = mb
        self.nose = nose
        self.gains = gains
        self.dir = Path(state_dir) / "backroom"
        self.looks_dir = self.dir / "looks"
        self.public = self.dir / "public" / "public.json"
        self.room_file = self.dir / "room.json"
        self.dopamine_file = self.dir / "dopamine.jsonl"
        self.executor_url = str(executor_url).rstrip("/")
        self.intent_token = intent_token
        self.fetch_board = fetch_board or pons_board
        self.http = http or LoopbackHttp()
        self.clock = clock
        self.spawn = spawn or _thread
        # the approach and avoidance MBON populations, recorded in every run so
        # their rates are on the record and in the stream. Nothing decides on
        # them: what a look is worth is read off the synapses (calibration.py).
        self.readout = calibration.readout(mb)

        self.in_room = False
        self.counters = {"visits": 0, "looks": 0, "commits": 0, "intents": 0,
                         "booked": 0, "refused": 0, "dislikes": 0, "busy": 0,
                         "sugar": 0, "shock": 0}
        self.mark_ref = {}         # token -> the ETH value this coin is judged against
        self.from_cost = set()     # tokens whose reference is a buy's cost, not a mark
        self.book_rev = {}         # token -> how many settlements this room has applied to it
        self.meta = {}             # token -> remembered card metadata, so a holding is still a card
        self.refs = {}             # token -> look ids that must survive pruning
        self.last_seq = 0
        # False until this room knows where it is in the executor's event
        # stream. Starting at 0 and asking for everything would replay a whole
        # ledger of settled sells as if they had just happened.
        self.seq_known = False
        self.last_intents = []
        self.last_dopamine = []

        self._lock = threading.Lock()
        self._board = {"cards": [], "updated": 0.0}
        self._board_busy = False
        self._board_error = None   # why the last fetch failed, for state() and the log
        self._dwell = None
        self._intent = None
        self._events_busy = False
        self._events_result = None
        self._events_at = 0.0
        self._smell = {}
        # What this visit has looked at: token -> the (approach, avoid) reading
        # of the fly's last look at that card. This is the room a card is judged
        # against, and it is emptied on the way in, on the way out and on every
        # delivery of dopamine (_forget_room), because a reading taken under
        # weights that have since learned is not what this visit is comparing.
        self._seen = {}
        self._look_seq = 0
        self._held = set()         # what the book holds, re-read once per step
        self._said_no_book = False
        self._load_room()

    def _say(self, msg):
        """One line into the roamer's log. The room is silent otherwise."""
        try:
            print(f"[backroom] {msg}", flush=True)
        except Exception:
            pass

    # -- the board --------------------------------------------------------
    def board(self):
        """What the page draws: the listing's coins, and every coin the book holds."""
        book = self.read_book()
        positions = book.get("positions") or []
        held = {p.get("token") for p in positions if p.get("token")}
        cards = [dict(c, held=c["token"] in held) for c in self._board["cards"]]
        holdings = []
        for p in positions:
            token = p.get("token")
            if not token:
                continue
            self.remember(token, {"name": p.get("name"), "symbol": p.get("symbol")})
            m = self.meta.get(token, {})
            holdings.append({"token": token, "name": m.get("name") or "",
                             "symbol": m.get("symbol") or "", "logo": m.get("logo") or "",
                             "market_cap_usd": m.get("market_cap_usd"),
                             "graduation_pct": m.get("graduation_pct"), "held": True})
        return {"cards": cards, "holdings": holdings, "updated": int(self._board["updated"])}

    def refresh_board(self, force=False):
        """Ask the listing for a new board, off this thread. The last good board stays until one arrives."""
        if self._board_busy:
            return False
        if not force and self._now() - self._board["updated"] < BOARD_MAX_AGE_S:
            return False
        self._board_busy = True
        self.spawn(self._board_worker)
        return True

    def _board_worker(self):
        try:
            coins = self.fetch_board() or []
            cards = []
            for coin in coins:
                token = str(_field(coin, "token") or "")
                if not TOKEN_RE.match(token):
                    continue
                if not _field(coin, "quote_is_native", True):
                    continue                      # only native-quote coins are settleable
                cards.append({"token": token,
                              "name": str(_field(coin, "name") or ""),
                              "symbol": str(_field(coin, "symbol") or ""),
                              "logo": str(_field(coin, "logo_url") or ""),
                              "market_cap_usd": _field(coin, "market_cap_usd"),
                              "graduation_pct": _field(coin, "progress_pct"),
                              "held": False})
                self.remember(token, {"name": cards[-1]["name"], "symbol": cards[-1]["symbol"],
                                      "description": str(_field(coin, "description") or ""),
                                      "logo": cards[-1]["logo"],
                                      "market_cap_usd": cards[-1]["market_cap_usd"],
                                      "graduation_pct": cards[-1]["graduation_pct"]})
                if len(cards) >= BOARD_CARDS:
                    break
            if cards:
                self._board = {"cards": cards, "updated": self._now()}
                self._board_error = None
                self._save_room()
        except Exception as exc:
            # The last good board must survive a failed fetch, so the failure
            # is swallowed - but silently it is indistinguishable from a fly
            # that never reached the room: an empty grid has no rects, no
            # dwell and no commit, and every published number agrees that
            # nothing is wrong. Said once, and kept where state() can show it.
            first, self._board_error = self._board_error is None, f"{type(exc).__name__}: {exc}"[:120]
            if first:
                self._say(f"the listing could not be read: {self._board_error}")
        finally:
            self._board_busy = False

    def remember(self, token, meta):
        """Keep a coin's words, so a holding that fell off the board is still a card and still smells."""
        cur = dict(self.meta.get(token) or {})
        for k, v in (meta or {}).items():
            if v not in (None, ""):
                cur[k] = v
        self.meta[token] = cur

    def read_book(self):
        """
        The executor's public book, read from disk. The room never writes it.

        A missing file is said out loud once, because it is indistinguishable
        from a fly that never chose to sell: with no book nothing is ever held,
        so no card is marked, no sell can be sent and no mark can be asked for.
        The usual cause is the two processes resolving FLY_STATE_DIR
        differently.
        """
        try:
            book = json.loads(self.public.read_text(encoding="utf-8"))
        except FileNotFoundError:
            if not self._said_no_book:
                self._said_no_book = True
                self._say(f"no paper book at {self.public} - the executor writes it; "
                          "until it exists the room holds nothing")
            return {}
        except Exception as exc:
            if not self._said_no_book:
                self._said_no_book = True
                self._say(f"the paper book at {self.public} cannot be read: {str(exc)[:90]}")
            return {}
        self._said_no_book = False
        return book

    def _refresh_held(self):
        """
        What the book says the fly holds, read once a step.

        This is the authority on held, not the page: web/backroom.html repaints
        data-held on its own twenty-second poll, and a whole dwell fits inside
        that window, so a coin bought seconds ago still draws as unheld. The
        attribute is there for the fly to look at; the side of a commit is
        decided here.
        """
        self._held = {p.get("token") for p in (self.read_book().get("positions") or [])
                      if p.get("token")}
        return self._held

    @property
    def mode(self):
        """paper or live, as the executor's own book reports it. This build is paper."""
        return str(self.read_book().get("mode") or "paper")

    def smell_of(self, token):
        """What this coin smells of, computed once per coin and kept."""
        if token not in self._smell:
            m = self.meta.get(token) or {}
            self._smell[token] = self.nose.smell(m.get("name") or "", m.get("symbol") or "",
                                                 m.get("description") or "")
        return self._smell[token]

    # -- being in the room ------------------------------------------------
    async def enter(self, page):
        """
        The fly walked in. Refresh the board without waiting for it.

        The visit starts with an empty room behind it: nothing it saw on its
        last visit is a reference for what it sees now, so the first card it
        looks at has nothing to be judged against and commits nothing.
        """
        self.in_room = True
        self.counters["visits"] += 1
        self._dwell = None
        self._seen = {}
        self.refresh_board(force=True)
        self._save_room()

    def leave(self):
        """The fly walked out. A half-finished dwell is not evidence of anything."""
        self.in_room = False
        self._dwell = None
        self._seen = {}
        self._save_room()

    async def step(self, page, img, cx, cy, seed):
        """
        One control step in the room, in place of pilot.step.

        One run of the brain, and it is the fly's actual experience: the page
        through its eye, plus the smell of whatever card the cursor is inside.
        That run gives the cursor move, the stop and the Kenyon cells that
        fired, so the reading, the eligibility and the commit all come off the
        same moment. What the card is judged against is the room the fly has
        already walked: the reading of its last look at each other card in this
        visit. Nothing is run twice and nothing blank is run at all.
        """
        self._collect_intent()
        self._refresh_held()
        rects = await self._rects(page)
        card = self._card_at(rects, cx, cy)
        token = card["token"] if card else None
        if token is not None and (self._dwell is None or self._dwell["token"] != token):
            # Arriving at a card. Whatever was eligible before it - another
            # card, the bare ground, the web page the fly was on before the
            # room - stops being eligible, so a lesson delivered during this
            # dwell can only reach this coin.
            self.mb.forget_trace()

        smell = self.smell_of(token) if token else None
        dx, dy, click, hz, info = self.pilot.step(
            img, cx, cy, gains=self.gains, seed=seed, detail=True,
            extra_drive=(self.nose.drive(smell) if smell else None),
            extra_record=self.readout)

        # the one place a weight may move later: what fired now becomes eligible
        self.mb.observe(info.get("fired"))
        self.mb.forget()

        dwell = self._advance_dwell(card)
        if dwell is not None:
            own = calibration.syn_drive(self.mb, info.get("fired"))
            if not calibration.has_reading(own):
                # No Kenyon cell fired, so the mushroom body carried nothing
                # about this card. calibration.leaning scores that 0.0, which
                # is a quarter of the range above every card the room has ever
                # measured, so spending it as a drive or as a reference is
                # spending a number the brain did not produce. The step is
                # counted as a blind one and is otherwise not a look.
                dwell["blind"] += 1
            else:
                others = self._room_behind(token)
                dwell["steps"] += 1
                dwell["A"] += own[0]
                dwell["V"] += own[1]
                dwell["kc"] += self._kc_fired(info.get("fired"))
                dwell["lean_sum"] += calibration.leaning(own)
                dwell["cursor"] = [float(cx), float(cy)]
                self.counters["looks"] += 1
                if others:
                    dwell["drive_sum"] += calibration.relative(own, [r for _, r in others])
                    dwell["pairs"] += 1
                    dwell["reference"] = [t for t, _ in others]
                # this look becomes part of the room the next card is judged against
                self._seen[token] = own

        drive = self._drive_of(dwell)
        if click and dwell is not None and dwell["steps"] >= self.dwell_min:
            self._commit(img, cx, cy, seed, card, dwell, drive, smell)
            self._dwell = None                     # a commit spends the evidence
        elif click:
            pass                                   # a stop off a card, or too soon, is nothing

        if self._dwell is not None and self._dwell.get("held"):
            self._learn_from_mark()

        info["backroom"] = {"token": token, "drive": drive,
                            "dwell_steps": int(self._dwell["steps"]) if self._dwell else 0}
        return dx, dy, click, hz, info

    # -- what is under the cursor ----------------------------------------
    async def _rects(self, page):
        try:
            raw = await page.evaluate(RECTS_JS)
        except Exception:
            return []
        out = []
        for r in raw or []:
            token = str(r.get("token") or "")
            w, h = float(r.get("w") or 0.0), float(r.get("h") or 0.0)
            if not TOKEN_RE.match(token) or w <= 0.0 or h <= 0.0:
                continue
            out.append({"token": token, "held": bool(r.get("held")),
                        "x": float(r.get("x") or 0.0), "y": float(r.get("y") or 0.0),
                        "w": w, "h": h})
        return out

    @staticmethod
    def _card_at(rects, cx, cy):
        for r in rects:
            if r["x"] <= cx < r["x"] + r["w"] and r["y"] <= cy < r["y"] + r["h"]:
                return r
        return None

    # -- dwell ------------------------------------------------------------
    def _advance_dwell(self, card):
        """
        A dwell is consecutive steps inside one card. Leaving it, or entering
        another, ends it.

        `steps` counts the steps that read something - a step whose run fired
        no Kenyon cell measured nothing and is counted as `blind` instead - so
        the two steps of evidence a commit needs are two readings.
        """
        if card is None:
            self._dwell = None
            return None
        d = self._dwell
        if d is None or d["token"] != card["token"]:
            d = {"token": card["token"], "steps": 0, "blind": 0, "A": 0.0, "V": 0.0,
                 "kc": 0, "lean_sum": 0.0,
                 "pairs": 0, "drive_sum": 0.0, "reference": [],
                 "cursor": None, "mark": None, "card": card}
            self._dwell = d
        d["held"] = card["token"] in self._held      # the book, not the page
        d["card"] = card
        return d

    def _kc_fired(self, fired):
        """
        How many Kenyon cells that run fired, counted the way the offline gate
        counts them (backroom_screen.py: np.isin(kc, fired)), so the number in
        a look record can be read against the gate's 0.050 to 0.235.
        """
        kc = getattr(self.mb, "kc", None)
        if fired is None or kc is None or not len(kc):
            return 0
        return int(np.isin(np.asarray(kc), np.asarray(fired)).sum())

    def _drive_of(self, dwell):
        """Each look of the dwell against the room behind it, averaged."""
        if not dwell or dwell["steps"] <= 0 or dwell["pairs"] <= 0:
            return None
        return float(np.clip(dwell["drive_sum"] / float(dwell["pairs"]), -1.0, 1.0))

    def _room_behind(self, token):
        """
        The other cards this visit has looked at: [(token, (approach, avoid))].

        This is the other arm of the T-maze. A fly in a T-maze is offered two
        arms and chooses between them, and that comparison is the only form the
        offline gate could measure a lesson in (build/backroom_screen.json): an
        absolute reading, against a blank frame or against nothing, carries
        almost none of what was learned, because learning here is partly global
        and moves every coin's reading together.

        The gate can afford to look at every coin at the same seed. A walking
        fly cannot: it has one run a step and that run is its own experience, so
        the room it is compared with is the one it has already walked. Each card
        counts once, at the fly's most recent look at it, and the card under the
        cursor is not part of its own reference. Until the fly has looked at a
        second card there is no reference at all, and a stop commits nothing.

        A card whose look fired no Kenyon cell is not in here. It is not a
        measurement of that card, and leaning() would score it 0.0, which is
        about a quarter of the range above every card this project has
        measured (calibration.has_reading).
        """
        return sorted((t, r) for t, r in self._seen.items()
                      if t != token and calibration.has_reading(r))

    # -- the commit -------------------------------------------------------
    def _commit(self, img, cx, cy, seed, card, dwell, drive, smell):
        at = self._now()
        token = card["token"]
        m = self.meta.get(token) or {}
        symbol = m.get("symbol") or ""
        self.counters["commits"] += 1
        look_id = self._write_look(at, img, cx, cy, seed, card, dwell, drive, smell)

        if drive is None:
            # The fly stopped on the first card it has looked at this visit, so
            # this card leans against nothing and there is no reading to act on.
            # The stop is on the record; no intent goes anywhere.
            self._note_intent(at, symbol, "none", None, "no reference",
                              "the fly has looked at no other card in this visit")
            return
        if drive == 0.0:
            return                                  # a stop that said nothing
        if drive > 0.0:
            self._send_intent(token, "buy", drive, look_id, at, symbol)
            return
        if token in self._held:                      # the book, not the page's data-held
            self._send_intent(token, "sell", drive, look_id, at, symbol)
            return
        # disliking a coin it does not hold is not a short and never will be
        self.counters["dislikes"] += 1
        self._note_intent(at, symbol, "none", drive, "dislike", "the fly does not hold this coin")

    def _send_intent(self, token, side, drive, look_id, at, symbol):
        if self._intent is not None:
            self.counters["busy"] += 1
            self._note_intent(at, symbol, side, drive, "busy", "an intent was already in flight")
            return
        body = {"token": token, "side": side, "drive": float(drive),
                "seen_at": float(at), "look_id": look_id}
        self._intent = {"body": body, "symbol": symbol, "at": at, "done": False, "result": None}
        self.counters["intents"] += 1
        self.refs.setdefault(token, []).append(look_id)
        self._save_room()
        self.spawn(self._intent_worker, self._intent)

    def _intent_worker(self, slot):
        # The transport's own answer is kept beside the body's: the executor's
        # 4xx bodies describe themselves as "error" and "forbidden", so the
        # code is the only thing that reliably says whether it took the intent.
        out = {"status": "unreachable", "reason": "no answer from the executor", "http": None}
        try:
            status, obj = self.http.post_json(
                self.executor_url + "/intent", dict(slot["body"]),
                headers={"X-Fly-Intent": self.intent_token},
                timeout=CHAIN_TIMEOUT_S)
            if isinstance(obj, dict) and obj:
                out = dict(obj)
                out.setdefault("status", f"http {status}")
            else:
                out = {"status": f"http {status}", "reason": ""}
            out["http"] = status
        except Exception as exc:
            out = {"status": "unreachable", "reason": str(exc)[:120], "http": None}
        slot["result"] = out
        slot["done"] = True

    def _collect_intent(self):
        """Pick up an intent posted on an earlier step. Booking is decided by the event stream."""
        slot = self._intent
        if slot is None or not slot.get("done"):
            return
        res = slot.get("result") or {}
        status = str(res.get("status") or "unknown")
        self._note_intent(slot["at"], slot["symbol"], slot["body"]["side"],
                          slot["body"]["drive"], status, res.get("reason"))
        # A look stops being pinned only when the answer proves nothing was
        # booked. No answer proves nothing: the executor has to quote the chain
        # before it replies, and in the first paper run it booked a buy several
        # seconds after the room had given up waiting for it. Dropping the
        # reference there leaves an open position whose own buy look can be
        # pruned away, which is the one thing pruning must never do.
        if answered_unbooked(status, res.get("http")):
            self._drop_ref(slot["body"]["token"], slot["body"]["look_id"])
        self._intent = None
        self._save_room()

    def _note_intent(self, at, symbol, side, drive, status, reason=None):
        self.last_intents.append({"at": int(at), "symbol": symbol, "side": side,
                                  "drive": None if drive is None else round(float(drive), 6),
                                  "status": status,
                                  "reason": (str(reason)[:120] if reason else None)})
        self.last_intents = self.last_intents[-10:]

    # -- look records -----------------------------------------------------
    def _write_look(self, at, img, cx, cy, seed, card, dwell, drive, smell):
        """
        Everything needed to recompute this decision later, and the picture the
        fly's eye actually had. `backroom.py replay <look_id>` re-runs it.

        A and V are sums over the dwell's reading steps and A_mean and V_mean
        are those divided by the steps; `leaning` is the mean of the per-step
        ratios, which is what the drive was computed from, and
        `leaning_of_sums` is (A-V)/(A+V) of the sums. The two differ - by up to
        0.042 in the first paper run, which is the size of the whole lesson the
        gate measured - so both are written down rather than leaving a reader
        to recompute one from the other and get the wrong number.

        `kc_mean` and `kc_frac` are how many Kenyon cells fired per step, and
        what share of the brain that is: a look far below the gate's 0.050 to
        0.235 fired the lowest-threshold cells only and says little about which
        coin it was.

        `on_card` is how much of the eye window was the card being judged. The
        offline gate held the picture, the cursor and the seed identical so
        that only smell differed; the room cannot, because the fly stops where
        it stops, and the 300x210 window overlaps the neighbouring cards. It
        was 0.295 to 0.854 over the first paper run's nine commits, and a
        reader of one look should be able to see which.

        The reference is the other cards the visit had looked at, each with the
        leaning its own last look gave and with its smell, because those
        readings were taken at other cards and at other moments and cannot be
        reproduced from this record alone.
        """
        self._look_seq += 1
        look_id = f"{int(at * 1000):013d}-{self._look_seq:04d}"
        h, w = int(img.shape[0]), int(img.shape[1])
        # exactly the region FlyEye.look samples, clipped to the screen the
        # same way it clips, so a replay sees every pixel that drove the retina
        x0 = max(0, min(w, int(round(cx - FOV_W / 2.0))))
        y0 = max(0, min(h, int(round(cy - FOV_H / 2.0))))
        x1 = max(x0, min(w, int(round(cx + FOV_W / 2.0))))
        y1 = max(y0, min(h, int(round(cy + FOV_H / 2.0))))
        m = self.meta.get(card["token"]) or {}
        steps = max(1, int(dwell["steps"]))
        reference = [{"token": t, "leaning": calibration.leaning(self._seen[t]),
                      "smell": self.smell_of(t)}
                     for t in dwell["reference"]
                     if t in self._seen and calibration.has_reading(self._seen[t])]
        kc = getattr(self.mb, "kc", None)
        kc_cells = max(1, 0 if kc is None else len(kc))
        # how much of the eye window was the card this record is about
        ox = max(0.0, min(float(x1), card["x"] + card["w"]) - max(float(x0), card["x"]))
        oy = max(0.0, min(float(y1), card["y"] + card["h"]) - max(float(y0), card["y"]))
        window = float(max(1, (x1 - x0) * (y1 - y0)))
        record = {
            "look_id": look_id, "at": at, "token": card["token"],
            "symbol": m.get("symbol") or "", "name": m.get("name") or "",
            "smell": smell, "card_rect": [card["x"], card["y"], card["w"], card["h"]],
            "cursor": [float(cx), float(cy)], "seed": int(seed), "drive": drive,
            "A": dwell["A"], "V": dwell["V"], "leaning": dwell["lean_sum"] / float(steps),
            "A_mean": dwell["A"] / float(steps), "V_mean": dwell["V"] / float(steps),
            "leaning_of_sums": calibration.leaning((dwell["A"], dwell["V"])),
            "kc_mean": dwell["kc"] / float(steps),
            "kc_frac": dwell["kc"] / float(steps) / float(kc_cells),
            "on_card": round(ox * oy / window, 4),
            "dwell_steps": int(dwell["steps"]), "blind_steps": int(dwell["blind"]),
            "pairs": int(dwell["pairs"]),
            # the room this card was judged against: the other cards the visit
            # had looked at, each with what its own last look gave and what it
            # smells of, so a replay can say what the comparison was made of
            "reference": reference,
            "reference_leaning": (float(np.mean([r["leaning"] for r in reference]))
                                  if reference else None),
            "calibration": str(getattr(self.mb, "calibration", calibration.CHOSEN)),
            "sides_sha": str(getattr(self.mb, "sides_sha", "")),
            "board_item": dict(card),
            "crop": [x0, y0, x1 - x0, y1 - y0], "frame": [h, w],
            "mode": self.mode,
        }
        self.looks_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write(self.looks_dir / f"{look_id}.json",
                      json.dumps(record, indent=1).encode("utf-8"))
        self._write_png(self.looks_dir / f"{look_id}.png", img[y0:y1, x0:x1])
        self._prune_looks()
        return look_id

    @staticmethod
    def _write_png(path, crop):
        """
        The eye window, as 8-bit grey.

        The frame the eye sampled was already 8-bit (a JPEG screencast decoded
        to 'L' and divided by 255), so rounding back to bytes is lossless and a
        replay sees exactly what the fly saw.
        """
        try:
            from PIL import Image
            arr = np.rint(np.clip(np.asarray(crop, dtype=np.float32), 0.0, 1.0) * 255.0)
            Image.fromarray(arr.astype(np.uint8), mode="L").save(str(path))
            return True
        except Exception:
            return False

    def _read_look(self, look_id):
        if not look_id:
            return None
        try:
            return json.loads((self.looks_dir / f"{look_id}.json").read_text(encoding="utf-8"))
        except Exception:
            return None

    def _read_crop(self, look_id):
        try:
            from PIL import Image
            with Image.open(str(self.looks_dir / f"{look_id}.png")) as im:
                return np.asarray(im.convert("L"), dtype=np.float32) / 255.0
        except Exception:
            return None

    def protected_looks(self):
        """Look ids that may never be pruned: an open position's buys, and anything still in flight."""
        keep = set()
        open_tokens = {p.get("token") for p in (self.read_book().get("positions") or [])}
        pending = self._intent["body"]["token"] if self._intent else None
        for token, ids in self.refs.items():
            if token in open_tokens or token == pending:
                keep.update(ids)
        if self._intent:
            keep.add(self._intent["body"]["look_id"])
        return keep

    def _prune_looks(self):
        try:
            files = sorted(self.looks_dir.glob("*.json"))
        except Exception:
            return
        over = len(files) - int(self.max_looks)
        if over <= 0:
            return
        keep = self.protected_looks()
        for p in files:
            if over <= 0:
                break
            if p.stem in keep:
                continue
            for q in (p, p.with_suffix(".png")):
                try:
                    q.unlink()
                except OSError:
                    pass
            over -= 1

    def _drop_ref(self, token, look_id):
        ids = self.refs.get(token)
        if not ids or not look_id:
            return
        left = [i for i in ids if i != look_id]
        if left:
            self.refs[token] = left
        else:
            self.refs.pop(token, None)

    # -- learning ---------------------------------------------------------
    def _learn_from_mark(self):
        """
        Looking at a coin it holds, while that coin is worth more or less than
        it was the last time the fly looked.

        The first step of a dwell asks the executor what the position would
        return if it were sold now. A later step in the same dwell, once that
        answer is back, pairs the smell and sight of the card - already
        observed this step - with reward or punishment dopamine. One delivery
        per dwell; if the fly walks off before the answer arrives, nothing is
        delivered and the reference is left alone.

        Two things stop a lesson that would be about nothing. A mark quoted
        across a settlement measures a position that has since changed, and is
        dropped. And the very first mark on a position replaces the reference
        rather than being judged against it, because until then the reference
        is what the coin cost, which a sale price can never match.
        """
        d = self._dwell
        token = d["token"]
        mark = d.get("mark")
        if mark is None:
            # The reference is captured here, with the request, and not read
            # when the answer lands: the executor needs seconds to quote the
            # chain, and a settlement applied in between moves mark_ref under
            # it. book_rev is how the room notices that happened.
            d["mark"] = {"done": False, "delivered": False, "value": None,
                         "ref": self.mark_ref.get(token), "rev": self.book_rev.get(token, 0),
                         "from_cost": token in self.from_cost}
            self.spawn(self._mark_worker, token, d["mark"])
            return
        if mark["delivered"] or not mark["done"]:
            return
        mark["delivered"] = True
        value = mark.get("value")
        if value is None:
            return                                   # unquotable: nothing measured, nothing learned
        if self.book_rev.get(token, 0) != mark.get("rev"):
            # a buy or a sell of this coin settled while it was being quoted,
            # so the number in hand measures a position that no longer exists
            self._say(f"a mark on {token[:10]} crossed a settlement; nothing learned from it")
            return
        ref = mark.get("ref")
        if not ref or float(ref) <= 0.0 or mark.get("from_cost"):
            # A fresh position's reference is what it cost; a mark is what
            # selling it would return. The venue's fee and the creator tax come
            # off on the way in and again on the way out, so the first look at a
            # coin the fly has just bought is below its cost whatever the coin
            # did - a punishment for having bought at all. The first mark sets
            # the reference and teaches nothing. CHOSEN.
            self.mark_ref[token] = float(value)
            self.from_cost.discard(token)
            self._save_room()
            return
        q = float(value) / float(ref)
        self._deliver(token, "look", q, None)
        self.mark_ref[token] = float(value)
        self._save_room()

    def _mark_worker(self, token, slot):
        try:
            _, obj = self.http.post_json(self.executor_url + "/marks", {"tokens": [token]},
                                         headers={"X-Fly-Intent": self.intent_token},
                                         timeout=CHAIN_TIMEOUT_S)
            mark = ((obj or {}).get("marks") or {}).get(token) or {}
            value = mark.get("value_eth")
            slot["value"] = None if value is None else float(value)
            slot["venue"] = mark.get("venue")
            slot["block"] = mark.get("block")
        except Exception:
            slot["value"] = None
        finally:
            slot["done"] = True

    def _deliver(self, token, kind, q, look_id):
        """
        Reward or punishment at the Kenyon cell synapse, and one line in the log.

        A delivery moves weights, and every reading the visit is holding was
        taken under the old ones. syn_drive is base * gain summed over the
        synapses whose Kenyon cell fired, so a stored reading is priced at the
        gains of the moment it was taken; and the shift a lesson makes is
        almost entirely common to every card (build/backroom_screen.json: 12
        pairings moved the trained coin's leaning -0.0254 and an untouched
        coin's -0.0267). Comparing a card read after the lesson against cards
        read before it hands that whole common shift to the drive - the one
        thing the relative rule exists to cancel. So the room behind the fly is
        emptied here, and the dwell in progress drops what it has accumulated:
        the visit has to walk two cards again before it can commit, which is
        the same price enter() already pays.
        """
        sign = sign_for(q)
        amount = amount_for(q)
        if sign == 0 or amount <= 0.0:
            return 0
        hit = int(self.mb.dopamine(sign, amount))
        self.mb.apply()
        self.mb.save()
        self._forget_room()
        self.counters["sugar" if sign > 0 else "shock"] += 1
        m = self.meta.get(token) or {}
        record = {"at": self._now(), "token": token, "symbol": m.get("symbol") or "",
                  "kind": kind, "q": float(q), "amount": float(amount), "sign": int(sign),
                  "synapses_hit": hit, "look_id": look_id, "mode": self.mode}
        try:
            self.dopamine_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.dopamine_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + chr(10))
        except Exception:
            pass
        self.last_dopamine.append(record)
        self.last_dopamine = self.last_dopamine[-10:]
        return hit

    def _forget_room(self):
        """
        Drop every reading taken under weights that have since moved.

        Called from _deliver, which is the only place in this build where a
        weight changes. The dwell in progress keeps its identity, its held flag
        and its mark - none of those are readings - and starts its evidence
        again.
        """
        self._seen = {}
        d = self._dwell
        if d is not None:
            d.update({"steps": 0, "blind": 0, "A": 0.0, "V": 0.0, "kc": 0, "lean_sum": 0.0,
                      "pairs": 0, "drive_sum": 0.0, "reference": []})

    # -- what the executor settled ---------------------------------------
    def poll_events(self):
        """
        Apply anything the executor has settled since last time.

        Called by roam between steps, because applying an event can run the
        brain: a settled sell is paired with the coin all over again. The fetch
        itself waits on a worker; the applying happens here.
        """
        pending, self._events_result = self._events_result, None
        if pending is not None:
            self._apply_events(pending)
        now = self._now()
        if not self._events_busy and now - self._events_at >= EVENTS_EVERY_S:
            self._events_busy = True
            self._events_at = now
            self.spawn(self._events_worker, int(self.last_seq))

    def _events_worker(self, after):
        try:
            _, obj = self.http.get_json(f"{self.executor_url}/events?after={int(after)}",
                                        headers={"X-Fly-Intent": self.intent_token})
            if isinstance(obj, dict):
                self._events_result = obj
        except Exception:
            pass
        finally:
            self._events_busy = False

    def _apply_events(self, obj):
        try:
            last = None if obj.get("last") is None else int(obj["last"])
        except (TypeError, ValueError):
            last = None
        if obj.get("error"):
            # the executor says its own ledger did not replay, so the history
            # it can serve is a prefix of the real one; a prefix is not a history
            self._say(f"the executor will not vouch for its events: {str(obj['error'])[:90]}")
            return
        if not self.seq_known:
            # First contact, with no room.json to say where this room got to.
            # Everything the executor still holds happened before this room
            # existed, and applying it would run a brain pass and deliver
            # dopamine for every settled sell a second time - on a mushroom
            # body that is stored separately and already learned from them.
            # Take its place in the stream without applying any of it.
            self.seq_known = True
            self.last_seq = last or 0
            self._say(f"no place on disk: joining the executor's event stream at {self.last_seq}")
            self._save_room()
            return
        if last is not None and last < self.last_seq:
            # the executor's ledger was rotated or replaced and its numbering
            # restarted; following it down is the only way to see what it sends
            self._say(f"the executor is at event {last} and this room was at {self.last_seq}; "
                      "its ledger restarted, so the room follows it back")
            self.last_seq = last
        for ev in obj.get("events") or []:
            try:
                self._apply_event(ev)
            except Exception as exc:
                self._say(f"an event could not be applied: {str(exc)[:90]}")
        if last is not None:
            self.last_seq = max(int(self.last_seq), last)
        self._save_room()

    def _apply_event(self, ev):
        seq = ev.get("seq")
        if seq is not None:
            self.last_seq = max(int(self.last_seq), int(seq))
        token = ev.get("token")
        kind = str(ev.get("kind") or "")
        side = str(ev.get("side") or "")
        if kind == "refused":
            self.counters["refused"] += 1
            self._drop_ref(token, ev.get("look_id"))
            return
        if kind != "booked" or not token:
            return
        self.counters["booked"] += 1
        # every settlement on this coin moves what a mark means, so a mark that
        # was already in flight when this arrived can be recognised and dropped
        self.book_rev[token] = int(self.book_rev.get(token, 0)) + 1
        eth = float(ev.get("eth") or 0.0)
        gas = float(ev.get("gas_eth") or 0.0)
        if side == "buy":
            # what the coin cost. That is a cost and not a mark, so the first
            # look at this position only resets the reference (_learn_from_mark)
            self.mark_ref[token] = float(self.mark_ref.get(token, 0.0)) + eth + gas
            self.from_cost.add(token)
            return
        if side != "sell":
            return
        fraction = float(ev.get("fraction") or 0.0)
        ref = float(self.mark_ref.get(token, 0.0))
        if fraction > 0.0 and ref > 0.0:
            if token in self.from_cost:
                # The reference is still what the position cost, because the
                # fly never looked at this coin while it held it. A sale price
                # can never match a cost - the venue's fee and the creator tax
                # come off on the way in and again on the way out - so this
                # measures the round trip's fees, not what the coin did.
                # _learn_from_mark refuses to teach on exactly that comparison,
                # and the two paths have to agree: otherwise the same flat
                # round trip teaches shock 0.060 when the fly happened not to
                # look at the coin and 0.0105 when it did, which is gaze
                # history deciding the size of a lesson. CHOSEN, measured.
                self._say(f"a sell of {str(token)[:10]} settled against what it cost rather "
                          "than against a mark; nothing is taught from it")
            else:
                self._represent(ev, (eth - gas) / (ref * fraction))
        left = ref * max(0.0, 1.0 - fraction)
        if fraction >= 1.0 or left <= 0.0:
            self.mark_ref.pop(token, None)
            self.from_cost.discard(token)
            self.refs.pop(token, None)               # the position is closed; its looks may age out
        else:
            self.mark_ref[token] = left

    def _represent(self, ev, q):
        """
        A settled sell arrives while the fly is somewhere else, so the coin is
        shown to it again: that commit's own eye window, back in its place on
        an otherwise empty room, plus the coin's smell. Dopamine has to fall in
        the same moment as the smell or the synapse it should change is not
        eligible, so this runs, observes and delivers in one go.

        With no stored picture there is no lesson. A look record or its PNG can
        be missing - a failed write, a prune, an event carrying a look_id this
        room never wrote - and what is left to show the fly is then a uniform
        ground-grey frame, which this project measured as a stronger stimulus
        than a card (13,075 approach MBONs against 6,796) and one every card
        shares: the lesson would be carved into the whole board instead of into
        the coin. Losing it is better than that.

        Nothing the fly looked at before this moment may take the lesson, and
        nothing it looks at after may take it either, so eligibility is wiped on
        the way in and again on the way out. Without the second wipe the
        re-presented coin is still eligible at 55% on the next step, and the
        mark lesson of whatever card the fly is dwelling on lands on it too.
        """
        token = ev.get("token")
        look_id = ev.get("look_id")
        look = self._read_look(look_id)
        if look is None or self._read_crop(look_id) is None:
            self._say(f"a settled sell of {str(token)[:10]} has no stored picture "
                      f"({look_id or 'no look id'}); nothing is taught from it")
            return
        frame, cx, cy = self._look_scene(look)
        smell = (look or {}).get("smell") or (self.smell_of(token) if token else None)
        drive = dict(self.pilot.eye.look(frame, cx, cy))
        for k, v in (self.nose.drive(smell) if smell else {}).items():
            drive[k] = v
        seed = int(self._now() * 1000.0) % (1 << 30)
        # Nothing the fly looked at before this moment may take the lesson. The
        # trace decays but never clears, so the six steps before a settlement
        # landed - whatever web page the fly happened to be on while the
        # executor spent fifteen seconds quoting - were eligible for it, and the
        # step just before took 55% of it. The coin is shown again precisely so
        # the lesson is the coin's.
        self.mb.forget_trace()
        r = self.fb.run(drive, steps=self.pilot.sim_steps, gains=self.gains,
                        record=self.readout, seed=seed)
        self.mb.observe(r.get("_fired"))
        self._deliver(token, "sell", q, look_id)
        # and out again: this coin's cells stop being eligible the moment its
        # own lesson is spent, so the card the fly is dwelling on cannot charge
        # its next lesson to them
        self.mb.forget_trace()

    def _look_scene(self, look):
        """An empty room with one look's eye window put back where it was."""
        h, w = (look or {}).get("frame") or [800, 1280]
        frame = np.full((int(h), int(w)), GROUND, dtype=np.float32)
        cx, cy = (look or {}).get("cursor") or [w / 2.0, h / 2.0]
        crop = self._read_crop((look or {}).get("look_id")) if look else None
        if crop is not None:
            x0, y0 = int(look["crop"][0]), int(look["crop"][1])
            frame[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = crop
        return frame, float(cx), float(cy)

    # -- reporting --------------------------------------------------------
    def state(self):
        d = self._dwell
        now = None
        if d is not None:
            m = self.meta.get(d["token"]) or {}
            now = {"token": d["token"], "symbol": m.get("symbol") or "",
                   "name": m.get("name") or "", "held": bool(d.get("held")),
                   "drive": self._drive_of(d), "dwell_steps": int(d["steps"])}
        c = self.counters
        # looks counts every step that read a card, commits the stops that
        # became decisions. They were the same number by construction until
        # 2026-09-12, which published nine looks for a fly that had looked at a
        # card on most of 1,592 room steps.
        return {"in_room": bool(self.in_room), "visits": c["visits"], "looks": c["looks"],
                "commits": c["commits"], "intents": c["intents"], "booked": c["booked"],
                "refused": c["refused"], "dislikes": c["dislikes"], "busy": c["busy"],
                # how many cards this visit has looked at: with fewer than two
                # there is nothing to judge a card against and a stop commits
                # nothing, which is worth being able to see from outside
                "seen": len(self._seen),
                "now": now, "last_intents": self.last_intents[-10:],
                "learning": {"sugar": c["sugar"], "shock": c["shock"],
                             "last": self.last_dopamine[-10:]},
                "board_size": len(self._board["cards"]),
                "board_error": self._board_error,
                "board_updated": int(self._board["updated"])}

    # -- persistence ------------------------------------------------------
    def _now(self):
        return float(self.clock())

    def _save_room(self):
        with self._lock:
            try:
                _atomic_write(self.room_file, json.dumps({
                    "mark_ref": self.mark_ref, "from_cost": sorted(self.from_cost),
                    "book_rev": self.book_rev, "meta": self.meta, "refs": self.refs,
                    "counters": self.counters, "last_seq": int(self.last_seq),
                    "seq_known": bool(self.seq_known),
                    "last_intents": self.last_intents, "last_dopamine": self.last_dopamine,
                    "look_seq": int(self._look_seq), "cards": self._board["cards"],
                    "updated": int(self._now()),
                }, indent=1).encode("utf-8"))
            except Exception as exc:
                # losing this file loses the room's place in the event stream,
                # which is worth saying out loud rather than swallowing
                self._say(f"could not write {self.room_file}: {str(exc)[:90]}")

    def _load_room(self):
        try:
            d = json.loads(self.room_file.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return                                   # a new room; poll_events finds its place
        except Exception as exc:
            self._say(f"could not read {self.room_file}: {str(exc)[:90]} - "
                      "starting with no memory of earlier visits")
            return
        self.mark_ref = {str(k): float(v) for k, v in (d.get("mark_ref") or {}).items()}
        self.from_cost = {str(t) for t in (d.get("from_cost") or [])}
        self.book_rev = {str(k): int(v) for k, v in (d.get("book_rev") or {}).items()}
        self.meta = dict(d.get("meta") or {})
        self.refs = {str(k): list(v) for k, v in (d.get("refs") or {}).items()}
        self.counters.update({k: int(v) for k, v in (d.get("counters") or {}).items()
                              if k in self.counters})
        self.last_seq = int(d.get("last_seq") or 0)
        self.seq_known = True                        # this room has a place in the stream
        self.last_intents = list(d.get("last_intents") or [])[-10:]
        self.last_dopamine = list(d.get("last_dopamine") or [])[-10:]
        self._look_seq = int(d.get("look_seq") or 0)
        # the board is kept for the page to have something to draw, but it is
        # stale by definition, so the next visit refetches it
        self._board = {"cards": list(d.get("cards") or []), "updated": 0.0}


# --------------------------------------------------------------------------
# replay: recompute one stored decision
# --------------------------------------------------------------------------
def replay(look_id, state_dir=None, brain=None):
    """
    Re-run one look on the real brain and print what it gives now.

    The record holds the seed, the cursor, the smell and the eye window, and
    the eye window is the whole of what the retina sampled, so this reproduces
    that run exactly - unless the mushroom body has learned since, which is the
    interesting case. It recomputes A, V and the leaning the way the room read
    them, off the synapses whose Kenyon cells fired.

    The reference is not re-run. The cards this look was judged against were
    read at their own rectangles, at their own moments in that visit, and the
    record keeps the leaning each of them gave; running their smells here at
    this cursor would be a different measurement wearing the same name. So the
    drive below is this look's new leaning against those stored ones.
    """
    state_dir = Path(state_dir or os.environ.get("FLY_STATE_DIR") or (ROOT / "build"))
    path = state_dir / "backroom" / "looks" / f"{look_id}.json"
    look = json.loads(path.read_text(encoding="utf-8"))

    from flysim import FlyBrain
    from mushroom import MushroomBody
    from olfaction import Nose
    try:
        from pumpui import FlyPilot
    except ImportError:                          # the public copy calls it flyeye
        from flyeye import FlyPilot

    fb = brain or FlyBrain()
    gains = calibration.gains_for(fb, look.get("calibration") or calibration.CHOSEN)
    mb = MushroomBody(fb, calibration=look.get("calibration") or calibration.CHOSEN)
    pilot = FlyPilot(fb, sim_steps=60)
    nose = Nose(fb)
    nose.max_hz = calibration.SETTINGS[calibration.CHOSEN]["odour_max_hz"]
    readout = calibration.readout(mb)

    h, w = look["frame"]
    # the page's own ground grey, a constant of the room and not of the look
    frame = np.full((int(h), int(w)), GROUND, dtype=np.float32)
    try:
        from PIL import Image
        with Image.open(str(path.with_suffix(".png"))) as im:
            crop = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
        x0, y0 = int(look["crop"][0]), int(look["crop"][1])
        frame[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = crop
        saw = "the stored eye window"
    except Exception:
        saw = "an empty room (no stored picture)"

    cx, cy = look["cursor"]
    seed = int(look["seed"])
    smell = look.get("smell")
    _, _, click, hz, info = pilot.step(frame, cx, cy, gains=gains, seed=seed, detail=True,
                                       extra_drive=(nose.drive(smell) if smell else None),
                                       extra_record=readout)
    own = calibration.syn_drive(mb, info.get("fired"))
    # what the other cards leaned when the fly looked at them, as recorded
    refs = [{"token": r.get("token"), "leaning": float(r.get("leaning") or 0.0)}
            for r in look.get("reference") or []]
    steps = max(1, int(look.get("dwell_steps") or 1))
    drive_now = None
    if refs:
        drive_now = calibration.relative_leaning(calibration.leaning(own),
                                                 [r["leaning"] for r in refs])
    out = {"look_id": look_id, "token": look["token"], "symbol": look.get("symbol"),
           "saw": saw, "dwell_steps": steps, "seed": seed,
           # stored A/V are the dwell's sums and its mean leaning; the ones
           # below are this one look
           "stored": {"A": look["A"], "V": look["V"], "leaning": look.get("leaning"),
                      "leaning_of_sums": look.get("leaning_of_sums"),
                      "kc_frac": look.get("kc_frac"), "on_card": look.get("on_card"),
                      "blind_steps": look.get("blind_steps"),
                      "reference": [r.get("token") for r in look.get("reference") or []],
                      "reference_leaning": look.get("reference_leaning"),
                      "drive": look["drive"]},
           "now": {"A": own[0], "V": own[1], "leaning": calibration.leaning(own),
                   "kc_frac": (float(np.isin(np.asarray(mb.kc), np.asarray(info.get("fired"))).mean())
                               if info.get("fired") is not None and len(mb.kc) else None),
                   "reference_leaning": (float(np.mean([r["leaning"] for r in refs]))
                                         if refs else None),
                   "drive": drive_now},
           "click_now": bool(click), "stop_hz": round(float(hz.get("stop", 0.0)), 1),
           "calibration": look.get("calibration"), "sides_sha_then": look.get("sides_sha"),
           "sides_sha_now": mb.sides_sha, "mode": look.get("mode")}
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "replay":
        replay(sys.argv[2], state_dir=(sys.argv[3] if len(sys.argv) > 3 else None))
    else:
        print(__doc__.strip().splitlines()[-1].strip())
