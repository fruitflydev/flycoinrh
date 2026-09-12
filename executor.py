"""
The paper executor: the one process that turns the fly's commits into trades,
and in this build it does not trade at all.

It listens on 127.0.0.1 only, and the backroom hands it one thing: an intent,
{token, side, drive, seen_at, look_id}, with a secret header this boot made up.
Any other field in the body is a 400. There is no command, route, file or
environment variable here that places, sizes, cancels or reverses a trade, and
none that a person can reach from outside the machine.

WHAT IT DOES WITH AN INTENT
Re-reads the coin's market from the chain - the factory record gives the phase,
the phase gives the venue - quotes the exact trade at a pinned block, estimates
the gas, and books the fill against a make-believe balance. The quote is real
and live; the balance is not. FLY_BACKROOM_LIVE=1 does not turn that into real
money: it stops the process, because the live path is not built here.

SIZE COMES FROM THE BRAIN AND THE BALANCES, NOTHING ELSE
A buy spends floor(drive x (balance - gas reserve)). A sell sells
floor(|drive| x the position). drive = 1 spends every free wei and drive = -1
sells the lot. There is no minimum, maximum, cap, stop, target, cooldown,
allowlist or price test anywhere in this file, and adding one would make the
fly's output stop being the fly's.

WHAT IS MEASURED AND WHAT IS CHOSEN
MEASURED: the factory record and phase, the curve reserves, the quotes (pons
reproduces the launchpad's own arithmetic on a block-pinned state), the ERC-20
decimals, and eth_gasPrice.
CHOSEN by people: the paper starting balance, the gas unit counts in GAS_UNITS
(estimates, until a live trade measures them), the gas reserve formula, the
20-second intent age, the 300-bps slippage figure (recorded only - a paper fill
happens at the quote), and the refusal reasons themselves.

  FLY_STATE_DIR      where backroom/ledger.paper.jsonl lives
  FLY_EXECUTOR_PORT  loopback port, default 4671
  FLY_INTENT_TOKEN   the per-boot secret; required
  FLY_BACKROOM_PAPER_ETH  the paper balance to start from, default 1.0
  FLY_INTENT_MAX_AGE_S    default 20, held to 5-60 s (intent_max_age)
  FLY_SLIPPAGE_BPS        default 300, recorded only

None of these places, sizes, cancels or reverses a trade. The paper balance is
how much pretend money exists, not a limit on any order; the intent age is
clamped precisely so that it cannot become a way to refuse everything the fly
decides. Both are printed at boot.
"""
import hmac
import json
import math
import os
import re
import sys
import threading
import time
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pons
import tradebook

ROOT = Path(__file__).parent
HOST = "127.0.0.1"                       # the only interface, in every mode
MODE = "paper"
FIELDS = ("token", "side", "drive", "seen_at", "look_id")
ADDRESS = re.compile(r"^0x[0-9a-fA-F]{40}$")
MAX_BODY = 64 * 1024
MAX_MARKS = 64

# Gas units per trade. CHOSEN estimates, from what comparable calls cost on
# this chain; nothing here has been measured on a real trade, because this
# build never makes one. They matter twice: they come out of the paper balance,
# and they set the reserve that keeps a buy from leaving a position unsellable.
GAS_UNITS = {"curve": {"buy": 160_000, "sell": 140_000},
             "pool": {"buy": 220_000, "sell": 220_000}}
APPROVE_UNITS = {"erc20": 46_000, "permit2": 60_000}
# What one exit costs at worst: both approvals plus a pool sell.
EXIT_UNITS = APPROVE_UNITS["erc20"] + APPROVE_UNITS["permit2"] + GAS_UNITS["pool"]["sell"]
# A first sell pays for its approvals once per coin per venue: the curve pulls
# the coin itself, a pool pulls it through Permit2.
APPROVALS = {"curve": ("erc20",), "pool": ("erc20", "permit2")}

LIVE_REFUSAL = "live trading is not built; this build is paper only"


class _No(Exception):
    """A reason to refuse. Carries the string the fly and the ledger will see."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def _short(exc):
    return f"{type(exc).__name__}: {exc}"[:160]


def _floor(scale, whole):
    """floor(scale x whole) with the float's exact value, so no wei is invented."""
    return math.floor(Fraction(scale) * int(whole))


def address(value):
    """The checksummed address, or None if that string is not one."""
    if not isinstance(value, str) or not ADDRESS.match(value):
        return None
    try:
        return pons.to_checksum_address(value)
    except ValueError:
        return None                  # mixed case whose checksum does not add up


def check_body(body):
    """The intent body, exactly. Returns (fields, None) or (None, why not)."""
    if not isinstance(body, dict):
        return None, "the body must be an object"
    if set(body) != set(FIELDS):
        missing = sorted(set(FIELDS) - set(body))
        extra = sorted(set(body) - set(FIELDS))
        return None, f"the body must be exactly {list(FIELDS)} (missing {missing}, extra {extra})"
    token = address(body["token"])
    if token is None:
        return None, "token must be 0x and 40 hex characters"
    if body["side"] not in ("buy", "sell"):
        return None, "side must be 'buy' or 'sell'"
    for name in ("drive", "seen_at"):
        v = body[name]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None, f"{name} must be a number"
        if v != v or v in (float("inf"), float("-inf")):
            return None, f"{name} must be a real number"
    if not isinstance(body["look_id"], str):
        return None, "look_id must be a string"
    return {"token": token, "side": body["side"], "drive": float(body["drive"]),
            "seen_at": float(body["seen_at"]), "look_id": body["look_id"]}, None


AGE_MIN_S, AGE_MAX_S = 5.0, 60.0


def intent_max_age(value):
    """
    How old an intent may be before it is refused, held inside a band.

    An intent is refused as stale when the fly's look is older than this, which
    exists because a chain quote takes seconds and a look the fly has walked
    away from should not become a trade. Read from FLY_INTENT_MAX_AGE_S, and
    therefore from .env, it was also a way to cancel every trade the fly will
    ever make: FLY_INTENT_MAX_AGE_S=0 refuses every intent before the chain is
    touched, the ledger fills with refusals, and every page still says the fly
    is trading. Nothing outside the brain may place, size, cancel or reverse a
    trade, so the setting is clamped to a band in which it can only ever be
    what it is for - 5 s is shorter than the fastest quote seen and 60 s is
    longer than the slowest - and a value outside it is said out loud at boot.
    Anything unreadable is the default.
    """
    try:
        age = float(value)
    except (TypeError, ValueError):
        return 20.0
    if age != age:                          # not a number at all
        return 20.0
    return min(AGE_MAX_S, max(AGE_MIN_S, age))


class Executor:
    """
    The paper book behind the loopback API. One intent at a time: the lock is
    the whole of the ordering, and a second intent while one is in flight is
    told 'busy' rather than queued, so the fly never trades on a look it has
    already walked away from.
    """

    def __init__(self, chain, ledger, intent_token, max_age_s=20.0, slippage_bps=300,
                 clock=time.time):
        self.chain = chain
        self.ledger = ledger
        self.intent_token = intent_token or ""
        self.max_age_s = intent_max_age(max_age_s)
        self.slippage_bps = int(slippage_bps)
        self.clock = clock
        self.lock = threading.Lock()
        self._meta = {}

    # ---- read-only routes -----------------------------------------------
    def health(self):
        return {"ok": True, "mode": MODE}

    def book(self):
        """
        The published book, or an explicit failure.

        A ledger that did not replay has no book. Answering with the one
        rebuilt from the lines that did parse would report an untouched paper
        balance and no positions while the file on disk records trades - a
        confident wrong answer, which is worse than none. The intent route
        already fails closed; so do the two read-only ones.
        """
        if not self.ledger.ok:
            return 503, {"mode": MODE, "status": "error",
                         "error": self.ledger.error or "ledger unreadable"}
        return 200, self.ledger.public(at=self.clock())

    def events(self, after=0):
        if not self.ledger.ok:
            # a truncated history read as a whole one would let a room replay
            # a prefix of the ledger as if that were everything that happened
            return 503, {"events": [], "last": None, "status": "error",
                         "error": self.ledger.error or "ledger unreadable"}
        try:
            after = int(after)
        except (TypeError, ValueError):
            after = 0
        return 200, {"events": self.ledger.events_after(after), "last": self.ledger.book.seq}

    def marks(self, body):
        """
        What selling each whole paper position would return right now, net of
        whatever the venue takes, at one pinned block. It reads the chain and
        nothing else: no ledger entry, no book change, no trade.
        """
        if not isinstance(body, dict) or set(body) != {"tokens"}:
            return 400, {"status": "error", "reason": "the body must be exactly ['tokens']"}
        tokens = body["tokens"]
        if not isinstance(tokens, list) or len(tokens) > MAX_MARKS:
            return 400, {"status": "error", "reason": f"tokens must be a list of at most {MAX_MARKS}"}
        # The sizes are read under the lock an intent holds, and each mark says
        # which size it was quoted for. Quoting is nineteen sequential eth_calls
        # - fourteen to fifteen seconds against the public RPC - and a sell can
        # book inside that window, so a mark read off a book that was changing
        # would report the value of a position that no longer exists.
        if not self.lock.acquire(blocking=False):
            return 200, {"marks": {str(t)[:64]: self._mark(None, None, "busy") for t in tokens}}
        marks, held = {}, {}
        try:
            for raw in tokens:
                token = address(raw)
                if token is None:
                    marks[str(raw)[:64]] = self._mark(None, None, "not an address")
                    continue
                if not self.ledger.ok:
                    marks[raw] = self._mark(None, None, "ledger unreadable")
                    continue
                pos = self.ledger.book.held(token)
                if pos is None or pos["tokens_raw"] <= 0:
                    marks[raw] = self._mark(None, None, "not held")
                    continue
                held[raw] = (token, int(pos["tokens_raw"]))
        finally:
            self.lock.release()
        block = None
        for raw, (token, tokens_raw) in held.items():
            try:
                if block is None:
                    block = self._block()
                launch, venue = self._market(token, block)
                out, _used = self._quote(launch, venue, token, "sell", tokens_raw, block)
                marks[raw] = {"value_eth": out / tradebook.WEI, "venue": venue, "block": block,
                              "tokens_raw": str(tokens_raw), "reason": None}
            except _No as no:
                marks[raw] = self._mark(None, block, no.reason)
        return 200, {"marks": marks}

    @staticmethod
    def _mark(value, block, reason):
        return {"value_eth": value, "venue": None, "block": block, "tokens_raw": None,
                "reason": reason}

    # ---- the intent -----------------------------------------------------
    def intent(self, body):
        fields, why = check_body(body)
        if why:
            return 400, {"status": "error", "reason": why}
        if not self.lock.acquire(blocking=False):
            return 409, {"status": "busy"}
        try:
            return self._intent(fields)
        finally:
            self.lock.release()

    def _intent(self, f):
        led = self.ledger
        token, side, drive = f["token"], f["side"], f["drive"]
        ctx = {"token": token, "side": side, "drive": drive, "look_id": f["look_id"]}
        if not led.ok:
            # The book cannot be trusted, so nothing is booked and nothing is
            # written: the refusal itself would go into the file that is broken.
            ev = tradebook.loose_event("ledger unreadable", at=self.clock(), **ctx)
            return 200, {"status": "refused", "reason": ev["reason"], "event": ev}
        iid = led.intent(token, side, drive, f["seen_at"], f["look_id"])
        meta = {}

        def refuse(reason):
            ev = led.refused(iid, reason, **ctx, **meta)
            return 200, {"status": "refused", "reason": reason, "event": ev}

        if self.clock() - f["seen_at"] > self.max_age_s:
            return refuse("stale")
        if side == "buy" and not 0 < drive <= 1:
            return refuse("a buy needs drive in (0, 1]")
        if side == "sell" and not -1 <= drive < 0:
            return refuse("a sell needs drive in [-1, 0)")
        pos = led.book.held(token)
        if side == "sell" and (pos is None or pos["tokens_raw"] <= 0):
            return refuse("not held")
        try:
            block = self._block()
            launch, venue = self._market(token, block)
            symbol, name, decimals = self._token_meta(token, block)
            meta = {"symbol": symbol, "name": name, "venue": venue, "block": block}
            price = self._gas_price()

            if side == "buy":
                free = led.book.balance_wei - self._reserve(price)
                if free <= 0:
                    return refuse("nothing to spend")
                amount = _floor(drive, free)
                if amount <= 0:
                    return refuse("nothing to spend")
            else:
                amount = _floor(-drive, pos["tokens_raw"])
                if amount <= 0:
                    return refuse("nothing to sell")

            out, used = self._quote(launch, venue, token, side, amount, block)
        except _No as no:
            return refuse(no.reason)
        if out <= 0 or used <= 0:
            return refuse(f"the {venue} quotes nothing")

        approvals = [w for w in APPROVALS[venue]
                     if side == "sell" and not led.book.approved(token, venue, w)]
        units = GAS_UNITS[venue][side] + sum(APPROVE_UNITS[w] for w in approvals)
        gas_wei = units * price
        eth_wei, tokens_raw = (used, out) if side == "buy" else (out, used)
        # Recorded because a live build would need it; a paper fill takes the
        # quote itself, so this number never changes what is booked.
        min_out = out * (10_000 - self.slippage_bps) // 10_000

        led.quoted(iid, venue=venue, block=block, gas_price_wei=str(price), gas_units=units,
                   approvals=approvals, eth_wei=str(eth_wei), tokens_raw=str(tokens_raw),
                   gas_wei=str(gas_wei), min_out=str(min_out), slippage_bps=self.slippage_bps,
                   quoted_out=str(out))

        if side == "buy" and eth_wei + gas_wei > led.book.balance_wei:
            return refuse("nothing to spend")
        if side == "sell" and gas_wei > led.book.balance_wei + eth_wei:
            return refuse("not enough ETH left for the gas")

        ev = led.booked(iid, token=token, side=side, venue=venue, block=block, eth_wei=eth_wei,
                        tokens_raw=tokens_raw, gas_wei=gas_wei, symbol=symbol, name=name,
                        decimals=decimals, drive=drive, look_id=f["look_id"], approvals=approvals)
        return 200, {"status": "booked", "event": ev}

    # ---- the chain ------------------------------------------------------
    def _block(self):
        try:
            return self.chain.block_number()
        except Exception as exc:                       # noqa: BLE001
            raise _No(f"unquotable: {_short(exc)}")

    def _gas_price(self):
        try:
            return pons.gas_price(self.chain)
        except Exception as exc:                       # noqa: BLE001
            raise _No(f"unquotable: {_short(exc)}")

    def _market(self, token, block):
        """
        The venue, re-read every time. A coin that graduated while the fly held
        it is sold through its pool, because nothing here remembers a venue.
        """
        try:
            launch = pons.launched(self.chain, token, block=block)
        except Exception as exc:                       # noqa: BLE001
            raise _No(f"unquotable: {_short(exc)}")
        if not launch.exists:
            raise _No("the pons v2 factory has no record of this token")
        if not launch.native_quote:
            raise _No("not a native-quote coin")
        if launch.phase == pons.Phase.SETTLING:
            raise _No("settling into its pool")
        if launch.phase == pons.Phase.CLOSED:
            raise _No("closed")
        return launch, "curve" if launch.phase == pons.Phase.CURVE else "pool"

    def _quote(self, launch, venue, token, side, amount, block):
        """
        (out, used): tokens out and wei spent for a buy, wei out and tokens in
        for a sell. A curve buy past the sellable supply spends less than it
        was given, which is why 'used' is answered rather than assumed.
        """
        try:
            if venue == "curve":
                state = pons.curve_state(self.chain, launch.curve, block=block)
                if side == "buy":
                    q = pons.quote_curve_buy(state, amount)
                    return q["tokens_out"], q["spent"]
                return pons.quote_curve_sell(state, amount)["quote_out"], amount
            key = pons.pool_key(launch)
            out, _gas = pons.quote_pool_side(self.chain, key, token, side, amount, block=block)
            return out, amount
        except _No:
            raise
        except Exception as exc:                       # noqa: BLE001
            raise _No(f"unquotable: {_short(exc)}")

    def _token_meta(self, token, block):
        if token not in self._meta:
            try:
                decimals = pons.erc20_decimals(self.chain, token, block=block)
            except Exception as exc:                   # noqa: BLE001
                raise _No(f"unquotable: {_short(exc)}")
            self._meta[token] = (pons.erc20_text(self.chain, token, "symbol", block=block),
                                 pons.erc20_text(self.chain, token, "name", block=block),
                                 decimals)
        return self._meta[token]

    def _reserve(self, price):
        """
        Enough gas kept back to get out of everything held plus the position
        this buy would open, at twice the current price. A buy never spends
        into it; a sell may.
        """
        return (len(self.ledger.book.positions) + 1) * EXIT_UNITS * 2 * int(price)


# --------------------------------------------------------------------------
# the loopback server
# --------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "flybrain-executor"

    def log_message(self, fmt, *args):
        """Quiet on purpose: a request line would carry the fly's every look."""

    @property
    def executor(self):
        return self.server.executor

    def _authed(self):
        given = self.headers.get("X-Fly-Intent") or ""
        want = self.executor.intent_token
        return bool(want) and hmac.compare_digest(given, want)

    def _reply(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return None, "Content-Length is not a number"
        if length <= 0:
            return None, "no body"
        if length > MAX_BODY:
            return None, "body too large"
        try:
            return json.loads(self.rfile.read(length).decode("utf-8")), None
        except (ValueError, UnicodeDecodeError) as exc:
            return None, f"the body is not JSON: {exc}"

    def _failed(self, exc):
        """
        Anything that got past the routes, answered rather than dropped.

        socketserver's default is to log a traceback and close the connection,
        which the room reads as "unreachable" - and "unreachable" is
        deliberately treated as proof of nothing, so the look stays pinned for
        the life of the install and the fly waits for a settlement that will
        never be written. A 500 is an answer: it says the executor broke, which
        is true, and the room can act on it.
        """
        try:
            return self._reply(500, {"status": "error",
                                     "reason": f"{type(exc).__name__}: {str(exc)[:120]}"})
        except Exception:
            return None

    def do_GET(self):                                  # noqa: N802 - http.server's name
        try:
            return self._get()
        except Exception as exc:
            return self._failed(exc)

    def do_POST(self):                                 # noqa: N802
        try:
            return self._post()
        except Exception as exc:
            return self._failed(exc)

    def _get(self):
        route = urlparse(self.path)
        if route.path == "/health":
            return self._reply(200, self.executor.health())
        if not self._authed():
            return self._reply(403, {"status": "forbidden"})
        if route.path == "/book":
            return self._reply(*self.executor.book())
        if route.path == "/events":
            after = (parse_qs(route.query).get("after") or ["0"])[0]
            return self._reply(*self.executor.events(after))
        return self._reply(404, {"status": "error", "reason": "no such route"})

    def _post(self):
        route = urlparse(self.path)
        if not self._authed():
            return self._reply(403, {"status": "forbidden"})
        if route.path not in ("/intent", "/marks"):
            return self._reply(404, {"status": "error", "reason": "no such route"})
        body, why = self._body()
        if why:
            return self._reply(400, {"status": "error", "reason": why})
        if route.path == "/intent":
            code, payload = self.executor.intent(body)
        else:
            code, payload = self.executor.marks(body)
        return self._reply(code, payload)


def make_server(executor, port):
    """Loopback only. Nothing outside this machine can reach the fly's book."""
    server = ThreadingHTTPServer((HOST, int(port)), Handler)
    server.daemon_threads = True
    server.executor = executor
    return server


def setting(name, default=None):
    """
    One of this process's settings, resolved the way roam and the room resolve
    theirs.

    roam reads everything through launch.load_env, which reads .env and then
    lays the process's own FLY_* variables over it. Reading os.environ alone
    here meant a FLY_STATE_DIR written only into .env put the executor's ledger
    in one directory and the room's reader in another, with no error anywhere:
    nothing would ever look held, which is indistinguishable from a fly that
    never chose to sell. The same split would have set the two halves talking
    to different ports. run_all.py hands every child the resolved values; this
    is the same resolution for a run started by hand. load_env is imported
    softly, because requirements-executor.txt does not carry what launch.py
    imports, and it honours the supervisor's deny list, so reading the file
    here cannot hand this process a credential it was started without.
    """
    try:
        try:
            from launch import load_env
        except ImportError:                # the public copy calls it envcfg
            from envcfg import load_env
        value = load_env().get(name)
    except Exception:
        value = os.environ.get(name)
    return default if value in (None, "") else value


def state_dir():
    """Where the ledger and the published book live."""
    return setting("FLY_STATE_DIR", str(ROOT / "build"))


def main():
    if setting("FLY_BACKROOM_LIVE") == "1":
        print(LIVE_REFUSAL, file=sys.stderr)
        return 2
    # the one setting that is never read from a file: it is minted per boot by
    # the supervisor and handed to exactly two processes
    token = os.environ.get("FLY_INTENT_TOKEN")
    if not token:
        print("FLY_INTENT_TOKEN is not set; the executor will not answer without one",
              file=sys.stderr)
        return 2
    where = tradebook.paths(state_dir())
    start_wei = int(Fraction(setting("FLY_BACKROOM_PAPER_ETH", "1.0")) * tradebook.WEI)
    ledger = tradebook.Ledger(where["ledger"], book_path=where["book"], public_path=where["public"],
                              start_wei=start_wei)
    asked = setting("FLY_INTENT_MAX_AGE_S", "20")
    executor = Executor(pons.Chain(), ledger, token,
                        max_age_s=asked,
                        slippage_bps=int(setting("FLY_SLIPPAGE_BPS", "300")))
    port = int(setting("FLY_EXECUTOR_PORT", "4671"))
    server = make_server(executor, port)
    print(f"executor on http://{HOST}:{port} in {MODE} mode, ledger {where['ledger']}"
          + ("" if ledger.ok else f" - REFUSING EVERYTHING: {ledger.error}"), flush=True)
    # the settings that exist, named at boot, because a setting nobody can see
    # is a setting nobody can check against what the site says
    print(f"paper balance {start_wei / tradebook.WEI} ETH, intents stale after "
          f"{executor.max_age_s:g} s"
          + ("" if str(asked) == f"{executor.max_age_s:g}"
             else f" (FLY_INTENT_MAX_AGE_S={asked} is outside "
                  f"{AGE_MIN_S:g}-{AGE_MAX_S:g} s and was held to the band)"),
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
