"""
The fly's paper trade book: a write-ahead ledger, and the book derived from it.

One rule: nothing is booked before it is written down. Every intent is appended
to a JSONL file and flushed to disk before the outcome that follows it, so a
power cut can leave an intent with no outcome - recovery refuses it as
interrupted - but can never leave a fill nobody asked for.

This build is paper. There is no key here and nothing leaves the machine: the
amounts come from live on-chain quotes (pons.py) and are booked against a
make-believe balance. Every file says paper in its name and in its contents,
and the live ledger name is refused outright rather than half-handled.

The book is never the source of truth; the ledger is. Delete book.paper.json
and public.json and they come back identical on the next boot, because they are
a replay of the ledger and nothing else.

WHAT IS MEASURED AND WHAT IS CHOSEN
Nothing in this module is measured. It does arithmetic on numbers other modules
measure: quotes from the coin's own contracts, the gas price from the chain.
People chose the paper starting balance, the convention that a buy's estimated
gas joins the cost of the position while a sell's comes out of the proceeds,
and the decision to keep only the newest 50 trades in the published file.

Amounts are kept as integers - wei, and raw token units - and written as
decimal strings, because a float cannot hold 18 decimals without drifting. The
floats in an event are for display only.
"""
import json
import os
import time
from pathlib import Path

WEI = 10 ** 18
PAPER_LEDGER = "ledger.paper.jsonl"
LIVE_LEDGER = "ledger.live.jsonl"
TRADES_KEPT = 50            # how many booked trades the published file carries
EVENTS_KEPT = 5000          # how many events stay in memory for GET /events


class LedgerError(Exception):
    """The ledger on disk does not describe a book that could exist."""


def paths(state_dir):
    """Where the backroom keeps its files. The executor owns all three."""
    room = Path(state_dir) / "backroom"
    return {"room": room, "ledger": room / PAPER_LEDGER, "book": room / "book.paper.json",
            "public": room / "public" / "public.json"}


def atomic_write(path, data: bytes):
    """Write then rename, so a reader never sees half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _int(v, what):
    try:
        return int(str(v))
    except (TypeError, ValueError):
        raise LedgerError(f"{what} is not a whole number: {v!r}")


def loose_event(reason, token="", side="", drive=0.0, look_id="", at=None, symbol="", name=""):
    """
    A refusal that could not be written down, because the ledger itself is the
    thing that is broken. It carries seq 0 so nobody mistakes it for a record.
    """
    return {"seq": 0, "at": at if at is not None else time.time(), "kind": "refused", "side": side,
            "token": token, "symbol": symbol, "name": name, "drive": drive, "look_id": look_id,
            "reason": reason, "venue": None, "block": None, "eth": 0.0, "tokens": 0.0,
            "gas_eth": 0.0, "fraction": None, "realised_eth": None, "cost_basis_eth": 0.0}


class Book:
    """
    The state a ledger replays into: one balance, the open positions, the
    counters, and the events in the order they happened. apply() is the only
    thing that changes it, so the incremental book and a book rebuilt from the
    file on disk cannot drift apart.
    """

    def __init__(self, mode="paper"):
        self.mode = mode
        self.start_wei = 0
        self.balance_wei = 0
        self.positions = {}          # token -> dict with raw amounts
        self.counts = {"buys": 0, "sells": 0, "refused": 0}
        self.approvals = set()       # (token, venue, which) already paid for once
        self.open_intents = {}       # id -> the intent entry still without an outcome
        self.intents = 0
        self.events = []
        self.seq = 0
        self.opened = False

    # ---- replay ---------------------------------------------------------
    def apply(self, e):
        if not isinstance(e, dict):
            raise LedgerError("a ledger line is not an object")
        kind = e.get("kind")
        if kind == "open":
            return self._open(e)
        if not self.opened:
            raise LedgerError(f"{kind!r} before the ledger was opened")
        if kind == "intent":
            return self._intent(e)
        if kind == "quoted":
            return self._quoted(e)
        if kind == "booked":
            return self._terminal(e, self._booked)
        if kind == "refused":
            return self._terminal(e, self._refused)
        raise LedgerError(f"unknown ledger entry {kind!r}")

    def _open(self, e):
        if self.opened:
            raise LedgerError("the ledger is opened twice")
        if self.seq or self.intents:
            raise LedgerError("the ledger opens after it has already been used")
        if e.get("mode") != self.mode:
            raise LedgerError(f"this is a {e.get('mode')!r} ledger, not {self.mode!r}")
        self.start_wei = self.balance_wei = _int(e.get("start_wei"), "start_wei")
        if self.start_wei < 0:
            raise LedgerError("the paper balance starts below zero")
        self.opened = True
        return None

    def _intent(self, e):
        i = _int(e.get("id"), "intent id")
        if i != self.intents + 1:
            raise LedgerError(f"intent {i} is out of order")
        self.intents = i
        self.open_intents[i] = e
        return None

    def _quoted(self, e):
        i = _int(e.get("id"), "intent id")
        if i not in self.open_intents:
            raise LedgerError(f"a quote for intent {i}, which is not open")
        return None

    def _terminal(self, e, fn):
        i = _int(e.get("id"), "intent id")
        if i not in self.open_intents:
            raise LedgerError(f"{e.get('kind')} for intent {i}, which is not open")
        seq = _int(e.get("seq"), "seq")
        if seq != self.seq + 1:
            raise LedgerError(f"event {seq} is out of order")
        extra = fn(e)
        self.open_intents.pop(i)
        self.seq = seq
        ev = self._event(e, seq, extra)
        self.events.append(ev)
        if len(self.events) > EVENTS_KEPT:
            del self.events[:len(self.events) - EVENTS_KEPT]
        return ev

    def _booked(self, e):
        token = e.get("token")
        side = e.get("side")
        venue = e.get("venue")
        eth = _int(e.get("eth_wei"), "eth_wei")
        raw = _int(e.get("tokens_raw"), "tokens_raw")
        gas = _int(e.get("gas_wei"), "gas_wei")
        if eth < 0 or raw < 0 or gas < 0:
            raise LedgerError("a booked amount is below zero")
        if side == "buy":
            pos = self.positions.get(token)
            if pos is None:
                pos = {"token": token, "symbol": e.get("symbol", ""), "name": e.get("name", ""),
                       "decimals": int(e.get("decimals", 18)), "tokens_raw": 0, "cost_wei": 0,
                       "venue": venue, "opened_at": e.get("at")}
                self.positions[token] = pos
            self.balance_wei -= eth + gas
            pos["tokens_raw"] += raw
            pos["cost_wei"] += eth + gas
            pos["venue"] = venue
            self.counts["buys"] += 1
            extra = {"fraction": None, "realised_wei": None, "cost_basis_wei": pos["cost_wei"]}
        elif side == "sell":
            pos = self.positions.get(token)
            if pos is None:
                raise LedgerError(f"a sell of {token}, which the book does not hold")
            held = pos["tokens_raw"]
            if raw > held or held <= 0:
                raise LedgerError(f"a sell of {raw} raw units of {token}, which holds {held}")
            cost_before = pos["cost_wei"]
            cost_after = cost_before * (held - raw) // held
            self.balance_wei += eth - gas
            pos["tokens_raw"] = held - raw
            pos["cost_wei"] = cost_after
            pos["venue"] = venue
            self.counts["sells"] += 1
            extra = {"fraction": raw / held, "realised_wei": eth - gas - (cost_before - cost_after),
                     "cost_basis_wei": cost_after}
            if pos["tokens_raw"] == 0:
                self.positions.pop(token)
        else:
            raise LedgerError(f"a booked entry with side {side!r}")
        for which in e.get("approvals") or []:
            self.approvals.add((token, venue, which))
        if self.balance_wei < 0:
            raise LedgerError("the balance went below zero")
        return extra

    def _refused(self, e):
        if not e.get("reason"):
            raise LedgerError("a refusal with no reason")
        self.counts["refused"] += 1
        pos = self.positions.get(e.get("token"))
        return {"fraction": None, "realised_wei": None, "cost_basis_wei": pos["cost_wei"] if pos else 0}

    def _event(self, e, seq, extra):
        dec = int(e.get("decimals", 18))
        raw = _int(e.get("tokens_raw", 0), "tokens_raw")
        realised = extra.get("realised_wei")
        return {"seq": seq, "at": e.get("at"), "kind": e.get("kind"), "side": e.get("side"),
                "token": e.get("token"), "symbol": e.get("symbol", ""), "name": e.get("name", ""),
                "drive": e.get("drive"), "look_id": e.get("look_id"), "reason": e.get("reason"),
                "venue": e.get("venue"), "block": e.get("block"),
                "eth": _int(e.get("eth_wei", 0), "eth_wei") / WEI, "tokens": raw / (10 ** dec),
                "gas_eth": _int(e.get("gas_wei", 0), "gas_wei") / WEI,
                "fraction": extra.get("fraction"),
                "realised_eth": None if realised is None else realised / WEI,
                "cost_basis_eth": extra.get("cost_basis_wei", 0) / WEI}

    # ---- what the rest of the process reads -----------------------------
    def held(self, token):
        return self.positions.get(token)

    def approved(self, token, venue, which):
        return (token, venue, which) in self.approvals

    def public(self, address=None, at=None):
        """The published view: what a position is, what it cost, what was booked."""
        return {"mode": self.mode, "address": address, "start_eth": self.start_wei / WEI,
                "eth_balance": self.balance_wei / WEI,
                "positions": [{"token": p["token"], "symbol": p["symbol"], "name": p["name"],
                               "tokens": p["tokens_raw"] / (10 ** int(p["decimals"])),
                               "cost_eth": p["cost_wei"] / WEI, "venue": p["venue"],
                               "opened_at": p["opened_at"]} for p in self.positions.values()],
                "trades": [ev for ev in self.events if ev["kind"] == "booked"][-TRADES_KEPT:],
                "counts": dict(self.counts), "updated": at if at is not None else time.time()}

    def state(self):
        """The whole book, amounts intact, for book.paper.json and for tests."""
        return {"mode": self.mode, "start_wei": str(self.start_wei), "balance_wei": str(self.balance_wei),
                "positions": {t: {**p, "tokens_raw": str(p["tokens_raw"]), "cost_wei": str(p["cost_wei"])}
                              for t, p in self.positions.items()},
                "counts": dict(self.counts), "seq": self.seq, "intents": self.intents,
                "approvals": sorted(list(a) for a in self.approvals),
                "open_intents": sorted(self.open_intents)}


class Ledger:
    """
    The write-ahead file and the book it makes. Every method that appends
    flushes and fsyncs before it returns, and every outcome rewrites
    book.paper.json and public/public.json by atomic replace.

    If the file on disk cannot be replayed - a broken line, a fill for an
    intent that is not there, a balance below zero - the ledger comes up with
    ok False and stays that way. The executor then refuses every intent. A book
    that might be wrong is worse than no book.
    """

    def __init__(self, path, book_path=None, public_path=None, mode="paper", start_wei=WEI,
                 clock=time.time):
        self.path = Path(path)
        if mode != "paper":
            raise ValueError("this build is paper only")
        if self.path.name == LIVE_LEDGER:
            raise ValueError(f"{LIVE_LEDGER} belongs to a live build that does not exist here")
        room = self.path.parent
        self.book_path = Path(book_path) if book_path else room / "book.paper.json"
        self.public_path = Path(public_path) if public_path else room / "public" / "public.json"
        self.mode = mode
        self.start_wei = int(start_wei)
        self.clock = clock
        self.book = Book(mode)
        self.ok = True
        self.error = None
        self.publish_error = None       # why the derived files are stale, said once
        self.load()

    # ---- boot -----------------------------------------------------------
    def load(self):
        self.book = Book(self.mode)
        self.ok, self.error = True, None
        try:
            # errors="replace" on purpose: a corrupt byte has to become a line
            # that is not JSON, which the loop below turns into ok=False and a
            # book that refuses everything. Decoding strictly raised
            # UnicodeDecodeError - a ValueError, not an OSError - out of this
            # constructor instead, and the executor then died at startup with a
            # traceback and no /health, which is the one failure shape that
            # turns fail-closed into fail-to-start. Unclean shutdown is exactly
            # what a write-ahead ledger is built for.
            lines = (self.path.read_text(encoding="utf-8", errors="replace").splitlines()
                     if self.path.exists() else [])
        except OSError as exc:
            self.ok, self.error = False, f"ledger unreadable: {exc}"
            return
        try:
            for n, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except ValueError as exc:
                    raise LedgerError(f"line {n} is not JSON: {exc}")
                self.book.apply(entry)
        except LedgerError as exc:
            self.ok, self.error = False, str(exc)
            return
        if not self.book.opened:
            self._append({"kind": "open", "at": self.clock(), "mode": self.mode,
                          "start_wei": str(self.start_wei)}, apply=True)
        self.recover()
        if self.ok:
            self.write_book()

    def recover(self):
        """
        An intent with no outcome is an intent the process died inside. Nothing
        was booked, because the outcome is written after the intent and before
        the book changes, so it is refused and the fly is told.
        """
        for i in sorted(self.book.open_intents):
            e = self.book.open_intents[i]
            self.refused(i, "interrupted", token=e.get("token"), side=e.get("side"),
                         drive=e.get("drive"), look_id=e.get("look_id"), write=False)

    # ---- appending ------------------------------------------------------
    def _append(self, entry, apply=True):
        if not self.ok:
            raise LedgerError(self.error or "ledger unreadable")
        line = json.dumps(entry, separators=(",", ":")) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        if apply:
            try:
                return self.book.apply(entry)
            except LedgerError as exc:      # the file and the book disagree: stop trading
                self.ok, self.error = False, str(exc)
                raise
        return None

    def intent(self, token, side, drive, seen_at, look_id):
        """Write the ask down first. Returns the id every later entry carries."""
        i = self.book.intents + 1
        self._append({"kind": "intent", "at": self.clock(), "id": i, "token": token, "side": side,
                      "drive": drive, "seen_at": seen_at, "look_id": look_id})
        return i

    def quoted(self, intent_id, **fields):
        """What the chain answered, before anything is booked against it."""
        self._append({"kind": "quoted", "at": self.clock(), "id": intent_id, **fields})

    def booked(self, intent_id, token, side, venue, block, eth_wei, tokens_raw, gas_wei,
               symbol="", name="", decimals=18, drive=None, look_id="", approvals=(), **extra):
        entry = {"kind": "booked", "at": self.clock(), "seq": self.book.seq + 1, "id": intent_id,
                 "token": token, "symbol": symbol, "name": name, "decimals": int(decimals),
                 "side": side, "drive": drive, "look_id": look_id, "venue": venue, "block": block,
                 "eth_wei": str(int(eth_wei)), "tokens_raw": str(int(tokens_raw)),
                 "gas_wei": str(int(gas_wei)), "approvals": list(approvals), **extra}
        ev = self._append(entry)
        self.write_book()
        return ev

    def refused(self, intent_id, reason, token="", side="", drive=None, look_id="", symbol="",
                name="", venue=None, block=None, write=True):
        entry = {"kind": "refused", "at": self.clock(), "seq": self.book.seq + 1, "id": intent_id,
                 "token": token, "symbol": symbol, "name": name, "side": side, "drive": drive,
                 "look_id": look_id, "venue": venue, "block": block, "reason": reason}
        ev = self._append(entry)
        if write:
            self.write_book()
        return ev

    # ---- derived files --------------------------------------------------
    def public(self, at=None):
        return self.book.public(at=at)

    def write_book(self):
        """
        Republish the two derived files.

        Both are a replay of the ledger, which is already on disk and fsynced
        before this runs, so failing to write them loses nothing that is not
        rebuilt on the next boot. Failing loudly, on the other hand, loses the
        HTTP answer: os.replace over a file another process is reading raises
        PermissionError on Windows, and public.json is read about twice a room
        step. Unguarded, that exception left the executor's reply unsent, the
        room recorded "unreachable", and because no answer is deliberately
        treated as proof of nothing, that look stayed pinned for good.
        """
        if not self.ok:
            return                      # a book that might be wrong is not published
        at = self.clock()
        try:
            atomic_write(self.book_path, json.dumps(self.book.state(), indent=1).encode("utf-8"))
            atomic_write(self.public_path,
                         json.dumps(self.book.public(at=at), indent=1).encode("utf-8"))
            self.publish_error = None
        except OSError as exc:
            first, self.publish_error = self.publish_error is None, f"{type(exc).__name__}: {exc}"
            if first:
                print(f"[tradebook] the published book could not be written: "
                      f"{self.publish_error} - the ledger is intact and the files are "
                      f"rewritten on the next booking", flush=True)

    def events_after(self, after):
        return [ev for ev in self.book.events if ev["seq"] > int(after)]


def rebuild(path, mode="paper"):
    """
    Replay a ledger file into a book without touching anything on disk. Used by
    the tests, and by anyone who wants to check the published book by hand.
    """
    book = Book(mode)
    for n, line in enumerate(
            Path(path).read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except ValueError as exc:
            raise LedgerError(f"line {n} is not JSON: {exc}")
        book.apply(entry)
    return book
