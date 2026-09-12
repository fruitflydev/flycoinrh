"""
The paper executor against a fake chain: no network, no key, nothing on chain.

The fake answers the same JSON-RPC the real one does - the factory record, the
curve's thirteen reads, the pons quoter, decimals, symbol, name, gas price - so
the quotes here run through the real pons arithmetic and only the transport is
make-believe.

What these tests are really for: proving there is no cap. drive 1.0 has to
spend every free wei and drive -1.0 has to sell the whole position, because the
size is the fly's output and nothing here is allowed to overrule it.

  py -m pytest -q test_executor.py
"""
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from fractions import Fraction
from pathlib import Path
from unittest import mock

from eth_abi import decode, encode

import executor
import pons
import tradebook

ROOT = Path(__file__).parent
WEI = tradebook.WEI
TOKEN = pons.to_checksum_address("0x94fa961993480f2786d91d756476a766432d5ab7")
CURVE = pons.to_checksum_address("0xfca246b2d47ce1f7ae255f5c5eda8374af09b2b6")
GOOGL = pons.to_checksum_address("0x2e0847e8910a9732eb3fb1bb4b70a580adad4fe3")
ME = pons.to_checksum_address("0x6ce4085efb52a6ebdb7d6989beb8860847f4b42a")
SECRET = "a" * 64
NOW = 1_780_000_000.0
S = pons.selector


def ret(types, vals):
    return "0x" + encode(types, vals).hex()


class World:
    """One coin on this chain, and every read the executor makes about it."""

    def __init__(self):
        self.block = 60_500_000
        self.gas_price = 365_000_000          # about 0.365 gwei, as this chain runs
        self.phase = pons.Phase.CURVE
        self.exists = True
        self.pair = pons.ZERO
        self.decimals = 18
        self.symbol = "EREBUS"
        self.name = "erebus"
        self.pool_rate = 1000                 # tokens per wei in the graduated pool
        self.hook = None                      # lets a test pause inside a call
        self.dead = None                      # an RPC error to raise instead of answering
        self.curve = dict(quote_reserve=4_646_157_047_298_115_655,
                          token_reserve=361_589_154_842_919_513_074_871_874,
                          sellable=300_000_000 * 10 ** 18, reserved=0,
                          real_quote=2_966_157_047_298_115_655,
                          phantom=1_680_000_000_000_000_000,
                          threshold=4_200_000_000_000_000_000,
                          fee_bps=100, tax_bps=0, graduated=False, ready=False)

    def state(self):
        """The same CurveState pons.curve_state builds, for the test's own maths."""
        c = self.curve
        return pons.CurveState(curve=CURVE, token=TOKEN, pair_token=self.pair, native_quote=True,
                               graduated=c["graduated"], ready_to_graduate=c["ready"],
                               quote_reserve=c["quote_reserve"], token_reserve=c["token_reserve"],
                               sellable_tokens=c["sellable"], reserved_tokens=c["reserved"],
                               real_quote_reserve=c["real_quote"], phantom_quote=c["phantom"],
                               graduation_threshold=c["threshold"], fee_bps=c["fee_bps"],
                               creator_tax_bps=c["tax_bps"])

    def answer(self, sel, tx):
        c = self.curve
        if sel == S("getLaunchedToken(address)"):
            return ret([pons.LAUNCH_TUPLE], [(TOKEN, CURVE, ME, ME, self.pair, c["threshold"], 0, 200,
                                              c["tax_bps"], False, int(self.phase), 0, 0, 0, self.exists)])
        if sel == S("getReserves()"):
            return ret(["uint256", "uint256"], [c["quote_reserve"], c["token_reserve"]])
        if sel == S("token()"):
            return ret(["address"], [TOKEN])
        if sel == S("pairToken()"):
            return ret(["address"], [self.pair])
        if sel == S("isNativeQuote()"):
            return ret(["bool"], [pons.is_zero(self.pair)])
        if sel == S("graduated()"):
            return ret(["bool"], [c["graduated"]])
        if sel == S("readyToGraduate()"):
            return ret(["bool"], [c["ready"]])
        if sel == S("sellableTokens()"):
            return ret(["uint256"], [c["sellable"]])
        if sel == S("reservedTokens()"):
            return ret(["uint256"], [c["reserved"]])
        if sel == S("realQuoteReserve()"):
            return ret(["uint256"], [c["real_quote"]])
        if sel == S("phantomQuote()"):
            return ret(["uint256"], [c["phantom"]])
        if sel == S("graduationThreshold()"):
            return ret(["uint256"], [c["threshold"]])
        if sel == S("creatorTaxBps()"):
            return ret(["uint256"], [c["tax_bps"]])
        if sel == S("feeBps()"):
            return ret(["uint256"], [c["fee_bps"]])
        if sel == S("decimals()"):
            return ret(["uint8"], [self.decimals])
        if sel == S("symbol()"):
            return ret(["string"], [self.symbol])
        if sel == S("name()"):
            return ret(["string"], [self.name])
        if sel == S(pons.QUOTER_SIG):
            (_key, zfo, amount, _hook), = decode([pons.QUOTE_PARAMS], bytes.fromhex(tx["data"][10:]))
            out = amount * self.pool_rate if zfo else amount // self.pool_rate
            return ret(["uint256", "uint256"], [out, 50_000])
        raise AssertionError(f"the fake chain was asked for {sel}")


class FakeResponse:
    def __init__(self, result):
        self.result = result

    def raise_for_status(self):
        pass

    def json(self):
        return {"jsonrpc": "2.0", "id": 1, "result": self.result}


class FakeSession:
    def __init__(self, world):
        self.world = world
        self.posts = []

    def post(self, url, json=None, **kw):
        w = self.world
        self.posts.append(json)
        method = json["method"]
        if w.dead:
            raise w.dead
        if method == "eth_blockNumber":
            return FakeResponse(hex(w.block))
        if method == "eth_gasPrice":
            return FakeResponse(hex(w.gas_price))
        if method != "eth_call":
            raise AssertionError(f"the fake chain was asked to {method}")
        tx = json["params"][0]
        sel = tx["data"][:10]
        if w.hook:
            w.hook(sel)
        return FakeResponse(w.answer(sel, tx))


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.where = tradebook.paths(self.tmp.name)
        self.world = World()
        self.session = FakeSession(self.world)

    def make(self, start_wei=WEI, max_age_s=20.0, clock=None):
        self.ledger = tradebook.Ledger(self.where["ledger"], book_path=self.where["book"],
                                       public_path=self.where["public"], start_wei=start_wei,
                                       clock=lambda: NOW)
        self.ex = executor.Executor(pons.Chain(session=self.session), self.ledger, SECRET,
                                    max_age_s=max_age_s, clock=clock or (lambda: NOW))
        return self.ex

    def body(self, side="buy", drive=0.5, token=TOKEN, seen_at=None, look_id="look-1"):
        return {"token": token, "side": side, "drive": drive,
                "seen_at": NOW if seen_at is None else seen_at, "look_id": look_id}

    def send(self, **kw):
        return self.ex.intent(self.body(**kw))

    def entries(self, kind=None):
        text = self.where["ledger"].read_text(encoding="utf-8")
        out = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
        return [e for e in out if kind is None or e["kind"] == kind]

    def reserve(self, positions=0):
        return (positions + 1) * executor.EXIT_UNITS * 2 * self.world.gas_price

    def buy(self, drive=0.5):
        code, payload = self.send(side="buy", drive=drive)
        self.assertEqual((code, payload["status"]), (200, "booked"), payload)
        return payload["event"]


class BodyShape(Base):
    def setUp(self):
        super().setUp()
        self.make()

    def test_an_extra_field_is_refused_before_anything_else(self):
        body = self.body()
        body["amount"] = 10 ** 18
        code, payload = self.ex.intent(body)
        self.assertEqual(code, 400)
        self.assertIn("amount", payload["reason"])
        self.assertEqual(self.entries("intent"), [])         # nothing was even written down

    def test_a_missing_field_is_refused(self):
        body = self.body()
        del body["look_id"]
        self.assertEqual(self.ex.intent(body)[0], 400)

    def test_junk_fields_are_refused(self):
        for body in ({**self.body(), "token": "0x1234"},
                     {**self.body(), "side": "short"},
                     {**self.body(), "drive": "0.5"},
                     {**self.body(), "drive": True},
                     {**self.body(), "seen_at": None},
                     {**self.body(), "look_id": 7},
                     {**self.body(), "drive": float("nan")},
                     ["token", "buy"]):
            self.assertEqual(self.ex.intent(body)[0], 400, body)

    def test_the_marks_body_is_exact_too(self):
        self.assertEqual(self.ex.marks({"tokens": [TOKEN], "why": 1})[0], 400)
        self.assertEqual(self.ex.marks({"token": TOKEN})[0], 400)
        self.assertEqual(self.ex.marks({"tokens": [TOKEN] * 200})[0], 400)


class Sizing(Base):
    def test_a_buy_spends_the_floor_of_drive_times_free_eth(self):
        self.make()
        ev = self.buy(drive=0.5)
        free = WEI - self.reserve()
        value = math.floor(Fraction(0.5) * free)
        quote = pons.quote_curve_buy(self.world.state(), value)
        booked = self.entries("booked")[-1]
        self.assertEqual(booked["eth_wei"], str(quote["spent"]))
        self.assertEqual(booked["tokens_raw"], str(quote["tokens_out"]))
        self.assertEqual(int(self.entries("quoted")[-1]["gas_wei"]),
                         executor.GAS_UNITS["curve"]["buy"] * self.world.gas_price)
        self.assertEqual(self.ledger.book.balance_wei,
                         WEI - quote["spent"] - executor.GAS_UNITS["curve"]["buy"] * self.world.gas_price)
        self.assertEqual(ev["venue"], "curve")
        self.assertEqual((ev["symbol"], ev["name"]), ("EREBUS", "erebus"))

    def test_drive_one_spends_every_free_wei(self):
        self.make()
        self.buy(drive=1.0)
        free = WEI - self.reserve()
        quote = pons.quote_curve_buy(self.world.state(), free)
        booked = self.entries("booked")[-1]
        self.assertEqual(int(self.entries("quoted")[-1]["eth_wei"]), quote["spent"])
        self.assertEqual(booked["eth_wei"], str(quote["spent"]))
        self.assertEqual(quote["spent"], free)             # the whole free balance, no cap

    def test_the_reserve_grows_with_the_number_of_positions(self):
        self.make()
        self.buy(drive=0.1)
        before = self.ledger.book.balance_wei
        self.buy(drive=1.0)
        spent = int(self.entries("booked")[-1]["eth_wei"])
        self.assertEqual(spent, before - self.reserve(positions=1))

    def test_a_sell_sells_the_floor_of_drive_times_the_position(self):
        self.make()
        self.buy(drive=0.5)
        held = self.ledger.book.positions[TOKEN]["tokens_raw"]
        code, payload = self.send(side="sell", drive=-0.25)
        self.assertEqual(payload["status"], "booked", payload)
        qty = math.floor(Fraction(0.25) * held)
        self.assertEqual(self.entries("booked")[-1]["tokens_raw"], str(qty))
        self.assertAlmostEqual(payload["event"]["fraction"], qty / held)
        self.assertEqual(self.ledger.book.positions[TOKEN]["tokens_raw"], held - qty)

    def test_drive_minus_one_sells_the_lot(self):
        self.make()
        self.buy(drive=0.5)
        held = self.ledger.book.positions[TOKEN]["tokens_raw"]
        code, payload = self.send(side="sell", drive=-1.0)
        self.assertEqual(self.entries("booked")[-1]["tokens_raw"], str(held))
        self.assertEqual(payload["event"]["fraction"], 1.0)
        self.assertEqual(self.ledger.book.positions, {})

    def test_a_balance_too_small_to_reserve_gas_has_nothing_to_spend(self):
        self.make(start_wei=1000)
        code, payload = self.send(drive=1.0)
        self.assertEqual((payload["status"], payload["reason"]), ("refused", "nothing to spend"))

    def test_a_drive_too_small_to_move_a_wei_has_nothing_to_spend(self):
        self.make()
        code, payload = self.send(drive=1e-30)
        self.assertEqual(payload["reason"], "nothing to spend")


class Consistency(Base):
    def test_a_buy_needs_a_positive_drive(self):
        self.make()
        for drive in (-0.5, 0.0, 1.5, -1.0):
            code, payload = self.send(side="buy", drive=drive)
            self.assertEqual(payload["status"], "refused")
            self.assertIn("(0, 1]", payload["reason"])

    def test_a_sell_needs_a_negative_drive(self):
        self.make()
        for drive in (0.5, 0.0, -1.5):
            code, payload = self.send(side="sell", drive=drive)
            self.assertIn("[-1, 0)", payload["reason"])

    def test_a_sell_of_a_coin_the_book_does_not_hold(self):
        self.make()
        code, payload = self.send(side="sell", drive=-0.5)
        self.assertEqual(payload["reason"], "not held")

    def test_a_stale_look_is_refused(self):
        self.make(max_age_s=20.0)
        code, payload = self.send(seen_at=NOW - 21)
        self.assertEqual(payload["reason"], "stale")
        self.assertEqual([e["kind"] for e in self.entries()], ["open", "intent", "refused"])

    def test_a_look_inside_the_window_is_not_stale(self):
        self.make(max_age_s=20.0)
        code, payload = self.send(seen_at=NOW - 19)
        self.assertEqual(payload["status"], "booked")

    def test_the_age_cannot_be_set_to_refuse_everything(self):
        """
        FLY_INTENT_MAX_AGE_S is read from .env, and at 0 it vetoed every trade
        the fly would ever make: the room commits, the ledger fills with
        "stale", and every page still says the fly is trading. Nothing outside
        the brain may cancel a trade, so the setting is held to a band in which
        it can only be what it is for.
        """
        self.assertEqual(executor.intent_max_age(0), executor.AGE_MIN_S)
        self.assertEqual(executor.intent_max_age(-5), executor.AGE_MIN_S)
        self.assertEqual(executor.intent_max_age(1e9), executor.AGE_MAX_S)
        self.assertEqual(executor.intent_max_age("nonsense"), 20.0)
        self.assertEqual(executor.intent_max_age(None), 20.0)
        self.assertEqual(executor.intent_max_age("30"), 30.0)
        self.assertEqual(executor.intent_max_age(20.0), 20.0)

    def test_a_zero_age_still_takes_a_fresh_look(self):
        self.make(max_age_s=0)
        code, payload = self.send(seen_at=NOW - 1)
        self.assertEqual(payload["status"], "booked")
        self.assertEqual(self.ex.max_age_s, executor.AGE_MIN_S)

    def test_a_refusal_is_an_event_with_no_money_in_it(self):
        self.make()
        ev = self.send(side="sell", drive=-0.5)[1]["event"]
        self.assertEqual((ev["kind"], ev["eth"], ev["tokens"], ev["gas_eth"]), ("refused", 0.0, 0.0, 0.0))
        self.assertEqual(self.ledger.book.balance_wei, WEI)


class Venues(Base):
    def test_settling_and_closed_are_refused(self):
        for phase, reason in ((pons.Phase.SETTLING, "settling into its pool"),
                              (pons.Phase.CLOSED, "closed")):
            self.setUp()
            self.make()
            self.world.phase = phase
            code, payload = self.send()
            self.assertEqual(payload["reason"], reason)

    def test_a_token_the_factory_never_launched_is_refused(self):
        self.make()
        self.world.exists = False
        self.assertIn("no record", self.send()[1]["reason"])

    def test_a_coin_quoted_in_something_other_than_eth_is_refused(self):
        self.make()
        self.world.pair = GOOGL
        self.assertEqual(self.send()[1]["reason"], "not a native-quote coin")

    def test_a_curve_that_quotes_nothing_refuses_the_sell(self):
        self.make()
        self.buy(drive=0.5)
        self.world.curve["ready"] = True            # graduation is due; the curve stops paying out
        code, payload = self.send(side="sell", drive=-1.0)
        self.assertEqual(payload["reason"], "the curve quotes nothing")
        self.assertIn(TOKEN, self.ledger.book.positions)

    def test_a_coin_that_graduated_while_it_was_held_is_sold_through_the_pool(self):
        self.make()
        self.buy(drive=0.5)
        held = self.ledger.book.positions[TOKEN]["tokens_raw"]
        self.world.phase = pons.Phase.POOL         # the curve filled while the fly was away
        code, marks = self.ex.marks({"tokens": [TOKEN]})
        self.assertEqual(marks["marks"][TOKEN]["venue"], "pool")
        self.assertAlmostEqual(marks["marks"][TOKEN]["value_eth"], (held // self.world.pool_rate) / WEI)
        code, payload = self.send(side="sell", drive=-1.0)
        ev = payload["event"]
        self.assertEqual((ev["status"] if "status" in ev else ev["kind"], ev["venue"]), ("booked", "pool"))
        self.assertEqual(int(self.entries("booked")[-1]["eth_wei"]), held // self.world.pool_rate)

    def test_an_rpc_that_will_not_answer_refuses_rather_than_guesses(self):
        self.make()
        self.world.dead = RuntimeError("eth_call: {'code': -32000}")
        code, payload = self.send()
        self.assertTrue(payload["reason"].startswith("unquotable:"), payload["reason"])
        self.assertEqual(self.ledger.book.balance_wei, WEI)


class Gas(Base):
    def test_the_first_curve_sell_pays_for_its_approval_once(self):
        self.make()
        self.buy(drive=0.5)
        price = self.world.gas_price
        self.send(side="sell", drive=-0.25)
        first = int(self.entries("booked")[-1]["gas_wei"])
        self.send(side="sell", drive=-0.25)
        second = int(self.entries("booked")[-1]["gas_wei"])
        self.assertEqual(first, (executor.GAS_UNITS["curve"]["sell"] + executor.APPROVE_UNITS["erc20"]) * price)
        self.assertEqual(second, executor.GAS_UNITS["curve"]["sell"] * price)

    def test_a_first_pool_sell_pays_for_both_approvals(self):
        self.make()
        self.buy(drive=0.5)
        self.world.phase = pons.Phase.POOL
        price = self.world.gas_price
        self.send(side="sell", drive=-0.25)
        booked = self.entries("booked")[-1]
        self.assertEqual(booked["approvals"], ["erc20", "permit2"])
        self.assertEqual(int(booked["gas_wei"]),
                         (executor.GAS_UNITS["pool"]["sell"] + executor.APPROVE_UNITS["erc20"]
                          + executor.APPROVE_UNITS["permit2"]) * price)

    def test_gas_comes_out_of_the_balance_and_into_the_cost(self):
        self.make()
        ev = self.buy(drive=0.5)
        gas = executor.GAS_UNITS["curve"]["buy"] * self.world.gas_price
        spent = int(self.entries("booked")[-1]["eth_wei"])
        self.assertEqual(self.ledger.book.positions[TOKEN]["cost_wei"], spent + gas)
        self.assertAlmostEqual(ev["gas_eth"], gas / WEI)


class Marks(Base):
    def test_a_coin_that_is_not_held_has_no_mark(self):
        self.make()
        code, payload = self.ex.marks({"tokens": [TOKEN]})
        self.assertEqual(code, 200)
        self.assertEqual(payload["marks"][TOKEN],
                         {"value_eth": None, "venue": None, "block": None, "tokens_raw": None,
                          "reason": "not held"})

    def test_a_mark_is_what_the_whole_position_would_return(self):
        self.make()
        self.buy(drive=0.5)
        held = self.ledger.book.positions[TOKEN]["tokens_raw"]
        payload = self.ex.marks({"tokens": [TOKEN]})[1]
        expected = pons.quote_curve_sell(self.world.state(), held)["quote_out"]
        self.assertAlmostEqual(payload["marks"][TOKEN]["value_eth"], expected / WEI)
        self.assertEqual(payload["marks"][TOKEN]["venue"], "curve")
        self.assertEqual(payload["marks"][TOKEN]["block"], self.world.block)

    def test_a_mark_never_changes_the_book(self):
        self.make()
        self.buy(drive=0.5)
        before_file = self.where["ledger"].read_bytes()
        before_book = self.ledger.book.state()
        self.ex.marks({"tokens": [TOKEN, "0xnope"]})
        self.assertEqual(self.where["ledger"].read_bytes(), before_file)
        self.assertEqual(self.ledger.book.state(), before_book)

    def test_a_mark_says_which_size_it_was_quoted_for(self):
        # a quote takes seconds and a sell can book inside that window, so the
        # answer has to carry the position it measured
        self.make()
        self.buy(drive=0.5)
        held = self.ledger.book.positions[TOKEN]["tokens_raw"]
        payload = self.ex.marks({"tokens": [TOKEN]})[1]
        self.assertEqual(payload["marks"][TOKEN]["tokens_raw"], str(held))

    def test_a_mark_is_never_read_off_a_book_that_is_changing(self):
        self.make()
        self.buy(drive=0.5)
        self.ex.lock.acquire()                       # an intent is being booked
        try:
            payload = self.ex.marks({"tokens": [TOKEN]})[1]
        finally:
            self.ex.lock.release()
        self.assertEqual(payload["marks"][TOKEN]["reason"], "busy")
        self.assertIsNone(payload["marks"][TOKEN]["value_eth"])

    def test_a_mark_on_a_dead_rpc_says_so_rather_than_lying(self):
        self.make()
        self.buy(drive=0.5)
        self.world.dead = RuntimeError("no")
        payload = self.ex.marks({"tokens": [TOKEN]})[1]
        self.assertIsNone(payload["marks"][TOKEN]["value_eth"])
        self.assertTrue(payload["marks"][TOKEN]["reason"].startswith("unquotable:"))


class OneAtATime(Base):
    def test_a_second_intent_while_one_is_in_flight_is_busy(self):
        self.make()
        started, release, out = threading.Event(), threading.Event(), []

        def hook(sel):
            if sel == S("getLaunchedToken(address)") and not started.is_set():
                started.set()
                release.wait(10)

        self.world.hook = hook
        first = threading.Thread(target=lambda: out.append(self.send(drive=0.5)), daemon=True)
        first.start()
        self.assertTrue(started.wait(10))
        code, payload = self.send(drive=0.25, look_id="look-2")
        release.set()
        first.join(10)
        self.assertEqual((code, payload), (409, {"status": "busy"}))
        self.assertEqual(out[0][1]["status"], "booked")
        self.assertEqual(len(self.entries("intent")), 1)      # the busy one is not written down


class FailClosed(Base):
    def test_an_unreadable_ledger_refuses_everything(self):
        self.make()
        self.buy(drive=0.1)
        with open(self.where["ledger"], "a", encoding="utf-8") as fh:
            fh.write("{not json\n")
        self.make()                                   # boot again on the broken file
        self.assertFalse(self.ledger.ok)
        code, payload = self.send(drive=0.5)
        self.assertEqual((code, payload["status"], payload["reason"]), (200, "refused", "ledger unreadable"))
        self.assertEqual(payload["event"]["seq"], 0)
        marks = self.ex.marks({"tokens": [TOKEN]})[1]
        self.assertEqual(marks["marks"][TOKEN]["reason"], "ledger unreadable")

    def test_an_interrupted_intent_comes_back_as_an_event(self):
        self.make()
        self.ledger.intent(TOKEN, "buy", 0.5, NOW, "look-lost")
        self.make()
        self.assertEqual(self.ex.events(0)[1]["events"][-1]["reason"], "interrupted")

    def test_a_broken_ledger_publishes_no_book_and_no_events(self):
        """
        A book that might be wrong must not be served as if it were right. The
        replay stops at the bad line, so the book in memory is a prefix: a full
        paper balance and no positions, while the file on disk records trades.
        """
        self.make()
        self.buy(drive=0.1)
        with open(self.where["ledger"], "a", encoding="utf-8") as fh:
            fh.write("{not json\n")
        self.make()
        self.assertFalse(self.ledger.ok)
        code, book = self.ex.book()
        self.assertEqual(code, 503)
        self.assertNotIn("eth_balance", book)
        self.assertTrue(book["error"])
        code, events = self.ex.events(0)
        self.assertEqual(code, 503)
        self.assertEqual(events["events"], [])
        self.assertIsNone(events["last"])
        # and what it wrote before it broke is still on disk, untouched
        self.assertEqual(json.loads(self.where["public"].read_text())["counts"]["buys"], 1)


class Api(Base):
    """The HTTP surface, over a real socket on the loopback interface."""

    def setUp(self):
        super().setUp()
        self.make()
        self.server = executor.make_server(self.ex, 0)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.port = self.server.server_address[1]

    def call(self, path, body=None, token=SECRET, method=None):
        url = f"http://127.0.0.1:{self.port}{path}"
        req = urllib.request.Request(url, method=method or ("POST" if body is not None else "GET"))
        if token is not None:
            req.add_header("X-Fly-Intent", token)
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, data, timeout=15) as r:
                return r.status, json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def test_it_binds_the_loopback_interface_only(self):
        self.assertEqual(self.server.server_address[0], "127.0.0.1")
        self.assertEqual(executor.HOST, "127.0.0.1")

    def test_health_needs_no_secret_and_says_paper(self):
        self.assertEqual(self.call("/health", token=None), (200, {"ok": True, "mode": "paper"}))

    def test_every_other_route_needs_the_secret(self):
        for path, body in (("/intent", self.body()), ("/marks", {"tokens": [TOKEN]}),
                           ("/book", None), ("/events", None)):
            self.assertEqual(self.call(path, body, token=None)[0], 403, path)
            self.assertEqual(self.call(path, body, token="b" * 64)[0], 403, path)
        self.assertEqual(self.entries("intent"), [])

    def test_an_intent_over_the_socket_is_booked(self):
        code, payload = self.call("/intent", self.body())
        self.assertEqual((code, payload["status"]), (200, "booked"))
        self.assertEqual(payload["event"]["side"], "buy")
        code, book = self.call("/book")
        self.assertEqual(book["mode"], "paper")
        self.assertIsNone(book["address"])
        self.assertEqual(book["counts"], {"buys": 1, "sells": 0, "refused": 0})
        self.assertEqual(len(book["positions"]), 1)

    def test_events_after_a_sequence_number(self):
        self.call("/intent", self.body(look_id="a"))
        self.call("/intent", self.body(side="sell", drive=-2.0, look_id="b"))
        code, all_events = self.call("/events?after=0")
        self.assertEqual([e["seq"] for e in all_events["events"]], [1, 2])
        self.assertEqual(all_events["last"], 2)
        code, rest = self.call("/events?after=1")
        self.assertEqual([e["seq"] for e in rest["events"]], [2])
        self.assertEqual(rest["events"][0]["kind"], "refused")

    def test_a_body_that_is_not_json_is_a_four_hundred(self):
        url = f"http://127.0.0.1:{self.port}/intent"
        req = urllib.request.Request(url, method="POST")
        req.add_header("X-Fly-Intent", SECRET)
        try:
            urllib.request.urlopen(req, b"{nope", timeout=15)
            self.fail("a broken body was accepted")
        except urllib.error.HTTPError as exc:
            self.assertEqual(exc.code, 400)

    def test_a_broken_ledger_answers_the_read_routes_with_a_failure(self):
        self.ex.ledger.ok = False
        self.ex.ledger.error = "line 3 is not JSON"
        for path in ("/book", "/events"):
            code, payload = self.call(path)
            self.assertEqual((code, payload["status"]), (503, "error"), path)
            self.assertIn("line 3", payload["error"])

    def test_there_is_no_other_route(self):
        self.assertEqual(self.call("/trade", {"token": TOKEN})[0], 404)
        self.assertEqual(self.call("/admin")[0], 404)

    def test_a_broken_executor_answers_rather_than_dropping_the_connection(self):
        """
        socketserver's default for an unhandled exception is a traceback and a
        closed connection. The room reads that as "unreachable", which it
        deliberately treats as proof of nothing - so the look stays pinned for
        the life of the install and the fly waits for a settlement nobody will
        ever write. A 500 is an answer.
        """
        def boom(body):
            raise RuntimeError("the book fell over")
        self.ex.intent = boom
        code, payload = self.call("/intent", self.body())
        self.assertEqual(code, 500)
        self.assertEqual(payload["status"], "error")
        self.assertIn("the book fell over", payload["reason"])

    def test_a_broken_read_route_answers_too(self):
        def boom():
            raise RuntimeError("no book")
        self.ex.book = boom
        self.assertEqual(self.call("/book")[0], 500)


class PaperOnly(unittest.TestCase):
    def test_live_stops_the_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, FLY_BACKROOM_LIVE="1", FLY_INTENT_TOKEN=SECRET,
                       FLY_STATE_DIR=tmp, FLY_EXECUTOR_PORT="4699")
            done = subprocess.run([sys.executable, str(ROOT / "executor.py")], env=env, cwd=str(ROOT),
                                  capture_output=True, text=True, timeout=120)
            self.assertNotEqual(done.returncode, 0)
            self.assertIn(executor.LIVE_REFUSAL, done.stderr)
            self.assertFalse((Path(tmp) / "backroom").exists())   # it stopped before touching anything

    def test_it_will_not_answer_without_a_secret(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {k: v for k, v in os.environ.items() if k != "FLY_INTENT_TOKEN"}
            env.update(FLY_STATE_DIR=tmp, FLY_EXECUTOR_PORT="4698")
            env.pop("FLY_BACKROOM_LIVE", None)
            done = subprocess.run([sys.executable, str(ROOT / "executor.py")], env=env, cwd=str(ROOT),
                                  capture_output=True, text=True, timeout=120)
            self.assertNotEqual(done.returncode, 0)
            self.assertIn("FLY_INTENT_TOKEN", done.stderr)

    def test_settings_are_resolved_the_way_the_room_resolves_them(self):
        """
        A FLY_STATE_DIR written only into .env used to put the ledger in one
        directory and the room's reader in another, with no error anywhere.
        """
        try:
            import launch
        except ImportError:              # the public copy calls it envcfg
            import envcfg as launch
        with mock.patch.object(launch, "load_env",
                               return_value={"FLY_STATE_DIR": "C:/only-in-the-file",
                                             "FLY_EXECUTOR_PORT": "4999"}):
            self.assertEqual(executor.state_dir(), "C:/only-in-the-file")
            self.assertEqual(executor.setting("FLY_EXECUTOR_PORT", "4671"), "4999")
            self.assertEqual(executor.setting("FLY_NOT_SET", "fallback"), "fallback")

    def test_the_source_cannot_reach_a_key_or_the_network(self):
        for name in ("executor.py", "tradebook.py"):
            src = (ROOT / name).read_text(encoding="utf-8").lower()
            for word in ("eth_account", "sign", "send_raw", "sendrawtransaction", "private key",
                         "rhprovider", "rhwallet", "keystore", "mnemonic"):
                self.assertNotIn(word, src, f"{name} contains {word!r}")

    def test_the_only_transactions_it_knows_about_are_quotes(self):
        src = (ROOT / "executor.py").read_text(encoding="utf-8")
        for builder in ("curve_buy_tx", "curve_sell_tx", "pool_swap_tx", "erc20_approve_tx",
                        "permit2_approve_tx"):
            self.assertNotIn(builder, src)        # it never even builds calldata


if __name__ == "__main__":
    unittest.main()
