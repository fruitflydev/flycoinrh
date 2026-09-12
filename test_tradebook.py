"""
The paper ledger on its own: entries in write-ahead order, recovery of an
intent the process died inside, fail-closed on anything it cannot replay, and a
book rebuilt from the file that matches the book kept in memory wei for wei.

  py -m pytest -q test_tradebook.py
"""
import json
import tempfile
import unittest
from pathlib import Path

import tradebook

WEI = tradebook.WEI
TOKEN = "0x94FA961993480f2786D91d756476a766432d5Ab7"
OTHER = "0x4eB990547bce4a982432CA88Cf5FAe7eeD1A2d35"
NOW = 1_780_000_000.0


class Clock:
    """A clock that moves one second per entry, so order is visible in the file."""

    def __init__(self, t=NOW):
        self.t = t

    def __call__(self):
        self.t += 1.0
        return self.t


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.where = tradebook.paths(self.tmp.name)
        self.clock = Clock()

    def ledger(self, start_wei=WEI):
        return tradebook.Ledger(self.where["ledger"], book_path=self.where["book"],
                                public_path=self.where["public"], start_wei=start_wei,
                                clock=self.clock)

    def lines(self):
        text = self.where["ledger"].read_text(encoding="utf-8")
        return [json.loads(ln) for ln in text.splitlines() if ln.strip()]

    def kinds(self):
        return [e["kind"] for e in self.lines()]

    def write(self, entries):
        """Put a ledger on disk by hand, to test what a replay makes of it."""
        self.where["ledger"].parent.mkdir(parents=True, exist_ok=True)
        self.where["ledger"].write_text("".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")

    def buy(self, led, eth_wei=10 ** 17, tokens_raw=10 ** 22, gas_wei=10 ** 14, token=TOKEN,
            venue="curve", drive=0.5, approvals=()):
        i = led.intent(token, "buy", drive, NOW, "look-buy")
        led.quoted(i, venue=venue, block=60_500_000)
        return led.booked(i, token=token, side="buy", venue=venue, block=60_500_000, eth_wei=eth_wei,
                          tokens_raw=tokens_raw, gas_wei=gas_wei, symbol="EREBUS", name="erebus",
                          decimals=18, drive=drive, look_id="look-buy", approvals=approvals)

    def sell(self, led, eth_wei, tokens_raw, gas_wei=10 ** 14, token=TOKEN, venue="curve",
             drive=-0.5, approvals=("erc20",)):
        i = led.intent(token, "sell", drive, NOW, "look-sell")
        led.quoted(i, venue=venue, block=60_500_001)
        return led.booked(i, token=token, side="sell", venue=venue, block=60_500_001, eth_wei=eth_wei,
                          tokens_raw=tokens_raw, gas_wei=gas_wei, symbol="EREBUS", name="erebus",
                          decimals=18, drive=drive, look_id="look-sell", approvals=approvals)


class WriteAhead(Base):
    def test_the_ask_is_on_disk_before_the_fill(self):
        led = self.ledger()
        ev = self.buy(led)
        self.assertEqual(self.kinds(), ["open", "intent", "quoted", "booked"])
        self.assertEqual(ev["seq"], 1)
        self.assertEqual(led.book.balance_wei, WEI - 10 ** 17 - 10 ** 14)

    def test_every_line_is_flushed_as_it_is_written(self):
        led = self.ledger()
        i = led.intent(TOKEN, "buy", 0.5, NOW, "look")
        self.assertEqual(self.kinds(), ["open", "intent"])       # readable by another process now
        led.refused(i, "stale", token=TOKEN, side="buy", drive=0.5, look_id="look")
        self.assertEqual(self.kinds(), ["open", "intent", "refused"])

    def test_amounts_are_kept_as_whole_numbers_in_strings(self):
        led = self.ledger()
        self.buy(led, eth_wei=499881010000000001, tokens_raw=39721281065908980248457)
        booked = self.lines()[-1]
        self.assertEqual(booked["eth_wei"], "499881010000000001")
        self.assertEqual(booked["tokens_raw"], "39721281065908980248457")
        self.assertEqual(led.book.positions[TOKEN]["tokens_raw"], 39721281065908980248457)

    def test_the_open_entry_is_written_once(self):
        self.ledger()
        again = self.ledger()
        self.assertEqual(self.kinds(), ["open"])
        self.assertEqual(again.book.start_wei, WEI)


class Recovery(Base):
    def test_an_interrupted_intent_is_refused_on_the_next_boot(self):
        led = self.ledger()
        led.intent(TOKEN, "buy", 0.5, NOW, "look-lost")
        again = self.ledger()                     # as if the process had died there
        self.assertEqual(self.kinds(), ["open", "intent", "refused"])
        self.assertEqual(self.lines()[-1]["reason"], "interrupted")
        self.assertEqual(again.book.counts["refused"], 1)
        self.assertEqual(again.book.events[-1]["look_id"], "look-lost")
        self.assertTrue(again.ok)

    def test_a_recovered_ledger_takes_the_next_intent(self):
        led = self.ledger()
        led.intent(TOKEN, "buy", 0.5, NOW, "look-lost")
        again = self.ledger()
        self.assertEqual(again.intent(TOKEN, "buy", 0.5, NOW, "look-next"), 2)
        self.assertEqual(again.book.balance_wei, WEI)          # nothing was booked


class FailClosed(Base):
    def test_a_broken_line_stops_everything(self):
        led = self.ledger()
        self.buy(led)
        with open(self.where["ledger"], "a", encoding="utf-8") as fh:
            fh.write("{not json\n")
        again = self.ledger()
        self.assertFalse(again.ok)
        self.assertIn("not JSON", again.error)
        with self.assertRaises(tradebook.LedgerError):
            again.intent(TOKEN, "buy", 0.5, NOW, "look")

    def test_a_fill_with_no_intent_is_inconsistent(self):
        self.write([{"kind": "open", "at": NOW, "mode": "paper", "start_wei": str(WEI)},
                    {"kind": "booked", "at": NOW, "seq": 1, "id": 1, "token": TOKEN, "side": "buy",
                     "venue": "curve", "block": 1, "eth_wei": "1", "tokens_raw": "1", "gas_wei": "0"}])
        led = self.ledger()
        self.assertFalse(led.ok)
        self.assertIn("not open", led.error)

    def test_a_balance_below_zero_is_inconsistent(self):
        self.write([{"kind": "open", "at": NOW, "mode": "paper", "start_wei": "1000"},
                    {"kind": "intent", "at": NOW, "id": 1, "token": TOKEN, "side": "buy",
                     "drive": 1.0, "seen_at": NOW, "look_id": "x"},
                    {"kind": "booked", "at": NOW, "seq": 1, "id": 1, "token": TOKEN, "side": "buy",
                     "venue": "curve", "block": 1, "eth_wei": "5000", "tokens_raw": "1", "gas_wei": "0"}])
        led = self.ledger()
        self.assertFalse(led.ok)
        self.assertIn("below zero", led.error)

    def test_events_out_of_order_are_inconsistent(self):
        self.write([{"kind": "open", "at": NOW, "mode": "paper", "start_wei": str(WEI)},
                    {"kind": "intent", "at": NOW, "id": 1, "token": TOKEN, "side": "buy",
                     "drive": 1.0, "seen_at": NOW, "look_id": "x"},
                    {"kind": "refused", "at": NOW, "seq": 7, "id": 1, "reason": "stale"}])
        led = self.ledger()
        self.assertFalse(led.ok)
        self.assertIn("out of order", led.error)

    def test_a_corrupt_byte_fails_closed_rather_than_failing_to_start(self):
        """
        The writer is ASCII, so this needs byte corruption rather than a torn
        line - which is the unclean-shutdown case the write-ahead file exists
        for. Decoded strictly it raised UnicodeDecodeError (a ValueError, not
        an OSError) straight out of the constructor: the executor died at
        startup with a traceback, no /health and no reason anywhere, and under
        the supervisor that is eight restarts and then the whole fly.
        """
        led = self.ledger()
        self.buy(led)
        with open(self.where["ledger"], "ab") as fh:
            fh.write(b'{"kind": "booked", "at": 1, "token": "\xe2\x28\xa1"}\n')
        again = self.ledger()
        self.assertFalse(again.ok)                      # fail-closed, with a reason
        self.assertTrue(again.error)
        with self.assertRaises(tradebook.LedgerError):  # and still refusing to trade
            again.intent(TOKEN, "buy", 0.5, NOW, "look")

    def test_the_same_corruption_does_not_stop_a_rebuild_from_saying_why(self):
        self.write([{"kind": "open", "at": NOW, "mode": "paper", "start_wei": str(WEI)}])
        with open(self.where["ledger"], "ab") as fh:
            fh.write(b'{"kind": "\xff\xfe"}\n')
        with self.assertRaises(tradebook.LedgerError):
            tradebook.rebuild(self.where["ledger"])

    def test_a_book_that_cannot_be_published_does_not_lose_the_booking(self):
        """
        os.replace over a file another process holds open raises PermissionError
        on Windows, and public.json is read about twice a room step. Unguarded
        it escaped Ledger.booked, the executor's HTTP reply was never sent, and
        the room recorded "unreachable" - which is deliberately treated as proof
        of nothing, so that look stayed pinned for good. The ledger itself is
        already fsynced, so the derived files are allowed to be stale.
        """
        led = self.ledger()
        real = tradebook.atomic_write
        calls = []

        def refuse(path, data):
            calls.append(path)
            raise PermissionError(5, "Access is denied")

        tradebook.atomic_write = refuse
        try:
            ev = self.buy(led)
        finally:
            tradebook.atomic_write = real
        self.assertEqual(ev["kind"], "booked")
        self.assertIsNotNone(led.publish_error)
        self.assertTrue(calls)
        # the ledger is intact, so the next write republishes both files
        led.write_book()
        self.assertIsNone(led.publish_error)
        self.assertEqual(json.loads(self.where["public"].read_text())["counts"]["buys"], 1)

    def test_a_broken_ledger_does_not_republish_the_book(self):
        led = self.ledger()
        self.buy(led)
        good = self.where["public"].read_bytes()
        with open(self.where["ledger"], "a", encoding="utf-8") as fh:
            fh.write("{not json\n")
        self.ledger()
        self.assertEqual(self.where["public"].read_bytes(), good)   # the last good book stands


class PaperOnly(Base):
    def test_the_live_ledger_name_is_refused(self):
        with self.assertRaises(ValueError):
            tradebook.Ledger(Path(self.tmp.name) / "backroom" / tradebook.LIVE_LEDGER)

    def test_a_live_mode_is_refused(self):
        with self.assertRaises(ValueError):
            tradebook.Ledger(self.where["ledger"], mode="live")

    def test_a_ledger_written_by_another_mode_is_not_replayed(self):
        self.write([{"kind": "open", "at": NOW, "mode": "live", "start_wei": str(WEI)}])
        led = self.ledger()
        self.assertFalse(led.ok)
        self.assertIn("live", led.error)

    def test_the_published_book_says_paper_and_has_no_address(self):
        led = self.ledger()
        self.buy(led)
        public = json.loads(self.where["public"].read_text(encoding="utf-8"))
        self.assertEqual(public["mode"], "paper")
        self.assertIsNone(public["address"])
        self.assertEqual(public["counts"], {"buys": 1, "sells": 0, "refused": 0})


class Arithmetic(Base):
    def test_a_buy_carries_its_gas_into_the_cost(self):
        led = self.ledger()
        ev = self.buy(led, eth_wei=5 * 10 ** 17, gas_wei=10 ** 14)
        pos = led.book.positions[TOKEN]
        self.assertEqual(pos["cost_wei"], 5 * 10 ** 17 + 10 ** 14)
        self.assertAlmostEqual(ev["cost_basis_eth"], (5 * 10 ** 17 + 10 ** 14) / WEI)
        self.assertIsNone(ev["fraction"])
        self.assertIsNone(ev["realised_eth"])

    def test_a_sell_takes_its_share_of_the_cost_with_it(self):
        led = self.ledger()
        self.buy(led, eth_wei=5 * 10 ** 17, tokens_raw=10 ** 22, gas_wei=10 ** 14)
        cost = 5 * 10 ** 17 + 10 ** 14
        ev = self.sell(led, eth_wei=3 * 10 ** 17, tokens_raw=5 * 10 ** 21, gas_wei=2 * 10 ** 14)
        self.assertEqual(ev["fraction"], 0.5)
        self.assertEqual(led.book.positions[TOKEN]["cost_wei"], cost // 2)
        realised = 3 * 10 ** 17 - 2 * 10 ** 14 - (cost - cost // 2)
        self.assertAlmostEqual(ev["realised_eth"], realised / WEI)
        self.assertEqual(led.book.balance_wei, WEI - cost + 3 * 10 ** 17 - 2 * 10 ** 14)

    def test_selling_all_of_it_closes_the_position(self):
        led = self.ledger()
        self.buy(led, tokens_raw=10 ** 22)
        ev = self.sell(led, eth_wei=10 ** 17, tokens_raw=10 ** 22, drive=-1.0)
        self.assertEqual(ev["fraction"], 1.0)
        self.assertEqual(ev["cost_basis_eth"], 0.0)
        self.assertEqual(led.book.positions, {})
        self.assertEqual(led.public()["positions"], [])

    def test_a_sell_of_more_than_is_held_is_inconsistent(self):
        led = self.ledger()
        self.buy(led, tokens_raw=10 ** 22)
        with self.assertRaises(tradebook.LedgerError):
            self.sell(led, eth_wei=1, tokens_raw=10 ** 23)
        self.assertFalse(led.ok)

    def test_approvals_are_remembered_per_coin_and_venue(self):
        led = self.ledger()
        self.buy(led)
        self.assertFalse(led.book.approved(TOKEN, "curve", "erc20"))
        self.sell(led, eth_wei=10 ** 16, tokens_raw=10 ** 21, approvals=("erc20",))
        self.assertTrue(led.book.approved(TOKEN, "curve", "erc20"))
        self.assertFalse(led.book.approved(TOKEN, "pool", "erc20"))
        self.assertFalse(led.book.approved(OTHER, "curve", "erc20"))

    def test_only_the_newest_fifty_trades_are_published(self):
        led = self.ledger()
        for _ in range(55):
            self.buy(led, eth_wei=10 ** 15, tokens_raw=10 ** 20, gas_wei=10 ** 13)
        public = json.loads(self.where["public"].read_text(encoding="utf-8"))
        self.assertEqual(len(public["trades"]), 50)
        self.assertEqual(public["trades"][-1]["seq"], 55)
        self.assertEqual(public["counts"]["buys"], 55)


class Rebuild(Base):
    def test_the_file_alone_rebuilds_the_book(self):
        led = self.ledger()
        self.buy(led, eth_wei=3 * 10 ** 17, tokens_raw=10 ** 22, gas_wei=10 ** 14)
        i = led.intent(TOKEN, "sell", 0.5, NOW, "bad")
        led.refused(i, "a sell needs drive in [-1, 0)", token=TOKEN, side="sell", drive=0.5, look_id="bad")
        self.sell(led, eth_wei=2 * 10 ** 17, tokens_raw=3 * 10 ** 21, gas_wei=2 * 10 ** 14)
        self.buy(led, token=OTHER, eth_wei=10 ** 16, tokens_raw=10 ** 20, venue="pool")
        rebuilt = tradebook.rebuild(self.where["ledger"])
        self.assertEqual(rebuilt.state(), led.book.state())
        self.assertEqual(rebuilt.events, led.book.events)
        self.assertEqual(rebuilt.public(at=1.0), led.book.public(at=1.0))
        self.assertEqual(rebuilt.balance_wei, led.book.balance_wei)

    def test_the_book_on_disk_is_the_replayed_book(self):
        led = self.ledger()
        self.buy(led)
        on_disk = json.loads(self.where["book"].read_text(encoding="utf-8"))
        self.assertEqual(on_disk, tradebook.rebuild(self.where["ledger"]).state())


class LooseEvent(unittest.TestCase):
    def test_an_unwritable_refusal_is_marked_as_no_record(self):
        ev = tradebook.loose_event("ledger unreadable", token=TOKEN, side="buy", drive=0.5,
                                   look_id="look", at=NOW)
        self.assertEqual((ev["seq"], ev["kind"], ev["reason"]), (0, "refused", "ledger unreadable"))
        self.assertEqual((ev["eth"], ev["tokens"], ev["gas_eth"]), (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
