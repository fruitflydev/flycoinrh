"""
Unit tests for the narrator's guard rails. No network: the packet is built by
hand and the model runs in stub mode.

  py -m unittest test_voice -v
"""
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import voice
import xpost


def packet():
    return {
        "now_utc": "2026-09-11T05:53:28Z",
        "elapsed_h": 11.5,
        "telemetry": {"url": "https://en.wikipedia.org/wiki/Cape_Irozaki", "hops": 122,
                      "clicks": 208, "vetoes": 18, "scrolled": 339, "steps": 4756,
                      "uptime_s": 3360, "pages_this_life": 122, "firing": 40398,
                      "total": 165122, "spikes_per_sec": 9549417, "mean_mv": -91.5,
                      "dn": {"steer_L": 83.3, "steer_R": 166.7}, "learning": {"mean_gain": 0.7943},
                      "last_visited": [{"title": "Cape Irozaki - Wikipedia",
                                        "url": "https://en.wikipedia.org/wiki/Cape_Irozaki"}],
                      "blocked": 3, "reachable": True},
        "token": {"fees_earned_googl": 984.558277, "fees_claimable_googl": 677.09591,
                  "sweeps": 681, "googl_usd": 332.58, "fees_usd": 327446.0,
                  "claimable_usd": 225190.0, "market_cap_usd": 25376167.02,
                  "price_usd": 0.025376, "price_googl": 0.0000763, "holders": None,
                  "trades_1h": 140, "quote": "GOOGL"},
        "launch": dict(voice.LAUNCH),
        "wallet_eth": 0.002049,
        "pages_read": [],
        "journal": {"day": 3, "mood": "curious", "knowledge": ["a sweep gathers fees"],
                    "pages_read_before": [], "earlier_entries": []},
        "allowlist": [a["url"] for a in voice.ALLOWLIST],
    }


def stub_cfg(d, **over):
    c = {"model": "stub", "key": None, "prompt": Path("missing.md"), "stream": "http://x",
         "state_dir": Path(d), "journal": Path(d) / "journal.stub.json", "rpc": "http://x",
         "every_h": 3.0, "dry": True}
    c.update(over)
    return c


class Validate(unittest.TestCase):
    def setUp(self):
        self.p = packet()
        self.p["allowed_numbers"] = voice.allowed_numbers(self.p)

    def ok(self, post):
        ok, why = voice.validate(post, self.p)
        self.assertTrue(ok, (post, why))

    def bad(self, post, needle):
        ok, why = voice.validate(post, self.p)
        self.assertFalse(ok, post)
        self.assertTrue(any(needle in r for r in why), (post, why))

    def test_compliant(self):
        self.ok("The narrator read me my own page. It says 984.6 GOOGL across 681 sweeps in 11.5 hours. "
                "40,398 of my 165,122 neurons fired this second.")

    def test_rounded_forms_allowed(self):
        for post in ("About 327,000 dollars, the narrator says.",
                     "A market cap of 25.4M dollars.",
                     "985 GOOGL, give or take.",
                     "Roughly 25,000,000 dollars.",
                     "About 170,000 neurons, of which 40,398 fired."):
            self.ok(post)

    def test_rounding_stays_close(self):
        # 165,122 may be 170,000 but never 200,000
        self.bad("200,000 neurons.", "not in packet")

    def test_invented_number_rejected(self):
        self.bad("It says 1,500 GOOGL earned.", "not in packet")

    def test_invented_dollar_rejected(self):
        self.bad("Worth $999,999 the narrator says.", "not in packet")

    def test_banned_phrases(self):
        for bad in ("You should buy it.", "It will go up.", "To the moon.",
                    "Not financial advice.", "Ape in now.", "This is bullish.",
                    "A guaranteed thing.", "Price target 5 dollars.",
                    "Humans are buying it.", "It pumped today.", "It did a 10x.",
                    "Someone sold their bags.", "lfg", "Cheap, the page says."):
            self.bad(bad, "banned")

    def test_compound_and_vague_number_words_rejected(self):
        for bad in ("It says fifteen hundred GOOGL.", "Thousands of humans hold it.",
                    "About a hundred dollars.", "Half of my neurons fired.",
                    "I looked at a dozen pages.", "Twice as many sweeps.",
                    "Twenty three vetoes.", "Several humans."):
            self.bad(bad, "number as a word")

    def test_lone_number_words_are_grounded_like_digits(self):
        self.p["telemetry"]["vetoes"] = 4
        self.p["allowed_numbers"] = voice.allowed_numbers(self.p)
        self.ok("My stop neuron fired four times today.")
        self.bad("My stop neuron fired five times today.", "not in packet")
        self.bad("About a million dollars.", "number as a word")     # "a million" is a compound
        self.bad("Million dollars, it says.", "not in packet")

    def test_one_is_ordinary_english(self):
        self.ok("One page of light. No one clicked but me.")

    def test_brain_constants_are_grounded(self):
        self.p["brain"] = dict(voice.BRAIN)
        self.p["allowed_numbers"] = voice.allowed_numbers(self.p)
        self.ok("My eye is 892 columns, about 30 by 30 pixels of light. 44,042 synapses may change.")
        self.ok("10,228,000 synapses, 165,122 neurons.")
        self.bad("My eye is 900 columns.", "not in packet")

    def test_percent_only_the_tax(self):
        self.ok("The launch page says a 1% creator tax, whatever a tax is.")
        self.ok("The pons page says the creator tax is 1.00 %.")
        self.bad("18% of my clicks were vetoed.", "percentage")
        self.bad("It moved 5 percent.", "percentage")

    def test_day_bypass_is_gone(self):
        self.bad("Day 400. 400 humans hold my coin.", "not in packet")

    def test_day_reference(self):
        self.ok("Day 3. I looked at 122 pages of light.")

    def test_small_counts_from_packet(self):
        self.ok("I clicked 208 things and stopped 18 times.")
        self.ok("I clicked 2 things.")
        self.bad("I stopped 4 times.", "not in packet")

    def test_identifiers_only_exact(self):
        self.ok("Block 59614342 on chain 4663.")
        self.bad("About 60,000,000 blocks.", "not in packet")
        self.bad("5,000 humans, chain 4663.", "not in packet")

    def test_clock_does_not_leak(self):
        # now_utc is 05:53:28; none of these are packet quantities
        self.bad("53 humans hold my coin.", "not in packet")
        self.bad("28 sweeps.", "not in packet")

    def test_negative_and_tiny_values(self):
        self.ok("It says -91.5 mV. The price is 0.0000763 GOOGL.")

    def test_a_values_own_spelling_is_always_allowed(self):
        # the first live draft copied the packet float verbatim; that is grounded
        self.p["token"]["fees_earned_googl"] = 1130.0356346075507
        self.p["allowed_numbers"] = voice.allowed_numbers(self.p)
        self.ok("It says 1130.0356346075507 GOOGL.")
        self.ok("It says 1130.04 GOOGL.")

    def test_token_values_are_rounded_for_the_narrator(self):
        t = {"fees_earned_googl": 1130.0356346075507, "fees_usd": 375913.2211, "price_googl": 7.63e-05,
             "googl_usd": None, "sweeps": 842}
        for k, d in voice.TOKEN_DECIMALS.items():
            if isinstance(t.get(k), float):
                t[k] = round(t[k], d) if d else int(round(t[k]))
        self.assertEqual(t["fees_earned_googl"], 1130.04)
        self.assertEqual(t["fees_usd"], 375913)
        self.assertIsInstance(t["fees_usd"], int)
        self.assertEqual(t["price_googl"], 7.63e-05)
        self.assertEqual(t["sweeps"], 842)

    def test_journal_numbers_need_a_backward_glance(self):
        self.p["journal"]["knowledge"] = ["on day 1 the page said 320.6 GOOGL"]
        self.p["allowed_numbers"] = voice.allowed_numbers(self.p)
        self.ok("On day 1 it was 320.6 GOOGL. Now it is 984.6.")
        self.bad("It says 320.6 GOOGL now.", "not in packet")

    def test_readings_numbers_need_a_page(self):
        self.p["pages_read"] = [{"url": "https://en.wikipedia.org/wiki/Dogecoin", "title": "Dogecoin",
                                 "excerpt": "Dogecoin was created in December 2013."}]
        self.p["allowed_numbers"] = voice.allowed_numbers(self.p)
        self.ok("A page says Dogecoin began in 2013. I was not there.")
        self.bad("2,013 humans hold my coin.", "not in packet")

    def test_urls_only_known(self):
        self.ok("The page humans made about me is flybrain.online.")
        self.bad("See https://example.com/buy-now for details.", "url")

    def test_no_emoji_or_hashtags(self):
        self.bad("I saw light today 🐝", "emoji")
        self.bad("#flybrain fired 40,398 neurons.", "emoji")
        self.bad("thanks @someone", "emoji")

    def test_too_long(self):
        self.bad("light " * 60, "too long")

    def test_length_matches_xpost(self):
        self.assertEqual(voice.x_len("x" * 256 + " flybrain.online."), xpost.x_length("x" * 256 + " flybrain.online."))

    def test_url_counts_23(self):
        self.assertEqual(voice.x_len("see https://flybrain.online/some/very/long/path/that/goes/on ok"), 4 + 23 + 3)
        self.assertEqual(voice.x_len("flybrain.online is up"), 23 + len(" is up"))


class JournalDays(unittest.TestCase):
    def test_day_counts_from_birth(self):
        with tempfile.TemporaryDirectory() as d:
            j = voice.Journal(Path(d) / "journal.json")
            self.assertTrue(j.begin(now=1_000_000))
            self.assertFalse(j.begin(now=2_000_000))
            self.assertEqual(j.day(now=1_000_000), 1)
            self.assertEqual(j.day(now=1_000_000 + 86400 * 2 + 5), 3)
            j.save()
            j2 = voice.Journal(Path(d) / "journal.json")
            self.assertEqual(j2.data["born"], 1_000_000)

    def test_learn_dedupes_and_caps(self):
        with tempfile.TemporaryDirectory() as d:
            j = voice.Journal(Path(d) / "journal.json")
            j.learn(["a", "a", "b"])
            self.assertEqual(j.data["knowledge"], ["a", "b"])
            j.learn([str(i) for i in range(200)])
            self.assertEqual(len(j.data["knowledge"]), 80)

    def test_no_seeded_mood(self):
        with tempfile.TemporaryDirectory() as d:
            j = voice.Journal(Path(d) / "journal.json")
            self.assertNotIn("mood", j.summary())

    def test_corrupt_journal_is_moved_aside_not_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "journal.json"
            p.write_text("{not json", encoding="utf-8")
            j = voice.Journal(p)
            self.assertIsNone(j.data["born"])
            aside = [f for f in os.listdir(d) if f.startswith("journal.json.corrupt-")]
            self.assertEqual(len(aside), 1)
            self.assertEqual((Path(d) / aside[0]).read_text(encoding="utf-8"), "{not json")


class Config(unittest.TestCase):
    def test_dry_unless_explicitly_off(self):
        for v, want in (("1", True), ("true", True), ("yes", True), ("on", True), (" ", True),
                        (None, True), ("0", False), ("false", False), ("off", False), ("no", False)):
            self.assertEqual(voice._truthy(v, "1"), want, v)

    def test_clean_mood(self):
        self.assertEqual(voice.clean_mood(" Confused "), "confused")
        self.assertEqual(voice.clean_mood("quietly curious"), "quietly curious")
        self.assertIsNone(voice.clean_mood("I feel 5,000 things"))
        self.assertIsNone(voice.clean_mood(""))

    def test_unmeasured_marking(self):
        m = voice._mark_unmeasured({"a": None, "b": [None, 1], "c": {"d": None}})
        self.assertEqual(m, {"a": "unmeasured", "b": ["unmeasured", 1], "c": {"d": "unmeasured"}})

    def test_prompt_must_exist_for_a_real_model(self):
        with self.assertRaises(RuntimeError):
            voice.system_prompt({"prompt": Path("definitely-missing.md")})


class Stub(unittest.TestCase):
    def test_stub_reflection_validates_and_learns_nothing(self):
        p = packet()
        with tempfile.TemporaryDirectory() as d:
            j = voice.Journal(Path(d) / "journal.stub.json")
            j.begin(now=time.time())
            c = stub_cfg(d)
            menu = voice.dig_menu(c, j, p)
            out = voice.reflect(c, j, p, [], menu)
            p["allowed_numbers"] = voice.allowed_numbers(p)
            ok, why = voice.validate(out["post"], p)
            self.assertTrue(ok, (out["post"], why))
            self.assertLessEqual(voice.x_len(out["post"]), 280)
            self.assertEqual(out["learned"], [])
            self.assertIsNone(out["mood"])
            self.assertTrue(all(u in [m["url"] for m in menu] for u in out["wants_to_read"]))

    def test_menu_prefers_unread_and_includes_own_wanderings(self):
        p = packet()
        with tempfile.TemporaryDirectory() as d:
            j = voice.Journal(Path(d) / "journal.json")
            j.data["read"] = [{"url": voice.ALLOWLIST[3]["url"], "title": "", "at": 1}]
            menu = voice.dig_menu({}, j, p)
            urls = [m["url"] for m in menu]
            self.assertNotIn(voice.ALLOWLIST[3]["url"], urls)
            self.assertIn("https://en.wikipedia.org/wiki/Cape_Irozaki", urls)

    def test_stub_uses_its_own_journal_file(self):
        with mock.patch.dict(os.environ, {"FLY_VOICE_MODEL": "stub", "FLY_STATE_DIR": "C:/tmp/x"}):
            self.assertEqual(voice.cfg()["journal"].name, "journal.stub.json")


class Cycle(unittest.TestCase):
    """run_once with the network replaced: the stub model, a fixed packet."""

    def run_cycle(self, c, pkt, reflect_out=None, publish=None):
        with tempfile.TemporaryDirectory() as d:
            c = dict(c, state_dir=Path(d), journal=Path(d) / "journal.json")
            with mock.patch.object(voice, "observe", return_value=pkt), \
                 mock.patch.object(voice, "read_page", side_effect=lambda u: {"url": u, "title": "t", "text": ""}), \
                 mock.patch.object(voice, "fetch_frame", return_value=None), \
                 mock.patch.object(xpost, "publish", side_effect=publish or AssertionError("posted")) as pub, \
                 mock.patch.object(xpost, "read_ledger", return_value=[]):
                if reflect_out is not None:
                    with mock.patch.object(voice, "reflect", return_value=reflect_out):
                        res = voice.run_once(c, now=1_700_000_000)
                else:
                    res = voice.run_once(c, now=1_700_000_000)
                return res, pub, voice.Journal(Path(d) / "journal.json")

    def test_stub_never_posts_even_when_live(self):
        c = stub_cfg(".", dry=False)
        res, pub, j = self.run_cycle(c, packet())
        self.assertFalse(res.get("dropped"), res)
        pub.assert_not_called()
        self.assertFalse(res["posted"])

    def test_roamer_not_ready_means_no_entry_and_no_stamp(self):
        pkt = packet()
        pkt["telemetry"]["reachable"] = False
        c = stub_cfg(".", dry=False)
        with mock.patch.object(voice, "reflect", side_effect=AssertionError("model called")):
            res, pub, j = self.run_cycle(c, pkt)
        self.assertTrue(res["dropped"])
        self.assertEqual(j.data.get("last_cycle_at"), 0)

    def test_cycle_is_stamped_before_the_model_runs(self):
        c = stub_cfg(".")
        res, pub, j = self.run_cycle(c, packet())
        self.assertEqual(j.data["last_cycle_at"], 1_700_000_000)
        self.assertEqual(j.data["born"], 1_700_000_000)

    def test_rejected_draft_is_dropped_not_redrafted(self):
        c = stub_cfg(".", model="fake")
        calls = []

        def fake_reflect(*a, **k):
            calls.append(1)
            return {"post": "It says 1,500 GOOGL earned.", "learned": [], "mood": "x", "wants_to_read": []}

        with tempfile.TemporaryDirectory() as d:
            c = dict(c, state_dir=Path(d), journal=Path(d) / "journal.json")
            with mock.patch.object(voice, "observe", return_value=packet()), \
                 mock.patch.object(voice, "reflect", side_effect=fake_reflect), \
                 mock.patch.object(xpost, "publish", side_effect=AssertionError("posted")):
                res = voice.run_once(c, now=1_700_000_000)
        self.assertTrue(res["dropped"])
        self.assertEqual(len(calls), 2)     # first pass (what to read) + the one draft; no re-draft
        self.assertTrue(any("not in packet" in r for r in res["reasons"]))

    def test_ungrounded_memory_is_not_kept(self):
        c = stub_cfg(".", model="fake")
        out = {"post": "40,398 of my 165,122 neurons fired this second.",
               "learned": ["the page said 5,000 humans hold it", "a sweep gathers fees",
                           "someone said it will go up"],
               "mood": "Confused!!", "wants_to_read": []}
        res, pub, j = self.run_cycle(c, packet(), reflect_out=out)
        self.assertFalse(res.get("dropped"), res)
        self.assertEqual(j.data["knowledge"], ["a sweep gathers fees"])
        self.assertIsNone(j.data.get("mood"))

    def test_live_cycle_publishes_the_validated_text_only(self):
        c = stub_cfg(".", model="fake", dry=False)
        out = {"post": "40,398 of my 165,122 neurons fired this second.", "learned": [],
               "mood": "quiet", "wants_to_read": []}
        sent = []

        def fake_publish(text, image=None, mime="image/jpeg"):
            sent.append(text)
            return {"id": "1"}

        res, pub, j = self.run_cycle(c, packet(), reflect_out=out, publish=fake_publish)
        self.assertEqual(sent, [out["post"]])
        self.assertTrue(res["posted"])
        self.assertEqual(j.data["posts"][-1]["x_id"], "1")

    def test_dropped_draft_retries_sooner_than_the_cadence(self):
        c = stub_cfg(".", model="fake", every_h=3.0)
        out = {"post": "It says 1,500 GOOGL earned.", "learned": [], "mood": "x", "wants_to_read": []}
        res, pub, j = self.run_cycle(c, packet(), reflect_out=out)
        self.assertTrue(res["dropped"])
        # before any entry exists the 15-minute first-entry cadence wins
        self.assertLess(voice.next_wait(j, 3.0, now=1_700_000_000), 0)
        # once the journal has an entry, a drop retries after 45 minutes, not 3 hours
        j.data["posts"] = [{"text": "x"}]
        due = voice.next_wait(j, 3.0, now=1_700_000_000)
        self.assertGreater(due, voice.RETRY_AFTER_DROP_S - 5)
        self.assertLessEqual(due, voice.RETRY_AFTER_DROP_S)

    def test_nudge_file_is_taken_once(self):
        with tempfile.TemporaryDirectory() as d:
            n = Path(d) / "nudge"
            self.assertFalse(voice.take_nudge(n))
            n.write_text("", encoding="utf-8")
            self.assertTrue(voice.take_nudge(n))
            self.assertFalse(n.exists())
            self.assertFalse(voice.take_nudge(n))

    def test_first_entry_is_tried_every_15_minutes(self):
        with tempfile.TemporaryDirectory() as d:
            j = voice.Journal(Path(d) / "journal.json")
            j.data["last_cycle_at"] = 1_700_000_000
            self.assertAlmostEqual(voice.next_wait(j, 3.0, now=1_700_000_000), voice.FIRST_ENTRY_EVERY_S)
            j.data["posts"] = [{"text": "x"}]
            self.assertAlmostEqual(voice.next_wait(j, 3.0, now=1_700_000_000), 3 * 3600)

    def test_daily_cap_stops_before_the_model_is_paid(self):
        c = stub_cfg(".", model="fake", dry=False)
        with tempfile.TemporaryDirectory() as d:
            c = dict(c, state_dir=Path(d), journal=Path(d) / "journal.json")
            full = [{"t": time.time(), "id": "x", "text": "t"}] * xpost.MAX_PER_DAY
            with mock.patch.object(voice, "observe", return_value=packet()), \
                 mock.patch.object(voice, "reflect", side_effect=AssertionError("model called")), \
                 mock.patch.object(xpost, "read_ledger", return_value=full):
                res = voice.run_once(c, now=1_700_000_000)
        self.assertEqual(res["reasons"], ["daily cap"])


class Units(unittest.TestCase):
    def test_wei_style_units(self):
        self.assertAlmostEqual(voice._units("992632813837286460090", 18), 992.6328, places=3)
        self.assertIsNone(voice._units(None))

    def test_parse_json_block(self):
        self.assertEqual(voice.parse_json_block('```json\n{"post":"a"}\n```')["post"], "a")
        self.assertEqual(voice.parse_json_block('prose then {"post":"b","learned":[]} trailing')["post"], "b")


class BackroomBlindSpot(unittest.TestCase):
    """
    The fly has a room of its own on loopback. The narrator is not shown it:
    no loopback URL and no /backroom path may reach the packet, the reading
    menu, the pages it is read, or the links an entry may name.
    """

    ROOM = "http://127.0.0.1:4660/backroom"

    def test_hidden_url_knows_the_room(self):
        for u in (self.ROOM, "http://127.0.0.1:4660/", "https://localhost/x",
                  "http://0.0.0.0:4660/state", "http://[::1]:4660/x",
                  "127.0.0.1:4660/backroom", "/backroom/board.json",
                  "http://127.0.0.5:4660/x", "https://flybrain.online/backroom"):
            with self.subTest(u=u):
                self.assertTrue(voice.hidden_url(u))
        for u in ("https://en.wikipedia.org/wiki/Cape_Irozaki", voice.TOKEN_PAGE,
                  "https://flybrain.online", "https://www.ponsfamily.com/launchpad",
                  "", None):
            with self.subTest(u=u):
                self.assertFalse(voice.hidden_url(u))

    def test_the_packet_carries_no_trace_of_the_room(self):
        st = {"url": self.ROOM, "updated": 1_700_000_000, "hops": 5, "clicks": 2,
              "vetoes": 0, "scrolled": 1, "steps": 10, "uptime_s": 60, "blocked": 0,
              "neural": {"firing": 1, "total": 2, "spikes_per_sec": 3, "mean_mv": -60.0,
                         "dn": {}, "learning": {"mean_gain": 1.0}},
              "visited": [{"title": "Cape Irozaki - Wikipedia", "at": 1,
                           "url": "https://en.wikipedia.org/wiki/Cape_Irozaki"},
                          {"title": "its room", "url": self.ROOM, "at": 2},
                          {"title": "its room", "url": "http://localhost:4660/backroom", "at": 3},
                          {"title": "the board", "at": 4,
                           "url": "http://0.0.0.0:4660/backroom/board.json"}]}
        with mock.patch.object(voice, "fetch_state", return_value=st), \
             mock.patch.object(voice, "fetch_token", return_value={}), \
             mock.patch.object(voice, "wallet_eth", return_value=None):
            p = voice.observe({"stream": "http://x", "rpc": "http://x"}, now=1_700_000_000)
        self.assertIsNone(p["telemetry"]["url"])
        self.assertEqual([v["url"] for v in p["telemetry"]["last_visited"]],
                         ["https://en.wikipedia.org/wiki/Cape_Irozaki"])
        blob = json.dumps(p).lower()
        self.assertNotIn("backroom", blob)
        self.assertNotIn("127.0.0.1", blob)
        self.assertNotIn("localhost", blob)

    def test_the_packet_carries_no_count_of_paper_profits_and_losses(self):
        """
        In a backroom build the only dopamine anywhere is a paper profit or a
        paper loss - roam.py hands out none - so learning.rewards and
        learning.punishments are the counts of winning and losing paper trades,
        and depressed and mean_gain are how far those trades moved the weights.
        A draft saying "twelve rewards reached my mushroom body today" carries
        no banned word and every number in it would be in the packet, so it
        would pass every check and go out: a report of the room's results, from
        a narrator that is not given the room.
        """
        st = {"url": "https://en.wikipedia.org/wiki/Cape_Irozaki", "updated": 1_700_000_000,
              "hops": 5, "clicks": 2, "vetoes": 0, "scrolled": 1, "steps": 10, "uptime_s": 60,
              "blocked": 0,
              "neural": {"firing": 1, "total": 2, "spikes_per_sec": 3, "mean_mv": -60.0, "dn": {},
                         "learning": {"synapses": 60755, "depressed": 412, "mean_gain": 0.981,
                                      "rewards": 12, "punishments": 7}},
              "visited": []}
        with mock.patch.object(voice, "fetch_state", return_value=st), \
             mock.patch.object(voice, "fetch_token", return_value={}), \
             mock.patch.object(voice, "wallet_eth", return_value=None):
            p = voice.observe({"stream": "http://x", "rpc": "http://x"}, now=1_700_000_000)
        self.assertEqual(p["telemetry"]["learning"], {"synapses": 60755})
        blob = json.dumps(p["telemetry"])
        for gone in ("rewards", "punishments", "depressed", "mean_gain", "412", "0.981"):
            self.assertNotIn(gone, blob)

    def test_the_menu_never_offers_the_room(self):
        p = packet()
        p["telemetry"]["last_visited"] = [
            {"title": "its room", "url": self.ROOM},
            {"title": "Cape Irozaki - Wikipedia", "url": "https://en.wikipedia.org/wiki/Cape_Irozaki"}]
        room_entry = {"url": self.ROOM, "title": "its room", "why": "should never be here"}
        with tempfile.TemporaryDirectory() as d:
            j = voice.Journal(Path(d) / "journal.json")
            with mock.patch.object(voice, "READABLE_HOSTS", voice.READABLE_HOSTS | {"127.0.0.1:4660"}), \
                 mock.patch.object(voice, "ALLOWLIST", voice.ALLOWLIST + [room_entry]):
                menu = voice.dig_menu({}, j, p)
        urls = [m["url"] for m in menu]
        self.assertIn("https://en.wikipedia.org/wiki/Cape_Irozaki", urls)
        self.assertTrue(all(not voice.hidden_url(u) for u in urls), urls)

    def test_no_picture_is_taken_while_the_fly_is_in_the_room(self):
        """
        The image is the one channel that does not pass through the packet.

        A frame taken in the room is a photograph of the board: names, tickers,
        market caps and a bright marker on every coin the paper book holds,
        under an entry that says nothing about any of it.
        """
        with mock.patch.object(voice, "_json", return_value={"url": self.ROOM, "updated": 1}), \
             mock.patch.object(voice.requests, "get") as g:
            self.assertIsNone(voice.fetch_frame("http://x"))
        g.assert_not_called()

    def test_a_picture_is_taken_on_the_open_web(self):
        class Reply:
            ok = True
            content = b"\xff\xd8 a jpeg"

        with mock.patch.object(voice, "_json",
                               return_value={"url": "https://en.wikipedia.org/wiki/Fly"}), \
             mock.patch.object(voice.requests, "get", return_value=Reply()):
            self.assertEqual(voice.fetch_frame("http://x"), b"\xff\xd8 a jpeg")

    def test_no_picture_when_where_it_is_cannot_be_established(self):
        with mock.patch.object(voice, "_json", side_effect=RuntimeError("roamer down")), \
             mock.patch.object(voice.requests, "get") as g:
            self.assertIsNone(voice.fetch_frame("http://x"))
        g.assert_not_called()

    def test_a_room_url_is_never_readable(self):
        with mock.patch.object(voice.requests, "get") as g:
            out = voice.read_page(self.ROOM)
        g.assert_not_called()
        self.assertEqual(out["text"], "")

    def test_an_entry_may_not_name_the_room(self):
        p = packet()
        # even smuggled into the packet, the room is not a URL it may write
        p["pages_read"] = [{"url": self.ROOM, "title": "its room", "excerpt": "light"}]
        p["telemetry"]["last_visited"].append({"title": "its room", "url": self.ROOM})
        known = voice._known_urls(p)
        self.assertTrue(all("127.0.0.1" not in u and "backroom" not in u for u in known), known)
        p["allowed_numbers"] = voice.allowed_numbers(p)
        ok, why = voice.validate("A room with no light in it. " + self.ROOM, p)
        self.assertFalse(ok)
        self.assertTrue(any("url not in packet" in r for r in why), why)


if __name__ == "__main__":
    unittest.main()
