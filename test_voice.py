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

    def test_number_words_rejected(self):
        for bad in ("It says fifteen hundred GOOGL.", "Thousands of humans hold it.",
                    "About a million dollars.", "Half of my neurons fired.",
                    "I looked at a dozen pages.", "Twice as many sweeps."):
            self.bad(bad, "number as a word")

    def test_one_is_ordinary_english(self):
        self.ok("One page of light. No one clicked but me.")

    def test_percent_only_the_tax(self):
        self.ok("The launch page says a 1% creator tax, whatever a tax is.")
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
                t[k] = round(t[k], d)
        self.assertEqual(t["fees_earned_googl"], 1130.04)
        self.assertEqual(t["fees_usd"], 375913.0)
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


if __name__ == "__main__":
    unittest.main()
