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

import voice


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
                      "last_events": ["arrived: Cape Irozaki"], "reachable": True},
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


class Validate(unittest.TestCase):
    def setUp(self):
        self.p = packet()
        self.p["allowed_numbers"] = voice.allowed_numbers(self.p)

    def test_compliant(self):
        ok, why = voice.validate(
            "The narrator read me my own page. It says 984.6 GOOGL across 681 sweeps in 11.5 hours. "
            "40,398 of my 165,122 neurons fired this second.", self.p)
        self.assertTrue(ok, why)

    def test_rounded_forms_allowed(self):
        for post in ("About 327,000 dollars, the narrator says.",
                     "A market cap of 25.4M dollars.",
                     "985 GOOGL, give or take.",
                     "Roughly 25,000,000 dollars."):
            ok, why = voice.validate(post, self.p)
            self.assertTrue(ok, (post, why))

    def test_invented_number_rejected(self):
        ok, why = voice.validate("It says 1,500 GOOGL earned.", self.p)
        self.assertFalse(ok)
        self.assertTrue(any("not in packet" in r for r in why), why)

    def test_invented_dollar_rejected(self):
        ok, why = voice.validate("Worth $999,999 the narrator says.", self.p)
        self.assertFalse(ok)

    def test_banned_phrases(self):
        for bad in ("You should buy it.", "It will go up.", "To the moon.",
                    "Not financial advice.", "Ape in now.", "This is bullish.",
                    "A guaranteed thing.", "Price target 5 dollars."):
            ok, why = voice.validate(bad, self.p)
            self.assertFalse(ok, bad)
            self.assertTrue(any("banned" in r for r in why) or any("not in packet" in r for r in why), (bad, why))

    def test_url_counts_23(self):
        self.assertEqual(voice.x_len("see https://flybrain.online/some/very/long/path/that/goes/on ok"), 4 + 23 + 3)
        self.assertEqual(voice.x_len("flybrain.online is up"), 23 + len(" is up"))

    def test_too_long(self):
        ok, why = voice.validate("light " * 60, self.p)
        self.assertFalse(ok)
        self.assertTrue(any("too long" in r for r in why))

    def test_day_reference(self):
        ok, why = voice.validate("Day 3. I looked at 122 pages of light.", self.p)
        self.assertTrue(ok, why)

    def test_small_counts_ok(self):
        ok, why = voice.validate("I clicked 2 things and stopped 4 times.", self.p)
        self.assertTrue(ok, why)

    def test_journal_numbers_are_grounded(self):
        self.p["journal"]["knowledge"] = ["on day 1 the page said 320.6 GOOGL"]
        self.p["allowed_numbers"] = voice.allowed_numbers(self.p)
        ok, why = voice.validate("On day 1 it was 320.6 GOOGL. Now it is 984.6.", self.p)
        self.assertTrue(ok, why)

    def test_readings_numbers_are_grounded(self):
        self.p["pages_read"] = [{"url": "https://en.wikipedia.org/wiki/Dogecoin", "title": "Dogecoin",
                                 "excerpt": "Dogecoin was created in December 2013."}]
        self.p["allowed_numbers"] = voice.allowed_numbers(self.p)
        ok, why = voice.validate("A page says Dogecoin began in 2013. I was not there.", self.p)
        self.assertTrue(ok, why)


class JournalDays(unittest.TestCase):
    def test_day_counts_from_birth(self):
        with tempfile.TemporaryDirectory() as d:
            j = voice.Journal(Path(d) / "journal.json")
            j.begin(now=1_000_000)
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


class Stub(unittest.TestCase):
    def test_stub_reflection_validates(self):
        p = packet()
        with tempfile.TemporaryDirectory() as d:
            j = voice.Journal(Path(d) / "journal.json")
            j.begin(now=time.time())
            c = {"model": "stub", "key": None, "prompt": Path("missing.md"), "stream": "http://x"}
            menu = voice.dig_menu(c, j, p)
            out = voice.reflect(c, j, p, [], menu)
            self.assertIn("post", out)
            p["allowed_numbers"] = voice.allowed_numbers(p)
            ok, why = voice.validate(out["post"], p)
            self.assertTrue(ok, (out["post"], why))
            self.assertLessEqual(voice.x_len(out["post"]), 280)
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


class Units(unittest.TestCase):
    def test_wei_style_units(self):
        self.assertAlmostEqual(voice._units("992632813837286460090", 18), 992.6328, places=3)
        self.assertIsNone(voice._units(None))

    def test_parse_json_block(self):
        self.assertEqual(voice.parse_json_block('```json\n{"post":"a"}\n```')["post"], "a")
        self.assertEqual(voice.parse_json_block('prose then {"post":"b","learned":[]} trailing')["post"], "b")


if __name__ == "__main__":
    unittest.main()
