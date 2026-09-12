"""
web/backrooms.html, held to the DESIGN: nothing to press, the honesty banner
verbatim at the top, no script or style fetched from anywhere, a canvas room,
bars for each fly, the transcript, and a link to the dictionary JSON.

  py -m pytest -q test_backrooms_page.py
"""
import re
import unittest
from html.parser import HTMLParser
from pathlib import Path

import backrooms

PAGE = Path(__file__).parent / "web" / "backrooms.html"
FORBIDDEN = {"button", "form", "input", "select", "textarea"}


class Tags(HTMLParser):
    """Every start tag, its attributes and the order they appear in."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tags = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))

    handle_startendtag = handle_starttag


class Page(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = PAGE.read_text(encoding="utf-8")
        p = Tags()
        p.feed(cls.html)
        cls.tags = p.tags
        cls.script = cls.html.split("<script>", 1)[1]

    def test_there_is_nothing_to_press(self):
        found = sorted({t for t, _ in self.tags} & FORBIDDEN)
        self.assertEqual(found, [], f"the page must hold no controls, found {found}")

    def test_no_inline_handlers(self):
        for tag, attrs in self.tags:
            bad = [k for k in attrs if k.startswith("on")]
            self.assertEqual(bad, [], f"<{tag}> carries {bad}")
        self.assertEqual(re.findall(r"\son[a-z]+\s*=", self.html.split("<script>")[0]), [])

    def test_the_banner_is_verbatim_and_first_in_the_body(self):
        for para in backrooms.BANNER_PARAGRAPHS:
            self.assertIn(para, self.html)
        body = self.html.split("<body>", 1)[1]
        first = re.search(r"<(\w+)", body).group(1)
        self.assertEqual(first, "header")
        self.assertIn('class="banner"', body.split(">", 1)[0] + ">")
        head = body.split("</header>", 1)[0]
        for para in backrooms.BANNER_PARAGRAPHS:
            self.assertIn(para, head)

    def test_the_banner_stays_at_the_top(self):
        self.assertRegex(self.html, r"\.banner\{[^}]*position:sticky;top:0")

    def test_no_external_script_style_or_font(self):
        self.assertEqual([t for t, a in self.tags if t in ("link", "iframe", "img")], [])
        for tag, attrs in self.tags:
            if tag == "script":
                self.assertNotIn("src", attrs)
        self.assertEqual(re.findall(r"https?://[^\s'\"<)]+", self.html), [])
        self.assertNotIn("@import", self.html)

    def test_it_links_to_the_dictionary_json(self):
        hrefs = [a.get("href") for t, a in self.tags if t == "a"]
        self.assertIn("/dictionary.json", hrefs)
        self.assertTrue(all(h.startswith("/") for h in hrefs), hrefs)

    def test_it_reads_only_state_transcript_and_the_socket(self):
        urls = sorted(set(re.findall(r"fetch\(\s*['\"]([^'\"]+)", self.script)))
        self.assertEqual(urls, ["/state", "/transcript?after="])
        self.assertIn("'/ws'", self.script)
        self.assertIn("new WebSocket(", self.script)
        self.assertNotIn(".send(", self.script)

    def test_the_room_the_bars_and_the_transcript_exist(self):
        ids = {a.get("id") for _, a in self.tags}
        for need in ("room", "transcript", "barsA", "barsB", "posA", "posB",
                     "status", "how", "room-note"):
            self.assertIn(need, ids)
        canvas = [a for t, a in self.tags if t == "canvas"]
        self.assertEqual(len(canvas), 1)
        self.assertEqual(canvas[0]["id"], "room")

    def test_the_ground_is_near_black(self):
        m = re.search(r"--ground:\s*#([0-9a-fA-F]{6})", self.html)
        self.assertIsNotNone(m)
        r, g, b = (int(m.group(1)[i:i + 2], 16) for i in (0, 2, 4))
        self.assertLess(0.299 * r + 0.587 * g + 0.114 * b, 12)

    def test_the_ring_is_3_mm_and_the_arena_20_mm(self):
        self.assertIn("near_mm", self.script)
        self.assertIn("|| 3", self.script)
        self.assertIn("|| 20", self.script)
        self.assertIn("the faint ring around each is 3 mm", self.html)
        self.assertIn("the square is 20 mm", self.html)

    def test_heading_convention_matches_the_captioner(self):
        # 0 = +x, counter-clockwise positive; canvas y runs down so sin is negated
        self.assertIn("uy = -Math.sin(th)", self.script)
        self.assertIn("cy = H - f.y * ppm", self.script)

    def test_the_frames_are_read_by_the_names_the_loop_sends(self):
        for key in backrooms.FLY_FRAME_KEYS:
            if key in ("turn",):
                continue
            self.assertIn("." + key, self.script, key)
        self.assertIn("f.flies", self.script)
        self.assertIn("f.lines", self.script)
        self.assertIn("m.type === 'hello'", self.script)
        self.assertIn("m.type === 'frame'", self.script)
        self.assertIn("ln.seq", self.script)

    def test_the_bar_scale_is_stated(self):
        self.assertIn("GROUP_HZ = 100", self.script)
        self.assertIn("0 to 100 Hz scale", self.html)
        self.assertIn("0 to 200 Hz for the drive rows", self.html)

    def test_uncertain_groups_are_marked(self):
        self.assertIn("meta.confidence === 'uncertain'", self.script)
        self.assertIn("'UNCERTAIN'", self.script)

    def test_lines_carry_timestamps(self):
        self.assertIn("clock(ln.at)", self.script)
        self.assertIn("'t ' + fmt(ln.t, 1) + ' s'", self.script)


if __name__ == "__main__":
    unittest.main()
