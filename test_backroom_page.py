"""
web/backroom.html, checked against the rails the roamer already has.

The fly walks this page with the same brain and the same click rule it uses on
the open web, so the page has to be a place where a click cannot commit
anything: no links, no controls, and no word on it that roam.VETO would stop
the fly from clicking anyway. The room's own logic depends on two more things
being true of the page - that every card carries its coin's address, and that
the ground it is drawn on is the grey backroom.py assumes when it puts a stored
look back into an empty room.

  py -m pytest -q test_backroom_page.py
"""
import re
import unittest
from html.parser import HTMLParser
from pathlib import Path

import backroom
import roam

PAGE = Path(__file__).parent / "web" / "backroom.html"
FORBIDDEN = {"a", "button", "form", "input", "select", "textarea"}


class Tags(HTMLParser):
    """Every start tag and its attributes, template contents included."""

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

    def test_there_is_nothing_to_press(self):
        found = sorted({t for t, _ in self.tags} & FORBIDDEN)
        self.assertEqual(found, [], f"the room must hold no controls, found {found}")

    def test_no_inline_handlers(self):
        for tag, attrs in self.tags:
            bad = [k for k in attrs if k.startswith("on")]
            self.assertEqual(bad, [], f"<{tag}> carries {bad}")
        self.assertEqual(re.findall(r"\son[a-z]+\s*=", self.html), [])

    def test_no_word_the_roamer_vetoes(self):
        # stricter than needed on purpose: the whole file, not only its text,
        # so a class name or a comment can never become a reason to veto
        hits = sorted({m.group(0).lower() for m in roam.VETO.finditer(self.html)})
        self.assertEqual(hits, [], f"the page says {hits}")

    def test_no_address_the_roamer_blocks(self):
        hits = sorted({m.group(0).lower() for m in roam.BLOCK.finditer(self.html)})
        self.assertEqual(hits, [])

    def test_every_card_shape_carries_its_coin(self):
        templates = [i for i, (t, _) in enumerate(self.tags) if t == "template"]
        self.assertGreaterEqual(len(templates), 2, "a card shape and a holding shape")
        for i in templates:
            tag, attrs = self.tags[i + 1]
            self.assertIn("data-token", attrs, f"the {tag} shape must carry data-token")
            self.assertIn("data-held", attrs)
        self.assertIn("setAttribute('data-token'", self.html)
        self.assertIn("setAttribute('data-held'", self.html)

    def test_the_ground_is_the_grey_the_room_redraws(self):
        m = re.search(r"--ground:\s*(#[0-9a-fA-F]{6})", self.html)
        self.assertIsNotNone(m, "the page must declare its ground colour")
        self.assertEqual(m.group(1).lower(), backroom.GROUND_CSS.lower())
        self.assertAlmostEqual(backroom.grey601(m.group(1)), backroom.GROUND, places=3)
        self.assertAlmostEqual(round(backroom.GROUND, 3), 0.047, places=3)

    def test_the_grid_is_fixed_so_rectangles_do_not_move(self):
        self.assertIn("grid-template-columns:repeat(4,280px)", self.html)
        self.assertIn("grid-auto-rows:200px", self.html)
        self.assertIn("width:280px;height:200px", self.html)

    def test_it_reads_the_board_the_room_publishes(self):
        self.assertIn("/backroom/board.json", self.html)
        self.assertIn("EVERY_MS = 20000", self.html)
        self.assertIn("setInterval(load, EVERY_MS)", self.html)

    def test_it_asks_for_nothing_but_that_board(self):
        urls = re.findall(r"fetch\(\s*['\"]([^'\"]+)", self.html)
        self.assertEqual(urls, ["/backroom/board.json"])
        self.assertEqual(re.findall(r"src\s*=", self.html), [])

    def test_a_logo_that_does_not_load_leaves_a_plain_square(self):
        self.assertIn("--logo:", self.html)
        self.assertIn("background-color:var(--logo)", self.html)

    def test_a_held_coin_is_marked_brightly(self):
        self.assertIn('.card[data-held="1"] .mark', self.html)


if __name__ == "__main__":
    unittest.main()
