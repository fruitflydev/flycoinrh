import json
import unittest
from pathlib import Path

import numpy as np

import mb_sides
from mushroom import sides_sha

ROOT = Path(__file__).parent
REAL = ROOT / "build" / "mb_sides.json"


class Synthetic(unittest.TestCase):
    def setUp(self):
        self.bodies = np.array([10, 11, 20, 21, 30])
        self.types = np.array(["PAM01", "PPL101", "MBON01", "MBON11", "KCg"])

    def build(self, edges, min_syn=3):
        pre, post, w = (np.array(c) for c in zip(*edges))
        return mb_sides.build(self.bodies, self.types, [(pre, post, w)], min_syn)

    def test_side_follows_input_not_output(self):
        d = self.build([
            (10, 20, 5),      # PAM -> MBON01, input
            (20, 11, 500),    # MBON01 -> PPL1, output: must not count
            (11, 21, 7),      # PPL1 -> MBON11, input
            (21, 10, 900),    # MBON11 -> PAM, output: must not count
        ])
        self.assertEqual(d["types"]["MBON01"]["side"], "PAM")
        self.assertEqual(d["types"]["MBON11"]["side"], "PPL1")
        self.assertEqual(d["totals"], {"pam": 5, "ppl1": 7})

    def test_pairs_below_min_syn_are_ignored(self):
        d = self.build([(10, 21, 2), (11, 21, 3)])
        self.assertEqual(d["types"]["MBON11"]["pam_syn"], 0)
        self.assertEqual(d["types"]["MBON11"]["side"], "PPL1")

    def test_tie_or_no_input_has_no_side(self):
        d = self.build([(10, 20, 4), (11, 20, 4)])
        self.assertIsNone(d["types"]["MBON01"]["side"])
        self.assertIsNone(d["types"]["MBON11"]["side"])
        self.assertNotIn("20", d["bodies"])

    def test_weak_flag(self):
        d = self.build([(10, 20, 19), (11, 21, 20)])
        self.assertTrue(d["types"]["MBON01"]["weak"])
        self.assertFalse(d["types"]["MBON11"]["weak"])

    def test_sha_is_the_side_table_hash(self):
        d = self.build([(10, 20, 5), (11, 21, 7)])
        self.assertEqual(d["sides_sha"], sides_sha({"MBON01": "PAM", "MBON11": "PPL1"}))


@unittest.skipUnless(REAL.exists(), "build/mb_sides.json not generated")
class RealTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = json.loads(REAL.read_text(encoding="utf-8"))

    def side(self, t):
        return self.d["types"][t]["side"]

    def test_known_pam_types(self):
        for t in ("MBON01", "MBON02", "MBON03", "MBON05", "MBON04", "MBON10"):
            self.assertEqual(self.side(t), "PAM", t)

    def test_known_ppl1_types(self):
        for t in ("MBON11", "MBON12", "MBON13", "MBON14"):
            self.assertEqual(self.side(t), "PPL1", t)

    def test_totals(self):
        self.assertEqual(self.d["totals"], {"pam": 27090, "ppl1": 10819})

    def test_type_counts(self):
        sides = [e["side"] for e in self.d["types"].values()]
        self.assertEqual((sides.count("PAM"), sides.count("PPL1")), (15, 22))


if __name__ == "__main__":
    unittest.main()
