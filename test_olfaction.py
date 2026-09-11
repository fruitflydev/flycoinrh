"""
The nose, without the connectome: DoOR is read from disk, the brain is faked.

  py -m unittest test_olfaction -v
"""
import re
import unittest

import numpy as np

import olfaction

try:
    DOOR = olfaction.Door()
except FileNotFoundError:
    DOOR = None


class FakeBrain:
    """Just enough of FlyBrain: a type per neuron and where(type_re=)."""

    def __init__(self, glomeruli, per=3):
        self.types = [f"ORN_{g}" for g in glomeruli for _ in range(per)] + ["KCab-m"] * 5

    def where(self, type_re=None):
        rx = re.compile(type_re)
        return np.flatnonzero([bool(rx.search(t)) for t in self.types])


@unittest.skipIf(DOOR is None, "DoOR data not present")
class Door(unittest.TestCase):
    def test_row_name_shift_is_handled(self):
        # the mapping file's data rows carry an extra leading row-name column
        self.assertEqual(DOOR.glomerulus_of["Or42b"], "DM1")
        self.assertEqual(DOOR.glomerulus_of["Or22a"], "DM2")
        self.assertEqual(DOOR.glomerulus_of["Gr21a.Gr63a"], "V")
        self.assertEqual(DOOR.glomerulus_of["Ir64a.DC4"], "DC4")

    def test_merged_sensilla_are_skipped(self):
        self.assertNotIn("Or33b", DOOR.glomerulus_of)         # "DM5+DM3"
        self.assertNotIn("ac3A", DOOR.glomerulus_of)          # "DL2d/v"

    def test_every_convention_odorant_is_in_door(self):
        missing = [o for o in olfaction.WORD_ODOURS if o not in DOOR.key_of]
        self.assertEqual(missing, [])

    def test_known_responses(self):
        # acetic acid is the Ir64a/DC4 channel's best odorant; CO2 drives V fully
        self.assertAlmostEqual(DOOR.profile(DOOR.key_of["acetic acid"])["DC4"], 1.0, places=2)
        self.assertAlmostEqual(DOOR.profile(DOOR.key_of["carbon dioxide"])["V"], 1.0, places=2)
        self.assertGreater(DOOR.profile(DOOR.key_of["geosmin"])["DA2"], 0.5)

    def test_profiles_are_bounded(self):
        for name in ("ethyl acetate", "isopentyl acetate", "benzaldehyde"):
            vals = DOOR.profile(DOOR.key_of[name]).values()
            self.assertTrue(all(0.0 <= v <= 1.0 for v in vals))


@unittest.skipIf(DOOR is None, "DoOR data not present")
class Nose(unittest.TestCase):
    def setUp(self):
        gloms = sorted(set(DOOR.glomerulus_of.values()))
        self.fb = FakeBrain(gloms)
        self.nose = olfaction.Nose(self.fb, door=DOOR)

    def test_glomeruli_found_in_the_brain(self):
        self.assertGreaterEqual(len(self.nose.orn), 40)
        self.assertEqual(self.nose.cells, 3 * len(self.nose.orn))

    def test_words_name_real_smells(self):
        s = self.nose.smell("Banana Republic", "BANANA")
        self.assertEqual([o["name"] for o in s["odorants"]], ["isopentyl acetate"])
        self.assertEqual(s["odorants"][0]["why"], "word:banana")
        s = self.nose.smell("Mud Frog", "MUD")
        self.assertIn("geosmin", [o["name"] for o in s["odorants"]])

    def test_description_smells_are_fainter(self):
        by_name = self.nose.smell("vinegar", "VIN")
        by_desc = self.nose.smell("xq", "XQ", "a coin that smells of vinegar")
        self.assertEqual(by_desc["odorants"][0]["weight"], olfaction.DESCRIPTION_WEIGHT)
        self.assertAlmostEqual(by_desc["profile"]["DC4"], by_name["profile"]["DC4"] * olfaction.DESCRIPTION_WEIGHT, places=3)

    def test_no_smell_word_means_a_stable_hash_blend(self):
        a = self.nose.smell("PEPE", "PEPE", "the frog")
        b = self.nose.smell("pepe", "pepe", "a totally different description")
        self.assertEqual(a, b)                          # same name and symbol, same smell
        self.assertTrue(all(o["why"] == "hash" for o in a["odorants"]))
        self.assertLessEqual(len(a["odorants"]), olfaction.HASH_ODORANTS)
        self.assertNotEqual(a["profile"], self.nose.smell("DOGE", "DOGE")["profile"])

    def test_money_and_sugar_have_no_smell(self):
        for w in ("sugar", "sweet", "money", "moon", "gold", "pump"):
            self.assertNotIn(w, olfaction.WORD_TO_ODORANT)

    def test_drive_targets_only_receptor_neurons(self):
        s = self.nose.smell("banana", "BAN")
        d = self.nose.drive(s)
        self.assertTrue(d)
        orn = set(np.flatnonzero([t.startswith("ORN_") for t in self.fb.types]).tolist())
        for idx, hz in d.items():
            self.assertTrue(set(idx) <= orn)
            self.assertTrue(0 < hz <= olfaction.MAX_HZ)


if __name__ == "__main__":
    unittest.main()
