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
        # CO2's strongest receptor reports 1.0 raw; the profile subtracts its spontaneous rate
        raw = DOOR.response[DOOR.key_of["carbon dioxide"]]
        adjusted = [float(v) - (0.0 if s in ("NA", "") else float(s))
                    for r, v, s in zip(DOOR.receptors, raw, DOOR.sfr)
                    if DOOR.glomerulus_of.get(r) == "V" and v not in ("NA", "")]
        self.assertAlmostEqual(max(float(v) for r, v in zip(DOOR.receptors, raw)
                                   if DOOR.glomerulus_of.get(r) == "V" and v not in ("NA", "")), 1.0, places=2)
        self.assertAlmostEqual(DOOR.profile(DOOR.key_of["carbon dioxide"])["V"], max(adjusted), places=6)
        self.assertLess(DOOR.profile(DOOR.key_of["carbon dioxide"])["V"], 1.0)
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


class SpontaneousFiring(unittest.TestCase):
    def test_sfr_row_is_kept_and_is_not_an_odorant(self):
        self.assertIsNotNone(DOOR.sfr)
        self.assertNotIn("SFR", DOOR.odorants)

    def test_response_at_or_below_baseline_is_not_driven(self):
        d = olfaction.Door.__new__(olfaction.Door)
        d.receptors = ["R1", "R2", "R3", "R4"]
        d.glomerulus_of = {"R1": "G1", "R2": "G2", "R3": "G3", "R4": "G4"}
        d.sfr = ["0.2", "0.2", "NA", "0.1"]
        d.response = {"k": ["0.2", "0.1", "0.3", "0.6"]}
        p = d.profile("k")
        self.assertEqual(p["G1"], 0.0)          # at baseline
        self.assertEqual(p["G2"], 0.0)          # below baseline: inhibition is not driven
        self.assertAlmostEqual(p["G3"], 0.3)    # no baseline measured: taken as 0
        self.assertAlmostEqual(p["G4"], 0.5)


class EqualSniff(unittest.TestCase):
    """Scaling every coin to the same total odour, so no coin shouts."""

    class FakeFly:
        def where(self, type_re=None, **kw):
            return np.array([0, 1], dtype=np.int64)

    def nose(self, equal_sniff=0.0):
        return olfaction.Nose(self.FakeFly(), door=DOOR, equal_sniff=equal_sniff)

    def test_off_by_default(self):
        plain = self.nose().smell("Banana", "BNNA")
        self.assertGreater(sum(plain["profile"].values()), 0)
        self.assertEqual(plain["profile"], self.nose(0.0).smell("Banana", "BNNA")["profile"])

    def test_every_coin_is_scaled_toward_the_same_total_but_not_to_it(self):
        # A glomerulus is never driven above the strongest response DoOR
        # measured for it, so a coin whose smell sits in very few glomeruli
        # cannot reach the target total: it lands at one unit per glomerulus.
        # "Every coin is smelled equally loudly" was the claim in the module
        # docstrings and on the site, and this is why it is not true: in the
        # first paper run the totals were 1.00, 1.24, 1.48, 1.49 and 2.00, and
        # the quiet looks read 226-239 approach a step against 24,275-29,211.
        n = self.nose(2.0)
        for name, symbol in (("Banana", "BNNA"), ("Mud", "MUD"), ("Moon Dog", "MDOG")):
            profile = n.smell(name, symbol)["profile"]
            total = sum(profile.values())
            self.assertAlmostEqual(total, min(2.0, float(len(profile))), places=2, msg=name)

    def test_a_loud_coin_no_longer_swamps_a_quiet_one(self):
        loud = sum(self.nose().smell("Mud", "MUD")["profile"].values())
        quiet = sum(self.nose().smell("Banana", "BNNA")["profile"].values())
        self.assertGreater(quiet / loud, 1.0)          # unscaled, banana spreads much wider
        loud2 = sum(self.nose(2.0).smell("Mud", "MUD")["profile"].values())
        quiet2 = sum(self.nose(2.0).smell("Banana", "BNNA")["profile"].values())
        self.assertLessEqual(max(loud2, quiet2) / min(loud2, quiet2), 2.0)

    def test_shape_is_kept(self):
        plain = self.nose().smell("Mud", "MUD")["profile"]
        scaled = self.nose(2.0).smell("Mud", "MUD")["profile"]
        self.assertEqual(sorted(plain), sorted(scaled))
        top = max(plain, key=plain.get)
        self.assertEqual(top, max(scaled, key=scaled.get))

    def test_never_above_one(self):
        for g, v in self.nose(50.0).smell("Banana", "BNNA")["profile"].items():
            self.assertLessEqual(v, 1.0, g)

    def test_the_smell_says_how_loud_it_actually_is(self):
        """
        The achieved total travels with the smell, so a look record, a panel or
        a reader can tell a coin that reached the target from one that could
        not. Nothing decides on it.
        """
        for name, symbol in (("Banana", "BNNA"), ("Mud", "MUD")):
            s = self.nose(2.0).smell(name, symbol)
            self.assertAlmostEqual(s["loudness"], sum(s["profile"].values()), places=3, msg=name)
            self.assertEqual(s["loudness_target"], 2.0)
            self.assertLessEqual(s["loudness"], 2.0 + 1e-6, msg=name)

    def test_a_one_glomerulus_coin_cannot_be_made_as_loud_as_a_wide_one(self):
        n = self.nose(2.0)
        narrow = n.smell("Mud", "MUD")
        wide = n.smell("Banana", "BNNA")
        self.assertLess(len(narrow["profile"]), len(wide["profile"]))
        self.assertLessEqual(narrow["loudness"], wide["loudness"])


if __name__ == "__main__":
    unittest.main()
