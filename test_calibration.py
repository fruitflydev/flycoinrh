"""
calibration.py without the connectome: the brain and mushroom body are faked.

  py -m unittest test_calibration -v
"""
import unittest

import numpy as np

import calibration


class FakeBrain:
    def __init__(self):
        self.type_names = np.array(["APL", "DM1_lPN", "DA1_vPN", "VP2+_adPN", "KCab-m", "KCg-m", "MBON01",
                                    "ORN_DM1", "ORN_V", "LPN_a", "PAM01"])
        self.n_types = len(self.type_names)


class FakeMB:
    reward_side = np.array([10, 11])      # PAM compartments -> avoidance
    punish_side = np.array([20, 21, 22])  # PPL1 compartments -> approach


class Gains(unittest.TestCase):
    def setUp(self):
        self.fb = FakeBrain()

    def test_stock_is_none(self):
        self.assertIsNone(calibration.gains_for(self.fb, "stock"))
        self.assertIsNone(calibration.gains_for(self.fb, {}))

    def test_only_matched_types_are_scaled(self):
        g = calibration.gains_for(self.fb, {"PN": 0.3, "APL": 10, "KC": 0.3})
        by = dict(zip(self.fb.type_names, g))
        self.assertAlmostEqual(by["APL"], 10.0, places=5)
        self.assertAlmostEqual(by["DM1_lPN"], 0.3, places=5)
        self.assertAlmostEqual(by["DA1_vPN"], 0.3, places=5)
        self.assertAlmostEqual(by["KCab-m"], 0.3, places=5)
        for untouched in ("MBON01", "ORN_DM1", "LPN_a", "PAM01"):
            self.assertEqual(by[untouched], 1.0, untouched)

    def test_clock_neuron_lpn_is_not_a_projection_neuron(self):
        self.assertNotIn("LPN_a", calibration.matched_types(self.fb, "PN"))

    def test_thermo_hygro_vp_projection_neurons_are_in_the_pn_group(self):
        # the measured settings scaled these too; the disclosure says so
        self.assertIn("VP2+_adPN", calibration.matched_types(self.fb, "PN"))
        self.assertIn("thermo- and\nhygrosensory VP", calibration.__doc__)

    def test_named_settings_match_their_gains(self):
        for name, s in calibration.SETTINGS.items():
            g = calibration.gains_for(self.fb, name)
            self.assertEqual(g is None, not s["gains"], name)

    def test_bad_inputs_refused(self):
        with self.assertRaises(KeyError):
            calibration.gains_for(self.fb, {"DAN": 2})
        with self.assertRaises(ValueError):
            calibration.gains_for(self.fb, {"APL": 0})

    def test_dtype_is_float32_for_the_simulator(self):
        self.assertEqual(calibration.gains_for(self.fb, "pn05_apl10_kc03").dtype, np.float32)

    def test_the_chosen_setting_is_a_measured_calibration(self):
        chosen = calibration.SETTINGS[calibration.CHOSEN]
        self.assertTrue(chosen["gains"])                      # not the stock brain
        for key in ("mean_leak", "kc_pct", "pattern_overlap", "roam_clicks", "roam_max_dn_shift_sd"):
            self.assertIn(key, chosen)
        self.assertLess(chosen["mean_leak"], calibration.SETTINGS["stock"]["mean_leak"])


class Readout(unittest.TestCase):
    def test_approach_is_ppl1_side_and_avoid_is_pam_side(self):
        rec = calibration.readout(FakeMB())
        self.assertEqual(rec["approach"].tolist(), [20, 21, 22])
        self.assertEqual(rec["avoid"].tolist(), [10, 11])

    def test_valence_sign(self):
        self.assertGreater(calibration.valence({"approach": [300, 300, 300], "avoid": [100, 100]}), 0)
        self.assertLess(calibration.valence({"approach": [50, 50, 50], "avoid": [200, 200]}), 0)
        self.assertEqual(calibration.valence({"approach": [120, 80, 100], "avoid": [100, 100]}), 0.0)

    def test_reward_raises_valence_through_the_measured_rule(self):
        # reward depresses KC input to PAM-compartment (avoidance) MBONs; with
        # avoidance output lower and approach unchanged, valence must rise
        before = calibration.valence({"approach": [200, 200, 200], "avoid": [200, 200]})
        after = calibration.valence({"approach": [200, 200, 200], "avoid": [150, 150]})
        self.assertGreater(after, before)


class SynapseReading(unittest.TestCase):
    """
    How a look is read: the Kenyon cells that fired, times the weights dopamine
    changes, summed on each side.
    """

    class MB:
        def __init__(self):
            self.fb = type("FB", (), {"n": 8})()
            self.pre = np.array([1, 2, 3, 4])          # one Kenyon cell per synapse
            self.side = np.array([-1, -1, 1, 1])       # approach, approach, avoid, avoid
            self.base = np.array([2.0, 3.0, 5.0, 7.0])
            self.gain = np.ones(4)

    def test_only_the_cells_that_fired_are_counted(self):
        mb = self.MB()
        self.assertEqual(calibration.syn_drive(mb, np.array([1, 3])), (2.0, 5.0))
        self.assertEqual(calibration.syn_drive(mb, np.array([2, 4])), (3.0, 7.0))
        self.assertEqual(calibration.syn_drive(mb, np.array([1, 2, 3, 4])), (5.0, 12.0))
        self.assertEqual(calibration.syn_drive(mb, None), (0.0, 0.0))

    def test_depression_can_only_lower_the_side_it_touches(self):
        mb = self.MB()
        before = calibration.syn_drive(mb, np.array([1, 3]))
        mb.gain[2] = 0.5                               # reward depresses an avoid synapse
        after = calibration.syn_drive(mb, np.array([1, 3]))
        self.assertEqual(after[0], before[0])
        self.assertLess(after[1], before[1])
        self.assertGreater(calibration.leaning(after), calibration.leaning(before))


class Leaning(unittest.TestCase):
    """(A - V) / (A + V), the scale-free part of the rule."""

    def test_sign(self):
        self.assertGreater(calibration.leaning((300.0, 100.0)), 0)
        self.assertLess(calibration.leaning((100.0, 300.0)), 0)
        self.assertEqual(calibration.leaning((100.0, 100.0)), 0.0)

    def test_doubling_both_sides_changes_nothing(self):
        self.assertEqual(calibration.leaning((30.0, 10.0)), calibration.leaning((60.0, 20.0)))

    def test_bounded(self):
        self.assertEqual(calibration.leaning((1e9, 0.0)), 1.0)
        self.assertEqual(calibration.leaning((0.0, 1e9)), -1.0)

    def test_silence_leans_nowhere(self):
        self.assertEqual(calibration.leaning((0.0, 0.0)), 0.0)


class Relative(unittest.TestCase):
    """A card against the other cards, which is the comparison the gate measured."""

    def test_a_card_that_leans_like_the_room_is_zero(self):
        self.assertEqual(calibration.relative((300.0, 100.0), [(300.0, 100.0)]), 0.0)
        self.assertEqual(calibration.relative((300.0, 100.0),
                                              [(30.0, 10.0), (600.0, 200.0)]), 0.0)

    def test_sign(self):
        self.assertGreater(calibration.relative((300.0, 100.0), [(100.0, 100.0)]), 0)
        self.assertLess(calibration.relative((100.0, 300.0), [(100.0, 100.0)]), 0)

    def test_it_is_the_mean_of_the_other_cards(self):
        want = calibration.leaning((300.0, 100.0)) - (calibration.leaning((100.0, 100.0))
                                                      + calibration.leaning((100.0, 300.0))) / 2.0
        self.assertAlmostEqual(
            calibration.relative((300.0, 100.0), [(100.0, 100.0), (100.0, 300.0)]), want)

    def test_bounded(self):
        self.assertEqual(calibration.relative((1e9, 0.0), [(0.0, 1e9)]), 1.0)
        self.assertEqual(calibration.relative((0.0, 1e9), [(1e9, 0.0)]), -1.0)

    def test_loudness_does_not_decide_it(self):
        # a coin that fires five times as many Kenyon cells leans the same way
        quiet = calibration.relative((30.0, 10.0), [(10.0, 10.0)])
        loud = calibration.relative((150.0, 50.0), [(10.0, 10.0)])
        self.assertAlmostEqual(quiet, loud)

    def test_silence_is_zero(self):
        self.assertEqual(calibration.relative((0.0, 0.0), [(0.0, 0.0)]), 0.0)

    def test_no_other_card_is_no_comparison(self):
        self.assertEqual(calibration.relative((300.0, 100.0), []), 0.0)

    def test_stored_leanings_give_the_same_answer_as_the_readings(self):
        """What a replay has in hand, against what the room had."""
        own, others = (300.0, 100.0), [(100.0, 100.0), (100.0, 300.0)]
        self.assertEqual(
            calibration.relative_leaning(calibration.leaning(own),
                                         [calibration.leaning(o) for o in others]),
            calibration.relative(own, others))


class AnEmptyReading(unittest.TestCase):
    """
    A run in which no Kenyon cell fired measured nothing, and 0.0 is not a
    neutral score for it: every leaning measured in the first paper run was
    between -0.18 and -0.31, so an empty reading sits about a quarter of the
    range above every real card. has_reading is how a caller tells the two
    apart; the room drops them and the offline gate never had one.
    """

    def test_nothing_fired_is_not_a_reading(self):
        self.assertFalse(calibration.has_reading((0.0, 0.0)))

    def test_anything_at_all_is(self):
        for r in ((1e-9, 0.0), (0.0, 1e-9), (300.0, 100.0), (100.0, 100.0)):
            self.assertTrue(calibration.has_reading(r), r)

    def test_it_is_the_case_leaning_cannot_speak_for(self):
        self.assertEqual(calibration.leaning((0.0, 0.0)), 0.0)
        self.assertEqual(calibration.leaning((100.0, 100.0)), 0.0)

    def test_what_an_empty_reference_entry_would_have_done(self):
        """The recorded look 1789229112324-0009, with and without it."""
        own, other = -0.27423, -0.20659
        self.assertAlmostEqual(calibration.relative_leaning(own, [0.0, other]), -0.170935, places=6)
        self.assertAlmostEqual(calibration.relative_leaning(own, [other]), -0.06764, places=5)


class Contrast(unittest.TestCase):
    def test_zero_when_stimulus_equals_blank(self):
        self.assertEqual(calibration.contrast((120.0, 80.0), (120.0, 80.0)), 0.0)

    def test_sign(self):
        self.assertGreater(calibration.contrast((150.0, 80.0), (120.0, 80.0)), 0)
        self.assertLess(calibration.contrast((120.0, 110.0), (120.0, 80.0)), 0)

    def test_bounded(self):
        for s, b in (((1e6, 0.0), (0.0, 0.0)), ((0.0, 1e6), (0.0, 0.0)), ((0.0, 0.0), (0.0, 5.0))):
            self.assertLessEqual(abs(calibration.contrast(s, b)), 1.0)

    def test_fixed_population_imbalance_cancels(self):
        self.assertEqual(calibration.contrast((300.0, 100.0), (300.0, 100.0)), 0.0)

    def test_silence_is_zero(self):
        self.assertEqual(calibration.contrast((0.0, 0.0), (0.0, 0.0)), 0.0)


if __name__ == "__main__":
    unittest.main()
