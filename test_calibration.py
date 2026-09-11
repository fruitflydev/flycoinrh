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


if __name__ == "__main__":
    unittest.main()
