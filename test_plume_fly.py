"""
The plume coupling on a fake brain: DoOR is read from disk, the connectome is
not. One optional class loads the real brain once, and only when asked:

  py -m pytest -q test_plume_fly.py
  PLUME_REAL_BRAIN=1 py -m pytest -q -s test_plume_fly.py -k RealBrain

The defaults under test are protocol v2 (JO-E only, per-side equalisation,
state carried across a trial, 120 LIF steps per world step, smoothed
commands); the published v1 coupling is tested through plume_fly.V1.
"""
import math
import os
import re
import time
import unittest

import numpy as np

import olfaction
import plume_fly

try:
    DOOR = olfaction.Door()
except FileNotFoundError:
    DOOR = None

DEFAULT_JO = (("JO-CA1", 3), ("JO-EV1", 3), ("JO-CM", 2))


class FakeBrain:
    """
    Just enough of FlyBrain: types, bodies, where(type_re=) and a run() that
    echoes each recorded neuron's drive rate back as its firing rate, so the
    plumbing from drive to readout can be checked exactly. The state it
    returns counts the windows chained so far, so carrying can be checked.
    """

    def __init__(self, glomeruli, per=2, jo=DEFAULT_JO):
        types = [f"ORN_{g}" for g in glomeruli for _ in range(per)]
        for t, n in jo:
            types += [t] * n
        types += ["DNa02", "DNa02", "DNa01", "DNa01", "MDN", "MDN", "DNp09", "DNp09", "KCab-m", "KCab-m"]
        self.types = np.array(types)
        self.n = len(types)
        self.bodies = np.arange(self.n) + 10_000
        self.type_names, self.type_code = np.unique(self.types, return_inverse=True)
        self.n_types = len(self.type_names)
        self.calls = []

    def where(self, type_re=None, **_):
        rx = re.compile(type_re, re.I)
        return np.flatnonzero([bool(rx.search(t)) for t in self.types])

    def run(self, drive, steps, gains=None, record=None, seed=0, state=None):
        self.calls.append(dict(drive=drive, steps=steps, gains=gains, record=record, seed=seed, state=state))
        per_neuron = np.zeros(self.n)
        for k, r in drive.items():
            per_neuron[list(k)] = r
        out = {name: per_neuron[np.asarray(idx, dtype=np.int64)] for name, idx in (record or {}).items()}
        out["_fired"] = np.flatnonzero(per_neuron > 0)
        windows = 0 if state is None else int(state["v"][0])
        out["_state"] = {"v": np.array([windows + 1]), "refr": np.zeros(1, dtype=np.int32),
                         "rng": {"windows": windows + 1}}
        return out


def fake_motor(fb):
    dn = lambda t: fb.where(type_re=rf"^{t}$")
    a02, a01 = dn("DNa02"), dn("DNa01")
    return {"steer_L": a02[:1], "steer_R": a02[1:], "fwd_L": a01[:1], "fwd_R": a01[1:],
            "back": dn("MDN"), "stop": dn("DNp09")}


def fake_root_side(fb, unsided=1):
    """JO cells alternate L, R; the last `unsided` JO cells get no side."""
    side = np.array([""] * fb.n, dtype=object)
    jo = fb.where(type_re=plume_fly.JO_TYPE_RE)
    for j, i in enumerate(jo):
        side[i] = "L" if j % 2 == 0 else "R"
    for i in jo[len(jo) - unsided:]:
        side[i] = ""
    return side.astype(str)


def make_fly(glomeruli=None, jo=DEFAULT_JO, unsided=1, **kw):
    if glomeruli is None:
        glomeruli = sorted(set(DOOR.glomerulus_of.values()))
    fb = FakeBrain(glomeruli, jo=jo)
    fly = plume_fly.PlumeFly(fb, gains=None, motor=fake_motor(fb),
                             root_side=fake_root_side(fb, unsided), **kw)
    return fb, fly


def hand_run(rate_of):
    """A run() that returns rate_of(name, n_cells) for every recorded group, with a state."""
    def run(drive, steps, gains=None, record=None, seed=0, state=None, **_):
        out = {k: rate_of(k, len(record[k])) for k in record}
        out["_fired"] = np.array([], dtype=np.int64)
        out["_state"] = {"v": np.zeros(1), "refr": np.zeros(1, dtype=np.int32), "rng": {}}
        return out
    return run


def orn_rates(fb, drive):
    """{neuron index: rate} over ORN cells in a drive dict."""
    out = {}
    for k, v in drive.items():
        for i in k:
            if fb.types[i].startswith("ORN_"):
                out[int(i)] = float(v)
    return out


def jo_indices(fb, drive, prefix):
    """Indices of cells whose type starts with prefix that appear in a drive dict."""
    return {int(i) for k in drive for i in k if fb.types[i].startswith(prefix)}


@unittest.skipIf(DOOR is None, "DoOR data not present")
class OdourDrive(unittest.TestCase):
    def test_scales_linearly_with_c_and_is_zero_at_zero(self):
        fb, fly = make_fly()
        full = orn_rates(fb, fly.drives(1.0, 0.0))
        self.assertTrue(full)
        self.assertGreater(max(full.values()), 0.0)
        for c in (0.25, 0.5):
            part = orn_rates(fb, fly.drives(c, 0.0))
            self.assertEqual(set(part), set(full))
            for i in full:
                self.assertAlmostEqual(part[i], full[i] * c, places=9)
        zero = orn_rates(fb, fly.drives(0.0, 0.0))
        self.assertEqual(set(zero), set(full))
        self.assertTrue(all(v == 0.0 for v in zero.values()))

    def test_c_is_clipped_to_unit(self):
        fb, fly = make_fly()
        full = orn_rates(fb, fly.drives(1.0, 0.0))
        self.assertEqual(orn_rates(fb, fly.drives(3.0, 0.0)), full)
        self.assertTrue(all(v == 0.0 for v in orn_rates(fb, fly.drives(-0.5, 0.0)).values()))

    def test_every_orn_index_comes_from_the_odorant_glomeruli(self):
        fb, fly = make_fly()
        profile = DOOR.profile(DOOR.key_of["ethyl acetate"])
        driven = {g for g, v in profile.items() if v > 0}
        rates = orn_rates(fb, fly.drives(1.0, 0.0))
        seen = set()
        for i, hz in rates.items():
            g = fb.types[i][len("ORN_"):]
            self.assertIn(g, driven)
            self.assertAlmostEqual(hz, profile[g] * plume_fly.ODOUR_HZ, places=6)
            seen.add(g)
        # every glomerulus the brain has and the odorant drives is driven
        self.assertEqual(seen, driven & set(fly.nose.orn))
        self.assertGreaterEqual(len(seen), 20)

    def test_unknown_odorant_is_refused(self):
        fb = FakeBrain(["DM1"])
        with self.assertRaises(KeyError):
            plume_fly.PlumeFly(fb, motor=fake_motor(fb), root_side=fake_root_side(fb),
                               odorant="not an odorant")


class WindEncoding(unittest.TestCase):
    """The base cosine per antenna, unchanged from v1."""

    def test_headwind_is_equal_and_high(self):
        left, right = plume_fly.wind_rates(0.0, 100.0)
        self.assertAlmostEqual(left, right, places=9)
        self.assertAlmostEqual(left, 100.0 * (0.5 + 0.5 * np.cos(np.deg2rad(45))), places=9)
        self.assertGreater(left, 50.0)

    def test_wind_from_the_left_drives_the_left_antenna_harder(self):
        for deg in (20, 45, 90, 135):
            left, right = plume_fly.wind_rates(np.deg2rad(deg), 100.0)
            self.assertGreater(left, right, deg)
            left2, right2 = plume_fly.wind_rates(np.deg2rad(-deg), 100.0)
            self.assertGreater(right2, left2, -deg)
            self.assertAlmostEqual(left, right2, places=9)     # mirror symmetric

    def test_tailwind_is_equal_and_low(self):
        left, right = plume_fly.wind_rates(np.pi, 100.0)
        self.assertAlmostEqual(left, right, places=9)
        self.assertLess(left, 20.0)
        self.assertGreaterEqual(left, 0.0)

    def test_rates_stay_inside_zero_and_wind_hz(self):
        for deg in range(-180, 181, 15):
            left, right = plume_fly.wind_rates(np.deg2rad(deg), 100.0)
            self.assertTrue(0.0 <= left <= 100.0)
            self.assertTrue(0.0 <= right <= 100.0)

    def test_no_wind_sense_is_constant_and_symmetric(self):
        for deg in (-120, -45, 0, 30, 90, 180):
            self.assertEqual(plume_fly.wind_rates(np.deg2rad(deg), 100.0, wind_sense=False), (50.0, 50.0))


@unittest.skipIf(DOOR is None, "DoOR data not present")
class WindDrive(unittest.TestCase):
    def test_left_and_right_jo_get_their_own_scaled_rates(self):
        fb, fly = make_fly(["DM1", "VM7d"])
        self.assertEqual((fly.jo_left.size, fly.jo_right.size), (1, 2))     # the JO-E cells of the fake
        d = fly.wind_drive(np.deg2rad(60))
        left, right = plume_fly.wind_rates(np.deg2rad(60), fly.wind_hz)
        self.assertAlmostEqual(d[tuple(fly.jo_left.tolist())], left * fly.scale_left)
        self.assertAlmostEqual(d[tuple(fly.jo_right.tolist())], right * fly.scale_right)
        self.assertEqual((fly.scale_left, fly.scale_right), (1.5, 0.75))
        # the direction survives the scaling: the summed totals keep the base ratio
        tl, tr = d[tuple(fly.jo_left.tolist())] * 1, d[tuple(fly.jo_right.tolist())] * 2
        self.assertAlmostEqual(tl / tr, left / right)

    def test_v1_rates_are_the_unscaled_cosine(self):
        fb, fly = make_fly(["DM1", "VM7d"], **plume_fly.V1)
        d = fly.wind_drive(np.deg2rad(60))
        left, right = plume_fly.wind_rates(np.deg2rad(60), fly.wind_hz)
        self.assertEqual(d[tuple(fly.jo_left.tolist())], left)
        self.assertEqual(d[tuple(fly.jo_right.tolist())], right)
        self.assertGreater(left, right)

    def test_jo_cells_without_rootside_get_no_drive_v1(self):
        fb, fly = make_fly(["DM1"], **plume_fly.V1)
        jo = fb.where(type_re=plume_fly.JO_TYPE_RE)
        self.assertEqual(len(jo), 8)
        self.assertEqual(fly.jo_unsided.size, 1)
        self.assertEqual(fly.jo_left.size + fly.jo_right.size + fly.jo_unsided.size, len(jo))
        driven = {i for k in fly.drives(1.0, 0.3) for i in k}
        for i in fly.jo_unsided:
            self.assertNotIn(int(i), driven)
        for i in np.concatenate([fly.jo_left, fly.jo_right]):
            self.assertIn(int(i), driven)

    def test_jo_e_cells_without_rootside_get_no_drive_v2(self):
        # the last JO cell of this fake is a JO-EV1, and it is the unsided one
        fb, fly = make_fly(["DM1"], jo=(("JO-CA1", 2), ("JO-EV1", 4)))
        self.assertEqual((fly.jo_left.size, fly.jo_right.size, fly.jo_unsided.size), (2, 1, 1))
        self.assertEqual(fly.jo_silent.size, 2)
        driven = {i for k in fly.drives(1.0, 0.3) for i in k}
        for i in fly.jo_unsided:
            self.assertNotIn(int(i), driven)
            self.assertTrue(fb.types[i].startswith("JO-E"))
        for i in np.concatenate([fly.jo_left, fly.jo_right]):
            self.assertIn(int(i), driven)
        self.assertEqual(fly.jo_all.size, 6)
        self.assertEqual(fly.jo_left.size + fly.jo_right.size + fly.jo_unsided.size + fly.jo_silent.size, 6)

    def test_none_mode_is_tonic_v1_control(self):
        fb, fly = make_fly(["DM1"], **plume_fly.V1)
        for deg in (-90, 0, 60):
            d = fly.wind_drive(np.deg2rad(deg), wind_sense=False)
            self.assertEqual(d[tuple(fly.jo_left.tolist())], 0.5 * fly.wind_hz)
            self.assertEqual(d[tuple(fly.jo_right.tolist())], 0.5 * fly.wind_hz)
        fb, fly = make_fly(["DM1"])
        for deg in (-90, 0, 60):
            d = fly.wind_drive(np.deg2rad(deg), wind_mode="none")
            self.assertEqual(d[tuple(fly.jo_left.tolist())], 0.5 * fly.wind_hz * fly.scale_left)
            self.assertEqual(d[tuple(fly.jo_right.tolist())], 0.5 * fly.wind_hz * fly.scale_right)

    def test_drive_keys_do_not_overlap(self):
        fb, fly = make_fly()
        d = fly.drives(1.0, 0.5)
        seen = []
        for k in d:
            seen.extend(k)
        self.assertEqual(len(seen), len(set(seen)))


class SideEqualisation(unittest.TestCase):
    """Both antennae deliver the same total cell-Hz at the same deflection (v2, E)."""

    def make(self, n_left, n_right, **kw):
        fb = FakeBrain(["DM1"], jo=(("JO-EV1", n_left + n_right),))
        side = np.array([""] * fb.n, dtype=object)
        jo = fb.where(type_re=plume_fly.JO_TYPE_RE)
        for j, i in enumerate(jo):
            side[i] = "L" if j < n_left else "R"
        return plume_fly.PlumeFly(fb, gains=None, motor=fake_motor(fb), root_side=side.astype(str), **kw)

    def test_scales_are_mean_count_over_side_count(self):
        self.assertEqual(plume_fly.side_scales(4, 4), (1.0, 1.0))
        self.assertEqual(plume_fly.side_scales(5, 3), (0.8, 4.0 / 3.0))
        l, r = plume_fly.side_scales(157, 110)
        self.assertAlmostEqual(l, 133.5 / 157)
        self.assertAlmostEqual(r, 133.5 / 110)
        self.assertEqual(plume_fly.side_scales(5, 3, equalise=False), (1.0, 1.0))
        self.assertEqual(plume_fly.side_scales(5, 0), (0.5, 0.0))

    def test_totals_equal_at_equal_deflection(self):
        fly = self.make(5, 3)
        for phi in (0.0, np.pi):
            left, right = fly.wind_rates_sides(phi)
            self.assertAlmostEqual(left * 5, right * 3)
        # and at an oblique wind the totals keep the base cosine's ratio
        left, right = fly.wind_rates_sides(np.deg2rad(60))
        bl, br = plume_fly.wind_rates(np.deg2rad(60), fly.wind_hz)
        self.assertAlmostEqual((left * 5) / (right * 3), bl / br)

    def test_the_smaller_side_can_exceed_wind_hz(self):
        fly = self.make(5, 3)
        left, right = fly.wind_rates_sides(np.deg2rad(-45))     # full-on right antenna
        self.assertAlmostEqual(right, fly.wind_hz * 4.0 / 3.0)
        self.assertGreater(right, fly.wind_hz)
        self.assertLess(right * plume_fly.LIF_DT_MS / 1000.0, 0.1)   # spike probability per LIF step

    def test_step_records_delivered_and_realised_cell_hz_per_side(self):
        fly = self.make(5, 3)
        _, _, info = fly.step(0.0, 0.0, seed=1)                 # the fake brain echoes drive as rate
        self.assertAlmostEqual(info["jo_left_drive_cell_hz"], info["jo_right_drive_cell_hz"])
        self.assertAlmostEqual(info["jo_left_cell_hz"], info["jo_left_drive_cell_hz"])
        self.assertAlmostEqual(info["jo_right_cell_hz"], info["jo_right_drive_cell_hz"])
        self.assertAlmostEqual(info["jo_total_cell_hz"], info["jo_total_drive_cell_hz"])
        base, _ = plume_fly.wind_rates(0.0, fly.wind_hz)
        self.assertAlmostEqual(info["jo_total_cell_hz"], 8 * base)
        self.assertAlmostEqual(info["jo_left_hz"], base * 0.8)
        self.assertAlmostEqual(info["jo_right_hz"], base * 4.0 / 3.0)

    def test_describe_reports_counts_scales_and_totals(self):
        d = self.make(5, 3).describe()
        self.assertEqual((d["jo_left"], d["jo_right"], d["jo_silent"], d["jo_all"]), (5, 3, 0, 8))
        self.assertEqual((d["side_scale_left"], d["side_scale_right"]), (0.8, 4.0 / 3.0))
        self.assertTrue(d["equalise_sides"])
        t = d["jo_drive_totals"]
        self.assertEqual((t["scale_left"], t["scale_right"]), (0.8, 4.0 / 3.0))
        self.assertAlmostEqual(t["cases"]["headwind"]["left_minus_right_cell_hz"], 0.0)
        self.assertAlmostEqual(t["cases"]["headwind"]["equivalent_phi_deg"], 0.0)
        self.assertAlmostEqual(t["cases"]["control_no_wind_sense"]["equivalent_phi_deg"], 0.0)
        self.assertGreater(t["cases"]["wind_from_left"]["equivalent_phi_deg"], 0.0)
        self.assertLess(t["cases"]["wind_from_right"]["equivalent_phi_deg"], 0.0)

    def test_the_connectomes_jo_e_split_is_equalised(self):
        l, r = plume_fly.side_scales(157, 110)
        d = plume_fly.jo_drive_totals(157, 110, 100.0, l, r)
        hw = d["cases"]["headwind"]
        self.assertAlmostEqual(hw["total_cell_hz_left"], 85.355 * 133.5, delta=0.5)
        self.assertAlmostEqual(hw["total_cell_hz_right"], 85.355 * 133.5, delta=0.5)
        self.assertAlmostEqual(hw["left_minus_right_cell_hz"], 0.0, places=6)
        self.assertAlmostEqual(hw["equivalent_phi_deg"], 0.0, places=6)
        self.assertAlmostEqual(hw["per_cell_hz_right"], 85.355 * 133.5 / 110, delta=0.01)
        self.assertAlmostEqual(d["cases"]["wind_from_left"]["equivalent_phi_deg"], 90.0)
        self.assertAlmostEqual(d["cases"]["tailwind"]["left_minus_right_cell_hz"], 0.0, places=6)

    def test_v1_leaves_the_split_alone(self):
        fly = self.make(5, 3, **plume_fly.V1)
        self.assertEqual((fly.scale_left, fly.scale_right), (1.0, 1.0))
        _, _, info = fly.step(0.0, 0.0, seed=1)
        left, right = plume_fly.wind_rates(0.0, fly.wind_hz)
        self.assertAlmostEqual(info["jo_total_cell_hz"], 5 * left + 3 * right)
        _, _, info = fly.step(0.0, np.pi, seed=1, wind_sense=False)
        self.assertAlmostEqual(info["jo_total_cell_hz"], 8 * 50.0)


@unittest.skipIf(DOOR is None, "DoOR data not present")
class JoClasses(unittest.TestCase):
    """v2 drives JO-E and names it; JO-C is counted, recorded and never driven."""

    def test_only_jo_e_is_driven_by_default(self):
        fb, fly = make_fly(["DM1"])
        d = fly.drives(1.0, np.deg2rad(30))
        self.assertEqual(jo_indices(fb, d, "JO-C"), set())
        self.assertEqual(jo_indices(fb, d, "JO-E"), set(fly.jo_left.tolist()) | set(fly.jo_right.tolist()))
        jo_c = set(fb.where(type_re=r"^JO-C").tolist())
        self.assertEqual(set(fly.jo_silent.tolist()), jo_c)
        self.assertEqual(fly.jo_silent.size, 5)
        desc = fly.describe()
        self.assertEqual(desc["jo_driven_types"], ["JO-EV1"])
        self.assertEqual(desc["jo_silent_types"], ["JO-CA1", "JO-CM"])
        self.assertFalse(desc["jo_ce_coactivated"])
        self.assertEqual(desc["jo_driven_re"], r"^JO-E")

    def test_jo_c_is_never_driven_in_any_mode_at_any_angle(self):
        fb, fly = make_fly(["DM1"])
        fly.reset_state(1)
        for mode in plume_fly.WIND_MODES:
            for deg in (-150, -45, 0, 30, 90, 180):
                fly.step(1.0, np.deg2rad(deg), seed=3, wind_mode=mode)
                self.assertEqual(jo_indices(fb, fb.calls[-1]["drive"], "JO-C"), set(), (mode, deg))

    def test_jo_c_is_recorded_as_silent(self):
        fb, fly = make_fly(["DM1"])
        _, _, info = fly.step(1.0, 0.0, seed=3)
        rec = fb.calls[-1]["record"]
        self.assertIn("jo_silent", rec)
        np.testing.assert_array_equal(rec["jo_silent"], fly.jo_silent)
        self.assertEqual(info["jo_silent_hz"], 0.0)          # the fake echoes drive: none

    def test_v1_drives_both_classes_alike(self):
        fb, fly = make_fly(["DM1"], **plume_fly.V1)
        self.assertEqual(fly.jo_silent.size, 0)
        d = fly.drives(1.0, np.deg2rad(30))
        self.assertEqual(len(jo_indices(fb, d, "JO-C")), 4)    # the 5 JO-C cells minus the unsided one
        self.assertTrue(fly.describe()["jo_ce_coactivated"])


@unittest.skipIf(DOOR is None, "DoOR data not present")
class ShuffledMode(unittest.TestCase):
    """The matched control (v2, F): same drive computation, an angle drawn each step, heading ignored."""

    def test_draws_a_new_angle_in_range_each_step(self):
        fb, fly = make_fly(["DM1"])
        fly.reset_state(3)
        drawn = []
        for k in range(50):
            _, _, info = fly.step(0.5, 0.0, seed=k, wind_mode="shuffled")
            drawn.append(info["phi_drive"])
            self.assertEqual(info["phi"], 0.0)
            self.assertEqual(info["wind_mode"], "shuffled")
            self.assertTrue(info["wind_sense"])
            left, right = fly.wind_rates_sides(info["phi_drive"])
            self.assertAlmostEqual(info["jo_left_drive_hz"], left)
            self.assertAlmostEqual(info["jo_right_drive_hz"], right)
            self.assertAlmostEqual(fb.calls[-1]["drive"][tuple(fly.jo_left.tolist())], left)
        self.assertTrue(all(-np.pi <= p < np.pi for p in drawn))
        self.assertGreater(len(set(drawn)), 45)

    def test_ignores_the_heading(self):
        fb_a, a = make_fly(["DM1"])
        fb_b, b = make_fly(["DM1"])
        a.reset_state(3)
        b.reset_state(3)
        rng = np.random.default_rng(0)
        for k in range(20):
            _, _, ia = a.step(0.5, 0.0, seed=k, wind_mode="shuffled")
            _, _, ib = b.step(0.5, rng.uniform(-np.pi, np.pi), seed=k, wind_mode="shuffled")
            self.assertEqual(ia["phi_drive"], ib["phi_drive"])
            self.assertEqual(fb_a.calls[-1]["drive"], fb_b.calls[-1]["drive"])

    def test_reproducible_from_the_trial_seed_and_different_across_seeds(self):
        fb, fly = make_fly(["DM1"])
        def sequence(seed):
            fly.reset_state(seed)
            return [fly.step(0.0, 0.0, seed=k, wind_mode="shuffled")[2]["phi_drive"] for k in range(10)]
        self.assertEqual(sequence(3), sequence(3))
        self.assertNotEqual(sequence(3), sequence(4))

    def test_wind_mode_uses_the_true_angle(self):
        fb, fly = make_fly(["DM1"])
        for deg in (-60, 0, 120):
            _, _, info = fly.step(0.0, np.deg2rad(deg), seed=1)
            self.assertEqual(info["phi_drive"], info["phi"])
            self.assertEqual(info["wind_mode"], "wind")

    def test_statistics_match_the_wind_sense_over_uniform_headings(self):
        fb, fly = make_fly(["DM1"])
        fly.reset_state(11)
        left, right = [], []
        for k in range(3000):
            _, _, info = fly.step(0.0, 0.0, seed=k, wind_mode="shuffled")
            left.append(info["jo_left_drive_hz"])
            right.append(info["jo_right_drive_hz"])
        # the cosine's mean over uniform angles is 0.5, so the shuffled mean per cell is 0.5 x wind_hz x scale
        self.assertAlmostEqual(np.mean(left), 0.5 * fly.wind_hz * fly.scale_left, delta=0.03 * fly.wind_hz)
        self.assertAlmostEqual(np.mean(right), 0.5 * fly.wind_hz * fly.scale_right, delta=0.03 * fly.wind_hz)
        # while the direction information is zero: the drawn angle is not the heading
        self.assertGreater(np.std(left), 0.2 * fly.wind_hz)

    def test_wind_sense_false_is_the_none_mode_and_bad_modes_are_refused(self):
        fb, fly = make_fly(["DM1"])
        _, _, info = fly.step(0.0, np.deg2rad(90), seed=2, wind_sense=False)
        self.assertEqual(info["wind_mode"], "none")
        self.assertFalse(info["wind_sense"])
        self.assertEqual(info["jo_left_drive_hz"], 0.5 * fly.wind_hz * fly.scale_left)
        self.assertEqual(info["jo_right_drive_hz"], 0.5 * fly.wind_hz * fly.scale_right)
        self.assertEqual(info["orn_hz"], 0.0)
        with self.assertRaises(ValueError):
            fly.step(0.0, 0.0, seed=2, wind_mode="bogus")


class Smoothing(unittest.TestCase):
    """The command low-pass (v2, B): tau = 150 ms, three world steps, from rest."""

    def test_alpha_arithmetic(self):
        s = plume_fly.Smoother(0.150, 0.05)
        self.assertAlmostEqual(s.alpha, 1.0 - math.exp(-1.0 / 3.0))
        self.assertAlmostEqual(s.alpha, 0.2835, places=4)
        self.assertAlmostEqual(s.describe()["steps_per_tau"], 3.0)
        self.assertFalse(s.describe()["passthrough"])
        self.assertEqual(plume_fly.SMOOTH_TAU_S, 0.150)
        self.assertEqual(plume_fly.WORLD_DT_S, 0.05)

    def test_unit_step_response(self):
        s = plume_fly.Smoother(0.150, 0.05)
        for k in range(1, 13):
            y, _ = s.update(1.0, 0.0)
            self.assertAlmostEqual(y, 1.0 - math.exp(-k / 3.0), places=12)
            if k == 3:
                self.assertAlmostEqual(y, 0.632, places=3)

    def test_zero_tau_passes_through(self):
        s = plume_fly.Smoother(0.0, 0.05)
        self.assertEqual(s.alpha, 1.0)
        self.assertEqual(s.update(0.3, -0.2), (0.3, -0.2))
        self.assertEqual(s.update(-1.0, 0.7), (-1.0, 0.7))
        self.assertTrue(s.describe()["passthrough"])

    def test_reset_returns_to_rest(self):
        s = plume_fly.Smoother()
        s.update(1.0, 1.0)
        s.reset()
        self.assertEqual(tuple(s.y), (0.0, 0.0))
        self.assertAlmostEqual(s.update(1.0, 1.0)[0], s.alpha)

    def test_channel_count_and_dt_are_checked(self):
        with self.assertRaises(ValueError):
            plume_fly.Smoother(0.1, 0.0)
        with self.assertRaises(ValueError):
            plume_fly.Smoother().update(1.0)

    def test_step_returns_the_smoothed_command_and_records_the_raw_one(self):
        fb, fly = make_fly(["DM1"])
        fly.fb.run = hand_run(lambda k, n: np.full(n, 450.0 if k == "steer_R" else 0.0))
        a = fly.smoother.alpha
        fly.reset_state(1)
        turn, speed, info = fly.step(0.0, 0.0, seed=0)
        self.assertAlmostEqual(info["turn_raw"], 1.0)
        self.assertAlmostEqual(turn, a)
        self.assertAlmostEqual(info["turn"], a)
        turn, speed, info = fly.step(0.0, 0.0, seed=1)
        self.assertAlmostEqual(info["turn_raw"], 1.0)
        self.assertAlmostEqual(turn, 1.0 - (1.0 - a) ** 2)
        self.assertEqual(speed, 0.0)
        fly.reset_state(2)
        turn, _, _ = fly.step(0.0, 0.0, seed=0)
        self.assertAlmostEqual(turn, a)

    def test_v1_has_no_smoothing(self):
        fb, fly = make_fly(["DM1"], **plume_fly.V1)
        fly.fb.run = hand_run(lambda k, n: np.full(n, 450.0 if k == "steer_R" else 0.0))
        turn, _, info = fly.step(0.0, 0.0, seed=0)
        self.assertEqual(turn, 1.0)
        self.assertEqual(info["turn_raw"], 1.0)
        self.assertTrue(fly.describe()["smoothing"]["passthrough"])


@unittest.skipIf(DOOR is None, "DoOR data not present")
class StateCarry(unittest.TestCase):
    """The brain state is carried within a trial and dropped at reset_state() (v2, A)."""

    def test_state_is_carried_within_a_trial_and_dropped_at_reset(self):
        fb, fly = make_fly(["DM1"])
        fly.reset_state(5)
        infos = [fly.step(0.0, 0.0, seed=k)[2] for k in range(3)]
        self.assertIsNone(fb.calls[0]["state"])
        self.assertEqual(int(fb.calls[1]["state"]["v"][0]), 1)
        self.assertEqual(int(fb.calls[2]["state"]["v"][0]), 2)
        self.assertEqual([i["state_carried"] for i in infos], [False, True, True])
        self.assertTrue(fly.state_carried)
        fly.reset_state()
        self.assertFalse(fly.state_carried)
        fly.step(0.0, 0.0, seed=9)
        self.assertIsNone(fb.calls[3]["state"])

    def test_the_trial_seed_seeds_the_brain_and_later_seeds_are_only_recorded(self):
        fb, fly = make_fly(["DM1"])
        fly.reset_state(5)
        infos = [fly.step(0.0, 0.0, seed=s)[2] for s in (100, 101)]
        self.assertEqual([c["seed"] for c in fb.calls], [5, 5])
        self.assertEqual([i["seed"] for i in infos], [100, 101])
        self.assertEqual([i["trial_seed"] for i in infos], [5, 5])
        self.assertEqual(fly.trial_seed, 5)
        # without a seed at reset, the first step's seed names the trial
        fb.calls.clear()
        fly.reset_state()
        infos = [fly.step(0.0, 0.0, seed=s)[2] for s in (100, 101)]
        self.assertEqual([c["seed"] for c in fb.calls], [100, 100])
        self.assertEqual([i["trial_seed"] for i in infos], [100, 100])

    def test_a_fresh_fly_needs_no_reset(self):
        fb, fly = make_fly(["DM1"])
        _, _, info = fly.step(0.0, 0.0, seed=42)
        self.assertEqual(info["trial_seed"], 42)
        self.assertFalse(info["state_carried"])
        _, _, info = fly.step(0.0, 0.0, seed=43)
        self.assertTrue(info["state_carried"])

    def test_without_carry_every_step_is_fresh_v1(self):
        fb, fly = make_fly(["DM1"], **plume_fly.V1)
        fly.reset_state(5)
        infos = [fly.step(0.0, 0.0, seed=s)[2] for s in (100, 101)]
        self.assertEqual([c["state"] for c in fb.calls], [None, None])
        self.assertEqual([c["seed"] for c in fb.calls], [100, 101])
        self.assertEqual([i["state_carried"] for i in infos], [False, False])
        self.assertFalse(fly.state_carried)
        self.assertTrue(fly.describe()["memoryless"])

    def test_a_brain_that_cannot_carry_state_is_an_error_under_carry(self):
        fb, fly = make_fly(["DM1"])
        fly.fb.run = lambda drive, steps, gains=None, record=None, seed=0, state=None: {
            **{k: np.zeros(len(record[k])) for k in record}, "_fired": np.array([], dtype=np.int64)}
        with self.assertRaises(KeyError):
            fly.step(0.0, 0.0, seed=0)


class MotorConversions(unittest.TestCase):
    def rates(self, **kw):
        r = {k: 0.0 for k in plume_fly.MOTOR_NAMES}
        r.update(kw)
        return r

    def test_turn_sign_is_right_minus_left(self):
        turn, _, _ = plume_fly.PlumeFly.motor_from_rates(self.rates(steer_R=450.0))
        self.assertAlmostEqual(turn, 1.0)
        turn, _, _ = plume_fly.PlumeFly.motor_from_rates(self.rates(steer_L=225.0))
        self.assertAlmostEqual(turn, -0.5)
        turn, _, _ = plume_fly.PlumeFly.motor_from_rates(self.rates(steer_L=300.0, steer_R=300.0))
        self.assertEqual(turn, 0.0)

    def test_forward_is_the_mean_of_both_sides(self):
        _, speed, parts = plume_fly.PlumeFly.motor_from_rates(self.rates(fwd_L=450.0, fwd_R=0.0))
        self.assertAlmostEqual(parts["forward_n"], 0.5)
        self.assertAlmostEqual(speed, 0.5)
        _, speed, _ = plume_fly.PlumeFly.motor_from_rates(self.rates(fwd_L=450.0, fwd_R=450.0))
        self.assertAlmostEqual(speed, 1.0)

    def test_speed_is_clipped_to_unit(self):
        _, speed, _ = plume_fly.PlumeFly.motor_from_rates(self.rates(fwd_L=900.0, fwd_R=900.0))
        self.assertEqual(speed, 1.0)
        _, speed, _ = plume_fly.PlumeFly.motor_from_rates(self.rates(back=900.0))
        self.assertEqual(speed, -1.0)
        _, speed, _ = plume_fly.PlumeFly.motor_from_rates(self.rates(fwd_L=450.0, fwd_R=450.0, back=225.0))
        self.assertAlmostEqual(speed, 0.5)

    def test_stop_gates_speed(self):
        base = self.rates(fwd_L=450.0, fwd_R=450.0)
        _, speed, _ = plume_fly.PlumeFly.motor_from_rates(dict(base, stop=450.0))
        self.assertEqual(speed, 0.0)
        _, speed, _ = plume_fly.PlumeFly.motor_from_rates(dict(base, stop=225.0))
        self.assertAlmostEqual(speed, 0.5)
        _, speed, _ = plume_fly.PlumeFly.motor_from_rates(dict(base, stop=900.0))
        self.assertEqual(speed, 0.0)                   # stop past 450 Hz does not reverse
        _, speed, _ = plume_fly.PlumeFly.motor_from_rates(dict(base, back=900.0, stop=225.0))
        self.assertAlmostEqual(speed, -0.5)

    def test_matches_the_roamer_formula_on_random_rates(self):
        rng = np.random.default_rng(3)
        for _ in range(50):
            hz = {k: float(rng.uniform(0, 600)) for k in plume_fly.MOTOR_NAMES}
            turn, speed, _ = plume_fly.PlumeFly.motor_from_rates(hz)
            t = (hz["steer_R"] - hz["steer_L"]) / 450.0
            f = (hz["fwd_L"] + hz["fwd_R"]) / 2.0 / 450.0
            s = np.clip(f - hz["back"] / 450.0, -1, 1) * (1.0 - np.clip(hz["stop"] / 450.0, 0, 1))
            self.assertAlmostEqual(turn, t, places=12)
            self.assertAlmostEqual(speed, float(s), places=12)


@unittest.skipIf(DOOR is None, "DoOR data not present")
class Step(unittest.TestCase):
    def test_one_run_with_the_right_arguments(self):
        fb, fly = make_fly(["DM1", "VM7d"], sim_steps=60)
        gains = np.ones(fb.n_types, dtype=np.float32)
        fly.gains = gains
        turn, speed, info = fly.step(0.5, np.deg2rad(30), seed=17)
        self.assertEqual(len(fb.calls), 1)
        call = fb.calls[0]
        self.assertEqual(call["steps"], 60)
        self.assertEqual(call["seed"], 17)
        self.assertIsNone(call["state"])
        self.assertIs(call["gains"], gains)
        self.assertEqual(call["drive"], fly.drives(0.5, np.deg2rad(30)))
        for k in plume_fly.MOTOR_NAMES + ("orn", "jo_left", "jo_right", "jo_silent"):
            self.assertIn(k, call["record"])

    def test_default_window_is_120_steps(self):
        fb, fly = make_fly(["DM1"])
        fly.step(0.0, 0.0, seed=0)
        self.assertEqual(fb.calls[0]["steps"], 120)
        self.assertEqual(plume_fly.SIM_STEPS, 120)
        self.assertEqual(plume_fly.SIM_STEPS_V1, 60)
        self.assertEqual(plume_fly.V1["sim_steps"], 60)

    def test_info_carries_rates_and_the_echoed_inputs(self):
        fb, fly = make_fly(["DM1", "VM7d"])
        turn, speed, info = fly.step(1.0, np.deg2rad(30), seed=1)
        # the fake brain echoes drive as rate: motor cells get no drive, so they are silent
        for k in plume_fly.MOTOR_NAMES:
            self.assertEqual(info[k], 0.0)
        self.assertEqual((turn, speed), (0.0, 0.0))
        left, right = fly.wind_rates_sides(np.deg2rad(30))
        self.assertAlmostEqual(info["jo_left_hz"], left)
        self.assertAlmostEqual(info["jo_right_hz"], right)
        self.assertGreater(info["orn_hz"], 0.0)
        self.assertEqual(info["fired"], fly.orn.size + fly.jo_left.size + fly.jo_right.size)
        self.assertEqual(info["c"], 1.0)
        self.assertTrue(info["wind_sense"])
        for k in ("forward_n", "back_n", "stop_n", "turn", "speed", "turn_raw", "speed_raw", "seed",
                  "trial_seed", "phi", "phi_drive", "wind_mode", "state_carried", "jo_silent_hz",
                  "jo_left_cell_hz", "jo_right_cell_hz", "jo_total_cell_hz", "jo_total_drive_cell_hz"):
            self.assertIn(k, info)

    def test_hz_rates_survive_next_to_the_normalised_parts(self):
        fb, fly = make_fly(["DM1"], **plume_fly.V1)
        # drive the motor cells directly through a hand-made run so the Hz and /450 values differ
        fly.fb.run = hand_run(lambda k, n: np.full(n, 450.0) if k == "back" else np.zeros(n))
        turn, speed, info = fly.step(0.0, 0.0, seed=0)
        self.assertEqual(info["back"], 450.0)          # Hz, as recorded
        self.assertEqual(info["back_n"], 1.0)          # rate / 450
        self.assertEqual(info["stop"], 0.0)
        self.assertEqual(speed, -1.0)

    def test_describe_lists_the_constants(self):
        fb, fly = make_fly(["DM1", "VM7d"])
        d = fly.describe()
        self.assertEqual(d["odorant"], "ethyl acetate")
        self.assertEqual(d["odour_hz"], 200.0)
        self.assertEqual(d["wind_hz"], 100.0)
        self.assertEqual(d["sim_steps"], 120)
        self.assertEqual(d["brain_ms_per_world_step"], 24.0)
        self.assertEqual(d["world_dt_s"], 0.05)
        self.assertTrue(d["carry_state"])
        self.assertEqual(d["jo_all"], 8)
        self.assertEqual(d["jo_left"] + d["jo_right"] + d["jo_unsided"], 3)
        self.assertEqual(d["jo_silent"], 5)
        self.assertEqual(set(d["profile"]), {"DM1", "VM7d"})
        self.assertEqual(d["wind_modes"], ["wind", "shuffled", "none"])
        fb, fly = make_fly(["DM1", "VM7d"], **plume_fly.V1)
        d = fly.describe()
        self.assertEqual(d["sim_steps"], 60)
        self.assertEqual(d["jo_left"] + d["jo_right"] + d["jo_unsided"], 8)
        self.assertEqual(d["jo_silent"], 0)
        self.assertFalse(d["carry_state"])


class Quantisation(unittest.TestCase):
    """The readout's granularity is disclosed, and the numbers are right for each window."""

    def test_lif_step_matches_the_simulator(self):
        import flysim
        self.assertEqual(plume_fly.LIF_DT_MS, flysim.Params.dt)

    def test_one_spike_in_twelve_milliseconds_is_83_hz(self):
        q = plume_fly.readout_quanta(60)
        self.assertAlmostEqual(q["window_ms"], 12.0)
        self.assertAlmostEqual(q["rate_quantum_hz"], 1000.0 / 12.0)
        q = plume_fly.readout_quanta(120)
        self.assertAlmostEqual(q["window_ms"], 24.0)
        self.assertAlmostEqual(q["rate_quantum_hz"], 1000.0 / 24.0)

    def test_turn_quantum_with_one_steering_cell_per_side(self):
        fb, fly = make_fly(["DM1"])
        q = fly.describe()["readout_quanta"]
        self.assertAlmostEqual(q["turn_quantum"], (1000.0 / 24.0) / 450.0)            # 0.0926 (v2)
        # which the world turns into 0.0926 x 180 deg/s x 0.05 s = 0.83 deg per step, before smoothing
        self.assertAlmostEqual(q["turn_quantum"] * 180.0 * 0.05, 0.8333, places=3)
        self.assertAlmostEqual(q["forward_quantum"], (1000.0 / 24.0) / (2 * 450.0))
        self.assertAlmostEqual(q["back_quantum"], (1000.0 / 24.0) / (2 * 450.0))      # the fake MDN has 2 cells
        # v1's shorter window doubles every quantum: the floor belongs to the readout, not the fly
        q1 = plume_fly.readout_quanta(60, fly.motor)
        self.assertAlmostEqual(q1["turn_quantum"], 2 * q["turn_quantum"])
        self.assertAlmostEqual(q1["turn_quantum"] * 180.0 * 0.05, 1.6667, places=3)

    def test_a_step_moves_the_raw_turn_in_whole_quanta(self):
        fb, fly = make_fly(["DM1"], sim_steps=60)
        counts = {"steer_L": 2, "steer_R": 5}      # spikes in the window
        fly.fb.run = hand_run(lambda k, n: np.full(n, counts.get(k, 0) / 0.012))
        turn, speed, info = fly.step(0.0, 0.0, seed=0)
        quantum = fly.describe()["readout_quanta"]["turn_quantum"]
        self.assertAlmostEqual(info["turn_raw"], 3 * quantum)
        self.assertAlmostEqual(turn, fly.smoother.alpha * 3 * quantum)     # smoothed from rest


class PopulationImbalance(unittest.TestCase):
    """The rootSide split makes the summed drive asymmetric under v1; describe() says by how much."""

    def make(self, n_left, n_right, **kw):
        fb = FakeBrain(["DM1"], jo=(("JO-EV1", n_left + n_right),))
        side = np.array([""] * fb.n, dtype=object)
        jo = fb.where(type_re=plume_fly.JO_TYPE_RE)
        for j, i in enumerate(jo):
            side[i] = "L" if j < n_left else "R"
        return plume_fly.PlumeFly(fb, gains=None, motor=fake_motor(fb), root_side=side.astype(str), **kw)

    def test_equal_populations_give_a_symmetric_headwind_and_a_direction_free_control(self):
        for kw in ({}, plume_fly.V1):
            d = self.make(4, 4, **kw).describe()["jo_drive_totals"]
            self.assertEqual(d["population_ratio_left_over_right"], 1.0)
            self.assertAlmostEqual(d["cases"]["headwind"]["left_minus_right_cell_hz"], 0.0)
            self.assertAlmostEqual(d["cases"]["headwind"]["equivalent_phi_deg"], 0.0)
            self.assertAlmostEqual(d["cases"]["control_no_wind_sense"]["equivalent_phi_deg"], 0.0)
            self.assertAlmostEqual(d["cases"]["wind_from_left"]["equivalent_phi_deg"], 90.0)
            self.assertAlmostEqual(d["cases"]["wind_from_right"]["equivalent_phi_deg"], -90.0)

    def test_the_connectomes_203_to_132_split_is_a_31_degree_headwind_bias_under_v1(self):
        d = plume_fly.jo_drive_totals(203, 132, 100.0)
        hw = d["cases"]["headwind"]
        self.assertAlmostEqual(hw["per_cell_hz_left"], hw["per_cell_hz_right"])       # symmetric per cell
        self.assertAlmostEqual(hw["per_cell_hz_left"], 85.355, places=2)
        self.assertAlmostEqual(hw["total_cell_hz_left"], 203 * 85.355, delta=0.5)
        self.assertAlmostEqual(hw["total_cell_hz_right"], 132 * 85.355, delta=0.5)
        self.assertAlmostEqual(hw["left_minus_right_cell_hz"], 6060.2, delta=0.5)
        self.assertAlmostEqual(hw["equivalent_phi_deg"], 30.8, delta=0.2)
        ctl = d["cases"]["control_no_wind_sense"]
        self.assertAlmostEqual(ctl["left_minus_right_cell_hz"], 3550.0)
        self.assertAlmostEqual(ctl["equivalent_phi_deg"], 17.4, delta=0.2)
        self.assertAlmostEqual(d["population_ratio_left_over_right"], 203 / 132)
        self.assertEqual((d["scale_left"], d["scale_right"]), (1.0, 1.0))
        # a downwind-facing fly gets far less total drive than the control does
        self.assertAlmostEqual(d["cases"]["tailwind"]["total_cell_hz"], 335 * 14.645, delta=1.0)
        self.assertAlmostEqual(ctl["total_cell_hz"], 335 * 50.0)

    def test_step_records_the_summed_jo_drive(self):
        fly = self.make(5, 3, **plume_fly.V1)
        _, _, info = fly.step(0.0, 0.0, seed=1)                # the fake brain echoes drive as rate
        left, right = plume_fly.wind_rates(0.0, fly.wind_hz)
        self.assertAlmostEqual(info["jo_total_cell_hz"], 5 * left + 3 * right)
        _, _, info = fly.step(0.0, np.pi, seed=1, wind_sense=False)
        self.assertAlmostEqual(info["jo_total_cell_hz"], 8 * 50.0)
        fly = self.make(5, 3)
        _, _, info = fly.step(0.0, 0.0, seed=1)
        self.assertAlmostEqual(info["jo_left_cell_hz"], info["jo_right_cell_hz"])
        self.assertAlmostEqual(info["jo_total_cell_hz"], 8 * left)


class Docstring(unittest.TestCase):
    """The MEASURED/CHOSEN split: role attributions, the brain setting and every v2 change are filed as choices."""

    def test_roles_and_setting_are_chosen_not_measured(self):
        doc = plume_fly.__doc__
        measured = doc[doc.index("MEASURED"):doc.index("CHOSEN, and said so")]
        chosen = doc[doc.index("CHOSEN, and said so"):]
        for phrase in ("forward", "backward walking", "calibration.CHOSEN", "steering", "stays silent",
                       "150 ms", "shuffled", "carried", "the larger class", "133.5"):
            self.assertNotIn(phrase, measured, phrase)
            self.assertIn(phrase, chosen, phrase)
        for phrase in ("203", "132", "157", "110", "267", "68 cells"):
            self.assertIn(phrase, measured, phrase)
        for phrase in ("named in v1's own report", "41.7 Hz", "1.67 deg", "0.283", "uniformly on [-pi, pi)"):
            self.assertIn(phrase, doc, phrase)
        d = make_fly(["DM1"])[1].describe()
        for k in ("readout_quanta", "jo_drive_totals", "jo_ce_coactivated", "memoryless", "state", "lif_dt_ms",
                  "smoothing", "side_scale_left", "side_scale_right", "jo_driven_re", "jo_silent", "wind_modes"):
            self.assertIn(k, d)


def free_ram_gb():
    try:
        import subprocess
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        return float(out) / (1024 * 1024)
    except Exception:
        return float("nan")


@unittest.skipUnless(os.environ.get("PLUME_REAL_BRAIN") == "1", "set PLUME_REAL_BRAIN=1 to load the connectome")
class RealBrain(unittest.TestCase):
    """
    One brain, loaded once: the JO-E split is what the docstring says, both
    antennae get the same cell-Hz in a headwind, and with the state carried
    across 10 consecutive steps at c = 1 the motor rates are finite and a
    step is quick.
    """

    def test_jo_e_split_equalised_drive_and_ten_carried_steps(self):
        ram = free_ram_gb()
        print(f"\nfree RAM before load: {ram:.1f} GB")
        self.assertGreater(ram, 6.0, "need more than 6 GB free to load a FlyBrain")
        import calibration
        import flysim
        t0 = time.time()
        fb = flysim.FlyBrain()
        gains = calibration.gains_for(fb, calibration.CHOSEN)
        fly = plume_fly.PlumeFly(fb, gains=gains)
        d = fly.describe()
        print(f"load + setup: {time.time() - t0:.1f} s")
        print(f"JO-E left {d['jo_left']} right {d['jo_right']} unsided {d['jo_unsided']}; "
              f"JO-C silent {d['jo_silent']}; all {d['jo_all']}; scales L {d['side_scale_left']:.4f} "
              f"R {d['side_scale_right']:.4f}; headwind cell-Hz per side "
              f"{d['jo_drive_totals']['cases']['headwind']['total_cell_hz_left']:.0f} / "
              f"{d['jo_drive_totals']['cases']['headwind']['total_cell_hz_right']:.0f}")
        self.assertEqual((d["jo_left"], d["jo_right"], d["jo_unsided"]), (157, 110, 0))
        self.assertEqual(d["jo_silent"], 68)
        self.assertEqual(d["jo_all"], 335)
        self.assertEqual(d["jo_driven_types"], ["JO-ED1", "JO-ED2_a", "JO-ED2_b", "JO-ED2_c", "JO-EV1",
                                                "JO-EV2", "JO-EV3", "JO-EV4", "JO-EV5", "JO-EV6"])
        self.assertEqual(fly.orn_all.size, 2635)                 # every ORN_ cell
        self.assertGreater(fly.orn.size, 1000)

        fly.reset_state(0)
        times, infos = [], []
        for k in range(10):
            t = time.time()
            turn, speed, info = fly.step(1.0, 0.0, seed=k)
            times.append(time.time() - t)
            infos.append(info)
            print(f"step {k}: {times[-1]:.2f} s  carried={info['state_carried']}  turn={turn:+.3f} "
                  f"(raw {info['turn_raw']:+.3f}) speed={speed:+.3f} (raw {info['speed_raw']:+.3f})  "
                  f"orn={info['orn_hz']:.1f} Hz  jo cell-Hz L/R {info['jo_left_cell_hz']:.0f}/"
                  f"{info['jo_right_cell_hz']:.0f} (delivered {info['jo_left_drive_cell_hz']:.0f}/"
                  f"{info['jo_right_drive_cell_hz']:.0f})  jo-c={info['jo_silent_hz']:.1f} Hz  "
                  f"fired={info['fired']}  " + " ".join(f"{n}={info[n]:.0f}" for n in plume_fly.MOTOR_NAMES))
        print(f"mean step time: {np.mean(times):.2f} s (first {times[0]:.2f} s, rest {np.mean(times[1:]):.2f} s)")
        self.assertEqual([i["state_carried"] for i in infos], [False] + [True] * 9)
        for info in infos:
            self.assertGreater(info["orn_hz"], 0.0)
            self.assertGreater(info["jo_left_hz"], 0.0)
            self.assertGreater(info["jo_right_hz"], 0.0)
            self.assertAlmostEqual(info["jo_left_drive_cell_hz"], info["jo_right_drive_cell_hz"], places=6)
            for k in plume_fly.MOTOR_NAMES + ("turn", "speed", "turn_raw", "speed_raw"):
                self.assertTrue(np.isfinite(info[k]), k)
        self.assertLess(np.mean(times), 3.0)


@unittest.skipIf(DOOR is None, "DoOR data not present")
class ProtocolArgument(unittest.TestCase):
    """PlumeFly(protocol=...) names the coupling; explicit arguments win; begin_trial is reset_state."""

    def settings(self, fly):
        return dict(sim_steps=fly.sim_steps, jo_driven_re=fly.jo_driven_re, equalise_sides=fly.equalise_sides,
                    carry_state=fly.carry_state, smooth_tau_s=fly.smoother.tau_s)

    def test_named_protocols_resolve_to_their_settings(self):
        _, v2 = make_fly(["DM1"], protocol="v2")
        _, v1 = make_fly(["DM1"], protocol="v1")
        _, plain = make_fly(["DM1"])
        self.assertEqual(self.settings(v2), dict(plume_fly.V2))
        self.assertEqual(self.settings(v1), dict(plume_fly.V1))
        self.assertEqual(self.settings(plain), dict(plume_fly.V2))
        self.assertEqual((v2.protocol, v1.protocol, plain.protocol), ("v2", "v1", "v2"))
        self.assertEqual((v2.protocol_overrides, v1.protocol_overrides), ([], []))
        with self.assertRaises(ValueError):
            make_fly(["DM1"], protocol="v3")

    def test_the_v1_dict_without_a_name_is_recognised_as_v1(self):
        _, fly = make_fly(["DM1"], **plume_fly.V1)
        self.assertEqual(fly.protocol, "v1")
        self.assertEqual(fly.describe()["protocol"], "v1")

    def test_an_explicit_argument_wins_over_the_protocol_and_is_listed(self):
        _, fly = make_fly(["DM1"], protocol="v2", smooth_tau_s=0.0)
        self.assertTrue(fly.carry_state)
        self.assertEqual(fly.jo_driven_re, plume_fly.JO_DRIVEN_RE)
        self.assertTrue(fly.smoother.describe()["passthrough"])
        self.assertEqual(fly.protocol, "v2")
        self.assertEqual(fly.protocol_overrides, ["smooth_tau_s"])
        d = fly.describe()
        self.assertEqual((d["protocol"], d["protocol_overrides"]), ("v2", ["smooth_tau_s"]))
        # the runner's use: a v2 fly that passes its commands through, so the
        # runner's low-pass is the only one
        fly.fb.run = hand_run(lambda k, n: np.full(n, 450.0 if k == "steer_R" else 0.0))
        fly.reset_state(1)
        turn, _, info = fly.step(0.0, 0.0, seed=0)
        self.assertEqual((turn, info["turn_raw"]), (1.0, 1.0))
        _, custom = make_fly(["DM1"], protocol="v1", carry_state=True)
        self.assertEqual((custom.protocol, custom.protocol_overrides), ("v1", ["carry_state"]))
        _, unnamed = make_fly(["DM1"], sim_steps=60)
        self.assertEqual(unnamed.protocol, "custom")

    def test_begin_trial_is_reset_state(self):
        fb, fly = make_fly(["DM1"])
        self.assertIs(plume_fly.PlumeFly.begin_trial, plume_fly.PlumeFly.reset_state)
        fly.step(0.0, 0.0, seed=3)
        self.assertTrue(fly.state_carried)
        fly.begin_trial(9)
        self.assertFalse(fly.state_carried)
        self.assertEqual(fly.trial_seed, 9)
        self.assertEqual(tuple(fly.smoother.y), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
