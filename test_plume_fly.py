"""
The plume coupling on a fake brain: DoOR is read from disk, the connectome is
not. One optional class loads the real brain once, and only when asked:

  py -m pytest -q test_plume_fly.py
  PLUME_REAL_BRAIN=1 py -m pytest -q -s test_plume_fly.py -k RealBrain
"""
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


class FakeBrain:
    """
    Just enough of FlyBrain: types, bodies, where(type_re=) and a run() that
    echoes each recorded neuron's drive rate back as its firing rate, so the
    plumbing from drive to readout can be checked exactly.
    """

    def __init__(self, glomeruli, per=2, jo=(("JO-CA1", 3), ("JO-EV1", 3), ("JO-CM", 2))):
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

    def run(self, drive, steps, gains=None, record=None, seed=0):
        self.calls.append(dict(drive=drive, steps=steps, gains=gains, record=record, seed=seed))
        per_neuron = np.zeros(self.n)
        for k, r in drive.items():
            per_neuron[list(k)] = r
        out = {name: per_neuron[np.asarray(idx, dtype=np.int64)] for name, idx in (record or {}).items()}
        out["_fired"] = np.flatnonzero(per_neuron > 0)
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


def make_fly(glomeruli=None, **kw):
    if glomeruli is None:
        glomeruli = sorted(set(DOOR.glomerulus_of.values()))
    fb = FakeBrain(glomeruli)
    fly = plume_fly.PlumeFly(fb, gains=None, motor=fake_motor(fb), root_side=fake_root_side(fb), **kw)
    return fb, fly


def orn_rates(fb, drive):
    """{neuron index: rate} over ORN cells in a drive dict."""
    out = {}
    for k, v in drive.items():
        for i in k:
            if fb.types[i].startswith("ORN_"):
                out[int(i)] = float(v)
    return out


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
    def test_left_and_right_jo_get_their_own_rates(self):
        fb, fly = make_fly(["DM1", "VM7d"])
        d = fly.wind_drive(np.deg2rad(60))
        left, right = plume_fly.wind_rates(np.deg2rad(60), fly.wind_hz)
        self.assertEqual(d[tuple(fly.jo_left.tolist())], left)
        self.assertEqual(d[tuple(fly.jo_right.tolist())], right)
        self.assertGreater(left, right)

    def test_jo_cells_without_rootside_get_no_drive(self):
        fb, fly = make_fly(["DM1"])
        jo = fb.where(type_re=plume_fly.JO_TYPE_RE)
        self.assertEqual(len(jo), 8)
        self.assertEqual(fly.jo_unsided.size, 1)
        self.assertEqual(fly.jo_left.size + fly.jo_right.size + fly.jo_unsided.size, len(jo))
        driven = {i for k in fly.drives(1.0, 0.3) for i in k}
        for i in fly.jo_unsided:
            self.assertNotIn(int(i), driven)
        for i in np.concatenate([fly.jo_left, fly.jo_right]):
            self.assertIn(int(i), driven)

    def test_no_wind_sense_control_is_tonic_and_symmetric(self):
        fb, fly = make_fly(["DM1"])
        for deg in (-90, 0, 60):
            d = fly.wind_drive(np.deg2rad(deg), wind_sense=False)
            self.assertEqual(d[tuple(fly.jo_left.tolist())], 0.5 * fly.wind_hz)
            self.assertEqual(d[tuple(fly.jo_right.tolist())], 0.5 * fly.wind_hz)

    def test_drive_keys_do_not_overlap(self):
        fb, fly = make_fly()
        d = fly.drives(1.0, 0.5)
        seen = []
        for k in d:
            seen.extend(k)
        self.assertEqual(len(seen), len(set(seen)))


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
        self.assertIs(call["gains"], gains)
        self.assertEqual(call["drive"], fly.drives(0.5, np.deg2rad(30)))
        for k in plume_fly.MOTOR_NAMES + ("orn", "jo_left", "jo_right"):
            self.assertIn(k, call["record"])

    def test_info_carries_rates_and_the_echoed_inputs(self):
        fb, fly = make_fly(["DM1", "VM7d"])
        turn, speed, info = fly.step(1.0, np.deg2rad(30), seed=1)
        # the fake brain echoes drive as rate: motor cells get no drive, so they are silent
        for k in plume_fly.MOTOR_NAMES:
            self.assertEqual(info[k], 0.0)
        self.assertEqual((turn, speed), (0.0, 0.0))
        left, right = plume_fly.wind_rates(np.deg2rad(30), fly.wind_hz)
        self.assertAlmostEqual(info["jo_left_hz"], left)
        self.assertAlmostEqual(info["jo_right_hz"], right)
        self.assertGreater(info["orn_hz"], 0.0)
        self.assertEqual(info["fired"], fly.orn.size + fly.jo_left.size + fly.jo_right.size)
        self.assertEqual(info["c"], 1.0)
        self.assertTrue(info["wind_sense"])
        for k in ("forward_n", "back_n", "stop_n", "turn", "speed", "seed"):
            self.assertIn(k, info)

    def test_hz_rates_survive_next_to_the_normalised_parts(self):
        fb, fly = make_fly(["DM1"])
        # drive the motor cells directly through a hand-made run so the Hz and /450 values differ
        fly.fb.run = lambda drive, steps, gains=None, record=None, seed=0: {
            **{k: np.full(len(record[k]), 450.0) if k == "back" else np.zeros(len(record[k])) for k in record},
            "_fired": np.array([], dtype=np.int64)}
        turn, speed, info = fly.step(0.0, 0.0, seed=0)
        self.assertEqual(info["back"], 450.0)          # Hz, as recorded
        self.assertEqual(info["back_n"], 1.0)          # rate / 450
        self.assertEqual(info["stop"], 0.0)
        self.assertEqual(speed, -1.0)

    def test_no_wind_sense_reaches_the_brain(self):
        fb, fly = make_fly(["DM1"])
        _, _, info = fly.step(0.0, np.deg2rad(90), seed=2, wind_sense=False)
        self.assertEqual(info["jo_left_hz"], 0.5 * fly.wind_hz)
        self.assertEqual(info["jo_right_hz"], 0.5 * fly.wind_hz)
        self.assertEqual(info["orn_hz"], 0.0)
        self.assertFalse(info["wind_sense"])

    def test_describe_lists_the_constants(self):
        fb, fly = make_fly(["DM1", "VM7d"])
        d = fly.describe()
        self.assertEqual(d["odorant"], "ethyl acetate")
        self.assertEqual(d["odour_hz"], 200.0)
        self.assertEqual(d["wind_hz"], 100.0)
        self.assertEqual(d["sim_steps"], 60)
        self.assertEqual(d["jo_left"] + d["jo_right"] + d["jo_unsided"], 8)
        self.assertEqual(set(d["profile"]), {"DM1", "VM7d"})


class Quantisation(unittest.TestCase):
    """The readout's granularity is disclosed, and the numbers are right for this window."""

    def test_lif_step_matches_the_simulator(self):
        import flysim
        self.assertEqual(plume_fly.LIF_DT_MS, flysim.Params.dt)

    def test_one_spike_in_twelve_milliseconds_is_83_hz(self):
        q = plume_fly.readout_quanta(60)
        self.assertAlmostEqual(q["window_ms"], 12.0)
        self.assertAlmostEqual(q["rate_quantum_hz"], 1000.0 / 12.0)

    def test_turn_quantum_with_one_steering_cell_per_side(self):
        fb, fly = make_fly(["DM1"])
        q = fly.describe()["readout_quanta"]
        self.assertAlmostEqual(q["turn_quantum"], (1000.0 / 12.0) / 450.0)            # 0.185
        # which the world turns into 0.185 x 180 deg/s x 0.05 s = 1.67 deg per step
        self.assertAlmostEqual(q["turn_quantum"] * 180.0 * 0.05, 1.6667, places=3)
        self.assertAlmostEqual(q["forward_quantum"], (1000.0 / 12.0) / (2 * 450.0))
        self.assertAlmostEqual(q["back_quantum"], (1000.0 / 12.0) / (2 * 450.0))      # the fake MDN has 2 cells
        # a longer window lowers every quantum: the floor belongs to the readout, not the fly
        self.assertAlmostEqual(plume_fly.readout_quanta(120, fly.motor)["turn_quantum"], q["turn_quantum"] / 2)

    def test_a_step_moves_the_turn_in_whole_quanta(self):
        fb, fly = make_fly(["DM1"])
        counts = {"steer_L": 2, "steer_R": 5}      # spikes in the window
        fly.fb.run = lambda drive, steps, gains=None, record=None, seed=0: {
            **{k: np.full(len(record[k]), counts.get(k, 0) / 0.012) for k in record},
            "_fired": np.array([], dtype=np.int64)}
        turn, speed, info = fly.step(0.0, 0.0, seed=0)
        self.assertAlmostEqual(turn, 3 * fly.describe()["readout_quanta"]["turn_quantum"])


class PopulationImbalance(unittest.TestCase):
    """The rootSide split makes the summed drive asymmetric; describe() says by how much."""

    def make(self, n_left, n_right):
        fb = FakeBrain(["DM1"], jo=(("JO-CA1", n_left + n_right),))
        side = np.array([""] * fb.n, dtype=object)
        jo = fb.where(type_re=plume_fly.JO_TYPE_RE)
        for j, i in enumerate(jo):
            side[i] = "L" if j < n_left else "R"
        return plume_fly.PlumeFly(fb, gains=None, motor=fake_motor(fb), root_side=side.astype(str))

    def test_equal_populations_give_a_symmetric_headwind_and_a_direction_free_control(self):
        d = self.make(4, 4).describe()["jo_drive_totals"]
        self.assertEqual(d["population_ratio_left_over_right"], 1.0)
        self.assertAlmostEqual(d["cases"]["headwind"]["left_minus_right_cell_hz"], 0.0)
        self.assertAlmostEqual(d["cases"]["headwind"]["equivalent_phi_deg"], 0.0)
        self.assertAlmostEqual(d["cases"]["control_no_wind_sense"]["equivalent_phi_deg"], 0.0)
        self.assertAlmostEqual(d["cases"]["wind_from_left"]["equivalent_phi_deg"], 90.0)
        self.assertAlmostEqual(d["cases"]["wind_from_right"]["equivalent_phi_deg"], -90.0)

    def test_the_connectomes_203_to_132_split_is_a_31_degree_headwind_bias(self):
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
        # a downwind-facing fly gets far less total drive than the control does
        self.assertAlmostEqual(d["cases"]["tailwind"]["total_cell_hz"], 335 * 14.645, delta=1.0)
        self.assertAlmostEqual(ctl["total_cell_hz"], 335 * 50.0)

    def test_step_records_the_summed_jo_drive(self):
        fly = self.make(5, 3)
        _, _, info = fly.step(0.0, 0.0, seed=1)                # the fake brain echoes drive as rate
        left, right = plume_fly.wind_rates(0.0, fly.wind_hz)
        self.assertAlmostEqual(info["jo_total_cell_hz"], 5 * left + 3 * right)
        _, _, info = fly.step(0.0, np.pi, seed=1, wind_sense=False)
        self.assertAlmostEqual(info["jo_total_cell_hz"], 8 * 50.0)


class Docstring(unittest.TestCase):
    """The MEASURED/CHOSEN split: role attributions and the brain setting are filed as choices."""

    def test_roles_and_setting_are_chosen_not_measured(self):
        doc = plume_fly.__doc__
        measured = doc[doc.index("MEASURED"):doc.index("CHOSEN, and said so")]
        chosen = doc[doc.index("CHOSEN, and said so"):]
        for phrase in ("forward", "backward walking", "calibration.CHOSEN", "steering"):
            self.assertNotIn(phrase, measured, phrase)
            self.assertIn(phrase, chosen, phrase)
        for phrase in ("203", "132", "restarts every neuron from rest", "83.3 Hz", "driven identically",
                       "uniformly distributed"):
            self.assertIn(phrase, doc, phrase)
        d = make_fly(["DM1"])[1].describe()
        for k in ("readout_quanta", "jo_drive_totals", "jo_ce_coactivated", "memoryless", "lif_dt_ms"):
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
    """One brain, loaded once: at c = 1 in a headwind the sensory populations fire and the motor rates are finite."""

    def test_sensory_populations_fire_and_a_step_is_quick(self):
        ram = free_ram_gb()
        print(f"\nfree RAM before load: {ram:.1f} GB")
        self.assertGreater(ram, 6.0, "need more than 6 GB free to load a FlyBrain")
        import calibration
        import flysim
        t0 = time.time()
        fb = flysim.FlyBrain()
        gains = calibration.gains_for(fb, calibration.CHOSEN)
        fly = plume_fly.PlumeFly(fb, gains=gains)
        print(f"load + setup: {time.time() - t0:.1f} s; {fly.describe()}")
        self.assertEqual(fly.jo_left.size + fly.jo_right.size + fly.jo_unsided.size, 335)
        self.assertEqual(fly.orn_all.size, 2635)                 # every ORN_ cell
        self.assertEqual(fly.orn.size, sum(fly.nose.orn[g].size for g in fly.profile))
        self.assertGreater(fly.orn.size, 1000)

        fly.step(1.0, 0.0, seed=0)                     # warm-up
        times, infos = [], []
        for seed in range(3):
            t = time.time()
            turn, speed, info = fly.step(1.0, 0.0, seed=seed)
            times.append(time.time() - t)
            infos.append(info)
            print(f"seed {seed}: {times[-1]:.2f} s  turn={turn:+.3f} speed={speed:+.3f}  "
                  f"orn={info['orn_hz']:.1f} Hz jo_L={info['jo_left_hz']:.1f} jo_R={info['jo_right_hz']:.1f} Hz  "
                  f"fired={info['fired']}  "
                  + " ".join(f"{k}={info[k]:.0f}" for k in plume_fly.MOTOR_NAMES))
        print(f"mean step time: {np.mean(times):.2f} s")
        for info in infos:
            self.assertGreater(info["orn_hz"], 0.0)
            self.assertGreater(info["jo_left_hz"], 0.0)
            self.assertGreater(info["jo_right_hz"], 0.0)
            for k in plume_fly.MOTOR_NAMES + ("turn", "speed"):
                self.assertTrue(np.isfinite(info[k]), k)
        self.assertLess(np.mean(times), 2.0)


if __name__ == "__main__":
    unittest.main()
