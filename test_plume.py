"""
The wind tunnel without the brain: every constant is checked against what the
world actually does, and the sign conventions are pinned down so the coupling
layer cannot drift.

  py -m pytest -q test_plume.py
"""
import math
import time
import unittest

import numpy as np

import plume
from plume import World

SX, SY = plume.SOURCE


def run(world, turn, speed, n):
    return [world.step(turn, speed) for _ in range(n)]


def field_trace(seed, x, y, steps):
    """The plume at a fixed point over `steps` world steps (fly not involved)."""
    w = World(seed=seed)
    out = []
    for _ in range(steps):
        w._advance_plume()
        out.append(float(w.field(x, y)))
    return np.array(out)


class TestDeterminism(unittest.TestCase):

    def test_same_seed_same_trial(self):
        a, b = World(seed=3), World(seed=3)
        self.assertEqual(a.start, b.start)
        self.assertEqual(run(a, 0.3, 0.7, 120), run(b, 0.3, 0.7, 120))

    def test_different_seed_differs(self):
        a, b = World(seed=3), World(seed=4)
        self.assertNotEqual((a.start["y"], a.start["heading"]),
                            (b.start["y"], b.start["heading"]))

    def test_odour_off_keeps_the_same_start_and_wind(self):
        # paired conditions: the blank world is the same world with c masked
        on, off = World(seed=5, odour=True), World(seed=5, odour=False)
        self.assertEqual((on.start["y"], on.start["heading"]),
                         (off.start["y"], off.start["heading"]))
        self.assertEqual(off.start["c"], 0.0)
        self.assertGreater(float(off.field(SX + 0.1, SY)), 0.0)
        self.assertEqual(float(off.concentration(SX + 0.1, SY)), 0.0)
        ra, rb = run(on, 0.1, 1.0, 50), run(off, 0.1, 1.0, 50)
        self.assertEqual([r["x"] for r in ra], [r["x"] for r in rb])
        self.assertTrue(all(r["c"] == 0.0 for r in rb))


class TestPlume(unittest.TestCase):

    def test_start_pose_is_in_the_stated_ranges(self):
        for seed in range(20):
            s = World(seed=seed).start
            self.assertEqual(s["x"], plume.START_X)
            self.assertTrue(plume.START_Y[0] <= s["y"] <= plume.START_Y[1])
            self.assertTrue(-math.pi < s["heading"] <= 2 * math.pi)
            self.assertEqual(s["t"], 0.0)
            self.assertFalse(s["reached"])

    def test_derived_constants(self):
        # D from the width requirement: sd 0.03 m at 0.3 m downwind
        t = 0.3 / plume.WIND_SPEED
        self.assertAlmostEqual(math.sqrt(plume.R0 ** 2 + 2 * plume.DIFFUSION * t), 0.03, places=9)
        self.assertGreater(plume.Q, 0.0)
        self.assertEqual(plume.RELEASE_EVERY, 1)         # one puff per 0.05 s step
        self.assertEqual(plume.WARMUP_STEPS, 120)
        self.assertGreater(plume.WARMUP_S, plume.TRANSIT_S)

    def test_centreline_reads_one_at_ten_centimetres(self):
        # the calibration target, on the steady meander-free plume
        self.assertAlmostEqual(float(plume.straight_plume_concentration(SX + 0.1, SY)), 1.0, places=9)
        # with the meander on the reading wanders below 1 but stays about 1
        for seed in range(3):
            tr = field_trace(seed, SX + 0.1, SY, 400)
            self.assertLess(tr.max(), 1.1)
            self.assertGreater(np.median(tr), 0.5)

    def test_width_at_thirty_centimetres(self):
        c0 = float(plume.straight_plume_concentration(SX + 0.3, SY))
        c1 = float(plume.straight_plume_concentration(SX + 0.3, SY + 0.03))
        # one sd off the centreline reads exp(-1/2) of the peak
        self.assertAlmostEqual(c1 / c0, math.exp(-0.5), delta=0.02)

    def test_falls_off_crosswind_and_downwind(self):
        cl = float(plume.straight_plume_concentration(SX + 0.1, SY))
        offsets = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08]
        cs = [float(plume.straight_plume_concentration(SX + 0.1, SY + d)) for d in offsets]
        self.assertTrue(all(a > b for a, b in zip(cs, cs[1:])), cs)
        cs = [float(plume.straight_plume_concentration(SX + 0.1, SY - d)) for d in offsets]
        self.assertTrue(all(a > b for a, b in zip(cs, cs[1:])), cs)
        self.assertLess(cs[-1], 0.05 * cl)
        downwind = [float(plume.straight_plume_concentration(SX + d, SY)) for d in (0.1, 0.2, 0.3, 0.4, 0.5)]
        self.assertTrue(all(a > b for a, b in zip(downwind, downwind[1:])), downwind)
        # essentially nothing upwind of the source
        self.assertLess(float(plume.straight_plume_concentration(SX - 0.02, SY)), 0.01)

    def test_intermittent_at_thirty_centimetres(self):
        # MEASURED with the chosen meander (sd 0.02 m/s, tau 1 s): on the
        # centreline 0.3 m downwind the plume drops out about once per 10 s
        # (7 upward and 10 downward crossings of 0.05 pooled over eight 10 s
        # windows, in the plume 92% of the time). That is intermittent, but
        # milder than the design's "a few times per window"; the constants
        # were left as specified rather than tuned to the test.
        ups = downs = 0
        seen_high = seen_low = 0
        for seed in range(8):
            tr = field_trace(seed, SX + 0.3, SY, 200)      # 10 s
            above = tr > plume.ODOUR_THRESHOLD
            ups += int(np.sum(~above[:-1] & above[1:]))
            downs += int(np.sum(above[:-1] & ~above[1:]))
            seen_high += int(tr.max() > 0.3)
            seen_low += int(tr.min() < plume.ODOUR_THRESHOLD)
        self.assertGreaterEqual(ups, 4)
        self.assertGreaterEqual(downs, 4)
        self.assertEqual(seen_high, 8)                     # every window has real plume
        self.assertGreaterEqual(seen_low, 5)               # most windows also lose it

    def test_puffs_leave_the_arena(self):
        w = World(seed=0)
        self.assertLessEqual(w.n_puffs, int(plume.TRANSIT_S / plume.DT) + 2)
        self.assertGreater(w.n_puffs, 50)

    def test_concentration_accepts_arrays(self):
        w = World(seed=0)
        xs = np.linspace(0, plume.ARENA_X, 7)
        ys = np.full(7, SY)
        c = w.concentration(xs, ys)
        self.assertEqual(c.shape, (7,))
        self.assertTrue(np.all(c >= 0))
        xg, yg, C = w.snapshot(nx=30, ny=15)
        self.assertEqual(C.shape, (15, 30))
        self.assertGreater(C.max(), 0.5)


class TestWind(unittest.TestCase):

    def test_headwind_is_zero(self):
        self.assertAlmostEqual(float(plume.wind_direction_relative(math.pi)), 0.0)
        self.assertAlmostEqual(float(World.wind_direction_relative(-math.pi)), 0.0)

    def test_crosswind_signs(self):
        # heading +y: the wind (from -x) comes from the fly's LEFT: +pi/2
        self.assertAlmostEqual(float(plume.wind_direction_relative(math.pi / 2)), math.pi / 2)
        # heading -y: from the right: -pi/2
        self.assertAlmostEqual(float(plume.wind_direction_relative(-math.pi / 2)), -math.pi / 2)
        # tailwind: |phi| = pi
        self.assertAlmostEqual(abs(float(plume.wind_direction_relative(0.0))), math.pi)

    def test_wraps_and_broadcasts(self):
        phi = plume.wind_direction_relative(np.array([math.pi, math.pi / 2, 3 * math.pi]))
        np.testing.assert_allclose(phi, [0.0, math.pi / 2, 0.0], atol=1e-12)
        self.assertTrue(np.all(phi > -math.pi) and np.all(phi <= math.pi))


class TestFly(unittest.TestCase):

    def test_step_moves_by_speed_vmax_dt(self):
        w = World(seed=1, odour=False)
        w.x, w.y, w.heading = 0.3, 0.15, math.pi / 4
        s = w.step(0.0, 1.0)
        d = plume.V_MAX * plume.DT
        self.assertAlmostEqual(s["x"], 0.3 + d * math.cos(math.pi / 4))
        self.assertAlmostEqual(s["y"], 0.15 + d * math.sin(math.pi / 4))
        self.assertAlmostEqual(s["heading"], math.pi / 4)
        self.assertAlmostEqual(s["vx"], plume.V_MAX * math.cos(math.pi / 4))
        self.assertAlmostEqual(s["vy"], plume.V_MAX * math.sin(math.pi / 4))
        self.assertAlmostEqual(s["t"], plume.DT)
        # half speed, and speed is clipped to 1
        w.x, w.y, w.heading = 0.3, 0.15, 0.0
        self.assertAlmostEqual(w.step(0.0, 0.5)["x"], 0.3 + 0.5 * d)
        w.x = 0.3
        self.assertAlmostEqual(w.step(0.0, 5.0)["x"], 0.3 + d)
        # negative speed backs up
        w.x = 0.3
        self.assertAlmostEqual(w.step(0.0, -1.0)["x"], 0.3 - d)

    def test_turn_rate_and_sign(self):
        w = World(seed=1, odour=False)
        w.x, w.y, w.heading = 0.3, 0.15, math.pi
        rate = math.radians(180.0) * plume.DT           # 9 degrees per step
        s = w.step(1.0, 0.0)                            # full right turn
        self.assertAlmostEqual(s["heading"], math.pi - rate)   # clockwise
        self.assertEqual(s["x"], 0.3)                   # no walking
        s = w.step(-0.5, 0.0)                           # half left turn
        self.assertAlmostEqual(s["heading"], math.pi - rate + 0.5 * rate)
        s = w.step(-3.0, 0.0)                           # clipped to 1
        self.assertAlmostEqual(s["heading"], plume.wrap_angle(math.pi - 0.5 * rate + rate))
        # heading stays wrapped to (-pi, pi]
        for _ in range(60):
            s = w.step(1.0, 0.0)
            self.assertTrue(-math.pi < s["heading"] <= math.pi)

    def test_walls_clamp_and_count(self):
        w = World(seed=1, odour=False)
        w.x, w.y, w.heading = 0.3, plume.ARENA_Y - 0.0025, math.pi / 2   # walking +y
        states = run(w, 0.0, 1.0, 6)
        self.assertEqual(states[-1]["y"], plume.ARENA_Y)
        # 2.5 free steps then clamped: contacts counted per clamped step
        self.assertEqual(states[1]["wall_contacts"], 0)
        self.assertEqual(states[-1]["wall_contacts"], 4)
        w.x, w.heading = 0.0005, math.pi                                  # half a step from the upwind wall
        s = w.step(0.0, 1.0)
        self.assertEqual(s["x"], 0.0)
        self.assertEqual(s["wall_contacts"], 5)
        self.assertTrue(0.0 <= s["x"] <= plume.ARENA_X and 0.0 <= s["y"] <= plume.ARENA_Y)

    def test_reached_within_three_centimetres(self):
        w = World(seed=2, odour=False)
        w.x, w.y, w.heading = SX + 0.031, SY, math.pi        # 3.1 cm downwind, heading upwind
        s = w.step(0.0, 1.0)                                 # moves 1 mm closer
        self.assertTrue(s["reached"])
        self.assertTrue(s["done"])
        w2 = World(seed=2, odour=False)
        w2.x, w2.y, w2.heading = SX + 0.05, SY, 0.0          # 5 cm away, heading away
        self.assertFalse(w2.step(0.0, 1.0)["reached"])

    def test_done_after_trial_steps(self):
        w = World(seed=2, odour=False)
        states = run(w, 0.0, 0.0, plume.TRIAL_STEPS)
        self.assertFalse(states[-2]["done"])
        self.assertTrue(states[-1]["done"])
        self.assertAlmostEqual(states[-1]["t"], plume.TRIAL_STEPS * plume.DT)

    def test_log_has_one_row_per_step(self):
        w = World(seed=6)
        self.assertEqual(len(w.log), 0)
        run(w, 0.2, 0.8, 37)
        self.assertEqual(len(w.log), 37)
        tr = w.trajectory()
        for k in ("t", "x", "y", "heading", "c", "vx", "vy", "turn", "speed"):
            self.assertEqual(tr[k].shape, (37,))
        np.testing.assert_allclose(tr["t"], plume.DT * np.arange(1, 38))
        self.assertEqual(w.state, w.log[-1])
        self.assertEqual(w.start["step"], 0)

    def test_state_carries_the_nose_and_antennae_inputs(self):
        w = World(seed=7)
        s = w.step(0.0, 1.0)
        self.assertAlmostEqual(s["c"], float(w.concentration(s["x"], s["y"])))
        self.assertAlmostEqual(s["phi"], float(plume.wind_direction_relative(s["heading"])))

    def test_full_trial_is_fast(self):
        w = World(seed=0)
        t0 = time.perf_counter()
        run(w, 0.1, 1.0, plume.TRIAL_STEPS)
        self.assertLess(time.perf_counter() - t0, 0.5)


class TestTrialDesign(unittest.TestCase):
    """
    Consequences of the chosen constants, pinned as arithmetic so the outputs
    cannot be misread as properties of the brain: the source is only just
    reachable, and the start band lies inside the plume.
    """

    def test_source_needs_92_percent_of_top_speed_straight_upwind(self):
        self.assertAlmostEqual(plume.TRIAL_S, 20.0)
        self.assertAlmostEqual(plume.REACH_MIN_S, 18.5)
        self.assertAlmostEqual(plume.REACH_MIN_SPEED, 0.925)
        self.assertLess(plume.REACH_MIN_S, plume.TRIAL_S)        # reachable, but only just

    def test_start_band_lies_inside_the_straight_plume(self):
        d = plume.start_band_straight_plume()
        self.assertAlmostEqual(d["c_centre"], 0.517, places=2)
        self.assertAlmostEqual(d["c_edges"][0], 0.181, places=2)
        self.assertAlmostEqual(d["c_edges"][0], d["c_edges"][1], places=9)
        self.assertGreater(d["c_min"], plume.ODOUR_THRESHOLD)
        self.assertEqual(d["fraction_of_band_above_threshold"], 1.0)
        self.assertAlmostEqual(d["plume_sd_m_at_start_x"], 0.0345, delta=0.001)
        self.assertAlmostEqual(d["reach_min_speed_fraction"], 0.925)
        # and so, with the warm-up, most seeds are born in odour
        above = sum(World(seed).start["c"] > plume.ODOUR_THRESHOLD for seed in range(20))
        self.assertGreaterEqual(above, 10)


if __name__ == "__main__":
    unittest.main()
