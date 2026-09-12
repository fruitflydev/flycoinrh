"""
The plume runner without the brain and without the plume: hand-made trials
where every answer is known, and fakes for the world and the fly.

  py -m pytest -q test_plume_experiment.py
"""
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

import plume_experiment as pe

DT = 0.05


def make_trial(x, y, heading, c, seed=0, condition="odour", reached=False, walls=0,
               turn=None, speed=None, rates=None):
    x, y, heading, c = (np.asarray(a, float) for a in (x, y, heading, c))
    n = len(x) - 1
    return {
        "seed": seed, "condition": condition, "wind_sense": True, "dt": DT,
        "steps": n, "steps_run": n, "reached": reached, "wall_contacts": walls,
        "t": np.arange(len(x)) * DT, "x": x, "y": y, "heading": heading, "c": c,
        "phi": np.zeros(n), "turn": np.zeros(n) if turn is None else np.asarray(turn, float),
        "speed": np.ones(n) if speed is None else np.asarray(speed, float),
        "rates": rates or {}, "elapsed_s": 0.0, "sec_per_step": 0.0,
    }


def straight_trial(progress, n=40, seed=0, condition="odour", c=None):
    """A fly walking straight upwind by `progress` metres in n steps, no odour unless given."""
    x = 0.45 - progress * np.arange(n + 1) / n
    y = np.full(n + 1, 0.15)
    h = np.full(n + 1, math.pi)
    return make_trial(x, y, h, np.zeros(n + 1) if c is None else c, seed=seed, condition=condition)


def surge_trial():
    """20 samples outside, then inside: -vx 0.005 m/s before, 0.015 m/s after. Surge 0.010."""
    n = 60
    c = np.zeros(n + 1)
    c[20:] = 1.0
    v = np.where(np.arange(n) < 20, -0.005, -0.015)
    x = 0.45 + np.concatenate(([0.0], np.cumsum(v * DT)))
    return make_trial(x, np.full(n + 1, 0.15), np.full(n + 1, math.pi), c)


def cast_trial():
    """
    10 samples outside, 50 inside walking straight (encounter at 10), loss at 60,
    then 40 steps of zig-zag: heading alternates by exactly +-90 degrees and y
    alternates by 0.0005 m, so |vy| = 0.01 m/s and the heading-change spread is 90.
    """
    n = 100
    c = np.zeros(n + 1)
    c[10:60] = 1.0
    x = 0.45 - 0.0005 * np.arange(n + 1)
    y = np.full(n + 1, 0.15)
    h = np.full(n + 1, math.pi)
    for j in range(61, n + 1):
        if (j - 60) % 2 == 1:
            h[j] = math.pi + math.pi / 2
            y[j] = 0.15 + 0.0005
    return make_trial(x, y, h, c)


class Events(unittest.TestCase):
    def test_encounter_needs_half_a_second_below(self):
        c = np.zeros(30)
        c[10:] = 1.0
        enc, loss, run = pe.events(c, DT)
        self.assertEqual(enc, [10])
        self.assertEqual(loss, [])
        self.assertEqual(run[10], 10)

    def test_nine_samples_below_is_not_enough(self):
        c = np.zeros(30)
        c[9:] = 1.0
        enc, loss, _ = pe.events(c, DT)
        self.assertEqual(enc, [])

    def test_exactly_the_threshold_counts_as_below(self):
        c = np.full(30, pe.THRESHOLD)
        c[12:] = 0.06
        enc, _, _ = pe.events(c, DT)
        self.assertEqual(enc, [12])

    def test_loss_needs_a_quarter_second_above(self):
        c = np.zeros(40)
        c[10:15] = 1.0          # 5 samples above: a loss at 15
        enc, loss, run = pe.events(c, DT)
        self.assertEqual(enc, [10])
        self.assertEqual(loss, [15])
        self.assertEqual(run[15], 5)
        c = np.zeros(40)
        c[10:14] = 1.0          # 4 samples above: too brief to have been lost
        enc, loss, _ = pe.events(c, DT)
        self.assertEqual(enc, [10])
        self.assertEqual(loss, [])

    def test_history_must_be_inside_the_trace(self):
        c = np.zeros(30)
        c[5:] = 1.0             # only 5 samples of history: not an encounter
        enc, _, _ = pe.events(c, DT)
        self.assertEqual(enc, [])
        c = np.ones(30)
        c[3:] = 0.0             # 3 samples above: not a loss
        _, loss, _ = pe.events(c, DT)
        self.assertEqual(loss, [])

    def test_brief_dips_are_not_losses_and_reentries_need_fresh_history(self):
        c = np.zeros(60)
        c[10:30] = 1.0
        c[32:50] = 1.0          # a 2-sample dip: loss at 30, but no encounter at 32
        enc, loss, _ = pe.events(c, DT)
        self.assertEqual(enc, [10])
        self.assertEqual(loss, [30, 50])


class Surge(unittest.TestCase):
    def test_hand_made_surge_is_exact(self):
        m = pe.metrics(surge_trial())
        self.assertEqual(m["n_encounters"], 1)
        self.assertEqual(m["n_p1_events"], 1)
        self.assertAlmostEqual(m["p1_surge"], 0.010, places=9)
        self.assertEqual(m["n_losses"], 0)
        self.assertTrue(math.isnan(m["p2_vy"]))

    def test_window_is_clipped_at_the_end_of_the_trace(self):
        n = 55
        c = np.zeros(n + 1)
        c[50:] = 1.0            # encounter at 50, only 5 steps follow
        v = np.where(np.arange(n) < 50, -0.005, -0.02)
        x = 0.45 + np.concatenate(([0.0], np.cumsum(v * DT)))
        m = pe.metrics(make_trial(x, np.full(n + 1, 0.15), np.full(n + 1, math.pi), c))
        self.assertEqual(m["n_encounters"], 1)
        self.assertAlmostEqual(m["p1_surge"], 0.015, places=9)

    def test_two_encounters_are_averaged(self):
        n = 120
        c = np.zeros(n + 1)
        c[20:40] = 1.0
        c[60:] = 1.0
        v = np.full(n, -0.005)
        v[20:40] = -0.015       # surge 0.010 at the first
        v[60:80] = -0.025       # before window is steps 40..59 at -0.005: surge 0.020
        x = 0.45 + np.concatenate(([0.0], np.cumsum(v * DT)))
        m = pe.metrics(make_trial(x, np.full(n + 1, 0.15), np.full(n + 1, math.pi), c))
        self.assertEqual(m["n_p1_events"], 2)
        self.assertAlmostEqual(m["p1_surge"], 0.015, places=9)


class Cast(unittest.TestCase):
    def test_hand_made_cast_is_exact(self):
        m = pe.metrics(cast_trial())
        self.assertEqual(m["n_encounters"], 1)
        self.assertEqual(m["n_losses"], 1)
        self.assertEqual(m["n_p2_events"], 1)
        self.assertAlmostEqual(m["p2_vy"], 0.010, places=9)
        self.assertAlmostEqual(m["p2_dh_deg"], 90.0, places=6)

    def test_before_window_is_only_the_time_inside(self):
        # inside for 8 samples only (encounter at 10, loss at 18): the before
        # window is steps 10..17, and those zig-zag as hard as the after steps
        n = 70
        c = np.zeros(n + 1)
        c[10:18] = 1.0
        y = np.full(n + 1, 0.15)
        h = np.full(n + 1, math.pi)
        for j in range(11, n + 1):
            if (j - 10) % 2 == 1:
                h[j] = math.pi + math.pi / 2
                y[j] = 0.15 + 0.0005
        m = pe.metrics(make_trial(0.45 - 0.0005 * np.arange(n + 1), y, h, c))
        self.assertEqual(m["n_p2_events"], 1)
        self.assertAlmostEqual(m["p2_vy"], 0.0, places=9)
        self.assertAlmostEqual(m["p2_dh_deg"], 0.0, places=6)

    def test_heading_change_wraps(self):
        h = np.array([3.1, -3.1, 3.1, -3.1])
        d = pe.wrap(np.diff(h))
        self.assertTrue(np.all(np.abs(d) < 0.2))


class NoEncounter(unittest.TestCase):
    def test_contributes_nothing_and_is_counted(self):
        blank = straight_trial(0.2)
        m = pe.metrics(blank)
        self.assertEqual(m["n_encounters"], 0)
        self.assertEqual(m["n_p1_events"], 0)
        self.assertTrue(math.isnan(m["p1_surge"]))
        self.assertTrue(math.isnan(m["p2_vy"]))
        s = pe.summarise({"odour": [surge_trial(), dict(blank, seed=1)]})
        self.assertEqual(s["no_encounter"]["odour"], 1)
        self.assertEqual(s["conditions"]["odour"]["p1_surge"]["n"], 1)
        self.assertEqual(s["predictions"]["P1_surge"]["n_trials"], 1)
        self.assertEqual(s["predictions"]["P1_surge"]["verdict"], "undetermined")

    def test_a_loss_without_an_encounter_is_never_used(self):
        n = 60
        c = np.zeros(n + 1)
        c[:8] = 1.0             # starts inside, leaves at 8: a loss by the rule
        y = np.full(n + 1, 0.15)
        h = np.full(n + 1, math.pi)
        for j in range(9, n + 1):
            if j % 2:
                h[j] = math.pi / 2
                y[j] = 0.151
        m = pe.metrics(make_trial(0.45 - 0.0005 * np.arange(n + 1), y, h, c))
        self.assertEqual(m["n_losses"], 1)
        self.assertEqual(m["n_encounters"], 0)
        self.assertEqual(m["n_p2_events"], 0)
        self.assertTrue(math.isnan(m["p2_dh_deg"]))


class Paired(unittest.TestCase):
    def trials(self):
        return {
            "odour": [straight_trial(p, seed=i) for i, p in enumerate((0.3, 0.2, 0.4))],
            "blank": [straight_trial(p, seed=i, condition="blank") for i, p in enumerate((0.1, 0.1, 0.1))]
                     + [straight_trial(0.9, seed=7, condition="blank")],   # unpaired seed: ignored
            "nowind": [straight_trial(p, seed=i, condition="nowind") for i, p in enumerate((0.0, 0.1, 0.2))],
        }

    def test_paired_difference_and_se(self):
        s = pe.summarise(self.trials())
        ob = s["paired"]["odour-blank"]
        self.assertEqual(ob["seeds"], [0, 1, 2])
        self.assertEqual(ob["n"], 3)
        self.assertAlmostEqual(ob["mean"], 0.2, places=9)
        self.assertAlmostEqual(ob["se"], 0.1 / math.sqrt(3), places=9)
        self.assertEqual(ob["verdict"], "supported")
        on = s["paired"]["odour-nowind"]
        self.assertAlmostEqual(on["mean"], 0.2, places=9)
        self.assertAlmostEqual(on["se"], 0.1 / math.sqrt(3), places=9)
        self.assertEqual(s["predictions"]["P3_upwind_progress"]["verdict"], "supported")
        self.assertEqual(s["conditions"]["blank"]["n_trials"], 4)

    def test_p3_fails_when_the_effect_is_within_two_se(self):
        t = self.trials()
        t["odour"] = [straight_trial(p, seed=i) for i, p in enumerate((0.3, 0.0, 0.1))]   # diffs 0.2, -0.1, 0.0
        s = pe.summarise(t)
        ob = s["paired"]["odour-blank"]
        self.assertAlmostEqual(ob["mean"], 0.1 / 3, places=9)
        self.assertEqual(ob["verdict"], "not supported")
        self.assertEqual(s["predictions"]["P3_upwind_progress"]["verdict"], "not supported")

    def test_p4_is_a_plain_comparison_of_fractions(self):
        t = self.trials()
        t["odour"][0]["reached"] = True
        s = pe.summarise(t)
        p4 = s["predictions"]["P4_source_reached"]
        self.assertAlmostEqual(p4["odour"], 1 / 3)
        self.assertEqual(p4["blank"], 0.0)
        self.assertEqual(p4["verdict"], "supported")
        t["odour"][0]["reached"] = False
        self.assertEqual(pe.summarise(t)["predictions"]["P4_source_reached"]["verdict"], "not supported")

    def test_baseline_comes_from_blank(self):
        s = pe.summarise(self.trials())
        self.assertAlmostEqual(s["baseline_blank"]["mean_speed_cmd"]["mean"], 1.0)
        self.assertAlmostEqual(s["baseline_blank"]["mean_turn_cmd"]["mean"], 0.0)


class Verdicts(unittest.TestCase):
    def test_more_than_two_se_in_the_predicted_direction(self):
        self.assertEqual(pe.verdict(0.5, 0.2), "supported")
        self.assertEqual(pe.verdict(0.4, 0.2), "not supported")      # exactly 2 SE is not more than
        self.assertEqual(pe.verdict(-0.5, 0.1), "not supported")
        self.assertEqual(pe.verdict(0.0, 0.0), "not supported")
        self.assertEqual(pe.verdict(1.0, float("nan")), "undetermined")
        self.assertEqual(pe.verdict(1.0, 0.1, n=1), "undetermined")
        self.assertEqual(pe.verdict(None, None), "undetermined")

    def test_fraction_verdict(self):
        self.assertEqual(pe.verdict_fraction(0.5, 0.25), "supported")
        self.assertEqual(pe.verdict_fraction(0.25, 0.25), "not supported")
        self.assertEqual(pe.verdict_fraction(float("nan"), 0.1), "undetermined")

    def test_stats(self):
        self.assertTrue(math.isnan(pe.stats([1.0])["se"]))
        self.assertEqual(pe.stats([])["n"], 0)
        s = pe.stats([1.0, 2.0, 3.0, float("nan")])
        self.assertEqual(s["n"], 3)
        self.assertAlmostEqual(s["mean"], 2.0)
        self.assertAlmostEqual(s["se"], 1.0 / math.sqrt(3))

    def test_p2_needs_both_halves(self):
        t = {"odour": [cast_trial(), dict(cast_trial(), seed=1), dict(cast_trial(), seed=2)]}
        s = pe.summarise(t)
        p2 = s["predictions"]["P2_cast"]
        # three identical trials: SE 0, effect positive: supported on both
        self.assertEqual(p2["vy"]["verdict"], "supported")
        self.assertEqual(p2["heading_change_deg"]["verdict"], "supported")
        self.assertEqual(p2["verdict"], "supported")


class Deterministic(unittest.TestCase):
    def test_same_input_same_output_whatever_the_order(self):
        t = {
            "odour": [surge_trial(), dict(cast_trial(), seed=1), straight_trial(0.1, seed=2)],
            "blank": [straight_trial(0.05, seed=i, condition="blank") for i in range(3)],
            "nowind": [straight_trial(0.02, seed=i, condition="nowind") for i in range(3)],
        }
        a = json.dumps(pe._clean(pe.summarise(t)), sort_keys=True)
        b = json.dumps(pe._clean(pe.summarise(t)), sort_keys=True)
        self.assertEqual(a, b)
        shuffled = {"nowind": t["nowind"], "blank": t["blank"][::-1], "odour": t["odour"][::-1]}
        c = json.dumps(pe._clean(pe.summarise(shuffled)), sort_keys=True)
        self.assertEqual(a, c)


class Ladder(unittest.TestCase):
    def test_budget_ladder(self):
        self.assertEqual(pe.ladder(12, 0.2, 400, 3), (12, "kept"))
        self.assertEqual(pe.ladder(12, 0.3, 400, 3)[0], 10)
        self.assertEqual(pe.ladder(12, 0.5, 400, 3)[0], 8)
        self.assertEqual(pe.ladder(12, 5.0, 400, 3)[0], 8)        # never below 8
        self.assertEqual(pe.ladder(2, 5.0, 80, 3), (2, "kept"))    # quick mode is left alone
        self.assertEqual(pe.ladder(12, float("nan"), 400, 3), (12, "kept"))


# ---- fakes for run_trial --------------------------------------------------------------

class FakeWorld:
    """The design's kinematics with no plume: a field function stands in for the puffs."""
    dt = DT

    def __init__(self, seed, odour=True, x=0.45, y=0.15, heading=math.pi, field=None,
                 heading_unit="rad"):
        self.seed, self.odour = seed, odour
        self.x, self.y, self.t = x, y, 0.0
        self.heading_unit = heading_unit
        self._h = heading
        self.field = field or (lambda x, y, t: 0.0)
        self.wall_contacts, self.reached = 0, False
        self.trajectory = []

    @property
    def heading(self):
        return math.degrees(self._h) if self.heading_unit == "deg" else self._h

    def concentration(self, x, y):
        return float(self.field(x, y, self.t)) if self.odour else 0.0

    def wind_direction_relative(self, heading):
        h = math.radians(heading) if self.heading_unit == "deg" else heading
        return float(pe.wrap(math.pi - h))

    def step(self, turn, speed):
        self._h += float(np.clip(turn, -1, 1)) * math.pi * DT
        self.x += speed * 0.02 * DT * math.cos(self._h)
        self.y += speed * 0.02 * DT * math.sin(self._h)
        if not 0 <= self.x <= 0.6 or not 0 <= self.y <= 0.3:
            self.wall_contacts += 1
        self.x, self.y = min(max(self.x, 0.0), 0.6), min(max(self.y, 0.0), 0.3)
        self.t += DT
        self.reached = math.hypot(self.x - 0.05, self.y - 0.15) <= 0.03
        st = {"x": self.x, "y": self.y, "heading": self.heading, "c": self.concentration(self.x, self.y),
              "t": self.t, "reached": self.reached, "wall_contacts": self.wall_contacts}
        self.trajectory.append(st)
        return st


class FakeFly:
    def __init__(self, turn=0.0, speed=1.0, shape="tuple3"):
        self.turn, self.speed, self.shape = turn, speed, shape
        self.calls = []

    def step(self, c, phi, wind_sense=True, seed=0):
        self.calls.append((c, phi, wind_sense, seed))
        info = {"steer_L": 1.0, "steer_R": 2.0, "fwd_L": 3.0, "fwd_R": 4.0, "back": 0.0,
                "stop": 0.0, "orn_hz": c * 10, "jo_L": 5.0, "jo_R": 6.0, "fired": 12, "note": "text",
                "c": c, "phi": phi, "seed": seed, "wind_sense": wind_sense}   # echoes the runner owns
        if self.shape == "dict":
            return dict(info, turn=self.turn, speed=self.speed)
        if self.shape == "tuple2":
            return self.turn, self.speed
        return self.turn, self.speed, info


class RunTrial(unittest.TestCase):
    def test_walks_upwind_and_stops_at_the_source(self):
        w, f = FakeWorld(3), FakeFly()
        tr = pe.run_trial(w, f, 400, 3, True, condition="blank")
        self.assertTrue(tr["reached"])
        self.assertEqual(tr["steps_run"], 370)
        self.assertEqual(len(tr["x"]), 371)
        self.assertEqual(len(tr["turn"]), 370)
        self.assertEqual(len(tr["rates"]["steer_L"]), 370)
        self.assertNotIn("note", tr["rates"])
        for owned in ("c", "phi", "seed", "wind_sense"):
            self.assertNotIn(owned, tr["rates"])
        self.assertAlmostEqual(tr["dt"], DT)
        self.assertNotIn("mean_seed", pe.metrics(tr))
        self.assertAlmostEqual(tr["x"][0], 0.45)
        self.assertAlmostEqual(tr["x"][-1], 0.08, places=9)
        self.assertEqual(tr["condition"], "blank")
        m = pe.metrics(tr)
        self.assertAlmostEqual(m["progress"], 0.37, places=9)
        self.assertAlmostEqual(m["mean_ground_speed_m_s"], 0.02, places=9)
        # headwind: phi is 0 when the fly faces -x
        self.assertAlmostEqual(tr["phi"][0], 0.0, places=9)

    def test_per_step_seeds_are_distinct_and_paired_across_conditions(self):
        fa, fb_ = FakeFly(), FakeFly()
        pe.run_trial(FakeWorld(5, odour=True), fa, 30, 5, True)
        pe.run_trial(FakeWorld(5, odour=False), fb_, 30, 5, False)
        seeds_a = [c[3] for c in fa.calls]
        seeds_b = [c[3] for c in fb_.calls]
        self.assertEqual(len(set(seeds_a)), 30)
        self.assertEqual(seeds_a, seeds_b)
        self.assertTrue(all(c[2] is True for c in fa.calls))
        self.assertTrue(all(c[2] is False for c in fb_.calls))
        other = FakeFly()
        pe.run_trial(FakeWorld(6), other, 30, 6, True)
        self.assertNotEqual(seeds_a, [c[3] for c in other.calls])

    def test_odour_reaches_the_fly_from_the_world(self):
        field = lambda x, y, t: 1.0 if x < 0.3 else 0.0
        f = FakeFly()
        tr = pe.run_trial(FakeWorld(0, field=field), f, 200, 0, True)
        self.assertEqual(tr["c"][0], 0.0)
        self.assertEqual(tr["c"][-1], 1.0)
        self.assertEqual(f.calls[0][0], 0.0)
        self.assertEqual(f.calls[-1][0], 1.0)
        self.assertEqual(pe.metrics(tr)["n_encounters"], 1)

    def test_other_return_shapes(self):
        for shape in ("dict", "tuple2"):
            tr = pe.run_trial(FakeWorld(0), FakeFly(shape=shape), 10, 0, True)
            self.assertEqual(tr["steps_run"], 10)
            self.assertAlmostEqual(tr["speed"][0], 1.0)

    def test_degree_headings_are_converted(self):
        tr = pe.run_trial(FakeWorld(0, heading_unit="deg"), FakeFly(turn=0.5), 4, 0, True)
        self.assertAlmostEqual(tr["heading"][0], math.pi)
        self.assertAlmostEqual(tr["heading"][1], math.pi + 0.5 * math.pi * DT)

    def test_turning_bias_is_measured(self):
        tr = pe.run_trial(FakeWorld(0), FakeFly(turn=0.5, speed=0.0), 20, 0, True, condition="blank")
        m = pe.metrics(tr)
        self.assertAlmostEqual(m["mean_turn_cmd"], 0.5)
        self.assertAlmostEqual(m["mean_heading_rate_deg_s"], 90.0, places=6)
        self.assertAlmostEqual(m["mean_ground_speed_m_s"], 0.0)


try:
    import plume as _plume
except ImportError:
    _plume = None


@unittest.skipIf(_plume is None, "plume.py not present")
class RealWorldFakeFly(unittest.TestCase):
    """The runner against the real wind tunnel, with a fly that just walks upwind."""

    def test_straight_upwind_walk_reaches_the_source(self):
        w = _plume.World(0, odour=True)
        # face into the wind whatever the seeded heading: one perfect turn command per step
        class Homing(FakeFly):
            def step(self, c, phi, wind_sense=True, seed=0):
                self.calls.append((c, phi, wind_sense, seed))
                return float(np.clip(-phi, -1, 1)), 1.0, {"fired": 1}   # phi > 0: wind from the left, turn left
        f = Homing()
        tr = pe.run_trial(w, f, 400, 0, True, condition="odour")
        self.assertTrue(tr["reached"])
        self.assertLess(tr["steps_run"], 400)
        self.assertAlmostEqual(tr["dt"], _plume.DT)
        self.assertAlmostEqual(tr["x"][0], _plume.START_X)
        self.assertEqual(len(tr["c"]), tr["steps_run"] + 1)
        m = pe.metrics(tr)
        self.assertGreater(m["progress"], 0.3)
        self.assertGreaterEqual(m["n_encounters"], 1)          # it walks up the plume's centreline
        self.assertGreater(m["time_in_plume_s"], 1.0)
        self.assertAlmostEqual(abs(tr["phi"][-1]), 0.0, places=1)   # headwind at the end

    def test_paired_seeds_share_start_and_wind(self):
        a = pe.run_trial(_plume.World(4, odour=True), FakeFly(speed=0.0), 20, 4, True)
        b = pe.run_trial(_plume.World(4, odour=False), FakeFly(speed=0.0), 20, 4, False)
        self.assertEqual(a["x"][0], b["x"][0])
        self.assertEqual(a["y"][0], b["y"][0])
        self.assertEqual(a["heading"][0], b["heading"][0])
        self.assertTrue(np.all(b["c"] == 0.0))

    def test_snapshot_from_the_real_world(self):
        pic = pe.plume_snapshot(_plume.World(0, odour=True), steps=0, grid=(24, 12))
        self.assertEqual(pic.shape, (12, 24))
        self.assertGreater(pic.max(), 0.05)
        self.assertLessEqual(pic.max(), 1.0)


class MainWithFakes(unittest.TestCase):
    """The CLI end to end with fake modules in place of the brain, the tunnel and the coupling."""

    def test_quick_run_logs_every_trial_and_writes_outputs(self):
        import sys
        import types

        fake_flysim = types.ModuleType("flysim")
        class FakeBrain:
            n = 7
        fake_flysim.FlyBrain = FakeBrain
        fake_cal = types.ModuleType("calibration")
        fake_cal.CHOSEN = "fake_setting"
        fake_cal.gains_for = lambda fb, setting: None
        fake_plume = types.ModuleType("plume")
        fake_plume.World = FakeWorld
        fake_plume.ARENA_X, fake_plume.ARENA_Y, fake_plume.SOURCE, fake_plume.REACH_RADIUS = 0.6, 0.3, (0.05, 0.15), 0.03
        fake_plume.SOME_CONSTANT = 42
        fake_fly = types.ModuleType("plume_fly")
        class Fly(FakeFly):
            def __init__(self, fb, gains, **kw):
                super().__init__(turn=0.1, speed=0.5)
                self.kw = kw
            def describe(self):
                return {"kw": self.kw}
        fake_fly.PlumeFly = Fly
        saved = {k: sys.modules.get(k) for k in ("flysim", "calibration", "plume", "plume_fly")}
        sys.modules.update({"flysim": fake_flysim, "calibration": fake_cal, "plume": fake_plume, "plume_fly": fake_fly})
        real_ram = pe.free_ram_gb
        pe.free_ram_gb = lambda: 100.0
        try:
            with tempfile.TemporaryDirectory() as d:
                out = Path(d) / "plume"
                rc = pe.main(["--quick", "1", "--out", str(out), "--odorant", "ethyl acetate"])
                self.assertEqual(rc, 0)
                log = (Path(d) / "plume_quick_experiment.log").read_text(encoding="utf-8").splitlines()
                trial_lines = [l for l in log if " trial " in l]
                self.assertEqual(len(trial_lines), 6)
                self.assertIn("seed=1 cond=nowind", trial_lines[-1])
                data = json.loads((Path(d) / "plume_quick_experiment.json").read_text(encoding="utf-8"))
                self.assertFalse(data["partial"])
                self.assertEqual(data["run"]["seeds"], [0, 1])
                self.assertEqual(data["run"]["steps"], pe.QUICK_STEPS)
                self.assertEqual(data["run"]["seed_ladder"], "kept")
                self.assertEqual(data["run"]["design_seeds"], pe.DEFAULT_SEEDS)
                self.assertEqual(data["run"]["design_budget_s"], pe.BUDGET_S)
                self.assertIn("mean of clip(c, 0, 1)", data["plume_picture"])
                self.assertEqual(data["constants"]["world"]["SOME_CONSTANT"], 42)
                self.assertEqual(data["constants"]["fly_instance"]["kw"]["odorant"], "ethyl acetate")
                self.assertEqual(data["constants"]["fly_instance"]["kw"]["sim_steps"], pe.SIM_STEPS)
                self.assertEqual(len(data["summary"]["per_trial"]), 6)
                self.assertTrue((Path(d) / "plume_quick_trajectories.npz").exists())
                self.assertTrue((Path(d) / "plume_quick_trajectories.png").exists())
                # the orchestrator's key=value form, naming the JSON itself
                kv = Path(d) / "kv"
                rc = pe.main(["quick=1", f"out={kv / 'plume_quick.json'}", f"log={kv / 'run.log'}"])
                self.assertEqual(rc, 0)
                self.assertTrue((kv / "plume_quick_experiment.json").exists())
                self.assertTrue((kv / "plume_quick_trajectories.png").exists())
                self.assertEqual(len([l for l in (kv / "run.log").read_text(encoding="utf-8").splitlines() if " trial " in l]), 6)
        finally:
            pe.free_ram_gb = real_ram
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

    def test_refuses_to_load_a_brain_without_free_ram(self):
        real_ram = pe.free_ram_gb
        pe.free_ram_gb = lambda: 2.0
        try:
            with tempfile.TemporaryDirectory() as d:
                rc = pe.main(["--quick", "1", "--out", str(Path(d) / "plume")])
                self.assertEqual(rc, 2)
        finally:
            pe.free_ram_gb = real_ram


class Cli(unittest.TestCase):
    """key=value tokens and an --out that names the JSON itself."""

    def test_key_value_tokens_become_flags(self):
        self.assertEqual(pe.normalise_argv(["quick=1", "out=a/b.json", "--seeds", "3", "budget_min=80"]),
                         ["--quick", "1", "--out", "a/b.json", "--seeds", "3", "--budget-min", "80"])
        self.assertEqual(pe.normalise_argv(["--out=x"]), ["--out=x"])    # argparse's own form is left alone

    def test_out_prefix(self):
        self.assertEqual(pe.out_prefix_from("build/plume"), Path("build/plume"))
        self.assertEqual(pe.out_prefix_from("build/plume_experiment.json"), Path("build/plume"))
        self.assertEqual(pe.out_prefix_from("build/plume", quick=True), Path("build/plume_quick"))
        self.assertEqual(pe.out_prefix_from("s/plume_quick.json", quick=True), Path("s/plume_quick"))
        self.assertEqual(pe.out_prefix_from("s/plume_quick_experiment.json", quick=True), Path("s/plume_quick"))
        # the design's real outputs come from the design's own JSON path
        pre = pe.out_prefix_from("build/plume_experiment.json")
        self.assertEqual(pre.with_name(pre.name + "_experiment.json"), Path("build/plume_experiment.json"))
        self.assertEqual(pre.with_name(pre.name + "_trajectories.npz"), Path("build/plume_trajectories.npz"))


class Outputs(unittest.TestCase):
    def two_trials(self):
        return {"odour": [surge_trial()], "blank": [straight_trial(0.2, seed=1, condition="blank")]}

    def test_render_writes_a_png(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "sub" / "traj.png"
            plume = np.random.default_rng(0).random((30, 60))
            pe.render(self.two_trials(), p, plume=plume)
            self.assertTrue(p.exists())
            self.assertEqual(p.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertGreater(p.stat().st_size, 1000)

    def test_pil_fallback_writes_a_png(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "pil.png"
            pe._render_pil(self.two_trials(), p, np.zeros((30, 60)), pe.ARENA, pe.SOURCE,
                           pe.SOURCE_RADIUS, ["odour", "blank"])
            self.assertEqual(p.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_write_outputs_json_npz_png(self):
        with tempfile.TemporaryDirectory() as d:
            prefix = Path(d) / "plume"
            summary, j, z, png = pe.write_outputs(self.two_trials(), {"constants": {"runner": pe.CONSTANTS}, "steps": 60}, prefix)
            data = json.loads(j.read_text(encoding="utf-8"))
            self.assertFalse(data["partial"])
            self.assertIn("P1_surge", data["summary"]["predictions"])
            self.assertEqual(data["constants"]["runner"]["THRESHOLD"], 0.05)
            self.assertEqual(len(data["simulator_limits"]), 4)
            self.assertTrue(any("restarted from rest" in s for s in data["simulator_limits"]))
            self.assertIn("design_limits", data)
            self.assertIn("sensitivity", data["summary"])
            self.assertIn("disclosures", data["summary"])
            self.assertIsNone(data["summary"]["predictions"]["P1_surge"]["se"])   # one trial: nan -> null
            with np.load(z) as npz:
                self.assertEqual(list(npz["condition"]), ["odour", "blank"])
                self.assertEqual(list(npz["length"]), [61, 41])
                self.assertTrue(np.isnan(npz["x"][1, 41:]).all())
                self.assertAlmostEqual(float(npz["x"][1, 40]), 0.25)
            self.assertTrue(png.exists())

    def test_clean_makes_json_safe(self):
        out = pe._clean({"a": np.float64("nan"), "b": np.int64(3), "c": np.array([1.5, 2.5]), "d": np.bool_(True)})
        self.assertEqual(out, {"a": None, "b": 3, "c": [1.5, 2.5], "d": True})
        json.dumps(out)

    def test_module_constants_lists_only_uppercase_simple_values(self):
        c = pe.module_constants(pe)
        self.assertEqual(c["THRESHOLD"], 0.05)
        self.assertIn("CONSTANTS", c)
        self.assertNotIn("np", c)
        self.assertNotIn("Log", c)


class WindowDisclosure(unittest.TestCase):
    """
    Every P1 event's window lengths are recorded, and the (non-preregistered)
    full-window check excludes clipped events while the preregistered value
    keeps counting them.
    """

    def test_an_encounter_in_the_last_second_is_recorded_as_clipped(self):
        n = 55
        c = np.zeros(n + 1)
        c[50:] = 1.0
        v = np.where(np.arange(n) < 50, -0.005, -0.02)
        x = 0.45 + np.concatenate(([0.0], np.cumsum(v * DT)))
        m = pe.metrics(make_trial(x, np.full(n + 1, 0.15), np.full(n + 1, math.pi), c))
        self.assertAlmostEqual(m["p1_surge"], 0.015, places=9)              # preregistered value untouched
        self.assertEqual(m["p1_events"], [{"k": 50, "t_s": 2.5, "n_before": 20, "n_after": 5,
                                           "full_window": False, "surge": m["p1_surge"]}])
        self.assertEqual(m["n_p1_events_clipped"], 1)
        self.assertEqual(m["n_p1_events_full"], 0)
        self.assertTrue(math.isnan(m["p1_surge_full"]))

    def test_full_windows_only_drops_the_clipped_event_and_keeps_the_rest(self):
        n = 120
        c = np.zeros(n + 1)
        c[20:40] = 1.0
        c[119:] = 1.0                 # a second encounter with a one-step after window
        v = np.full(n, -0.005)
        v[20:40] = -0.015             # surge 0.010, full windows
        v[119:] = -0.105              # a huge one-step "surge" of 0.100
        x = 0.45 + np.concatenate(([0.0], np.cumsum(v * DT)))
        m = pe.metrics(make_trial(x, np.full(n + 1, 0.15), np.full(n + 1, math.pi), c))
        self.assertEqual(m["n_p1_events"], 2)
        self.assertAlmostEqual(m["p1_surge"], (0.010 + 0.100) / 2, places=9)     # preregistered: both count
        self.assertAlmostEqual(m["p1_surge_full"], 0.010, places=9)               # sensitivity: only the full one
        self.assertEqual(m["n_p1_events_full"], 1)
        self.assertEqual([e["n_after"] for e in m["p1_events"]], [20, 1])


class LossDisclosure(unittest.TestCase):
    """
    A loss of the plume the fly was born in is flagged as preceding the first
    encounter, the after-window re-entry fraction is measured, and the
    post-encounter sensitivity check uses only losses of a plume the fly found.
    """

    def trial(self, n, c, zigzag_from=None):
        y = np.full(n + 1, 0.15)
        h = np.full(n + 1, math.pi)
        if zigzag_from is not None:
            for j in range(zigzag_from + 1, n + 1):
                if (j - zigzag_from) % 2 == 1:
                    h[j] = math.pi + math.pi / 2
                    y[j] = 0.15 + 0.0005
        return make_trial(0.45 - 0.0005 * np.arange(n + 1), y, h, c)

    def test_loss_before_the_first_encounter_is_flagged(self):
        # born inside (30 samples), lost at 30, re-encountered at 60, lost again at 90
        n = 140
        c = np.zeros(n + 1)
        c[:30] = 1.0
        c[60:90] = 1.0
        m = pe.metrics(self.trial(n, c))
        self.assertEqual((m["n_encounters"], m["n_losses"], m["n_p2_events"]), (1, 2, 2))   # preregistered: both losses count
        self.assertEqual([(e["k"], e["after_first_encounter"]) for e in m["p2_events"]], [(30, False), (90, True)])
        self.assertEqual(m["n_p2_before_first_encounter"], 1)
        self.assertEqual(m["n_p2_events_post_encounter"], 1)
        self.assertTrue(m["starts_above_threshold"])
        self.assertAlmostEqual(m["c_start"], 1.0)
        self.assertAlmostEqual(m["fraction_in_plume"], 60 / 141, places=9)

    def test_after_window_reentry_fraction(self):
        # loss at 30; 10 of the 40 after-samples (31..70) are back above threshold
        n = 100
        c = np.zeros(n + 1)
        c[:30] = 1.0
        c[50:60] = 1.0
        c[80:] = 1.0
        m = pe.metrics(self.trial(n, c))
        self.assertEqual(m["n_encounters"], 2)
        losses = [e for e in m["p2_events"] if e["k"] == 30]
        self.assertEqual(len(losses), 1)
        self.assertAlmostEqual(losses[0]["after_above_fraction"], 10 / 40, places=9)
        self.assertFalse(losses[0]["after_first_encounter"])
        self.assertAlmostEqual(m["p2_after_above_fraction"], np.mean([e["after_above_fraction"] for e in m["p2_events"]]))

    def test_post_encounter_sensitivity_uses_only_losses_after_an_encounter(self):
        # born inside, lost at 20 walking straight; found at 40; lost at 70 and then zig-zagging
        n = 130
        c = np.zeros(n + 1)
        c[:20] = 1.0
        c[40:70] = 1.0
        m = pe.metrics(self.trial(n, c, zigzag_from=70))
        self.assertEqual(m["n_p2_events"], 2)
        self.assertAlmostEqual(m["p2_vy"], 0.010 / 2, places=9)                  # preregistered: the mean of both
        self.assertAlmostEqual(m["p2_dh_deg"], 90.0 / 2, places=6)
        self.assertAlmostEqual(m["p2_vy_post_encounter"], 0.010, places=9)       # sensitivity: the found-and-lost one
        self.assertAlmostEqual(m["p2_dh_deg_post_encounter"], 90.0, places=6)
        self.assertEqual(m["n_p2_events_post_encounter"], 1)


class SummaryDisclosure(unittest.TestCase):
    """
    The summary carries the labelled sensitivity block, the disclosure counts,
    the summed JO drive and the notes; the preregistered verdicts are not
    touched by any of them.
    """

    def test_sensitivity_and_disclosures_are_labelled_and_separate(self):
        s = pe.summarise({"odour": [surge_trial(), dict(cast_trial(), seed=1)],
                          "blank": [straight_trial(0.2, condition="blank"), straight_trial(0.1, seed=1, condition="blank")]})
        self.assertIn("not preregistered", s["sensitivity"]["note"])
        self.assertEqual(set(s["sensitivity"]) - {"note"},
                         {"P1_full_windows_only", "P2_full_windows_only", "P2_losses_after_an_encounter_only"})
        self.assertIn("would_be_verdict", s["sensitivity"]["P1_full_windows_only"])
        self.assertNotIn("would_be_verdict", s["predictions"]["P1_surge"])
        d = s["disclosures"]["odour"]
        self.assertEqual((d["n_trials"], d["n_start_above_threshold"], d["n_p1_events"], d["n_p2_events"]), (2, 0, 2, 1))
        self.assertEqual(d["n_p1_events_clipped"], 1)      # cast_trial's encounter at sample 10 has only 10 before-steps
        self.assertIn("92.5", s["predictions"]["P4_source_reached"]["design_note"])
        self.assertIn("restarted from rest", s["predictions"]["P1_surge"]["protocol_note"])
        self.assertIn("restarted from rest", s["predictions"]["P2_cast"]["protocol_note"])
        self.assertIn("total JO drive", s["predictions"]["P3_upwind_progress"]["confound_note"])
        self.assertAlmostEqual(s["predictions"]["P1_surge"]["effect_in_se"],
                               s["predictions"]["P1_surge"]["effect"] / s["predictions"]["P1_surge"]["se"])
        self.assertIn("readout-limited", s["baseline_blank"]["note"])
        self.assertEqual(s["predictions"]["P1_surge"]["statement"],
                         "upwind velocity rises in the 1 s after an encounter (odour), > 2 SE")   # the statements are verbatim

    def test_jo_totals_from_per_side_means(self):
        rates = {"jo_left_hz": np.full(40, 30.0), "jo_right_hz": np.full(40, 10.0)}
        tr = dict(straight_trial(0.2), rates=rates)
        s = pe.summarise({"odour": [tr]}, jo_cells=(203, 132))
        self.assertAlmostEqual(s["conditions"]["odour"]["mean_jo_total_cell_hz"]["mean"], 203 * 30.0 + 132 * 10.0)
        self.assertAlmostEqual(s["disclosures"]["odour"]["mean_jo_total_cell_hz"], 203 * 30.0 + 132 * 10.0)
        # a trial that recorded the total itself is left alone
        tr2 = dict(straight_trial(0.2), rates=dict(rates, jo_total_cell_hz=np.full(40, 1.0)))
        s2 = pe.summarise({"odour": [tr2]}, jo_cells=(203, 132))
        self.assertAlmostEqual(s2["conditions"]["odour"]["mean_jo_total_cell_hz"]["mean"], 1.0)
        # and without the cell counts nothing is invented
        self.assertNotIn("mean_jo_total_cell_hz", pe.summarise({"odour": [tr]})["conditions"]["odour"])


class RunRecord(unittest.TestCase):
    """The seed decision is written into the run record, with what the runner's own ladder would have done."""

    def test_as_designed_and_quick(self):
        self.assertEqual(pe.ladder_note({"requested_seeds": 12, "budget_s": pe.BUDGET_S, "steps": 400}), "as designed")
        self.assertIsNone(pe.ladder_note({"requested_seeds": 2, "quick": True}))

    def test_hand_cut_seeds_and_raised_budget_are_named(self):
        note = pe.ladder_note({"requested_seeds": 10, "budget_s": 4500.0, "steps": 400, "sec_per_brain_run": 0.311})
        self.assertIn("requested_seeds 10 differs from the design's 12", note)
        self.assertIn("budget_s 4500 differs from the design's 3600", note)
        self.assertIn("would have given 8 seeds (dropped 12 -> 8)", note)
        self.assertIn("from the requested 10 at the design budget it would have given 8", note)


class Reanalysis(unittest.TestCase):
    """
    A finished run's summary and picture can be recomputed from its saved
    trajectories without a brain; the preregistered numbers come back
    identical and the check is recorded in the JSON.
    """

    def test_round_trip_reproduces_the_preregistered_metrics(self):
        odour0 = dict(surge_trial(), wall_contacts=3,
                      rates={"jo_left_hz": np.full(60, 30.0), "jo_right_hz": np.full(60, 10.0), "steer_L": np.full(60, 2.0)})
        trials = {"odour": [odour0, dict(cast_trial(), seed=1)],
                  "blank": [straight_trial(0.2, condition="blank"), straight_trial(0.1, seed=1, condition="blank")],
                  "nowind": [straight_trial(0.05, condition="nowind"), straight_trial(0.02, seed=1, condition="nowind")]}
        run_info = {"constants": {"runner": pe.CONSTANTS, "fly_instance": {"jo_left": 203, "jo_right": 132}},
                    "requested_seeds": 2, "steps": 100, "budget_s": pe.BUDGET_S, "sec_per_brain_run": 0.01}
        with tempfile.TemporaryDirectory() as d:
            prefix = Path(d) / "plume"
            _, j0, _, _ = pe.write_outputs(trials, run_info, prefix)
            old = json.loads(j0.read_text(encoding="utf-8"))
            _, j1, z1, png = pe.reanalyse(str(j0), note="test")
            new = json.loads(j1.read_text(encoding="utf-8"))
            same, worst, mismatches = pe.compare_preregistered(old["summary"]["predictions"], new["summary"]["predictions"])
            self.assertTrue(same, mismatches)
            self.assertEqual(worst, 0.0)
            rean = new["run"]["reanalyses"][-1]
            self.assertTrue(rean["preregistered_metrics_unchanged"])
            self.assertEqual(rean["why"], "test")
            self.assertEqual(new["run"]["requested_seeds"], 2)                  # the run record is kept
            self.assertEqual(new["run"]["design_seeds"], pe.DEFAULT_SEEDS)
            self.assertIn("seed_decision", new["run"])
            # recorded rate means, command means and wall contacts come through; the rest is recomputed
            row = [r for r in new["summary"]["per_trial"] if r["condition"] == "odour" and r["seed"] == 0][0]
            self.assertAlmostEqual(row["mean_steer_L"], 2.0)
            self.assertAlmostEqual(row["mean_jo_total_cell_hz"], 203 * 30.0 + 132 * 10.0)
            self.assertEqual(row["wall_contacts"], 3)
            self.assertAlmostEqual(row["p1_surge"], 0.010, places=9)
            self.assertEqual(new["summary"]["baseline_blank"]["mean_speed_cmd"]["mean"],
                             old["summary"]["baseline_blank"]["mean_speed_cmd"]["mean"])
            self.assertTrue(png.exists())
            self.assertIn("mean of clip(c, 0, 1)", new["plume_picture"])
            self.assertEqual(old["plume_picture"], None)                        # the first write had no picture

    def test_a_changed_number_is_caught(self):
        a = {"P1_surge": {"effect": 0.1, "se": 0.01, "verdict": "supported"}}
        b = {"P1_surge": {"effect": 0.2, "se": 0.01, "verdict": "supported"}}
        same, worst, mismatches = pe.compare_preregistered(a, b)
        self.assertFalse(same)
        self.assertEqual(mismatches, ["P1_surge.effect"])
        self.assertAlmostEqual(worst, 0.1)
        self.assertTrue(pe.compare_preregistered(a, a)[0])


class PlumeAverage(unittest.TestCase):
    """The picture behind the trajectories is the plume the run's flies saw on average, and says so."""

    def test_time_averaged_plume_is_bounded_and_captioned(self):
        import plume
        field, label = pe.plume_average(plume.World, [0, 1], 20, every=10)
        self.assertEqual(field.shape, (pe.SNAPSHOT_GRID[1], pe.SNAPSHOT_GRID[0]))
        self.assertTrue(np.all(field >= 0) and np.all(field <= 1))
        self.assertIn("seeds 0..1", label)
        ny, nx = field.shape
        self.assertGreater(field[ny // 2, nx // 4], 0.5)      # 0.1 m downwind on the centreline: odorous
        self.assertLess(field[0, nx - 1], 0.05)               # the far corner: clean
        self.assertEqual(pe.plume_average(plume.World, [], 20), (None, ""))

    def test_render_captions_the_plume(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "traj.png"
            trials = {"odour": [surge_trial()]}
            pe.render(trials, p, plume=np.zeros((30, 60)), plume_label="a caption")
            self.assertTrue(p.exists())
            pe._render_pil(trials, Path(d) / "pil.png", np.zeros((30, 60)), pe.ARENA, pe.SOURCE,
                           pe.SOURCE_RADIUS, ["odour"], "a caption")
            self.assertTrue((Path(d) / "pil.png").exists())


if __name__ == "__main__":
    unittest.main()
