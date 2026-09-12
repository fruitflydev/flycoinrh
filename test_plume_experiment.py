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
                 heading_unit="rad", protocol="v1"):
        self.seed, self.odour, self.protocol = seed, odour, protocol
        if protocol == "v2" and x == 0.45:
            x = 0.25
        self.x, self.y, self.t = x, y, 0.0
        self.heading_unit = heading_unit
        self._h = heading
        self.field = field or (lambda x, y, t: 0.0)
        self.start_c_field = float(self.field(self.x, self.y, 0.0))
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


# ---- protocol v2 ---------------------------------------------------------------------

def born_inside_trial(n=140, zigzag_from=None):
    """
    Born inside (30 samples), lost at 30, found at 60, lost at 100 after 40
    samples inside with 40 steps to go: v1 counts both losses, v2 only the
    second (the first precedes any encounter and has under 2 s inside).
    """
    c = np.zeros(n + 1)
    c[:30] = 1.0
    c[60:100] = 1.0
    y = np.full(n + 1, 0.15)
    h = np.full(n + 1, math.pi)
    if zigzag_from is not None:
        for j in range(zigzag_from + 1, n + 1):
            if (j - zigzag_from) % 2 == 1:
                h[j] = math.pi + math.pi / 2
                y[j] = 0.15 + 0.0005
    return make_trial(0.45 - 0.0005 * np.arange(n + 1), y, h, c)


class V2Windows(unittest.TestCase):
    """
    Protocol v2's window rules are the primary P1/P2 values: an encounter
    without a full window on both sides is excluded, and a loss counts only
    if it follows an encounter and has full windows; v1's rule stays under
    its own name for the side-by-side reading.
    """

    def test_encounter_without_a_full_before_window_is_excluded(self):
        # cast_trial's encounter is at sample 10: only 10 before-steps, the window needs 20
        m1, m2 = pe.metrics(cast_trial(), "v1"), pe.metrics(cast_trial(), "v2")
        self.assertEqual((m1["n_p1_events"], m1["p1_rule"]), (1, "any_window"))
        self.assertEqual((m2["n_p1_events"], m2["p1_rule"]), (0, "full_window"))
        self.assertTrue(math.isnan(m2["p1_surge"]))
        self.assertEqual(m2["n_p1_events_any_window"], 1)
        self.assertAlmostEqual(m2["p1_surge_any_window"], m1["p1_surge"], places=12)
        self.assertEqual(m2["n_encounters"], 1)                      # the event itself is still an encounter

    def test_encounter_in_the_last_second_is_excluded_and_full_ones_count(self):
        n = 55
        c = np.zeros(n + 1)
        c[50:] = 1.0
        v = np.where(np.arange(n) < 50, -0.005, -0.02)
        x = 0.45 + np.concatenate(([0.0], np.cumsum(v * DT)))
        m = pe.metrics(make_trial(x, np.full(n + 1, 0.15), np.full(n + 1, math.pi), c), "v2")
        self.assertEqual(m["n_p1_events"], 0)
        self.assertTrue(math.isnan(m["p1_surge"]))
        self.assertAlmostEqual(m["p1_surge_any_window"], 0.015, places=9)
        # two encounters with full windows: both count, exactly as under v1
        n = 120
        c = np.zeros(n + 1)
        c[20:40] = 1.0
        c[60:] = 1.0
        v = np.full(n, -0.005)
        v[20:40] = -0.015
        v[60:80] = -0.025
        x = 0.45 + np.concatenate(([0.0], np.cumsum(v * DT)))
        m = pe.metrics(make_trial(x, np.full(n + 1, 0.15), np.full(n + 1, math.pi), c), "v2")
        self.assertEqual(m["n_p1_events"], 2)
        self.assertAlmostEqual(m["p1_surge"], 0.015, places=9)
        self.assertAlmostEqual(m["p1_surge"], pe.metrics(make_trial(x, np.full(n + 1, 0.15), np.full(n + 1, math.pi), c), "v1")["p1_surge"])

    def test_loss_before_any_encounter_is_excluded(self):
        m1, m2 = pe.metrics(born_inside_trial(), "v1"), pe.metrics(born_inside_trial(), "v2")
        self.assertEqual((m1["n_losses"], m1["n_p2_events"], m1["p2_rule"]), (2, 2, "any_window"))
        self.assertEqual((m2["n_losses"], m2["n_p2_events"], m2["p2_rule"]), (2, 1, "full_window_after_encounter"))
        self.assertEqual(m2["n_p2_events_any_window"], 2)
        self.assertEqual(m2["n_p2_before_first_encounter"], 1)
        self.assertEqual(m2["n_p2_events_full_post_encounter"], 1)
        # the one that counts is the found-and-lost one at 100, with full windows: 40 in, 40 after
        used = [e for e in m2["p2_events"] if e["full_window"] and e["after_first_encounter"]]
        self.assertEqual([e["k"] for e in used], [100])
        self.assertEqual([(e["k"], e["full_window"], e["after_first_encounter"]) for e in m2["p2_events"]],
                         [(30, False, False), (100, True, True)])
        # and a zig-zag after that loss gives the same numbers as v1's post-encounter sensitivity check
        z = pe.metrics(born_inside_trial(zigzag_from=100), "v2")
        self.assertAlmostEqual(z["p2_vy"], 0.010, places=9)
        self.assertAlmostEqual(z["p2_dh_deg"], 90.0, places=6)
        self.assertAlmostEqual(z["p2_vy"], z["p2_vy_post_encounter"], places=12)

    def test_loss_needs_two_seconds_inside_and_two_seconds_after(self):
        # found at 10, lost at 18: only 8 samples inside, v1 counts it, v2 does not
        n = 70
        c = np.zeros(n + 1)
        c[10:18] = 1.0
        y = np.full(n + 1, 0.15)
        h = np.full(n + 1, math.pi)
        tr = make_trial(0.45 - 0.0005 * np.arange(n + 1), y, h, c)
        self.assertEqual(pe.metrics(tr, "v1")["n_p2_events"], 1)
        self.assertEqual(pe.metrics(tr, "v2")["n_p2_events"], 0)
        self.assertTrue(math.isnan(pe.metrics(tr, "v2")["p2_vy"]))
        # found at 20, inside for 50, lost at 70 with only 10 steps left: v2 excludes the clipped after-window
        n = 80
        c = np.zeros(n + 1)
        c[20:70] = 1.0
        tr = make_trial(0.45 - 0.0005 * np.arange(n + 1), np.full(n + 1, 0.15), np.full(n + 1, math.pi), c)
        self.assertEqual(pe.metrics(tr, "v1")["n_p2_events"], 1)
        self.assertEqual(pe.metrics(tr, "v2")["n_p2_events"], 0)
        self.assertEqual(pe.metrics(tr, "v2")["n_p2_events_clipped"], 1)

    def test_v2_summary_uses_its_rule_and_keeps_v1_rule_as_sensitivity(self):
        t = {"odour": [cast_trial(), dict(born_inside_trial(zigzag_from=100), seed=1)],
             "blank": [straight_trial(0.1, seed=i, condition="blank") for i in range(2)],
             "shuffled": [straight_trial(0.05, seed=i, condition="shuffled") for i in range(2)]}
        s = pe.summarise(t, protocol="v2")
        self.assertEqual(s["protocol"], "v2")
        self.assertEqual(list(s["conditions"]), ["odour", "blank", "shuffled"])
        p = s["predictions"]
        self.assertEqual(p["P1_surge"]["rule"], "full_window")
        # cast_trial's encounter at 10 lacks a full before-window; the born-inside trial's at 60 has both
        self.assertEqual(p["P1_surge"]["n_events"], 1)
        self.assertEqual(p["P1_surge"]["n_events_any_window"], 2)
        self.assertEqual(p["P1_surge"]["n_trials"], 1)
        self.assertEqual(p["P2_cast"]["rule"], "full_window_after_encounter")
        # cast_trial's loss at 60 (50 inside, 40 after) and the born-inside trial's at 100 count; its loss at 30 does not
        self.assertEqual(p["P2_cast"]["n_events"], 2)
        self.assertEqual(p["P2_cast"]["n_events_any_window"], 3)
        self.assertEqual(p["P2_cast"]["n_events_before_first_encounter"], 1)
        self.assertIn("full", p["P1_surge"]["statement"])
        self.assertIn("follow an encounter", p["P2_cast"]["statement"])
        self.assertIn("carried", p["P1_surge"]["protocol_note"])
        self.assertNotIn("restarted from rest", p["P1_surge"]["protocol_note"])
        self.assertIn("42.5", p["P4_source_reached"]["design_note"])
        self.assertEqual(set(s["sensitivity"]) - {"note"},
                         {"P1_any_window_v1_rule", "P2_any_window_v1_rule", "P2_full_windows_only", "P2_losses_after_an_encounter_only"})
        self.assertIn("not preregistered for v2", s["sensitivity"]["note"])
        self.assertEqual(s["sensitivity"]["P1_any_window_v1_rule"]["n_trials"], 2)
        # v1 on the same trials reads the same events under its own rule
        s1 = pe.summarise({"odour": t["odour"]}, protocol="v1")
        self.assertEqual(s1["predictions"]["P1_surge"]["n_events"], 2)
        self.assertAlmostEqual(s1["predictions"]["P1_surge"]["effect"], s["sensitivity"]["P1_any_window_v1_rule"]["effect"])


class V2Paired(unittest.TestCase):
    """The paired SE and the 2 SE bar on odour - shuffled, and the v2 condition set."""

    def trials(self, odour=(0.3, 0.2, 0.4), shuffled=(0.1, 0.1, 0.1)):
        return {
            "odour": [straight_trial(p, seed=i) for i, p in enumerate(odour)],
            "blank": [straight_trial(p, seed=i, condition="blank") for i, p in enumerate((0.05, 0.05, 0.05))],
            "shuffled": [straight_trial(p, seed=i, condition="shuffled") for i, p in enumerate(shuffled)],
        }

    def test_odour_minus_shuffled_se_and_verdicts(self):
        s = pe.summarise(self.trials(), protocol="v2")
        self.assertEqual(set(s["paired"]), {"odour-blank", "odour-shuffled"})
        osh = s["paired"]["odour-shuffled"]
        self.assertEqual(osh["seeds"], [0, 1, 2])
        self.assertAlmostEqual(osh["mean"], 0.2, places=9)
        self.assertAlmostEqual(osh["se"], 0.1 / math.sqrt(3), places=9)    # 0.2 > 2 x 0.0577
        self.assertEqual(osh["verdict"], "supported")
        p3 = s["predictions"]["P3_upwind_progress"]
        self.assertEqual(set(p3) - {"statement", "verdict", "control_note"}, {"odour-blank", "odour-shuffled"})
        self.assertIn("odour > shuffled", p3["statement"])
        self.assertEqual(p3["verdict"], "supported")
        self.assertIn("shuffled", s["predictions"]["P4_source_reached"])
        self.assertNotIn("nowind", s["predictions"]["P4_source_reached"])

    def test_within_two_se_is_not_supported(self):
        s = pe.summarise(self.trials(odour=(0.3, 0.0, 0.1)), protocol="v2")   # diffs vs shuffled: 0.2, -0.1, 0.0
        osh = s["paired"]["odour-shuffled"]
        self.assertAlmostEqual(osh["mean"], 0.1 / 3, places=9)
        self.assertAlmostEqual(osh["se"], np.std([0.2, -0.1, 0.0], ddof=1) / math.sqrt(3), places=9)
        self.assertEqual(osh["verdict"], "not supported")
        self.assertEqual(s["predictions"]["P3_upwind_progress"]["verdict"], "not supported")
        # exactly 2 SE is not more than 2 SE
        se = np.std([0.2, -0.1, 0.0], ddof=1) / math.sqrt(3)
        self.assertEqual(pe.verdict(2 * se, se), "not supported")
        self.assertEqual(pe.verdict(2 * se + 1e-12, se), "supported")

    def test_one_half_passing_is_not_supported(self):
        s = pe.summarise(self.trials(shuffled=(0.3, 0.2, 0.4)), protocol="v2")   # odour == shuffled
        self.assertEqual(s["paired"]["odour-blank"]["verdict"], "supported")
        self.assertEqual(s["paired"]["odour-shuffled"]["verdict"], "not supported")
        self.assertEqual(s["predictions"]["P3_upwind_progress"]["verdict"], "not supported")


class FacingUpwind(unittest.TestCase):
    def test_fraction_of_steps_with_cos_phi_positive(self):
        self.assertAlmostEqual(pe.metrics(straight_trial(0.2))["facing_upwind_fraction"], 1.0)
        n = 20
        down = make_trial(0.3 + 0.001 * np.arange(n + 1), np.full(n + 1, 0.15), np.zeros(n + 1), np.zeros(n + 1))
        self.assertAlmostEqual(pe.metrics(down)["facing_upwind_fraction"], 0.0)
        h = np.full(n + 1, math.pi)
        h[:10] = 0.0                                  # the heading at the start of each step counts: 10 of 20 steps
        half = make_trial(np.full(n + 1, 0.3), np.full(n + 1, 0.15), h, np.zeros(n + 1))
        self.assertAlmostEqual(pe.metrics(half)["facing_upwind_fraction"], 0.5)
        s = pe.summarise({"blank": [straight_trial(0.1, condition="blank"), dict(down, seed=1, condition="blank")]})
        self.assertAlmostEqual(s["conditions"]["blank"]["facing_upwind_fraction"]["mean"], 0.5)
        self.assertAlmostEqual(s["baseline_blank"]["facing_upwind_fraction"]["mean"], 0.5)
        self.assertAlmostEqual(s["disclosures"]["blank"]["facing_upwind_fraction"], 0.5)


class Smoothing(unittest.TestCase):
    """The motor low-pass in run_trial: the world gets the smoothed command, the record keeps both."""

    def test_exponential_smoothing_from_rest(self):
        alpha = 1.0 - math.exp(-DT / 0.15)
        self.assertAlmostEqual(pe.smooth_alpha(DT, 0.15), alpha)
        self.assertEqual(pe.smooth_alpha(DT, None), 1.0)
        tr = pe.run_trial(FakeWorld(0), FakeFly(turn=0.5, speed=1.0), 30, 0, True, smooth_tau_s=0.15)
        self.assertAlmostEqual(tr["smooth_alpha"], alpha)
        self.assertEqual(tr["smooth_tau_s"], 0.15)
        np.testing.assert_allclose(tr["turn_raw"], 0.5)
        np.testing.assert_allclose(tr["speed_raw"], 1.0)
        expect = 1.0 - (1.0 - alpha) ** (np.arange(30) + 1)
        np.testing.assert_allclose(tr["turn"], 0.5 * expect, atol=1e-12)
        np.testing.assert_allclose(tr["speed"], expect, atol=1e-12)
        self.assertLess(tr["speed"][0], 0.3)
        self.assertGreater(tr["speed"][-1], 0.99)
        # the world moved by the smoothed speed, not the raw one (FakeWorld turns, then walks along the new heading)
        self.assertAlmostEqual(tr["x"][1] - tr["x"][0], expect[0] * 0.02 * DT * math.cos(tr["heading"][1]), places=12)
        m = pe.metrics(tr)
        self.assertAlmostEqual(m["mean_turn_raw"], 0.5)
        self.assertAlmostEqual(m["mean_speed_raw"], 1.0)
        self.assertAlmostEqual(m["mean_turn_cmd"], float(np.mean(0.5 * expect)))
        self.assertLess(m["mean_speed_cmd"], m["mean_speed_raw"])

    def test_no_smoothing_leaves_raw_and_world_commands_equal(self):
        tr = pe.run_trial(FakeWorld(0), FakeFly(turn=0.3, speed=0.7), 10, 0, True)
        np.testing.assert_array_equal(tr["turn"], tr["turn_raw"])
        np.testing.assert_array_equal(tr["speed"], tr["speed_raw"])
        self.assertEqual(tr["smooth_alpha"], 1.0)

    def test_command_distributions(self):
        a = pe.run_trial(FakeWorld(0), FakeFly(turn=0.5, speed=1.0), 30, 0, True, smooth_tau_s=0.15)
        b = pe.run_trial(FakeWorld(1), FakeFly(turn=-1.5, speed=-0.2), 10, 1, True, smooth_tau_s=0.15)
        d = pe.command_distributions([a, b])
        self.assertEqual(set(d) - {"note"}, {"turn_raw", "speed_raw", "turn", "speed"})
        self.assertEqual(d["turn_raw"]["n"], 40)
        self.assertAlmostEqual(d["turn_raw"]["fraction_negative"], 0.25)
        self.assertAlmostEqual(d["turn_raw"]["fraction_beyond_clip"], 0.25)   # -1.5 is beyond the world's clip
        self.assertAlmostEqual(d["speed_raw"]["q50"], 1.0)
        self.assertAlmostEqual(d["speed_raw"]["min"], -0.2)
        self.assertLessEqual(d["speed"]["max"], 1.0)
        self.assertLess(d["turn"]["min"], 0.0)
        s = pe.summarise({"odour": [a, dict(b, seed=1)]}, protocol="v2")
        self.assertEqual(s["conditions"]["odour"]["commands"]["turn_raw"]["n"], 40)
        self.assertEqual(pe.command_distributions([straight_trial(0.1)])["turn"]["n"], 40)
        self.assertEqual(pe.command_distributions([dict(straight_trial(0.1), turn=np.array([]), speed=np.array([]))]), {})


class Shuffled(unittest.TestCase):
    """The shuffled control: the fly's antennae get a random angle, the record keeps the true one."""

    def test_fly_gets_a_random_angle_independent_of_the_heading(self):
        f = FakeFly(turn=0.0, speed=0.0)
        tr = pe.run_trial(FakeWorld(3), f, 200, 3, True, condition="shuffled", shuffled=True)
        self.assertTrue(tr["shuffled"])
        np.testing.assert_allclose(tr["phi"], 0.0, atol=1e-12)                # the fly stands facing the wind
        given = np.array([c[1] for c in f.calls])
        np.testing.assert_array_equal(given, tr["phi_fly"])
        self.assertTrue(np.all(given > -math.pi) and np.all(given <= math.pi))
        self.assertGreater(np.std(given), 1.0)                                  # uniform on (-pi, pi]: sd 1.81
        self.assertLess(abs(np.mean(given)), 0.5)
        self.assertGreater(np.mean(np.abs(given) > math.pi / 2), 0.35)         # half the angles are tailwind-ish
        self.assertEqual(len(set(np.round(given, 12))), 200)
        self.assertTrue(all(c[2] is True for c in f.calls))                     # wind sense stays on

    def test_deterministic_per_seed_and_distinct_from_the_brain_seed(self):
        a = pe.run_trial(FakeWorld(4), FakeFly(), 20, 4, True, shuffled=True)
        b = pe.run_trial(FakeWorld(4, odour=False), FakeFly(), 20, 4, True, shuffled=True)
        np.testing.assert_array_equal(a["phi_fly"], b["phi_fly"])
        c = pe.run_trial(FakeWorld(5), FakeFly(), 20, 5, True, shuffled=True)
        self.assertFalse(np.array_equal(a["phi_fly"], c["phi_fly"]))
        self.assertNotEqual(pe.shuffle_seed(4), pe.step_seed(4, 0))
        self.assertNotEqual(pe.shuffle_seed(4), pe.shuffle_seed(5))
        plain = pe.run_trial(FakeWorld(4), FakeFly(), 20, 4, True)
        np.testing.assert_array_equal(plain["phi_fly"], plain["phi"])
        self.assertFalse(plain["shuffled"])
        self.assertEqual(pe.condition_flags("v2", "shuffled"), (True, True, True))
        self.assertEqual(pe.condition_flags("v2", "blank"), (False, True, False))
        self.assertEqual(pe.condition_flags("v1", "nowind"), (True, False, False))


class CarryingFly(FakeFly):
    def __init__(self, *a, method="begin_trial", **kw):
        super().__init__(*a, **kw)
        self.begun = []
        setattr(self, method, self._begin)

    def _begin(self, seed):
        self.begun.append(int(seed))


class StateCarry(unittest.TestCase):
    """v2 tells the fly when a trial starts, and refuses a fly that cannot be told."""

    def test_begin_trial_is_called_once_per_trial_with_the_trial_seed(self):
        f = CarryingFly()
        tr = pe.run_trial(FakeWorld(7), f, 15, 7, True, state_carry=True)
        self.assertEqual(f.begun, [pe.trial_seed(7)])
        self.assertEqual(pe.trial_seed(7), pe.step_seed(7, 0))
        self.assertEqual(tr["begin_trial_method"], "begin_trial")
        self.assertTrue(tr["state_carry"])
        pe.run_trial(FakeWorld(8), f, 15, 8, True, state_carry=True)
        self.assertEqual(f.begun, [pe.trial_seed(7), pe.trial_seed(8)])
        # per-step seeds are still passed (a state-carrying fly uses the first only)
        self.assertEqual([c[3] for c in f.calls][:2], [pe.step_seed(7, 0), pe.step_seed(7, 1)])

    def test_synonyms_and_refusal(self):
        for name in ("start_trial", "reset_trial", "reset"):
            f = CarryingFly(method=name)
            tr = pe.run_trial(FakeWorld(1), f, 3, 1, True, state_carry=True)
            self.assertEqual((tr["begin_trial_method"], f.begun), (name, [pe.trial_seed(1)]))
        with self.assertRaises(TypeError):
            pe.run_trial(FakeWorld(1), FakeFly(), 3, 1, True, state_carry=True)
        plain = pe.run_trial(FakeWorld(1), CarryingFly(), 3, 1, True)      # v1 never calls it
        self.assertIsNone(plain["begin_trial_method"])
        self.assertEqual(pe.begin_trial_name(FakeFly()), None)

    def test_make_fly_and_make_world_enforce_the_v2_interface(self):
        import types
        mod = types.ModuleType("plume_fly")

        class OldFly(FakeFly):
            def __init__(self, fb, gains, **kw):
                super().__init__()
                self.kw = kw

        class NewFly(CarryingFly):
            def __init__(self, fb, gains, protocol="v1", **kw):
                super().__init__()
                self.protocol, self.kw = protocol, kw

        mod.PlumeFly = OldFly
        self.assertIsInstance(pe.make_fly(mod, None, None, "v1", sim_steps=60), OldFly)
        with self.assertRaises(TypeError):
            pe.make_fly(mod, None, None, "v2", sim_steps=120)
        mod.PlumeFly = NewFly
        fly = pe.make_fly(mod, None, None, "v2", sim_steps=120)
        self.assertEqual((fly.protocol, fly.kw), ("v2", {"sim_steps": 120}))
        self.assertEqual(pe.make_fly(mod, None, None, "v1").protocol, "v1")

        class NewFlyNoReset(FakeFly):
            def __init__(self, fb, gains, protocol="v1", **kw):
                super().__init__()
        mod.PlumeFly = NewFlyNoReset
        with self.assertRaises(TypeError):
            pe.make_fly(mod, None, None, "v2")

        wmod = types.ModuleType("plume")
        wmod.World = FakeWorld
        self.assertEqual(pe.make_world(wmod, 0, True, "v2").protocol, "v2")
        self.assertEqual(pe.make_world(wmod, 0, True, "v1").protocol, "v1")

        class OldWorld(FakeWorld):
            def __init__(self, seed, odour=True):
                super().__init__(seed, odour)
        wmod.World = OldWorld
        self.assertEqual(pe.make_world(wmod, 0, True, "v1").protocol, "v1")
        with self.assertRaises(TypeError):
            pe.make_world(wmod, 0, True, "v2")


class StartRule(unittest.TestCase):
    """Every trial's start is checked against the threshold, from the field and from the nose, and reported."""

    def test_counts_and_seeds(self):
        t = {"odour": [dict(straight_trial(0.1, seed=0), c_start_field=0.2), dict(straight_trial(0.1, seed=1), c_start_field=0.01)],
             "blank": [dict(straight_trial(0.1, seed=0, condition="blank"), c_start_field=0.2),
                       dict(straight_trial(0.1, seed=1, condition="blank"), c_start_field=0.01)],
             "shuffled": [dict(straight_trial(0.1, seed=0, condition="shuffled"), c_start_field=0.2),
                          dict(straight_trial(0.1, seed=1, condition="shuffled"), c_start_field=0.01)]}
        s = pe.summarise(t, protocol="v2")
        sr = s["start_rule"]
        self.assertEqual((sr["n_trials"], sr["n_trials_above_threshold_field"], sr["seeds_above_threshold_field"]), (6, 3, [0]))
        self.assertEqual(sr["n_trials_above_threshold_nose"], 0)      # the straight trials carry c = 0 at the nose
        self.assertEqual(sr["expected_violations"], 0)
        self.assertEqual(s["conditions"]["blank"]["n_start_above_threshold_field"], 1)
        self.assertEqual(s["conditions"]["blank"]["n_start_above_threshold"], 0)
        m = pe.metrics(t["odour"][0], "v2")
        self.assertTrue(m["starts_above_threshold_field"])
        self.assertFalse(m["starts_above_threshold"])
        self.assertAlmostEqual(m["c_start_field"], 0.2)
        # without a field value the nose decides, as under v1
        m = pe.metrics(surge_trial())
        self.assertFalse(m["starts_above_threshold_field"])
        self.assertTrue(math.isnan(m["c_start_field"]))
        self.assertTrue(pe.metrics(dict(surge_trial(), c=np.ones(61)))["starts_above_threshold_field"])
        # run_trial reads it off the world
        tr = pe.run_trial(FakeWorld(0, field=lambda x, y, t: 0.3), FakeFly(), 5, 0, True)
        self.assertAlmostEqual(tr["c_start_field"], 0.3)
        self.assertTrue(pe.metrics(tr)["starts_above_threshold_field"])


class JoMatch(unittest.TestCase):
    """The realised total JO drive per condition and the within-15 % statement."""

    def test_statement(self):
        conds = {"odour": {"mean_jo_total_cell_hz": {"mean": 10000.0}}, "shuffled": {"mean_jo_total_cell_hz": {"mean": 11000.0}}}
        j = pe.jo_match(conds)
        self.assertAlmostEqual(j["relative_difference"], 1000.0 / 10500.0)
        self.assertTrue(j["within_tolerance"])
        self.assertIn("within the 15 %", j["statement"])
        conds["shuffled"]["mean_jo_total_cell_hz"]["mean"] = 15000.0
        j = pe.jo_match(conds)
        self.assertFalse(j["within_tolerance"])
        self.assertIn("NOT within", j["statement"])
        self.assertIsNone(pe.jo_match({"odour": {}})["within_tolerance"])

    def test_from_trial_rates(self):
        def tr(seed, cond, total, left=None, right=None):
            rates = {"jo_total_cell_hz": np.full(40, total), "jo_left_hz": np.full(40, 30.0), "jo_right_hz": np.full(40, 30.0)}
            if left is not None:
                rates["jo_left_cell_hz"] = np.full(40, left)
                rates["jo_right_cell_hz"] = np.full(40, right)
            return dict(straight_trial(0.1, seed=seed, condition=cond), rates=rates)
        t = {"odour": [tr(0, "odour", 9000.0, 4500.0, 4500.0), tr(1, "odour", 11000.0, 5500.0, 5500.0)],
             "blank": [tr(0, "blank", 9000.0), tr(1, "blank", 11000.0)],
             "shuffled": [tr(0, "shuffled", 10500.0), tr(1, "shuffled", 10500.0)]}
        s = pe.summarise(t, protocol="v2", jo_cells=(150, 150))
        jd = s["jo_drive"]
        self.assertAlmostEqual(jd["odour"]["mean_total_cell_hz"], 10000.0)
        self.assertAlmostEqual(jd["odour"]["mean_left_cell_hz"], 5000.0)          # the fly's own per-side number wins
        self.assertAlmostEqual(jd["blank"]["mean_left_cell_hz"], 30.0 * 150)        # else per-cell mean x count
        self.assertAlmostEqual(jd["odour_vs_shuffled"]["relative_difference"], 500.0 / 10250.0)
        self.assertTrue(jd["odour_vs_shuffled"]["within_tolerance"])
        self.assertAlmostEqual(s["disclosures"]["shuffled"]["mean_jo_total_cell_hz"], 10500.0)
        self.assertEqual(pe.jo_cells_of({"constants": {"fly_instance": {"jo_e_left": 100, "jo_e_right": 90, "jo_left": 203, "jo_right": 132}}}), (100, 90))
        self.assertEqual(pe.jo_cells_of({"constants": {"fly_instance": {"jo_left": 203, "jo_right": 132}}}), (203, 132))
        self.assertIsNone(pe.jo_cells_of({"constants": {}}))


class LadderV2(unittest.TestCase):
    def test_ten_to_eight_never_lower(self):
        self.assertEqual(pe.ladder(10, 0.6, 400, 3, 9000.0, (10, 8), 8), (10, "kept"))       # 120 min
        self.assertEqual(pe.ladder(10, 0.8, 400, 3, 9000.0, (10, 8), 8)[0], 8)               # 160 min -> 8
        self.assertEqual(pe.ladder(10, 5.0, 400, 3, 9000.0, (10, 8), 8), (8, "dropped 10 -> 8 (floor)"))
        self.assertEqual(pe.ladder(8, 5.0, 400, 3, 9000.0, (10, 8), 8), (8, "kept"))
        self.assertEqual(pe.PROTOCOLS["v2"]["seed_ladder"], (10, 8))
        self.assertEqual(pe.PROTOCOLS["v2"]["min_seeds"], 8)
        self.assertEqual(pe.PROTOCOLS["v2"]["budget_s"], 9000.0)
        self.assertEqual(pe.PROTOCOLS["v2"]["sim_steps"], 120)
        self.assertEqual(pe.ladder_note({"protocol": "v2", "requested_seeds": 10, "budget_s": 9000.0, "steps": 400}), "as designed")
        note = pe.ladder_note({"protocol": "v2", "requested_seeds": 10, "budget_s": 9000.0, "steps": 400, "sec_per_brain_run": 0.8})
        self.assertIn("would have given 8 seeds (dropped 10 -> 8)", note)


def fake_v1_json(path, npz_trials=None):
    """A minimal v1 record with v1's headline numbers, and optionally a trajectories NPZ next to it."""
    data = {
        "protocol": "v1",
        "run": {"seeds": list(range(10)), "steps": 400, "sim_steps": 60, "sec_per_brain_run": 0.311, "elapsed_s": 3900.0},
        "summary": {
            "predictions": {
                "P1_surge": {"effect": 0.00215, "se": 0.00144, "n_trials": 9, "n_events": 16, "verdict": "not supported"},
                "P2_cast": {"vy": {"effect": -0.00085, "se": 0.00026, "verdict": "not supported"},
                            "heading_change_deg": {"effect": 0.062, "se": 0.188, "verdict": "not supported"},
                            "n_events": 21, "verdict": "not supported"},
                "P3_upwind_progress": {"odour-blank": {"mean": -0.0128, "se": 0.0048, "n": 10, "verdict": "not supported"},
                                       "odour-nowind": {"mean": 0.0379, "se": 0.0130, "n": 10, "verdict": "supported"},
                                       "verdict": "not supported"},
                "P4_source_reached": {"odour": 0.0, "blank": 0.0, "nowind": 0.0, "verdict": "not supported"},
            },
            "conditions": {
                "odour": {"progress": {"mean": 0.0458, "se": 0.0051}, "n_encounters": {"mean": 1.6, "se": 0.4},
                          "n_losses": {"mean": 2.3, "se": 0.5}, "time_in_plume_s": {"mean": 17.0, "se": 1.0},
                          "wall_contacts": {"mean": 0.0, "se": 0.0}, "mean_jo_total_cell_hz": {"mean": 8696.0, "se": 100.0},
                          "n_start_above_threshold": 9, "n_trials": 10, "n_reached": 0},
                "blank": {"progress": {"mean": 0.0586, "se": 0.0071}, "n_trials": 10, "n_reached": 0, "n_start_above_threshold": 0},
                "nowind": {"progress": {"mean": 0.0079, "se": 0.0142}, "mean_jo_total_cell_hz": {"mean": 15361.0, "se": 50.0},
                           "n_trials": 10, "n_reached": 0},
            },
        },
    }
    Path(path).write_text(json.dumps(data), encoding="utf-8")
    if npz_trials is not None:
        pe.save_trajectories(npz_trials, Path(path).with_name(Path(path).name.replace("_experiment.json", "_trajectories.npz")))
    return data


class V1Comparison(unittest.TestCase):
    """The side-by-side table reads v1's numbers from its JSON and never invents a missing one."""

    def v2_summary(self):
        t = {"odour": [straight_trial(0.3, seed=i) for i in range(3)],
             "blank": [straight_trial(0.1, seed=i, condition="blank") for i in range(3)],
             "shuffled": [straight_trial(0.2, seed=i, condition="shuffled") for i in range(3)]}
        return pe.summarise(t, protocol="v2")

    def rows_by_metric(self, table):
        return {r["metric"]: r for r in table["rows"]}

    def test_table_from_a_fake_v1_json(self):
        with tempfile.TemporaryDirectory() as d:
            j = Path(d) / "plume_experiment.json"
            fake_v1_json(j)
            s = self.v2_summary()
            table = pe.v1_comparison(j, s, {"seeds": [0, 1, 2], "steps": 400, "sim_steps": 120, "sec_per_brain_run": 0.62, "elapsed_s": 100.0})
            self.assertTrue(table["available"])
            self.assertEqual((table["v1_control"], table["v2_control"]), ("nowind", "shuffled"))
            self.assertIn("not the same experiment", table["note"])
            r = self.rows_by_metric(table)
            self.assertAlmostEqual(r["P1 surge: effect (m/s upwind, after - before)"]["v1"], 0.00215)
            self.assertIsNone(r["P1 surge: effect (m/s upwind, after - before)"]["v2"])       # no encounters in the straight trials: nan -> None
            self.assertEqual(r["P1 surge: verdict"]["v1"], "not supported")
            self.assertEqual(r["P1 surge: verdict"]["v2"], "undetermined")
            self.assertEqual(r["P1 surge: n events"]["v1"], 16)
            self.assertIn("full windows", r["P1 surge: n events"]["note"])
            self.assertAlmostEqual(r["P3 progress odour - blank (m): mean"]["v1"], -0.0128)
            self.assertAlmostEqual(r["P3 progress odour - blank (m): mean"]["v2"], 0.2, places=9)
            self.assertEqual(r["P3 progress odour - blank (m): verdict"]["v2"], "supported")
            sc = r["P3 progress odour - second control (m): mean"]
            self.assertAlmostEqual(sc["v1"], 0.0379)
            self.assertAlmostEqual(sc["v2"], 0.1, places=9)
            self.assertIn("nowind", sc["note"])
            self.assertIn("shuffled", sc["note"])
            self.assertEqual(r["P4 source reached: odour"]["v1"], 0.0)
            self.assertEqual(r["P4 source reached: second control"]["v1"], 0.0)
            self.assertEqual(r["P4 source reached: second control"]["v2"], 0.0)
            self.assertAlmostEqual(r["odour: progress (m upwind): mean"]["v1"], 0.0458)
            self.assertAlmostEqual(r["odour: progress (m upwind): mean"]["v2"], 0.3, places=9)
            self.assertAlmostEqual(r["odour: total JO drive (cell-Hz): mean"]["v1"], 8696.0)
            self.assertIsNone(r["odour: total JO drive (cell-Hz): mean"]["v2"])                 # the fake fly reported none
            self.assertEqual(r["odour: trials starting above threshold"]["v1"], 9)
            self.assertEqual(r["odour: trials starting above threshold"]["v2"], 0)
            self.assertIsNone(r["odour: facing upwind (fraction of steps)"]["v1"])              # no NPZ: not invented
            self.assertAlmostEqual(r["odour: facing upwind (fraction of steps)"]["v2"], 1.0)
            self.assertAlmostEqual(r["second control (nowind / shuffled): progress (m upwind): mean"]["v1"], 0.0079)
            self.assertAlmostEqual(r["second control (nowind / shuffled): progress (m upwind): mean"]["v2"], 0.2, places=9)
            self.assertEqual(r["run: seeds"]["v1"], 10)
            self.assertEqual(r["run: seeds"]["v2"], 3)
            self.assertEqual(r["run: sim_steps"]["v1"], 60)
            self.assertEqual(r["run: sim_steps"]["v2"], 120)
            self.assertAlmostEqual(r["run: sec_per_brain_run"]["v1"], 0.311)

    def test_facing_upwind_comes_from_the_v1_trajectories_when_present(self):
        with tempfile.TemporaryDirectory() as d:
            j = Path(d) / "plume_experiment.json"
            n = 20
            down = make_trial(0.3 + 0.001 * np.arange(n + 1), np.full(n + 1, 0.15), np.zeros(n + 1), np.zeros(n + 1),
                              seed=1, condition="odour")
            fake_v1_json(j, npz_trials={"odour": [straight_trial(0.1), down],
                                        "blank": [straight_trial(0.1, condition="blank")]})
            table = pe.v1_comparison(j, self.v2_summary())
            r = self.rows_by_metric(table)
            self.assertAlmostEqual(r["odour: facing upwind (fraction of steps)"]["v1"], 0.5)
            self.assertAlmostEqual(r["blank: facing upwind (fraction of steps)"]["v1"], 1.0)
            self.assertIsNone(r["second control (nowind / shuffled): facing upwind (fraction of steps)"]["v1"])
            self.assertEqual(pe.facing_upwind_from_npz(j.with_name("plume_trajectories.npz")), {"odour": 0.5, "blank": 1.0})

    def test_missing_v1_json_is_reported_not_invented(self):
        table = pe.v1_comparison(Path("nowhere") / "plume_experiment.json", self.v2_summary())
        self.assertFalse(table["available"])
        self.assertEqual(table["rows"], [])


class OutputGuard(unittest.TestCase):
    """The published v1 files are never written by a new run, however the prefix is spelled."""

    def test_published_prefix_is_refused(self):
        for bad in (pe.V1_PREFIX, str(pe.V1_PREFIX), pe.out_prefix_from(str(pe.V1_JSON)),
                    Path("build") / "plume" if Path("build").resolve() == pe.BUILD.resolve() else pe.V1_PREFIX):
            with self.assertRaises(ValueError):
                pe.assert_not_published(bad)
        for ok in (pe.BUILD / "plume_v2", pe.BUILD / "plume_quick", pe.BUILD / "plume_v2_quick", pe.BUILD / "plume_rerun",
                   pe.out_prefix_from(str(pe.V1_JSON), quick=True)):
            pe.assert_not_published(ok)
        self.assertEqual([p.name for p in pe.PUBLISHED_V1],
                         ["plume_experiment.json", "plume_trajectories.npz", "plume_trajectories.png", "plume_report.md"])
        self.assertEqual(pe.output_paths(pe.BUILD / "plume_v2")[0], pe.BUILD / "plume_v2_experiment.json")

    def test_write_outputs_and_main_refuse_the_published_prefix(self):
        trials = {"odour": [surge_trial()], "blank": [straight_trial(0.2, seed=1, condition="blank")]}
        watched = list(pe.PUBLISHED_V1) + [pe.V1_PREFIX.with_name("plume_experiment.log")]   # the guard runs before a log is opened
        before = {p: (p.stat().st_mtime_ns if p.exists() else None) for p in watched}
        with self.assertRaises(ValueError):
            pe.write_outputs(trials, {"constants": {"runner": pe.CONSTANTS}}, pe.V1_PREFIX, protocol="v2")
        with self.assertRaises(ValueError):
            pe.write_outputs(trials, {"constants": {"runner": pe.CONSTANTS}}, str(pe.V1_JSON)[:-len("_experiment.json")])
        real_ram = pe.free_ram_gb
        pe.free_ram_gb = lambda: (_ for _ in ()).throw(AssertionError("main must refuse before touching RAM or a brain"))
        try:
            self.assertEqual(pe.main(["--protocol", "v2", "--out", str(pe.V1_PREFIX)]), 4)
            self.assertEqual(pe.main(["--protocol", "v1", "--out", str(pe.V1_JSON)]), 4)
            self.assertEqual(pe.main(["--reanalyse", str(pe.V1_JSON)]), 4)          # without --allow-published
        finally:
            pe.free_ram_gb = real_ram
        after = {p: (p.stat().st_mtime_ns if p.exists() else None) for p in watched}
        self.assertEqual(before, after)

    def test_v2_default_prefix_is_plume_v2(self):
        self.assertEqual(pe.PROTOCOLS["v2"]["default_out"], "plume_v2")
        self.assertEqual(pe.out_prefix_from(str(pe.BUILD / "plume_v2"), quick=True), pe.BUILD / "plume_v2_quick")
        self.assertEqual(pe.out_prefix_from("build/plume_v2_experiment.json"), Path("build/plume_v2"))


class MainWithFakesV2(unittest.TestCase):
    """The v2 CLI end to end with fakes: three v2 conditions, the record's v2 blocks, and a reanalysis round trip."""

    def test_v2_quick_run(self):
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
        fake_plume.start_rule_v2_check = lambda n: {"n_start_above_threshold": 9, "seeds_above_threshold": [0, 16], "n_seeds": n}
        fake_fly = types.ModuleType("plume_fly")
        made = []

        class Fly(CarryingFly):
            def __init__(self, fb, gains, protocol="v1", **kw):
                super().__init__(turn=0.1, speed=0.5)
                self.protocol, self.kw = protocol, kw
                made.append(self)

            def step(self, c, phi, wind_sense=True, seed=0):
                turn, speed, info = super().step(c, phi, wind_sense, seed)
                info.update({"jo_total_cell_hz": 10000.0 + 100.0 * math.cos(phi), "jo_left_cell_hz": 5000.0, "jo_right_cell_hz": 5000.0})
                return turn, speed, info

            def describe(self):
                return {"kw": self.kw, "protocol": self.protocol, "jo_e_left": 100, "jo_e_right": 90}

        fake_fly.PlumeFly = Fly
        saved = {k: sys.modules.get(k) for k in ("flysim", "calibration", "plume", "plume_fly")}
        sys.modules.update({"flysim": fake_flysim, "calibration": fake_cal, "plume": fake_plume, "plume_fly": fake_fly})
        real_ram, real_v1 = pe.free_ram_gb, pe.V1_JSON
        pe.free_ram_gb = lambda: 100.0
        try:
            with tempfile.TemporaryDirectory() as d:
                v1 = Path(d) / "v1" / "plume_experiment.json"
                v1.parent.mkdir()
                fake_v1_json(v1)
                pe.V1_JSON = v1
                out = Path(d) / "plume_v2"
                rc = pe.main(["--protocol", "v2", "--quick", "1", "--out", str(out)])
                self.assertEqual(rc, 0)
                self.assertEqual(len(made), 1)
                self.assertEqual(made[0].protocol, "v2")
                self.assertEqual(made[0].kw["sim_steps"], 120)
                self.assertEqual(made[0].kw["smooth_tau_s"], 0.0)          # the fly passes through; the runner smooths once
                self.assertEqual(made[0].begun, [pe.trial_seed(0)] * 3 + [pe.trial_seed(1)] * 3)
                log = (Path(d) / "plume_v2_quick_experiment.log").read_text(encoding="utf-8").splitlines()
                trial_lines = [l for l in log if " trial " in l]
                self.assertEqual(len(trial_lines), 6)
                self.assertIn("seed=1 cond=shuffled", trial_lines[-1])
                self.assertIn("upwind=", trial_lines[0])
                self.assertTrue(any("protocol v2" in l for l in log))
                self.assertTrue(any("start rule v2 over 50 seeds: 9" in l for l in log))
                self.assertTrue(any("JO drive:" in l for l in log))
                data = json.loads((Path(d) / "plume_v2_quick_experiment.json").read_text(encoding="utf-8"))
                self.assertEqual(data["protocol"], "v2")
                self.assertFalse(data["partial"])
                self.assertEqual(data["run"]["protocol"], "v2")
                self.assertEqual(data["run"]["seeds"], [0, 1])
                self.assertEqual(data["run"]["sim_steps"], 120)
                self.assertEqual(data["run"]["smooth_tau_s"], 0.15)
                self.assertTrue(data["run"]["state_carry"])
                self.assertEqual(data["run"]["begin_trial_method"], "begin_trial")
                self.assertEqual(data["run"]["smoothing_applied_by"], "runner")
                self.assertEqual(data["run"]["shuffled_wind_applied_by"], "runner")
                self.assertEqual(data["run"]["design_seeds"], 10)
                self.assertEqual(data["run"]["design_budget_s"], 9000.0)
                self.assertEqual([c["letter"] for c in data["protocol_changes"]], list("ABCDEFGH"))
                self.assertTrue(all("named_in_v1_report" in c for c in data["protocol_changes"]))
                self.assertEqual(data["constants"]["world_start_rule_v2"]["n_start_above_threshold"], 9)
                self.assertEqual(data["constants"]["fly_instance"]["protocol"], "v2")
                self.assertEqual(len(data["simulator_limits"]), 4)
                self.assertFalse(any("restarted from rest" in s for s in data["simulator_limits"]))
                self.assertTrue(any("carries its state" in s for s in data["simulator_limits"]))
                s = data["summary"]
                self.assertEqual(list(s["conditions"]), ["odour", "blank", "shuffled"])
                self.assertEqual(set(s["paired"]), {"odour-blank", "odour-shuffled"})
                self.assertEqual(s["predictions"]["P1_surge"]["rule"], "full_window")
                self.assertIn("odour_vs_shuffled", s["jo_drive"])
                self.assertIsNotNone(s["jo_drive"]["odour_vs_shuffled"]["within_tolerance"])
                self.assertAlmostEqual(s["jo_drive"]["odour"]["mean_left_cell_hz"], 5000.0)
                self.assertEqual(s["start_rule"]["n_trials"], 6)
                self.assertEqual(s["start_rule"]["n_trials_above_threshold_field"], 0)
                self.assertIn("turn_raw", s["conditions"]["odour"]["commands"])
                self.assertLess(s["conditions"]["odour"]["mean_speed_cmd"]["mean"], 0.5)      # smoothed from rest
                self.assertAlmostEqual(s["conditions"]["odour"]["mean_speed_raw"]["mean"], 0.5)
                self.assertTrue(data["v1_comparison"]["available"])
                self.assertEqual(data["v1_comparison"]["v2_control"], "shuffled")
                rows = {r["metric"]: r for r in data["v1_comparison"]["rows"]}
                self.assertEqual(rows["run: seeds"]["v2"], 2)
                self.assertEqual(rows["run: sim_steps"]["v2"], 120)
                self.assertEqual(len(s["per_trial"]), 6)
                self.assertEqual({r["condition"] for r in s["per_trial"]}, {"odour", "blank", "shuffled"})
                with np.load(Path(d) / "plume_v2_quick_trajectories.npz") as z:
                    self.assertEqual(list(z["condition"]), ["odour", "odour", "blank", "blank", "shuffled", "shuffled"])
                    for k in ("turn", "speed", "turn_raw", "speed_raw", "phi", "phi_fly", "c_start_field"):
                        self.assertIn(k, z.files)
                    self.assertAlmostEqual(float(z["speed_raw"][0, 0]), 0.5)
                    self.assertLess(float(z["speed"][0, 0]), 0.5)
                    self.assertNotEqual(float(z["phi_fly"][4, 0]), float(z["phi"][4, 0]))   # shuffled rows
                    self.assertEqual(float(z["phi_fly"][0, 0]), float(z["phi"][0, 0]))
                self.assertTrue((Path(d) / "plume_v2_quick_trajectories.png").exists())
                # the reanalysis of a v2 record keeps the protocol and the numbers, rebuilding commands from the NPZ
                _, j1, _, png = pe.reanalyse(str(Path(d) / "plume_v2_quick_experiment.json"), note="v2 round trip")
                new = json.loads(j1.read_text(encoding="utf-8"))
                self.assertEqual(new["protocol"], "v2")
                same, worst, mismatches = pe.compare_preregistered(data["summary"]["predictions"], new["summary"]["predictions"], protocol="v2")
                self.assertTrue(same, mismatches)
                self.assertTrue(new["run"]["reanalyses"][-1]["preregistered_metrics_unchanged"])
                self.assertEqual(new["run"]["design_seeds"], 10)
                self.assertAlmostEqual(new["summary"]["conditions"]["odour"]["mean_speed_raw"]["mean"], 0.5)
                self.assertAlmostEqual(new["summary"]["conditions"]["odour"]["mean_speed_cmd"]["mean"],
                                       data["summary"]["conditions"]["odour"]["mean_speed_cmd"]["mean"])
                self.assertEqual(new["summary"]["start_rule"]["n_trials"], 6)
                self.assertIn("turn_raw", new["summary"]["conditions"]["odour"]["commands"])
                self.assertEqual(new["summary"]["conditions"]["odour"]["commands"]["turn_raw"]["n"],
                                 data["summary"]["conditions"]["odour"]["commands"]["turn_raw"]["n"])
                # a v1 quick run through the same fakes is untouched by the v2 machinery
                rc = pe.main(["--quick", "1", "--out", str(Path(d) / "plume")])
                self.assertEqual(rc, 0)
                d1 = json.loads((Path(d) / "plume_quick_experiment.json").read_text(encoding="utf-8"))
                self.assertEqual(d1["protocol"], "v1")
                self.assertEqual(list(d1["summary"]["conditions"]), ["odour", "blank", "nowind"])
                self.assertNotIn("v1_comparison", d1)
                self.assertEqual(made[-1].protocol, "v1")
                self.assertEqual(made[-1].begun, [])
                self.assertNotIn("smooth_tau_s", made[-1].kw)              # v1: no smoothing anywhere
                self.assertEqual(d1["run"]["smoothing_applied_by"], "none")
                self.assertAlmostEqual(d1["summary"]["conditions"]["odour"]["mean_speed_cmd"]["mean"], 0.5)
        finally:
            pe.free_ram_gb, pe.V1_JSON = real_ram, real_v1
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

    def test_v2_refuses_a_fly_or_world_without_the_protocol(self):
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
        fake_fly = types.ModuleType("plume_fly")

        class OldFly(FakeFly):
            def __init__(self, fb, gains, **kw):
                super().__init__()
        fake_fly.PlumeFly = OldFly
        saved = {k: sys.modules.get(k) for k in ("flysim", "calibration", "plume", "plume_fly")}
        sys.modules.update({"flysim": fake_flysim, "calibration": fake_cal, "plume": fake_plume, "plume_fly": fake_fly})
        real_ram = pe.free_ram_gb
        pe.free_ram_gb = lambda: 100.0
        try:
            with tempfile.TemporaryDirectory() as d:
                self.assertEqual(pe.main(["--protocol", "v2", "--quick", "1", "--out", str(Path(d) / "plume_v2")]), 3)
                log = (Path(d) / "plume_v2_quick_experiment.log").read_text(encoding="utf-8")
                self.assertIn("refusing to run", log)
                self.assertFalse((Path(d) / "plume_v2_quick_experiment.json").exists())
        finally:
            pe.free_ram_gb = real_ram
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v


class V2Wiring(unittest.TestCase):
    """The real fly's names are accepted, and the fly's echoed raw commands stay the runner's."""

    def test_reset_state_counts_as_a_trial_start_method(self):
        f = CarryingFly(method="reset_state")
        tr = pe.run_trial(FakeWorld(2), f, 3, 2, True, state_carry=True)
        self.assertEqual((tr["begin_trial_method"], f.begun), ("reset_state", [pe.trial_seed(2)]))
        self.assertEqual(pe.BEGIN_TRIAL_NAMES[0], "begin_trial")      # the alias is found first when both exist

    def test_echoed_raw_commands_are_not_recorded_twice(self):
        class EchoFly(FakeFly):
            def step(self, c, phi, wind_sense=True, seed=0):
                turn, speed, info = super().step(c, phi, wind_sense, seed)
                info.update({"turn_raw": turn, "speed_raw": speed, "phi_drive": phi, "trial_seed": 1})
                return turn, speed, info
        tr = pe.run_trial(FakeWorld(0), EchoFly(turn=0.2, speed=0.4), 5, 0, True, smooth_tau_s=0.15)
        self.assertNotIn("turn_raw", tr["rates"])
        self.assertNotIn("speed_raw", tr["rates"])
        self.assertIn("phi_drive", tr["rates"])
        self.assertEqual(list(tr["turn_raw"]), [0.2] * 5)
        self.assertLess(tr["turn"][0], 0.2)
        m = pe.metrics(tr, "v2")
        self.assertAlmostEqual(m["mean_turn_raw"], 0.2)


if __name__ == "__main__":
    unittest.main()
