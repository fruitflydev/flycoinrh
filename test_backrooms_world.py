"""
The backrooms world on a fake brain: no graph is loaded, the eye is a fake
that reads the window's mean luminance, and the brain echoes each driven
cell's rate back as its firing rate so the plumbing from geometry to drive to
record can be checked exactly. One optional class loads the real brain once,
and only when asked:

  py -m pytest -q test_backrooms_world.py
  BACKROOMS_REAL_BRAIN=1 py -m pytest -q -s test_backrooms_world.py -k RealBrain

What matters and why:
  * the arena is deterministic in its seed, caps speed and turn, clamps at
    the walls, and its bearing helper has the sign the docstring promises;
  * the other fly's ellipse moves across the frame with bearing, grows as
    distance shrinks, and is not in the eye window when it is behind;
  * smell and sound fall with distance and are zero at 20 mm and beyond;
  * two bodies on one brain keep separate states and separate seeds;
  * a whole room step is a pure function of the seed;
  * the module imports no language model and no network library.
"""
import copy
import json
import math
import os
import re
import time
import unittest
from pathlib import Path

import numpy as np

import backrooms_dictionary as bd
import backrooms_world as bw
from plume_fly import MOTOR_NAMES, PlumeFly

ROOT = Path(__file__).parent

FAKE_TYPES = (
    ["L1"] * 6 + ["L2"] * 6 + ["ORN_DA1"] * 4 + ["JO-A1"] * 3 + ["JO-B1_a"] * 2
    + ["ps1 MN"] * 2 + ["i1 MN", "hg1 MN", "DLMn a, b"]
    + ["DNa02", "DNa02", "DNa01", "DNa01", "MDN", "MDN", "DNp09", "DNp09"]
    + ["pC1_1a", "pC1_1a", "LC10a", "LC10a", "KCab-m", "KCab-m", "ORN_DM1"]
)


class FakeBrain:
    """
    Just enough of FlyBrain: types, bodies, n, where(type_re=) and a run()
    that returns each recorded cell's drive rate (plus any `spont` rate set
    by index) as its firing rate. The returned state counts the windows
    chained on it and remembers the seed of the window that started the
    chain, so carrying and seeding can be checked per body.
    """

    def __init__(self, types=FAKE_TYPES, spont=None):
        self.types = np.array(types, dtype=str)
        self.n = len(self.types)
        self.bodies = np.arange(self.n) + 10_000
        self.spont = dict(spont or {})
        self.calls = []

    def where(self, type_re=None, **_):
        rx = re.compile(type_re, re.I)
        return np.flatnonzero([bool(rx.search(t)) for t in self.types])

    def run(self, drive, steps, gains=None, record=None, seed=0, spike_log=False, state=None):
        self.calls.append(dict(steps=steps, seed=seed, state=copy.deepcopy(state),
                               keys=[tuple(k) for k in drive]))
        per = np.zeros(self.n)
        for k, r in drive.items():
            rr = np.asarray(r, dtype=np.float64)
            per[list(k)] = rr if rr.ndim else float(rr)
        for i, hz in self.spont.items():
            per[int(i)] += float(hz)
        out = {name: per[np.asarray(idx, dtype=np.int64)] for name, idx in (record or {}).items()}
        out["_fired"] = np.flatnonzero(per > 0)
        windows = 0 if state is None else int(state["v"][0])
        first_seed = seed if state is None else int(state["v"][1])
        out["_state"] = {"v": np.array([windows + 1, first_seed]),
                         "refr": np.zeros(1, dtype=np.int32), "rng": {"windows": windows + 1}}
        return out


class FakeEye:
    """L1 cells at the eye window's mean luminance x 180 Hz, L2 at (1 - mean) x 108 Hz."""

    def __init__(self, fb, fov=(bw.EYE_FOV_W, bw.EYE_FOV_H)):
        self.on_idx = fb.where(type_re="^L1$")
        self.off_idx = fb.where(type_re="^L2$")
        self.fov_w, self.fov_h = fov

    def look(self, img, cx, cy):
        H, W = img.shape
        x0, x1 = int(cx - self.fov_w / 2), int(cx + self.fov_w / 2)
        y0, y1 = int(cy - self.fov_h / 2), int(cy + self.fov_h / 2)
        m = float(img[max(y0, 0):min(y1, H), max(x0, 0):min(x1, W)].mean())
        return {tuple(self.on_idx): np.full(len(self.on_idx), m * 180.0, dtype=np.float32),
                tuple(self.off_idx): np.full(len(self.off_idx), (1.0 - m) * 108.0, dtype=np.float32)}


def fake_sides(fb):
    """Paired descending neurons: the first of each type is L, the second R."""
    side = np.array([""] * fb.n, dtype=object)
    for t in ("DNa02", "DNa01"):
        idx = fb.where(type_re=rf"^{t}$")
        side[idx[0]], side[idx[1]] = "L", "R"
    return side.astype(str)


def make_parts(spont=None):
    fb = FakeBrain(spont=spont)
    eye = FakeEye(fb)
    groups = bd.present_groups(fb)
    motor = bw.motor_groups(fb, fake_sides(fb))
    return fb, eye, groups, motor


def make_body(name="A", seed=0, spont=None, groups=None):
    fb, eye, g, motor = make_parts(spont)
    return fb, bw.FlyBody(name, fb, eye, groups if groups is not None else g, motor, seed=seed)


def flat(viewer_xy=(10.0, 10.0), heading=0.0):
    return bw.Fly("A", viewer_xy[0], viewer_xy[1], heading)


def other_at(viewer, bearing_deg, distance_mm, heading=None):
    """A fly placed at the given bearing and distance from the viewer."""
    ang = viewer.heading + math.radians(bearing_deg)
    x, y = viewer.x + distance_mm * math.cos(ang), viewer.y + distance_mm * math.sin(ang)
    return bw.Fly("B", x, y, ang if heading is None else heading)


def dark(img, ground=bw.GROUND_GREY):
    return np.argwhere(img < ground)


def window(img, ch):
    cx, cy = ch.gaze
    return img[int(cy - ch.fov_h / 2):int(cy + ch.fov_h / 2), int(cx - ch.fov_w / 2):int(cx + ch.fov_w / 2)]


# =============================================================================

class Honesty(unittest.TestCase):
    def test_module_imports_no_model_and_no_network(self):
        src = (ROOT / "backrooms_world.py").read_text(encoding="utf-8")
        for bad in ("openai", "anthropic", "transformers", "llama_cpp", "requests",
                    "httpx", "urllib", "socket", "aiohttp"):
            self.assertIsNone(re.search(rf"^\s*(import|from)\s+{bad}\b", src, re.M),
                              f"backrooms_world imports {bad}")

    def test_describe_is_plain_json(self):
        fb, eye, groups, motor = make_parts()
        room = bw.Room(fb, eye, groups, motor, seed=3)
        json.dumps(room.describe())

    def test_describe_measures_the_sight_channel(self):
        fb, eye, groups, motor = make_parts()
        room = bw.Room(fb, eye, groups, motor, seed=3)
        d = room.describe()
        sm = d["sight_measured"]
        self.assertEqual(len(sm["cases"]), 10)
        self.assertEqual(sm["ground"]["L1_columns"], 6)
        for c in sm["cases"]:
            for k in ("L1_columns_on_fly", "L2_columns_on_fly", "L1_mean_minus_ground_hz", "in_window"):
                self.assertIn(k, c)
        near = [c for c in sm["cases"] if c["distance_mm"] == 5.0 and c["pose"] == "side_on"][0]
        self.assertTrue(near["in_window"])
        self.assertGreaterEqual(near["L1_columns_on_fly"], 1)      # the fake eye's mean drops
        self.assertLess(near["L1_mean_minus_ground_hz"], 0.0)
        self.assertAlmostEqual(d["timing"]["brain_under_run_factor"], 50.0 / 12.0)
        self.assertIn("4.2x faster", d["timing"]["note"])
        self.assertIn("side_scales", d["bodies"]["A"])
        json.dumps(d)


class ArenaGeometry(unittest.TestCase):
    def test_starts_are_seeded_and_inside_the_margin(self):
        a1, a2, a3 = bw.Arena(seed=5), bw.Arena(seed=5), bw.Arena(seed=6)
        for f, g in zip(a1.flies, a2.flies):
            self.assertEqual((f.x, f.y, f.heading), (g.x, g.y, g.heading))
        self.assertNotEqual((a1.A.x, a1.A.y), (a3.A.x, a3.A.y))
        for f in a1.flies + a3.flies:
            self.assertGreaterEqual(f.x, bw.START_MARGIN_MM)
            self.assertLessEqual(f.x, bw.ARENA_MM - bw.START_MARGIN_MM)
            self.assertGreaterEqual(f.y, bw.START_MARGIN_MM)
            self.assertLessEqual(f.y, bw.ARENA_MM - bw.START_MARGIN_MM)
            self.assertGreaterEqual(f.heading, -math.pi)
            self.assertLess(f.heading, math.pi)

    def test_bearing_sign_and_distance(self):
        a = bw.Arena(start=[(5.0, 5.0, 0.0), (5.0, 10.0, 0.0)])     # A faces +x, B is straight "up"
        self.assertAlmostEqual(a.distance(), 5.0)
        self.assertAlmostEqual(a.bearing("A"), 90.0)                  # to A's left
        self.assertAlmostEqual(a.bearing("B"), -90.0)                 # A is to B's right
        a = bw.Arena(start=[(5.0, 5.0, 0.0), (10.0, 5.0, 0.0)])
        self.assertAlmostEqual(a.bearing("A"), 0.0)                   # dead ahead
        self.assertAlmostEqual(a.bearing("B"), -180.0)                # A is behind B
        a = bw.Arena(start=[(5.0, 5.0, 0.0), (5.0, 0.0, math.pi / 2)])
        self.assertAlmostEqual(a.bearing("A"), -90.0)                 # to A's right
        self.assertAlmostEqual(a.bearing("B"), 0.0)                   # B faces +y, A is ahead
        g = a.geometry()
        self.assertEqual(set(g), {"t", "time_s", "A", "B", "distance_mm", "bearing_deg",
                                  "turned_deg", "moved_mm"})

    def test_speed_and_turn_caps(self):
        a = bw.Arena(start=[(10.0, 10.0, 0.0), (2.0, 2.0, 0.0)])
        a.step((0.0, 1.0), (0.0, 0.0))
        self.assertAlmostEqual(a.A.x, 11.0)                           # 20 mm/s x 50 ms
        a.step((0.0, 5.0), (0.0, 0.0))                                # clipped to 1
        self.assertAlmostEqual(a.A.x, 12.0)
        a.step((1.0, 0.0), (0.0, 0.0))                                # right turn = clockwise
        self.assertAlmostEqual(math.degrees(a.A.heading), -18.0)      # 360 deg/s x 50 ms
        a.step((-3.0, 0.0), (0.0, 0.0))                               # clipped to -1: back to 0
        self.assertAlmostEqual(math.degrees(a.A.heading), 0.0)
        self.assertEqual(a.t, 4)
        self.assertAlmostEqual(a.time_s, 0.2)
        self.assertEqual((a.B.x, a.B.y), (2.0, 2.0))                  # B was told to stand still

    def test_turn_then_walk_and_backing_up(self):
        a = bw.Arena(start=[(10.0, 10.0, 0.0), (2.0, 2.0, 0.0)])
        a.step((1.0, 1.0), (0.0, -1.0))
        h = math.radians(-18.0)
        self.assertAlmostEqual(a.A.x, 10.0 + math.cos(h))
        self.assertAlmostEqual(a.A.y, 10.0 + math.sin(h))
        self.assertAlmostEqual(a.B.x, 1.0)                            # backed up along +x
        g = a.geometry()
        self.assertAlmostEqual(g["turned_deg"]["A"], -18.0)
        self.assertAlmostEqual(g["moved_mm"]["B"], -1.0)

    def test_walls_clamp(self):
        a = bw.Arena(start=[(19.5, 10.0, 0.0), (10.0, 0.5, -math.pi / 2)])
        for _ in range(5):
            a.step((0.0, 1.0), (0.0, 1.0))
        self.assertEqual(a.A.x, 20.0)
        self.assertAlmostEqual(a.A.y, 10.0)
        self.assertEqual(a.B.y, 0.0)
        self.assertAlmostEqual(a.B.x, 10.0)

    def test_trajectory_is_deterministic_in_seed_and_commands(self):
        cmds = [((math.sin(k), math.cos(k)), (math.cos(k), -math.sin(k))) for k in range(30)]
        logs = []
        for _ in range(2):
            a = bw.Arena(seed=11)
            logs.append([a.step(*c) for c in cmds])
        self.assertEqual(json.dumps(logs[0]), json.dumps(logs[1]))


class Sight(unittest.TestCase):
    def setUp(self):
        self.ch = bw.Channels()
        self.v = flat()

    def test_frame_is_the_ground_with_one_dark_ellipse(self):
        img = self.ch.sight_frame(self.v, other_at(self.v, 0.0, 5.0))
        self.assertEqual(img.shape, (bw.FRAME_H, bw.FRAME_W))
        self.assertEqual(img.dtype, np.float32)
        vals = np.unique(img)
        self.assertEqual(list(vals), [np.float32(bw.FLY_GREY), np.float32(bw.GROUND_GREY)])
        self.assertGreater(len(dark(img)), 0)

    def test_ellipse_moves_across_the_frame_with_bearing(self):
        xs = []
        for b in (-60.0, -30.0, 0.0, 30.0, 60.0):
            img = self.ch.sight_frame(self.v, other_at(self.v, b, 5.0))
            px = dark(img)
            self.assertGreater(len(px), 0, b)
            xs.append(float(px[:, 1].mean()))
        # positive bearing = the other is to the viewer's left = left of centre
        self.assertTrue(all(x1 < x0 for x0, x1 in zip(xs, xs[1:])), xs)
        self.assertAlmostEqual(xs[2], bw.FRAME_W / 2, delta=1.0)
        e = self.ch.ellipse_of(self.v, other_at(self.v, 30.0, 5.0))
        self.assertAlmostEqual(e["cx"], bw.FRAME_W / 2 - 30.0 * bw.PX_PER_DEG)

    def test_ellipse_grows_as_distance_shrinks(self):
        areas, rxs, raws, cys = [], [], [], []
        for d in (15.0, 10.0, 5.0, 2.0, 1.0):
            img = self.ch.sight_frame(self.v, other_at(self.v, 0.0, d))
            areas.append(len(dark(img)))
            e = self.ch.ellipse_of(self.v, other_at(self.v, 0.0, d))
            rxs.append(e["rx"])
            raws.append(e["rx_raw"])
            cys.append(e["cy"])
        # the geometry grows monotonically; the drawn size is floored at one
        # column spacing, so it is flat at long range and grows from there
        self.assertTrue(all(r1 > r0 for r0, r1 in zip(raws, raws[1:])), raws)
        self.assertTrue(all(r1 >= r0 for r0, r1 in zip(rxs, rxs[1:])), rxs)
        self.assertTrue(all(a1 >= a0 for a0, a1 in zip(areas, areas[1:])), areas)
        self.assertTrue(all(a1 > a0 for a0, a1 in zip(areas[1:], areas[2:])), areas)
        self.assertEqual(rxs[0], bw.MIN_RADIUS_PX)
        self.assertGreater(rxs[-1], bw.MIN_RADIUS_PX)
        # a nearer fly on the floor sits lower in the picture
        self.assertTrue(all(c1 > c0 for c0, c1 in zip(cys, cys[1:])), cys)
        self.assertTrue(all(c > bw.FRAME_H / 2 for c in cys))

    def test_a_fly_behind_is_not_in_the_eye_window(self):
        behind = other_at(self.v, 180.0, 3.0)
        e = self.ch.ellipse_of(self.v, behind)
        self.assertFalse(e["in_window"])
        img = self.ch.sight_frame(self.v, behind)
        self.assertTrue((window(img, self.ch) == np.float32(bw.GROUND_GREY)).all())
        self.assertGreater(len(dark(img)), 0)            # drawn on the frame, just not seen
        ahead = other_at(self.v, 0.0, 3.0)
        self.assertTrue(self.ch.ellipse_of(self.v, ahead)["in_window"])
        self.assertGreater(len(dark(window(self.ch.sight_frame(self.v, ahead), self.ch))), 0)

    def test_side_on_is_wider_than_head_on(self):
        los = 0.0                                        # the other is dead ahead along +x
        side_on = other_at(self.v, 0.0, 5.0, heading=los + math.pi / 2)
        head_on = other_at(self.v, 0.0, 5.0, heading=los + math.pi)
        rs, rh = self.ch.ellipse_of(self.v, side_on)["rx"], self.ch.ellipse_of(self.v, head_on)["rx"]
        self.assertGreater(rs, rh)
        self.assertAlmostEqual(rs / rh, bw.BODY_LENGTH_MM / bw.BODY_WIDTH_MM, delta=0.1)

    def test_a_far_fly_is_floored_at_one_column_spacing(self):
        # at 20 mm the raw ellipse is 6 x 2.4 px, under one column spacing
        # (9.0 px for the eye's 892 distinct positions on 300 x 210): it
        # would fall between the eye's point samples, so both semi-axes are
        # floored
        self.assertAlmostEqual(bw.column_spacing_px(892), 9.0, delta=0.1)
        self.assertAlmostEqual(bw.MIN_RADIUS_PX, bw.column_spacing_px(bw.N_EYE_COLUMNS), delta=0.1)
        self.assertEqual(bw.N_EYE_COLUMNS, 892)
        # distinct positions, not cells: two lobes on the same coordinates count once
        class Eye:
            on_uv = (np.array([0.1, 0.5, 0.1, 0.9]), np.array([0.2, 0.5, 0.2, 0.9]))
        self.assertEqual(bw.distinct_columns(Eye()), 3)
        self.assertEqual(bw.distinct_columns(object()), bw.N_EYE_COLUMNS)
        e = self.ch.ellipse_of(self.v, other_at(self.v, 0.0, 20.0))
        self.assertLess(e["rx_raw"], bw.MIN_RADIUS_PX)
        self.assertLess(e["ry_raw"], bw.MIN_RADIUS_PX)
        self.assertEqual((e["rx"], e["ry"]), (bw.MIN_RADIUS_PX, bw.MIN_RADIUS_PX))
        near = self.ch.ellipse_of(self.v, other_at(self.v, 0.0, 2.0))
        self.assertEqual((near["rx"], near["ry"]), (near["rx_raw"], near["ry_raw"]))
        raw = bw.Channels(min_radius_px=0.0).ellipse_of(self.v, other_at(self.v, 0.0, 20.0))
        self.assertEqual((raw["rx"], raw["ry"]), (raw["rx_raw"], raw["ry_raw"]))
        self.assertEqual(self.ch.describe()["min_radius_px"], bw.MIN_RADIUS_PX)

    def test_the_ground_gives_the_ellipse_contrast(self):
        # the first draft's ground of 0.05 gave L1 9 vs 0 Hz and no measurable
        # signal; a mid-grey ground gives 90 vs 0 Hz on L1 and 54 vs 108 on L2
        self.assertEqual(bw.GROUND_GREY, 0.5)
        c = self.ch.describe()["retinal_contrast_hz"]
        self.assertGreaterEqual(c["on_ground"] - c["on_fly"], 0.5 * bw.EYE_MAX_HZ)
        self.assertGreaterEqual(c["off_fly"] - c["off_ground"], 0.25 * bw.EYE_MAX_HZ)
        self.assertIn("one pixel per lamina column", self.ch.describe()["sampling"])
        self.assertIn("875 of 875", self.ch.describe()["sampling"])

    def test_flies_on_top_of_each_other_still_render(self):
        img = self.ch.sight_frame(self.v, bw.Fly("B", self.v.x, self.v.y, 0.0))
        self.assertEqual(img.shape, (bw.FRAME_H, bw.FRAME_W))
        self.assertTrue(np.isfinite(img).all())


class SmellAndSound(unittest.TestCase):
    def setUp(self):
        self.ch = bw.Channels()

    def test_smell_falls_linearly_and_is_zero_from_20_mm(self):
        s = self.ch.smell_hz
        self.assertAlmostEqual(s(0.0), 200.0)
        self.assertAlmostEqual(s(5.0), 150.0)
        self.assertAlmostEqual(s(10.0), 100.0)
        self.assertAlmostEqual(s(19.0), 10.0)
        for d in (20.0, 20.5, 25.0, 28.3, 100.0):
            self.assertEqual(s(d), 0.0, d)
        vals = [s(d) for d in np.linspace(0, 30, 61)]
        self.assertTrue(all(b <= a for a, b in zip(vals, vals[1:])))

    def test_sound_is_proportional_to_song_capped_and_falls_with_distance(self):
        h, full = self.ch.sound_hz, bw.SONG_FULL_HZ
        self.assertAlmostEqual(full, 8 * 1000.0 / 2.2)          # 3,636 Hz: 8 cells at the refractory ceiling
        self.assertAlmostEqual(bw.song_full_hz(3, 2.2), 3 * 1000.0 / 2.2)
        self.assertAlmostEqual(h(full, 0.0), 200.0)
        self.assertAlmostEqual(h(full / 2, 0.0), 100.0)
        self.assertAlmostEqual(h(full / 4, 0.0), 50.0)
        self.assertAlmostEqual(h(2 * full, 0.0), 200.0)         # capped at the ceiling
        self.assertAlmostEqual(h(full, 10.0), 100.0)
        self.assertAlmostEqual(h(full / 2, 10.0), 50.0)
        self.assertAlmostEqual(h(2417.0, 5.88), 200.0 * 2417.0 / full * (1 - 5.88 / 20.0))
        for d in (20.0, 21.0, 30.0):
            self.assertEqual(h(full, d), 0.0, d)
        self.assertEqual(h(0.0, 1.0), 0.0)
        self.assertEqual(h(-50.0, 1.0), 0.0)
        vals = [h(full, d) for d in np.linspace(0, 30, 61)]
        self.assertTrue(all(b <= a for a, b in zip(vals, vals[1:])))


class Bodies(unittest.TestCase):
    def test_drive_is_eye_plus_smell_plus_sound(self):
        fb, body = make_body()
        img = bw.Channels().sight_frame(flat(), other_at(flat(), 0.0, 4.0))
        d = body.drive(img, 150.0, 60.0)
        orn = tuple(fb.where(type_re="^ORN_DA1$").tolist())
        jo = tuple(np.unique(np.concatenate([fb.where(type_re="^JO-A"), fb.where(type_re="^JO-B")])).tolist())
        # no sides given: one uniform per-cell rate, as an array the brain takes per cell
        np.testing.assert_allclose(d[orn], 150.0)
        np.testing.assert_allclose(d[jo], 60.0)
        self.assertEqual(len(d[orn]), 4)
        self.assertEqual(len(d[jo]), 5)
        on, off = tuple(body.eye.on_idx), tuple(body.eye.off_idx)
        self.assertIn(on, d)
        self.assertIn(off, d)
        self.assertEqual(len(d), 4)
        self.assertTrue((body.drive(img, -5.0, -1.0)[orn] == 0.0).all())     # never a negative rate
        # the fake eye saw the ellipse: the window mean is below the ground grey
        self.assertLess(float(np.asarray(d[on]).mean()), bw.GROUND_GREY * 180.0)
        self.assertFalse(body.describe()["side_scales"]["ORN_DA1"]["equalised"])

    def test_sides_equalise_the_total_cell_hz_per_antenna(self):
        """
        ORN_DA1 3 L / 1 R, JO-A 1 L / 2 R, JO-B 1 L / 1 R on the fake brain:
        with one uniform rate the right antenna would get a third of the
        left's cVA and twice its JO-A; with root sides given, each side's
        total is the same, the nominal rate is what the record reports, and
        the group's mean per-cell rate is unchanged.
        """
        fb, eye, groups, motor = make_parts()
        sides = np.array([""] * fb.n, dtype=object)
        orn = fb.where(type_re="^ORN_DA1$"); joa = fb.where(type_re="^JO-A"); job = fb.where(type_re="^JO-B")
        sides[orn] = ["L", "L", "L", "R"]
        sides[joa] = ["L", "R", "R"]
        sides[job] = ["L", "R"]
        sides = sides.astype(str)
        body = bw.FlyBody("A", fb, eye, groups, motor, sides=sides)
        img = np.full((bw.FRAME_H, bw.FRAME_W), bw.GROUND_GREY, dtype=np.float32)
        d = body.drive(img, 150.0, 60.0)
        per = np.zeros(fb.n)
        for k, v in d.items():
            per[list(k)] = np.asarray(v, dtype=float)
        for idx, hz in ((orn, 150.0), (joa, 60.0), (job, 60.0)):
            left, right = idx[sides[idx] == "L"], idx[sides[idx] == "R"]
            self.assertAlmostEqual(per[left].sum(), per[right].sum(), places=4)   # same total per antenna
            self.assertAlmostEqual(per[idx].mean(), hz, places=4)                  # the group mean is the nominal rate
        np.testing.assert_allclose(per[orn], [100.0, 100.0, 100.0, 300.0])
        np.testing.assert_allclose(per[joa], [90.0, 45.0, 45.0])
        np.testing.assert_allclose(per[job], [60.0, 60.0])
        s = body.describe()["side_scales"]
        self.assertEqual(s["ORN_DA1"], {"equalised": True, "n": 4, "n_L": 3, "n_R": 1, "n_other": 0,
                                        "scale_L": 2.0 / 3.0, "scale_R": 2.0,
                                        "cell_hz_R_over_L_uniform": 1.0 / 3.0, "cell_hz_R_over_L_now": 1.0})
        self.assertAlmostEqual(s["JO_A"]["cell_hz_R_over_L_uniform"], 2.0)
        self.assertAlmostEqual(s["JO_A"]["cell_hz_R_over_L_now"], 1.0)
        json.dumps(s)
        r = body.step(img, 150.0, 60.0)
        self.assertEqual(r["in"]["smell_hz"], 150.0)                 # the nominal rate is what is recorded
        self.assertAlmostEqual(r["rates"]["ORN_DA1"], 150.0, places=4)   # the fake brain echoes the per-cell drive
        self.assertAlmostEqual(r["rates"]["JO_A"], 60.0, places=4)
        # a cell with no side gets the nominal rate; a side with no cells gets no scale error
        sides2 = sides.copy(); sides2[orn[3]] = ""
        body2 = bw.FlyBody("A", fb, eye, groups, motor, sides=sides2)
        per2 = np.asarray(body2.drive(img, 150.0, 60.0)[tuple(orn.tolist())], dtype=float)
        np.testing.assert_allclose(per2, [75.0, 75.0, 75.0, 150.0])
        self.assertEqual(body2.describe()["side_scales"]["ORN_DA1"]["n_other"], 1)
        self.assertEqual(body2.describe()["side_scales"]["ORN_DA1"]["scale_R"], 0.0)

    def test_the_real_side_split_would_be_lopsided_without_equalisation(self):
        # the numbers the dictionary JSON reports for the real antennal groups
        for n_l, n_r, uniform in ((51, 105, 105 / 51), (18, 32, 32 / 18), (60, 29, 29 / 60)):
            sides = np.array(["L"] * n_l + ["R"] * n_r + [""] * 3, dtype=str)
            scale, s = bw.per_side_scales(np.arange(n_l + n_r + 3), sides)
            self.assertAlmostEqual(s["cell_hz_R_over_L_uniform"], uniform)
            self.assertAlmostEqual(s["cell_hz_R_over_L_now"], 1.0)
            self.assertAlmostEqual(scale[:n_l].sum(), scale[n_l:n_l + n_r].sum())
            self.assertTrue((scale[-3:] == 1.0).all())

    def test_two_bodies_on_one_brain_keep_separate_states_and_seeds(self):
        fb, eye, groups, motor = make_parts()
        a = bw.FlyBody("A", fb, eye, groups, motor, seed=101)
        b = bw.FlyBody("B", fb, eye, groups, motor, seed=202)
        img = np.full((bw.FRAME_H, bw.FRAME_W), bw.GROUND_GREY, dtype=np.float32)
        ra1 = a.step(img, 10.0, 0.0)
        rb1 = b.step(img, 10.0, 0.0)
        ra2 = a.step(img, 10.0, 0.0)
        self.assertEqual((a.windows, b.windows), (2, 1))
        self.assertEqual((ra1["state_carried"], rb1["state_carried"], ra2["state_carried"]),
                         (False, False, True))
        self.assertEqual(int(a.state["v"][0]), 2)
        self.assertEqual(int(b.state["v"][0]), 1)
        self.assertEqual(int(a.state["v"][1]), 101)                  # each chain started on its own seed
        self.assertEqual(int(b.state["v"][1]), 202)
        self.assertEqual([c["seed"] for c in fb.calls], [101, 202, 101])
        self.assertIsNone(fb.calls[0]["state"])
        self.assertIsNone(fb.calls[1]["state"])
        self.assertEqual(int(fb.calls[2]["state"]["v"][0]), 1)      # A's own, not B's
        self.assertEqual(fb.calls[2]["steps"], bw.SIM_STEPS)
        a.reset()
        self.assertFalse(a.state_carried)
        self.assertEqual(a.windows, 0)

    def test_rates_counts_and_inputs_for_every_group(self):
        fb, body = make_body()
        img = np.full((bw.FRAME_H, bw.FRAME_W), bw.GROUND_GREY, dtype=np.float32)
        r = body.step(img, 150.0, 60.0)
        for k in list(bd.present_groups(fb)) + list(MOTOR_NAMES):
            self.assertIn(k, r["rates"], k)
            self.assertIn(k, r["counts"], k)
        self.assertEqual(r["rates"]["ORN_DA1"], 150.0)
        self.assertEqual(r["rates"]["JO_A"], 60.0)
        self.assertEqual(r["rates"]["JO_B"], 60.0)
        self.assertEqual(r["out"], {"ORN_DA1": 150.0, "JO_A": 60.0, "JO_B": 60.0})
        self.assertEqual(r["in"]["smell_hz"], 150.0)
        self.assertEqual(r["in"]["sound_hz"], 60.0)
        self.assertAlmostEqual(r["in"]["eye_on_hz"], bw.GROUND_GREY * 180.0, places=4)
        self.assertAlmostEqual(r["in"]["eye_off_hz"], (1 - bw.GROUND_GREY) * 108.0, places=4)
        # counts: 4 ORN_DA1 cells at 150 Hz over 12 ms = 7.2 spikes -> 7
        self.assertEqual(r["counts"]["ORN_DA1"], int(np.rint(150.0 * 4 * 0.012)))
        self.assertEqual(r["rates"]["pC1"], 0.0)                     # nothing drove it
        self.assertEqual(r["counts"]["pC1"], 0)
        self.assertEqual(r["song_hz"], 0.0)
        self.assertEqual(r["song_group"], "song_pulse_mn")
        self.assertEqual(r["song_cells"], 3)                          # ps1 x 2 + i1
        self.assertEqual(r["fly"], "A")
        json.dumps(r)

    def test_song_is_the_summed_rate_of_the_pulse_group(self):
        fb0 = FakeBrain()
        ps1, i1, hg1 = fb0.where(type_re="^ps1 MN$"), fb0.where(type_re="^i1 MN$"), fb0.where(type_re="^hg1 MN$")
        spont = {int(ps1[0]): 100.0, int(ps1[1]): 40.0, int(i1[0]): 50.0, int(hg1[0]): 999.0}
        fb, body = make_body(spont=spont)
        img = np.full((bw.FRAME_H, bw.FRAME_W), bw.GROUND_GREY, dtype=np.float32)
        r = body.step(img, 0.0, 0.0)
        self.assertAlmostEqual(r["song_hz"], 190.0)                   # hg1 (sine) is not counted
        self.assertAlmostEqual(r["rates"]["song_pulse_mn"], 190.0 / 3)
        self.assertAlmostEqual(r["rates"]["song_sine_hg1"], 999.0)
        self.assertAlmostEqual(r["rates"]["wing_mn_all"], (190.0 + 999.0) / 5)

    def test_motor_conversions_are_plume_flys(self):
        fb0 = FakeBrain()
        a02, a01, mdn, p09 = (fb0.where(type_re=rf"^{t}$") for t in ("DNa02", "DNa01", "MDN", "DNp09"))
        spont = {int(a02[0]): 100.0, int(a02[1]): 300.0, int(a01[0]): 200.0, int(a01[1]): 100.0,
                 int(mdn[0]): 50.0, int(p09[1]): 90.0}
        fb, body = make_body(spont=spont)
        img = np.full((bw.FRAME_H, bw.FRAME_W), bw.GROUND_GREY, dtype=np.float32)
        r = body.step(img, 0.0, 0.0)
        motor = {"steer_L": 100.0, "steer_R": 300.0, "fwd_L": 200.0, "fwd_R": 100.0, "back": 25.0, "stop": 45.0}
        self.assertEqual(r["motor"], motor)
        turn, speed, parts = PlumeFly.motor_from_rates(motor)
        self.assertEqual((r["turn"], r["speed"]), (turn, speed))
        self.assertAlmostEqual(r["turn"], 200.0 / 450.0)
        self.assertEqual({k: r[k] for k in parts}, parts)
        self.assertEqual(r["rates"]["DNa02"], 200.0)                  # the dictionary's unsplit group

    def test_eye_overlapping_smell_or_sound_is_refused(self):
        fb, eye, groups, motor = make_parts()
        eye.on_idx = np.concatenate([eye.on_idx, fb.where(type_re="^ORN_DA1$")[:1]])
        with self.assertRaises(ValueError):
            bw.FlyBody("A", fb, eye, groups, motor)

    def test_song_group_falls_back_to_all_wing_motor_neurons_or_refuses(self):
        fb, eye, groups, motor = make_parts()
        g = {k: v for k, v in groups.items() if k != "song_pulse_mn"}
        body = bw.FlyBody("A", fb, eye, g, motor)
        self.assertEqual(body.song_key, "wing_mn_all")
        self.assertEqual(body.describe()["song_cells"], 5)
        g = {k: v for k, v in groups.items() if k not in ("song_pulse_mn", "wing_mn_all")}
        with self.assertRaises(KeyError):
            bw.FlyBody("A", fb, eye, g, motor)
        with self.assertRaises(KeyError):
            bw.FlyBody("A", fb, eye, {k: v for k, v in groups.items() if k != "ORN_DA1"}, motor)
        with self.assertRaises(KeyError):
            bw.FlyBody("A", fb, eye, {k: v for k, v in groups.items() if k not in ("JO_A", "JO_B")}, motor)

    def test_motor_groups_match_the_roamers_selection(self):
        fb = FakeBrain()
        m = bw.motor_groups(fb, fake_sides(fb))
        a02, a01 = fb.where(type_re="^DNa02$"), fb.where(type_re="^DNa01$")
        self.assertEqual(list(m["steer_L"]), [a02[0]])
        self.assertEqual(list(m["steer_R"]), [a02[1]])
        self.assertEqual(list(m["fwd_L"]), [a01[0]])
        self.assertEqual(list(m["fwd_R"]), [a01[1]])
        self.assertEqual(list(m["back"]), list(fb.where(type_re="^MDN$")))
        self.assertEqual(list(m["stop"]), list(fb.where(type_re="^DNp09$")))
        self.assertEqual(set(m), set(MOTOR_NAMES))


def strip_timing(rec):
    rec = copy.deepcopy(rec)
    rec.pop("step_s", None)
    for n in ("A", "B"):
        rec[n].pop("brain_s", None)
    return rec


class WholeRoom(unittest.TestCase):
    def test_a_room_step_is_a_pure_function_of_the_seed(self):
        logs = []
        for _ in range(2):
            fb, eye, groups, motor = make_parts(spont={26: 120.0, 27: 60.0})   # DNa02 L/R: a steady turn
            room = bw.Room(fb, eye, groups, motor, seed=9)
            logs.append([strip_timing(room.step()) for _ in range(6)])
        self.assertEqual(json.dumps(logs[0]), json.dumps(logs[1]))
        self.assertEqual([r["t"] for r in logs[0]], list(range(6)))
        self.assertNotEqual(json.dumps(logs[0][0]["after"]["A"]), json.dumps(logs[0][5]["after"]["A"]))
        fb, eye, groups, motor = make_parts(spont={26: 120.0, 27: 60.0})
        other = [strip_timing(bw.Room(fb, eye, groups, motor, seed=10).step()) for _ in range(2)]
        self.assertNotEqual(json.dumps(other[0]["before"]), json.dumps(logs[0][0]["before"]))

    def test_sound_is_the_others_previous_song_and_smell_is_the_distance(self):
        fb0 = FakeBrain()
        ps1 = fb0.where(type_re="^ps1 MN$")
        fb, eye, groups, motor = make_parts(spont={int(ps1[0]): 120.0})   # both flies sing 120 Hz summed
        room = bw.Room(fb, eye, groups, motor, seed=2)
        d0 = room.arena.distance()
        r0 = room.step()
        self.assertEqual(r0["sound_hz"], {"A": 0.0, "B": 0.0})           # nothing sung yet
        self.assertAlmostEqual(r0["smell_hz"], room.channels.smell_hz(d0))
        self.assertAlmostEqual(r0["A"]["song_hz"], 120.0)
        d1 = room.arena.distance()
        r1 = room.step()
        self.assertAlmostEqual(r1["sound_hz"]["A"], room.channels.sound_hz(120.0, d1))
        self.assertAlmostEqual(r1["sound_hz"]["B"], room.channels.sound_hz(120.0, d1))
        self.assertAlmostEqual(r1["A"]["in"]["sound_hz"], r1["sound_hz"]["A"])
        self.assertEqual(r1["seen"]["A"]["distance_mm"], max(d1, bw.MIN_DISTANCE_MM))
        self.assertEqual(room.bodies["A"].seed, 5)
        self.assertEqual(room.bodies["B"].seed, 6)
        self.assertEqual(r1["A"]["window"], 2)
        # the room's song reference follows the song group it actually has: 3 fake cells
        self.assertAlmostEqual(room.song_full_hz, 3 * 1000.0 / 2.2)
        self.assertAlmostEqual(room.channels.song_full, room.song_full_hz)
        self.assertAlmostEqual(r1["sound_hz"]["A"], 200.0 * 120.0 / room.song_full_hz * (1 - d1 / 20.0))
        self.assertAlmostEqual(room.describe()["song_full_hz"], room.song_full_hz)


class WorldAdapter(unittest.TestCase):
    """
    backrooms.Loop's declared record shape, from a World wrapping a fake room:
    no brain is loaded. What is held to: the flat per-fly record carries the
    arena's numbers after the step in the loop's units (degrees, mm/s, deg/s),
    rates and counts for exactly the present dictionary groups, the three
    drive keys, the six motor rates and the song; and the whole record is
    plain JSON.
    """

    FLY_KEYS = ("x", "y", "heading", "speed", "turn", "distance", "bearing",
                "rates", "counts", "drive", "motor", "song")

    def make(self, seed=4, spont=None):
        fb, eye, groups, motor = make_parts(spont)
        return bw.World(seed=seed, room=bw.Room(fb, eye, groups, motor, seed=seed))

    def test_record_is_the_loops_shape_with_the_arenas_numbers(self):
        w = self.make(spont={26: 120.0, 27: 60.0, 28: 200.0, 29: 200.0})   # DNa02 L/R and DNa01: turn and walk
        self.assertEqual(sorted(w.groups), sorted(bd.present_groups(w.fb)))
        for k, v in w.groups.items():
            np.testing.assert_array_equal(v, bd.present_groups(w.fb)[k], err_msg=k)
        for expect in (1, 2, 3):
            r = w.step()
            self.assertEqual(sorted(r), sorted(["step", "t", "flies", "smell_hz", "sound_hz", "seen", "step_s"]))
            self.assertEqual(r["step"], expect)
            self.assertAlmostEqual(r["t"], expect * bw.WORLD_DT_S)
            self.assertEqual(sorted(r["flies"]), ["A", "B"])
            after = w.room.arena.geometry()          # the arena as left by the step
            for n, o in (("A", "B"), ("B", "A")):
                f = r["flies"][n]
                for k in self.FLY_KEYS:
                    self.assertIn(k, f, k)
                self.assertEqual((f["x"], f["y"]), (after[n]["x"], after[n]["y"]))
                self.assertEqual(f["heading"], after[n]["heading_deg"])
                self.assertEqual(f["distance"], after["distance_mm"])
                self.assertEqual(f["bearing"], after["bearing_deg"][n])
                self.assertAlmostEqual(f["speed"], after["moved_mm"][n] / bw.WORLD_DT_S)
                self.assertAlmostEqual(f["turn"], after["turned_deg"][n] / bw.WORLD_DT_S)
                # the same geometry recomputed from the record's own x, y, heading
                me = bw.Fly(n, f["x"], f["y"], math.radians(f["heading"]))
                ot = bw.Fly(o, r["flies"][o]["x"], r["flies"][o]["y"], 0.0)
                self.assertAlmostEqual(bw.distance_mm(me, ot), f["distance"])
                self.assertAlmostEqual(bw.bearing_deg(me, ot), f["bearing"], places=9)
                self.assertEqual(sorted(f["rates"]), sorted(w.groups))
                self.assertEqual(sorted(f["counts"]), sorted(w.groups))
                self.assertNotIn("steer_L", f["rates"])
                self.assertEqual(sorted(f["drive"]), ["JO_A", "JO_B", "ORN_DA1"])
                self.assertEqual(sorted(f["motor"]), sorted(MOTOR_NAMES))
                self.assertEqual(f["drive"]["JO_A"], f["drive"]["JO_B"])
                self.assertAlmostEqual(f["drive"]["ORN_DA1"], r["smell_hz"])
                self.assertAlmostEqual(f["drive"]["JO_A"], r["sound_hz"][n])
                self.assertEqual(f["window"], expect)
                self.assertEqual(f["state_carried"], expect > 1)
            # the steady DNa02 / DNa01 drive walks and turns both flies
            self.assertNotEqual(r["flies"]["A"]["speed"], 0.0)
            self.assertNotEqual(r["flies"]["A"]["turn"], 0.0)
            # the record is plain JSON before the loop touches it
            json.dumps(r)

    def test_meta_is_the_rooms_describe_plus_the_conventions(self):
        w = self.make()
        m = w.meta()
        self.assertEqual(m["world_class"], "backrooms_world.World")
        self.assertEqual(m["brain_class"], bw.BRAIN_CLASS)
        self.assertIn("arena", m)
        self.assertIn("bodies", m)
        self.assertIn("record", m)
        json.dumps(m)

    def test_a_world_step_is_deterministic_in_the_seed(self):
        recs = []
        for _ in range(2):
            w = self.make(seed=7, spont={26: 120.0, 27: 60.0})
            recs.append([w.step() for _ in range(4)])
        for a, b in zip(*recs):
            for r in (a, b):
                r.pop("step_s")
                for n in ("A", "B"):
                    r["flies"][n].pop("brain_s")
            self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))


# =============================================================================

@unittest.skipUnless(os.environ.get("BACKROOMS_REAL_BRAIN") == "1",
                     "set BACKROOMS_REAL_BRAIN=1 to load the connectome")
class RealBrain(unittest.TestCase):
    """
    One brain, loaded once after the RAM check: the eye, the dictionary
    groups and the roamer's motor split on the real type strings, then two
    flies for ten world steps with per-fly motor, ORN_DA1 and JO rates and
    the step time printed and written to build/backrooms_world_check.json.
    """

    OUT = ROOT / "build" / "backrooms_world_check.json"

    def test_two_flies_ten_steps(self):
        t0 = time.time()
        fb = bw.load_brain()
        gains = bw.room_gains(fb)
        room = bw.build_room(fb, gains, seed=1)
        print(f"load + setup: {time.time() - t0:.1f} s")
        desc = room.describe()
        body = room.bodies["A"]
        print(f"groups recorded: {len(body.groups)}; recorded cells: {desc['bodies']['A']['recorded_cells']}; "
              f"smell cells {desc['bodies']['A']['smell_cells']}; sound cells {desc['bodies']['A']['sound_cells']}; "
              f"song {desc['bodies']['A']['song_group']} x {desc['bodies']['A']['song_cells']}")
        self.assertEqual(body.smell_idx.size, 204)
        self.assertEqual(body.sound_idx.size, 139)
        self.assertEqual(desc["bodies"]["A"]["song_cells"], 8)
        self.assertAlmostEqual(room.song_full_hz, 8 * 1000.0 / fb.p.refractory)
        print(f"song reference: {room.song_full_hz:.1f} Hz summed (8 cells x 1000 / {fb.p.refractory} ms)")
        self.assertEqual({k: v.size for k, v in body.motor.items()},
                         {"steer_L": 1, "steer_R": 1, "fwd_L": 1, "fwd_R": 1, "back": 4, "stop": 2})
        # the same cells the roamer reads (pumpui.FlyPilot; its annotations path is relative)
        import pumpui
        pilot = pumpui.FlyPilot(fb, eye=body.eye, sim_steps=bw.SIM_STEPS)
        for k in MOTOR_NAMES:
            np.testing.assert_array_equal(body.motor[k], pilot.motor[k], err_msg=k)
        # the eye and the two other channels touch disjoint cells
        eye_idx = np.concatenate([body.eye.on_idx, body.eye.off_idx])
        self.assertEqual(np.intersect1d(eye_idx, np.concatenate([body.smell_idx, body.sound_idx])).size, 0)

        steps = []
        for k in range(10):
            r = room.step()
            steps.append(r)
            for n in ("A", "B"):
                x = r[n]
                m = x["motor"]
                print(f"t={r['t']:2d} {n}: {x['brain_s']:.2f} s  carried={x['state_carried']}  "
                      f"turn={x['turn']:+.3f} speed={x['speed']:+.3f}  "
                      f"steer L/R {m['steer_L']:.0f}/{m['steer_R']:.0f} fwd L/R {m['fwd_L']:.0f}/{m['fwd_R']:.0f} "
                      f"back {m['back']:.0f} stop {m['stop']:.0f}  "
                      f"ORN_DA1 in {x['in']['smell_hz']:.1f} out {x['out']['ORN_DA1']:.1f}  "
                      f"JO in {x['in']['sound_hz']:.1f} out A {x['out']['JO_A']:.1f} B {x['out']['JO_B']:.1f}  "
                      f"song {x['song_hz']:.0f}  fired {x['fired']}")
            print(f"      d={r['after']['distance_mm']:.2f} mm  bearing A {r['after']['bearing_deg']['A']:+.0f} "
                  f"B {r['after']['bearing_deg']['B']:+.0f}  step {r['step_s']:.2f} s")
        times = [r["step_s"] for r in steps]
        brain = [r[n]["brain_s"] for r in steps for n in ("A", "B")]
        print(f"world step: mean {np.mean(times):.2f} s (first {times[0]:.2f}, rest {np.mean(times[1:]):.2f}); "
              f"brain window: mean {np.mean(brain):.2f} s; "
              f"{np.mean(times[1:]) / bw.WORLD_DT_S:.0f}x slower than life")
        self.assertEqual([r["A"]["state_carried"] for r in steps], [False] + [True] * 9)
        self.assertEqual([r["B"]["state_carried"] for r in steps], [False] + [True] * 9)
        for r in steps:
            for n in ("A", "B"):
                for k in ("turn", "speed", "song_hz"):
                    self.assertTrue(np.isfinite(r[n][k]), k)
                self.assertGreaterEqual(r[n]["out"]["ORN_DA1"], 0.0)
        self.assertLess(np.mean(times[1:]), 6.0)
        out = {"what": "one real-brain check of backrooms_world: two flies, ten world steps",
               "describe": desc,
               "step_s": times, "brain_s": brain,
               "steps": [strip_timing(r) for r in steps]}
        self.OUT.write_text(json.dumps(out, indent=1, sort_keys=True, default=float) + "\n", encoding="utf-8")
        print(f"wrote {self.OUT}")


if __name__ == "__main__":
    unittest.main()
