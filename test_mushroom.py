import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import scipy.sparse as sp

import mushroom
from mushroom import MushroomBody, sides_sha


class FakeBrain:
    """0,1 Kenyon cells; 2 MBON01 (PAM); 3 MBON11 (PPL1); 4 MBON99 (no table entry); 5 PAM01."""

    def __init__(self):
        self.types = np.array(["KCab", "KCg", "MBON01", "MBON11", "MBON99", "PAM01"])
        self.n = len(self.types)
        pre = [0, 0, 0, 1, 1, 2]
        post = [2, 3, 4, 2, 3, 5]          # the last edge is MBON01 -> PAM01, an output
        W = sp.csc_matrix((np.ones(len(pre), np.float32), (post, pre)), shape=(self.n, self.n))
        W.sort_indices()
        self.W = W
        self.indptr, self.indices = W.indptr, W.indices
        self.wdata = W.data.astype(np.float32)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.sides = self.write_sides({"MBON01": "PAM", "MBON11": "PPL1"})
        self.store = self.tmp / "state" / "mb_gains.v2.npz"
        self.now = [1_000_000.0]

    def write_sides(self, table, name="mb_sides.json"):
        p = self.tmp / name
        p.write_text(json.dumps({"types": {t: {"side": s} for t, s in table.items()}}), encoding="utf-8")
        return p

    def mb(self, **kw):
        kw.setdefault("sides", self.sides)
        kw.setdefault("store", self.store)
        kw.setdefault("clock", lambda: self.now[0])
        kw.setdefault("calibration", "pn05_apl10_kc03")
        kw.setdefault("half_life_h", 1.0)
        return MushroomBody(FakeBrain(), **kw)

    def gain_of(self, mb, pre, post):
        i = np.flatnonzero((mb.pre == pre) & (mb.post == post))
        self.assertEqual(len(i), 1)
        return float(mb.gain[i[0]])


class Sides(Base):
    def test_sides_come_from_the_table(self):
        mb = self.mb()
        self.assertEqual(mb.reward_side.tolist(), [2])
        self.assertEqual(mb.punish_side.tolist(), [3])
        self.assertEqual(mb.unassigned, ["MBON99"])

    def test_output_edges_do_not_decide_a_side(self):
        mb = self.mb(sides=self.write_sides({"MBON01": "PPL1", "MBON11": "PAM"}, "flipped.json"))
        self.assertEqual(mb.reward_side.tolist(), [3])
        self.assertEqual(mb.punish_side.tolist(), [2])

    def test_sha_ignores_order(self):
        self.assertEqual(sides_sha({"a": "PAM", "b": "PPL1"}), sides_sha({"b": "PPL1", "a": "PAM"}))

    def test_a_missing_side_table_says_where_it_comes_from(self):
        # build/ is gitignored, so an image can be built without this file; the
        # error has to name the thing that writes it rather than a bare path
        with self.assertRaises(FileNotFoundError) as caught:
            self.mb(sides=self.tmp / "not-here.json")
        self.assertIn("mb_sides.py", str(caught.exception))


class Dopamine(Base):
    def test_reward_depresses_only_eligible_synapses_onto_pam_side(self):
        mb = self.mb()
        mb.observe(np.array([0]))
        self.assertEqual(mb.dopamine(+1, 1.0), 1)
        self.assertLess(self.gain_of(mb, 0, 2), 1.0)
        for pre, post in ((0, 3), (0, 4), (1, 2), (1, 3)):
            self.assertEqual(self.gain_of(mb, pre, post), 1.0)

    def test_punishment_depresses_only_eligible_synapses_onto_ppl1_side(self):
        mb = self.mb()
        mb.observe(np.array([0]))
        self.assertEqual(mb.dopamine(-1, 1.0), 1)
        self.assertLess(self.gain_of(mb, 0, 3), 1.0)
        for pre, post in ((0, 2), (0, 4), (1, 2), (1, 3)):
            self.assertEqual(self.gain_of(mb, pre, post), 1.0)

    def test_stale_trace_is_not_eligible(self):
        mb = self.mb()
        mb.observe(np.array([0]))
        for _ in range(6):                  # 0.55 ** 6 = 0.028, under the 0.05 cutoff
            mb.observe(np.array([], dtype=np.int64))
        self.assertEqual(mb.dopamine(+1, 1.0), 0)

    def test_zero_amount_is_no_event(self):
        mb = self.mb()
        mb.observe(np.array([0]))
        self.assertEqual(mb.dopamine(+1, 0.0), 0)
        self.assertEqual(mb.events, {"reward": 0, "punish": 0})

    def test_amount_scales_the_step(self):
        a, b = self.mb(), self.mb()
        for mb, amount in ((a, 1.0), (b, 0.25)):
            mb.observe(np.array([0]))
            mb.dopamine(+1, amount)
        self.assertAlmostEqual(1 - self.gain_of(a, 0, 2), 4 * (1 - self.gain_of(b, 0, 2)), places=5)

    def test_floor(self):
        mb = self.mb()
        for _ in range(500):
            mb.observe(np.array([0]))
            mb.dopamine(+1, 1.0)
        self.assertAlmostEqual(self.gain_of(mb, 0, 2), mb.floor, places=6)


class Forgetting(Base):
    def test_half_life_is_wall_clock(self):
        mb = self.mb()
        mb.gain[:] = 0.5
        self.now[0] += 3600.0
        mb.forget()
        np.testing.assert_allclose(mb.gain, 0.75, rtol=1e-5)

    def test_many_small_steps_equal_one_big_one(self):
        a, b = self.mb(), self.mb()
        a.gain[:] = 0.4
        b.gain[:] = 0.4
        for _ in range(100):
            self.now[0] += 36.0
            a.forget()
        b.forget()
        np.testing.assert_allclose(a.gain, b.gain, rtol=1e-4)

    def test_no_time_no_change(self):
        mb = self.mb()
        mb.gain[:] = 0.5
        mb.forget()
        np.testing.assert_array_equal(mb.gain, 0.5)


class Applying(Base):
    """What the simulation reads has to be what the mushroom body says it is."""

    def test_forgetting_writes_the_decayed_gains_into_the_simulation(self):
        mb = self.mb()
        mb.observe(np.array([0]))
        mb.dopamine(+1, 1.0)
        mb.apply()
        lesson = mb.fb.wdata[mb.pos].copy()
        self.now[0] += 3600.0
        mb.forget()
        # without this the fly walked on the weights it booted with for as long
        # as no trade settled, while stats() reported the decayed ones
        self.assertFalse(np.array_equal(mb.fb.wdata[mb.pos], lesson))
        np.testing.assert_allclose(mb.fb.wdata[mb.pos], mb.base * mb.gain, rtol=1e-6)

    def test_no_time_passing_writes_nothing(self):
        mb = self.mb()
        mb.fb.wdata[mb.pos] = 0.0            # a sentinel nothing should overwrite
        mb.forget()
        np.testing.assert_array_equal(mb.fb.wdata[mb.pos], 0.0)


class Trace(Base):
    def test_wiping_the_trace_makes_everything_ineligible(self):
        mb = self.mb()
        mb.observe(np.array([0]))
        mb.forget_trace()
        self.assertEqual(mb.dopamine(+1, 1.0), 0)
        self.assertTrue((mb.gain == 1.0).all())

    def test_what_fires_after_the_wipe_is_eligible_again(self):
        mb = self.mb()
        mb.observe(np.array([0]))            # the old page
        mb.forget_trace()
        mb.observe(np.array([1]))            # what the lesson is about
        self.assertEqual(mb.dopamine(+1, 1.0), 1)
        self.assertEqual(self.gain_of(mb, 0, 2), 1.0)
        self.assertLess(self.gain_of(mb, 1, 2), 1.0)


class Persistence(Base):
    def trained(self):
        mb = self.mb()
        mb.observe(np.array([0]))
        mb.dopamine(+1, 1.0)
        self.assertTrue(mb.save())
        return mb

    def test_round_trip(self):
        a = self.trained()
        b = self.mb()
        self.assertTrue(b.loaded)
        np.testing.assert_array_equal(a.gain, b.gain)
        self.assertEqual(b.events["reward"], 1)

    def test_rejects_other_sides(self):
        self.trained()
        b = self.mb(sides=self.write_sides({"MBON01": "PPL1", "MBON11": "PAM"}, "flipped.json"))
        self.assertFalse(b.loaded)
        self.assertTrue((b.gain == 1.0).all())

    def test_rejects_other_calibration(self):
        self.trained()
        self.assertFalse(self.mb(calibration="stock").loaded)

    def test_old_unversioned_store_is_never_read(self):
        self.trained()
        self.store.rename(self.store.with_name("mb_gains.npz"))
        self.assertFalse(self.mb().loaded)

    def test_time_away_is_forgotten_on_load(self):
        a = self.trained()
        g = self.gain_of(a, 0, 2)
        self.now[0] += 3600.0
        b = self.mb()
        self.assertTrue(b.loaded)
        self.assertAlmostEqual(self.gain_of(b, 0, 2), 1.0 - (1.0 - g) * 0.5, places=5)


class WhereTheGainsLive(unittest.TestCase):
    """
    FLY_STATE_DIR decides where everything the fly has learned is kept, and it
    has to be the same directory the ledger, the room and the looks resolve to.
    This module used to read the process environment only, so a run started by
    hand with FLY_STATE_DIR in .env split the two apart silently.
    """

    def test_a_value_only_in_the_env_file_is_found(self):
        stub = types.ModuleType("launch")
        stub.load_env = lambda: {"FLY_STATE_DIR": "C:/only-in-the-file"}
        with mock.patch.dict(sys.modules, {"launch": stub}):
            self.assertEqual(mushroom.setting("FLY_STATE_DIR"), "C:/only-in-the-file")

    def test_without_launch_the_environment_still_answers(self):
        with mock.patch.dict(sys.modules, {"launch": None}), \
             mock.patch.dict(os.environ, {"FLY_STATE_DIR": "C:/from-the-environment"}):
            self.assertEqual(mushroom.setting("FLY_STATE_DIR"), "C:/from-the-environment")

    def test_an_unset_name_is_the_default(self):
        stub = types.ModuleType("launch")
        stub.load_env = lambda: {}
        with mock.patch.dict(sys.modules, {"launch": stub}):
            self.assertEqual(mushroom.setting("FLY_NOT_SET", "fallback"), "fallback")

    def test_the_store_is_named_after_it(self):
        self.assertEqual(mushroom.STORE.name, "mb_gains.v2.npz")


if __name__ == "__main__":
    unittest.main()
