"""Paired controls and report contracts without graph archives."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import courtship_experiment as ce
import backrooms_world as bw
from test_backrooms_world import make_parts


class FakeFly:
    def __init__(self, name, seed, ignores_sound=False):
        self.name, self.seed, self.window = name, seed, 0
        self.ignores_sound = ignores_sound

    def step(self, frame, smell, sound):
        self.window += 1
        answer = 0 if self.ignores_sound else sound * (self.window % 3 != 0)
        return dict(turn=0.05 * np.sin(self.window + self.seed), speed=0.02 + self.seed * 0.01,
            song_hz=float((self.window * 37 + self.seed * 13) % 101) if self.name == "A" else 0.,
            her_answer=dict(vpodn_hz=answer, pc1_hz=answer/2), **{"in": {"sound_hz": sound}})


class FakeRoom(ce.ExperimentRoom):
    def __init__(self, seed, condition, song_sound=None, brains=None, ignores_sound=False):
        self.arena = bw.Arena(seed=seed)
        self.channels = bw.Channels()
        self.song = dict(A=0., B=0.)
        self.bodies = {n: FakeFly(n, seed, ignores_sound) for n in ("A", "B")}
        self.configure(seed, condition, song_sound)


def trials(seed=0, steps=12, ignores_sound=False):
    result = {}
    sound = None
    for c in ce.CONDITIONS:
        result[c] = ce.run_trial(FakeRoom(seed, c, sound, ignores_sound=ignores_sound), steps, seed, c)
        if c == "song":
            sound = result[c][1]["her_sound_hz"]
    return result


class Paired(unittest.TestCase):
    def test_seed_start_and_multiset(self):
        r = trials(3)
        self.assertEqual(set(r), {"song", "silence", "shuffled", "dark"})
        for condition in r:
            self.assertEqual(r[condition][0]["seed"], 3)
            self.assertEqual(r[condition][0]["start"], r["song"][0]["start"])
        self.assertEqual(r["song"][0]["start"], r["silence"][0]["start"])
        self.assertEqual(r["song"][0]["start"], r["shuffled"][0]["start"])
        np.testing.assert_array_equal(np.sort(r["song"][1]["her_sound_hz"]), np.sort(r["shuffled"][1]["her_sound_hz"]))
        self.assertFalse(np.array_equal(r["song"][1]["her_sound_hz"], r["shuffled"][1]["her_sound_hz"]))
        self.assertNotEqual(r["song"][0]["start"], trials(4)["song"][0]["start"])

    def test_silence_and_male_path(self):
        r = trials(2, ignores_sound=True)
        self.assertTrue(np.all(r["silence"][1]["her_sound_hz"] == 0))
        self.assertGreater(r["song"][1]["her_sound_hz"].sum(), 0)
        for key in ("A_x", "A_y", "A_heading_rad", "song_hz", "his_sound_hz"):
            np.testing.assert_array_equal(r["song"][1][key], r["silence"][1][key])

    def test_hook_changes_actual_body_input(self):
        fb, eye, groups, motor = make_parts()
        room = ce.ExperimentRoom(fb, eye, groups, motor)
        room.configure(0, "silence")
        room.song["A"] = 1000
        room.song["B"] = 500
        expected = room.channels.sound_hz(500, room.arena.distance())
        r = room.step()
        self.assertEqual(r["B"]["in"]["sound_hz"], 0)
        self.assertEqual(r["A"]["in"]["sound_hz"], expected)

    def test_dark_eye_and_sound(self):
        from test_courtship import female_fake
        male, _, _, _ = make_parts()
        for eye_setting in ("blind", "luminance"):
            with patch.object(ce, "FEMALE_EYE", eye_setting):
                room = ce.build_room(2, "dark", brains=(male, female_fake()))
            body = room.bodies["B"]
            self.assertIsInstance(body.eye, ce.BlindEye)
            room.song["A"] = 1000
            r = room.step()
            self.assertEqual(r["B"]["in"]["sound_hz"], 0)
            drive = body.drive(np.ones((bw.FRAME_H, bw.FRAME_W)), 0, room.listener_sound(1000))
            self.assertEqual(set(drive), {tuple(body.sound_idx)})
            np.testing.assert_array_equal(drive[tuple(body.sound_idx)], 0)


class Verdicts(unittest.TestCase):
    def test_threshold(self):
        self.assertEqual(ce.verdict(.5, .2, 2), "supported")
        for mean, se in ((.4, .2), (0, 0), (-1, .1)):
            self.assertEqual(ce.verdict(mean, se, 2), "not supported")
        self.assertEqual(ce.verdict(1, None, 1), "undetermined")

    def test_paired_values_and_direction(self):
        rows = [r[0] for seed in (0, 1) for r in trials(seed).values()]
        s = ce.summarise(rows)
        self.assertEqual(s["predictions"]["P1"]["paired_differences"],
                         [rows[0]["vpodn_hz"]-rows[1]["vpodn_hz"], rows[4]["vpodn_hz"]-rows[5]["vpodn_hz"]])
        self.assertEqual(s["predictions"]["P4"]["paired_differences"][0], rows[1]["last_distance_mm"]-rows[0]["last_distance_mm"])


class Deterministic(unittest.TestCase):
    def test_repeat(self):
        a, b = trials(4), trials(4)
        self.assertEqual(json.dumps([v[0] for v in a.values()], sort_keys=True), json.dumps([v[0] for v in b.values()], sort_keys=True))
        for c in a:
            for k in a[c][1]:
                np.testing.assert_array_equal(a[c][1][k], b[c][1][k])


class RunTrial(unittest.TestCase):
    def test_outcome_boundary_and_constant_correlation(self):
        row, t = trials()["song"]
        t["vpodn_hz"][:] = 0
        t["vpodn_hz"][:2] = 1
        self.assertFalse(ce.outcome(0, "song", t, row["start"])["accept"])
        t["vpodn_hz"][2] = 1
        self.assertTrue(ce.outcome(0, "song", t, row["start"])["accept"])
        self.assertIsNone(ce.correlation([1, 1], [1, 2]))
        self.assertAlmostEqual(ce.correlation([1, 2, 3], [3, 2, 1]), -1)

    def test_missing_replay(self):
        with self.assertRaises(ValueError):
            FakeRoom(0, "shuffled")


class MainWithFakes(unittest.TestCase):
    def test_quick_outputs_and_reanalysis(self):
        with tempfile.TemporaryDirectory() as td:
            prefix = Path(td) / "run"
            self.assertEqual(ce.main(["--quick", "1", "--out", str(prefix)], room_factory=FakeRoom), 0)
            for p in ce.paths(prefix):
                self.assertTrue(p.is_file())
                self.assertGreater(p.stat().st_size, 0)
            jp = ce.paths(prefix)[0]
            data = json.loads(jp.read_text())
            self.assertEqual(len(data["outcomes"]), 8)
            self.assertEqual({(r["seed"], r["condition"]) for r in data["outcomes"]}, {(s, c) for s in (0, 1) for c in ce.CONDITIONS})
            self.assertEqual(data["steps"], 80)
            self.assertEqual(data["budget_ladder"]["selected"], 2)
            self.assertEqual(data["summary"]["P0"]["expected_active_windows"], 0)
            self.assertEqual(data["summary"]["P0"]["per_seed"], [dict(seed=s, condition=c, active_windows=0) for s in (0, 1) for c in ("silence", "dark")])
            report = ce.paths(prefix)[3].read_text()
            self.assertIn("## P0: baseline (descriptive, no verdict)", report)
            self.assertIn("FEMALE_EXC_SCALE = 1.0; FEMALE_EYE = blind", report)
            self.assertIn("Seed 1, dark: 0 active windows.", report)
            self.assertIn("<!-- interpretation: to be written after the run -->", report)
            self.assertIn("outcomes differ across seeds", report)
            self.assertEqual(ce.main(["--reanalyse", str(jp)]), 0)
            self.assertEqual(json.loads(jp.read_text())["outcomes"], data["outcomes"])


class Cli(unittest.TestCase):
    def test_published_file_identity(self):
        with tempfile.TemporaryDirectory() as td:
            published = Path(td) / "published"
            alias = Path(td) / "alias"
            ce.paths(published)[2].write_bytes(b"protected")
            os.link(ce.paths(published)[2], ce.paths(alias)[0])
            with patch.object(ce, "PUBLISHED", published):
                with self.assertRaises(ValueError):
                    ce.assert_not_published(alias)

    def test_budget_ladder(self):
        self.assertEqual(ce.budget_ladder(10, 100, 4000)["selected"], 10)
        self.assertEqual(ce.budget_ladder(10, 100, 3200)["selected"], 8)
        self.assertTrue(ce.budget_ladder(10, 100, 100)["budget_exceeded"])
        self.assertEqual(ce.budget_ladder(2, 100, 100)["selected"], 2)

    def test_published_refusal(self):
        self.assertEqual(ce.main(["--out", str(ce.PUBLISHED)], room_factory=FakeRoom), 4)

    def test_invalid_steps(self):
        with self.assertRaises(SystemExit):
            ce.main(["--steps", "0"], room_factory=FakeRoom)


class Outputs(unittest.TestCase):
    def test_spread_and_protocol(self):
        rows = [r[0] for seed in (0, 1) for r in trials(seed).values()]
        spread = ce.summarise(rows)["seed_spread"]
        self.assertEqual(spread["silence"]["accept"], 0)
        self.assertEqual(spread["song"]["accept"], 2)
        identical = [dict(r, approach=True, retreat=False) for r in rows]
        self.assertIn("every seed gives the same outcome", ce.summarise(identical)["seed_spread"]["song"]["sentence"])
        self.assertEqual(ce.PROTOCOLS["v2"]["default_seeds"], 10)
        self.assertEqual(ce.PROTOCOLS["v2"]["steps"], 400)
