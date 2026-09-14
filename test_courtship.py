"""Listener selectors, window rates, and the delayed room sound channel."""
import os
import time
import unittest

import numpy as np
import pytest

import backrooms_dictionary as bd
import backrooms_world as bw
from courtship import BlindEye, HerBody, female_groups, female_motor, load_female
from flysim import BUILD, FlyBrain
from test_backrooms_world import FakeBrain, FakeEye, fake_sides, make_parts


def female_fake(types=None):
    if types is None:
        types = (["L1", "L2", "JO-A", "JO-A", "JO-B", "JO-B"]
                 + [f"pC1{letter}" for letter in "abcde" for _ in range(2)]
                 + ["DNp37", "DNp37", "DNa02", "DNa02", "DNa01", "DNa01"]
                 + ["MDN", "DNp09", "MN9", "JO-A1", "pC1a_extra", "DNp370"])
    fb = FakeBrain(types)
    fb.soma_side = fake_sides(fb)
    return fb


def her(fb=None):
    fb = female_fake() if fb is None else fb
    return HerBody("B", fb, FakeEye(fb), female_groups(fb), female_motor(fb), seed=2)


@pytest.mark.parametrize("override, expected", [(None, .5), (1.0, 1.0), (.25, .25)])
def test_load_female_scale_once(tmp_path, monkeypatch, override, expected):
    from types import SimpleNamespace
    import courtship
    path = tmp_path / "female.npz"
    np.savez(path, exc_scale=.5, soma_side=np.array(["L", "R"]))
    def raw_brain(path, p):
        return SimpleNamespace(wdata=np.array([8., -6., 0., 4.], dtype=np.float32),
                               W=SimpleNamespace(data=None))
    monkeypatch.setattr(courtship, "FlyBrain", raw_brain)
    for _ in range(2):
        fb = load_female(path, exc_scale=override)
        assert fb.exc_scale == expected
        np.testing.assert_array_equal(fb.wdata, [8 * expected, -6, 0, 4 * expected])
        np.testing.assert_array_equal(fb.W.data, fb.wdata)


def test_blind_body_only_sound():
    fb = female_fake()
    eye = BlindEye(fb)
    assert eye.on_idx.dtype == eye.off_idx.dtype == np.dtype('int64')
    assert eye.on_idx.size == eye.off_idx.size == 0
    body = HerBody("B", fb, eye, female_groups(fb), female_motor(fb))
    frame = np.ones((bw.FRAME_H, bw.FRAME_W), dtype=np.float32)
    assert eye.look(frame, 0, 0) == {}
    drive = body.drive(frame, 999, 80)
    assert set(drive) == {tuple(body.sound_idx)}
    assert () not in drive
    np.testing.assert_array_equal(drive[tuple(body.sound_idx)], body.sound_scale * 80)
    result = body.step(frame, 999, 80)
    assert result['in'] == dict(smell_hz=0., sound_hz=80., eye_on_hz=0., eye_off_hz=0.)


def test_dark_is_blind_and_silent():
    import courtship_experiment as ce
    male, _, _, _ = make_parts()
    with unittest.mock.patch.object(ce, "FEMALE_EYE", "luminance"):
        room = ce.build_room(3, "dark", brains=(male, female_fake()))
    body = room.bodies['B']
    assert isinstance(body.eye, BlindEye)
    room.song['A'] = 1000.
    result = room.step()
    assert result['B']['in'] == dict(smell_hz=0., sound_hz=0., eye_on_hz=0., eye_off_hz=0.)


def test_groups_and_motor():
    fb = female_fake()
    groups = female_groups(fb)
    assert {k: len(v) for k, v in groups.items()} == {
        "JO_A": 2, "JO_B": 2, "pC1": 10, "vpoDN": 2}
    motor = female_motor(fb)
    assert set(motor) == set(bw.MOTOR_NAMES)
    for key, typ, side in (("steer_L", "DNa02", "L"), ("steer_R", "DNa02", "R"),
                           ("fwd_L", "DNa01", "L"), ("fwd_R", "DNa01", "R")):
        assert fb.types[motor[key]].tolist() == [typ]
        assert fb.soma_side[motor[key]].tolist() == [side]
    fb.types[fb.types == "MDN"] = "absent"
    assert female_motor(fb)["back"].size == 0


@pytest.mark.parametrize("missing", ["DNp37", "JO-A", "JO-B"])
def test_missing_required_group(missing):
    fb = female_fake()
    fb.types[fb.types == missing] = "absent"
    with pytest.raises(KeyError, match={"DNp37": "vpoDN", "JO-A": "JO_A", "JO-B": "JO_B"}[missing]):
        female_groups(fb)


def test_five_windows_mean_answer_and_ignored_smell():
    body = her()
    for i, hz in zip(body.groups["vpoDN"], [20.0, 60.0]):
        body.fb.spont[i] = hz
    for i, hz in zip(body.groups["pC1"], range(10)):
        body.fb.spont[i] = hz
    frame = np.zeros((bw.FRAME_H, bw.FRAME_W), dtype=np.float32)
    for window in range(1, 6):
        r = body.step(frame, 999.0, 80.0)
        assert r["her_answer"] == body.answer == {"vpodn_hz": 40.0, "pc1_hz": 4.5}
        assert all(np.isfinite(v) for v in body.answer.values())
        assert r["out"] == {"JO_A": 80.0, "JO_B": 80.0}
        assert r["song_hz"] == r["in"]["smell_hz"] == 0.0
        assert r["window"] == window
        assert r["state_carried"] == (window > 1)
        assert int(body.state["v"][0]) == window
    assert not hasattr(body, "song_key")
    assert len(body.fb.calls[-1]["keys"]) == 3
    body.reset()
    assert body.answer == {"vpodn_hz": 0.0, "pc1_hz": 0.0}


def test_room_twenty_steps_and_previous_song():
    fb, eye, groups, motor = make_parts()
    body = her()
    # CHOSEN: explicit motor/song rates exercise movement and delayed sound.
    for i in groups[bw.SONG_KEY]:
        fb.spont[i] = 100.0
    for brain, motors in ((fb, motor), (body.fb, body.motor)):
        for k in ("fwd_L", "fwd_R"):
            for i in motors[k]:
                brain.spont[i] = 100.0
    room = bw.Room(fb, eye, groups, motor, body_b=body)
    before = [(f.x, f.y) for f in (room.arena.A, room.arena.B)]
    previous = 0.0
    heard = []
    for _ in range(20):
        d = room.arena.distance()
        r = room.step()
        assert r["A"]["song_hz"] == 100.0 * len(groups[bw.SONG_KEY])
        assert r["B"]["song_hz"] == 0.0
        assert r["sound_hz"]["B"] == room.channels.sound_hz(previous, d)
        assert r["sound_hz"]["A"] == 0.0
        assert r["B"]["out"]["JO_A"] == pytest.approx(r["sound_hz"]["B"])
        assert np.isfinite(r["B"]["her_answer"]["vpodn_hz"])
        heard.append(r["sound_hz"]["B"])
        previous = r["A"]["song_hz"]
    assert max(heard) > 0.0
    for start, fly in zip(before, (room.arena.A, room.arena.B)):
        assert (fly.x, fly.y) != start


@unittest.skipUnless(os.environ.get("COURTSHIP_REAL_BRAIN") == "1",
                     "set COURTSHIP_REAL_BRAIN=1 to load both connectomes")
class RealBrain(unittest.TestCase):
    def test_room_twenty_steps(self):
        start = time.perf_counter()
        female = load_female()
        groups = female_groups(female)
        self.assertEqual(len(groups["pC1"]), 10)
        self.assertEqual(len(groups["vpoDN"]), 2)
        for k in ("JO_A", "JO_B"):
            self.assertGreater(len(groups[k]), 0)
        with np.load(BUILD / "graph_female.npz") as z:
            raw = z["data"]
            self.assertAlmostEqual(float(female.wdata[female.wdata > 0].sum()),
                                   float(raw[raw > 0].sum() * z["exc_scale"]), delta=1.0)
            self.assertAlmostEqual(float(female.wdata[female.wdata < 0].sum()),
                                   float(raw[raw < 0].sum()), delta=1.0)
        male = FlyBrain()
        # CHOSEN: synthetic luminance eyes isolate graph integration; annotation
        # files are not required. Unknown male soma sides leave paired motors empty.
        male_motor = bw.motor_groups(male, np.full(male.n, "", dtype=str))
        body = her(female)
        room = bw.Room(male, FakeEye(male), bd.present_groups(male), male_motor, body_b=body)
        print("female groups:", {k: len(v) for k, v in groups.items()})
        print("female motor:", {k: len(v) for k, v in body.motor.items()})
        print("female MN9:", len(female.where(type_re="^MN9$")))
        setup = time.perf_counter() - start
        moving = {"A": 0, "B": 0}
        for window in range(20):
            r = room.step()
            self.assertGreaterEqual(r["A"]["song_hz"], 0.0)
            self.assertEqual(r["B"]["song_hz"], 0.0)
            for value in body.answer.values():
                self.assertTrue(np.isfinite(value))
            for k in ("A", "B"):
                self.assertEqual(r[k]["state_carried"], window > 0)
                moving[k] += int(r["after"]["moved_mm"][k] != 0)
        print(f"setup={setup:.3f}s; 20 steps={time.perf_counter() - start - setup:.3f}s; "
              f"total={time.perf_counter() - start:.3f}s; moving windows={moving}; answer={body.answer}")
