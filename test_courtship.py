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
from test_backrooms_world import FakeBrain, FakeEye, fake_sides, FAKE_TYPES


@pytest.mark.parametrize('frequency,band', [(150, 0), (1000, 1)])
def test_wave_ear_frequency_bands(frequency, band):
    from courtship import WaveEar
    ear = WaveEar()
    ear.hear(.01*np.sin(2*np.pi*frequency*np.arange(1103)/22050))
    levels = ear.band_rms.mean(axis=0)
    assert levels[band] > 10*levels[1-band]


def test_wave_ear_carried_constant_spikes_and_equalisation():
    from courtship import WaveEar
    from unittest.mock import patch
    body = her()
    # A fake spike generator uses a carried integer clock; resetting any subrun
    # changes the event sequence. Rates are actual recorded spike counts / time.
    logs = []
    def run(drive, steps, gains=None, record=None, seed=0, state=None):
        start = 0 if state is None else state['clock']
        events = np.flatnonzero(np.arange(start, start+steps) % 7 == 0) + start
        logs.extend(events.tolist())
        return {'all': np.full(len(record['all']), len(events)/(steps*.0002)),
                '_state': {'clock': start+steps}}
    body.fb.run = run
    with patch.object(WaveEar, 'hear', return_value=np.full((10, 2), 40.)):
        body.ear.clipped = np.zeros((10, 2), bool)
        result = body.step(np.zeros((800, 1280)), 0, np.zeros(1103))
    split = list(logs)
    logs.clear()
    expected = run({}, 250, record={'all': body.rec_idx})
    assert split == logs
    assert body.state == expected['_state']
    assert result['her_answer']['vpodn_hz'] == pytest.approx(expected['all'][0])
    assert result['counts']['vpoDN'] == 2*len(logs)
    # Per-side scaling is the same drive path for every waveform subwindow.
    body = her(female_fake(female_fake().types.tolist() + ['JO-A', 'JO-B']))
    for key in ('JO_A', 'JO_B'):
        idx = body.groups[key]
        body.fb.soma_side[idx] = ['L', 'L', 'R']
    body = her(body.fb)
    rates = body.ear.hear(.01*np.sin(2*np.pi*150*np.arange(1103)/22050))
    for pair in rates:
        drive = body.drive(np.zeros((800, 1280)), 0, pair)[tuple(body.sound_idx)]
        for key in ('JO_A', 'JO_B'):
            pos = np.searchsorted(body.sound_idx, body.groups[key])
            assert drive[pos[:2]].sum() == pytest.approx(drive[pos[2]])
            assert drive[pos[2]] == pytest.approx(2*drive[pos[0]])


class MaleFake(FakeBrain):
    def __init__(self, spont=None):
        super().__init__(list(FAKE_TYPES) + ["ORN_VA1v", "contact", "P1", "hg1", "pIP10"], spont)

    def where(self, type_re=None, receptor=None, **kwargs):
        if receptor is not None:
            assert receptor == "^putative_ppk23$"
            return np.flatnonzero(self.types == "contact")
        return super().where(type_re=type_re, **kwargs)


def make_parts(spont=None):
    fb = MaleFake(spont)
    return fb, FakeEye(fb), bd.present_groups(fb), bw.motor_groups(fb, fake_sides(fb))


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


def test_male_readouts_are_measured_population_mean_and_sums():
    fb, eye, groups, motor = make_parts()
    groups["P1"] = np.array([0, 1])
    groups["song_sine_hg1"] = np.array([2, 3])
    body = bw.FlyBody("A", fb, eye, groups, motor)
    rates = np.arange(body.rec_idx.size, dtype=float) + 1
    result = body.absorb({"all": rates, "_state": {}}, {}, 0, 0, time.time())
    assert result["p1_hz"] == rates[body.rec_pos["P1"]].mean()
    assert result["pulse_hz"] == rates[body.rec_pos["song_pulse_mn"]].sum()
    assert result["sine_hz"] == rates[body.rec_pos["song_sine_hg1"]].sum()
    assert result["song_hz"] == result["pulse_hz"]
    del groups["P1"]
    body = bw.FlyBody("A", fb, eye, groups, motor)
    assert body.step(np.zeros((800, 1280)), 0, 0)["p1_hz"] is None


def test_pair_routes_separate_jo_populations():
    body = her()
    result = body.step(np.zeros((800, 1280)), 999, (80, 23))
    assert result["out"] == {"JO_A": 80., "JO_B": 23.}
    assert result["in"]["sound_hz"] == (80., 23.)
    assert result["sound_hz"] == 80.


@pytest.mark.parametrize("missing", ["song_pulse_mn", "song_sine_hg1"])
def test_missing_song_group_is_unavailable_and_protocol_refuses(missing):
    fb, eye, groups, motor = make_parts()
    del groups[missing]
    body = bw.FlyBody("A", fb, eye, groups, motor)
    result = body.step(np.zeros((800, 1280)), 0, 0)
    assert result["pulse_hz" if missing == "song_pulse_mn" else "sine_hz"] is None
    with pytest.raises(ValueError, match="requires both male song groups"):
        bw.Room(fb, eye, groups, motor, body_b=her())


def test_pair_clipping_measured_before_override():
    fb, eye, groups, motor = make_parts()
    room = bw.Room(fb, eye, groups, motor, body_b=her())
    room.listener_sound = lambda sound: (0., 0.)
    for factors, expected in [((1., 1.), (False, False)),
                              ((1.01, .5), (True, False)),
                              ((.5, 1.01), (False, True))]:
        room.song_pair = tuple(f * ref for f, ref in zip(factors, room.song_pair_full_hz))
        result = room.step()
        assert result["sound_clipped"] == expected
        assert result["B"]["in"]["sound_hz"] == (0., 0.)


def test_dictionary_drive_uses_cached_scales_and_overlap(monkeypatch):
    fb, eye, groups, motor = make_parts()
    groups["partial_eye"] = eye.on_idx[:1]
    groups["partial_sound"] = groups["JO_A"][:1]
    body = bw.FlyBody("A", fb, eye, groups, motor)
    def unexpected(*args, **kwargs):
        raise AssertionError("per-step group computation")
    monkeypatch.setattr(bw, "per_side_scales", unexpected)
    monkeypatch.setattr(np, "intersect1d", unexpected)
    frame = np.zeros((800, 1280))
    actual = body.drive(frame, {bw.SMELL_KEY: 123.}, 45.)
    np.testing.assert_array_equal(actual[tuple(body.smell_idx)], body.smell_scale * 123.)
    for group in ("partial_eye", "partial_sound"):
        with pytest.raises(ValueError, match="overlaps"):
            body.drive(frame, {group: 123.}, 45.)


@pytest.mark.parametrize("distance", [1.9, 2.0, 2.1, 10.0])
def test_female_scent_contact_boundary_and_zero_cva(distance):
    fb, eye, groups, motor = make_parts()
    room = bw.Room(fb, eye, groups, motor, body_b=her())
    room.arena.A.x, room.arena.A.y = 5., 5.
    room.arena.B.x, room.arena.B.y = 5. + distance, 5.
    result = room.step()["A"]
    expected = room.channels.smell_hz(distance)
    assert result["rates"]["female_scent_orn"] == pytest.approx(expected)
    assert result["rates"]["female_scent_contact"] == pytest.approx(expected if distance <= bw.CONTACT_MM else 0.)
    assert result["rates"]["ORN_DA1"] == 0.
    assert result["in"]["smell_hz"]["ORN_DA1"] == 0.


def test_room_pair_uses_previous_window_and_own_full_rates():
    fb, eye, groups, motor = make_parts()
    from types import SimpleNamespace
    fb.p = SimpleNamespace(refractory=3.0)
    room = bw.Room(fb, eye, groups, motor, body_b=her())
    references = tuple(bw.song_full_hz(groups[k].size, fb.p.refractory)
                       for k in ("song_pulse_mn", "song_sine_hg1"))
    assert tuple(groups[k].size for k in ("song_pulse_mn", "song_sine_hg1")) != (8, 2)
    assert room.song_pair_full_hz == references
    room.song_pair = (references[0] * .25, references[1] * .75)
    expected = room.channels.sound_max * room.channels.falloff(room.arena.distance())
    result = room.step()
    assert result["B"]["in"]["sound_hz"] == pytest.approx((expected * .25, expected * .75))
    assert result["A"]["in"]["sound_hz"] == 0.
    assert room.song_pair == (result["A"]["pulse_hz"], result["A"]["sine_hz"])


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
        fb = load_female(path, exc_scale=override, brain_class=raw_brain)
        assert fb.exc_scale == expected
        np.testing.assert_array_equal(fb.wdata, [8 * expected, -6, 0, 4 * expected])
        np.testing.assert_array_equal(fb.W.data, fb.wdata)


@pytest.mark.parametrize("as_string", [False, True])
def test_load_female_class_plumbing(tmp_path, monkeypatch, as_string):
    from types import SimpleNamespace
    path = tmp_path / "female.npz"
    np.savez(path, soma_side=np.array(["L", "R"]))
    calls = []
    class Selected:
        def __init__(self, path, p):
            calls.append((path, p))
            self.wdata = np.array([8., -6.], dtype=np.float32)
            self.W = SimpleNamespace(data=None)
    monkeypatch.setattr(bw, "SelectedFemale", Selected, raising=False)
    from flysim import Params
    p = Params()
    fb = load_female(path, p, .25, "backrooms_world.SelectedFemale" if as_string else Selected)
    assert isinstance(fb, Selected)
    assert calls == [(path, p)]
    np.testing.assert_array_equal(fb.wdata, [2., -6.])


@pytest.mark.skipif(os.environ.get("COURTSHIP_REAL_BRAIN") != "1", reason="set COURTSHIP_REAL_BRAIN=1")
def test_real_scaled_female_gpu_window():
    from flysim_gpu import FlyBrainGPU
    cpu = load_female(exc_scale=.5)
    gpu = load_female(exc_scale=.5, brain_class=FlyBrainGPU)
    np.testing.assert_array_equal(cpu.wdata, gpu.wdata)
    drive = {tuple(cpu.where(type_re="^JO-A$")): 120.,
             tuple(cpu.where(type_re="^JO-B$")): 80.}
    a = cpu.run(drive, 60, seed=17)
    b = gpu.run(drive, 60, seed=17)
    assert gpu.weights_are_current()
    assert a["_total_hz"] == b["_total_hz"], f"CPU={a['_total_hz']!r}, GPU={b['_total_hz']!r}"


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
    assert result['in'] == dict(smell_hz=0., sound_hz=(80., 80.), eye_on_hz=0., eye_off_hz=0.)
    assert result['sound_hz'] == 80.


def test_dark_is_blind_and_silent(tmp_path):
    import courtship_experiment as ce
    male, _, _, _ = make_parts()
    with unittest.mock.patch.object(ce, "FEMALE_EYE", "luminance"):
        room = ce.build_room(3, "dark", brains=(male, female_fake()),
                             annotations_path=tmp_path / "missing")
    body = room.bodies['B']
    assert isinstance(body.eye, BlindEye)
    room.song['A'] = 1000.
    result = room.step()
    assert result['B']['in'] == dict(smell_hz=0., sound_hz=(0., 0.), eye_on_hz=0., eye_off_hz=0.)


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
        expected = room.channels.sound_hz(previous * room.channels.song_full / room.song_pair_full_hz[0], d)
        assert r["sound_hz"]["B"] == (expected, 0.)
        assert r["sound_hz"]["A"] == 0.0
        assert r["B"]["out"]["JO_A"] == pytest.approx(expected)
        assert np.isfinite(r["B"]["her_answer"]["vpodn_hz"])
        heard.append(r["sound_hz"]["B"][0])
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


def test_default_room_without_contribution_module(monkeypatch):
    """The core must build and step its default room with this module absent."""
    import sys
    monkeypatch.setitem(sys.modules, "courtship", None)
    fb, eye, groups, motor = make_parts()
    room = bw.Room(fb, eye, groups, motor)
    result = room.step()
    assert result["A"]["window"] == result["B"]["window"] == 1
