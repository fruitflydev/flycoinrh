"""
The captioner, checked on synthetic records so no brain is loaded.

What matters and why:
  * it is deterministic and a pure function of the record (two runs agree,
    the streaming and the array forms agree, the order is fixed);
  * a flat trace - all zero, all constant, or a slow drift - writes nothing,
    so an idle room stays silent and a line always means a crossing;
  * a synthetic P1 burst gives exactly one 'fires' and one 'falls' line with
    the rate, the time and the words that the templates and the dictionary
    prescribe;
  * every geometric event fires once per crossing and is re-armed only by
    the band the module states;
  * no word reaches a line that is not in the templates, the dictionary's
    names, roles and citation keys, or a fly's name (the allow-list regex).

  py -m pytest -q test_backrooms_captions.py
"""
import copy
import re
import unittest

import numpy as np

import backrooms as br
import backrooms_dictionary as bd

DT = br.WORLD_DT_S
# measured counts from build/backrooms_dictionary.json, pinned here so these
# tests need no file on disk
SIZES = {"ORN_DA1": 204, "JO_A": 50, "JO_B": 89, "LC10a": 275, "P1": 86,
         "pC1": 156, "pIP10": 2, "Tk_FruM": 5, "song_pulse_mn": 8,
         "song_ps1": 2, "song_sine_hg1": 2, "wing_mn_all": 66,
         "DNa02": 2, "DNa01": 2, "MDN": 4, "DNp09": 2}
Q2 = br.SPIKE_HZ / 2.0          # one spike in a 2-cell group: 41.667 Hz


def flat_record(n, sizes=SIZES):
    """
    Both flies still, 7 mm apart, each facing the other, every rate zero:
    A at (5, 10) heading 0, B at (12, 10) heading 180. Nothing crosses.
    """
    t = np.arange(n) * DT

    def fly(x, y, h):
        return {"x": np.full(n, x), "y": np.full(n, y), "heading": np.full(n, h),
                "rates": {k: np.zeros(n) for k in sizes},
                "drive": {k: np.zeros(n) for k in ("ORN_DA1", "JO_A", "JO_B")},
                "song": np.zeros(n)}

    return {"t": t, "A": fly(5.0, 10.0, 0.0), "B": fly(12.0, 10.0, 180.0)}


def rich_record(n=400):
    """
    One record that produces every kind of line at least once, for the
    vocabulary and schema checks. Built from flat_record by editing slices.
    """
    r = flat_record(n)
    A, B = r["A"], r["B"]
    A["rates"]["P1"][20:40] = 40.0
    A["rates"]["pC1"][30:50] = 30.0
    B["rates"]["JO_A"][60:80] = 63.0
    B["drive"]["JO_A"][60:80] = 40.0
    B["rates"]["JO_B"][60:80] = 30.0
    B["drive"]["JO_B"][60:80] = 40.0
    A["rates"]["ORN_DA1"][90:110] = 120.0
    A["drive"]["ORN_DA1"][90:110] = 100.0
    A["rates"]["pIP10"][120:140] = Q2
    B["rates"]["Tk_FruM"][120:140] = 100.0
    B["rates"]["song_sine_hg1"][120:140] = br.SPIKE_HZ
    A["song"][150:190] = 300.0
    # A turns away from B and back (3 deg per step)
    A["heading"][200:231] = np.arange(31) * 3.0
    A["heading"][231:260] = 90.0
    A["heading"][260:291] = 90.0 - np.arange(31) * 3.0
    # A walks to B (5 mm/s), stops, backs off (5 mm/s) past 10 mm, stops
    A["x"][300:321] = 5.0 + np.arange(21) * 0.25
    A["x"][321:330] = 10.0
    A["x"][330:371] = 10.0 - np.arange(41) * 0.25
    A["x"][371:] = 0.0
    A["rates"]["MDN"][330:371] = 83.0
    A["rates"]["DNp09"][321:330] = Q2
    return r


def kinds(lines):
    return [ln["kind"] for ln in lines]


class Determinism(unittest.TestCase):
    def test_two_runs_agree_and_seq_is_consecutive(self):
        r = rich_record()
        a = br.caption_record(r, SIZES)
        b = br.caption_record(copy.deepcopy(r), SIZES)
        self.assertEqual(a, b)
        self.assertGreater(len(a), 10)
        self.assertEqual([ln["seq"] for ln in a], list(range(len(a))))

    def test_streaming_equals_the_array_form(self):
        r = rich_record()
        cap = br.Captioner(SIZES)
        got = []
        for i in range(len(r["t"])):
            flies = {}
            for f in ("A", "B"):
                x = r[f]
                flies[f] = {"x": x["x"][i], "y": x["y"][i], "heading": x["heading"][i],
                            "rates": {k: v[i] for k, v in x["rates"].items()},
                            "drive": {k: v[i] for k, v in x["drive"].items()},
                            "song": x["song"][i]}
            got.extend(cap.update(r["t"][i], flies))
        self.assertEqual(got, br.caption_record(r, SIZES))

    def test_line_schema(self):
        for ln in br.caption_record(rich_record(), SIZES):
            self.assertEqual(set(ln), {"seq", "t", "fly", "kind", "group", "value", "text"})
            self.assertIsInstance(ln["seq"], int)
            self.assertIsInstance(ln["t"], float)
            self.assertIn(ln["fly"], br.FLIES)
            self.assertIn(ln["kind"], br.KINDS)
            self.assertIsInstance(ln["group"], str)
            self.assertIsInstance(ln["value"], float)
            self.assertTrue(ln["text"].startswith(ln["fly"] + "  "))
            if ln["group"]:
                self.assertIn(ln["group"], bd.DICTIONARY)

    def test_every_kind_appears_in_the_rich_record(self):
        self.assertEqual(set(kinds(br.caption_record(rich_record(), SIZES))), set(br.KINDS))

    def test_seq_continues_from_seq0(self):
        cap = br.Captioner(SIZES, seq0=17)
        r = flat_record(60)
        r["A"]["rates"]["P1"][20:40] = 40.0
        lines = []
        for i in range(60):
            flies = {f: {"x": r[f]["x"][i], "y": r[f]["y"][i], "heading": r[f]["heading"][i],
                         "rates": {k: v[i] for k, v in r[f]["rates"].items()},
                         "drive": {}, "song": 0.0} for f in ("A", "B")}
            lines.extend(cap.update(r["t"][i], flies))
        self.assertEqual(lines[0]["seq"], 17)

    def test_groups_are_emitted_in_dictionary_order(self):
        r = flat_record(100)
        r["A"]["rates"]["P1"][50:70] = 40.0
        r["A"]["rates"]["pC1"][50:70] = 40.0
        r["A"]["rates"]["LC10a"][50:70] = 40.0
        order = [ln["group"] for ln in br.caption_record(r, SIZES) if ln["kind"] == "onset"]
        want = [k for k in bd.DICTIONARY if k in order]
        self.assertEqual(order, want)


class FlatTraces(unittest.TestCase):
    def test_all_zero_writes_nothing(self):
        self.assertEqual(br.caption_record(flat_record(500), SIZES), [])

    def test_constant_nonzero_writes_nothing(self):
        r = flat_record(500)
        for f in ("A", "B"):
            for k in r[f]["rates"]:
                r[f]["rates"][k][:] = 30.0
            r[f]["drive"]["ORN_DA1"][:] = 100.0
            r[f]["song"][:] = 400.0
        self.assertEqual(br.caption_record(r, SIZES), [])

    def test_constant_high_from_the_first_sample_writes_nothing(self):
        r = flat_record(300)
        r["A"]["rates"]["pIP10"][:] = 200.0
        r["A"]["song"][:] = 1000.0
        self.assertEqual(br.caption_record(r, SIZES), [])

    def test_slow_drift_writes_nothing(self):
        # 0.02 Hz per step: the 10 s baseline lags by about 1 Hz, so the
        # rate is never twice its baseline once it is above the floor
        n = 5000
        r = flat_record(n)
        r["A"]["rates"]["P1"][:] = np.arange(n) * 0.02
        self.assertEqual(br.caption_record(r, SIZES), [])

    def test_still_flies_facing_each_other_write_nothing(self):
        r = flat_record(50)
        r["A"]["heading"][:] = 25.0      # inside the 30 degree cone from the start
        r["B"]["heading"][:] = 200.0
        self.assertEqual(br.caption_record(r, SIZES), [])


class RateEvents(unittest.TestCase):
    P1_ROLE = "courtship command neurons (P1 = pMP4 = pMP-e), fru+ dsx+"

    def test_p1_burst_gives_one_onset_and_one_offset(self):
        r = flat_record(600)
        r["A"]["rates"]["P1"][300:340] = 40.0
        lines = br.caption_record(r, SIZES)
        self.assertEqual(len(lines), 2)
        on, off = lines
        self.assertEqual((on["kind"], on["fly"], on["group"], on["value"]), ("onset", "A", "P1", 40.0))
        self.assertAlmostEqual(on["t"], 302 * DT, places=3)      # third confirming step
        self.assertEqual(on["text"], f"A  P1 fires 40 Hz ({self.P1_ROLE}, Kimura 2008; uncertain match)")
        self.assertEqual((off["kind"], off["fly"], off["group"], off["value"]), ("offset", "A", "P1", 0.0))
        self.assertAlmostEqual(off["t"], 342 * DT, places=3)
        self.assertEqual(off["text"], f"A  P1 falls to 0 Hz ({self.P1_ROLE}, Kimura 2008; uncertain match)")

    def test_measured_groups_carry_no_uncertain_suffix(self):
        r = flat_record(200)
        r["B"]["rates"]["LC10a"][100:120] = 20.0
        lines = br.caption_record(r, SIZES)
        self.assertEqual(lines[0]["text"],
                         "B  LC10a fires 20 Hz (small-object tracking visual projection neurons "
                         "used in courtship pursuit, fru+, Ribeiro 2018)")
        self.assertNotIn("uncertain", lines[0]["text"])

    def test_a_two_step_blip_writes_nothing(self):
        r = flat_record(200)
        r["A"]["rates"]["P1"][100:102] = 40.0
        self.assertEqual(br.caption_record(r, SIZES), [])

    def test_single_spikes_in_a_two_cell_group_write_nothing(self):
        r = flat_record(200)
        r["A"]["rates"]["pIP10"][::2] = Q2
        self.assertEqual(br.caption_record(r, SIZES), [])

    def test_sustained_spiking_in_a_two_cell_group_is_one_event_at_the_quantum(self):
        r = flat_record(200)
        r["A"]["rates"]["pIP10"][100:120] = Q2
        lines = br.caption_record(r, SIZES)
        self.assertEqual(kinds(lines), ["onset", "offset"])
        self.assertAlmostEqual(lines[0]["value"], Q2, places=3)
        self.assertEqual(lines[0]["text"],
                         "A  pIP10 fires 42 Hz (song descending neuron (drives both pulse "
                         "and sine song), von Philipsborn 2011)")

    def test_below_the_floor_writes_nothing(self):
        r = flat_record(200)
        r["A"]["rates"]["P1"][100:150] = 4.9            # floor for 86 cells is 5 Hz
        r["A"]["rates"]["pIP10"][100:150] = 30.0        # floor for 2 cells is 41.67 Hz
        self.assertEqual(br.caption_record(r, SIZES), [])

    def test_less_than_twice_the_baseline_writes_nothing(self):
        r = flat_record(600)
        r["A"]["rates"]["P1"][:] = 20.0
        r["A"]["rates"]["P1"][300:400] = 30.0           # 1.5 x baseline: no
        self.assertEqual(br.caption_record(r, SIZES), [])
        # the 10 s baseline at step 400 is the mean of 100 x 20 and 100 x 30
        # = 25 Hz; exactly 2 x that confirms, 60 Hz confirms with room
        r["A"]["rates"]["P1"][400:500] = 60.0
        lines = br.caption_record(r, SIZES)
        self.assertEqual(kinds(lines), ["onset", "offset"])
        self.assertEqual(lines[0]["value"], 60.0)
        self.assertAlmostEqual(lines[0]["t"], 402 * DT, places=3)
        self.assertEqual(lines[1]["value"], 20.0)       # back to the old level: falls
        self.assertAlmostEqual(lines[1]["t"], 502 * DT, places=3)

    def test_a_burst_cannot_raise_its_own_threshold_while_confirming(self):
        # exactly twice the baseline, held: the confirming samples must not be
        # averaged into the baseline or the third sample would miss 2 x base
        r = flat_record(400)
        r["A"]["rates"]["P1"][:] = 20.0
        r["A"]["rates"]["P1"][300:320] = 40.0
        lines = br.caption_record(r, SIZES)
        self.assertEqual(kinds(lines), ["onset", "offset"])
        self.assertAlmostEqual(lines[0]["t"], 302 * DT, places=3)

    def test_a_failed_confirmation_joins_the_baseline(self):
        # two 40 Hz samples, then quiet: a blip, and afterwards the baseline
        # has moved by exactly those two samples (2 x 40 + 198 x 0) / 200
        det = br.RateDetector(br.floor_hz(86), 200)     # floor 5 Hz
        for v in [0.0] * 200 + [40.0, 40.0] + [0.0]:
            self.assertIsNone(det.update(v))
        self.assertAlmostEqual(det.hist.mean, 80.0 / 200.0, places=9)
        self.assertEqual(det.pending, [])

    def test_a_rise_that_stays_up_is_written_once(self):
        r = flat_record(2000)
        r["A"]["rates"]["P1"][300:] = 40.0
        self.assertEqual(kinds(br.caption_record(r, SIZES)), ["onset"])

    def test_sensory_template_needs_a_nonzero_drive(self):
        r = flat_record(200)
        r["B"]["rates"]["JO_A"][100:120] = 63.0
        r["B"]["drive"]["JO_A"][100:120] = 40.0
        lines = br.caption_record(r, SIZES)
        self.assertEqual(lines[0]["text"],
                         "B  JO-A fires 63 Hz under 40 Hz of sound drive from A's song (sound-sensitive "
                         "Johnston's organ neurons, subgroup A, Kamikouchi 2009)")
        r["B"]["drive"]["JO_A"][:] = 0.0
        lines = br.caption_record(r, SIZES)
        self.assertEqual(lines[0]["text"],
                         "B  JO-A fires 63 Hz (sound-sensitive Johnston's organ neurons, "
                         "subgroup A, Kamikouchi 2009)")

    def test_smell_template(self):
        r = flat_record(200)
        r["A"]["rates"]["ORN_DA1"][100:120] = 120.0
        r["A"]["drive"]["ORN_DA1"][100:120] = 100.0
        lines = br.caption_record(r, SIZES)
        self.assertEqual(lines[0]["text"],
                         "A  ORN_DA1 fires 120 Hz under 100 Hz of cVA drive from B (cVA "
                         "pheromone receptor neurons (Or67d, glomerulus DA1), Kurtovic 2007)")
        self.assertEqual(lines[1]["text"],
                         "A  ORN_DA1 falls to 0 Hz (cVA pheromone receptor neurons (Or67d, "
                         "glomerulus DA1), Kurtovic 2007)")

    def test_motor_readout_and_song_groups_are_never_fires_lines(self):
        r = flat_record(200)
        for k in ("DNa02", "DNa01", "MDN", "DNp09", "song_pulse_mn", "song_ps1", "wing_mn_all"):
            r["A"]["rates"][k][100:150] = 200.0
        self.assertEqual(br.caption_record(r, SIZES), [])
        self.assertNotIn("song_sine_hg1", br.song_family(bd.DICTIONARY, "song_pulse_mn"))

    def test_a_group_without_a_size_is_not_captioned(self):
        sizes = dict(SIZES)
        del sizes["P1"]
        r = flat_record(200)
        r["A"]["rates"]["P1"][100:150] = 40.0
        self.assertEqual(br.caption_record(r, sizes), [])

    def test_floor_follows_the_cell_count(self):
        self.assertAlmostEqual(br.floor_hz(2), br.SPIKE_HZ / 2, places=6)
        self.assertAlmostEqual(br.floor_hz(8), br.SPIKE_HZ / 8, places=6)
        self.assertEqual(br.floor_hz(86), 5.0)
        self.assertEqual(br.floor_hz(204), 5.0)
        self.assertAlmostEqual(br.SPIKE_HZ, 1000.0 / 12.0, places=6)


class Song(unittest.TestCase):
    def test_song_burst_starts_and_ends_once(self):
        r = flat_record(300)
        r["A"]["song"][100:140] = 300.0
        lines = br.caption_record(r, SIZES)
        self.assertEqual(kinds(lines), ["song_start", "song_end"])
        s, e = lines
        self.assertEqual((s["fly"], s["group"], s["value"]), ("A", "song_pulse_mn", 300.0))
        self.assertAlmostEqual(s["t"], 102 * DT, places=3)
        self.assertEqual(s["text"],
                         "A  song starts: pulse-song wing motor neurons 300 Hz summed over 8 cells "
                         "(wing motor neurons wired from pulse-song premotor neurons, Shirangi 2013)")
        self.assertAlmostEqual(e["t"], 142 * DT, places=3)
        self.assertEqual(e["text"], "A  song ends: pulse-song wing motor neurons 0 Hz summed over 8 cells")

    def test_one_spike_per_window_is_not_a_song(self):
        r = flat_record(300)
        r["A"]["song"][100:200] = br.SPIKE_HZ          # 1 spike per window, below 2
        self.assertEqual(br.caption_record(r, SIZES), [])

    def test_the_fallback_group_is_named_as_such(self):
        sizes = {k: v for k, v in SIZES.items() if k != "song_pulse_mn"}
        r = flat_record(300)
        r["B"]["song"][100:140] = 300.0
        lines = br.caption_record(r, sizes)
        self.assertEqual(lines[0]["group"], "wing_mn_all")
        self.assertTrue(lines[0]["text"].startswith("B  song starts: wing motor neurons 300 Hz summed over 66 cells ("))


class Geometry(unittest.TestCase):
    def test_bearing_and_distance(self):
        d, b = br.geometry(0, 0, 0, 3, 4)
        self.assertAlmostEqual(d, 5.0)
        self.assertAlmostEqual(b, 53.130, places=3)
        d, b = br.geometry(0, 0, 90, 0, -1)
        self.assertAlmostEqual(b, -180.0)
        self.assertAlmostEqual(br.geometry(0, 0, 350, 1, 0)[1], 10.0)

    def test_own_turn_crossings_fire_once_each(self):
        r = flat_record(120)
        h = r["A"]["heading"]
        h[10:20] = 90.0                                   # bearing -90: not facing
        h[20:30] = 90.0 - np.arange(10) * 10.0            # sweeps in: 30 -> facing
        h[30:40] = 0.0
        h[40:50] = np.arange(10) * 10.0                   # sweeps out: 50 -> not facing
        h[50:60] = 90.0
        h[60:70] = 90.0 - np.arange(10) * 10.0            # in again
        lines = [ln for ln in br.caption_record(r, SIZES) if ln["kind"] in ("toward", "away")]
        self.assertEqual(kinds(lines), ["away", "toward", "away", "toward"])
        self.assertEqual([ln["value"] for ln in lines], [-90.0, -30.0, -50.0, -30.0])
        self.assertEqual(lines[1]["text"], "A  turns toward B: bearing -30 deg at 7.0 mm")
        self.assertEqual(lines[2]["text"], "A  turns away from B: bearing -50 deg at 7.0 mm")
        self.assertTrue(all(ln["fly"] == "A" for ln in lines))

    def test_a_crossing_from_the_others_motion_says_faces(self):
        r = flat_record(60)
        r["A"]["heading"][:] = 45.0                        # bearing -45 to B at (12, 10)
        y = r["B"]["y"]
        y[10:20] = 10.0 + np.arange(1, 11) * 0.5           # B rises to y = 15
        y[20:30] = 15.0 - np.arange(1, 11) * 0.5           # and comes back
        lines = [ln for ln in br.caption_record(r, SIZES) if ln["kind"] in ("toward", "away")]
        self.assertEqual(kinds(lines), ["toward", "away"])
        self.assertTrue(lines[0]["text"].startswith("A  faces B: bearing -29 deg at 7.3 mm"))
        self.assertTrue(lines[1]["text"].startswith("A  no longer faces B: bearing -41 deg at 7.0 mm"))

    def test_approach_and_leave_fire_once_per_crossing(self):
        r = flat_record(14)
        # distance sequence, B still at x = 12, A walks along x
        d = [7, 7, 5, 2, 3.5, 2.5, 5, 2, 6, 11, 12, 8, 11, 11]
        r["A"]["x"][:] = 12.0 - np.array(d, dtype=float)
        lines = [ln for ln in br.caption_record(r, SIZES) if ln["kind"] in ("approach", "leave")]
        self.assertEqual(kinds(lines), ["approach", "approach", "leave", "leave"])
        self.assertEqual([ln["value"] for ln in lines], [2.0, 2.0, 11.0, 11.0])
        self.assertEqual([ln["t"] for ln in lines], [round(k * DT, 3) for k in (3, 7, 9, 12)])
        self.assertEqual(lines[0]["text"], "A  approaches B: 2.0 mm")
        self.assertEqual(lines[2]["text"], "A  leaves B: 11.0 mm")

    def test_pair_events_name_the_fly_that_moved(self):
        r = flat_record(6)
        r["B"]["x"][:] = np.array([12.0, 12.0, 9.0, 6.0, 9.0, 16.0])   # B walks to A and away
        lines = [ln for ln in br.caption_record(r, SIZES) if ln["kind"] in ("approach", "leave")]
        self.assertEqual([(ln["kind"], ln["fly"]) for ln in lines], [("approach", "B"), ("leave", "B")])
        self.assertEqual(lines[0]["text"], "B  approaches A: 1.0 mm")
        self.assertEqual(lines[1]["text"], "B  leaves A: 11.0 mm")

    def test_starting_inside_the_bands_writes_nothing(self):
        r = flat_record(50)
        r["A"]["x"][:] = 10.5                              # 1.5 mm apart from the start
        self.assertEqual(br.caption_record(r, SIZES), [])
        r["A"]["x"][:] = 0.0                               # 12 mm apart from the start
        self.assertEqual(br.caption_record(r, SIZES), [])

    def test_back_and_stop_fire_once_each_and_rearm(self):
        r = flat_record(40)
        x = r["A"]["x"]
        x[2:12] = 5.0 + np.arange(1, 11) * 0.25            # forward 5 mm/s
        x[12:17] = 7.5 - np.arange(1, 6) * 0.25            # backward 5 mm/s
        x[17:22] = 6.25                                    # still
        x[22:25] = 6.25 - np.arange(1, 4) * 0.25           # backward again
        x[25:] = 5.5                                       # still again
        r["A"]["rates"]["MDN"][12:17] = 83.0
        r["A"]["rates"]["DNp09"][17:22] = Q2
        lines = [ln for ln in br.caption_record(r, SIZES) if ln["kind"] in ("back", "stop")]
        self.assertEqual(kinds(lines), ["back", "stop", "back", "stop"])
        self.assertEqual([ln["t"] for ln in lines], [round(k * DT, 3) for k in (12, 17, 22, 25)])
        self.assertEqual(lines[0]["text"],
                         "A  backs up at 5.0 mm/s (MDN 83 Hz, backward walking descending "
                         "neurons (moonwalker), Bidaye 2014)")
        self.assertEqual(lines[0]["group"], "MDN")
        self.assertEqual(lines[1]["text"],
                         "A  stops (DNp09 42 Hz, forward-walking descending neurons that freeze "
                         "the fly at strong activation; read as stop by the roamer, Bidaye 2020)")
        self.assertEqual(lines[1]["value"], 0.0)
        # the second backing step has MDN silent and the second stop DNp09
        # silent: neither neuron is named, so a silent cell is never printed
        # beside an event as if it caused it
        self.assertEqual(lines[2]["text"], "A  backs up at 5.0 mm/s")
        self.assertEqual(lines[2]["group"], "")
        self.assertEqual(lines[3]["text"], "A  stops")
        self.assertEqual(lines[3]["group"], "")

    def test_a_silent_motor_group_is_not_named_on_a_stop(self):
        # the real run's 96 stops had DNp09 at 0 Hz on 95: those lines must be plain
        r = flat_record(30)
        x = r["A"]["x"]
        x[2:12] = 5.0 + np.arange(1, 11) * 0.25
        x[12:] = 7.5
        lines = [ln for ln in br.caption_record(r, SIZES) if ln["kind"] == "stop"]
        self.assertEqual([(ln["text"], ln["group"]) for ln in lines], [("A  stops", "")])
        r["A"]["rates"]["DNp09"][12] = Q2
        lines = [ln for ln in br.caption_record(r, SIZES) if ln["kind"] == "stop"]
        self.assertEqual(lines[0]["group"], "DNp09")
        self.assertTrue(lines[0]["text"].startswith("A  stops (DNp09 42 Hz, "))

    def test_back_and_stop_without_motor_groups_use_the_plain_lines(self):
        sizes = {k: v for k, v in SIZES.items() if k not in ("MDN", "DNp09")}
        r = flat_record(30)
        x = r["A"]["x"]
        x[2:12] = 5.0 + np.arange(1, 11) * 0.25
        x[12:17] = 7.5 - np.arange(1, 6) * 0.25
        x[17:] = 6.25
        lines = [ln for ln in br.caption_record(r, sizes) if ln["kind"] in ("back", "stop")]
        self.assertEqual([ln["text"] for ln in lines], ["A  backs up at 5.0 mm/s", "A  stops"])
        self.assertEqual([ln["group"] for ln in lines], ["", ""])

    def test_a_fly_that_never_moved_fast_never_stops(self):
        r = flat_record(40)
        r["A"]["x"][:] = 5.0 + np.minimum(np.arange(40), 20) * 0.05    # 1 mm/s then still
        self.assertEqual([ln for ln in br.caption_record(r, SIZES) if ln["kind"] == "stop"], [])


class Citations(unittest.TestCase):
    WANT = {
        "DA1_PN": "Datta 2008", "DC1": "Ruta 2010", "DNa01": "Chen 2018",
        "DNa02": "Rayshubskiy 2020", "DNp09": "Bidaye 2020", "DNp13": "Ruta 2010",
        "JO_A": "Kamikouchi 2009", "JO_B": "Kamikouchi 2009", "LC1": "Ruta 2010",
        "LC10a": "Ribeiro 2018", "MDN": "Bidaye 2014", "ORN_DA1": "Kurtovic 2007",
        "P1": "Kimura 2008", "PPN1": "Kallman 2015", "TN1A": "Lillvis 2024",
        "Tk_FruM": "Asahina 2014", "aIPg": "Schretter 2020", "dMS2": "Lillvis 2024",
        "dMS9": "Lillvis 2024", "dPR1": "von Philipsborn 2011", "mAL": "Kimura 2005",
        "pC1": "Rideout 2010", "pC2l": "Deutsch 2019", "pCd": "Jung 2020",
        "pIP10": "von Philipsborn 2011", "pMP2": "Lillvis 2024",
        "song_ps1": "Shirangi 2013", "song_pulse_mn": "Shirangi 2013",
        "song_sine_hg1": "Shirangi 2013", "vAB3": "Clowney 2015",
        "vMS11": "von Philipsborn 2011", "vMS12": "Lillvis 2024",
        "vPR6": "von Philipsborn 2011", "vPR9": "Lillvis 2024",
        "wing_mn_all": "Takemura 2024",
    }

    def test_every_present_group_has_a_pinned_citation_key(self):
        present = {k for k, e in bd.DICTIONARY.items() if e["present"]}
        self.assertEqual(present, set(self.WANT))
        for k in present:
            self.assertEqual(br.cite_key(bd.DICTIONARY[k]["citation"]), self.WANT[k], k)

    def test_citation_key_shape(self):
        rx = re.compile(r"^(?:von |van |de )?[A-Z][A-Za-z'-]+ (?:19|20)\d\d$")
        for k in self.WANT:
            self.assertRegex(br.cite_key(bd.DICTIONARY[k]["citation"]), rx, k)

    def test_journal_volumes_are_not_years(self):
        self.assertEqual(br.cite_key("Clowney et al. 2015 Neuron 87:1036 Fig. 2-3"), "Clowney 2015")
        self.assertEqual(br.cite_key("Neuron 87:1036; no paper"), "")
        self.assertEqual(br.cite_key("MANC nomenclature: Takemura et al. 2024 eLife"), "Takemura 2024")


class Vocabulary(unittest.TestCase):
    """
    The allow-list is rebuilt here, independently of backrooms.vocabulary(),
    from the templates with their placeholders blanked, the present groups'
    names, roles and citation keys, and the fly names. Numbers are allowed.
    Every token of every line must match.
    """

    @classmethod
    def setUpClass(cls):
        words = set()
        for t in br.TEMPLATES.values():
            words.update(br.tokens(re.sub(r"\{[^}]*\}", " ", t)))
        words.update(br.tokens(br.UNCERTAIN_SUFFIX))
        for e in bd.DICTIONARY.values():
            if e["present"]:
                for field in (e["name"], e["role"], br.cite_key(e["citation"])):
                    words.update(br.tokens(field))
        words.update(br.FLIES)
        cls.words = words
        alts = "|".join(sorted((re.escape(w) for w in words), key=len, reverse=True))
        cls.allowed = re.compile(rf"^(?:{alts}|\d+)$")

    def test_module_vocabulary_matches_this_allow_list(self):
        self.assertEqual(br.vocabulary(), self.words)

    def test_every_token_of_every_line_is_allowed(self):
        lines = br.caption_record(rich_record(), SIZES)
        self.assertEqual(set(kinds(lines)), set(br.KINDS))
        for ln in lines:
            for tok in br.tokens(ln["text"]):
                self.assertRegex(tok, self.allowed, f"{tok!r} in {ln['text']!r}")

    def test_the_templates_carry_no_feeling_words(self):
        bad = re.compile(r"\b(feel|feels|want|wants|love|loves|hate|hates|happy|sad|afraid|"
                         r"angry|excited|curious|lonely|bored|hopes|thinks|knows|decides|"
                         r"tries|likes|enjoys|fears)\b", re.I)
        for t in br.TEMPLATES.values():
            self.assertIsNone(bad.search(t), t)
        self.assertIsNone(bad.search(br.HOW_LINES_ARE_MADE))

    def test_the_templates_carry_no_perception_verbs(self):
        # a sensory line reports a delivered drive and a measured rate; what
        # the fly heard, smelled or saw was not measured and is not claimed;
        # 'sings' would claim a pulse pattern nobody measured
        for w in ("hears", "hear", "smells", "smell", "sings", "sing", "sees", "see",
                  "perceives", "notices"):
            self.assertNotIn(w, br.vocabulary(), w)
        self.assertIn("fires", br.vocabulary())
        self.assertIn("drive", br.vocabulary())

    def test_the_allow_list_helper_agrees_with_this_test(self):
        allowed = br.allowed_pattern()
        for ln in br.caption_record(rich_record(), SIZES):
            for tok in br.tokens(ln["text"]):
                self.assertRegex(tok, allowed)

    def test_reload_matching_is_by_template_not_by_word(self):
        """
        Every line the captioner writes matches a template pattern; a line
        made of allowed words in another arrangement, a line with a role or
        citation the dictionary no longer carries, a foreign word, a wrong
        fly or kind, or no text at all does not.
        """
        pats = br.template_patterns()
        self.assertGreaterEqual(len(pats), len(br.TEMPLATES))       # one per template per named group
        for ln in br.caption_record(rich_record(), SIZES):
            self.assertTrue(br.line_is_from_the_code(ln, pats), ln["text"])
        p1 = self.P1_LINE
        self.assertTrue(br.line_is_from_the_code({"text": p1, "fly": "A", "kind": "onset"}, pats))
        self.assertTrue(br.line_is_from_the_code({"text": p1}, pats))
        self.assertTrue(br.line_is_from_the_code({"text": "B  turns toward A: bearing -30 deg at 7.0 mm"}, pats))
        self.assertTrue(br.line_is_from_the_code({"text": "A  stops"}, pats))
        for bad in ("A  feels lonely",
                    "A  P1 fires 5 Hz",                                    # allowed words, no template
                    "A  fires P1 40 Hz (courtship command neurons (P1 = pMP4 = pMP-e), fru+ dsx+, Kimura 2008; uncertain match)",
                    "A  pCd fires 87 Hz (dsx+ cluster required for cVA-evoked courtship and aggression, Zhou 2014; uncertain match)",
                    "B  stops (DNp09 0 Hz, stop descending neurons (the roamer's reading), Bidaye 2020)",
                    "A  hears it: JO-A 62 Hz, driven at 86 Hz by B's song (sound-sensitive Johnston's organ neurons, subgroup A, Kamikouchi 2009)",
                    "A  Tk-FruM fires 33 Hz (male-specific tachykinin neurons promoting aggression, Asahina 2014)",   # now needs the suffix
                    ""):
            self.assertFalse(br.line_is_from_the_code({"text": bad, "fly": "A", "kind": "onset"}, pats), bad)
        self.assertFalse(br.line_is_from_the_code({"text": p1, "fly": "C", "kind": "onset"}, pats))
        self.assertFalse(br.line_is_from_the_code({"text": p1, "fly": "A", "kind": "poem"}, pats))
        self.assertFalse(br.line_is_from_the_code(p1, pats))

    P1_LINE = ("A  P1 fires 40 Hz (courtship command neurons (P1 = pMP4 = pMP-e), fru+ dsx+, "
               "Kimura 2008; uncertain match)")

    def test_format_line(self):
        ln = br.caption_record(rich_record(), SIZES)[0]
        self.assertEqual(br.format_line(ln), f"{ln['t']:9.3f}  {ln['text']}")


@unittest.skipUnless(bd.JSON_PATH.exists(), "build/backrooms_dictionary.json not present")
class SizesFromDisk(unittest.TestCase):
    def test_sizes_from_json_match_the_pinned_counts(self):
        sizes = br.sizes_from_json()
        for k, n in SIZES.items():
            self.assertEqual(sizes[k], n, k)
        self.assertEqual(len(sizes), 35)
        self.assertNotIn("aDN", sizes)

    def test_captioned_keys_on_the_real_sizes(self):
        keys = br.captioned_keys(bd.DICTIONARY, br.sizes_from_json())
        # 35 present, minus the 4 motor readout groups, minus the song, its
        # part (ps1) and its container (all wing motor neurons); hg1 (sine
        # song) is a sibling, not the song, and keeps its line
        self.assertEqual(len(keys), 35 - 4 - 3)
        for k in ("DNa02", "DNa01", "MDN", "DNp09", "song_pulse_mn", "song_ps1", "wing_mn_all"):
            self.assertNotIn(k, keys)
        self.assertIn("song_sine_hg1", keys)
        self.assertIn("P1", keys)
        self.assertEqual(keys, [k for k in bd.DICTIONARY if k in keys])


if __name__ == "__main__":
    unittest.main()
