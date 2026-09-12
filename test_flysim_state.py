"""
State carry in flysim.FlyBrain.run, checked on a small synthetic graph so no
connectome has to be loaded. The property that matters for the plume
experiment: one run of N steps equals k consecutive runs of N / k steps that
carry state, spike for spike, with the same drive and no per-chunk reseeding.
The other property that matters for the roamer: with state=None nothing has
changed, which is checked against values captured before the state argument
existed.

  py -m pytest -q test_flysim_state.py
  PLUME_REAL_BRAIN=1 py -m pytest -q -s test_flysim_state.py -k RealBrain
"""
import copy
import hashlib
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp

import flysim

N_FAKE = 400
FAKE_GRAPH_SEED = 12345


def make_fake_graph(path, n=N_FAKE, seed=FAKE_GRAPH_SEED):
    """
    Write a graph.npz in the shape build_graph.py writes, small enough to
    load in milliseconds: n neurons in 40 types, about eight outgoing
    synapses each, 80 % excitatory, weights that are whole multiples of
    0.275 mV as in the real graph. Deterministic in `seed`, so the values
    captured before the state argument existed can be compared against.
    """
    rng = np.random.default_rng(seed)
    pre = np.repeat(np.arange(n), 8)
    post = rng.integers(0, n, size=pre.size)
    keep = pre != post
    pre, post = pre[keep], post[keep]
    count = rng.integers(1, 6, size=pre.size).astype(np.float32)
    sign = np.where(rng.random(n) < 0.8, 1.0, -1.0).astype(np.float32)
    w = (count * 0.275 * sign[pre]).astype(np.float32)
    W = sp.csr_matrix((w, (post, pre)), shape=(n, n), dtype=np.float32)
    W.sum_duplicates()
    types = np.array([f"T{i % 40:02d}" for i in range(n)], dtype=str)
    blank = np.array([""] * n, dtype=str)
    np.savez(
        path,
        data=W.data.astype(np.float32), indices=W.indices.astype(np.int32),
        indptr=W.indptr.astype(np.int32), shape=np.array(W.shape, dtype=np.int64),
        bodies=np.arange(n, dtype=np.int64) + 1000, sign=sign,
        types=types, superclass=blank, subclass=blank, receptor=blank, fru=blank,
        nt=np.where(sign > 0, "acetylcholine", "gaba").astype(str),
    )
    return path


def fake_brain():
    d = tempfile.mkdtemp(prefix="flysim_fake_")
    return flysim.FlyBrain(make_fake_graph(Path(d) / "graph.npz"))


def fake_drive():
    """Two driven groups, one strong and one weak, so the rng is consumed every step."""
    return {tuple(range(0, 40)): 200.0, tuple(range(40, 60)): 50.0}


def fake_record():
    return {"a": np.arange(0, 100), "b": np.arange(100, 400, 3), "c": np.array([7, 8, 9])}


def digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


# Captured on 2026-09-13 with the run() that had no state argument, on the
# synthetic graph above: drive fake_drive(), 60 steps, seed 7, record
# fake_record(). "gains" adds per-type gains from default_rng(3) and a spike
# log; "nodrive" is an empty drive dict. Each entry is (digest, _mean_mv,
# _spikes_per_sec, number of cells that fired).
BEFORE = {
    "plain": ("6c8cbd7665a2ac3ffcf0521645bf55018bbccecc570698aeec6e4a65cc1bbfbc",
              -51.5153694152832, 7000.0, 45),
    "gains": ("82e8dbfaba650c513cd404717db096c161fa8c9abaf83133c48fb279b37a5b9a",
              -51.3120002746582, 7666.666666666666, 53),
    "nodrive": ("b78965e6c26cb08ccb7f5bc977e1fa465fbc24c8515997ca7c25012fae28da18", -52.0),
}


def chained(fb, drive, chunks, record, gains=None, seed=7, spike_log=False, state=None):
    """Run the chunks in sequence, each continuing from the state the last returned."""
    outs = []
    for n in chunks:
        r = fb.run(drive, n, gains=gains, record=record, seed=seed, spike_log=spike_log, state=state)
        outs.append(r)
        state = r["_state"]
    return outs


def merged(outs, record, dt_ms=0.2):
    """What k chained windows add up to: summed spike counts, the union of fired cells, the joined log."""
    counts = {k: sum(np.rint(o[k] * (len(o["_spikes"]) * dt_ms / 1000.0)) for o in outs) for k in record}
    fired = np.unique(np.concatenate([o["_fired"] for o in outs]))
    log = [f for o in outs for f in o["_spikes"]]
    return counts, fired, log


class DefaultPathUnchanged(unittest.TestCase):
    """With state=None the run is what it was before the argument existed, to the byte."""

    def test_plain_run_matches_the_capture(self):
        fb = fake_brain()
        r = fb.run(fake_drive(), 60, record=fake_record(), seed=7)
        d, mv, sps, nf = BEFORE["plain"]
        self.assertEqual(digest(r["_fired"], r["a"], r["b"], r["c"]), d)
        self.assertEqual(r["_mean_mv"], mv)
        self.assertEqual(r["_spikes_per_sec"], sps)
        self.assertEqual(len(r["_fired"]), nf)

    def test_gains_and_spike_log_match_the_capture(self):
        fb = fake_brain()
        g = np.random.default_rng(3).uniform(0.5, 2.0, fb.n_types).astype(np.float32)
        r = fb.run(fake_drive(), 60, gains=g, record=fake_record(), seed=7, spike_log=True)
        d, mv, sps, nf = BEFORE["gains"]
        self.assertEqual(digest(r["_fired"], r["a"], r["b"], r["c"], *r["_spikes"]), d)
        self.assertEqual(r["_mean_mv"], mv)
        self.assertEqual(r["_spikes_per_sec"], sps)
        self.assertEqual(len(r["_fired"]), nf)

    def test_empty_drive_matches_the_capture(self):
        fb = fake_brain()
        r = fb.run({}, 60, record=fake_record(), seed=7)
        d, mv = BEFORE["nodrive"]
        self.assertEqual(digest(r["_fired"], r["a"]), d)
        self.assertEqual(r["_mean_mv"], mv)

    def test_state_none_is_the_default(self):
        fb = fake_brain()
        a = fb.run(fake_drive(), 30, record=fake_record(), seed=11)
        b = fb.run(fake_drive(), 30, record=fake_record(), seed=11, state=None)
        self.assertEqual(digest(a["_fired"], a["a"]), digest(b["_fired"], b["a"]))
        self.assertEqual(a["_mean_mv"], b["_mean_mv"])

    def test_the_returned_state_has_the_three_parts(self):
        fb = fake_brain()
        r = fb.run(fake_drive(), 10, record=fake_record(), seed=1)
        st = r["_state"]
        self.assertEqual(set(st), {"v", "refr", "rng"})
        self.assertEqual(st["v"].shape, (fb.n,))
        self.assertEqual(st["v"].dtype, np.float32)
        self.assertEqual(st["refr"].shape, (fb.n,))
        self.assertEqual(st["refr"].dtype, np.int32)
        self.assertEqual(st["rng"]["bit_generator"], "PCG64")


class ChunkedEqualsWhole(unittest.TestCase):
    """One run of N steps equals k chained runs of N / k steps, spike for spike."""

    def check(self, chunks, gains=None, seed=7):
        fb = fake_brain()
        drive, record = fake_drive(), fake_record()
        n = sum(chunks)
        whole = fb.run(drive, n, gains=gains, record=record, seed=seed, spike_log=True)
        parts = chained(fb, drive, chunks, record, gains=gains, seed=seed, spike_log=True)
        counts, fired, log = merged(parts, record)
        self.assertEqual(len(log), n)
        for step, (a, b) in enumerate(zip(whole["_spikes"], log)):
            np.testing.assert_array_equal(a, b, err_msg=f"step {step}")
        np.testing.assert_array_equal(whole["_fired"], fired)
        secs = n * fb.p.dt / 1000.0
        for k in record:
            np.testing.assert_array_equal(np.rint(whole[k] * secs), counts[k], err_msg=k)
        self.assertEqual(whole["_mean_mv"], parts[-1]["_mean_mv"])
        np.testing.assert_array_equal(whole["_state"]["v"], parts[-1]["_state"]["v"])
        np.testing.assert_array_equal(whole["_state"]["refr"], parts[-1]["_state"]["refr"])
        self.assertEqual(whole["_state"]["rng"], parts[-1]["_state"]["rng"])
        # the windows are not trivially empty: about 7,000 spikes/s over 400 cells
        self.assertGreater(sum(len(f) for f in log), 0.5 * 7000 * secs)

    def test_four_equal_chunks(self):
        self.check([30, 30, 30, 30])

    def test_uneven_chunks(self):
        self.check([7, 50, 63])

    def test_with_gains(self):
        fb = fake_brain()
        g = np.random.default_rng(5).uniform(0.5, 2.0, fb.n_types).astype(np.float32)
        self.check([40, 40, 40], gains=g)

    def test_one_step_chunks(self):
        self.check([1] * 25)

    def test_carrying_state_is_not_the_same_as_restarting(self):
        """The property has teeth: restarting from rest each chunk gives a different spike train."""
        fb = fake_brain()
        drive, record = fake_drive(), fake_record()
        whole = fb.run(drive, 120, record=record, seed=7, spike_log=True)
        restarted = [fb.run(drive, 30, record=record, seed=7, spike_log=True) for _ in range(4)]
        log = [f for o in restarted for f in o["_spikes"]]
        same = all(np.array_equal(a, b) for a, b in zip(whole["_spikes"], log))
        self.assertFalse(same)


class StateHandling(unittest.TestCase):
    def test_the_state_passed_in_is_not_modified(self):
        fb = fake_brain()
        r = fb.run(fake_drive(), 40, record=fake_record(), seed=2)
        st = r["_state"]
        before = copy.deepcopy(st)
        fb.run(fake_drive(), 40, record=fake_record(), state=st)
        np.testing.assert_array_equal(st["v"], before["v"])
        np.testing.assert_array_equal(st["refr"], before["refr"])
        self.assertEqual(st["rng"], before["rng"])

    def test_two_runs_from_one_state_agree(self):
        fb = fake_brain()
        st = fb.run(fake_drive(), 40, record=fake_record(), seed=2)["_state"]
        a = fb.run(fake_drive(), 40, record=fake_record(), state=st, spike_log=True)
        b = fb.run(fake_drive(), 40, record=fake_record(), state=st, spike_log=True)
        for x, y in zip(a["_spikes"], b["_spikes"]):
            np.testing.assert_array_equal(x, y)
        self.assertEqual(a["_mean_mv"], b["_mean_mv"])

    def test_seed_is_ignored_when_a_state_is_given(self):
        fb = fake_brain()
        st = fb.run(fake_drive(), 40, record=fake_record(), seed=2)["_state"]
        a = fb.run(fake_drive(), 40, record=fake_record(), state=st, seed=1, spike_log=True)
        b = fb.run(fake_drive(), 40, record=fake_record(), state=st, seed=999, spike_log=True)
        for x, y in zip(a["_spikes"], b["_spikes"]):
            np.testing.assert_array_equal(x, y)

    def test_a_state_for_another_brain_is_refused(self):
        fb = fake_brain()
        st = fb.run(fake_drive(), 5, record=fake_record(), seed=2)["_state"]
        bad = dict(st, v=st["v"][:-1])
        with self.assertRaises(ValueError):
            fb.run(fake_drive(), 5, record=fake_record(), state=bad)

    def test_the_returned_arrays_do_not_alias_the_input(self):
        fb = fake_brain()
        st = fb.run(fake_drive(), 5, record=fake_record(), seed=2)["_state"]
        nxt = fb.run(fake_drive(), 5, record=fake_record(), state=st)["_state"]
        self.assertFalse(np.shares_memory(st["v"], nxt["v"]))
        self.assertFalse(np.shares_memory(st["refr"], nxt["refr"]))


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
    """The connectome, once: the default path matches the pre-change capture and chunking holds."""

    BASELINE = Path(__file__).parent / "build" / "plume_v2_baseline_real.npz"

    def test_default_path_matches_capture_and_chunks_agree(self):
        ram = free_ram_gb()
        print(f"\nfree RAM before load: {ram:.1f} GB")
        self.assertGreater(ram, 6.0, "need more than 6 GB free to load a FlyBrain")
        import calibration
        fb = flysim.FlyBrain()
        gains = calibration.gains_for(fb, calibration.CHOSEN)
        if self.BASELINE.exists():
            import plume_fly
            z = np.load(self.BASELINE, allow_pickle=False)
            names = [k[4:] for k in z.files if k.startswith("idx_")]
            record = {k: z["idx_" + k] for k in names}
            jo_e = z["idx_jo_e"]
            side = plume_fly.root_side_of(fb)
            drive = {tuple(z["idx_orn"].tolist()): float(z["drive_orn_hz"]),
                     tuple(jo_e[side[jo_e] == "L"].tolist()): float(z["drive_jo_e_hz"]),
                     tuple(jo_e[side[jo_e] == "R"].tolist()): float(z["drive_jo_e_hz"])}
            r = fb.run(drive, int(z["steps"]), gains=gains, record=record, seed=int(z["seed"]))
            np.testing.assert_array_equal(r["_fired"], z["_fired"])
            self.assertEqual(r["_mean_mv"], float(z["_mean_mv"]))
            self.assertEqual(r["_spikes_per_sec"], float(z["_spikes_per_sec"]))
            for k in names:
                np.testing.assert_array_equal(r[k], z["rate_" + k], err_msg=k)
            print("real brain: default path matches the capture taken before the state argument existed")
        else:
            print("no baseline file; skipping the capture comparison")
            drive = {tuple(fb.where(type_re=r"^ORN_").tolist()): 100.0}
            record = {"dn": fb.where(type_re=r"^DNa02$")}
        whole = fb.run(drive, 60, gains=gains, record=record, seed=7, spike_log=True)
        parts = chained(fb, drive, [30, 30], record, gains=gains, seed=7, spike_log=True)
        log = [f for o in parts for f in o["_spikes"]]
        for a, b in zip(whole["_spikes"], log):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(whole["_mean_mv"], parts[-1]["_mean_mv"])
        print("real brain: 2 x 30 carried steps equal one run of 60, spike for spike")


if __name__ == "__main__":
    unittest.main()
