"""
The fly's own learning circuit.

Everything else in this project runs the connectome forward with fixed
weights. This is the one place a weight is allowed to change, and it changes
where a fly's weights actually change: the Kenyon cell (KC) to mushroom body
output neuron (MBON) synapse, under dopamine.

THE RULE (direction measured in flies, constants chosen)
A KC that was active shortly before dopamine arrives has that KC-to-MBON
synapse depressed, not strengthened (Hige et al. 2015, Cohn et al. 2015).
Learning in a fly is subtraction: the mushroom body starts able to drive every
response and experience carves away some of them. So there is no potentiation
here, only depression with a floor, and a drift back toward baseline over
wall-clock time that stands in for forgetting.

WHICH MBONS GET WHICH DOPAMINE (counted from the synapse table)
build/mb_sides.json, written by mb_sides.py, counts the synapses each MBON type
receives from PAM and from PPL1 dopamine neurons and puts the type on the side
that gives it more input. The simulator's weight matrix cannot answer this:
build_graph.py gives dopamine synapses sign 0 and drops them.

An earlier version of this file compared MBON output onto PAM and PPL1 instead
of their input. That put 14 of 37 MBON types on the wrong side, so reward and
punishment each landed partly in the other's compartments. Gains learned under
that split live in mb_gains.npz and are never read (see STORE).

WHAT THE SIDES MEAN FOR BEHAVIOUR
Reward dopamine (PAM) depresses KC input to PAM-compartment MBONs, which drive
avoidance. Punishment dopamine (PPL1) depresses KC input to PPL1-compartment
MBONs, which drive approach (Aso et al. 2014, eLife 3:e04580; measured for the
MBONs they tested, assumed for the rest). calibration.readout() reads them that
way.

WHAT IS HONEST ABOUT THIS AND WHAT IS NOT
* The circuit, the plasticity site and the direction of the rule are real.
* The reward signal is not. A real fly is rewarded by sugar and punished by
  shock. In the backroom, profit arrives as reward dopamine and loss as
  punishment dopamine. People chose that pairing.
* Learning is partly global: a lesson about one smell also shifts the response
  to similar smells (calibration.py and build/backroom_screen.json).
* The learning rate, floor, eligibility decay and memory half-life are chosen.
"""
import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
SIDES = ROOT / "build" / "mb_sides.json"
# Where learning is kept. On a host with a persistent volume, point
# FLY_STATE_DIR at it, or every redeploy wipes what the fly has learned. The
# name is versioned: mb_gains.npz was learned under the old output-based split
# and on the uncalibrated brain, so it is deliberately never read.
STORE = Path(os.environ.get("FLY_STATE_DIR", str(ROOT / "build"))) / "mb_gains.v2.npz"
# How long a lesson lasts, in wall-clock hours. CHOSEN: one training session
# leaves a fly a memory that fades over hours and is mostly gone within a day
# (Tully & Quinn 1985). At 6 h about 6% of a lesson is left after 24 h.
HALF_LIFE_H = float(os.environ.get("FLY_MB_HALF_LIFE_H", "") or 6.0)


def sides_sha(table):
    """Hash of a {MBON type: side} table; stored with gains so they never outlive their sides."""
    return hashlib.sha256(json.dumps(sorted(table.items())).encode("utf-8")).hexdigest()


def load_sides(path=SIDES):
    """
    {MBON type: 'PAM' | 'PPL1'} from mb_sides.json, and its hash.

    The file is derived, not downloaded, and it is gitignored with the rest of
    build/, so a fresh checkout or a deploy image can be missing it. When it is,
    say where it comes from rather than raising a bare path: without the table
    no MBON can be put on a dopamine side and there is no learning circuit at
    all. roam.load_brain catches this and roams on without a mushroom body.
    """
    p = Path(path)
    try:
        raw = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{p} is missing. It is written by mb_sides.py (run: py mb_sides.py), which "
            "counts each MBON type's PAM and PPL1 input from the synapse table.") from None
    d = json.loads(raw)
    table = {str(t): str(v["side"]) for t, v in d["types"].items()
             if v.get("side") in ("PAM", "PPL1")}
    return table, sides_sha(table)


class MushroomBody:
    """Dopamine-gated depression of KC to MBON synapses."""

    def __init__(self, fb, lr=0.06, floor=0.25, trace_decay=0.55, half_life_h=None,
                 calibration="stock", sides=SIDES, store=None, clock=time.time):
        self.fb = fb
        self.lr = lr                  # how hard one dopamine event depresses
        self.floor = floor            # a synapse is never silenced completely
        self.trace_decay = trace_decay
        self.half_life_s = 3600.0 * (HALF_LIFE_H if half_life_h is None else float(half_life_h))
        self.calibration = str(calibration)
        self.store = Path(store) if store is not None else STORE
        self.clock = clock

        types = np.asarray(fb.types).astype(str)
        sel = lambda p: np.where([bool(re.match(p, t)) for t in types])[0]
        self.kc = sel(r"^KC")
        self.mbon = sel(r"^MBON")

        table, self.sides_sha = load_sides(sides)
        mtypes = types[self.mbon]
        self.reward_side = self.mbon[np.array([table.get(t) == "PAM" for t in mtypes], dtype=bool)]
        self.punish_side = self.mbon[np.array([table.get(t) == "PPL1" for t in mtypes], dtype=bool)]
        self.unassigned = sorted({str(t) for t in mtypes if t not in table})

        # Positions in the weight array of every KC to MBON synapse, so a gain
        # can be written straight into the running simulation.
        is_mbon = np.zeros(fb.n, dtype=bool)
        is_mbon[self.mbon] = True
        side = np.zeros(fb.n, dtype=np.int8)
        side[self.reward_side] = 1
        side[self.punish_side] = -1

        pos, pre, post = [], [], []
        for k in self.kc:
            a, b = fb.indptr[k], fb.indptr[k + 1]
            tgt = fb.indices[a:b]
            hit = np.flatnonzero(is_mbon[tgt])
            for h in hit:
                pos.append(a + h)
                pre.append(k)
                post.append(tgt[h])

        self.pos = np.asarray(pos, dtype=np.int64)
        self.pre = np.asarray(pre, dtype=np.int64)
        self.post = np.asarray(post, dtype=np.int64)
        self.side = side[self.post] if len(self.post) else np.array([], np.int8)
        self.base = fb.wdata[self.pos].copy() if len(self.pos) else np.array([])
        self.gain = np.ones(len(self.pos), dtype=np.float32)

        # eligibility: which KCs fired recently, per synapse
        self.trace = np.zeros(len(self.pos), dtype=np.float32)
        self.events = {"reward": 0, "punish": 0}
        self.last_forget = float(self.clock())
        self.loaded = self.load()

    # -- the loop ---------------------------------------------------------
    def observe(self, fired):
        """
        Note which Kenyon cells just fired.

        Eligibility is why the rule is associative rather than global: only a
        synapse whose KC was active in the last moment is available to be
        depressed when dopamine arrives.
        """
        self.trace *= self.trace_decay
        if fired is None or not len(fired) or not len(self.pos):
            return
        active = np.zeros(self.fb.n, dtype=bool)
        active[fired] = True
        self.trace[active[self.pre]] = 1.0

    def dopamine(self, valence, amount=1.0):
        """
        valence  +1 reward (PAM-input MBONs), -1 punishment (PPL1-input MBONs).
        amount   0..1, how strong this event is.

        Depresses the eligible synapses in the addressed compartment and
        returns how many were hit. Nothing is potentiated, because that is not
        what the measured rule does.
        """
        amount = float(np.clip(amount, 0.0, 1.0))
        if not len(self.pos) or amount <= 0.0 or valence == 0:
            return 0
        want = 1 if valence > 0 else -1
        hit = (self.side == want) & (self.trace > 0.05)
        if not hit.any():
            return 0
        self.gain[hit] *= (1.0 - self.lr * amount * self.trace[hit])
        np.clip(self.gain, self.floor, 1.0, out=self.gain)
        self.events["reward" if want > 0 else "punish"] += 1
        return int(hit.sum())

    def forget_trace(self):
        """
        Drop eligibility entirely, so the next dopamine event can only reach
        synapses whose Kenyon cells fire after this call.

        The trace decays by trace_decay per observe() and never reaches zero on
        its own, so roughly the last six things the fly looked at stay eligible.
        That is right while it is walking - the lesson belongs to the moment -
        and wrong when a lesson is delivered for something the fly is not
        looking at any more, which is what a settled sell is. The offline gate
        keeps the same isolation by running six empty observes between pairings
        (backroom_screen.py). CHOSEN.
        """
        if len(self.trace):
            self.trace[:] = 0.0

    def forget(self, now=None):
        """
        Everything drifts back toward baseline with a wall-clock half-life, and
        the decayed gains go straight into the weights the simulation reads.

        Writing them here is the whole point: apply() used to be called only
        after a dopamine delivery, so on a run where nothing settled - days, on
        a long-lived roamer - the fly kept walking on the gains it booted with
        while stats() reported the decayed ones. The decay is a lesson fading,
        and a lesson that has faded has to have faded for the animal too.
        """
        now = float(self.clock() if now is None else now)
        dt = max(0.0, now - self.last_forget)
        self.last_forget = now
        if dt > 0 and len(self.gain) and self.half_life_s > 0:
            keep = np.float32(0.5 ** (dt / self.half_life_s))
            was = self.gain
            self.gain = (1.0 - (1.0 - self.gain) * keep).astype(np.float32)
            if not np.array_equal(was, self.gain):
                self.apply()

    def apply(self):
        """Write the learned gains into the weights the simulation reads."""
        if len(self.pos):
            self.fb.wdata[self.pos] = self.base * self.gain

    # -- reporting --------------------------------------------------------
    def stats(self):
        if not len(self.gain):
            return {"synapses": 0}
        return {
            "synapses": int(len(self.gain)),
            "reward_side": int((self.side == 1).sum()),
            "punish_side": int((self.side == -1).sum()),
            "depressed": int((self.gain < 0.995).sum()),
            "mean_gain": round(float(self.gain.mean()), 4),
            "min_gain": round(float(self.gain.min()), 4),
            "rewards": self.events["reward"],
            "punishments": self.events["punish"],
            "half_life_h": round(self.half_life_s / 3600.0, 3),
            "calibration": self.calibration,
        }

    # -- persistence ------------------------------------------------------
    def save(self):
        """Write the gains atomically. Returns True when written."""
        try:
            self.store.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.store.with_name(self.store.name + ".tmp")
            with open(tmp, "wb") as f:
                np.savez_compressed(
                    f, gain=self.gain, pos=self.pos,
                    sides_sha=np.array(self.sides_sha), calibration=np.array(self.calibration),
                    rewards=self.events["reward"], punishments=self.events["punish"],
                    at=float(self.clock()))
            os.replace(tmp, self.store)
            return True
        except Exception:
            return False

    def load(self):
        """
        Carry learning across runs, but only into the same synapses, the same
        side table and the same calibration. Anything else is discarded rather
        than misapplied. Time spent not running still counts as forgetting.
        """
        try:
            if not self.store.exists():
                return False
            z = np.load(self.store, allow_pickle=False)
            if (len(z["gain"]) != len(self.gain) or not np.array_equal(z["pos"], self.pos)
                    or str(z["sides_sha"]) != self.sides_sha
                    or str(z["calibration"]) != self.calibration):
                return False
            self.gain = z["gain"].astype(np.float32)
            self.events["reward"] = int(z["rewards"])
            self.events["punish"] = int(z["punishments"])
            self.last_forget = float(z["at"])
            self.forget()
            self.apply()
            return True
        except Exception:
            return False


if __name__ == "__main__":
    from flysim import FlyBrain
    fb = FlyBrain()
    mb = MushroomBody(fb)
    print("mushroom body")
    for k, v in mb.stats().items():
        print(f"  {k:14} {v}")
    print(f"  {'unassigned':14} {mb.unassigned}")
