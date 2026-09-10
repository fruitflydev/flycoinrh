"""
The fly's own learning circuit, driven by what happens on the internet.

Everything else in this project runs the connectome forward with fixed
weights. This is the one place a weight is allowed to change, and it changes
where a fly's weights actually change: the Kenyon cell to MBON synapse in the
mushroom body, under dopamine.

The rule is the one that has been measured in Drosophila. A Kenyon cell that
was active shortly before a dopaminergic neuron fires has *that* KC-to-MBON
synapse depressed - not strengthened. Learning in a fly is subtraction: the
mushroom body starts able to drive every response and experience carves away
the ones that did not pay. So there is no potentiation here, only depression
with a floor, and a slow drift back toward baseline that stands in for
forgetting.

Which MBONs count as reward-side and which as punishment-side is not
hardcoded from a table. It is read out of this connectome: for each MBON,
total PAM input weight is compared against total PPL1 input weight and the
stronger one wins. That split puts MBON01, 02 and 03 on the reward side and
MBON04, 10 and 11 on the punishment side, which is where the literature puts
them - a reassuring sign the split is finding real structure rather than
noise.

What is honest about this and what is not:

* The circuit, the plasticity site and the direction of the rule are real.
* The reward signal is not. A fly is rewarded by sugar, not by reaching a web
  page. Novelty stands in for it here, which is a modelling choice made by a
  person, and the fly has no say in it.
* The compartments are lopsided in this data - 942 KC-to-MBON edges sit on
  the reward side against 75 on the punishment side - so punishment has far
  less to work with than reward does. That asymmetry is in the measurement,
  not in the code.
"""
import re
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
STORE = ROOT / "build" / "mb_gains.npz"


class MushroomBody:
    """Dopamine-gated depression of KC to MBON synapses."""

    def __init__(self, fb, lr=0.06, floor=0.25, recover=0.0008, trace_decay=0.55):
        self.fb = fb
        self.lr = lr                  # how hard one dopamine event depresses
        self.floor = floor            # a synapse is never silenced completely
        self.recover = recover        # drift back toward 1.0, i.e. forgetting
        self.trace_decay = trace_decay

        types = np.asarray(fb.types).astype(str)
        sel = lambda p: np.where([bool(re.match(p, t)) for t in types])[0]
        self.kc = sel(r"^KC")
        self.mbon = sel(r"^MBON")
        pam, ppl1 = sel(r"^PAM"), sel(r"^PPL1")

        W = fb.W                      # CSC: column j = targets of presynaptic j
        pam_in = np.abs(np.asarray(W[pam][:, self.mbon].sum(axis=0)).ravel())
        ppl_in = np.abs(np.asarray(W[ppl1][:, self.mbon].sum(axis=0)).ravel())
        self.reward_side = self.mbon[pam_in > ppl_in]
        self.punish_side = self.mbon[ppl_in > pam_in]

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
        self.load()

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
        valence  +1 reward (PAM compartments), -1 punishment (PPL1 ones).

        Depresses the eligible synapses in the addressed compartment. Nothing
        is potentiated, because that is not what the measured rule does.
        """
        if not len(self.pos):
            return 0
        want = 1 if valence > 0 else -1
        hit = (self.side == want) & (self.trace > 0.05)
        if not hit.any():
            return 0
        self.gain[hit] *= (1.0 - self.lr * amount * self.trace[hit])
        np.clip(self.gain, self.floor, 1.0, out=self.gain)
        self.events["reward" if want > 0 else "punish"] += 1
        return int(hit.sum())

    def forget(self):
        """Everything drifts back toward baseline. Memory is not free."""
        if len(self.gain):
            self.gain += (1.0 - self.gain) * self.recover

    def apply(self):
        """Write the learned gains into the weights the simulation reads."""
        if len(self.pos):
            self.fb.wdata[self.pos] = self.base * self.gain

    # -- reporting --------------------------------------------------------
    def stats(self):
        if not len(self.gain):
            return {"synapses": 0}
        learned = int((self.gain < 0.995).sum())
        return {
            "synapses": int(len(self.gain)),
            "reward_side": int((self.side == 1).sum()),
            "punish_side": int((self.side == -1).sum()),
            "depressed": learned,
            "mean_gain": round(float(self.gain.mean()), 4),
            "min_gain": round(float(self.gain.min()), 4),
            "rewards": self.events["reward"],
            "punishments": self.events["punish"],
        }

    # -- persistence ------------------------------------------------------
    def save(self):
        try:
            STORE.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                STORE, gain=self.gain, pos=self.pos,
                rewards=self.events["reward"], punishments=self.events["punish"],
                at=time.time())
        except Exception:
            pass

    def load(self):
        """
        Carry learning across runs, but only if the graph still matches.

        A gain vector is meaningless against a different set of synapses, so a
        mismatch is discarded rather than misapplied.
        """
        try:
            if not STORE.exists():
                return False
            z = np.load(STORE)
            if len(z["gain"]) != len(self.gain) or not np.array_equal(z["pos"], self.pos):
                return False
            self.gain = z["gain"].astype(np.float32)
            self.events["reward"] = int(z["rewards"])
            self.events["punish"] = int(z["punishments"])
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
