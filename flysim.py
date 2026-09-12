"""
Leaky integrate-and-fire simulation over the male CNS connectome.

Model follows Shiu et al. 2024 (Nature) - the same approach that reproduced
sugar-evoked proboscis extension from the female connectome alone:

  - every neuron is a LIF unit with identical passive parameters
  - a presynaptic spike injects sign * n_synapses * 0.275 mV into each target
  - excitation/inhibition comes from predicted neurotransmitter, not from fitting

The one thing EM cannot tell you is synaptic *efficacy*. That is what stays
free here: `gains`, one scalar per cell type, multiplying that type's outgoing
weights. Wiring is fixed anatomy; gains are the trainable parameters.

Spike propagation uses a ragged-gather over CSC columns, so cost scales with
the number of neurons that actually fired, not with the 160k-neuron population.

A run starts from rest unless it is handed the `_state` a previous run
returned, in which case it continues where that run stopped. The default is
what every caller before the plume experiment relied on, and stays so.
"""
import numpy as np
import scipy.sparse as sp
from pathlib import Path

BUILD = Path(__file__).parent / "build"


class Params:
    v_rest = -52.0      # mV
    v_thresh = -45.0    # mV
    v_reset = -52.0     # mV
    tau_m = 20.0        # ms
    refractory = 2.2    # ms
    dt = 0.2            # ms


class FlyBrain:
    def __init__(self, graph_path=BUILD / "graph.npz", p=Params()):
        z = np.load(graph_path, allow_pickle=False)
        W = sp.csr_matrix(
            (z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"])
        )
        # CSC: column j holds every postsynaptic target of presynaptic neuron j
        self.W = W.tocsc()
        self.indptr = self.W.indptr
        self.indices = self.W.indices
        self.wdata = self.W.data.astype(np.float32)

        self.n = W.shape[0]
        self.bodies = z["bodies"]
        self.types = z["types"].astype(str)
        self.superclass = z["superclass"].astype(str)
        self.subclass = z["subclass"].astype(str)
        self.receptor = z["receptor"].astype(str)
        self.fru = z["fru"].astype(str)
        self.nt = z["nt"].astype(str)
        self.p = p

        # cell-type codes, for per-type trainable gains
        self.type_names, self.type_code = np.unique(self.types, return_inverse=True)
        self.n_types = len(self.type_names)

        self.body_to_i = {int(b): i for i, b in enumerate(self.bodies)}
        self.decay = np.float32(np.exp(-p.dt / p.tau_m))
        self.refr_steps = int(np.ceil(p.refractory / p.dt))

    # ---- population selection -------------------------------------------

    def where(self, *, type_re=None, superclass=None, subclass=None,
              receptor=None, fru=None):
        """Index array of neurons matching the given annotation filters."""
        import re
        m = np.ones(self.n, dtype=bool)
        if type_re is not None:
            rx = re.compile(type_re, re.I)
            m &= np.array([bool(rx.search(t)) for t in self.types])
        if superclass is not None:
            m &= np.isin(self.superclass, np.atleast_1d(superclass))
        if subclass is not None:
            m &= np.isin(self.subclass, np.atleast_1d(subclass))
        if receptor is not None:
            rx = re.compile(receptor, re.I)
            m &= np.array([bool(rx.search(t)) for t in self.receptor])
        if fru is not None:
            rx = re.compile(fru, re.I)
            m &= np.array([bool(rx.search(t)) for t in self.fru])
        return np.flatnonzero(m)

    # ---- simulation ------------------------------------------------------

    def run(self, drive, steps, gains=None, record=None, seed=0, spike_log=False,
            state=None):
        """
        drive     : dict {neuron_index_array: rate_hz} external Poisson input
        steps     : number of dt steps
        gains     : (n_types,) float32 multipliers on outgoing weights, or None
        record    : dict {name: neuron_index_array} populations to count spikes for
        spike_log : also return every spike as (step, neuron_indices), for rendering
        state     : None, or the '_state' dict a previous run returned. With
                    None every neuron starts at rest with no refractory time
                    left and a fresh rng seeded by `seed`, exactly as before
                    this argument existed. With a state the run continues from
                    it (membrane potentials, refractory counters and the rng's
                    bit_generator state) and `seed` is ignored, so k chained
                    runs of N / k steps reproduce one run of N steps spike for
                    spike. The dict passed in is copied, never modified.
        returns   : dict {name: spikes_per_neuron_per_second}, plus '_total',
                    and '_state', the state the run ends in, for the next call
        """
        p = self.p
        n = self.n

        if state is None:
            rng = np.random.default_rng(seed)
            v = np.full(n, p.v_rest, dtype=np.float32)
            refr = np.zeros(n, dtype=np.int32)
        else:
            v = np.array(state["v"], dtype=np.float32, copy=True)
            refr = np.array(state["refr"], dtype=np.int32, copy=True)
            if v.shape != (n,) or refr.shape != (n,):
                raise ValueError(
                    f"state is for {v.shape[0]} neurons, this brain has {n}")
            rng = np.random.default_rng()
            rng.bit_generator.state = state["rng"]

        if gains is None:
            gain_per_neuron = np.ones(n, dtype=np.float32)
        else:
            gain_per_neuron = gains[self.type_code].astype(np.float32)

        # External drive as per-step spike probability.
        # A value may be a scalar rate for the whole group, or a per-neuron
        # array the same length as the key - vision needs the latter, since
        # every retinotopic column sees a different part of the screen.
        if drive:
            idx_parts, p_parts = [], []
            for k, r in drive.items():
                ii = np.asarray(k, dtype=np.int64)
                rr = np.asarray(r, dtype=np.float32)
                if rr.ndim == 0:
                    rr = np.full(len(ii), float(rr), dtype=np.float32)
                elif len(rr) != len(ii):
                    raise ValueError(
                        f"drive rates ({len(rr)}) do not match neurons ({len(ii)})")
                idx_parts.append(ii)
                p_parts.append(np.clip(rr * p.dt / 1000.0, 0.0, 1.0))
            ext_idx = np.concatenate(idx_parts)
            ext_p = np.concatenate(p_parts).astype(np.float32)
        else:
            ext_idx = np.array([], dtype=np.int64)
            ext_p = np.array([], dtype=np.float32)

        record = record or {}
        counts = {k: np.zeros(len(v_), dtype=np.int64) for k, v_ in record.items()}
        total_spikes = 0
        # which neurons fired at all in this window - one bit each, so it costs
        # nothing and lets a caller show the population that actually woke up
        # rather than a rate averaged over everything
        ever = np.zeros(n, dtype=bool)
        log = [] if spike_log else None

        indptr, indices, wdata = self.indptr, self.indices, self.wdata
        thresh, rest, reset, decay = p.v_thresh, p.v_rest, p.v_reset, self.decay

        for _ in range(steps):
            # leak toward rest
            v = rest + (v - rest) * decay

            # external Poisson drive injected as suprathreshold kick
            if len(ext_idx):
                hit = ext_idx[rng.random(len(ext_idx)) < ext_p]
                if len(hit):
                    v[hit] = thresh + 1.0

            v[refr > 0] = reset
            fired = np.flatnonzero((v >= thresh) & (refr <= 0))

            if spike_log:
                log.append(fired.astype(np.int32))

            if len(fired):
                total_spikes += len(fired)
                ever[fired] = True
                refr[fired] = self.refr_steps
                v[fired] = reset

                # ragged gather of all outgoing synapses of the fired neurons
                starts = indptr[fired]
                cnt = indptr[fired + 1] - starts
                tot = int(cnt.sum())
                if tot:
                    off = np.repeat(starts - np.concatenate(([0], np.cumsum(cnt)[:-1])), cnt)
                    g = off + np.arange(tot)
                    tgt = indices[g]
                    val = wdata[g] * np.repeat(gain_per_neuron[fired], cnt)
                    v += np.bincount(tgt, weights=val, minlength=n).astype(np.float32)

                for name, sel in record.items():
                    counts[name] += np.isin(sel, fired)

            refr -= 1

        secs = steps * p.dt / 1000.0
        out = {k: c / secs for k, c in counts.items()}
        out["_total_hz"] = total_spikes / secs / n
        out["_spikes_per_sec"] = total_spikes / secs
        out["_fired"] = np.flatnonzero(ever)
        out["_mean_mv"] = float(v.mean())
        # where the brain is when the window ends, so the next call can go on
        # from here instead of from rest; v and refr are this run's own arrays
        # and the rng state is a fresh dict, so nothing here aliases the input
        out["_state"] = {"v": v, "refr": refr, "rng": rng.bit_generator.state}
        if spike_log:
            out["_spikes"] = log
        return out
