"""
Build a signed, sparse connectivity matrix from the FlyEM male CNS connectome v1.0.

Inputs (CC-BY, storage.googleapis.com/flyem-male-cns):
  data/connectome-weights.feather      pre -> post synapse counts
  data/body-neurotransmitters.feather  consensus neurotransmitter per body
  data/body-annotations.feather        cell type / superclass per body

Output:
  build/graph.npz   W (CSR, mV per presynaptic spike), body index, sign, type codes
                    and anatomical PAM/PPL1 -> MBON synapse counts

Sign convention follows Shiu et al. 2024 (Nature): acetylcholine excitatory,
GABA and glutamate inhibitory. Monoamines are modulatory in reality; they are
given zero fast weight here rather than being faked as excitatory.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path

MV_PER_SYNAPSE = 0.275  # Shiu et al. 2024
MIN_SYN = 3  # drop 1-2 synapse pairs: dominated by reconstruction noise

SIGN = {
    "acetylcholine": +1.0,
    "gaba": -1.0,
    "glutamate": -1.0,
    "dopamine": 0.0,
    "octopamine": 0.0,
    "serotonin": 0.0,
    "histamine": -1.0,  # histamine is inhibitory at fly photoreceptor synapses
    "unclear": 0.0,
    "unknown": 0.0,
}

ROOT = Path(__file__).parent
DATA = ROOT / "data"
BUILD = ROOT / "build"


def main():
    BUILD.mkdir(exist_ok=True)

    print(f"loading weights (keeping pairs with >= {MIN_SYN} synapses) ...")
    import pyarrow.feather as pf
    import pyarrow.compute as pc

    tbl = pf.read_table(DATA / "connectome-weights.feather")
    print(f"  {tbl.num_rows:,} pre->post pairs on disk")
    tbl = tbl.filter(pc.greater_equal(tbl.column("weight"), MIN_SYN))
    pre_a = tbl.column("body_pre").to_numpy()
    post_a = tbl.column("body_post").to_numpy()
    wt_a = tbl.column("weight").to_numpy().astype(np.float32)
    del tbl
    print(f"  {len(wt_a):,} pairs kept, {int(wt_a.sum()):,} synapses")

    print("loading annotations ...")
    ann = pd.read_feather(DATA / "body-annotations.feather")
    ann["t"] = ann["type"].fillna(ann["flywireType"]).fillna(ann["instance"]).fillna("")

    print("loading neurotransmitters ...")
    nt = pd.read_feather(DATA / "body-neurotransmitters.feather")
    nt = nt[["body", "consensus_nt"]].dropna(subset=["body"])
    nt = nt.drop_duplicates(subset=["body"])

    # The weights table is keyed on every EM segment, including millions of
    # unproofread fragments. Only 'Traced' bodies are actual reconstructed
    # neurons (165,122 of them - the number in the paper's headline).
    neurons = ann.loc[(ann.status == "Traced") & (ann.statusLabel != "Glia"), "bodyId"]
    bodies = np.sort(neurons.unique())
    n = len(bodies)
    print(f"  {n:,} traced neurons (of {ann.bodyId.nunique():,} annotated bodies)")

    # keep only edges where both endpoints are real neurons
    valid = np.zeros(int(max(pre_a.max(), post_a.max())) + 1, dtype=bool)
    valid[bodies] = True
    edge_ok = valid[pre_a] & valid[post_a]
    pre_a, post_a, wt_a = pre_a[edge_ok], post_a[edge_ok], wt_a[edge_ok]
    print(f"  {len(wt_a):,} neuron->neuron edges, {int(wt_a.sum()):,} synapses")

    idx = pd.Series(np.arange(n, dtype=np.int32), index=bodies)

    # per-body sign
    ntmap = nt.set_index("body")["consensus_nt"]
    nt_str = ntmap.reindex(bodies).fillna("unknown").str.lower().to_numpy().astype("U24")
    sign = np.array([SIGN.get(s, 0.0) for s in nt_str], dtype=np.float32)

    # per-body metadata aligned to index
    ann_i = ann.drop_duplicates(subset=["bodyId"]).set_index("bodyId")
    types = ann_i["t"].reindex(bodies).fillna("").to_numpy().astype(str)
    superclass = ann_i["superclass"].reindex(bodies).fillna("").to_numpy().astype(str)
    subclass = ann_i["subclass"].reindex(bodies).fillna("").to_numpy().astype(str)
    receptor = ann_i["receptorType"].reindex(bodies).fillna("").to_numpy().astype(str)
    fru = ann_i["fruDsx"].reindex(bodies).fillna("").to_numpy().astype(str)

    print("building sparse matrix ...")
    r = idx.loc[post_a].to_numpy()   # row = postsynaptic
    c = idx.loc[pre_a].to_numpy()    # col = presynaptic
    # Retain anatomical dopamine inputs separately from zero fast weights.
    is_dop = np.char.startswith(types, "PAM") | np.char.startswith(types, "PPL1")
    is_mbon = np.char.startswith(types, "MBON")
    dop = is_dop[c] & is_mbon[r]
    dop_pre, dop_post, dop_count = c[dop], r[dop], wt_a[dop]
    v = wt_a * MV_PER_SYNAPSE * sign[c]

    keep = v != 0.0
    W = sp.csr_matrix(
        (v[keep], (r[keep], c[keep])), shape=(n, n), dtype=np.float32
    )
    W.sum_duplicates()

    print(f"  W: {W.shape[0]:,} x {W.shape[1]:,}, {W.nnz:,} nonzero edges")
    print(f"  excitatory edges: {(W.data > 0).sum():,}")
    print(f"  inhibitory edges: {(W.data < 0).sum():,}")
    print(f"  dropped (modulatory/unknown NT): {(~keep).sum():,}")

    np.savez_compressed(
        BUILD / "graph.npz",
        data=W.data, indices=W.indices, indptr=W.indptr, shape=W.shape,
        bodies=bodies, sign=sign, types=types, superclass=superclass,
        subclass=subclass, receptor=receptor, fru=fru, nt=nt_str,
        dop_pre=dop_pre, dop_post=dop_post, dop_count=dop_count,
    )
    print(f"wrote {BUILD / 'graph.npz'}")


if __name__ == "__main__":
    main()
