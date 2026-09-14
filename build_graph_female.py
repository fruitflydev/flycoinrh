"""Build a signed sparse graph from the FlyWire FAFB v783 CSV exports."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

MV_PER_SYNAPSE = 0.275
SIGN = {"ACH": 1.0, "GABA": -1.0, "GLUT": -1.0}
NT = {"ACH": "acetylcholine", "GABA": "gaba", "GLUT": "glutamate",
      "DA": "dopamine", "SER": "serotonin", "OCT": "octopamine"}
ROOT = Path(__file__).parent
DATA = ROOT / "data" / "female"
BUILD = ROOT / "build"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eye", choices=("both", "left", "right"), default="both")
    # Largest tested scale meeting the ceiling and movement limits with both eyes.
    ap.add_argument("--exc-scale", type=float, default=0.5)
    # 100 and 120 Hz tie at 5/48 clicks; use the lower stopping threshold.
    ap.add_argument("--click-hz", type=float, default=100)
    ap.add_argument("--adapt", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--back-scale", type=float, default=1.0)
    args = ap.parse_args()
    if not np.isfinite(args.back_scale) or args.back_scale < 0:
        ap.error("--back-scale must be finite and nonnegative")
    print("loading annotations ...")
    ann = pd.read_csv(DATA / "classification.csv.gz", keep_default_na=False)
    bodies = ann.root_id.to_numpy()
    idx = pd.Index(bodies)
    n = len(bodies)
    print(f"  {n:,} neurons")
    cells = pd.read_csv(DATA / "consolidated_cell_types.csv.gz",
                        keep_default_na=False).set_index("root_id")
    types = cells.primary_type.reindex(bodies).fillna("").to_numpy().astype(str)
    superclass = ann.super_class.to_numpy().astype(str)
    subclass = ann.sub_class.to_numpy().astype(str)
    soma_side = ann.side.map({"left": "L", "right": "R"}).fillna("").to_numpy().astype(str)
    neurons = pd.read_csv(DATA / "neurons.csv.gz", keep_default_na=False).set_index("root_id")
    nt_cell = neurons.nt_type.reindex(bodies).fillna("")
    votes = pd.DataFrame({"type": types, "nt": nt_cell,
                          "score": pd.to_numeric(neurons.nt_type_score.reindex(bodies),
                                                 errors="coerce")})
    votes = votes.loc[(votes.type != "") & (votes.nt != "")]
    ranked = votes.groupby(["type", "nt"]).agg(
        count=("nt", "size"), score=("score", "mean")).reset_index()
    consensus = ranked.sort_values(
        ["type", "count", "score", "nt"], ascending=[True, False, False, True]
    ).drop_duplicates("type").set_index("type").nt
    nt = pd.Series(types, index=nt_cell.index).map(consensus).fillna(nt_cell)
    print(f"  NT labels changed: {(nt != nt_cell).sum():,}")
    print(f"  NT unknowns filled: {((nt_cell == '') & (nt != '')).sum():,}")
    for t in ("L1", "L2", "L3", "Mi1", "T4a"):
        print(f"  {t} consensus: {NT.get(consensus.get(t), 'unknown')}")
    sign = nt.map(SIGN).fillna(0).to_numpy(dtype=np.float32)
    nt_str = nt.map(NT).fillna("unknown").to_numpy().astype("U24")

    columns = pd.read_csv(DATA / "column_assignment.csv.gz")
    assigned = columns.column_id.notna() & columns.p.notna() & columns.q.notna()
    eye = args.eye
    selected = columns.hemisphere.isin(("left", "right") if eye == "both" else (eye,))
    # Both hemispheres sample the same screen in their shared raw axial coordinates.
    columns = columns.loc[assigned & selected].set_index("root_id").reindex(bodies)
    has_hex = columns.p.notna().to_numpy() & columns.q.notna().to_numpy()
    hex1 = columns.p.fillna(0).to_numpy(dtype=np.int32)
    hex2 = columns.q.fillna(0).to_numpy(dtype=np.int32)
    positions = pd.read_csv(DATA / "coordinates.csv.gz").drop_duplicates("root_id").set_index("root_id")
    positions = positions.position.reindex(bodies)
    soma = np.full((n, 3), np.nan, dtype=np.float32)
    have = positions.notna().to_numpy()
    soma[have] = np.stack([np.fromstring(p.strip("[]"), sep=" ", dtype=np.float32)
                          for p in positions[have]])

    print("loading weights (already filtered to >= 5 synapses) ...")
    connections = pd.read_csv(DATA / "connections_princeton.csv.gz",
                              usecols=["pre_root_id", "post_root_id", "syn_count"])
    print(f"  {len(connections):,} rows on disk")
    pairs = connections.groupby(["pre_root_id", "post_root_id"], sort=False).syn_count.sum()
    print(f"  {len(pairs):,} pairs, {int(pairs.sum()):,} synapses")
    c = idx.get_indexer(pairs.index.get_level_values("pre_root_id"))
    r = idx.get_indexer(pairs.index.get_level_values("post_root_id"))
    valid = (c >= 0) & (r >= 0)
    print(f"  {(~valid).sum():,} pairs outside neuron universe")
    c, r = c[valid], r[valid]
    v = pairs.to_numpy(dtype=np.float32)[valid] * MV_PER_SYNAPSE * sign[c]
    keep = v != 0.0
    W = sp.csr_matrix((v[keep], (r[keep], c[keep])), shape=(n, n), dtype=np.float32)
    W.sum_duplicates()
    print(f"  W: {n:,} x {n:,}, {W.nnz:,} nonzero edges")
    print(f"  excitatory edges: {(W.data > 0).sum():,}")
    print(f"  inhibitory edges: {(W.data < 0).sum():,}")
    print(f"  dropped (modulatory/unknown NT): {(~keep).sum():,}")
    for t in ("L1", "L2"):
        print(f"  {t} with hex: {((types == t) & has_hex).sum():,}")
        for hemisphere in ("left", "right"):
            mask = (types == t) & has_hex & (columns.hemisphere.to_numpy() == hemisphere)
            print(f"  {t} with hex {hemisphere}: {mask.sum():,}")
    for side in ("L", "R", ""):
        print(f"  DN side {side or 'unspecified'}: {((superclass == 'descending') & (soma_side == side)).sum():,}")
    print(f"  proboscis motor neurons: {(subclass == 'proboscis_motor_neuron').sum():,}")
    print(f"  eye hemisphere: {eye}")
    print(f"  exc_scale: {args.exc_scale}")
    print(f"  click_hz: {args.click_hz:g}")
    BUILD.mkdir(exist_ok=True)
    np.savez_compressed(
        BUILD / "graph_female.npz",
        data=W.data, indices=W.indices, indptr=W.indptr, shape=W.shape,
        bodies=bodies, sign=sign, types=types, superclass=superclass,
        subclass=subclass, receptor=np.full(n, ""), fru=np.full(n, ""), nt=nt_str,
        nt_cell=nt_cell.map(NT).fillna("unknown").to_numpy().astype("U24"),
        hex1=hex1, hex2=hex2, has_hex=has_hex, soma_side=soma_side,
        soma=soma, eye=np.array([eye]), exc_scale=np.float32(args.exc_scale),
        click_hz=np.float32(args.click_hz),
        adapt=np.bool_(args.adapt), back_scale=np.float32(args.back_scale),
    )
    print(f"wrote {BUILD / 'graph_female.npz'}")


if __name__ == "__main__":
    main()
