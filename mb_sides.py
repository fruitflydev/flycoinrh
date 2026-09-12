"""
Which dopamine cluster innervates each mushroom body output neuron (MBON).

  py mb_sides.py [--weights data/connectome-weights.feather] [--graph build/graph.npz]
                 [--out build/mb_sides.json] [--min-syn 3]

The simulator's weight matrix cannot answer this. build_graph.py gives
dopamine, octopamine and serotonin synapses sign 0 and drops them, so no
PAM->MBON or PPL1->MBON edge is in build/graph.npz. This reads the raw synapse
table instead, keeps neuron pairs joined by at least --min-syn synapses, and
for every MBON type sums the synapses it receives from PAM neurons and from
PPL1 neurons. The larger total names its side. A type with fewer than 20
synapses on its larger side is flagged weak.

Only input counts. An earlier mushroom.py compared MBON output onto the
dopamine neurons, which put 14 of 37 types on the wrong side.

mushroom.py loads the result and stores sides_sha next to learned gains, so
gains learned under a different table are never applied.
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from mushroom import sides_sha

ROOT = Path(__file__).parent
MIN_SYN = 3
WEAK_BELOW = 20


def build(bodies, types, batches, min_syn=MIN_SYN):
    """
    bodies, types : per-neuron arrays, as in build/graph.npz
    batches       : iterable of (body_pre, body_post, weight) arrays
    returns       : the mb_sides.json dict
    """
    bodies = np.asarray(bodies).astype(np.int64)
    types = np.asarray(types).astype(str)
    pam_b = bodies[np.char.startswith(types, "PAM")]
    ppl_b = bodies[np.char.startswith(types, "PPL1")]
    is_mbon = np.char.startswith(types, "MBON")
    mbon_b = bodies[is_mbon]
    type_of = dict(zip(mbon_b.tolist(), types[is_mbon].tolist()))

    received = {"PAM": defaultdict(int), "PPL1": defaultdict(int)}
    scanned = 0
    for pre, post, w in batches:
        pre = np.asarray(pre, dtype=np.int64)
        post = np.asarray(post, dtype=np.int64)
        w = np.asarray(w, dtype=np.int64)
        scanned += len(pre)
        keep = (w >= min_syn) & np.isin(post, mbon_b)
        if not keep.any():
            continue
        pre, post, w = pre[keep], post[keep], w[keep]
        for cluster, members in (("PAM", pam_b), ("PPL1", ppl_b)):
            hit = np.isin(pre, members)
            for b, s in zip(post[hit].tolist(), w[hit].tolist()):
                received[cluster][b] += int(s)

    per_type = {}
    for b, t in sorted(type_of.items()):
        e = per_type.setdefault(t, {"pam_syn": 0, "ppl1_syn": 0, "bodies": 0})
        e["pam_syn"] += received["PAM"].get(b, 0)
        e["ppl1_syn"] += received["PPL1"].get(b, 0)
        e["bodies"] += 1
    for e in per_type.values():
        p, q = e["pam_syn"], e["ppl1_syn"]
        e["side"] = "PAM" if p > q else "PPL1" if q > p else None
        e["weak"] = max(p, q) < WEAK_BELOW
    table = {t: e["side"] for t, e in per_type.items() if e["side"]}
    return {
        "method": (f"DAN->MBON input synapses from the raw synapse table, neuron pairs with >= {min_syn} "
                   "synapses; the cluster giving more input names the side. PAM side = reward dopamine, "
                   "its MBONs promote avoidance; PPL1 side = punishment dopamine, its MBONs promote "
                   "approach (Aso et al. 2014)"),
        "min_syn": min_syn,
        "scanned_edges": scanned,
        "totals": {"pam": sum(e["pam_syn"] for e in per_type.values()),
                   "ppl1": sum(e["ppl1_syn"] for e in per_type.values())},
        "types": dict(sorted(per_type.items())),
        "bodies": {str(b): table[t] for b, t in sorted(type_of.items()) if t in table},
        "sides_sha": sides_sha(table),
    }


def feather_batches(path, columns=("body_pre", "body_post", "weight")):
    """Stream the synapse table one record batch at a time."""
    import pyarrow as pa
    with pa.memory_map(str(path)) as src:
        reader = pa.ipc.open_file(src)
        for i in range(reader.num_record_batches):
            b = reader.get_batch(i)
            yield tuple(b.column(b.schema.get_field_index(c)).to_numpy() for c in columns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "data" / "connectome-weights.feather"))
    ap.add_argument("--graph", default=str(ROOT / "build" / "graph.npz"))
    ap.add_argument("--out", default=str(ROOT / "build" / "mb_sides.json"))
    ap.add_argument("--min-syn", type=int, default=MIN_SYN)
    a = ap.parse_args()
    t0 = time.time()
    z = np.load(a.graph, allow_pickle=False)
    d = build(z["bodies"], z["types"], feather_batches(a.weights), a.min_syn)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(d, indent=1), encoding="utf-8")
    print(f"scanned {d['scanned_edges']:,} edges in {time.time() - t0:.0f}s")
    print(f"DAN->MBON synapses: PAM {d['totals']['pam']:,}  PPL1 {d['totals']['ppl1']:,}")
    for t, e in d["types"].items():
        weak = "  (weak)" if e["weak"] else ""
        print(f"  {t:12} PAM {e['pam_syn']:5}  PPL1 {e['ppl1_syn']:5}  side {e['side']}{weak}")
    sides = [e["side"] for e in d["types"].values()]
    print(f"{sides.count('PAM')} PAM types, {sides.count('PPL1')} PPL1 types; "
          f"sides_sha {d['sides_sha'][:12]}; wrote {a.out}")


if __name__ == "__main__":
    main()
