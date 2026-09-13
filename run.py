"""FLYVUE connectome runner; imports only repo flysim.py and flyeye.py."""
import os
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image
import scipy.sparse as sp
from flysim import FlyBrain, Params
from flyeye import FlyEye
from feed import build_feed, fit_to_fov
from heatmap import make_heatmap_png
from order import first_seen

ROOT = Path(__file__).resolve().parent
GRAPH = ROOT / "build" / "graph.npz"
ANNOTATIONS = ROOT / "data" / "body-annotations.feather"
FOV_W, FOV_H = 300, 210
STEPS, DT_MS = 125, 0.2
MEMORY_CEILING = int(2.5 * 1024**3)
ATTRIBUTION = "Connectome © HHMI Janelia / FlyEM / Google Research, CC-BY"

@dataclass
class Simulation:
    heatmap_png: bytes
    spike_log: np.ndarray
    fired: np.ndarray
    soma_xy: np.ndarray
    first_seen: dict
    spikes_per_sec: float
    wall_clock_ms: float
    mode: str
    neuron_count: int
    unique_hex_count: int
    fixation: tuple[int, int]
    attribution: str = ATTRIBUTION

_STATE = {"fb": None, "eye": None, "soma_xy": None, "mode": None}


def _soma_xy(fb):
    import pandas as pd
    a = pd.read_feather(ANNOTATIONS, columns=["bodyId", "somaLocation"]).drop_duplicates("bodyId").set_index("bodyId")
    loc = a["somaLocation"].reindex(fb.bodies).to_numpy()
    xy = np.full((fb.n, 2), np.nan, dtype=np.float32)
    for i, value in enumerate(loc):
        if isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 3:
            try:
                xy[i] = (float(value[0]), float(value[2]))
            except (TypeError, ValueError):
                pass
    ok = np.isfinite(xy).all(axis=1)
    if ok.sum() < 100:
        raise RuntimeError("insufficient soma coordinates")
    lo, hi = np.nanmin(xy[ok], axis=0), np.nanmax(xy[ok], axis=0)
    xy[ok] = (xy[ok] - lo) / np.maximum(hi - lo, 1e-6)
    return xy


def _estimate_full_memory():
    try:
        return GRAPH.stat().st_size * 3.5
    except OSError:
        return 0


def _make_lite():
    """Build an in-memory visual/downstream subgraph; no separate lite module."""
    z = np.load(GRAPH, allow_pickle=False)
    W = sp.csr_matrix((z["data"].astype(np.float32), z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    types = z["types"].astype(str)
    frontier = np.flatnonzero((types == "L1") | (types == "L2"))
    chosen = set(map(int, frontier))
    for _ in range(4):
        if len(frontier) == 0:
            break
        target = np.unique(W[:, frontier].nonzero()[0])
        frontier = np.asarray([i for i in target if int(i) not in chosen], dtype=np.int64)
        chosen.update(map(int, frontier))
    sel = np.asarray(sorted(chosen), dtype=np.int64)
    sub = W[sel][:, sel].tocsc()
    fb = FlyBrain.__new__(FlyBrain)
    fb.W, fb.indptr, fb.indices = sub, sub.indptr, sub.indices
    fb.wdata = sub.data.astype(np.float32)
    fb.n = len(sel)
    fb.bodies = z["bodies"][sel]
    fb.types = types[sel]
    for key in ("superclass", "subclass", "receptor", "fru", "nt"):
        setattr(fb, key, z[key].astype(str)[sel])
    fb.p = Params()
    fb.type_names, fb.type_code = np.unique(fb.types, return_inverse=True)
    fb.n_types = len(fb.type_names)
    fb.body_to_i = {int(b): i for i, b in enumerate(fb.bodies)}
    fb.decay = np.float32(np.exp(-fb.p.dt / fb.p.tau_m))
    fb.refr_steps = int(np.ceil(fb.p.refractory / fb.p.dt))
    return fb


def _load():
    if _STATE["fb"] is not None:
        return _STATE["fb"], _STATE["eye"]
    force_full = os.environ.get("FLYVUE_FORCE_FULL") == "1"
    requested_lite = os.environ.get("FLYVUE_LITE") == "1"
    use_lite = requested_lite or (not force_full and _estimate_full_memory() > MEMORY_CEILING)
    if use_lite:
        fb = _make_lite()
        mode = "lite"
    else:
        fb = FlyBrain(GRAPH)
        mode = "full"
    eye = FlyEye(fb, str(ANNOTATIONS))
    _STATE.update(fb=fb, eye=eye, soma_xy=_soma_xy(fb), mode=mode)
    return fb, eye


def _dense_log(raw, steps, neuron_count):
    log = np.zeros((steps, neuron_count), dtype=np.bool_)
    for step, fired in enumerate(raw[:steps]):
        if len(fired):
            log[step, np.asarray(fired, dtype=np.int64)] = True
    return log


def run_fixation(img, cx, cy, seed=0):
    """Run one centered/FOV fixation; future tiling can call this repeatedly."""
    fb, eye = _load()
    image = img if isinstance(img, Image.Image) else Image.fromarray(np.asarray(img))
    image = fit_to_fov(image, FOV_W, FOV_H)
    cx, cy = int(cx), int(cy)
    if not (0 <= cx < image.width and 0 <= cy < image.height):
        raise ValueError("fixation is outside FOV")
    feed = build_feed(image, fb, eye=eye, fixation=(cx, cy), annotations_path=ANNOTATIONS)
    started = time.perf_counter()
    result = fb.run(feed.drive, steps=STEPS, seed=seed, spike_log=True)
    wall_ms = (time.perf_counter() - started) * 1000.0
    spike_log = _dense_log(result["_spikes"], STEPS, fb.n)
    fired = np.asarray(result["_fired"], dtype=np.int32)
    heat = make_heatmap_png(image, fired, _STATE["soma_xy"])
    return Simulation(
        heatmap_png=heat,
        spike_log=spike_log,
        fired=fired,
        soma_xy=_STATE["soma_xy"],
        first_seen=first_seen(spike_log, DT_MS),
        spikes_per_sec=float(result["_spikes_per_sec"]),
        wall_clock_ms=wall_ms,
        mode=_STATE["mode"],
        neuron_count=int(fb.n),
        unique_hex_count=int(feed.unique_hex_count),
        fixation=(cx, cy),
    )


def run_image(image, seed=0):
    return run_fixation(image, FOV_W // 2, FOV_H // 2, seed)


def health():
    fb, eye = _load()
    unique = set(zip(eye.on_h1.tolist(), eye.on_h2.tolist())) if hasattr(eye, "on_h1") else set()
    return {
        "mode": _STATE["mode"],
        "neuron_count": int(fb.n),
        "unique_hex_count": len(unique),
        "attribution": ATTRIBUTION,
    }
