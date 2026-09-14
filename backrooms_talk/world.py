"""
The four-fly room: backrooms_world's Arena, Channels and FlyBody, four
bodies on ONE loaded brain, stepped as one batched call on the GPU simulator.

Nothing here writes a word. Each world step:
  1. for every fly, from the positions before the step: the other three flies
     are drawn on its eye frame (Channels.sight_frame_many, each by
     ellipse_of as for one fly); its ORN_DA1 drive is Channels.smell_hz for
     each other fly, summed and clipped at one fly's ceiling; its JO-A / JO-B
     drive is Channels.sound_hz of each other fly's song from the previous
     window, summed and clipped the same way (the several-flies rule is
     CHOSEN and documented at Channels.smell_hz_many);
  2. the four brain windows run as the four columns of one
     FlyBrainGPU.run_batch call (backrooms_world.step_bodies), each column
     carrying its own membrane state and random stream;
  3. each fly's descending neurons give a turn and a speed by the roamer's
     mapping, and all four are applied to the arena from the same state.

Sound is heard one world step late, as in the two-fly room: every window
hears the song the others sang in the previous window.

The record a step returns is the captioner's per-fly shape (x, y, heading in
degrees, rates, drive, song) plus "drive_from": the one fly that delivered
a channel's drive when exactly one did (so a captioner line may name it), and
the brain numbers the readout uses (cells fired, the brain-wide mean rate).
"""
import time
from pathlib import Path

import numpy as np

import backrooms_dictionary as bd
import backrooms_world as bw

# CHOSEN: the four labels. The page shows a turn's `speaker` as sent and
# places it by `fly_index`; the names mean nothing beyond that.
FLY_NAMES = ("A", "B", "C", "D")

GPU_BRAIN_CLASS = "flysim_gpu.FlyBrainGPU"
REPO = Path(bw.__file__).resolve().parent

# where the connectome and the annotations are looked for, in order; a
# checkout whose build/ and data/ are empty (this one) uses the sibling
# flybrain checkout, which holds byte-identical simulator files
GRAPH_CANDIDATES = (REPO / "build" / "graph.npz", REPO.parent / "flybrain" / "build" / "graph.npz")
ANNOTATION_CANDIDATES = (REPO / "data" / "body-annotations.feather",
                         REPO.parent / "flybrain" / "data" / "body-annotations.feather")


def first_existing(paths, what):
    for p in paths:
        if Path(p).exists():
            return Path(p)
    raise FileNotFoundError(f"no {what} at any of: " + ", ".join(str(p) for p in paths))


class TalkRoom:
    """
    Four flies (any number >= 2 works) on one brain. `fb`, `eye`, `groups`,
    `motor`, `gains`, `sides` are exactly what backrooms_world.Room takes, so
    the tests build it on backrooms_world's fake brain.
    """

    def __init__(self, fb, eye, groups, motor, gains=None, seed=0, sim_steps=bw.SIM_STEPS,
                 names=FLY_NAMES, arena=None, channels=None, sides=None, min_radius_px=None):
        self.fb = fb
        self.names = tuple(names)
        self.seed = int(seed)
        self.arena = arena or bw.Arena(seed=self.seed, names=self.names)
        if tuple(f.name for f in self.arena.flies) != self.names:
            raise ValueError("the arena's flies are not these names")
        # CHOSEN: body i's seed is seed x n + 1 + i (the two-fly Room uses
        # seed x 2 + 1 and seed x 2 + 2), so the bodies' streams never coincide
        n = len(self.names)
        self.bodies = {name: bw.FlyBody(name, fb, eye, groups, motor, gains, sim_steps,
                                        seed=self.seed * n + 1 + i, sides=sides)
                       for i, name in enumerate(self.names)}
        first = self.bodies[self.names[0]]
        refractory = getattr(getattr(fb, "p", None), "refractory", bw.REFRACTORY_MS)
        self.song_full_hz = bw.song_full_hz(first.rec_pos[first.song_key].size, refractory)
        if channels is None:
            kw = {} if min_radius_px is None else {"min_radius_px": float(min_radius_px)}
            channels = bw.Channels(song_full_hz=self.song_full_hz, **kw)
        self.channels = channels
        self.song = {name: 0.0 for name in self.names}
        self.groups = dict(first.groups)
        self.sizes = {k: int(len(v)) for k, v in self.groups.items()}
        self.dt = float(self.arena.dt)
        self.n_neurons = int(fb.n)

    @property
    def step_n(self):
        return int(self.arena.t)

    def step(self):
        t0 = time.time()
        flies = self.arena.by_name
        ch = self.channels
        inputs, sources, heard, smelled = [], {}, {}, {}
        for name in self.names:
            me = flies[name]
            others = [flies[o] for o in self.names if o != name]
            dist = {o.name: bw.distance_mm(me, o) for o in others}
            smell = ch.smell_hz_many(dist.values())
            sound = ch.sound_hz_many([(self.song[o], d) for o, d in dist.items()])
            smell_src = [o for o, d in dist.items() if ch.smell_hz(d) > 0.0]
            sound_src = [o for o, d in dist.items() if ch.sound_hz(self.song[o], d) > 0.0]
            src = {}
            if len(smell_src) == 1:
                src[bw.SMELL_KEY] = smell_src[0]
            if len(sound_src) == 1:
                for k in bw.SOUND_KEYS:
                    src[k] = sound_src[0]
            sources[name], heard[name], smelled[name] = src, sound, smell
            inputs.append((ch.sight_frame_many(me, others), smell, sound))
        recs = dict(zip(self.names, bw.step_bodies([self.bodies[n] for n in self.names], inputs)))
        self.song = {name: float(recs[name]["song_hz"]) for name in self.names}
        self.arena.step_all({name: (recs[name]["turn"], recs[name]["speed"]) for name in self.names})

        out = {}
        for name in self.names:
            me, r = flies[name], recs[name]
            out[name] = {
                "x": float(me.x), "y": float(me.y), "heading": float(np.degrees(me.heading)),
                "speed": float(self.arena._last_move[name]) / self.dt,
                "turn": float(np.degrees(self.arena._last_turn[name])) / self.dt,
                "others": {o: {"distance": bw.distance_mm(me, flies[o]),
                               "bearing": bw.bearing_deg(me, flies[o])}
                           for o in self.names if o != name},
                "rates": {k: float(r["rates"][k]) for k in self.groups},
                "drive": {bw.SMELL_KEY: float(smelled[name]),
                          **{k: float(heard[name]) for k in bw.SOUND_KEYS}},
                "drive_from": sources[name],
                "motor": {k: float(v) for k, v in r["motor"].items()},
                "song": float(r["song_hz"]),
                "fired": int(r["fired"]),
                "active_fraction": float(r["fired"]) / float(self.n_neurons),
                "total_hz": r.get("total_hz"),
                "window": int(r["window"]), "state_carried": bool(r["state_carried"]),
            }
        return {"step": self.step_n, "t": float(self.arena.time_s), "flies": out,
                "step_s": time.time() - t0}

    def describe(self):
        first = self.bodies[self.names[0]]
        return {
            "names": list(self.names), "seed": self.seed,
            "arena": self.arena.describe(), "channels": self.channels.describe(),
            "body": first.describe(), "body_seeds": {n: b.seed for n, b in self.bodies.items()},
            "song_full_hz": self.song_full_hz, "n_neurons": self.n_neurons,
            "several_flies": "smell and sound: each other fly's one-fly contribution, summed and "
                             "clipped at one fly's ceiling; sight: every other fly drawn on the frame",
            "batched": callable(getattr(self.fb, "run_batch", None)),
        }


def build_talk_room(fb, gains=None, seed=0, annotations_path=None, sim_steps=bw.SIM_STEPS,
                    names=FLY_NAMES):
    """The real room: build_room's parts (eye, dictionary groups, motor split, root sides) for four flies."""
    ann = Path(annotations_path) if annotations_path else first_existing(ANNOTATION_CANDIDATES, "annotations")
    try:
        import pumpui as eye_module
    except ImportError:                     # the public copy calls it flyeye
        import flyeye as eye_module
    eye = eye_module.FlyEye(fb, annotations_path=str(ann))
    groups = bd.present_groups(fb)
    motor = bw.motor_groups(fb, bw.soma_sides(fb, ann))
    sides = bw.root_sides(fb, ann)
    min_radius = bw.column_spacing_px(bw.distinct_columns(eye))
    return TalkRoom(fb, eye, groups, motor, gains, seed=seed, sim_steps=sim_steps, names=names,
                    sides=sides, min_radius_px=min_radius)


def load_talk_room(seed=0, graph_path=None, annotations_path=None, say=print, ram_check=True,
                   brain_cls=GPU_BRAIN_CLASS):
    """One brain after backrooms_world's free-RAM check, the roamer's gains, four bodies."""
    graph = Path(graph_path) if graph_path else first_existing(GRAPH_CANDIDATES, "graph.npz")
    fb = bw.load_brain(cls=brain_cls, say=say, check=ram_check, graph_path=graph)
    return build_talk_room(fb, bw.room_gains(fb), seed=seed, annotations_path=annotations_path)
