"""A female listener and her descending-neuron answer in the shared room."""
import time

import numpy as np

from flysim import BUILD, FlyBrain, Params
from backrooms_world import (FlyBody, FRAME_W, FRAME_H, SIM_STEPS, LIF_DT_MS,
                             MOTOR_NAMES, PlumeFly, motor_groups, per_side_scales)


def female_groups(fb):
    """Female type names, under the room's channel keys."""
    patterns = {"JO_A": "^JO-A$", "JO_B": "^JO-B$",
                "pC1": "^pC1[a-e]$", "vpoDN": "^DNp37$"}
    groups = {k: np.asarray(fb.where(type_re=rx), dtype=np.int64)
              for k, rx in patterns.items()}
    for k in ("JO_A", "JO_B", "vpoDN"):
        if not groups[k].size:
            raise KeyError(f"no {k} cells matching {patterns[k]} in female graph")
    return groups


def female_motor(fb):
    """The six walking groups, split by the female graph's soma sides."""
    # Missing types stay empty: their mean motor rate is zero. Unknown sides
    # cannot supply a paired steering/forward group. MN9 is not in MOTOR_NAMES.
    sides = getattr(fb, "soma_side", np.full(fb.n, "", dtype=str))
    return motor_groups(fb, sides)


def load_female(path=BUILD / "graph_female.npz", p=Params(), exc_scale=None):
    """The stored factor calibrated the web-roaming fork; courtship loads raw for parity with the male."""
    fb = FlyBrain(path, p=p)
    with np.load(path, allow_pickle=False) as z:
        # The builder stores raw weights; the fork scaled positive weights
        # at load time. Apply that convention once here, including W itself.
        # CHOSEN: older archives without metadata retain unscaled weights.
        stored = float(z["exc_scale"]) if "exc_scale" in z else 1.0
        fb.exc_scale = stored if exc_scale is None else float(exc_scale)
        fb.soma_side = z["soma_side"].astype(str)
    if not np.isfinite(fb.exc_scale) or fb.exc_scale < 0:
        raise ValueError("exc_scale must be finite and non-negative")
    fb.wdata[fb.wdata > 0] *= fb.exc_scale
    fb.W.data = fb.wdata
    return fb


class BlindEye:
    """Supply no visual input to the listener."""

    def __init__(self, fb):
        self.on_idx = np.empty(0, dtype=np.int64)
        self.off_idx = np.empty(0, dtype=np.int64)

    def look(self, img, cx, cy):
        return {}


class HerBody(FlyBody):
    """Carry female state; accept and ignore smell, hear JO, read an answer."""

    def __init__(self, name, fb, eye, groups, motor, gains=None, sim_steps=SIM_STEPS,
                 seed=0, gaze=(FRAME_W / 2.0, FRAME_H / 2.0), sides=None):
        # Reuse FlyBody's step/state protocol without its male-only selectors.
        self.name, self.fb, self.eye = str(name), fb, eye
        self.gains, self.sim_steps, self.seed = gains, int(sim_steps), int(seed)
        self.gaze = tuple(map(float, gaze))
        self.state, self.windows = None, 0
        self.secs = self.sim_steps * LIF_DT_MS / 1000.0
        self.sides = getattr(fb, "soma_side", None) if sides is None else sides
        self.groups = {k: np.asarray(v, dtype=np.int64) for k, v in groups.items()}
        for k in ("JO_A", "JO_B", "vpoDN"):
            if k not in self.groups or not self.groups[k].size:
                raise KeyError(f"no {k} cells in female body")
        self.groups.setdefault("pC1", np.empty(0, dtype=np.int64))
        self.motor = {k: np.asarray(motor[k], dtype=np.int64) for k in MOTOR_NAMES}
        self.side_scales = {}
        indices, scales = [], []
        for k in ("JO_A", "JO_B"):
            idx = self.groups[k]
            scale, self.side_scales[k] = per_side_scales(idx, self.sides)
            indices.append(idx)
            scales.append(scale)
        self.sound_idx, first = np.unique(np.concatenate(indices), return_index=True)
        self.sound_scale = np.concatenate(scales)[first]
        eye_idx = np.concatenate([np.asarray(getattr(eye, k, []), dtype=np.int64)
                                  for k in ("on_idx", "off_idx")])
        if np.intersect1d(eye_idx, self.sound_idx).size:
            raise ValueError("eye cells overlap sound cells")
        self.rec_idx = np.unique(np.concatenate([*self.groups.values(), *self.motor.values()]))
        self.rec_pos = {k: np.searchsorted(self.rec_idx, v)
                        for k, v in {**self.groups, **self.motor}.items()}
        self.answer = {"vpodn_hz": 0.0, "pc1_hz": 0.0}

    def reset(self, seed=None):
        super().reset(seed)
        self.answer = {"vpodn_hz": 0.0, "pc1_hz": 0.0}

    def drive(self, frame, smell_hz, sound_hz):
        """Eye and sound only; smell_hz is accepted for Room compatibility."""
        drive = dict(self.eye.look(frame, *self.gaze))
        key = tuple(self.sound_idx.tolist())
        if any(np.intersect1d(k, self.sound_idx).size for k in drive):
            raise ValueError("sound drive overlaps the eye's own input")
        a, b = self.sound_pair(sound_hz)
        hz = np.zeros(self.sound_idx.size, dtype=np.float64)
        for group, rate in (("JO_A", a), ("JO_B", b)):
            hz[np.searchsorted(self.sound_idx, self.groups[group])] = rate
        drive[key] = (self.sound_scale * hz).astype(np.float32)
        return drive

    @staticmethod
    def sound_pair(sound):
        """A scalar retains the same nominal rate at both JO groups."""
        a, b = (sound, sound) if np.isscalar(sound) else sound
        return float(max(0., a)), float(max(0., b))

    def absorb(self, r, drive, smell_hz, sound_hz, t0):
        """Record mean answer rates in the same window telemetry as FlyBody."""
        carried = self.state is not None
        self.state = r["_state"]
        self.windows += 1
        per_cell = np.asarray(r["all"], dtype=np.float64)
        rates, counts = {}, {}
        for k, pos in self.rec_pos.items():
            v = per_cell[pos]
            rates[k] = float(v.mean()) if v.size else 0.0
            counts[k] = int(np.rint(v.sum() * self.secs))
        self.answer = {"vpodn_hz": rates["vpoDN"], "pc1_hz": rates["pC1"]}
        motor = {k: rates[k] for k in MOTOR_NAMES}
        turn, speed, parts = PlumeFly.motor_from_rates(motor)
        vals = [v for k, v in drive.items() if k != tuple(self.sound_idx.tolist())]
        vals = (vals + [np.zeros(1), np.zeros(1)])[:2]
        eye_rates = [float(np.asarray(v).mean()) if np.asarray(v).size else 0.0
                     for v in vals[:2]]
        fired = r.get("_fired")
        return {
            "fly": self.name, "window": self.windows, "state_carried": carried,
            "turn": float(turn), "speed": float(speed), **parts, "motor": motor,
            "song_hz": 0.0, "song_group": None, "song_cells": 0,
            "sound_hz": max(self.sound_pair(sound_hz)),
            "her_answer": dict(self.answer),
            "in": {"smell_hz": 0.0, "sound_hz": self.sound_pair(sound_hz),
                   "eye_on_hz": eye_rates[0], "eye_off_hz": eye_rates[1]},
            "out": {k: rates[k] for k in ("JO_A", "JO_B")},
            "rates": rates, "counts": counts,
            "fired": int(len(fired)) if fired is not None else 0,
            "total_hz": float(r["_total_hz"]) if "_total_hz" in r else None,
            "brain_s": time.time() - t0,
        }

    def describe(self):
        """Listener populations and delivered channels."""
        return {"name": self.name, "seed": self.seed, "sim_steps": self.sim_steps,
                "song_group": None, "song_cells": 0, "smell_cells": 0,
                "sound_cells": int(self.sound_idx.size),
                "recorded_cells": int(self.rec_idx.size),
                "groups": {k: int(v.size) for k, v in self.groups.items()},
                "motor_cells": {k: int(v.size) for k, v in self.motor.items()},
                "side_scales": self.side_scales,
                "drive_note": "smell ignored; JO sound equalised per soma side"}
