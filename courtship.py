"""A female listener and her descending-neuron answer in the shared room."""
import time

import numpy as np
from scipy.signal import butter, sosfiltfilt
from song import Singer, SAMPLE_RATE, PIP10_FULL, rms

from flysim import BUILD, FlyBrain, Params
import backrooms_world as bw
from backrooms_world import (FlyBody, FRAME_W, FRAME_H, SIM_STEPS, LIF_DT_MS,
                             MOTOR_NAMES, PlumeFly, motor_groups, per_side_scales)

BANDS = ((100, 500), (500, 2500))
SPSN_HZ = 50.0  # CHOSEN tonic unmated-state encoding; SpsP identity is uncertain.
FILTERS = [butter(4, band, btype="bandpass", fs=SAMPLE_RATE, output="sos")
           for band in BANDS]
# CHOSEN: one second (including filter edges) of the full pure pulse reference.
RMS_FULL = rms(sosfiltfilt(FILTERS[0], Singer().render(
    PIP10_FULL, 8., 0., seconds=1.), padlen=27))


class WaveEar:
    """The fork's two filters, read in ten 5 ms windows rather than 500 ms."""
    def hear(self, waveform):
        wave = np.asarray(waveform, dtype=float)
        if wave.ndim != 1 or wave.size not in (1102, 1103):
            raise ValueError("ear requires 50 ms of mono waveform at 22050 Hz")
        bands = [sosfiltfilt(sos, wave, padlen=min(27, wave.size-1)) for sos in FILTERS]
        self.band_rms = np.array([[rms(chunk) for chunk in np.array_split(band, 10)]
                                 for band in bands]).T
        self.clipped = self.band_rms > RMS_FULL
        return bw.SOUND_MAX_HZ*np.clip(self.band_rms/RMS_FULL, 0, 1)

    def describe(self):
        return dict(bands_hz=BANDS, filter_order=4, filter="sosfiltfilt", padlen=27,
                    subwindows=10, lif_steps_per_subwindow=25, rms_full=RMS_FULL,
                    reference="CHOSEN: JO-A RMS of one second of full pure pulse train",
                    note="CHOSEN: her brain runs in real time so pulse timing can reach it")


def female_groups(fb):
    """Female type names, under the room's channel keys."""
    patterns = {"JO_A": "^JO-A$", "JO_B": "^JO-B$",
                "pC1": "^pC1[a-e]$", "vpoDN": "^DNp37$",
                "SpsP": "^SpsP$", "oviDN": "^(oviDNa_a|oviDNa_b|oviDNb)$"}
    groups = {k: np.asarray(fb.where(type_re=rx), dtype=np.int64)
              for k, rx in patterns.items()}
    for k in ("JO_A", "JO_B", "vpoDN", "SpsP", "oviDN"):
        if not groups[k].size:
            raise KeyError(f"no {k} cells matching {patterns[k]} in female graph")
    return groups


def female_motor(fb):
    """The six walking groups, split by the female graph's soma sides."""
    # Missing types stay empty: their mean motor rate is zero. Unknown sides
    # cannot supply a paired steering/forward group. MN9 is not in MOTOR_NAMES.
    sides = getattr(fb, "soma_side", np.full(fb.n, "", dtype=str))
    return motor_groups(fb, sides)


def load_female(path=BUILD / "graph_female.npz", p=Params(), exc_scale=None, brain_class=FlyBrain):
    """The stored factor calibrated the web-roaming fork; courtship loads raw for parity with the male."""
    cls = bw.brain_class(brain_class) if isinstance(brain_class, str) else brain_class
    fb = cls(path, p=p)
    with np.load(path, allow_pickle=False) as z:
        # The builder stores raw weights; the fork scaled positive weights
        # at load time. Apply that convention once here, including W itself.
        # CHOSEN: older archives without metadata retain unscaled weights.
        stored = float(z["exc_scale"]) if "exc_scale" in z else 1.0
        fb.exc_scale = stored if exc_scale is None else float(exc_scale)
        fb.soma_side = z["soma_side"].astype(str)
    if not np.isfinite(fb.exc_scale) or fb.exc_scale < 0:
        raise ValueError("exc_scale must be finite and non-negative")
    # Indexed assignment notifies GPU tracked weights; run() refreshes the device.
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

    listens_in_pairs = True

    def __init__(self, name, fb, eye, groups, motor, gains=None, sim_steps=SIM_STEPS,
                 seed=0, gaze=(FRAME_W / 2.0, FRAME_H / 2.0), sides=None,
                 state="virgin"):
        # Reuse FlyBody's step/state protocol without its male-only selectors.
        self.name, self.fb, self.eye = str(name), fb, eye
        self.gains, self.sim_steps, self.seed = gains, int(sim_steps), int(seed)
        self.gaze = tuple(map(float, gaze))
        if state not in ("virgin", "mated"):
            raise ValueError("state must be virgin or mated")
        self.state, self.brain_state, self.windows = state, None, 0
        self.secs = self.sim_steps * LIF_DT_MS / 1000.0
        self.sides = getattr(fb, "soma_side", None) if sides is None else sides
        self.groups = {k: np.asarray(v, dtype=np.int64) for k, v in groups.items()}
        for k in ("JO_A", "JO_B", "vpoDN", "SpsP", "oviDN"):
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
        if np.intersect1d(self.groups["SpsP"], self.sound_idx).size:
            raise ValueError("SpsP cells overlap sound cells")
        eye_idx = np.concatenate([np.asarray(getattr(eye, k, []), dtype=np.int64)
                                  for k in ("on_idx", "off_idx")])
        if np.intersect1d(eye_idx, self.sound_idx).size:
            raise ValueError("eye cells overlap sound cells")
        self.rec_idx = np.unique(np.concatenate([*self.groups.values(), *self.motor.values()]))
        self.rec_pos = {k: np.searchsorted(self.rec_idx, v)
                        for k, v in {**self.groups, **self.motor}.items()}
        self.answer = dict(vpodn_hz=0., pc1_hz=0., ovidn_hz=0., spsp_hz=0.)
        self.ear = WaveEar()

    def step(self, frame, smell_hz, sound_hz):
        if not isinstance(sound_hz, np.ndarray) or sound_hz.ndim != 1 or sound_hz.size <= 2:
            t0 = time.time()
            drive = self.drive(frame, smell_hz, sound_hz)
            result = self.fb.run(drive, steps=self.sim_steps, gains=self.gains,
                                 record={"all": self.rec_idx}, seed=self.seed,
                                 state=self.brain_state)
            return self.absorb(result, drive, smell_hz, sound_hz, t0)
        t0 = time.time()
        pairs = self.ear.hear(sound_hz)
        state, results, drives = self.brain_state, [], []
        for pair in pairs:
            drive = self.drive(frame, smell_hz, pair)
            result = self.fb.run(drive, steps=25, gains=self.gains,
                                 record={"all": self.rec_idx}, seed=self.seed, state=state)
            state = result["_state"]
            results.append(result)
            drives.append(drive)
        combined = dict(results[-1])
        combined["all"] = np.mean([r["all"] for r in results], axis=0)
        if all("_total_hz" in r for r in results):
            combined["_total_hz"] = float(np.mean([r["_total_hz"] for r in results]))
        # _fired is an end-of-run event, not a count over the complete window.
        combined.pop("_fired", None)
        mean_drive = {k: np.mean([d[k] for d in drives], axis=0) for k in drives[0]}
        self.sim_steps, self.secs = 250, .05
        record = self.absorb(combined, mean_drive, smell_hz, pairs.mean(axis=0), t0)
        record["wave_rms"] = rms(sound_hz)
        record["ear_clipped"] = self.ear.clipped.any(axis=0).tolist()
        return record

    def reset(self, seed=None):
        self.brain_state, self.windows = None, 0
        if seed is not None:
            self.seed = int(seed)
        self.answer = dict(vpodn_hz=0., pc1_hz=0., ovidn_hz=0., spsp_hz=0.)

    @property
    def state_carried(self):
        return self.brain_state is not None

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
        idx = self.groups["SpsP"]
        if any(np.intersect1d(k, idx).size for k in drive):
            raise ValueError("SpsP drive overlaps another input")
        drive[tuple(idx)] = np.full(idx.size, SPSN_HZ if self.state == "virgin" else 0., np.float32)
        return drive

    @staticmethod
    def sound_pair(sound):
        """A scalar retains the same nominal rate at both JO groups."""
        a, b = (sound, sound) if np.isscalar(sound) else sound
        return float(max(0., a)), float(max(0., b))

    def absorb(self, r, drive, smell_hz, sound_hz, t0):
        """Record mean answer rates in the same window telemetry as FlyBody."""
        carried = self.brain_state is not None
        self.brain_state = r["_state"]
        self.windows += 1
        per_cell = np.asarray(r["all"], dtype=np.float64)
        rates, counts = {}, {}
        for k, pos in self.rec_pos.items():
            v = per_cell[pos]
            rates[k] = float(v.mean()) if v.size else 0.0
            counts[k] = int(np.rint(v.sum() * self.secs))
        self.answer = dict(vpodn_hz=rates["vpoDN"], pc1_hz=rates["pC1"],
                           ovidn_hz=rates["oviDN"], spsp_hz=rates["SpsP"])
        motor = {k: rates[k] for k in MOTOR_NAMES}
        turn, speed, parts = PlumeFly.motor_from_rates(motor)
        vals = [v for k, v in drive.items()
                if k not in (tuple(self.sound_idx), tuple(self.groups["SpsP"]))]
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
                "state": self.state, "spsp_drive_hz": SPSN_HZ if self.state == "virgin" else 0.,
                "song_group": None, "song_cells": 0, "smell_cells": 0,
                "sound_cells": int(self.sound_idx.size),
                "recorded_cells": int(self.rec_idx.size),
                "groups": {k: int(v.size) for k, v in self.groups.items()},
                "motor_cells": {k: int(v.size) for k, v in self.motor.items()},
                "side_scales": self.side_scales,
                "ear": self.ear.describe(),
                "drive_note": "smell ignored; JO sound equalised per soma side"}
