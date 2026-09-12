"""
The fly's eye and its descending-neuron readout.

Screen pixels enter through the 892 retinotopic hex columns the dataset
assigns to optic-lobe neurons, into L1 and L2 - the lamina cells directly
postsynaptic to photoreceptors R1-R6. The cursor comes back out of the
descending neurons a fly actually walks with.
"""
import numpy as np
from flysim import FlyBrain


class FlyEye:
    """Retinotopic sampling of the screen onto the fly's 892 hex columns."""

    def __init__(self, fb, annotations_path="data/body-annotations.feather"):
        import pandas as pd
        a = pd.read_feather(annotations_path).drop_duplicates("bodyId").set_index("bodyId")
        h1 = a["assignedOlHex1"].reindex(fb.bodies).to_numpy()
        h2 = a["assignedOlHex2"].reindex(fb.bodies).to_numpy()
        has = ~(np.isnan(h1.astype(float)) | np.isnan(h2.astype(float)))

        self.fb = fb
        types = fb.types
        # L1 = ON pathway input, L2 = OFF pathway input (both postsynaptic to R1-R6)
        self.on_mask = has & (types == "L1")
        self.off_mask = has & (types == "L2")

        # hex axial coordinates -> unit square
        def uv(mask):
            x = h1[mask].astype(np.float32) + 0.5 * h2[mask].astype(np.float32)
            y = h2[mask].astype(np.float32) * (np.sqrt(3) / 2)
            return x, y

        xs, ys = uv(self.on_mask | self.off_mask)
        self.x0, self.x1 = xs.min(), xs.max()
        self.y0, self.y1 = ys.min(), ys.max()

        self.on_idx = np.flatnonzero(self.on_mask)
        self.off_idx = np.flatnonzero(self.off_mask)
        # raw column indices, for unrolling the eye onto a rectangular display
        self.on_h1 = h1[self.on_mask].astype(int)
        self.on_h2 = h2[self.on_mask].astype(int)
        self.on_uv = self._to_uv(h1[self.on_mask], h2[self.on_mask])
        self.off_uv = self._to_uv(h1[self.off_mask], h2[self.off_mask])

    def _to_uv(self, h1, h2):
        x = h1.astype(np.float32) + 0.5 * h2.astype(np.float32)
        y = h2.astype(np.float32) * (np.sqrt(3) / 2)
        u = (x - self.x0) / (self.x1 - self.x0 + 1e-9)
        v = (y - self.y0) / (self.y1 - self.y0 + 1e-9)
        return np.clip(u, 0, 1), np.clip(v, 0, 1)

    def look(self, img, cx, cy, fov_w=300, fov_h=210, max_hz=180.0):
        """
        Sample the page around the cursor and return per-neuron drive rates.
        The fly's gaze follows its own cursor, so the view is egocentric.

        The window is deliberately narrow. With a wide view the whole form sits
        in frame at every cursor position, the retinal image barely changes as
        the fly moves, and there is no positional signal to learn from. A tight
        window makes what the fly sees depend on where it is.
        """
        H, W = img.shape

        def sample(uv):
            u, v = uv
            px = np.clip((cx - fov_w / 2 + u * fov_w).astype(int), 0, W - 1)
            py = np.clip((cy - fov_h / 2 + v * fov_h).astype(int), 0, H - 1)
            return img[py, px]

        lum_on = sample(self.on_uv)
        lum_off = sample(self.off_uv)
        # L1 carries light increments, L2 light decrements
        return {
            tuple(self.on_idx): np.clip(lum_on, 0, 1) * max_hz,
            tuple(self.off_idx): np.clip(1.0 - lum_off, 0, 1) * max_hz * 0.6,
        }


class FlyPilot:
    """
    Closed loop: the fly looks at the form, its descending neurons fire, the
    cursor moves, and it looks again.

    One control step is a short burst of brain time (20 ms by default). That is
    enough for the retina to drive the descending neurons through the two to
    three synapses between them, and short enough that a whole trajectory is
    affordable to simulate.
    """

    def __init__(self, fb, eye=None, sim_steps=100, click_hz=330.0):
        self.fb = fb
        self.eye = eye or FlyEye(fb)
        self.sim_steps = sim_steps
        self.click_hz = click_hz

        import pandas as pd
        a = pd.read_feather("data/body-annotations.feather")
        a = a.drop_duplicates("bodyId").set_index("bodyId")
        side = a["somaSide"].reindex(fb.bodies).fillna("").to_numpy().astype(str)

        def dn(t, s=None):
            sel = fb.where(type_re=rf"^{t}$")
            if s:
                sel = np.array([i for i in sel if side[i] == s], dtype=np.int64)
            return sel

        self.motor = {
            "steer_L": dn("DNa02", "L"),   # a fly turns by DNa02 asymmetry
            "steer_R": dn("DNa02", "R"),
            "fwd_L": dn("DNa01", "L"),
            "fwd_R": dn("DNa01", "R"),
            "back": dn("MDN"),             # Moonwalker: backward walking
            "stop": dn("DNp09"),
            "click": dn("MN9"),            # proboscis extension = commit
        }

    def step(self, img, cx, cy, gains=None, seed=0, detail=False,
             extra_drive=None, extra_record=None):
        """
        One control step.

        detail=True adds a fifth return value: what the rest of the brain was
        doing while the descending neurons decided. Existing callers unpack
        four and are unaffected.

        extra_drive adds input to the same run, as a FlyBrain drive dict (the
        backroom adds a coin's smell on the olfactory receptor neurons).
        extra_record adds populations to record in that run; their rates come
        back in info["extra"]. One run then gives the cursor, the commit, the
        mushroom body output and the Kenyon cells that fired, together.
        """
        drive = self.eye.look(img, cx, cy)
        if extra_drive:
            drive = dict(drive)
            for k, v in extra_drive.items():
                if k in drive:
                    raise ValueError("extra_drive overlaps the eye's own input")
                drive[k] = v
        record = self.motor
        if extra_record:
            clash = set(extra_record) & set(self.motor)
            if clash:
                raise ValueError(f"extra_record reuses motor names: {sorted(clash)}")
            record = dict(self.motor, **extra_record)
        r = self.fb.run(drive, steps=self.sim_steps, gains=gains,
                        record=record, seed=seed)
        hz = {k: float(r[k].mean()) for k in self.motor}

        # steering is the left/right difference; forward drive is the sum
        turn = (hz["steer_R"] - hz["steer_L"]) / 450.0
        fwd = (hz["fwd_L"] + hz["fwd_R"]) / 2.0 / 450.0
        back = hz["back"] / 450.0
        stop = hz["stop"] / 450.0

        speed = np.clip(fwd - back, -1, 1) * (1.0 - np.clip(stop, 0, 1))
        dx = np.clip(turn, -1, 1) * 90.0
        dy = -speed * 90.0            # forward walking moves up the page

        # The click is the fly stopping. MN9 was the obvious choice - proboscis
        # extension is the fly's commit action - but MN9 is a taste motor
        # neuron and measures a flat 0 Hz under visual drive, which made the
        # reward unreachable. DNp09, the stopping descending neuron, is
        # genuinely visually driven (167-417 Hz), so arriving and halting on a
        # target is the click.
        click = hz["stop"] >= self.click_hz and speed < 0.25
        if not detail:
            return dx, dy, click, hz

        fired = r.get("_fired")
        eye_idx = np.flatnonzero(self.eye.on_mask | self.eye.off_mask)
        motor_idx = np.concatenate([v for v in self.motor.values()])             if self.motor else np.array([], dtype=np.int64)
        info = {
            "firing": int(len(fired)) if fired is not None else 0,
            "spikes_per_sec": float(r.get("_spikes_per_sec", 0.0)),
            "mean_mv": float(r.get("_mean_mv", 0.0)),
            "visual": int(np.isin(eye_idx, fired).sum()) if fired is not None else 0,
            "motor": int(np.isin(motor_idx, fired).sum()) if fired is not None else 0,
            "fired": fired,
            "extra": {k: r[k] for k in (extra_record or {})},
            "turn_l": float(max(0.0, -turn)), "turn_r": float(max(0.0, turn)),
            "forward": float(max(0.0, speed)), "reverse": float(back),
            "click": float(stop),
        }

        # what the retina itself is doing: the drive rates that went into L1
        # and L2, at the hex columns they were sampled from. This is the
        # fly's actual visual field, not a picture of the screen.
        try:
            on_rate = np.asarray(list(drive.values())[0], dtype=np.float32)
            off_rate = np.asarray(list(drive.values())[1], dtype=np.float32)
            step_n = max(1, len(self.eye.on_uv[0]) // 190)
            u, v = self.eye.on_uv
            cols = [[round(float(u[i]), 3), round(float(v[i]), 3),
                     round(float(on_rate[i]) / 180.0, 3)]
                    for i in range(0, len(u), step_n)]
            info["vision"] = {
                "on_hz": round(float(on_rate.mean()), 1),
                "off_hz": round(float(off_rate.mean()), 1),
                "columns": len(u) + len(self.eye.off_uv[0]),
                "cols": cols,
            }
        except Exception:
            info["vision"] = None
        return dx, dy, click, hz, info

