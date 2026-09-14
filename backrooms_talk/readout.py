"""
The readout: what one fly's turn shows the model, built only from the
room's records since that fly's last turn.

Per world step, update(record) adds each fly's numbers to its running window
(and feeds the one backrooms.Captioner that watches all four flies, whose
event lines are kept on the window but not shown; EVENT_LINES = 0).
take(name) closes that fly's window and returns:

  basis   up to BASIS_MAX short strings, in this order:
            1-2. the TOP_GROUPS named groups whose window rate rose most above
                 their baseline: "<name> neurons: <rate> Hz now, baseline
                 <rate> Hz" (the dictionary's name; "neurons" is not added to
                 a name that already ends in it);
            3.   the distance walked in the window: "walked 4.1 mm in the
                 last 8.5 s";
            4-6. every other fly's distance and bearing now, one string per
                 fly, nearest first: "B is 4.0 mm away, 35 deg to the left".
          Every template here was checked against the safety filter over the
          whole range of its numbers (the tests do it again): "D 13.6 mm" read
          as a filtered word once digits are folded, so a fly's distance is
          written "D is 13.6 mm away".
  numbers the numeric summary the turn posts as meta (see TURN_SCHEMA.md);
          the song level and the drives are there, not in the basis.

Why this shape (CHOSEN, from a local probe of LFM2.5-1.2B-Instruct on the
first dry runs' readouts, nothing posted): with the dictionary's roles and
citations, the song line and the captioner's lines in the readout, and six
recent lines in the prompt, the model mostly wrote lines about "recorded
activity" and "patterns" with no number from the readout. Short strings,
one fact each, with the group lines first, made it quote the readout's own
numbers in nearly every sample. The roles and citations stay in the
dictionary and on the page; the model is not shown them.

Nothing here is a model; every number is from the simulation or a stated
choice. Baseline (CHOSEN): a group's mean rate over every world step of this
fly from the start of the run to the start of the window (the warm-up
included). Ranking (CHOSEN): (window - baseline) / max(baseline,
backrooms.floor_hz(cells)); the floor keeps a 2-cell group's single spike
from outranking a population's real change.
"""
import math

import numpy as np

import backrooms as br
import backrooms_dictionary as bd
import backrooms_world as bw

TOP_GROUPS = 2          # CHOSEN: named groups per readout
EVENT_LINES = 0         # CHOSEN: captioner lines per readout (0: none shown; see the module doc)
BASIS_MAX = 6           # CHOSEN (the task's cap): readout strings per turn
AHEAD_DEG = 0.5         # CHOSEN: |bearing| below this is written "straight ahead"


def bearing_words(b):
    """Bearing in degrees (+ = left, the room's convention) as words."""
    b = float(b)
    if abs(b) < AHEAD_DEG:
        return "straight ahead"
    return f"{abs(b):.0f} deg to the {'left' if b > 0 else 'right'}"


class _Window:
    """One fly's running sums: the open window, and the baseline behind it."""

    def __init__(self, keys):
        k = len(keys)
        self.win_rates = np.zeros(k)
        self.base_rates = np.zeros(k)
        self.win_n = 0
        self.base_n = 0
        self.song = self.smell = self.sound = self.active = self.brain_hz = 0.0
        self.brain_hz_n = 0
        self.walked = 0.0
        self.events = []
        self.t_start = None

    def close(self):
        """The window joins the baseline and a new one opens."""
        self.base_rates += self.win_rates
        self.base_n += self.win_n
        self.win_rates[:] = 0.0
        self.win_n = 0
        self.song = self.smell = self.sound = self.active = self.brain_hz = 0.0
        self.brain_hz_n = 0
        self.walked = 0.0
        self.events = []
        self.t_start = None


class Readout:
    def __init__(self, sizes, names, dictionary=None, dt=bw.WORLD_DT_S):
        self.dictionary = dictionary or bd.DICTIONARY
        self.names = tuple(names)
        self.sizes = {k: int(v) for k, v in sizes.items()}
        self.captioner = br.Captioner(self.sizes, dictionary=self.dictionary, flies=self.names, dt=dt)
        self.keys = list(self.captioner.keys)          # captioned groups: not motor, not the song
        self.song_key = self.captioner.song_key
        self.dt = float(dt)
        self.win = {n: _Window(self.keys) for n in self.names}
        self.last = None
        self.steps = 0

    def update(self, rec):
        """One world step's record (TalkRoom.step's shape). Returns the captioner's new lines."""
        t = float(rec["t"])
        flies = rec["flies"]
        lines = self.captioner.update(t, flies)
        for ln in lines:
            if ln["fly"] in self.win:
                self.win[ln["fly"]].events.append(ln)
        for n in self.names:
            f, w = flies[n], self.win[n]
            if w.t_start is None:
                w.t_start = t - self.dt
            w.win_rates += np.array([float(f["rates"].get(k, 0.0)) for k in self.keys])
            w.win_n += 1
            w.song += float(f["song"])
            w.smell += float(f["drive"].get(bw.SMELL_KEY, 0.0))
            w.sound += float(f["drive"].get(bw.SOUND_KEYS[0], 0.0))
            w.active += float(f.get("active_fraction", 0.0))
            if f.get("total_hz") is not None:
                w.brain_hz += float(f["total_hz"])
                w.brain_hz_n += 1
            w.walked += abs(float(f.get("speed", 0.0))) * self.dt
        self.last = rec
        self.steps += 1
        return lines

    def settle(self):
        """Every open window joins its baseline (after the warm-up): no fly's first turn reads the warm-up as its window."""
        for w in self.win.values():
            w.close()

    # ---- one turn ------------------------------------------------------------

    def top_groups(self, name):
        """[(key, window_hz, baseline_hz or None, score)] best first, TOP_GROUPS long."""
        w = self.win[name]
        if w.win_n == 0:
            return []
        win = w.win_rates / w.win_n
        base = w.base_rates / w.base_n if w.base_n else None
        rows = []
        for i, k in enumerate(self.keys):
            b = None if base is None else float(base[i])
            if win[i] <= 0.0 and not b:
                continue                       # silent then and now: nothing to rank
            floor = br.floor_hz(self.sizes[k])
            score = (float(win[i]) - (b or 0.0)) / max(b or 0.0, floor)
            rows.append((k, float(win[i]), b, score))
        rows.sort(key=lambda r: (-r[3], self.keys.index(r[0])))
        return rows[:TOP_GROUPS]

    def group_line(self, key, win_hz, base_hz):
        name = str(self.dictionary[key]["name"])
        label = name if name.endswith("neurons") else f"{name} neurons"
        base = f"baseline {base_hz:.0f} Hz" if base_hz is not None else "no baseline yet"
        return f"{label}: {win_hz:.0f} Hz now, {base}"

    @staticmethod
    def walked_line(walked_mm, secs):
        return f"walked {walked_mm:.1f} mm in the last {secs:.1f} s"

    @staticmethod
    def other_line(other, geo):
        """geo: {"distance", "bearing"} of another fly, as the room records it."""
        return f"{other} is {geo['distance']:.1f} mm away, {bearing_words(geo['bearing'])}"

    def song_line(self, song_hz, sound_hz, smell_hz):
        """Not shown to the model (the numbers go to meta); kept for the recorder and the tests."""
        song_name = self.dictionary[self.song_key]["name"] if self.song_key in self.dictionary else self.song_key
        n_song = self.sizes.get(self.song_key, 0)
        return (f"song level, summed over {n_song} {song_name}: {song_hz:.0f} Hz; "
                f"sound drive to JO-A and JO-B hearing neurons: {sound_hz:.0f} Hz; "
                f"cVA smell drive to ORN_DA1: {smell_hz:.0f} Hz")

    def take(self, name):
        """Close `name`'s window; return (basis strings, numbers)."""
        w = self.win[name]
        if w.win_n == 0 or self.last is None:
            raise ValueError(f"no world step in {name}'s window: step the world before a turn")
        f = self.last["flies"][name]
        secs = w.win_n * self.dt
        basis = []

        top = self.top_groups(name)
        for k, win_hz, base_hz, _ in top:
            basis.append(self.group_line(k, win_hz, base_hz))

        basis.append(self.walked_line(w.walked, secs))
        others = sorted(f["others"].items(), key=lambda kv: kv[1]["distance"])
        for o, g in others:
            basis.append(self.other_line(o, g))

        if EVENT_LINES > 0:
            for ln in w.events[-EVENT_LINES:]:
                basis.append(ln["text"])
        basis = basis[:BASIS_MAX]

        numbers = {
            "active_fraction": w.active / w.win_n,
            "brain_rate_hz": (w.brain_hz / w.brain_hz_n) if w.brain_hz_n else 0.0,
            "song_hz": w.song / w.win_n,
            "sound_drive_hz": w.sound / w.win_n,
            "smell_drive_hz": w.smell / w.win_n,
            "top_group_rate_hz": top[0][1] if top else 0.0,
            "top_group_baseline_hz": (top[0][2] or 0.0) if top else 0.0,
            "nearest_fly_mm": others[0][1]["distance"] if others else 0.0,
            "world_s": float(self.last["t"]),
            "window_steps": int(w.win_n),
        }
        for k, v in numbers.items():
            if isinstance(v, float) and not math.isfinite(v):
                raise ValueError(f"readout number {k} is not finite")
        w.close()
        return basis, numbers
