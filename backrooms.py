"""
backrooms.py - two flies in a 20 x 20 mm room, on one loaded connectome.

This file has two sections. SECTION 1, everything above the marker near the
end, is THE CAPTIONER: the only code that turns numbers into words. SECTION 2,
below the marker, is the world loop, the FastAPI server and the page feed; it
is added by the server agent and nothing in section 1 depends on it.

HOW A LINE IS MADE (this is what the page banner must say, verbatim or
paraphrased; the string HOW_LINES_ARE_MADE below is the canonical wording)

  There is no person and no language model anywhere in this program. Every
  line of the transcript is one of the fixed templates in TEMPLATES, filled
  with a fly's name ("A" or "B"), a neuron group's name and role as written in
  backrooms_dictionary.py (each with the paper it comes from), and numbers
  measured from the simulation. A line is written only when something
  crosses a threshold; nothing is written on a flat trace. The thresholds,
  the room, the channels and the phrasing are chosen, and are listed here as
  constants so a reader can check every line against the recorded numbers.

MEASURED (the inputs, recorded by the world loop once per 50 ms world step)
  * per fly, per dictionary group: the group's mean spike rate over the 60
    LIF steps (12 ms) the brain ran for that world step, in spikes per neuron
    per second, as flysim.FlyBrain.run reports it; so a group of n cells can
    only show multiples of 1000 / (12 * n) Hz, and a 2-cell descending neuron
    reads 41.7 Hz when one of its cells spikes once in the window;
  * per fly: the input rates the room delivered to ORN_DA1, JO-A and JO-B;
  * per fly: the song, defined as the SUMMED rate (Hz, over cells) of the
    pulse-song wing motor neurons (dictionary key song_pulse_mn: ps1, i1,
    iii1, b3; 8 cells); if that group were absent the sum over all named
    wing motor neurons would be used and the line would say so by name;
  * per fly: position (mm), heading (degrees, 0 along +x, counter-clockwise
    positive, measured the same way as atan2(dy, dx)).
  Distance, bearing, speed and the closing speed are computed here from
  those positions, so the captioner is a pure function of the record.

CHOSEN (every constant below is a choice, not a measurement)
  * A group "fires" when its rate has been at or above max(floor, 2 x its
    baseline) for CONFIRM_STEPS consecutive world steps, and "falls" when it
    has been below max(floor / 2, 1.5 x that baseline) for as many steps. The
    baseline is the mean of the group's last 10 s of samples taken while it
    was not firing (samples under a confirmation that fails count as quiet,
    the ones that confirm a burst do not; the baseline is frozen while the
    group is on, so a group that rises and stays up is reported once, not
    repeatedly). The floor is one spike in
    the group per 12 ms window, or 5 Hz, whichever is larger: it is derived
    from the group's measured cell count, so small groups need a whole spike
    and large groups need a change in the population rate. The confirmation
    steps stop a single spike from writing two lines.
  * The song "starts" when the summed pulse-motor rate has been at or above
    2 spikes per window for CONFIRM_STEPS steps and "ends" when it has been
    below 1 spike per window for as many steps.
  * A fly "faces" the other when the bearing to it is within 30 degrees of
    its heading and "no longer faces" it beyond 40 degrees (the 10-degree band
    is the hysteresis). The line says "turns toward" / "turns away from" only
    when the fly's own heading change in that step moved the bearing the same
    way; when the crossing came from the other fly's motion it says "faces" /
    "no longer faces", because the fly did not turn.
  * "approaches" is written when the distance drops below 3 mm, once, and
    again only after it has been at least 4 mm; "leaves" when it exceeds
    10 mm, once, and again only after it has been at most 9 mm. Both are
    pair events; the line carries the name of the fly whose own velocity
    closed (or opened) the distance more in that step.
  * "backs up" is written when a fly's velocity along its heading falls below
    -1 mm/s, once, re-armed at -0.5 mm/s; "stops" when its speed falls below
    0.5 mm/s after having been at least 2 mm/s. These are kinematic events
    of the room, so a fly pinned on a wall by its own drive "stops"; the
    line names MDN or DNp09 with its measured rate only when that group
    fired in the step (rate above zero), and is the plain line otherwise,
    so a silent neuron is never printed beside a stop as if it caused it.
    Under the roamer's mapping a negative speed needs MDN above DNa01, so
    every "backs up" line carries MDN; a "stops" line usually has DNp09
    silent (the fly stopped at a wall, or DNa01 fell), and DNp09 is in any
    case a forward-walking neuron that the roamer reads as stop.
  * The motor readout groups (DNa02, DNa01, MDN, DNp09) and the song groups
    are recorded but never written as "fires" lines: the first would restate
    the walking command, the second the song.
  * Groups the dictionary marks uncertain carry "; uncertain match" in the
    line, as rule 7 requires.

The captioner keeps state (baselines, arm flags, sequence number) so the
server can feed it one world step at a time; caption_record() runs a fresh
one over recorded arrays and is what the tests use. Both are deterministic:
there is no randomness here and the emitted order is fixed (fly A then B;
within a fly, groups in dictionary order, then the song; then the geometric
events).

  py -m pytest -q test_backrooms_captions.py test_backrooms_honesty.py
"""
import json
import math
import re
from collections import deque
from pathlib import Path

import backrooms_dictionary as bd

ROOT = Path(__file__).parent
BUILD = ROOT / "build"

# =============================================================================
# SECTION 1: THE CAPTIONER
# =============================================================================

FLIES = ("A", "B")

# ---- the clock. MEASURED: flysim.Params.dt is 0.2 ms; the DESIGN runs 60 of
#      them per 50 ms world step. Pinned here so every threshold below can be
#      read in seconds and spikes without opening the simulator. ---------------
WORLD_DT_S = 0.05
BRAIN_STEPS = 60
LIF_DT_MS = 0.2
WINDOW_MS = BRAIN_STEPS * LIF_DT_MS          # 12 ms of brain time per world step
SPIKE_HZ = 1000.0 / WINDOW_MS                # one spike in one window = 83.33 Hz


def rate_quantum_hz(n_cells):
    """The smallest non-zero mean rate a group of n cells can show in one window."""
    return SPIKE_HZ / float(n_cells)


# ---- CHOSEN detection constants ------------------------------------------------
BASELINE_S = 10.0             # running baseline window, world seconds
ONSET_RATIO = 2.0             # fires when rate >= max(floor, ONSET_RATIO x baseline)
OFFSET_RATIO = 1.5            # falls when rate <  max(floor x OFFSET_FLOOR_FRACTION, OFFSET_RATIO x baseline)
FLOOR_MIN_HZ = 5.0            # absolute floor for large groups
FLOOR_SPIKES = 1.0            # ... or this many spikes in the group per window, if larger
OFFSET_FLOOR_FRACTION = 0.5
CONFIRM_STEPS = 3             # consecutive world steps a condition must hold (150 ms)

SONG_KEY = "song_pulse_mn"    # the DESIGN's song group
SONG_FALLBACK_KEY = "wing_mn_all"
SONG_ON_SPIKES = 2.0          # summed spikes per window, start
SONG_OFF_SPIKES = 1.0         # summed spikes per window, end (below this)

FACING_DEG = 30.0             # |bearing| at or below this: faces the other
FACING_BAND_DEG = 10.0        # ... and above FACING_DEG + band: no longer faces
NEAR_MM = 3.0
NEAR_REARM_MM = 4.0
FAR_MM = 10.0
FAR_REARM_MM = 9.0
BACK_MM_S = 1.0               # velocity along heading below -BACK_MM_S: backs up
BACK_REARM_MM_S = 0.5         # re-armed once along-heading velocity >= -BACK_REARM_MM_S
STOP_MM_S = 0.5               # speed below this: stops
STOP_REARM_MM_S = 2.0         # re-armed once speed >= this

KINDS = ("onset", "offset", "song_start", "song_end", "toward", "away",
         "approach", "leave", "back", "stop")

# Every word the transcript can contain that is not a dictionary name, a
# dictionary role, a citation key, a fly name or a number is in one of these
# strings. The tests build an allow-list from them and refuse any other token.
TEMPLATES = {
    "onset": "{fly}  {name} fires {value:.0f} Hz ({role}, {cite}{unc})",
    # sensory lines state the delivered drive and the measured rate; they do
    # not say what the fly heard or smelled, because that was not measured
    "onset_smell": "{fly}  {name} fires {value:.0f} Hz under {drive:.0f} Hz of cVA drive "
                   "from {other} ({role}, {cite}{unc})",
    "onset_sound": "{fly}  {name} fires {value:.0f} Hz under {drive:.0f} Hz of sound drive "
                   "from {other}'s song ({role}, {cite}{unc})",
    "offset": "{fly}  {name} falls to {value:.0f} Hz ({role}, {cite}{unc})",
    # the song is a summed motor rate, not a measured pulse pattern, so the
    # line says 'song starts', never 'sings'
    "song_start": "{fly}  song starts: {name} {value:.0f} Hz summed over {n} cells ({role}, {cite}{unc})",
    "song_end": "{fly}  song ends: {name} {value:.0f} Hz summed over {n} cells",
    "toward": "{fly}  turns toward {other}: bearing {value:.0f} deg at {dist:.1f} mm",
    "faces": "{fly}  faces {other}: bearing {value:.0f} deg at {dist:.1f} mm",
    "away": "{fly}  turns away from {other}: bearing {value:.0f} deg at {dist:.1f} mm",
    "unfaces": "{fly}  no longer faces {other}: bearing {value:.0f} deg at {dist:.1f} mm",
    "approach": "{fly}  approaches {other}: {value:.1f} mm",
    "leave": "{fly}  leaves {other}: {value:.1f} mm",
    "back": "{fly}  backs up at {value:.1f} mm/s ({name} {rate:.0f} Hz, {role}, {cite})",
    "back_plain": "{fly}  backs up at {value:.1f} mm/s",
    "stop": "{fly}  stops ({name} {rate:.0f} Hz, {role}, {cite})",
    "stop_plain": "{fly}  stops",
}
UNCERTAIN_SUFFIX = "; uncertain match"

# The sensory entry points the room drives, and the template used when the
# recorded input drive to them was above zero in the step the line is written.
# With zero drive the plain "fires" template is used: the group fired on its
# own and the line must not say it heard or smelled anything.
SENSORY_TEMPLATES = {"ORN_DA1": "onset_smell", "JO_A": "onset_sound", "JO_B": "onset_sound"}
BACK_GROUP = "MDN"
STOP_GROUP = "DNp09"

HOW_LINES_ARE_MADE = (
    "Every line here is a fixed template filled with measured numbers. No person "
    "and no language model writes anything. A line names a fly (A or B), a neuron "
    "group by the name and role given in the dictionary (with the paper it comes "
    "from), and the number measured from the simulation at that step: a spike "
    "rate in Hz, a distance in mm, a bearing in degrees or a speed in mm/s. A line "
    "is written only when a rate has stayed at or above twice its 10 s baseline "
    "and above the group's floor for 3 consecutive steps (and once more when it "
    "falls back), when the summed pulse-song motor rate starts or ends, when a "
    "fly's bearing to the other crosses 30 degrees, when the distance drops below "
    "3 mm or exceeds 10 mm, or when a fly backs up or stops. Nothing is written "
    "on a flat trace. 'Backs up' and 'stops' are kinematic events of the room, "
    "read from positions, so a fly pinned on a wall by its own drive 'stops'; the "
    "line names MDN or DNp09 with its rate only when that group fired in the "
    "step, and DNp09 is a forward-walking neuron that the roamer's mapping reads "
    "as stop. A sensory line ('fires N Hz under M Hz of drive') states the rate "
    "the room delivered to those cells and the rate they fired at; it does not "
    "say what the fly heard or smelled. The song is the summed rate of the "
    "pulse-song wing motor neurons, not a measured pulse pattern: in this "
    "simulator those cells fire on every window, so the song is on from the "
    "first step, no start or end line is expected, and its level is shown as a "
    "number beside each fly; what the other fly's hearing cells receive is that "
    "level scaled by distance. The room, the channels, the thresholds and the "
    "phrasing are chosen; the neurons, their wiring and their spikes are the "
    "measurement. Each brain is run for 12 ms of neural time per 50 ms of arena "
    "time, so on the brains' own clock the room, the walking and the one-step "
    "sound delay move 4.2 times faster than life; separately, one 50 ms step "
    "takes several times longer than 50 ms of wall time, so the run is slower "
    "than life by the measured ratio shown in the status line. Groups marked "
    "'uncertain match' are named on evidence the dictionary calls uncertain."
)


# ---- small pure helpers --------------------------------------------------------

def floor_hz(n_cells):
    """
    The absolute floor for a group of n cells: FLOOR_SPIKES spikes in the group
    per window, or FLOOR_MIN_HZ, whichever is larger. Derived from the
    dictionary's measured count so that a 2-cell neuron needs a real spike
    and a 200-cell population needs a change in its rate. CHOSEN.
    """
    n = max(int(n_cells), 1)
    return max(FLOOR_MIN_HZ, FLOOR_SPIKES * SPIKE_HZ / n)


_YEAR = re.compile(r"(?<=[ (])((?:19|20)\d\d)(?=[ ;),.])")
_SURNAME = re.compile(r"\s*((?:von |van |de )?[A-Z][A-Za-z'-]+)")


def cite_key(citation):
    """
    'Surname YEAR' of the first paper in a dictionary citation string, e.g.
    'von Philipsborn 2011' from 'von Philipsborn et al. 2011 Neuron 69:509 (...)'.
    The transcript names one paper per line; the full citation, with figures,
    stays in the dictionary the page links to. Empty when no year is found
    (the absent entries), which never reaches a line because absent groups are
    never captioned.
    """
    m = _YEAR.search(citation)
    if not m:
        return ""
    prefix = citation[:m.start()]
    # the author list is what follows the last ': ' or '(' before the year
    seg = re.split(r":\s|\(", prefix)[-1]
    s = _SURNAME.match(seg)
    return f"{s.group(1)} {m.group(1)}" if s else ""


def wrap_deg(a):
    """Angle into [-180, 180)."""
    return (float(a) + 180.0) % 360.0 - 180.0


def geometry(x, y, heading, ox, oy):
    """(distance mm, bearing deg) from a fly at (x, y) facing `heading` to (ox, oy)."""
    dx, dy = float(ox) - float(x), float(oy) - float(y)
    dist = math.hypot(dx, dy)
    bearing = wrap_deg(math.degrees(math.atan2(dy, dx)) - float(heading))
    return dist, bearing


def sizes_from_json(path=bd.JSON_PATH):
    """{key: measured cell count} for the groups the dictionary JSON says resolve."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: int(e["counts"]["n"]) for k, e in d["groups"].items()
            if e["present"] and e["counts"]["n"] > 0}


def sizes_from_groups(groups):
    """{key: n} from backrooms_dictionary.groups(fb) / present_groups(fb)."""
    return {k: int(len(v)) for k, v in groups.items() if len(v)}


def song_family(dictionary, song_key):
    """
    The song group, its parts, and the groups that contain it: never 'fires'
    lines, because the song line already says it. A sibling inside the same
    container (hg1, the sine-song motor neuron, inside all wing motor
    neurons) is not the song and keeps its own line.
    """
    fam = {song_key}
    fam.update(dictionary[song_key]["contains"])
    for k, e in dictionary.items():
        if song_key in e["contains"]:
            fam.add(k)
    return fam


def captioned_keys(dictionary, sizes, song_key=SONG_KEY):
    """
    The groups that may produce 'fires' / 'falls' lines, in dictionary order:
    present, resolved to at least one cell, not the motor readout, not the song.
    """
    skip = song_family(dictionary, song_key)
    return [k for k, e in dictionary.items()
            if e["present"] and sizes.get(k, 0) > 0 and not e["motor"] and k not in skip]


_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+/-]*")


def tokens(text):
    """The words of a line, as the vocabulary test counts them."""
    return _TOKEN.findall(text)


def vocabulary(dictionary=None, flies=FLIES):
    """
    Every token a line may contain: the templates with their placeholders
    removed, the names, roles and citation keys of present groups, the fly
    names and the uncertain suffix. Numbers are allowed separately.
    """
    dictionary = dictionary or bd.DICTIONARY
    voc = set()
    for t in TEMPLATES.values():
        voc.update(tokens(re.sub(r"\{[^}]*\}", " ", t)))
    voc.update(tokens(UNCERTAIN_SUFFIX))
    for e in dictionary.values():
        if e["present"]:
            voc.update(tokens(e["name"]))
            voc.update(tokens(e["role"]))
            voc.update(tokens(cite_key(e["citation"])))
    voc.update(flies)
    return voc


# ---- detectors: each is a small state machine over one scalar trace -------------

class _Mean:
    """Running mean of the last `maxlen` values pushed, O(1) per push."""

    def __init__(self, maxlen):
        self.q = deque(maxlen=maxlen)
        self.sum = 0.0

    def push(self, v):
        if len(self.q) == self.q.maxlen:
            self.sum -= self.q[0]
        self.q.append(v)
        self.sum += v

    def __len__(self):
        return len(self.q)

    @property
    def mean(self):
        return self.sum / len(self.q) if self.q else 0.0


class RateDetector:
    """
    Onset / offset with hysteresis and confirmation over one group's rate.
    The baseline is the mean of the last `baseline_len` samples taken while
    the group was off. Samples under a confirmation are held back: if the
    confirmation fails they were quiet after all and join the baseline; if
    it succeeds they were the burst and never do. So a burst is measured
    against the quiet that preceded it and cannot raise its own threshold
    while it is being confirmed. The baseline is frozen at onset and used
    for the offset test; the samples that confirm the offset join it then.
    """

    def __init__(self, floor, baseline_len, confirm=CONFIRM_STEPS):
        self.floor = float(floor)
        self.hist = _Mean(baseline_len)
        self.confirm = int(confirm)
        self.on = False
        self.pending = []
        self.base = 0.0

    def _settle(self, rate):
        """The held samples and this one were quiet: into the baseline, in order."""
        for p in self.pending:
            self.hist.push(p)
        self.pending = []
        self.hist.push(rate)

    def update(self, rate):
        """Returns 'onset', 'offset' or None for this sample."""
        rate = float(rate)
        if not self.on:
            if len(self.hist) == 0:
                # first sample: it is its own history, nothing can be an event
                self.hist.push(rate)
                return None
            base = self.hist.mean
            if rate >= max(self.floor, ONSET_RATIO * base):
                self.pending.append(rate)
                if len(self.pending) >= self.confirm:
                    self.on, self.base, self.pending = True, base, []
                    return "onset"
                return None
            self._settle(rate)
            return None
        if rate < max(OFFSET_FLOOR_FRACTION * self.floor, OFFSET_RATIO * self.base):
            self.pending.append(rate)
            if len(self.pending) >= self.confirm:
                self.on = False
                self.pending.pop()          # this sample; _settle pushes it last
                self._settle(rate)
                return "offset"
            return None
        self.pending = []
        return None


class LevelDetector:
    """
    Two-state hysteresis on a scalar: enters 'on' at or above `on_at`, leaves
    it below `off_below`, each held for `confirm` samples. Used for the song.
    """

    def __init__(self, on_at, off_below, confirm=CONFIRM_STEPS):
        self.on_at, self.off_below, self.confirm = float(on_at), float(off_below), int(confirm)
        self.on = None          # unknown until the first sample sets it, silently
        self.run = 0

    def update(self, v):
        v = float(v)
        if self.on is None:
            # the first sample is the state, not an event: a song that is on
            # from the first step is reported when it ends, not for existing
            self.on = v >= self.on_at
            return None
        hit = (v < self.off_below) if self.on else (v >= self.on_at)
        self.run = self.run + 1 if hit else 0
        if self.run >= self.confirm:
            self.on, self.run = not self.on, 0
            return "start" if self.on else "end"
        return None


class Crossing:
    """
    One-sided crossing with re-arm: fires once when `cross(v)` is true while
    armed, then waits until `rearm(v)` is true before it can fire again. The
    first sample only sets the arm state, so a fly that begins inside the
    band gets no line for being there.
    """

    def __init__(self, cross, rearm):
        self.cross, self.rearm = cross, rearm
        self.armed = None

    def update(self, v):
        v = float(v)
        if self.armed is None:
            self.armed = bool(self.rearm(v))
            return False
        if self.armed and self.cross(v):
            self.armed = False
            return True
        if not self.armed and self.rearm(v):
            self.armed = True
        return False


class Facing:
    """Faces / no longer faces with a band; None at the first sample."""

    def __init__(self, deg=FACING_DEG, band=FACING_BAND_DEG):
        self.inside, self.outside = float(deg), float(deg) + float(band)
        self.facing = None

    def update(self, bearing):
        b = abs(float(bearing))
        if self.facing is None:
            self.facing = b <= self.inside
            return None
        if not self.facing and b <= self.inside:
            self.facing = True
            return "in"
        if self.facing and b > self.outside:
            self.facing = False
            return "out"
        return None


# ---- the captioner --------------------------------------------------------------

class Captioner:
    """
    Feed it one world step at a time; it returns the transcript lines that
    step produced, each {seq, t, fly, kind, group, value, text}. `sizes` is
    {dictionary key: measured cell count} (sizes_from_groups on the resolved
    groups, or sizes_from_json); it fixes each group's floor and which groups
    are captioned at all, so a group the brain does not resolve is never named.
    """

    def __init__(self, sizes, dictionary=None, song_key=SONG_KEY, flies=FLIES,
                 dt=WORLD_DT_S, seq0=0, confirm=CONFIRM_STEPS):
        self.dictionary = dictionary or bd.DICTIONARY
        self.sizes = {k: int(v) for k, v in sizes.items()}
        if song_key not in self.sizes and SONG_FALLBACK_KEY in self.sizes:
            song_key = SONG_FALLBACK_KEY
        self.song_key = song_key
        self.flies = tuple(flies)
        if len(self.flies) != 2:
            raise ValueError("the room holds exactly two flies")
        self.dt = float(dt)
        self.seq = int(seq0)
        self.confirm = int(confirm)
        self.keys = captioned_keys(self.dictionary, self.sizes, song_key)
        base_len = max(1, int(round(BASELINE_S / self.dt)))
        self.rate = {f: {k: RateDetector(floor_hz(self.sizes[k]), base_len, confirm)
                         for k in self.keys} for f in self.flies}
        self.song = {f: LevelDetector(SONG_ON_SPIKES * SPIKE_HZ, SONG_OFF_SPIKES * SPIKE_HZ, confirm)
                     for f in self.flies}
        self.facing = {f: Facing() for f in self.flies}
        self.back = {f: Crossing(lambda v: v < -BACK_MM_S, lambda v: v >= -BACK_REARM_MM_S)
                     for f in self.flies}
        self.stop = {f: Crossing(lambda v: v < STOP_MM_S, lambda v: v >= STOP_REARM_MM_S)
                     for f in self.flies}
        self.near = Crossing(lambda d: d < NEAR_MM, lambda d: d >= NEAR_REARM_MM)
        self.far = Crossing(lambda d: d > FAR_MM, lambda d: d <= FAR_REARM_MM)
        self.prev = {f: None for f in self.flies}     # (x, y, heading) of the last step

    # -- rendering -------------------------------------------------------------

    def _entry(self, key):
        return self.dictionary[key]

    def _fields(self, key):
        e = self._entry(key)
        return {"name": e["name"], "role": e["role"], "cite": cite_key(e["citation"]),
                "unc": UNCERTAIN_SUFFIX if e["confidence"] == bd.UNCERTAIN else ""}

    def _line(self, t, fly, kind, group, value, template, **fields):
        text = TEMPLATES[template].format(fly=fly, value=float(value), **fields)
        line = {"seq": self.seq, "t": round(float(t), 3), "fly": fly, "kind": kind,
                "group": group, "value": round(float(value), 3), "text": text}
        self.seq += 1
        return line

    def _motor_fields(self, rates, key):
        """
        (template suffix, fields) for the back / stop lines: the group and
        its rate only when it is recorded AND fired in this step. A silent
        group is not named, so the line never puts a neuron and a paper
        beside an event that neuron did not take part in.
        """
        if key in self.sizes and key in rates and key in self.dictionary and float(rates[key]) > 0.0:
            f = self._fields(key)
            return "", {"name": f["name"], "role": f["role"], "cite": f["cite"],
                        "rate": float(rates[key])}
        return "_plain", {}

    # -- one step --------------------------------------------------------------

    def update(self, t, flies):
        """
        `flies` is {"A": {"x", "y", "heading", "rates": {key: hz}, "drive":
        {key: hz}, "song": hz}, "B": {...}}. Returns the new lines, in a fixed
        order, possibly none.
        """
        a, b = self.flies
        other = {a: b, b: a}
        geo, vel = {}, {}
        for f in self.flies:
            s, o = flies[f], flies[other[f]]
            geo[f] = geometry(s["x"], s["y"], s["heading"], o["x"], o["y"])
            p = self.prev[f]
            if p is None:
                vel[f] = None
            else:
                vx, vy = (float(s["x"]) - p[0]) / self.dt, (float(s["y"]) - p[1]) / self.dt
                h = math.radians(float(s["heading"]))
                along = vx * math.cos(h) + vy * math.sin(h)
                dist = geo[f][0]
                if dist > 0.0:
                    ux, uy = (float(o["x"]) - float(s["x"])) / dist, (float(o["y"]) - float(s["y"])) / dist
                    closing = vx * ux + vy * uy
                else:
                    closing = 0.0
                vel[f] = {"speed": math.hypot(vx, vy), "along": along, "closing": closing,
                          "dheading": wrap_deg(float(s["heading"]) - p[2])}

        out = []
        # 1. circuits and song, fly by fly
        for f in self.flies:
            rates = flies[f].get("rates", {})
            drive = flies[f].get("drive", {})
            for k in self.keys:
                if k not in rates:
                    continue
                ev = self.rate[f][k].update(rates[k])
                if ev is None:
                    continue
                fields = self._fields(k)
                if ev == "onset":
                    tpl = "onset"
                    d = float(drive.get(k, 0.0))
                    if k in SENSORY_TEMPLATES and d > 0.0:
                        tpl = SENSORY_TEMPLATES[k]
                        fields.update(drive=d, other=other[f])
                    out.append(self._line(t, f, "onset", k, rates[k], tpl, **fields))
                else:
                    out.append(self._line(t, f, "offset", k, rates[k], "offset", **fields))
            song = flies[f].get("song")
            if song is not None and self.song_key in self.sizes:
                ev = self.song[f].update(song)
                if ev == "start":
                    fields = self._fields(self.song_key)
                    out.append(self._line(t, f, "song_start", self.song_key, song, "song_start",
                                          n=self.sizes[self.song_key], **fields))
                elif ev == "end":
                    fields = self._fields(self.song_key)
                    out.append(self._line(t, f, "song_end", self.song_key, song, "song_end",
                                          n=self.sizes[self.song_key], name=fields["name"]))
        # 2. facing, per fly
        for f in self.flies:
            dist, bearing = geo[f]
            ev = self.facing[f].update(bearing)
            if ev is None:
                continue
            v = vel[f]
            dh = v["dheading"] if v else 0.0
            # the fly's own turn moved the bearing toward zero iff dh has the
            # sign of the bearing it had before the turn (bearing + dh)
            prev_bearing = wrap_deg(bearing + dh)
            own = dh * prev_bearing
            if ev == "in":
                tpl = "toward" if own > 0.0 else "faces"
                out.append(self._line(t, f, "toward", "", bearing, tpl, other=other[f], dist=dist))
            else:
                tpl = "away" if own < 0.0 else "unfaces"
                out.append(self._line(t, f, "away", "", bearing, tpl, other=other[f], dist=dist))
        # 3. distance, pair events attributed to the fly that moved it more
        dist = geo[a][0]
        if self.near.update(dist):
            f = self._mover(vel, key="closing", sign=1.0)
            out.append(self._line(t, f, "approach", "", dist, "approach", other=other[f]))
        if self.far.update(dist):
            f = self._mover(vel, key="closing", sign=-1.0)
            out.append(self._line(t, f, "leave", "", dist, "leave", other=other[f]))
        # 4. backing and stopping, per fly, from the room's kinematics
        for f in self.flies:
            v = vel[f]
            if v is None:
                continue
            rates = flies[f].get("rates", {})
            if self.back[f].update(v["along"]):
                suffix, mf = self._motor_fields(rates, BACK_GROUP)
                out.append(self._line(t, f, "back", BACK_GROUP if not suffix else "",
                                      -v["along"], "back" + suffix, **mf))
            if self.stop[f].update(v["speed"]):
                suffix, mf = self._motor_fields(rates, STOP_GROUP)
                out.append(self._line(t, f, "stop", STOP_GROUP if not suffix else "",
                                      v["speed"], "stop" + suffix, **mf))
        for f in self.flies:
            s = flies[f]
            self.prev[f] = (float(s["x"]), float(s["y"]), float(s["heading"]))
        return out

    def _mover(self, vel, key, sign):
        """The fly whose `key` velocity component times `sign` is larger; A on ties or no velocity."""
        a, b = self.flies
        va, vb = vel[a], vel[b]
        if va is None or vb is None:
            return a
        return b if sign * vb[key] > sign * va[key] else a


def allowed_pattern(dictionary=None):
    """A regex matching one token of the vocabulary, or a number: the allow-list the tests use, compiled once."""
    alts = "|".join(sorted((re.escape(w) for w in vocabulary(dictionary)), key=len, reverse=True))
    return re.compile(rf"^(?:{alts}|\d+)$")


_NUMBER = r"-?\d+(?:\.\d+)?"
_PLACEHOLDER = re.compile(r"(\{\w+(?::[^}]*)?\})")


def template_patterns(dictionary=None, flies=FLIES):
    """
    Compiled regexes for every line this code can write now: each template's
    literal text with its placeholders replaced by what may fill them (a fly
    name; a number; and, for templates that name a group, that ONE group's
    name, role, citation key and its uncertain suffix or none, one pattern
    per present group, motor groups only for the back / stop templates). A
    line matches iff it is a current template filled with one current
    entry's strings; a line with a role or citation this code no longer
    carries, an uncertain group without its suffix, or allowed words in any
    other arrangement matches nothing.
    """
    dictionary = dictionary or bd.DICTIONARY
    present = [e for e in dictionary.values() if e["present"]]
    fly_alt = "(?:" + "|".join(re.escape(f) for f in flies) + ")"
    common = {"fly": fly_alt, "other": fly_alt, "value": _NUMBER, "drive": _NUMBER,
              "dist": _NUMBER, "rate": _NUMBER, "n": r"\d+"}

    def compile_(parts, fields):
        rx = "".join(fields[re.match(r"\{(\w+)", p).group(1)] if p.startswith("{") else re.escape(p)
                     for p in parts)
        return re.compile("^" + rx + "$")

    out = []
    for t in TEMPLATES.values():
        parts = [p for p in _PLACEHOLDER.split(t) if p]
        names = {re.match(r"\{(\w+)", p).group(1) for p in parts if p.startswith("{")}
        if not names & {"name", "role", "cite", "unc"}:
            out.append(compile_(parts, common))
            continue
        pool = [e for e in present if e["motor"]] if "rate" in names else present
        for e in pool:
            fields = dict(common, name=re.escape(e["name"]), role=re.escape(e["role"]),
                          cite=re.escape(cite_key(e["citation"])),
                          unc=re.escape(UNCERTAIN_SUFFIX) if e["confidence"] == bd.UNCERTAIN else "")
            out.append(compile_(parts, fields))
    return out


def line_is_from_the_code(row, patterns):
    """
    True when a stored transcript row could have been written by this code
    as it is now: its text is one of the templates filled with current
    dictionary strings and numbers (template_patterns), its fly is a fly
    name or empty, its kind a known kind or empty. The loop applies this to
    rows reloaded from build/backrooms_transcript.jsonl, so a line put into
    that file by hand (the one path by which a person could get words onto
    the page), or a line from an older version of the dictionary whose role
    or citation has since changed, is dropped and counted rather than
    served. A template match implies every word is in the vocabulary; the
    converse does not hold, which is why the match is by template.
    """
    text = row.get("text") if isinstance(row, dict) else None
    if not isinstance(text, str) or not text.strip():
        return False
    if row.get("fly", "") not in ("",) + tuple(FLIES):
        return False
    if row.get("kind", "") not in ("",) + tuple(KINDS):
        return False
    return any(p.match(text) is not None for p in patterns)


def caption_record(record, sizes, **kw):
    """
    The pure form: a fresh Captioner run over recorded arrays. `record` is
    {"t": array, "A": {"x", "y", "heading": arrays, "rates": {key: array},
    "drive": {key: array}, "song": array}, "B": {...}}. Returns every line.
    """
    cap = Captioner(sizes, **kw)
    n = len(record["t"])
    lines = []
    for i in range(n):
        flies = {}
        for f in cap.flies:
            r = record[f]
            flies[f] = {
                "x": float(r["x"][i]), "y": float(r["y"][i]), "heading": float(r["heading"][i]),
                "rates": {k: float(v[i]) for k, v in r.get("rates", {}).items()},
                "drive": {k: float(v[i]) for k, v in r.get("drive", {}).items()},
                "song": float(r["song"][i]) if "song" in r else None,
            }
        lines.extend(cap.update(record["t"][i], flies))
    return lines


def format_line(line):
    """One transcript line as the page and the log print it: 'ss.sss  text'."""
    return f"{line['t']:9.3f}  {line['text']}"


# =============================================================================
# SECTION 2: THE SERVER (world loop, recorder, FastAPI, --publish tunnel).
# Added by the server agent below this line. Nothing above depends on it.
# The server should: resolve groups with backrooms_dictionary.present_groups(fb),
# build sizes with sizes_from_groups, keep one Captioner for the run (seq0 =
# the last seq in build/backrooms_transcript.jsonl on restart), and call
# Captioner.update(t, flies) once per world step with the recorded numbers.
# =============================================================================

# SECTION 2 holds the server and the loop. It imports nothing that section 1
# does not already need except the web stack (FastAPI, uvicorn) at call time,
# so `import backrooms` stays cheap and test_backrooms_captions.py never
# starts a server. No brain is loaded here: the world holds the one brain.
#
# INTERFACE the loop expects of the world (CHOSEN here so the parts can be
# built apart; backrooms_world.World is the real one):
#
#   world.step() -> record
#       One 50 ms world step for both flies:
#         {"step": int, "t": float,            # world time in seconds
#          "flies": {"A": FLY, "B": FLY}}
#       FLY = {"x": mm, "y": mm,               # 0..20 inside the arena
#              "heading": deg,                 # 0 = +x, counter-clockwise +
#              "speed": mm/s, "turn": deg/s,
#              "distance": mm, "bearing": deg, # to / of the other fly
#              "rates": {group: Hz},           # every present dictionary group
#              "counts": {group: spikes},      # the same groups, per window
#              "drive": {"ORN_DA1","JO_A","JO_B"},  # input delivered, Hz
#              "motor": {"steer_L","steer_R","fwd_L","fwd_R","back","stop"},
#              "song": Hz}                     # summed pulse wing-MN rate
#       Extra keys pass through to the step log untouched; numpy scalars and
#       arrays are converted before anything is written or sent.
#   world.groups     {dictionary key: neuron index array}, i.e.
#                    backrooms_dictionary.present_groups(fb); sizes come from it.
#   world.meta()     (optional) what the world chose, copied into /state.
#
# The captioner is section 1's: Captioner.update(t, flies) once per step.
# The loop owns the transcript numbering: it sets the captioner's seq so the
# two agree, and re-stamps seq on every line it stores.
#
# WHAT THE LOOP DOES per step, in order: world.step -> captioner.update ->
# append the lines to build/backrooms_transcript.jsonl (one write, whole
# lines) -> append the record to build/backrooms_steps.jsonl (rotated at
# 64 MB, one older file kept) -> rewrite build/backrooms_state.json by
# write-then-rename -> hand one frame to every websocket. The brain step is
# CPU work and runs in a worker thread so the server keeps answering.
#
# MEMORY: main() checks free RAM before the world is built and refuses under
# 6 GB, because a loaded brain is 2.5 GB and other workflows hold brains.
#
#   py backrooms.py                        serve on http://localhost:4720
#   py backrooms.py --port 4720 --publish  also open a cloudflared quick tunnel
#   py -m pytest -q test_backrooms_server.py test_backrooms_page.py

import argparse
import asyncio
import os
import sys
import threading
import time
import traceback
from contextlib import asynccontextmanager

import numpy as np

PAGE = ROOT / "web" / "backrooms.html"
DICT_PATH = bd.JSON_PATH

# ---- the room's fixed numbers (CHOSEN; see DESIGN in the build notes) -------
ARENA_MM = 20.0                 # square arena side
N_NEURONS = 165122              # MEASURED: build/graph.npz
CHANNEL_ORDER = ["smell", "sound", "sight", "courtship", "aggression",
                 "song", "motor"]

KEEP_LINES = 5000               # transcript lines held in memory for /transcript
TRANSCRIPT_PAGE = 500           # most lines one /transcript call returns
FRAME_QUEUE = 30                # frames a slow watcher may fall behind by
STEPS_CAP_BYTES = 64 * 1024 * 1024
LAST_LINES_IN_STATE = 40

# The banner. web/backrooms.html carries these two paragraphs verbatim at the
# top of the page and test_backrooms_page.py checks that it does. The first
# is section 1's canonical account of how a line is made; the second says
# what the room is and what in it was chosen.
ROOM_NOTE = (
    "The room: two copies of one fruit fly connectome (165,122 neurons, leaky "
    "integrate-and-fire, one loaded graph carrying two state vectors) walk a "
    "20 by 20 mm arena, and each fly's walking comes from its own descending "
    "neurons. Each fly meets the other as a dark ellipse on a mid-grey ground in "
    "its eye (sampled one pixel per lamina column; both optic lobes carry the "
    "same column coordinates in this dataset, so both sample one window and "
    "there is no left-eye versus right-eye difference), as cVA on its ORN_DA1 "
    "receptor neurons and as song on its JO-A and JO-B hearing neurons, each by "
    "a mapping chosen by the people who built this and disclosed in the code. "
    "Smell and sound are delivered at the same total per antenna (each cell's "
    "rate scaled by its side's cell count), so they carry no left/right "
    "information by construction; how many columns the other fly covers at each "
    "distance, and the retinal rates on and off it, are measured and listed in "
    "the state. No synapse changes during the run: no dopamine is delivered and "
    "no weight is updated; the two brains do not learn from each other, and the "
    "only thing that adapts is the captioner's 10 s baseline. A world step takes "
    "longer than 50 ms of wall time; the measured ratio is in the status line. "
    "The dictionary the names come from, with the paper, figure and cell count "
    "behind each, is linked below."
)
BANNER_PARAGRAPHS = (HOW_LINES_ARE_MADE, ROOM_NOTE)


def say(*parts):
    """Print without ever raising on a console that cannot show a character."""
    try:
        print(*parts, flush=True)
    except Exception:
        try:
            print(*[str(p).encode("ascii", "replace").decode() for p in parts],
                  flush=True)
        except Exception:
            pass


# ---- plain data ------------------------------------------------------------

def plain(obj):
    """
    numpy and non-finite floats out, so that what is written and sent is JSON
    a browser can parse. NaN and inf become null rather than the bare NaN
    json.dumps would emit, which JSON.parse rejects.
    """
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return plain(obj.tolist())
    if isinstance(obj, np.generic):
        return plain(obj.item())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def dumps(obj, indent=None):
    return json.dumps(plain(obj), ensure_ascii=True, indent=indent,
                      separators=None if indent else (",", ":"))


RENAME_TRIES = 8                # Windows: a reader holding the target blocks the rename
RENAME_WAIT_S = 0.05


def atomic_write(path, data: bytes):
    """
    Write then rename, so a reader never sees half a file. os.replace is
    atomic on NTFS for a same-volume rename; the .tmp sits beside the target
    so it is on the same volume.

    On Windows a reader that has the target open without FILE_SHARE_DELETE
    (PowerShell's Get-Content, a text editor) makes os.replace raise
    PermissionError for as long as it holds the handle. MEASURED in the first
    15-minute run: 1 of 2,800 state writes failed that way while a monitor
    read the file. The rename is retried a few times over a fraction of a
    second and the error is re-raised if the target stays held, so the old
    file is left whole either way.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    for attempt in range(RENAME_TRIES):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == RENAME_TRIES - 1:
                raise
            time.sleep(RENAME_WAIT_S)


def append_jsonl(path, rows):
    """
    One write call for all of this step's rows, each a complete line. A crash
    mid-write can leave at most one torn line at the end of the file, and
    load_jsonl skips a line that does not parse.
    """
    if not rows:
        return 0
    data = "".join(dumps(r) + "\n" for r in rows).encode("utf-8")
    with open(path, "ab") as f:
        f.write(data)
    return len(data)


def load_jsonl(path, keep=None):
    """The rows of a jsonl file, last `keep` of them, torn lines skipped."""
    path = Path(path)
    if not path.exists():
        return []
    rows = deque(maxlen=keep) if keep else []
    with open(path, "rb") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except ValueError:
                continue
    return list(rows)


def load_group_meta(path=DICT_PATH):
    """
    What the page may say about each group, from the dictionary JSON: name,
    role words, citation, confidence, channel, measured cell counts. Absent
    groups (present False) are left out because the page must not print them.
    Missing file -> {} (main() refuses to start that way; tests use a fake).
    """
    path = Path(path)
    if not path.exists():
        return {}
    d = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for k, e in d.get("groups", {}).items():
        if not e.get("present", False):
            continue
        c = e.get("counts", {})
        out[k] = {
            "name": e.get("name", k),
            "role": e.get("role", ""),
            "citation": e.get("citation", ""),
            "confidence": e.get("confidence", "uncertain"),
            "channel": e.get("channel", ""),
            "motor": bool(e.get("motor", False)),
            "contains": list(e.get("contains", [])),
            "n": int(c.get("n", 0)),
            "L": int(c.get("L", 0)),
            "R": int(c.get("R", 0)),
        }
    return out


def captioner_meta():
    """Section 1's CHOSEN constants, so /state discloses every threshold."""
    return {
        "world_dt_s": WORLD_DT_S, "brain_steps": BRAIN_STEPS, "lif_dt_ms": LIF_DT_MS,
        "window_ms": WINDOW_MS, "spike_hz": SPIKE_HZ,
        "baseline_s": BASELINE_S, "onset_ratio": ONSET_RATIO,
        "offset_ratio": OFFSET_RATIO, "floor_min_hz": FLOOR_MIN_HZ,
        "floor_spikes": FLOOR_SPIKES, "offset_floor_fraction": OFFSET_FLOOR_FRACTION,
        "confirm_steps": CONFIRM_STEPS, "song_key": SONG_KEY,
        "song_on_spikes": SONG_ON_SPIKES, "song_off_spikes": SONG_OFF_SPIKES,
        "facing_deg": FACING_DEG, "facing_band_deg": FACING_BAND_DEG,
        "near_mm": NEAR_MM, "near_rearm_mm": NEAR_REARM_MM,
        "far_mm": FAR_MM, "far_rearm_mm": FAR_REARM_MM,
        "back_mm_s": BACK_MM_S, "back_rearm_mm_s": BACK_REARM_MM_S,
        "stop_mm_s": STOP_MM_S, "stop_rearm_mm_s": STOP_REARM_MM_S,
        "templates": dict(TEMPLATES), "uncertain_suffix": UNCERTAIN_SUFFIX,
    }


# ---- the loop --------------------------------------------------------------

FLY_FRAME_KEYS = ("x", "y", "heading", "speed", "turn", "distance", "bearing",
                  "rates", "drive", "motor", "song")


class Loop:
    """
    Steps the world, feeds the captioner, keeps the transcript, writes the
    files and hands frames to watchers. Holds no brain itself: the world does.

    tick() is synchronous and does one whole step; run() calls it in a worker
    thread forever. Tests drive tick() directly with fakes.
    """

    def __init__(self, world, captioner, out=BUILD, dictionary=DICT_PATH,
                 keep=KEEP_LINES):
        self.world = world
        self.captioner = captioner
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.state_path = self.out / "backrooms_state.json"
        self.transcript_path = self.out / "backrooms_transcript.jsonl"
        self.steps_path = self.out / "backrooms_steps.jsonl"
        self.groups = load_group_meta(dictionary)
        # reload: only rows the code could have written are served again;
        # the numbering continues past every row, kept or dropped, so a
        # dropped row's number is never reused
        rows = load_jsonl(self.transcript_path, keep)
        patterns = template_patterns()
        kept = [r for r in rows if line_is_from_the_code(r, patterns)]
        self.dropped_on_reload = len(rows) - len(kept)
        self.lines = deque(kept, maxlen=keep)
        self.seq = max([self._seq_of(r) for r in rows] + [0])
        if hasattr(captioner, "seq"):
            captioner.seq = self.seq + 1     # so its numbers match the store's
        self._steps_bytes = (self.steps_path.stat().st_size
                             if self.steps_path.exists() else 0)
        self.rotation_failures = 0          # step-log rotations refused by a held file
        self.state_write_failures = 0       # state renames refused by a held file
        self.lock = threading.Lock()
        self.queues = {}            # asyncio.Queue -> the event loop it lives on
        self.running = False
        self.started = time.time()
        self.step = 0
        self.t = 0.0
        self.wall_per_step = None   # EMA of seconds per tick, MEASURED
        self.tunnel = None
        self.state = self._base_state()
        self._state_text = dumps(self.state, indent=1)

    # -- state ---------------------------------------------------------------

    @staticmethod
    def _seq_of(row):
        try:
            return int(row.get("seq", 0))
        except (TypeError, ValueError, AttributeError):
            return 0

    def _meta(self, obj):
        m = getattr(obj, "meta", None)
        try:
            return plain(m()) if callable(m) else {}
        except Exception as exc:
            return {"error": str(exc)[:120]}

    def _base_state(self):
        return {
            "what": "two fly connectome copies in a 20 x 20 mm arena; the "
                    "transcript is templates over measured neuron groups",
            "how": list(BANNER_PARAGRAPHS),
            "started": self.started,
            "updated": self.started,
            "step": 0,
            "t": 0.0,
            "wall_per_step_s": None,
            "slowdown": None,
            "arena_mm": ARENA_MM,
            "world_dt_s": WORLD_DT_S,
            "brain_steps_per_world_step": BRAIN_STEPS,
            "brain_dt_ms": LIF_DT_MS,
            "brain_ms_per_world_step": WINDOW_MS,
            "near_mm": NEAR_MM,
            "far_mm": FAR_MM,
            "n_neurons": N_NEURONS,
            "channel_order": CHANNEL_ORDER,
            "groups": self.groups,
            "world": self._meta(self.world),
            "captioner": captioner_meta(),
            "tunnel": self.tunnel,
            "seq": self.seq,
            "lines_kept": len(self.lines),
            "lines_dropped_on_reload": self.dropped_on_reload,
            "reload_rule": "a stored transcript row is served again only if its text "
                           "is one of the code's templates filled with the current "
                           "dictionary's names, roles and citation keys and numbers "
                           "(backrooms.line_is_from_the_code); the rest, including "
                           "lines from an older dictionary, are dropped and counted here",
            "state_write_failures": self.state_write_failures,
            "rotation_failures": self.rotation_failures,
            "flies": {},
            "last_lines": list(self.lines)[-LAST_LINES_IN_STATE:],
            "files": {
                "state": str(self.state_path),
                "transcript": str(self.transcript_path),
                "steps": str(self.steps_path),
            },
        }

    def state_text(self):
        with self.lock:
            return self._state_text

    def snapshot(self):
        return json.loads(self.state_text())

    def set_tunnel(self, url):
        with self.lock:
            self.tunnel = url
            self.state["tunnel"] = url
            self._write_state()

    def _write_state(self):
        # caller holds the lock
        self._state_text = dumps(self.state, indent=1)
        atomic_write(self.state_path, (self._state_text + "\n").encode("utf-8"))

    # -- one step ------------------------------------------------------------

    def tick(self):
        """
        One world step: step, caption, log, write, offer a frame. Returns the
        frame. Wall time per step is measured over the whole of this call
        except the final offer, and smoothed with a 0.1 EMA.
        """
        t0 = time.perf_counter()
        rec = plain(self.world.step())
        step = int(rec.get("step", self.step + 1))
        t = float(rec.get("t", step * WORLD_DT_S))
        new = self.captioner.update(t, rec.get("flies") or {}) or []
        with self.lock:
            self.step, self.t = step, t
            now = time.time()
            lines = []
            for ln in new:
                self.seq += 1
                line = {
                    "seq": self.seq,
                    "step": self.step,
                    "t": self.t,
                    "at": now,
                    "fly": str(ln.get("fly", "") or ""),
                    "kind": str(ln.get("kind", "") or ""),
                    "group": str(ln.get("group", "") or ""),
                    "value": plain(ln.get("value")),
                    "text": str(ln["text"]),
                }
                self.lines.append(line)
                lines.append(line)
            append_jsonl(self.transcript_path, lines)
            self._log_step(rec)
            wall = time.perf_counter() - t0
            self.wall_per_step = (wall if self.wall_per_step is None
                                  else 0.9 * self.wall_per_step + 0.1 * wall)
            frame = self._frame(rec, lines, now)
            self._update_state(rec, frame)
            try:
                self._write_state()
            except PermissionError as exc:
                # Windows: a reader holds backrooms_state.json. The transcript
                # and step rows are already on disk and the in-memory state
                # (what /state serves) is current, so the step is complete;
                # the file catches up at the next step and the frame still
                # goes to every watcher rather than aborting the step here.
                self.state_write_failures += 1
                self.state["state_write_failures"] = self.state_write_failures
                self._state_text = dumps(self.state, indent=1)   # /state carries the count now
                say("state file held by a reader, kept in memory:", str(exc)[:80])
        self._offer(frame)
        return frame

    def _log_step(self, rec):
        """Every step, whole, to a jsonl that is rotated so forever fits on disk."""
        if self._steps_bytes > STEPS_CAP_BYTES:
            older = self.steps_path.with_name(self.steps_path.name + ".1")
            try:
                os.replace(self.steps_path, older)
                self._steps_bytes = 0
            except OSError:
                # a reader holds the file: the count is NOT reset, so the
                # rotation is tried again at the next step, not 64 MB later
                self.rotation_failures += 1
                self.state["rotation_failures"] = self.rotation_failures
                try:
                    self._steps_bytes = self.steps_path.stat().st_size
                except OSError:
                    pass
        row = dict(rec)
        row["at"] = time.time()
        self._steps_bytes += append_jsonl(self.steps_path, [row])

    def _frame(self, rec, lines, now):
        flies = {}
        for name, fly in (rec.get("flies") or {}).items():
            flies[name] = {k: fly.get(k) for k in FLY_FRAME_KEYS if k in fly}
        return {
            "type": "frame",
            "step": self.step,
            "t": self.t,
            "at": now,
            "wall_per_step_s": self.wall_per_step,
            "slowdown": (self.wall_per_step / WORLD_DT_S
                         if self.wall_per_step else None),
            "seq": self.seq,
            "flies": flies,
            "lines": lines,
        }

    def _update_state(self, rec, frame):
        s = self.state
        s["updated"] = frame["at"]
        s["step"] = self.step
        s["t"] = self.t
        s["wall_per_step_s"] = self.wall_per_step
        s["slowdown"] = frame["slowdown"]
        s["seq"] = self.seq
        s["lines_kept"] = len(self.lines)
        s["state_write_failures"] = self.state_write_failures
        s["rotation_failures"] = self.rotation_failures
        s["flies"] = rec.get("flies") or {}
        s["tunnel"] = self.tunnel
        s["last_lines"] = list(self.lines)[-LAST_LINES_IN_STATE:]

    # -- watchers ------------------------------------------------------------

    def subscribe(self):
        """A queue on the calling coroutine's event loop; frames arrive on it."""
        q = asyncio.Queue()
        self.queues[q] = asyncio.get_running_loop()
        return q

    def unsubscribe(self, q):
        self.queues.pop(q, None)

    @staticmethod
    def _put(q, frame):
        while q.qsize() >= FRAME_QUEUE:
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                break
        q.put_nowait(frame)

    def _offer(self, frame):
        """Hand the frame to every watcher's loop; safe from any thread."""
        for q, lp in list(self.queues.items()):
            try:
                lp.call_soon_threadsafe(self._put, q, frame)
            except RuntimeError:
                self.queues.pop(q, None)

    def lines_after(self, after, limit=TRANSCRIPT_PAGE):
        with self.lock:
            out = [ln for ln in self.lines if ln["seq"] > after]
        return out[:limit]

    # -- forever -------------------------------------------------------------

    async def run(self, steps=0):
        """
        Step until told to stop. A step that raises is reported and the loop
        waits two seconds and goes on, so one bad frame is not the end of the
        room. `steps` > 0 stops after that many, for a smoke run.
        """
        self.running = True
        done = 0
        while self.running and (steps <= 0 or done < steps):
            try:
                await asyncio.to_thread(self.tick)
                done += 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                say("step failed:", str(exc)[:160])
                traceback.print_exc()
                await asyncio.sleep(2)
        self.running = False


# ---- the public tunnel -----------------------------------------------------
# Copied from roam.start_tunnel rather than imported: importing roam loads the
# browser rails, the mushroom body and .env, none of which this room may
# touch. The lines are the same so a fix there is a fix here.

CFD = Path(os.path.expanduser("~/.claude/tools/cloudflared/cloudflared.exe"))
TUNNEL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
TUNNEL = {"url": None, "proc": None}


def tunnel_url(line):
    """The quick-tunnel address in one line of cloudflared's output, or None."""
    m = TUNNEL_RE.search(line or "")
    return m.group(0) if m else None


def start_tunnel(port, found):
    """
    Put this page on the public internet with a cloudflared quick tunnel.

    The address is random and changes every run, so it is written into the
    state file (via `found`) rather than hardcoded anywhere. Nothing else
    leaves the machine: no address is published to any repo or service.
    """
    import subprocess

    if os.environ.get("FLY_TUNNEL") == "0":
        say("FLY_TUNNEL=0 - this run stays on this machine")
        return None
    if not CFD.exists():
        say("no cloudflared at", CFD, "- this run stays on this machine")
        return None

    proc = subprocess.Popen(
        [str(CFD), "tunnel", "--no-autoupdate", "--url",
         f"http://127.0.0.1:{port}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace", bufsize=1)
    TUNNEL["proc"] = proc

    def watch():
        for line in proc.stdout:
            url = tunnel_url(line)
            if url and not TUNNEL["url"]:
                TUNNEL["url"] = url
                say("tunnel open:", url)
                try:
                    found(url)
                except Exception as exc:
                    say("could not record the tunnel:", str(exc)[:120])

    threading.Thread(target=watch, daemon=True).start()
    return proc


# ---- the server ------------------------------------------------------------

def make_app(loop, page=PAGE, dictionary=DICT_PATH, autostart=False,
             publish=False, port=None, steps=0):
    """
    The FastAPI app around a Loop. Routes:
      GET  /                 web/backrooms.html
      GET  /state            the latest state (same object as the state file)
      GET  /transcript?after=SEQ   lines with seq > SEQ, oldest first, <= 500
      GET  /dictionary.json  build/backrooms_dictionary.json
      WS   /ws               one "hello" carrying the state, then one frame
                             per world step; the client sends nothing
    With autostart the loop runs from startup; tests leave it off and tick.
    """
    from fastapi import FastAPI, Query, WebSocket
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse, Response

    @asynccontextmanager
    async def lifespan(app):
        task = None
        if autostart:
            task = asyncio.create_task(loop.run(steps))
        if publish and port:
            start_tunnel(port, loop.set_tunnel)
        try:
            yield
        finally:
            loop.running = False
            if task:
                task.cancel()
            proc = TUNNEL.get("proc")
            if proc and proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass

    app = FastAPI(lifespan=lifespan)
    # read-only routes; a page on another origin may read them
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["GET"], allow_headers=["*"])
    app.state.loop = loop
    page = Path(page)
    dictionary = Path(dictionary)

    @app.get("/")
    def index():
        if not page.exists():
            return Response("no page", status_code=404, media_type="text/plain")
        return FileResponse(str(page), media_type="text/html")

    @app.get("/state")
    def state():
        return Response(loop.state_text(), media_type="application/json")

    @app.get("/transcript")
    def transcript(after: int = Query(0, ge=0)):
        lines = loop.lines_after(after)
        return {"after": after, "seq": loop.seq, "lines": lines,
                "kept": len(loop.lines), "more": len(lines) >= TRANSCRIPT_PAGE}

    @app.get("/dictionary.json")
    def dictionary_json():
        if not dictionary.exists():
            return JSONResponse({"error": "no dictionary; run py "
                                          "backrooms_dictionary.py --write"},
                                status_code=404)
        return FileResponse(str(dictionary), media_type="application/json")

    @app.websocket("/ws")
    async def socket(ws: WebSocket):
        """A watcher. It is told the state, then every frame. It cannot send."""
        await ws.accept()
        q = loop.subscribe()
        recv = asyncio.ensure_future(ws.receive())
        get = asyncio.ensure_future(q.get())
        try:
            await ws.send_text(dumps({"type": "hello", "state": loop.snapshot()}))
            while True:
                done, _ = await asyncio.wait({recv, get},
                                             return_when=asyncio.FIRST_COMPLETED)
                if recv in done:
                    msg = recv.result()
                    if msg.get("type") == "websocket.disconnect":
                        break
                    recv = asyncio.ensure_future(ws.receive())   # ignored
                if get in done:
                    frame = get.result()
                    get = asyncio.ensure_future(q.get())
                    await ws.send_text(dumps(frame))
        except Exception:
            pass
        finally:
            loop.unsubscribe(q)
            for fut in (recv, get):
                fut.cancel()

    return app


# ---- memory guard ------------------------------------------------------------

def free_ram_gb():
    """Free RAM in GB (Linux /proc/meminfo, then PowerShell); None if it cannot be read."""
    import backrooms_world as world_module
    v = world_module.free_ram_gb()
    return None if v != v else v

# ---- the plug point ------------------------------------------------------------

def dictionary_is_current(path=DICT_PATH):
    """
    True when the JSON on disk is byte-identical to what
    backrooms_dictionary.build() makes now. The page's tooltips take roles
    and citations from the file and the transcript takes them from the code;
    the server refuses to start unless they are the same text, so the two
    cannot drift apart (build/ is writable, --check is not otherwise run).
    """
    path = Path(path)
    if not path.exists():
        return False
    return path.read_text(encoding="utf-8") == bd.dumps(bd.build())


def build_objects(seed=0, ram_check=True):
    """
    The world (which loads the one brain) and the captioner. This is the only
    place the parts meet. Sizes come from the groups the world resolved on its
    brain (so a group the brain does not resolve is never named), falling back
    to the dictionary JSON's counts if the world does not expose them.
    """
    import backrooms_world
    world = backrooms_world.World(seed=seed, ram_check=ram_check)
    groups = getattr(world, "groups", None)
    sizes = sizes_from_groups(groups) if groups else sizes_from_json(DICT_PATH)
    captioner = Captioner(sizes, dt=WORLD_DT_S)
    return world, captioner


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="two flies in a 20 x 20 mm room, on one loaded connectome")
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("PORT", "4720")))
    ap.add_argument("--publish", action="store_true",
                    help="open a cloudflared quick tunnel and record its URL "
                         "in the state file")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=0,
                    help="stop after this many world steps (0 = forever)")
    ap.add_argument("--no-ram-check", action="store_true")
    a = ap.parse_args(argv)

    if not dictionary_is_current():
        raise SystemExit(f"{DICT_PATH} is missing or stale (its text differs from "
                         "backrooms_dictionary.build()): run py "
                         "backrooms_dictionary.py --write")
    if not a.no_ram_check:
        free = free_ram_gb()
        if free is not None and free < 6.0:
            raise SystemExit(f"free RAM {free:.1f} GB < 6 GB; a loaded brain is "
                             "2.5 GB and two other workflows hold brains - "
                             "not loading")
        say(f"free RAM {free:.1f} GB" if free is not None else "free RAM unknown")

    import uvicorn
    world, captioner = build_objects(seed=a.seed, ram_check=not a.no_ram_check)
    loop = Loop(world, captioner)
    app = make_app(loop, autostart=True, publish=a.publish, port=a.port,
                   steps=a.steps)
    say(f"the backrooms - open http://localhost:{a.port}")
    uvicorn.run(app, host=os.environ.get("FLY_HOST", "127.0.0.1"),
                port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
