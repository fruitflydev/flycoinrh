"""
The backrooms world: one room, two flies, three channels between them, and
the coupling from each fly's senses to one shared connectome and back to its
legs.

Why this file exists. The backrooms page shows two copies of the male CNS
connectome walking in a 20 x 20 mm arena, each seeing, smelling and hearing
the other. Nothing here decides anything and nothing here writes a word: this
module turns geometry into spike rates on the sensory neurons that carry those
signals in a real fly, runs the brain for one window, and turns the descending
neurons' rates back into a turn and a speed with the roamer's conversions. The
transcript and the page are other files; they read the numbers this one
records.

MEASURED (this connectome, build/graph.npz; counts checked in this session)
  * ORN_DA1, the cVA receptor neurons: 204 cells. JO-A: 50 cells, JO-B: 89
    cells (the sound-sensitive Johnston's organ classes). Pulse-song wing
    motor neurons (ps1, i1, iii1, b3 MN): 8 cells. The roamer's motor types:
    DNa02 2, DNa01 2, MDN 4, DNp09 2. L1 1,776 and L2 1,779 lamina cells, of
    which pumpui.FlyEye keeps those with a hex column. The groups themselves
    are resolved by backrooms_dictionary, which cites each identity.
  * pumpui.FlyEye.look samples a 300 x 210 px window at the gaze point onto
    the hex columns, L1 at lum x 180 Hz and L2 at (1 - lum) x 108 Hz.
  * One 60-step window of flysim.FlyBrain is 60 x 0.2 ms = 12 ms of brain
    time, so every per-cell rate is a multiple of 83.3 Hz.

CHOSEN (every number below is a choice, disclosed by describe())
  * The room: 20 x 20 mm, a courtship-chamber scale (real assays use 10-30 mm
    chambers). World step 50 ms. Walking up to 20 mm/s, turning up to 360
    deg/s; the command is clipped to [-1, 1] and scales those. The fly turns,
    then walks along its new heading. Walls clamp the position. Starts are
    drawn from the seed inside a 2 mm margin.
  * Sight: a 1280 x 800 grey frame of the ground at 0.5 with the other fly
    as an ellipse at 0.0, so a lamina column on the ground gets L1 90 Hz /
    L2 54 Hz and one on the fly L1 0 / L2 108 Hz. (The first draft used a
    ground of 0.05: L1 9 vs 0 Hz, and the measured 15-minute run showed the
    other fly changing the retina's mean input by 0.2 Hz, i.e. no signal;
    the ground was raised so the ellipse has contrast.) The 300 px eye
    window at the frame centre spans the 180 deg in front of the fly (1.667
    px per degree), so a fly behind the viewer is drawn in the frame but not
    in the window: the fly looks where it faces. The image x axis is the
    viewer's right; a fly on the viewer's left appears left of centre.
    Vertical position and size come from ground-plane geometry with the eye
    1.0 mm above the floor and a body of 2.5 x 1.0 x 1.0 mm; the projected
    width follows the other's heading relative to the line of sight (side-on
    shows the length, head-on the width). Each semi-axis is floored at one
    lamina-column spacing, so the other fly always covers a few columns and
    cannot fall between the eye's point samples at long range. pumpui.FlyEye
    samples one pixel per column, and both optic lobes carry the same hex
    coordinates in this dataset (875 of 875 left L1 columns coincide with
    right ones; FlyEye normalises them together), so the two retinas sample
    the same window and nothing here is a left-eye versus right-eye
    difference. MEASURED: the eye's 1,767 L1 cells therefore sample 892
    distinct positions, 9.0 px apart over the 300 x 210 window
    (column_spacing_px(892)); build_room counts the distinct positions of
    the eye it has, not its cells (counting cells gave 6.4 px in the first
    run after this change, an underestimate, since coincident columns
    sample the same pixel). Room.sight_check() measures,
    through the eye the room has, how many columns the ellipse covers at
    each distance and the retinal rates on and off it; describe() carries
    the result. MEASURED with this eye: the other fly dead ahead covers 228
    L1 columns at 2 mm side-on, 38 at 5 mm and 14 at 20 mm (the floor), at
    L1 90 vs 0 Hz per column; in the 50-world-second run after the change,
    LC10a's mean rate was 0.60 Hz with the other fly in the window against
    0.31 Hz without (A) and 0.48 against 0.38 (B), correlation +0.06 / +0.04
    over 994 steps: the retina now carries the signal and LC10a shows a
    weak association with it, not an established pursuit response. None of
    this is a measured optical model of the compound eye.
  * Smell: ORN_DA1 on fly X is driven at 200 Hz x clip(1 - d / 20 mm, 0, 1)
    by the other fly's presence (males carry cVA; DoOR has no cVA entry, so
    the receptor ceiling of Hallem and Carlson 2006 is used directly).
  * Per-side equalisation of smell and sound (v2): the driven antennal
    groups are unevenly split between the antennae by rootSide (ORN_DA1 R
    105 / L 51 with 48 unsided; JO-A L 18 / R 32; JO-B L 60 / R 29), so one
    uniform per-cell rate would deliver 2.06x the cVA cell-Hz to the right
    antenna and 1.28x the song cell-Hz to the left at every step, a lateral
    signal unrelated to where the other fly is (the first 15-minute run did
    exactly that, undisclosed). Each cell's rate is therefore the nominal
    rate x (mean count per side / its own side's count), per group, as
    plume_fly.side_scales does for wind; unsided cells get the nominal
    rate. Both antennae then receive the same total cell-Hz and the drive
    carries no side information by construction. The nominal rate is what
    the record and the transcript report as the drive; the per-cell scales
    are in FlyBody.describe()["side_scales"]. MEASURED after the change, in
    the room: the mean DNa02 R - L fell from +229 / +134 Hz (A / B, first
    run, uniform) to +88 / +52 Hz (50 world seconds, equalised, ground grey
    also changed), so the room's steering bias shrank but did not vanish.
    MEASURED in a control with NO lateral input (blank ground frame, smell
    68 Hz and sound 40 Hz nominal held constant, 150 windows per body,
    seeds 3 and 4): equalised, A +314 and B -313 Hz (se 22 each), equal and
    opposite; uniform, A +38 and B -12 Hz. Under constant symmetric input
    one DNa02 cell can ignite and stay on, on a side set by the body's seed
    and carried state, so the steering readout is dominated by the brain's
    own persistent dynamics; the remaining bias in the room is not
    explained by the antennal split, and no behavioural link between the
    delivered sound and the turning is established here.
  * Sound: fly X's song is the summed rate of its pulse-song wing motor
    neurons. The other fly's JO-A and JO-B cells (equalised per side as
    above) are driven at 200 Hz x clip(song / song_full, 0, 1) x clip(1 -
    d / 20 mm, 0, 1), where song_full is the summed rate when every song cell fires at the
    model's own ceiling, 1000 / 2.2 ms refractory = 454.5 Hz per cell: 8
    cells give 3,636 Hz. The first draft used 240 Hz (8 cells at a real
    fly's 30 pulses/s); the real-brain check measured summed rates of
    1,250-3,750 Hz under this simulator's ignition, so that reference was
    saturated on every step and carried no song information. A 12 ms window
    can realise up to 6 spikes per cell (500 Hz), so the clip can still
    engage. Sound is heard one world step late: both brains run in turn on
    one graph, so each hears the song the other sang in the previous window.
  * Brain window: 60 LIF steps (12 ms) per 50 ms world step, the brain's
    state (membrane potentials, refractory counters, rng) carried per fly, so
    the two flies alternate on one loaded graph with two state vectors. Gains
    are the roamer's calibration (calibration.CHOSEN). Learning is off: no
    dopamine is delivered anywhere in this file.
  * Motor: the roamer's mapping, unchanged, through plume_fly's
    motor_from_rates: turn = (DNa02_R - DNa02_L) / 450, forward = mean(DNa01)
    / 450, back = MDN / 450, stop = DNp09 / 450, speed = clip(forward - back)
    x (1 - clip(stop)). turn > 0 is a right turn.

Swapping the brain class (a GPU port with the same run() contract) is the
one line BRAIN_CLASS below; nothing else in this file knows which class it
holds.

The server (backrooms.py section 2) steps a World, at the end of this file:
the room behind the loop's flat per-fly record shape. World(seed) loads the
brain; World(seed, room=...) wraps a ready room, which is how the tests run
it on a fake brain.

  py -m pytest -q test_backrooms_world.py
  BACKROOMS_REAL_BRAIN=1 py -m pytest -q -s test_backrooms_world.py -k RealBrain
"""
import math
import subprocess
import time
from pathlib import Path

import numpy as np

import backrooms_dictionary as bd
from plume_fly import LIF_DT_MS, MOTOR_NAMES, MOTOR_SCALE_HZ, PlumeFly

ROOT = Path(__file__).parent
ANNOTATIONS = ROOT / "data" / "body-annotations.feather"

# ---- the brain: the one line to change for a drop-in replacement -----------
BRAIN_CLASS = "flysim.FlyBrain"          # e.g. "flysim_gpu.FlyBrainGPU"
MIN_FREE_RAM_GB = 6.0                    # a loaded brain is about 2.5 GB
SIM_STEPS = 60                           # LIF steps per world step: 12 ms

# ---- the room ---------------------------------------------------------------
ARENA_MM = 20.0
WORLD_DT_S = 0.05
MAX_SPEED_MM_S = 20.0
MAX_TURN_DEG_S = 360.0
START_MARGIN_MM = 2.0
FLY_NAMES = ("A", "B")

# ---- sight ---------------------------------------------------------------------
FRAME_W, FRAME_H = 1280, 800
EYE_FOV_W, EYE_FOV_H = 300, 210          # pumpui.FlyEye.look's window
EYE_MAX_HZ = 180.0                       # pumpui.FlyEye.look's L1 ceiling
GROUND_GREY = 0.5                        # mid-grey ground: L1 90 Hz, L2 54 Hz on it (was 0.05: L1 9 Hz, no contrast)
FLY_GREY = 0.0                           # the other fly: L1 0 Hz, L2 108 Hz
N_EYE_COLUMNS = 892                      # MEASURED: distinct sampling positions of the eye (1,767 L1 cells; the 875 left columns all coincide with right ones)
MIN_RADIUS_PX = 9.0                      # CHOSEN: one column spacing (column_spacing_px(892) = 9.0); build_room measures it from its eye
FRONT_FOV_DEG = 180.0                    # what the eye window spans horizontally
PX_PER_DEG = EYE_FOV_W / FRONT_FOV_DEG   # 1.667
BODY_LENGTH_MM = 2.5
BODY_WIDTH_MM = 1.0
BODY_HEIGHT_MM = 1.0
EYE_HEIGHT_MM = 1.0
MIN_DISTANCE_MM = 0.1                    # below this the geometry is undefined

# ---- smell and sound -------------------------------------------------------------
RANGE_MM = 20.0                          # both channels reach zero here
SMELL_MAX_HZ = 200.0
SOUND_MAX_HZ = 200.0
REFRACTORY_MS = 2.2                      # flysim.Params.refractory; Room reads the brain's own when it has one
SONG_CELLS_DEFAULT = 8                   # MEASURED: ps1, i1, iii1, b3 MN in this dataset


def column_spacing_px(n_columns=N_EYE_COLUMNS, fov=(EYE_FOV_W, EYE_FOV_H)):
    """
    Mean centre-to-centre spacing, in px, of n hex sampling positions spread
    over the eye window: sqrt(area per position x 2 / sqrt 3). With 892
    positions on 300 x 210 px that is 9.0 px; an ellipse smaller than this
    can fall between the eye's point samples and vanish, which is why the
    ellipse's semi-axes are floored at it.
    """
    n = max(int(n_columns), 1)
    return math.sqrt(float(fov[0]) * float(fov[1]) / n * 2.0 / math.sqrt(3.0))


def distinct_columns(eye):
    """
    The number of distinct sampling positions of an eye: its L1 (on) column
    coordinates counted once per position. The two optic lobes' columns
    coincide in this dataset, so pumpui.FlyEye's 1,767 L1 cells sample 892
    positions; an eye without `on_uv` (a fake) is taken at N_EYE_COLUMNS.
    """
    uv = getattr(eye, "on_uv", None)
    if uv is None:
        return N_EYE_COLUMNS
    pts = np.round(np.stack([np.asarray(uv[0], dtype=np.float64).ravel(),
                             np.asarray(uv[1], dtype=np.float64).ravel()], axis=1), 6)
    return int(np.unique(pts, axis=0).shape[0])


def song_full_hz(n_cells=SONG_CELLS_DEFAULT, refractory_ms=REFRACTORY_MS):
    """The summed song rate at which the listener's drive reaches its ceiling: every song cell at 1000 / refractory."""
    return float(n_cells) * 1000.0 / float(refractory_ms)


SONG_FULL_HZ = song_full_hz()            # 3,636.4 Hz for the 8 pulse-song cells
SONG_KEY = "song_pulse_mn"
SONG_FALLBACK_KEY = "wing_mn_all"
SMELL_KEY = "ORN_DA1"
SOUND_KEYS = ("JO_A", "JO_B")


# =============================================================================
# geometry helpers
# =============================================================================

def wrap_deg(a):
    """Angle in degrees folded into [-180, 180)."""
    return (float(a) + 180.0) % 360.0 - 180.0


def wrap_rad(a):
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi


class Fly:
    """Position in mm (x right, y up), heading in radians counterclockwise from +x."""

    __slots__ = ("name", "x", "y", "heading")

    def __init__(self, name, x, y, heading):
        self.name = str(name)
        self.x = float(x)
        self.y = float(y)
        self.heading = float(heading)

    def copy(self):
        return Fly(self.name, self.x, self.y, self.heading)

    def as_dict(self):
        return {"name": self.name, "x": self.x, "y": self.y,
                "heading_rad": self.heading, "heading_deg": math.degrees(self.heading)}


def distance_mm(a, b):
    return math.hypot(b.x - a.x, b.y - a.y)


def bearing_deg(viewer, other):
    """
    Where the other fly is relative to the viewer's heading, in [-180, 180):
    0 dead ahead, positive to the viewer's LEFT, negative to the right (the
    plume fly's sign convention for wind, kept so the two files agree).
    """
    los = math.atan2(other.y - viewer.y, other.x - viewer.x)
    return wrap_deg(math.degrees(los - viewer.heading))


def line_of_sight_rad(viewer, other):
    return math.atan2(other.y - viewer.y, other.x - viewer.x)


# =============================================================================
# the arena
# =============================================================================

class Arena:
    """
    Two flies in a square room. step() applies both commands at once from
    the same previous state, so neither fly moves first. Deterministic in
    `seed`: the seed places the flies, and after that the trajectory is a
    function of the commands alone.
    """

    def __init__(self, seed=0, size_mm=ARENA_MM, dt_s=WORLD_DT_S,
                 max_speed_mm_s=MAX_SPEED_MM_S, max_turn_deg_s=MAX_TURN_DEG_S,
                 start=None, names=FLY_NAMES):
        self.seed = int(seed)
        self.size = float(size_mm)
        self.dt = float(dt_s)
        self.max_speed = float(max_speed_mm_s)
        self.max_turn = math.radians(float(max_turn_deg_s))
        self.t = 0
        if len(names) != 2:
            raise ValueError("an arena holds exactly two flies")
        if start is None:
            rng = np.random.default_rng(self.seed)
            lo, hi = START_MARGIN_MM, self.size - START_MARGIN_MM
            xy = rng.uniform(lo, hi, size=(2, 2))
            hd = rng.uniform(-math.pi, math.pi, size=2)
            start = [(xy[i, 0], xy[i, 1], hd[i]) for i in range(2)]
        self.flies = [Fly(n, *s) for n, s in zip(names, start)]
        self.by_name = {f.name: f for f in self.flies}
        self._last_turn = {n: 0.0 for n in names}
        self._last_move = {n: 0.0 for n in names}

    # ---- helpers --------------------------------------------------------------

    @property
    def A(self):
        return self.flies[0]

    @property
    def B(self):
        return self.flies[1]

    def other(self, fly):
        return self.flies[1] if fly is self.flies[0] else self.flies[0]

    @property
    def time_s(self):
        return self.t * self.dt

    def distance(self):
        return distance_mm(self.flies[0], self.flies[1])

    def bearing(self, name):
        """Bearing of the other fly as seen by the fly called `name`, degrees."""
        f = self.by_name[name]
        return bearing_deg(f, self.other(f))

    def geometry(self):
        """Everything the recorder and the page need about where the flies are."""
        a, b = self.flies
        return {
            "t": self.t, "time_s": self.time_s,
            "A": a.as_dict(), "B": b.as_dict(),
            "distance_mm": self.distance(),
            "bearing_deg": {"A": bearing_deg(a, b), "B": bearing_deg(b, a)},
            "turned_deg": {n: math.degrees(v) for n, v in self._last_turn.items()},
            "moved_mm": dict(self._last_move),
        }

    # ---- kinematics ------------------------------------------------------------

    def apply(self, fly, turn, speed):
        """One fly's step: turn (right is positive), then walk along the new heading, then the walls."""
        turn = float(np.clip(turn, -1.0, 1.0))
        speed = float(np.clip(speed, -1.0, 1.0))
        d_heading = -turn * self.max_turn * self.dt        # right turn = clockwise
        fly.heading = wrap_rad(fly.heading + d_heading)
        move = speed * self.max_speed * self.dt
        fly.x = float(np.clip(fly.x + move * math.cos(fly.heading), 0.0, self.size))
        fly.y = float(np.clip(fly.y + move * math.sin(fly.heading), 0.0, self.size))
        self._last_turn[fly.name] = d_heading
        self._last_move[fly.name] = move

    def step(self, cmd_a, cmd_b):
        """cmd = (turn, speed) per fly, both applied from the same state."""
        self.apply(self.flies[0], *cmd_a)
        self.apply(self.flies[1], *cmd_b)
        self.t += 1
        return self.geometry()

    def describe(self):
        return {
            "size_mm": self.size, "dt_s": self.dt, "seed": self.seed,
            "max_speed_mm_s": self.max_speed, "max_turn_deg_s": math.degrees(self.max_turn),
            "max_move_per_step_mm": self.max_speed * self.dt,
            "max_turn_per_step_deg": math.degrees(self.max_turn * self.dt),
            "start_margin_mm": START_MARGIN_MM,
            "order": "turn, then walk along the new heading, then clamp to the walls; "
                     "both flies from the same previous state",
            "conventions": "x right, y up, heading counterclockwise from +x; bearing positive "
                           "= the other fly is to the viewer's left; turn > 0 = right turn",
        }


# =============================================================================
# the channels
# =============================================================================

class Channels:
    """
    What each fly gets of the other: a frame for its eye, a rate for its
    cVA receptor neurons, a rate for its sound-sensitive Johnston's organ
    neurons. All three are pure functions of the geometry (and, for sound,
    the other's song rate).
    """

    def __init__(self, frame_w=FRAME_W, frame_h=FRAME_H, ground=GROUND_GREY,
                 fly_grey=FLY_GREY, px_per_deg=PX_PER_DEG, eye_height_mm=EYE_HEIGHT_MM,
                 body_mm=(BODY_LENGTH_MM, BODY_WIDTH_MM, BODY_HEIGHT_MM),
                 range_mm=RANGE_MM, smell_max_hz=SMELL_MAX_HZ, sound_max_hz=SOUND_MAX_HZ,
                 song_full_hz=SONG_FULL_HZ, fov=(EYE_FOV_W, EYE_FOV_H),
                 min_radius_px=MIN_RADIUS_PX):
        self.w, self.h = int(frame_w), int(frame_h)
        self.ground = float(ground)
        self.fly_grey = float(fly_grey)
        self.min_radius = float(min_radius_px)
        self.px_per_deg = float(px_per_deg)
        self.eye_height = float(eye_height_mm)
        self.body_length, self.body_width, self.body_height = (float(v) for v in body_mm)
        self.range = float(range_mm)
        self.smell_max = float(smell_max_hz)
        self.sound_max = float(sound_max_hz)
        self.song_full = float(song_full_hz)
        self.fov_w, self.fov_h = (int(v) for v in fov)
        self.gaze = (self.w / 2.0, self.h / 2.0)     # where FlyEye.look is pointed

    # ---- sight ---------------------------------------------------------------

    def ellipse_of(self, viewer, other):
        """
        The other fly on the viewer's frame: centre (cx, cy) and semi-axes
        (rx, ry) in pixels, plus whether any of it falls inside the eye
        window. Horizontal position is the bearing on an equirectangular
        strip; vertical position and both radii come from the ground-plane
        geometry in the module docstring. Each semi-axis is floored at
        min_radius (one column spacing); the unfloored values are returned
        as rx_raw, ry_raw so the record shows when the floor was in force.
        """
        d = max(distance_mm(viewer, other), MIN_DISTANCE_MM)
        b = bearing_deg(viewer, other)
        psi = other.heading - line_of_sight_rad(viewer, other)
        half_ext = math.hypot(0.5 * self.body_length * math.sin(psi),
                              0.5 * self.body_width * math.cos(psi))
        rx_raw = self.px_per_deg * math.degrees(math.atan(half_ext / d))
        ry_raw = self.px_per_deg * math.degrees(math.atan(0.5 * self.body_height / d))
        rx, ry = max(rx_raw, self.min_radius), max(ry_raw, self.min_radius)
        drop = math.degrees(math.atan((self.eye_height - 0.5 * self.body_height) / d))
        cx = self.gaze[0] - b * self.px_per_deg
        cy = self.gaze[1] + drop * self.px_per_deg
        x0, x1 = self.gaze[0] - self.fov_w / 2.0, self.gaze[0] + self.fov_w / 2.0
        y0, y1 = self.gaze[1] - self.fov_h / 2.0, self.gaze[1] + self.fov_h / 2.0
        in_window = (cx + rx > x0) and (cx - rx < x1) and (cy + ry > y0) and (cy - ry < y1)
        return {"cx": cx, "cy": cy, "rx": rx, "ry": ry, "rx_raw": rx_raw, "ry_raw": ry_raw,
                "bearing_deg": b, "distance_mm": d, "in_window": bool(in_window)}

    def sight_frame(self, viewer, other):
        """The viewer's 1280 x 800 grey frame: the ground with the other fly on it."""
        img = np.full((self.h, self.w), self.ground, dtype=np.float32)
        self.draw_ellipse(img, self.ellipse_of(viewer, other))
        return img

    def draw_ellipse(self, img, e):
        cx, cy, rx, ry = e["cx"], e["cy"], max(e["rx"], 0.5), max(e["ry"], 0.5)
        x0 = max(int(math.floor(cx - rx)), 0)
        x1 = min(int(math.ceil(cx + rx)) + 1, self.w)
        y0 = max(int(math.floor(cy - ry)), 0)
        y1 = min(int(math.ceil(cy + ry)) + 1, self.h)
        if x1 <= x0 or y1 <= y0:
            return 0
        yy, xx = np.ogrid[y0:y1, x0:x1]
        inside = ((xx + 0.5 - cx) / rx) ** 2 + ((yy + 0.5 - cy) / ry) ** 2 <= 1.0
        img[y0:y1, x0:x1][inside] = self.fly_grey
        return int(inside.sum())

    # ---- smell and sound ---------------------------------------------------------

    def falloff(self, d_mm):
        """clip(1 - d / range, 0, 1): one at contact, zero at the range and beyond."""
        return float(np.clip(1.0 - float(d_mm) / self.range, 0.0, 1.0))

    def smell_hz(self, d_mm):
        """ORN_DA1 drive from the other male's cVA at distance d."""
        return self.smell_max * self.falloff(d_mm)

    def sound_hz(self, song_hz, d_mm):
        """JO-A/JO-B drive from the other's song (summed pulse-MN rate) at distance d."""
        loud = float(np.clip(float(song_hz) / self.song_full, 0.0, 1.0))
        return self.sound_max * loud * self.falloff(d_mm)

    def describe(self):
        return {
            "frame": [self.w, self.h], "eye_window": [self.fov_w, self.fov_h],
            "gaze": list(self.gaze), "ground_grey": self.ground, "fly_grey": self.fly_grey,
            "front_fov_deg": self.fov_w / self.px_per_deg, "px_per_deg": self.px_per_deg,
            "eye_height_mm": self.eye_height,
            "body_mm": [self.body_length, self.body_width, self.body_height],
            "min_distance_mm": MIN_DISTANCE_MM,
            "range_mm": self.range, "smell_max_hz": self.smell_max,
            "sound_max_hz": self.sound_max, "song_full_hz": self.song_full,
            "min_radius_px": self.min_radius,
            "smell": "ORN_DA1 at smell_max_hz x clip(1 - d / range_mm, 0, 1), the nominal per-cell "
                     "rate; each cell is then scaled by its side's count (bodies.*.side_scales) so "
                     "both antennae receive the same total cell-Hz",
            "sound": "JO-A and JO-B at sound_max_hz x clip(song / song_full_hz, 0, 1) x clip(1 - d / "
                     "range_mm, 0, 1), the nominal per-cell rate, scaled per side like smell; the "
                     "song is the other's previous window",
            "sight": "ground at ground_grey, the other fly an ellipse at fly_grey; x from the bearing "
                     "(equirectangular, viewer's right = image right), y and size from ground-plane "
                     "geometry, each semi-axis floored at min_radius_px (one column spacing); the eye "
                     "window spans front_fov_deg, so a fly behind is not seen",
            "sampling": "pumpui.FlyEye.look samples one pixel per lamina column; both optic lobes "
                        "carry the same hex coordinates in this dataset (MEASURED: 875 of 875 left "
                        "L1 columns coincide with right ones) and are normalised together, so the "
                        "two retinas sample the same window and there is no left-eye versus "
                        "right-eye difference",
            "retinal_contrast_hz": {
                "on_ground": self.ground * EYE_MAX_HZ, "on_fly": self.fly_grey * EYE_MAX_HZ,
                "off_ground": (1.0 - self.ground) * EYE_MAX_HZ * 0.6,
                "off_fly": (1.0 - self.fly_grey) * EYE_MAX_HZ * 0.6,
                "note": "pumpui.FlyEye.look: L1 at lum x 180 Hz, L2 at (1 - lum) x 108 Hz",
            },
        }


# =============================================================================
# the brain side
# =============================================================================

def free_ram_gb():
    """Free physical memory in GB, from PowerShell; nan if it cannot be read."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
            capture_output=True, text=True, timeout=30).stdout.strip()
        return float(out) / (1024 * 1024)
    except Exception:
        return float("nan")


def brain_class(name=None):
    """Resolve 'module.Class' to the class; BRAIN_CLASS by default."""
    import importlib
    mod, _, cls = (name or BRAIN_CLASS).rpartition(".")
    return getattr(importlib.import_module(mod), cls)


def load_brain(min_free_gb=MIN_FREE_RAM_GB, cls=None, say=print):
    """
    The one brain this process holds. Refuses when free RAM is below
    min_free_gb (two other workflows on this machine hold a brain at times),
    so the check is not left to the caller.
    """
    ram = free_ram_gb()
    say(f"free RAM: {ram:.1f} GB (need > {min_free_gb:.1f})")
    if not ram > min_free_gb:
        raise MemoryError(f"free RAM {ram:.1f} GB is not above {min_free_gb:.1f} GB; not loading a brain")
    t0 = time.time()
    fb = brain_class(cls)()
    say(f"brain ready: {fb.n:,} neurons in {time.time() - t0:.1f} s ({cls or BRAIN_CLASS})")
    return fb


def room_gains(fb):
    """The roamer's calibration, so this is the same animal that roams."""
    import calibration
    return calibration.gains_for(fb, calibration.CHOSEN)


def soma_sides(fb, path=ANNOTATIONS):
    """Per-neuron somaSide ('L', 'R' or '') aligned with fb's neuron order."""
    import pandas as pd
    a = pd.read_feather(path, columns=["bodyId", "somaSide"])
    a = a.drop_duplicates("bodyId").set_index("bodyId")
    return a["somaSide"].reindex(fb.bodies).fillna("").to_numpy().astype(str)


def root_sides(fb, path=ANNOTATIONS):
    """
    Per-neuron rootSide ('L', 'R' or '') aligned with fb's neuron order: the
    side a sensory cell's root enters by. Antennal cells (ORN_DA1, JO-A,
    JO-B) have no somaSide, so this is the side that says which antenna a
    driven cell belongs to.
    """
    import pandas as pd
    a = pd.read_feather(path, columns=["bodyId", "rootSide"])
    a = a.drop_duplicates("bodyId").set_index("bodyId")
    s = a["rootSide"].reindex(fb.bodies).fillna("").to_numpy().astype(str)
    s[(s != "L") & (s != "R")] = ""
    return s


def per_side_scales(idx, sides):
    """
    Per-cell multipliers over the cells `idx` that make both sides receive
    the same total cell-Hz at one nominal rate: mean count per side over the
    side's own count (plume_fly.side_scales), 1.0 for cells with no side.
    Returns (scale array aligned with idx, summary dict). With sides None
    every scale is 1 and the summary says so.
    """
    idx = np.asarray(idx, dtype=np.int64)
    scale = np.ones(idx.size, dtype=np.float64)
    if sides is None:
        return scale, {"equalised": False, "n": int(idx.size), "n_L": 0, "n_R": 0,
                       "n_other": int(idx.size), "scale_L": 1.0, "scale_R": 1.0,
                       "cell_hz_R_over_L_uniform": None, "cell_hz_R_over_L_now": None}
    s = np.asarray(sides, dtype=str)[idx]
    left, right = s == "L", s == "R"
    n_l, n_r = int(left.sum()), int(right.sum())
    from plume_fly import side_scales
    k_l, k_r = side_scales(n_l, n_r, equalise=True)
    scale[left], scale[right] = k_l, k_r
    return scale, {"equalised": True, "n": int(idx.size), "n_L": n_l, "n_R": n_r,
                   "n_other": int(idx.size - n_l - n_r), "scale_L": float(k_l), "scale_R": float(k_r),
                   "cell_hz_R_over_L_uniform": (n_r / n_l if n_l else None),
                   "cell_hz_R_over_L_now": ((n_r * k_r) / (n_l * k_l) if n_l and k_l else None)}


def motor_groups(fb, side):
    """
    The roamer's six walking groups, selected exactly as pumpui.FlyPilot.motor
    selects them (exact type name, then somaSide for the paired ones). Done
    here rather than through FlyPilot so the annotations path is this file's
    absolute one and a fake brain can hand in its own sides.
    """
    side = np.asarray(side, dtype=str)

    def dn(t, s=None):
        sel = np.asarray(fb.where(type_re=rf"^{t}$"), dtype=np.int64)
        if s:
            sel = sel[side[sel] == s]
        return sel

    return {"steer_L": dn("DNa02", "L"), "steer_R": dn("DNa02", "R"),
            "fwd_L": dn("DNa01", "L"), "fwd_R": dn("DNa01", "R"),
            "back": dn("MDN"), "stop": dn("DNp09")}


class FlyBody:
    """
    One fly's brain state on the shared brain.

    step(frame, smell_hz, sound_hz) builds the drive (eye + ORN_DA1 + JO-A/B),
    runs sim_steps LIF steps continuing from this fly's own state, and
    returns the record: per-group mean rate and spike count for every
    dictionary group and the six motor groups, the inputs as delivered, the
    song rate (summed over the song group), and the roamer's turn and speed.

    `groups` is backrooms_dictionary.present_groups(fb) or any {key: index
    array}; `motor` is motor_groups(); `eye` is anything with look(img, cx,
    cy) returning a FlyBrain drive dict (pumpui.FlyEye, or a fake); `sides`
    is root_sides(fb) (per-neuron 'L' / 'R' / ''), which equalises the
    smell and sound drive per antenna; None delivers one uniform rate and
    describe() says so.
    """

    def __init__(self, name, fb, eye, groups, motor, gains=None, sim_steps=SIM_STEPS,
                 seed=0, gaze=(FRAME_W / 2.0, FRAME_H / 2.0), sides=None):
        self.name = str(name)
        self.fb = fb
        self.eye = eye
        self.gains = gains
        self.sim_steps = int(sim_steps)
        self.seed = int(seed)
        self.gaze = (float(gaze[0]), float(gaze[1]))
        self.state = None
        self.windows = 0
        self.secs = self.sim_steps * LIF_DT_MS / 1000.0
        self.sides = None if sides is None else np.asarray(sides, dtype=str)

        self.groups = {k: np.asarray(v, dtype=np.int64) for k, v in groups.items() if len(v)}
        self.motor = {k: np.asarray(motor[k], dtype=np.int64) for k in MOTOR_NAMES}
        missing = [k for k in MOTOR_NAMES if k not in motor]
        if missing:
            raise KeyError(f"motor groups missing: {missing}")
        if SMELL_KEY not in self.groups:
            raise KeyError(f"no {SMELL_KEY} cells: the smell channel has nowhere to go")
        sound_parts = [(k, self.groups[k]) for k in SOUND_KEYS if k in self.groups]
        if not sound_parts:
            raise KeyError(f"no {'/'.join(SOUND_KEYS)} cells: the sound channel has nowhere to go")
        self.smell_idx = self.groups[SMELL_KEY]
        # per-side equalisation, per driven group (JO-A and JO-B separately,
        # since they are distinct classes with their own L/R splits)
        self.side_scales = {}
        self.smell_scale, self.side_scales[SMELL_KEY] = per_side_scales(self.smell_idx, self.sides)
        cat_idx, cat_scale = [], []
        for k, v in sound_parts:
            sc, self.side_scales[k] = per_side_scales(v, self.sides)
            cat_idx.append(v)
            cat_scale.append(sc)
        cat_idx, cat_scale = np.concatenate(cat_idx), np.concatenate(cat_scale)
        self.sound_idx, first = np.unique(cat_idx, return_index=True)
        self.sound_scale = cat_scale[first]
        if SONG_KEY in self.groups:
            self.song_key = SONG_KEY
        elif SONG_FALLBACK_KEY in self.groups:
            self.song_key = SONG_FALLBACK_KEY
        else:
            raise KeyError(f"neither {SONG_KEY} nor {SONG_FALLBACK_KEY} present: no song to read")

        # the eye must not share a cell with what smell and sound drive, or
        # FlyBrain.run would carry two rates for one cell
        eye_idx = np.concatenate([np.asarray(getattr(eye, "on_idx", []), dtype=np.int64),
                                  np.asarray(getattr(eye, "off_idx", []), dtype=np.int64)])
        clash = np.intersect1d(eye_idx, np.concatenate([self.smell_idx, self.sound_idx]))
        if clash.size:
            raise ValueError(f"eye cells overlap smell/sound cells: {clash[:5].tolist()}")

        # one recorded array for everything, split afterwards: run() counts
        # spikes per recorded cell, so recording the union once and slicing is
        # the same numbers at a fraction of the per-step cost
        everything = [self.groups[k] for k in self.groups] + [self.motor[k] for k in MOTOR_NAMES]
        self.rec_idx = np.unique(np.concatenate(everything))
        self.rec_pos = {k: np.searchsorted(self.rec_idx, v) for k, v in self.groups.items()}
        self.rec_pos.update({k: np.searchsorted(self.rec_idx, self.motor[k]) for k in MOTOR_NAMES})

    @property
    def state_carried(self):
        return self.state is not None

    def reset(self, seed=None):
        self.state = None
        self.windows = 0
        if seed is not None:
            self.seed = int(seed)

    def drive(self, frame, smell_hz, sound_hz):
        """
        The full FlyBrain drive dict: eye first, then smell and sound as
        per-cell arrays, the nominal rate times each cell's side scale.
        """
        d = dict(self.eye.look(frame, self.gaze[0], self.gaze[1]))
        for idx, scale, hz in ((self.smell_idx, self.smell_scale, smell_hz),
                               (self.sound_idx, self.sound_scale, sound_hz)):
            key = tuple(idx.tolist())
            if key in d:
                raise ValueError("smell/sound drive overlaps the eye's own input")
            d[key] = (scale * float(max(0.0, hz))).astype(np.float32)
        return d

    def step(self, frame, smell_hz, sound_hz):
        t0 = time.time()
        drive = self.drive(frame, smell_hz, sound_hz)
        r = self.fb.run(drive, steps=self.sim_steps, gains=self.gains,
                        record={"all": self.rec_idx}, seed=self.seed, state=self.state)
        carried = self.state is not None
        self.state = r["_state"]          # a brain that cannot carry state is an error, not a restart
        self.windows += 1
        per_cell = np.asarray(r["all"], dtype=np.float64)

        rates, counts = {}, {}
        for k, pos in self.rec_pos.items():
            v = per_cell[pos]
            rates[k] = float(v.mean()) if v.size else 0.0
            counts[k] = int(np.rint(v.sum() * self.secs))
        motor = {k: rates[k] for k in MOTOR_NAMES}
        turn, speed, parts = PlumeFly.motor_from_rates(motor)
        song_hz = float(per_cell[self.rec_pos[self.song_key]].sum())

        vals = list(drive.values())
        on = np.asarray(vals[0], dtype=np.float64) if len(vals) > 0 else np.zeros(1)
        off = np.asarray(vals[1], dtype=np.float64) if len(vals) > 1 else np.zeros(1)
        fired = r.get("_fired")
        return {
            "fly": self.name, "window": self.windows, "state_carried": bool(carried),
            "turn": float(turn), "speed": float(speed), **parts,
            "motor": motor,
            "song_hz": song_hz, "song_group": self.song_key,
            "song_cells": int(self.rec_pos[self.song_key].size),
            "in": {"smell_hz": float(max(0.0, smell_hz)), "sound_hz": float(max(0.0, sound_hz)),
                   "eye_on_hz": float(on.mean()), "eye_off_hz": float(off.mean())},
            "out": {SMELL_KEY: rates[SMELL_KEY],
                    **{k: rates[k] for k in SOUND_KEYS if k in rates}},
            "rates": rates, "counts": counts,
            "fired": int(len(fired)) if fired is not None else 0,
            "brain_s": time.time() - t0,
        }

    def describe(self):
        return {
            "name": self.name, "seed": self.seed, "sim_steps": self.sim_steps,
            "lif_dt_ms": LIF_DT_MS, "window_ms": self.sim_steps * LIF_DT_MS,
            "rate_quantum_hz": 1000.0 / (self.sim_steps * LIF_DT_MS),
            "motor_scale_hz": MOTOR_SCALE_HZ,
            "song_group": self.song_key, "song_cells": int(self.rec_pos[self.song_key].size),
            "smell_cells": int(self.smell_idx.size), "sound_cells": int(self.sound_idx.size),
            "recorded_cells": int(self.rec_idx.size),
            "groups": {k: int(v.size) for k, v in self.groups.items()},
            "motor_cells": {k: int(v.size) for k, v in self.motor.items()},
            "gains": "custom" if self.gains is not None else "stock",
            "learning": "off: no dopamine, no synapse changed",
            "side_scales": self.side_scales,
            "drive_note": "smell and sound: each cell's rate = the nominal Hz recorded in 'in' x "
                          "that cell's side scale (mean count per side / its side's count, by "
                          "rootSide; unsided cells x 1), so both antennae receive the same total "
                          "cell-Hz and the drive carries no side information by construction",
        }


# =============================================================================
# the room: arena + channels + two bodies on one brain
# =============================================================================

class Room:
    """
    One world step: render each fly's frame, compute smell and sound from the
    geometry and the other's previous song, run A's window then B's on the
    shared brain, apply both commands to the arena at once. Everything is
    returned as plain numbers so it can be logged as it is.
    """

    def __init__(self, fb, eye, groups, motor, gains=None, seed=0, sim_steps=SIM_STEPS,
                 arena=None, channels=None, sides=None, min_radius_px=None):
        self.fb = fb
        self.arena = arena or Arena(seed=seed)
        self.bodies = {
            "A": FlyBody("A", fb, eye, groups, motor, gains, sim_steps, seed=seed * 2 + 1, sides=sides),
            "B": FlyBody("B", fb, eye, groups, motor, gains, sim_steps, seed=seed * 2 + 2, sides=sides),
        }
        # the song reference follows the song group actually present and the
        # brain's own refractory period, so a fallback group or another brain
        # class keeps the listener's scale honest
        refractory = getattr(getattr(fb, "p", None), "refractory", REFRACTORY_MS)
        self.song_full_hz = song_full_hz(self.bodies["A"].rec_pos[self.bodies["A"].song_key].size, refractory)
        if channels is None:
            kw = {} if min_radius_px is None else {"min_radius_px": float(min_radius_px)}
            channels = Channels(song_full_hz=self.song_full_hz, **kw)
        self.channels = channels
        self.song = {"A": 0.0, "B": 0.0}
        self.seed = int(seed)

    def step(self):
        t0 = time.time()
        a, b = self.arena.A, self.arena.B
        before = self.arena.geometry()
        d = self.arena.distance()
        smell = self.channels.smell_hz(d)
        heard = {"A": self.channels.sound_hz(self.song["B"], d),
                 "B": self.channels.sound_hz(self.song["A"], d)}
        seen = {"A": self.channels.ellipse_of(a, b), "B": self.channels.ellipse_of(b, a)}
        rec = {}
        for name, me, other in (("A", a, b), ("B", b, a)):
            frame = self.channels.sight_frame(me, other)
            rec[name] = self.bodies[name].step(frame, smell, heard[name])
        self.song = {n: rec[n]["song_hz"] for n in rec}
        after = self.arena.step((rec["A"]["turn"], rec["A"]["speed"]),
                                (rec["B"]["turn"], rec["B"]["speed"]))
        return {"t": before["t"], "time_s": before["time_s"], "before": before, "after": after,
                "smell_hz": smell, "sound_hz": heard, "seen": seen,
                "A": rec["A"], "B": rec["B"], "step_s": time.time() - t0}

    def sight_check(self, distances=(2.0, 3.0, 5.0, 10.0, 20.0)):
        """
        MEASURED through the eye this room has: with the other fly dead
        ahead at each distance, side-on and head-on, how many L1 and L2
        columns sample the ellipse rather than the ground, and the mean L1 /
        L2 drive with the fly in the window against a ground-only frame.
        This is the retinal signal the sight channel actually delivers, so a
        reader can see whether it is more than nothing (at the first draft's
        ground grey it was 0.2 Hz on the mean, i.e. nothing).
        """
        eye, ch = self.bodies["A"].eye, self.channels
        viewer = Fly("viewer", 10.0, 10.0, 0.0)
        blank = np.full((ch.h, ch.w), ch.ground, dtype=np.float32)
        v0 = list(eye.look(blank, ch.gaze[0], ch.gaze[1]).values())
        on0 = np.asarray(v0[0], dtype=np.float64).ravel()
        off0 = np.asarray(v0[1], dtype=np.float64).ravel() if len(v0) > 1 else np.zeros(1)
        cases = []
        for d in distances:
            for pose, hd in (("side_on", math.pi / 2.0), ("head_on", math.pi)):
                other = Fly("other", 10.0 + float(d), 10.0, hd)
                e = ch.ellipse_of(viewer, other)
                v = list(eye.look(ch.sight_frame(viewer, other), ch.gaze[0], ch.gaze[1]).values())
                on = np.asarray(v[0], dtype=np.float64).ravel()
                off = np.asarray(v[1], dtype=np.float64).ravel() if len(v) > 1 else np.zeros(1)
                cases.append({
                    "distance_mm": float(d), "pose": pose, "in_window": bool(e["in_window"]),
                    "rx_px": float(e["rx"]), "ry_px": float(e["ry"]),
                    "rx_raw_px": float(e["rx_raw"]), "ry_raw_px": float(e["ry_raw"]),
                    "L1_columns_on_fly": int((on < on0 - 1e-6).sum()),
                    "L2_columns_on_fly": int((off > off0 + 1e-6).sum()),
                    "L1_mean_hz": float(on.mean()), "L2_mean_hz": float(off.mean()),
                    "L1_mean_minus_ground_hz": float(on.mean() - on0.mean()),
                    "L2_mean_minus_ground_hz": float(off.mean() - off0.mean()),
                })
        return {"what": "the other fly dead ahead, through this room's eye: columns it covers and "
                        "the mean retinal drive against a ground-only frame (measured)",
                "ground": {"L1_hz": float(on0.mean()), "L2_hz": float(off0.mean()),
                           "L1_columns": int(on0.size), "L2_columns": int(off0.size)},
                "cases": cases}

    def describe(self):
        return {
            "brain_class": BRAIN_CLASS, "seed": self.seed,
            "arena": self.arena.describe(), "channels": self.channels.describe(),
            "bodies": {n: b.describe() for n, b in self.bodies.items()},
            "sight_measured": self.sight_check(),
            "timing": {"world_step_s": self.arena.dt,
                       "brain_ms_per_world_step": self.bodies["A"].sim_steps * LIF_DT_MS,
                       "brain_under_run_factor": self.arena.dt * 1000.0 / (self.bodies["A"].sim_steps * LIF_DT_MS),
                       "note": "12 ms of brain per 50 ms of world, so on the brains' own clock the "
                               "room moves 4.2x faster than life; the two flies run in turn on one "
                               "loaded graph, each carrying its own state; wall time per step is a "
                               "separate, measured ratio (the loop's slowdown)"},
            "sound_delay_steps": 1,
            "song_full_hz": self.song_full_hz,
            "song_full_note": "song cells x 1000 / refractory_ms: the summed rate with every song "
                              "cell at the model's ceiling; the listener's drive is proportional "
                              "to song / song_full, clipped at 1",
        }


def build_room(fb, gains=None, seed=0, annotations_path=ANNOTATIONS, sim_steps=SIM_STEPS):
    """The real thing: the eye from pumpui, the groups from the dictionary, the roamer's motor split."""
    try:
        import pumpui
    except ImportError:                     # the public copy calls it flyeye
        import flyeye as pumpui
    eye = pumpui.FlyEye(fb, annotations_path=str(annotations_path))
    groups = bd.present_groups(fb)
    motor = motor_groups(fb, soma_sides(fb, annotations_path))
    sides = root_sides(fb, annotations_path)
    # the ellipse floor is measured from the eye this room has: one spacing
    # of its distinct sampling positions (892 for pumpui.FlyEye, 9.0 px)
    min_radius = column_spacing_px(distinct_columns(eye))
    return Room(fb, eye, groups, motor, gains, seed=seed, sim_steps=sim_steps,
                sides=sides, min_radius_px=min_radius)


# =============================================================================
# what the server drives: the room behind backrooms.Loop's record shape
# =============================================================================

WORLD_CLASS = "backrooms_world.World"
DRIVE_KEYS = (SMELL_KEY,) + SOUND_KEYS          # the loop's drive dict: ORN_DA1, JO_A, JO_B


class World:
    """
    The object backrooms.Loop steps (section 2 of backrooms.py names it
    backrooms_world.World). It holds the one brain (loaded here, after the RAM
    check in load_brain, unless a ready room is handed in, which is how the
    tests run it without a graph) and turns Room.step()'s nested record into
    the flat per-fly record the loop, the captioner and the page read:

        {"step": int, "t": s, "flies": {"A": FLY, "B": FLY}, ...extras}
        FLY = {"x", "y": mm, "heading": deg ccw from +x,
               "speed": mm/s walked along the heading this step (negative
                        when backing; the command's move, before the walls),
               "turn": deg/s, ccw positive (so a right turn is negative),
               "distance": mm, "bearing": deg of the other fly, + = left,
               "rates", "counts": every present dictionary group,
               "drive": {"ORN_DA1", "JO_A", "JO_B"} nominal Hz delivered this
                        window (per-cell rates are scaled per side, see
                        FlyBody.describe()["side_scales"]),
               "motor": the six walking groups' rates, "song": summed Hz}

    Positions are those AFTER the step: the commands this step's brain
    windows produced, applied to the arena. So the velocity the captioner
    takes from successive positions is the motion the recorded rates caused,
    and a "turns toward" line names the window whose DNa02 rates turned it.
    Extra keys (the normalised command, the eye's own rates, the sensory
    output rates, cells fired, timing, the ellipse each fly was shown) pass
    through to the step log untouched; the loop converts numpy itself.
    """

    def __init__(self, seed=0, room=None, say=print):
        if room is None:
            fb = load_brain(say=say)
            room = build_room(fb, room_gains(fb), seed=seed)
        self.room = room
        self.fb = room.fb
        self.seed = int(seed)
        self.dt = float(room.arena.dt)
        # the present dictionary groups on this brain, {key: index array};
        # FlyBody keeps exactly present_groups(fb) (non-empty only)
        self.groups = {k: v for k, v in room.bodies["A"].groups.items()}
        self.step_n = 0

    def meta(self):
        """What the room chose, for /state: Room.describe() plus this adapter's conventions."""
        d = self.room.describe()
        d["world_class"] = WORLD_CLASS
        d["record"] = {
            "positions": "after the step: this window's commands applied",
            "speed": "mm/s along the heading, the command's move before the walls; negative = backing",
            "turn": "deg/s, counter-clockwise positive; a right turn is negative",
            "bearing": "deg, positive = the other fly is to this fly's left",
            "drive": "nominal Hz per cell delivered to ORN_DA1 (smell) and to JO_A and JO_B (sound) "
                     "this window, before the per-side scaling in bodies.*.side_scales",
            "rates": "mean Hz per cell over the 12 ms window, per present dictionary group",
            "counts": "spikes in the window, summed over the group",
        }
        return d

    def _fly(self, name, rec, after):
        g = after[name]
        return {
            "x": float(g["x"]), "y": float(g["y"]), "heading": float(g["heading_deg"]),
            "speed": float(after["moved_mm"][name]) / self.dt,
            "turn": float(after["turned_deg"][name]) / self.dt,
            "distance": float(after["distance_mm"]),
            "bearing": float(after["bearing_deg"][name]),
            "rates": {k: float(rec["rates"][k]) for k in self.groups},
            "counts": {k: int(rec["counts"][k]) for k in self.groups},
            "drive": {SMELL_KEY: float(rec["in"]["smell_hz"]),
                      **{k: float(rec["in"]["sound_hz"]) for k in SOUND_KEYS}},
            "motor": {k: float(v) for k, v in rec["motor"].items()},
            "song": float(rec["song_hz"]),
            # extras, for the step log
            "cmd": {"turn": float(rec["turn"]), "speed": float(rec["speed"])},
            "eye_hz": {"on": float(rec["in"]["eye_on_hz"]), "off": float(rec["in"]["eye_off_hz"])},
            "out": {k: float(v) for k, v in rec["out"].items()},
            "song_group": rec["song_group"], "song_cells": int(rec["song_cells"]),
            "fired": int(rec["fired"]), "window": int(rec["window"]),
            "state_carried": bool(rec["state_carried"]), "brain_s": float(rec["brain_s"]),
        }

    def step(self):
        r = self.room.step()
        after = r["after"]
        self.step_n = int(after["t"])
        return {
            "step": self.step_n,
            "t": float(after["time_s"]),
            "flies": {n: self._fly(n, r[n], after) for n in FLY_NAMES},
            "smell_hz": float(r["smell_hz"]),
            "sound_hz": {n: float(v) for n, v in r["sound_hz"].items()},
            "seen": r["seen"],
            "step_s": float(r["step_s"]),
        }
