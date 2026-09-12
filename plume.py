"""
A wind tunnel with an odour plume in it, for a walking fly. Pure numpy, no brain.

This is the world half of the plume-tracking experiment (after Alvarez-Salvado
et al. 2018 eLife and van Breugel and Dickinson 2014 Curr Biol): a rectangular
arena, a uniform wind, a Gaussian-puff plume that meanders so the odour comes
and goes, and a point fly that walks and turns. The brain never sees this
module directly; the coupling layer reads concentration and wind direction out
of a World, drives the connectome, and feeds the motor rates back into step().

MEASURED (from the literature, not tuned here)
  * Walking flies track plumes by surging upwind on an odour encounter and
    casting crosswind after losing it (Alvarez-Salvado et al. 2018), and wind
    direction is read as the difference between the two antennae (Suver et al.
    2019). The world only has to make encounters and losses happen.

CHOSEN, every one of them named in CONSTANTS below so a report can print them
  * Arena 0.6 m by 0.3 m, wind 0.12 m/s toward +x, source at (0.05, 0.15) so
    upwind is -x. Walls clamp the fly and count the contacts.
  * Puffs are released from the source every 0.05 s, advected downwind at the
    wind speed, and spread as r(t) = sqrt(r0^2 + 2 D t) with D set so the plume
    is 0.06 m wide (two standard deviations) 0.3 m downwind.
  * Meander: one Ornstein-Uhlenbeck crosswind velocity (sd 0.02 m/s, time
    constant 1 s) lives at the release point. Each puff is launched with the
    crosswind velocity in force at its release and keeps it, so consecutive
    puffs fan out in correlated directions and the plume snakes downwind. A
    spatially uniform gust that moves every airborne puff at once would be the
    other common choice; this one was picked because the fan makes the plume
    genuinely intermittent at a fixed point 0.3 m downwind.
  * Concentration is the sum over puffs of q / (2 pi r^2) exp(-d^2 / (2 r^2)),
    with q calibrated once, numerically, so a straight (meander-free) plume
    reads 1.0 on the centreline 0.1 m downwind. The odour drive uses
    clip(c, 0, 1), so the ORNs saturate near the source.
  * The plume is warmed up for 6 s before the trial clock starts (longer than
    the 4.6 s it takes a puff to cross the arena), so the fly starts in an
    established plume instead of an empty tunnel. The design text says
    nothing about a warm-up; this one was chosen before any data, and its
    consequence (below) was noticed only in review after the first run.
  * Fly: dt = 0.05 s per step (one 12 ms brain run stands for 50 ms of world
    time, the roamer's convention), top speed 0.02 m/s, top turn rate 180 deg/s.
  * Trial: start at x = 0.45 (0.4 m downwind of the source), y uniform in
    [0.10, 0.20], heading uniform, 400 steps (20 s) or until within 0.03 m of
    the source.

CONSEQUENCES OF THOSE CHOICES (arithmetic, not results; disclosed so a reader
of the outputs does not mistake them for properties of the brain)
  * The source is only just reachable. The fly must cover 0.45 - 0.05 - 0.03
    = 0.37 m in 20 s at a top speed of 0.02 m/s: 18.5 s of straight upwind
    walking at full speed, 92.5 % of what the trial allows. Any hesitation or
    turning makes the source unreachable, so "source reached" is not a
    discriminating test under these constants (REACH_MIN_S, REACH_MIN_SPEED).
  * The start band is inside the plume. On the meander-free plume the
    concentration at x = 0.45 is 0.52 on the centreline and 0.18 at the band
    edges y = 0.10 and 0.20 (start_band_straight_plume()), all above the 0.05
    threshold the metrics use, and the plume's 1-sd half-width there is about
    0.035 m. With the warm-up the fly is therefore born in odour on most
    seeds; encounters and losses in a trial are the meander sweeping over a
    fly that walks about 0.1 m in 20 s, not the fly finding the plume. A
    future preregistration would start the fly outside two sd of the
    centreline, or further downwind, and give it time to reach the source.

SIGN CONVENTIONS (stated once, used everywhere)
  * heading is in radians, counter-clockwise from +x. A fly heading into the
    wind (toward -x) has heading pi.
  * turn > 0 turns RIGHT (clockwise, heading decreases), matching the roamer's
    (R - L) / 450 readout where a positive value is a right turn.
  * wind_direction_relative(heading) is the angle the wind comes FROM relative
    to the heading, wrapped to (-pi, pi]: 0 is a headwind, +pi/2 is wind from
    the fly's LEFT (the fly heading +y, wind from -x), -pi/2 from its right.

Everything is deterministic under the seed: the fly's start pose, then the
meander noise, come from one numpy Generator in a fixed order, and the plume is
simulated identically whether odour is on or off so paired conditions share the
same start and the same wind.
"""
import math

import numpy as np

CONSTANTS = {
    # arena and wind
    "arena_x": 0.6,            # m, along the wind
    "arena_y": 0.3,            # m, across the wind
    "wind_speed": 0.12,        # m/s, toward +x
    "source": (0.05, 0.15),    # m
    # plume
    "puff_interval": 0.05,     # s between puff releases
    "meander_sd": 0.02,        # m/s, stationary sd of the OU crosswind velocity
    "meander_tau": 1.0,        # s, OU time constant
    "r0": 0.005,               # m, puff radius at release
    "width_2sd": 0.06,         # m, plume width (2 sd) ...
    "width_at": 0.3,           # m downwind ... at this distance; fixes D
    "calibrate_at": 0.1,       # m downwind where the centreline reads 1.0; fixes q
    "warmup_s": 6.0,           # s of plume before the trial clock starts
    # fly
    "dt": 0.05,                # s of world time per step
    "v_max": 0.02,             # m/s
    "turn_rate_deg": 180.0,    # deg/s at |turn| = 1
    # trial
    "start_x": 0.45,           # m
    "start_y": (0.10, 0.20),   # m, uniform
    "trial_steps": 400,        # 20 s
    "reach_radius": 0.03,      # m from the source counts as reached
    "odour_threshold": 0.05,   # the encounter/loss threshold the metrics use
}

ARENA_X = CONSTANTS["arena_x"]
ARENA_Y = CONSTANTS["arena_y"]
WIND_SPEED = CONSTANTS["wind_speed"]
SOURCE = tuple(CONSTANTS["source"])
WIND_FROM = math.pi           # the wind blows toward +x, so it comes from -x
PUFF_INTERVAL = CONSTANTS["puff_interval"]
MEANDER_SD = CONSTANTS["meander_sd"]
MEANDER_TAU = CONSTANTS["meander_tau"]
R0 = CONSTANTS["r0"]
WARMUP_S = CONSTANTS["warmup_s"]
DT = CONSTANTS["dt"]
V_MAX = CONSTANTS["v_max"]
TURN_RATE = math.radians(CONSTANTS["turn_rate_deg"])
START_X = CONSTANTS["start_x"]
START_Y = tuple(CONSTANTS["start_y"])
TRIAL_STEPS = CONSTANTS["trial_steps"]
REACH_RADIUS = CONSTANTS["reach_radius"]
ODOUR_THRESHOLD = CONSTANTS["odour_threshold"]

# Derived, not chosen: D from the width requirement.
_t_width = CONSTANTS["width_at"] / WIND_SPEED
DIFFUSION = ((CONSTANTS["width_2sd"] / 2.0) ** 2 - R0 ** 2) / (2.0 * _t_width)
TRANSIT_S = (ARENA_X - SOURCE[0]) / WIND_SPEED     # time for a puff to cross the arena
WARMUP_STEPS = int(round(WARMUP_S / DT))
RELEASE_EVERY = max(1, int(round(PUFF_INTERVAL / DT)))   # steps between puffs
# Derived, not chosen: how hard the trial makes the source to reach.
TRIAL_S = TRIAL_STEPS * DT                                          # 20 s
REACH_MIN_S = (START_X - SOURCE[0] - REACH_RADIUS) / V_MAX          # 18.5 s of straight full-speed upwind walking
REACH_MIN_SPEED = REACH_MIN_S / TRIAL_S                             # 0.925 of top speed, straight, for the whole trial


def _puff_sum(x, y, px, py, age, q):
    """Concentration at (x, y) from puffs at (px, py) of the given ages.

    x and y may be scalars or arrays of any matching shape; the puff axis is
    broadcast on the end and summed out.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if px.size == 0:
        return np.zeros(np.broadcast(x, y).shape, dtype=float)
    r2 = R0 ** 2 + 2.0 * DIFFUSION * age
    d2 = (x[..., None] - px) ** 2 + (y[..., None] - py) ** 2
    return (q / (2.0 * math.pi * r2) * np.exp(-d2 / (2.0 * r2))).sum(axis=-1)


def _straight_puffs():
    """The steady, meander-free plume: one puff per release, ages 0, dt, 2dt,
    ... up to leaving the arena. Used for calibration and for tests."""
    n = int(math.floor(TRANSIT_S / (RELEASE_EVERY * DT))) + 1
    age = np.arange(n) * RELEASE_EVERY * DT
    px = SOURCE[0] + WIND_SPEED * age
    py = np.full(n, SOURCE[1])
    keep = px <= ARENA_X
    return px[keep], py[keep], age[keep]


def _calibrate_q():
    px, py, age = _straight_puffs()
    unit = float(_puff_sum(SOURCE[0] + CONSTANTS["calibrate_at"], SOURCE[1], px, py, age, 1.0))
    return 1.0 / unit


Q = _calibrate_q()          # derived: puff strength for centreline 1.0 at 0.1 m


def wrap_angle(a):
    """Wrap to (-pi, pi]."""
    return np.pi - np.mod(np.pi - a, 2.0 * np.pi)


def wind_direction_relative(heading):
    """Angle the wind comes FROM relative to the heading (see the docstring)."""
    return wrap_angle(WIND_FROM - np.asarray(heading, dtype=float))


class World:
    """One trial's arena, plume and fly.

    World(seed, odour=True, meander=True)
      concentration(x, y)               plume at points (0 everywhere if odour=False)
      field(x, y)                       plume regardless of the odour flag (for pictures)
      wind_direction_relative(heading)  0 = headwind, +pi/2 = wind from the left
      step(turn, speed) -> state dict   {x, y, heading, c, t, reached, wall_contacts, ...}
      state                             the current state dict (a copy)
      start                             the state before any step
      log                               one state dict per step() call
      trajectory()                      the log as a dict of numpy arrays
      snapshot(nx, ny)                  (xs, ys, C) grid of the plume right now
    """

    def __init__(self, seed=0, odour=True, meander=True):
        self.seed = int(seed)
        self.odour = bool(odour)
        self.meander = bool(meander)
        self.rng = np.random.default_rng(self.seed)
        # fly start pose first, so it does not depend on the plume's noise
        self.x = START_X
        self.y = float(self.rng.uniform(*START_Y))
        self.heading = float(self.rng.uniform(0.0, 2.0 * math.pi))
        # plume state
        self._px = np.zeros(0)
        self._py = np.zeros(0)
        self._pvy = np.zeros(0)
        self._age = np.zeros(0)
        self._v = float(self.rng.normal(0.0, MEANDER_SD)) if self.meander else 0.0
        self._since_release = RELEASE_EVERY     # release on the first plume step
        for _ in range(WARMUP_STEPS):
            self._advance_plume()
        # trial state
        self.t = 0.0
        self.n_steps = 0
        self.wall_contacts = 0
        self.reached = self._near_source()
        self.log = []
        self.start = self._state(turn=0.0, speed=0.0, vx=0.0, vy=0.0)

    # ---- plume -------------------------------------------------------------

    def _advance_plume(self):
        """Move every airborne puff by dt, drop the ones that left the arena,
        release a new one when due, and advance the meander velocity."""
        if self._px.size:
            self._px = self._px + WIND_SPEED * DT
            self._py = self._py + self._pvy * DT
            self._age = self._age + DT
            keep = ((self._px >= 0.0) & (self._px <= ARENA_X)
                    & (self._py >= 0.0) & (self._py <= ARENA_Y))
            if not keep.all():
                self._px, self._py = self._px[keep], self._py[keep]
                self._pvy, self._age = self._pvy[keep], self._age[keep]
        self._since_release += 1
        if self._since_release >= RELEASE_EVERY:
            self._since_release = 0
            self._px = np.append(self._px, SOURCE[0])
            self._py = np.append(self._py, SOURCE[1])
            self._pvy = np.append(self._pvy, self._v)
            self._age = np.append(self._age, 0.0)
        if self.meander:
            # exact-in-distribution OU update: stationary sd stays MEANDER_SD
            a = math.exp(-DT / MEANDER_TAU)
            self._v = a * self._v + MEANDER_SD * math.sqrt(1.0 - a * a) * float(self.rng.normal())

    def field(self, x, y):
        """Plume concentration at (x, y), whatever the odour flag says."""
        return _puff_sum(x, y, self._px, self._py, self._age, Q)

    def concentration(self, x, y):
        """What the fly's nose gets: the plume, or nothing if odour is off."""
        if not self.odour:
            return np.zeros(np.broadcast(np.asarray(x, float), np.asarray(y, float)).shape)
        return self.field(x, y)

    def snapshot(self, nx=120, ny=60):
        """A grid of the plume right now, for pictures: (xs, ys, C[ny, nx])."""
        xs = np.linspace(0.0, ARENA_X, nx)
        ys = np.linspace(0.0, ARENA_Y, ny)
        gx, gy = np.meshgrid(xs, ys)
        return xs, ys, self.field(gx, gy)

    @property
    def n_puffs(self):
        return int(self._px.size)

    # ---- wind ----------------------------------------------------------------

    @staticmethod
    def wind_direction_relative(heading):
        return wind_direction_relative(heading)

    # ---- fly -----------------------------------------------------------------

    def _near_source(self):
        return bool(math.hypot(self.x - SOURCE[0], self.y - SOURCE[1]) <= REACH_RADIUS)

    def _state(self, turn, speed, vx, vy):
        c = float(self.concentration(self.x, self.y))
        return {
            "step": self.n_steps,
            "t": self.t,
            "x": self.x,
            "y": self.y,
            "heading": self.heading,
            "c": c,
            "phi": float(wind_direction_relative(self.heading)),
            "turn": float(turn),
            "speed": float(speed),
            "vx": float(vx),
            "vy": float(vy),
            "reached": self.reached,
            "wall_contacts": self.wall_contacts,
            "done": self.reached or self.n_steps >= TRIAL_STEPS,
        }

    @property
    def state(self):
        last = self.log[-1] if self.log else self.start
        return dict(last)

    def step(self, turn, speed):
        """Advance the world by dt with the fly commanding (turn, speed).

        turn in [-1, 1] (clipped): +1 is a full right turn at 180 deg/s.
        speed in [-1, 1] (clipped): +1 walks forward at v_max, negative backs up.
        Returns the new state dict and appends it to the log.
        """
        turn = float(np.clip(turn, -1.0, 1.0))
        speed = float(np.clip(speed, -1.0, 1.0))
        self._advance_plume()
        self.heading = float(wrap_angle(self.heading - turn * TURN_RATE * DT))
        step_len = speed * V_MAX * DT
        x0, y0 = self.x, self.y
        nx = x0 + step_len * math.cos(self.heading)
        ny = y0 + step_len * math.sin(self.heading)
        cx = min(max(nx, 0.0), ARENA_X)
        cy = min(max(ny, 0.0), ARENA_Y)
        if cx != nx or cy != ny:
            self.wall_contacts += 1
        self.x, self.y = cx, cy
        self.t += DT
        self.n_steps += 1
        self.reached = self.reached or self._near_source()
        state = self._state(turn, speed, (self.x - x0) / DT, (self.y - y0) / DT)
        self.log.append(state)
        return state

    def trajectory(self):
        """The log as arrays, one entry per step (the start pose is `start`)."""
        keys = ["step", "t", "x", "y", "heading", "c", "phi", "turn", "speed",
                "vx", "vy", "reached", "wall_contacts"]
        return {k: np.array([row[k] for row in self.log]) for k in keys}


def straight_plume_concentration(x, y):
    """The steady meander-free plume at (x, y): what q was calibrated against."""
    px, py, age = _straight_puffs()
    return _puff_sum(x, y, px, py, age, Q)


def start_band_straight_plume(n=21):
    """
    The meander-free plume across the start band, for disclosure: the
    concentration at the band's centre and edges, its minimum over the band,
    the fraction of the band above the odour threshold, and the plume's
    1-sd half-width at the start x. Says whether a fly can start outside the
    plume at all under these constants.
    """
    ys = np.linspace(START_Y[0], START_Y[1], n)
    c = np.asarray(straight_plume_concentration(np.full(n, START_X), ys), dtype=float)
    age = (START_X - SOURCE[0]) / WIND_SPEED
    return {
        "x": START_X, "y_band": list(START_Y), "threshold": ODOUR_THRESHOLD,
        "c_centre": float(straight_plume_concentration(START_X, SOURCE[1])),
        "c_edges": [float(c[0]), float(c[-1])],
        "c_min": float(c.min()), "c_max": float(c.max()),
        "fraction_of_band_above_threshold": float(np.mean(c > ODOUR_THRESHOLD)),
        "plume_sd_m_at_start_x": float(math.sqrt(R0 ** 2 + 2.0 * DIFFUSION * age)),
        "reach_min_s": REACH_MIN_S, "trial_s": TRIAL_S, "reach_min_speed_fraction": REACH_MIN_SPEED,
    }


if __name__ == "__main__":
    import time
    w = World(seed=0)
    t0 = time.perf_counter()
    for _ in range(TRIAL_STEPS):
        w.step(0.0, 1.0)
    dt_run = time.perf_counter() - t0
    print(f"D={DIFFUSION:.3e} m^2/s  q={Q:.3e}  transit={TRANSIT_S:.2f} s  "
          f"puffs airborne={w.n_puffs}  400 steps in {dt_run*1000:.0f} ms")
    print("centreline 0.1 m downwind, straight plume:",
          float(straight_plume_concentration(SOURCE[0] + 0.1, SOURCE[1])))
